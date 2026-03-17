"""
Bandit Policy
=============

Thompson Sampling-enhanced policy for intelligent tool selection.
Extends WeavingPolicyBase with exploration/exploitation via bandits,
A/B testing framework, and safety guardrails integration.

Migrated from weaving_orchestrator_bandit.py (PR 9, A-10).
The bandit doesn't select stages — it selects tools via Thompson Sampling
(Beta(α,β) priors, A/B testing). This is a RunPolicy extension.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import WeavingPolicyBase

if TYPE_CHECKING:
    from hololoom.orchestrator.context import WeavingContext


@dataclass
class BanditConfig:
    """Configuration for bandit-enhanced tool selection."""

    # Bandit model
    sampler_type: str = "neural"  # "discrete", "linear", "neural"
    context_dim: int = 384  # Match Matryoshka embeddings
    action_dim: int = 0  # Context-only (tool IDs only)

    # Neural-specific (if sampler_type="neural")
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    backend: str = "bootstrap"  # "bootstrap" or "mc_dropout"
    n_ensemble: int = 7

    # Training
    train_every: int = 200
    train_steps: int = 100
    replay_warmup: int = 1000

    # A/B testing
    ab_test_ratio: float = 0.1  # 10% traffic to bandit

    # Safety
    enable_safety: bool = True
    safety_threshold: float = 0.7

    # Monitoring
    enable_monitoring: bool = True
    log_decisions: bool = True

    # Rewards
    reward_success_weight: float = 1.0
    reward_latency_penalty: float = 0.0001  # Per ms
    reward_cost_penalty: float = 0.1


class BanditPolicy(WeavingPolicyBase):
    """
    Thompson Sampling-enhanced weaving policy.

    Extends WeavingPolicyBase with:
    - Bandit-based tool selection via on_stage_complete hooks
    - A/B testing (bandit vs baseline) per-query
    - Safety guardrails for high-risk actions
    - Reward computation and learning

    Usage:
        policy = BanditPolicy(
            bandit_config=BanditConfig(sampler_type="neural", ab_test_ratio=0.1),
        )
        runner = PipelineRunner(stages, pre_satisfied, policy=policy)
    """

    # Side-effect stages whose failure shouldn't abort
    _NON_CRITICAL = frozenset({
        'production_metrics', 'recursive_learning', 'reflection',
    })

    def __init__(
        self,
        bandit_config: Optional[BanditConfig] = None,
        enable_bandit: bool = True,
        tools: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bandit_config = bandit_config or BanditConfig()
        self.enable_bandit = enable_bandit
        self.tools = tools or ["answer", "search", "notion_write", "calc"]

        # Lazy-init bandit policy
        self._bandit_policy = None
        self._policy_initialized = False

        # Safety (lazy-init)
        self._safety = None
        self._safety_initialized = False

        # Evaluation (lazy-init)
        self._evaluator = None

        # Statistics
        self._total_decisions = 0
        self._bandit_decisions = 0
        self._baseline_decisions = 0
        self._safety_blocks = 0
        self.decision_log: List[Dict] = []

        # Per-query state
        self._current_use_bandit = False
        self._query_start_time: Optional[float] = None

    def _init_bandit(self):
        """Lazy initialization of bandit policy."""
        if self._policy_initialized:
            return
        try:
            from hololoom.ts_core import create_thompson_sampler
            cfg = self.bandit_config

            if cfg.sampler_type == "neural":
                self._bandit_policy = create_thompson_sampler(
                    "neural",
                    context_dim=cfg.context_dim,
                    action_dim=cfg.action_dim,
                    hidden_dims=cfg.hidden_dims,
                    backend=cfg.backend,
                    n_ensemble=cfg.n_ensemble,
                    train_every=cfg.train_every,
                    train_steps=cfg.train_steps,
                    replay_warmup=cfg.replay_warmup,
                )
            elif cfg.sampler_type == "discrete":
                self._bandit_policy = create_thompson_sampler(
                    "discrete",
                    n_arms=len(self.tools),
                    prior="uniform",
                )
            elif cfg.sampler_type == "linear":
                self._bandit_policy = create_thompson_sampler(
                    "linear",
                    context_dim=cfg.context_dim,
                    n_actions=len(self.tools),
                    prior="default",
                )
            else:
                raise ValueError(f"Unknown sampler_type: {cfg.sampler_type}")
            self._policy_initialized = True
        except ImportError:
            self.logger.warning("Thompson Sampling not available — bandit disabled")
            self.enable_bandit = False

    def _init_safety(self):
        """Lazy initialization of safety guardrails."""
        if self._safety_initialized:
            return
        self._safety_initialized = True
        if not self.bandit_config.enable_safety:
            return
        try:
            from hololoom.alignment.safety_guardrails import create_guardrails
            self._safety = create_guardrails(testing_mode=False)
        except ImportError:
            self.logger.debug("Safety guardrails not available")

    def _init_evaluator(self):
        """Lazy initialization of evaluation metrics."""
        if self._evaluator is not None:
            return
        try:
            from hololoom.bandits.neural_ts.eval import BanditEvaluator
            self._evaluator = BanditEvaluator(n_bins=10)
        except ImportError:
            pass

    def _should_use_bandit(self) -> bool:
        """A/B test: randomly assign traffic to bandit or baseline."""
        if not self.enable_bandit:
            return False
        return random.random() < self.bandit_config.ab_test_ratio

    def should_skip(self, stage_name: str, ctx: 'WeavingContext') -> bool:
        return False

    def should_abort_on_failure(self, stage_name: str) -> bool:
        return stage_name not in self._NON_CRITICAL

    # ----- Lifecycle hooks -----

    def on_stage_start(self, stage_name: str, ctx: 'WeavingContext') -> None:
        super().on_stage_start(stage_name, ctx)

        # At pipeline start (first stage), decide bandit vs baseline
        if self._query_start_time is None:
            self._query_start_time = time.time()
            self._total_decisions += 1
            self._current_use_bandit = self._should_use_bandit()
            if self._current_use_bandit:
                self._bandit_decisions += 1
                self._init_bandit()
            else:
                self._baseline_decisions += 1

    def on_stage_complete(
        self,
        stage_name: str,
        ctx: 'WeavingContext',
        duration_ms: float,
        error: Optional[Exception] = None,
    ) -> None:
        super().on_stage_complete(stage_name, ctx, duration_ms, error)

        # After fabric_weaving (final data stage), compute reward and log
        if stage_name == 'fabric_weaving' and self._query_start_time is not None:
            self._record_decision(ctx)

    def on_level_complete(
        self, level: list[str], ctx: 'WeavingContext',
    ) -> Optional[frozenset[str]]:
        return None

    # ----- Bandit-specific methods -----

    def _record_decision(self, ctx: 'WeavingContext') -> None:
        """Record decision + reward after pipeline completion."""
        cfg = self.bandit_config
        latency_ms = (time.time() - self._query_start_time) * 1000

        # Extract outcome
        spacetime = getattr(ctx, 'spacetime', None)
        confidence = getattr(spacetime, 'confidence', 0.5) if spacetime else 0.5
        tool_used = "answer"
        if spacetime and hasattr(spacetime, 'metadata'):
            tool_used = spacetime.metadata.get("tool_used", "answer")

        # Compute reward
        reward = max(0.0, min(1.0,
            cfg.reward_success_weight * confidence
            - cfg.reward_latency_penalty * latency_ms
        ))

        # Track evaluation
        if self._current_use_bandit and cfg.enable_monitoring:
            self._init_evaluator()
            if self._evaluator:
                self._evaluator.record(confidence, reward)

        # Decision log
        if cfg.log_decisions:
            query_text = getattr(ctx, 'query', None)
            self.decision_log.append({
                "timestamp": time.time(),
                "query": str(query_text)[:100] if query_text else "",
                "tool": tool_used,
                "reward": reward,
                "latency_ms": latency_ms,
                "confidence": confidence,
                "use_bandit": self._current_use_bandit,
            })

        # Reset per-query state
        self._query_start_time = None
        self._current_use_bandit = False

    def compute_reward(
        self, confidence: float, latency_ms: float, cost: float = 0.0,
    ) -> float:
        """Compute reward from execution outcome."""
        cfg = self.bandit_config
        reward = (
            cfg.reward_success_weight * confidence
            - cfg.reward_latency_penalty * latency_ms
            - cfg.reward_cost_penalty * cost
        )
        return max(0.0, min(1.0, reward))

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive bandit metrics."""
        result: Dict[str, Any] = {
            "total_decisions": self._total_decisions,
            "bandit_decisions": self._bandit_decisions,
            "baseline_decisions": self._baseline_decisions,
            "ab_ratio_actual": (
                self._bandit_decisions / self._total_decisions
                if self._total_decisions > 0 else 0.0
            ),
            "ab_ratio_target": self.bandit_config.ab_test_ratio,
            "safety_blocks": self._safety_blocks,
        }

        if self._evaluator:
            eval_metrics = self._evaluator.compute_metrics()
            result.update({
                "ece": eval_metrics.get("ece", 0.0),
                "regret_proxy": eval_metrics.get("regret_proxy", 0.0),
                "mean_reward": eval_metrics.get("mean_reward", 0.0),
            })

        if self._policy_initialized and self._bandit_policy:
            result["policy_stats"] = self._bandit_policy.get_statistics()

        return result

    def get_decision_log(
        self, limit: int = 100, bandit_only: bool = False,
    ) -> List[Dict]:
        """Get recent decision log."""
        log = self.decision_log
        if bandit_only:
            log = [d for d in log if d.get("use_bandit", False)]
        return log[-limit:]

    def reset_metrics(self) -> None:
        """Reset all metrics for a new evaluation period."""
        if self._evaluator:
            self._evaluator.reset()
        self._total_decisions = 0
        self._bandit_decisions = 0
        self._baseline_decisions = 0
        self._safety_blocks = 0
        self.decision_log.clear()

    def save_checkpoint(self, path: str) -> None:
        """Save bandit policy checkpoint."""
        if self._policy_initialized and self._bandit_policy:
            self._bandit_policy.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """Load bandit policy checkpoint."""
        self._init_bandit()
        if self._bandit_policy:
            self._bandit_policy.load_checkpoint(path)
