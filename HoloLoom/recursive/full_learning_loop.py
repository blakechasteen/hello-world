"""
Full Learning Loop - Phase 5
=============================
Background learning thread, Thompson Sampling prior updates, and policy
adaptation from outcomes.

This is the culmination of the recursive learning system:
- Continuous background learning from all interactions
- Automatic Thompson Sampling prior updates based on tool success
- Policy adapter weight adjustments based on outcomes
- Knowledge graph updates from learned patterns
- Self-improving retrieval weights

This module completes the vision: A truly self-improving knowledge system
that learns from every interaction and continuously adapts.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

from HoloLoom.documentation.types import Query, Spacetime, MemoryShard
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from Promptly.promptly.recursive_loops import Scratchpad

from .loop_integration import (
    LearningLoopEngine,
    LearningLoopConfig,
    LearnedPattern
)
from .hot_patterns import (
    HotPatternFeedbackEngine,
    HotPatternConfig,
    UsageRecord
)
from .advanced_refinement import (
    AdvancedRefiner,
    RefinementStrategy,
    RefinementResult
)


@dataclass
class ThompsonPriors:
    """
    Thompson Sampling priors for each tool.

    Each tool has Beta distribution parameters (alpha, beta):
    - alpha: Number of successes + 1
    - beta: Number of failures + 1
    """
    tool_priors: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(
        lambda: {"alpha": 1.0, "beta": 1.0}
    ))

    def update_success(self, tool: str, confidence: float):
        """Update priors after successful tool use"""
        self.tool_priors[tool]["alpha"] += confidence

    def update_failure(self, tool: str, confidence: float):
        """Update priors after unsuccessful tool use"""
        self.tool_priors[tool]["beta"] += (1.0 - confidence)

    def get_expected_reward(self, tool: str) -> float:
        """Get expected reward (mean of Beta distribution)"""
        alpha = self.tool_priors[tool]["alpha"]
        beta = self.tool_priors[tool]["beta"]
        return alpha / (alpha + beta)

    def get_uncertainty(self, tool: str) -> float:
        """Get uncertainty (variance of Beta distribution)"""
        alpha = self.tool_priors[tool]["alpha"]
        beta = self.tool_priors[tool]["beta"]
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))


@dataclass
class PolicyWeights:
    """Learned weights for policy adapters"""
    adapter_weights: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 1.0))
    adapter_successes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    adapter_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, adapter: str, success: bool):
        """Update adapter weights based on outcome"""
        self.adapter_total[adapter] += 1
        if success:
            self.adapter_successes[adapter] += 1

        # Recalculate weight (success rate with smoothing)
        total = self.adapter_total[adapter]
        successes = self.adapter_successes[adapter]
        self.adapter_weights[adapter] = (successes + 1) / (total + 2)  # Laplace smoothing

    def get_weight(self, adapter: str) -> float:
        """Get current weight for adapter"""
        return self.adapter_weights[adapter]

    def get_success_rate(self, adapter: str) -> float:
        """Get success rate for adapter"""
        total = self.adapter_total[adapter]
        if total == 0:
            return 0.5  # Unknown
        return self.adapter_successes[adapter] / total


@dataclass
class LearningMetrics:
    """Metrics tracking learning progress"""
    queries_processed: int = 0
    patterns_learned: int = 0
    thompson_updates: int = 0
    policy_updates: int = 0
    retrieval_updates: int = 0
    avg_confidence: float = 0.0
    learning_rate: float = 0.0  # Patterns learned per query

    def update(self, confidence: float):
        """Update metrics with new query result"""
        self.queries_processed += 1
        self.avg_confidence = (
            (self.avg_confidence * (self.queries_processed - 1) + confidence) /
            self.queries_processed
        )


class BackgroundLearner:
    """
    Background thread that continuously learns from accumulated experiences.

    This runs in the background, processing the reflection buffer and
    updating HoloLoom's internal models based on what works.
    """

    def __init__(
        self,
        orchestrator: WeavingOrchestrator,
        thompson_priors: ThompsonPriors,
        policy_weights: PolicyWeights,
        update_interval: float = 60.0  # Update every 60 seconds
    ):
        """
        Initialize background learner.

        Args:
            orchestrator: HoloLoom orchestrator to update
            thompson_priors: Thompson Sampling priors to maintain
            policy_weights: Policy adapter weights to maintain
            update_interval: Seconds between learning updates
        """
        self.orchestrator = orchestrator
        self.thompson_priors = thompson_priors
        self.policy_weights = policy_weights
        self.update_interval = update_interval
        self.logger = logging.getLogger(f"{__name__}.BackgroundLearner")

        # Learning state
        self.recent_spacetimes: deque = deque(maxlen=100)
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        """Start background learning loop"""
        if self.running:
            self.logger.warning("Background learner already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._learning_loop())
        self.logger.info("Background learner started")

    async def stop(self):
        """Stop background learning loop"""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        self.logger.info("Background learner stopped")

    def record_spacetime(self, spacetime: Spacetime):
        """Record spacetime for learning"""
        self.recent_spacetimes.append(spacetime)

    async def _learning_loop(self):
        """Main background learning loop"""
        while self.running:
            try:
                await asyncio.sleep(self.update_interval)

                if not self.recent_spacetimes:
                    continue

                self.logger.info("Running background learning update...")

                # 1. Update Thompson Sampling priors
                await self._update_thompson_priors()

                # 2. Update policy adapter weights
                await self._update_policy_weights()

                # 3. Update retrieval weights (TODO: implement)
                # await self._update_retrieval_weights()

                self.logger.info("Background learning update complete")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in learning loop: {e}")

    async def _update_thompson_priors(self):
        """Update Thompson Sampling priors from recent experiences"""
        tool_confidences = defaultdict(list)

        # Collect tool usage and confidence
        for spacetime in self.recent_spacetimes:
            tool = spacetime.trace.tool_selected
            confidence = spacetime.trace.tool_confidence
            tool_confidences[tool].append(confidence)

        # Update priors
        for tool, confidences in tool_confidences.items():
            avg_confidence = sum(confidences) / len(confidences)

            if avg_confidence >= 0.75:
                # Successful tool use
                self.thompson_priors.update_success(tool, avg_confidence)
            else:
                # Unsuccessful tool use
                self.thompson_priors.update_failure(tool, avg_confidence)

        self.logger.info(
            f"Updated Thompson priors for {len(tool_confidences)} tools"
        )

    async def _update_policy_weights(self):
        """Update policy adapter weights from recent experiences"""
        adapter_outcomes = defaultdict(list)

        # Collect adapter usage and outcomes
        for spacetime in self.recent_spacetimes:
            adapter = spacetime.trace.policy_adapter
            success = spacetime.trace.tool_confidence >= 0.75
            adapter_outcomes[adapter].append(success)

        # Update weights
        for adapter, outcomes in adapter_outcomes.items():
            for outcome in outcomes:
                self.policy_weights.update(adapter, outcome)

        self.logger.info(
            f"Updated policy weights for {len(adapter_outcomes)} adapters"
        )


class FullLearningEngine:
    """
    Complete self-improving orchestrator with full learning loop.

    This is the ultimate integration:
    - Phase 1: Scratchpad provenance tracking
    - Phase 2: Pattern learning from successful queries
    - Phase 3: Hot pattern feedback and adaptive retrieval
    - Phase 4: Advanced refinement strategies
    - Phase 5: Background learning with Thompson/policy updates

    The system continuously learns from every interaction and adapts
    its behavior to improve future performance.
    """

    def __init__(
        self,
        cfg: Config,
        shards: Optional[List[MemoryShard]] = None,
        enable_background_learning: bool = True,
        learning_update_interval: float = 60.0
    ):
        """
        Initialize full learning engine.

        Args:
            cfg: HoloLoom configuration
            shards: Memory shards
            enable_background_learning: Enable background learning thread
            learning_update_interval: Seconds between learning updates
        """
        self.cfg = cfg
        self.shards = shards or []
        self.logger = logging.getLogger(f"{__name__}.FullLearningEngine")

        # Core components
        self.scratchpad = Scratchpad()

        # Learning state
        self.thompson_priors = ThompsonPriors()
        self.policy_weights = PolicyWeights()
        self.metrics = LearningMetrics()

        # Will be initialized in __aenter__
        self.orchestrator: Optional[WeavingOrchestrator] = None
        self.hot_pattern_engine: Optional[HotPatternFeedbackEngine] = None
        self.advanced_refiner: Optional[AdvancedRefiner] = None
        self.background_learner: Optional[BackgroundLearner] = None

        self.enable_background_learning = enable_background_learning
        self.learning_update_interval = learning_update_interval

    async def __aenter__(self):
        """Async context manager entry"""
        # Initialize orchestrator
        self.orchestrator = WeavingOrchestrator(
            cfg=self.cfg,
            shards=self.shards
        )
        await self.orchestrator.__aenter__()

        # Initialize hot pattern engine
        self.hot_pattern_engine = HotPatternFeedbackEngine(
            cfg=self.cfg,
            shards=self.shards,
            hot_config=HotPatternConfig(
                enable_tracking=True,
                enable_adaptive_retrieval=True,
                heat_threshold=5.0,
                decay_rate=0.95
            ),
            learning_config=LearningLoopConfig(
                enable_pattern_learning=True,
                enable_auto_pruning=True
            )
        )
        await self.hot_pattern_engine.__aenter__()

        # Initialize advanced refiner
        self.advanced_refiner = AdvancedRefiner(
            orchestrator=self.orchestrator,
            scratchpad=self.scratchpad,
            enable_learning=True
        )

        # Initialize background learner
        if self.enable_background_learning:
            self.background_learner = BackgroundLearner(
                orchestrator=self.orchestrator,
                thompson_priors=self.thompson_priors,
                policy_weights=self.policy_weights,
                update_interval=self.learning_update_interval
            )
            await self.background_learner.start()
            self.logger.info("Background learning enabled")

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Stop background learner
        if self.background_learner:
            await self.background_learner.stop()

        # Cleanup components
        if self.hot_pattern_engine:
            await self.hot_pattern_engine.__aexit__(exc_type, exc_val, exc_tb)

        if self.orchestrator:
            await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)

    async def weave(
        self,
        query: Query,
        enable_refinement: bool = True,
        refinement_threshold: float = 0.75,
        max_refinement_iterations: int = 3
    ) -> Spacetime:
        """
        Process query with full learning loop.

        Flow:
        1. Weave with hot pattern engine (includes pattern learning)
        2. Track in scratchpad
        3. Refine if low confidence (using advanced strategies)
        4. Record for background learning
        5. Update Thompson priors immediately
        6. Return result

        Args:
            query: Query to process
            enable_refinement: Whether to refine low-confidence results
            refinement_threshold: Confidence threshold for refinement
            max_refinement_iterations: Max refinement iterations

        Returns:
            Final Spacetime result
        """
        # 1. Weave with hot pattern engine (includes learning)
        spacetime = await self.hot_pattern_engine.weave(query)

        # 2. Track in scratchpad
        self.scratchpad.add_entry(
            thought=f"Query: {query.text[:100]}",
            action=f"Tool: {spacetime.trace.tool_selected}, Adapter: {spacetime.trace.policy_adapter}",
            observation=f"Confidence: {spacetime.trace.tool_confidence:.2f}",
            score=spacetime.trace.tool_confidence
        )

        # 3. Refine if low confidence
        if enable_refinement and spacetime.trace.tool_confidence < refinement_threshold:
            self.logger.info(
                f"Low confidence ({spacetime.trace.tool_confidence:.2f}), triggering refinement"
            )

            refinement_result = await self.advanced_refiner.refine(
                query=query,
                initial_spacetime=spacetime,
                strategy=None,  # Auto-select
                max_iterations=max_refinement_iterations,
                quality_threshold=0.9
            )

            spacetime = refinement_result.final_spacetime

            # Log refinement to scratchpad
            self.scratchpad.add_entry(
                thought=f"Refinement: {refinement_result.strategy_used.value}",
                action=f"Iterations: {refinement_result.iterations}",
                observation=refinement_result.summary(),
                score=refinement_result.trajectory[-1].score()
            )

        # 4. Record for background learning
        if self.background_learner:
            self.background_learner.record_spacetime(spacetime)

        # 5. Update Thompson priors immediately (in addition to background updates)
        tool = spacetime.trace.tool_selected
        confidence = spacetime.trace.tool_confidence
        if confidence >= 0.75:
            self.thompson_priors.update_success(tool, confidence)
        else:
            self.thompson_priors.update_failure(tool, confidence)

        # 6. Update policy weights immediately
        adapter = spacetime.trace.policy_adapter
        success = confidence >= 0.75
        self.policy_weights.update(adapter, success)

        # 7. Update metrics
        self.metrics.update(confidence)
        self.metrics.thompson_updates += 1
        self.metrics.policy_updates += 1

        return spacetime

    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics.

        Returns:
            Dict with all learning metrics
        """
        stats = {
            "queries_processed": self.metrics.queries_processed,
            "avg_confidence": self.metrics.avg_confidence,
            "thompson_priors": {
                tool: {
                    "expected_reward": self.thompson_priors.get_expected_reward(tool),
                    "uncertainty": self.thompson_priors.get_uncertainty(tool),
                    **priors
                }
                for tool, priors in self.thompson_priors.tool_priors.items()
            },
            "policy_weights": {
                adapter: {
                    "weight": self.policy_weights.get_weight(adapter),
                    "success_rate": self.policy_weights.get_success_rate(adapter),
                    "total_uses": self.policy_weights.adapter_total[adapter]
                }
                for adapter in self.policy_weights.adapter_weights.keys()
            },
            "refinement_strategies": self.advanced_refiner.get_strategy_statistics() if self.advanced_refiner else {},
            "background_learning": {
                "enabled": self.enable_background_learning,
                "update_interval": self.learning_update_interval,
                "recent_experiences": len(self.background_learner.recent_spacetimes) if self.background_learner else 0
            }
        }

        # Add hot pattern stats if available
        if self.hot_pattern_engine:
            hot_patterns = self.hot_pattern_engine.hot_tracker.get_hot_patterns(limit=10)
            stats["hot_patterns"] = [
                {
                    "element_id": record.element_id,
                    "heat_score": record.heat_score,
                    "access_count": record.access_count,
                    "success_rate": record.success_rate,
                    "avg_confidence": record.avg_confidence
                }
                for record in hot_patterns
            ]

        # Add learned patterns if available
        if self.hot_pattern_engine:
            learned_patterns = self.hot_pattern_engine.learning_engine.pattern_learner.get_hot_patterns()
            stats["learned_patterns"] = [
                {
                    "motifs": pattern.motifs[:3],
                    "tool": pattern.tool,
                    "query_type": pattern.query_type,
                    "occurrences": pattern.occurrences,
                    "avg_confidence": pattern.avg_confidence
                }
                for pattern in learned_patterns[:10]
            ]

        return stats

    def get_scratchpad_history(self) -> str:
        """Get complete scratchpad history"""
        return self.scratchpad.get_history()

    def save_learning_state(self, path: str):
        """
        Save complete learning state to disk.

        Args:
            path: Directory to save state
        """
        import json
        import os

        os.makedirs(path, exist_ok=True)

        # Save Thompson priors
        with open(os.path.join(path, "thompson_priors.json"), "w") as f:
            json.dump(dict(self.thompson_priors.tool_priors), f, indent=2)

        # Save policy weights
        with open(os.path.join(path, "policy_weights.json"), "w") as f:
            json.dump({
                "weights": dict(self.policy_weights.adapter_weights),
                "successes": dict(self.policy_weights.adapter_successes),
                "total": dict(self.policy_weights.adapter_total)
            }, f, indent=2)

        # Save metrics
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump({
                "queries_processed": self.metrics.queries_processed,
                "avg_confidence": self.metrics.avg_confidence,
                "thompson_updates": self.metrics.thompson_updates,
                "policy_updates": self.metrics.policy_updates
            }, f, indent=2)

        # Save scratchpad
        with open(os.path.join(path, "scratchpad.txt"), "w") as f:
            f.write(self.get_scratchpad_history())

        self.logger.info(f"Saved learning state to {path}")


# Convenience function for quick usage
async def weave_with_full_learning(
    query: Query,
    config: Config,
    shards: Optional[List[MemoryShard]] = None,
    enable_refinement: bool = True,
    enable_background_learning: bool = True
) -> Spacetime:
    """
    Convenience function for one-off usage with full learning.

    Args:
        query: Query to process
        config: HoloLoom configuration
        shards: Memory shards
        enable_refinement: Enable refinement for low confidence
        enable_background_learning: Enable background learning thread

    Returns:
        Final Spacetime result
    """
    async with FullLearningEngine(
        cfg=config,
        shards=shards,
        enable_background_learning=enable_background_learning
    ) as engine:
        return await engine.weave(query, enable_refinement=enable_refinement)
