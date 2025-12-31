"""
Thompson Sampling Router for Ritual Phase Selection
====================================================

Created: December 30, 2025
Purpose: Thompson Sampling-based phase selection for ritual orchestration

Uses Bayesian Beta(α, β) priors to balance exploration and exploitation
when selecting which phase agents to invoke. Learns from historical
success rates to improve phase selection over time.

Mirrors HoloLoom/agentic/expert_router.py pattern.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import random
import logging

# Import from agent registration
from .agent_registration import (
    AgentCapability,
    RitualAgentInfo,
    RitualAgentRegistry,
    PHASE_CAPABILITY_MAP,
    get_capability_for_phase,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Selection Strategy Enum
# ============================================================================

class SelectionStrategy(Enum):
    """Strategies for selecting phases."""
    THOMPSON = "thompson"           # Pure Thompson Sampling
    EPSILON_GREEDY = "epsilon"      # ε-greedy with Thompson exploration
    SEQUENTIAL = "sequential"       # Fixed order (default ritual)
    ADAPTIVE = "adaptive"           # Context-aware selection (recommended)


# ============================================================================
# Thompson Prior (Standalone for per-context tracking)
# ============================================================================

@dataclass
class ThompsonPrior:
    """
    Beta distribution prior for Thompson Sampling.

    α (alpha): pseudo-count of successes + 1
    β (beta): pseudo-count of failures + 1

    Higher α → more confident in success
    Higher β → more confident in failure

    This is used for context-specific priors (e.g., "new_feature" vs "bug_fix")
    while RitualAgentInfo tracks global priors per agent.
    """
    alpha: float = 1.0  # Successes + 1 (Bayesian prior)
    beta: float = 1.0   # Failures + 1

    def sample(self) -> float:
        """Sample from Beta(α, β) distribution."""
        return random.betavariate(self.alpha, self.beta)

    def expected_reward(self) -> float:
        """E[X] = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)

    def update_success(self, confidence: float = 1.0) -> None:
        """Update prior on success: α ← α + confidence"""
        self.alpha += confidence

    def update_failure(self, confidence: float = 0.0) -> None:
        """Update prior on failure: β ← β + (1 - confidence)"""
        self.beta += (1.0 - confidence)

    def confidence_interval(self) -> Tuple[float, float]:
        """
        Return approximate 95% confidence interval using normal approximation.

        For Beta(α, β):
        - Mean = α / (α + β)
        - Variance = αβ / ((α + β)² × (α + β + 1))
        """
        total = self.alpha + self.beta
        mean = self.alpha / total
        variance = (self.alpha * self.beta) / (total ** 2 * (total + 1))
        std = variance ** 0.5

        # 95% CI: mean ± 1.96 × std
        lower = max(0.0, mean - 1.96 * std)
        upper = min(1.0, mean + 1.96 * std)

        return (lower, upper)

    @property
    def observations(self) -> int:
        """Total observations (excluding prior)."""
        return int(self.alpha + self.beta - 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "expected_reward": self.expected_reward(),
            "observations": self.observations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThompsonPrior":
        """Deserialize from dictionary."""
        return cls(
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
        )


# ============================================================================
# Ritual Phase Router
# ============================================================================

class RitualPhaseRouter:
    """
    Routes to ritual phases using Thompson Sampling.

    Learns which phases work best for different contexts:
    - Feature types (new feature, bug fix, refactor)
    - Code complexity (simple, moderate, complex)
    - Historical success rates per phase

    Supports 4 selection strategies:
    - THOMPSON: Pure Thompson Sampling (exploration/exploitation)
    - EPSILON_GREEDY: ε-greedy with Thompson exploration
    - SEQUENTIAL: Fixed order (default ritual flow)
    - ADAPTIVE: Context-aware with combined scoring (recommended)
    """

    def __init__(
        self,
        registry: RitualAgentRegistry,
        strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE,
        epsilon: float = 0.1,
        success_threshold: float = 0.75,
    ):
        """
        Initialize router.

        Args:
            registry: RitualAgentRegistry with registered phase agents
            strategy: Selection strategy (default: ADAPTIVE)
            epsilon: Exploration rate for epsilon-greedy (default: 0.1)
            success_threshold: Confidence threshold for success (default: 0.75)
        """
        self.registry = registry
        self.strategy = strategy
        self.epsilon = epsilon
        self.success_threshold = success_threshold

        # Context-specific Thompson priors: (agent_id, context_type) → ThompsonPrior
        # This enables learning different preferences per context
        self._context_priors: Dict[Tuple[str, str], ThompsonPrior] = {}

        # Track selection history for analysis
        self._selection_history: List[Dict[str, Any]] = []

        logger.info(
            f"RitualPhaseRouter initialized: strategy={strategy.value}, "
            f"epsilon={epsilon}, success_threshold={success_threshold}"
        )

    # ========================================================================
    # Prior Management
    # ========================================================================

    def _get_prior(self, agent_id: str, context_type: str = "default") -> ThompsonPrior:
        """Get or create prior for agent + context combination."""
        key = (agent_id, context_type)
        if key not in self._context_priors:
            self._context_priors[key] = ThompsonPrior()
        return self._context_priors[key]

    def _get_agent_with_context_prior(
        self,
        agent: RitualAgentInfo,
        context_type: str
    ) -> Tuple[RitualAgentInfo, ThompsonPrior]:
        """Get agent and its context-specific prior."""
        prior = self._get_prior(agent.id, context_type)
        return agent, prior

    # ========================================================================
    # Selection Methods
    # ========================================================================

    def select_phase(
        self,
        required_capability: AgentCapability,
        context_type: str = "default",
        top_k: int = 1
    ) -> List[RitualAgentInfo]:
        """
        Select best phase agent(s) for a capability.

        Uses Thompson Sampling to balance:
        - Exploitation: Use phases that worked well historically
        - Exploration: Try phases with uncertain performance

        Args:
            required_capability: Capability needed (e.g., PLANNING)
            context_type: Context for routing (e.g., "new_feature", "bug_fix")
            top_k: Number of agents to return (default: 1)

        Returns:
            List of RitualAgentInfo, sorted by selection score
        """
        # Find all agents with required capability
        candidates = self.registry.find_by_capability(required_capability)

        if not candidates:
            logger.warning(f"No agents found for capability: {required_capability.value}")
            return []

        # Apply selection strategy
        if self.strategy == SelectionStrategy.SEQUENTIAL:
            selected = self._select_sequential(candidates, top_k)

        elif self.strategy == SelectionStrategy.THOMPSON:
            selected = self._select_thompson(candidates, context_type, top_k)

        elif self.strategy == SelectionStrategy.EPSILON_GREEDY:
            selected = self._select_epsilon_greedy(candidates, context_type, top_k)

        elif self.strategy == SelectionStrategy.ADAPTIVE:
            selected = self._select_adaptive(candidates, context_type, top_k)

        else:
            # Fallback to sequential
            selected = self._select_sequential(candidates, top_k)

        # Record selection for history
        self._record_selection(required_capability, context_type, selected)

        return selected

    def select_phase_for_name(
        self,
        phase_name: str,
        context_type: str = "default",
        top_k: int = 1
    ) -> List[RitualAgentInfo]:
        """
        Select agent(s) by phase name (e.g., "awaken", "plan").

        Convenience wrapper around select_phase using PHASE_CAPABILITY_MAP.
        """
        capability = get_capability_for_phase(phase_name)
        if capability is None:
            logger.warning(f"Unknown phase name: {phase_name}")
            return []

        return self.select_phase(capability, context_type, top_k)

    # ========================================================================
    # Strategy Implementations
    # ========================================================================

    def _select_sequential(
        self,
        candidates: List[RitualAgentInfo],
        top_k: int
    ) -> List[RitualAgentInfo]:
        """Fixed order - sort by priority and return top-k."""
        sorted_candidates = sorted(candidates, key=lambda a: -a.priority)
        return sorted_candidates[:top_k]

    def _select_thompson(
        self,
        candidates: List[RitualAgentInfo],
        context_type: str,
        top_k: int
    ) -> List[RitualAgentInfo]:
        """Pure Thompson Sampling - sample from Beta distributions."""
        scores = []
        for agent in candidates:
            prior = self._get_prior(agent.id, context_type)
            sampled_score = prior.sample()
            scores.append((agent, sampled_score))

        # Sort by sampled score (highest first)
        scores.sort(key=lambda x: -x[1])
        return [s[0] for s in scores[:top_k]]

    def _select_epsilon_greedy(
        self,
        candidates: List[RitualAgentInfo],
        context_type: str,
        top_k: int
    ) -> List[RitualAgentInfo]:
        """ε-greedy with Thompson exploration."""
        if random.random() < self.epsilon:
            # Explore: random selection
            return random.sample(candidates, min(top_k, len(candidates)))
        else:
            # Exploit: use expected rewards
            scores = []
            for agent in candidates:
                prior = self._get_prior(agent.id, context_type)
                expected = prior.expected_reward()
                scores.append((agent, expected))

            scores.sort(key=lambda x: -x[1])
            return [s[0] for s in scores[:top_k]]

    def _select_adaptive(
        self,
        candidates: List[RitualAgentInfo],
        context_type: str,
        top_k: int
    ) -> List[RitualAgentInfo]:
        """
        Context-aware adaptive selection.

        Combines Thompson sampling with agent priority:
        - 70% weight to Thompson sampled score
        - 30% weight to agent priority (normalized to 0-1)

        This balances learned preferences with predefined importance.
        """
        scores = []
        for agent in candidates:
            prior = self._get_prior(agent.id, context_type)
            sampled = prior.sample()

            # Normalize priority to 0-1 (assuming priority is 0-100)
            priority_normalized = agent.priority / 100.0

            # Weighted combination
            combined = 0.7 * sampled + 0.3 * priority_normalized
            scores.append((agent, combined))

        scores.sort(key=lambda x: -x[1])
        return [s[0] for s in scores[:top_k]]

    # ========================================================================
    # Update Methods
    # ========================================================================

    def update(
        self,
        agent_id: str,
        context_type: str,
        success: bool,
        confidence: float = 1.0
    ) -> None:
        """
        Update Thompson prior based on phase outcome.

        Args:
            agent_id: ID of agent that was invoked
            context_type: Context in which invocation occurred
            success: Whether phase was successful
            confidence: Confidence in outcome (0.0-1.0)
        """
        prior = self._get_prior(agent_id, context_type)

        if success:
            prior.update_success(confidence)
            logger.debug(
                f"Updated {agent_id}[{context_type}] SUCCESS: "
                f"α={prior.alpha:.2f}, β={prior.beta:.2f}"
            )
        else:
            prior.update_failure(confidence)
            logger.debug(
                f"Updated {agent_id}[{context_type}] FAILURE: "
                f"α={prior.alpha:.2f}, β={prior.beta:.2f}"
            )

        # Also update the agent's global Thompson priors
        agent = self.registry.get_agent(agent_id)
        if agent:
            # Use record_invocation which updates global priors
            agent.record_invocation(
                success=success,
                latency_ms=0.0,  # Router doesn't track latency
                confidence=confidence
            )

    def update_from_result(
        self,
        agent_id: str,
        context_type: str,
        confidence: float
    ) -> None:
        """
        Update based on confidence score (convenience method).

        Success if confidence >= success_threshold.
        """
        success = confidence >= self.success_threshold
        self.update(agent_id, context_type, success, confidence)

    # ========================================================================
    # Recommendations & Analysis
    # ========================================================================

    def get_recommendations(
        self,
        context_type: str = "default"
    ) -> Dict[str, Dict[str, float]]:
        """
        Get expected rewards for all phases in a context.

        Returns:
            Dict mapping agent_id to metrics:
            {
                "agent_id": {
                    "expected_reward": 0.75,
                    "alpha": 10.0,
                    "beta": 5.0,
                    "confidence": 13.0,  # Total observations
                    "ci_lower": 0.52,
                    "ci_upper": 0.91,
                }
            }
        """
        recommendations = {}

        for (agent_id, ctx), prior in self._context_priors.items():
            if ctx == context_type:
                ci = prior.confidence_interval()
                recommendations[agent_id] = {
                    "expected_reward": prior.expected_reward(),
                    "alpha": prior.alpha,
                    "beta": prior.beta,
                    "confidence": prior.observations,
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                }

        return recommendations

    def get_best_agent_for_phase(
        self,
        phase_name: str,
        context_type: str = "default"
    ) -> Optional[Tuple[RitualAgentInfo, float]]:
        """
        Get the best agent for a phase with expected reward.

        Returns:
            Tuple of (agent, expected_reward) or None if no agents found
        """
        agents = self.select_phase_for_name(phase_name, context_type, top_k=1)
        if not agents:
            return None

        agent = agents[0]
        prior = self._get_prior(agent.id, context_type)
        return (agent, prior.expected_reward())

    def get_all_context_types(self) -> List[str]:
        """Get all context types that have been used."""
        return list(set(ctx for _, ctx in self._context_priors.keys()))

    # ========================================================================
    # Selection History
    # ========================================================================

    def _record_selection(
        self,
        capability: AgentCapability,
        context_type: str,
        selected: List[RitualAgentInfo]
    ) -> None:
        """Record selection for history/analysis."""
        import time

        self._selection_history.append({
            "timestamp": time.time(),
            "capability": capability.value,
            "context_type": context_type,
            "strategy": self.strategy.value,
            "selected_agents": [a.id for a in selected],
            "num_candidates": len(self.registry.find_by_capability(capability)),
        })

        # Keep history bounded (last 1000 selections)
        if len(self._selection_history) > 1000:
            self._selection_history = self._selection_history[-1000:]

    def get_selection_history(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent selection history."""
        return self._selection_history[-limit:]

    def get_selection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about selection patterns.

        Returns:
            Dict with selection statistics:
            - total_selections: int
            - by_strategy: Dict[str, int]
            - by_capability: Dict[str, int]
            - by_context: Dict[str, int]
        """
        stats = {
            "total_selections": len(self._selection_history),
            "by_strategy": {},
            "by_capability": {},
            "by_context": {},
        }

        for record in self._selection_history:
            # Strategy counts
            strat = record["strategy"]
            stats["by_strategy"][strat] = stats["by_strategy"].get(strat, 0) + 1

            # Capability counts
            cap = record["capability"]
            stats["by_capability"][cap] = stats["by_capability"].get(cap, 0) + 1

            # Context counts
            ctx = record["context_type"]
            stats["by_context"][ctx] = stats["by_context"].get(ctx, 0) + 1

        return stats

    # ========================================================================
    # Persistence
    # ========================================================================

    def save_state(self, path: str) -> None:
        """Save router state to JSON file."""
        import json

        state = {
            "strategy": self.strategy.value,
            "epsilon": self.epsilon,
            "success_threshold": self.success_threshold,
            "context_priors": {
                f"{agent_id}:{ctx}": prior.to_dict()
                for (agent_id, ctx), prior in self._context_priors.items()
            },
            "selection_history": self._selection_history[-100:],  # Last 100 only
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Router state saved to {path}")

    def load_state(self, path: str) -> None:
        """Load router state from JSON file."""
        import json

        with open(path, "r") as f:
            state = json.load(f)

        self.strategy = SelectionStrategy(state.get("strategy", "adaptive"))
        self.epsilon = state.get("epsilon", 0.1)
        self.success_threshold = state.get("success_threshold", 0.75)

        # Load context priors
        self._context_priors.clear()
        for key, prior_data in state.get("context_priors", {}).items():
            agent_id, ctx = key.split(":", 1)
            self._context_priors[(agent_id, ctx)] = ThompsonPrior.from_dict(prior_data)

        # Load selection history
        self._selection_history = state.get("selection_history", [])

        logger.info(f"Router state loaded from {path}")


# ============================================================================
# Factory Functions
# ============================================================================

def create_router(
    registry: Optional[RitualAgentRegistry] = None,
    strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE,
    epsilon: float = 0.1,
) -> RitualPhaseRouter:
    """
    Create a RitualPhaseRouter with optional registry.

    If no registry provided, creates one with default agents.
    """
    from .agent_registration import create_registry

    if registry is None:
        registry = create_registry()

    return RitualPhaseRouter(
        registry=registry,
        strategy=strategy,
        epsilon=epsilon,
    )


def create_thompson_router(
    registry: Optional[RitualAgentRegistry] = None
) -> RitualPhaseRouter:
    """Create router with pure Thompson Sampling strategy."""
    return create_router(
        registry=registry,
        strategy=SelectionStrategy.THOMPSON,
    )


def create_adaptive_router(
    registry: Optional[RitualAgentRegistry] = None
) -> RitualPhaseRouter:
    """Create router with adaptive strategy (recommended)."""
    return create_router(
        registry=registry,
        strategy=SelectionStrategy.ADAPTIVE,
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Strategy enum
    'SelectionStrategy',

    # Thompson prior
    'ThompsonPrior',

    # Main router
    'RitualPhaseRouter',

    # Factory functions
    'create_router',
    'create_thompson_router',
    'create_adaptive_router',
]