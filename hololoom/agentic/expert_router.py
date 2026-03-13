"""
Expert Router - Phase 7.2
=========================

Thompson Sampling-based agent selection with performance tracking.
Enables intelligent routing to the best agent for each capability.

Key Features:
- Thompson Sampling for exploration/exploitation balance
- Performance history tracking per agent
- Multiple selection strategies (FASTEST, RELIABLE, BALANCED, THOMPSON)
- Top-K selection for ensemble decisions

Author: Claude Code
Date: 2025-12-09
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import random
import math
from datetime import datetime
from collections import defaultdict

from hololoom.bandits.beta_arm import BetaArm
from .protocol import AgentCapability, AgentStatus, SelectionStrategy

if TYPE_CHECKING:
    from .multi_agent import AgentRegistry, AgentInfo


@dataclass
class ThompsonPrior:
    """
    Beta distribution prior for Thompson Sampling.

    Delegates Beta math to BetaArm (canonical implementation).
    Maintains α (successes) and β (failures) for each agent.
    """
    _arm: BetaArm = field(default_factory=BetaArm)

    @property
    def alpha(self) -> float:
        return self._arm.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._arm.alpha = value

    @property
    def beta(self) -> float:
        return self._arm.beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._arm.beta = value

    def sample(self) -> float:
        """Sample from Beta(alpha, beta) distribution."""
        return self._arm.sample()

    def expected_value(self) -> float:
        """Expected value E[X] = α / (α + β)."""
        return self._arm.expected_value()

    def update_success(self, confidence: float = 1.0) -> None:
        """Update prior after successful outcome."""
        self._arm.update(success=True, weight=confidence)

    def update_failure(self, confidence: float = 0.0) -> None:
        """Update prior after failed outcome."""
        self._arm.update(success=False, weight=(1.0 - confidence))

    def total_observations(self) -> int:
        """Total number of observations (excluding prior)."""
        return int(self._arm.total_observations())


@dataclass
class AgentScore:
    """
    Comprehensive score for an agent.

    Used to rank and select agents based on multiple factors.
    """
    agent_id: str
    total_score: float
    performance_score: float      # success_rate * avg_confidence
    availability_score: float     # 1.0 (IDLE), 0.5 (BUSY), 0.0 (OFFLINE)
    reliability_score: float      # Based on success/failure history
    thompson_sample: float        # Sample from Thompson prior
    expected_reward: float        # E[X] from Thompson prior

    def __lt__(self, other: "AgentScore") -> bool:
        return self.total_score < other.total_score


@dataclass
class RoutingDecision:
    """
    Result of the routing decision.

    Contains selected agent, score, and alternatives for ensemble.
    """
    selected_agent: str
    score: AgentScore
    alternatives: List[AgentScore] = field(default_factory=list)
    routing_reason: str = ""
    used_thompson: bool = False
    routing_time_ms: float = 0.0


class ExpertRouter:
    """
    Thompson Sampling-based expert agent router.

    Routes tasks to the best available agent using a combination of:
    - Historical performance (success rate, latency)
    - Availability status
    - Thompson Sampling for exploration/exploitation

    Example:
        router = ExpertRouter(registry)

        # Route to best agent
        decision = router.route(AgentCapability.CODE_REVIEW)
        agent_id = decision.selected_agent

        # After task completion, update performance
        router.update_performance(agent_id, success=True, confidence=0.9)
    """

    # Strategy weight configurations
    STRATEGY_WEIGHTS = {
        SelectionStrategy.FASTEST: {
            "performance": 0.1,
            "availability": 0.7,
            "reliability": 0.2,
            "thompson": 0.0
        },
        SelectionStrategy.RELIABLE: {
            "performance": 0.5,
            "availability": 0.2,
            "reliability": 0.3,
            "thompson": 0.0
        },
        SelectionStrategy.BALANCED: {
            "performance": 0.4,
            "availability": 0.3,
            "reliability": 0.3,
            "thompson": 0.0
        },
        SelectionStrategy.THOMPSON: {
            "performance": 0.0,
            "availability": 0.0,
            "reliability": 0.0,
            "thompson": 1.0
        },
        SelectionStrategy.RANDOM: {
            "performance": 0.0,
            "availability": 0.0,
            "reliability": 0.0,
            "thompson": 0.0  # Random ignores weights
        },
    }

    def __init__(
        self,
        registry: AgentRegistry,
        enable_thompson: bool = True,
        exploration_bonus: float = 0.1
    ):
        """
        Initialize expert router.

        Args:
            registry: Agent registry for discovery
            enable_thompson: Enable Thompson Sampling
            exploration_bonus: Bonus for less-explored agents
        """
        self._registry = registry
        self._enable_thompson = enable_thompson
        self._exploration_bonus = exploration_bonus

        # Thompson priors per agent
        self._priors: Dict[str, ThompsonPrior] = defaultdict(ThompsonPrior)

        # Performance history per agent
        self._performance: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "successes": 0,
                "failures": 0,
                "total_confidence": 0.0,
                "total_latency_ms": 0.0,
                "last_used": None
            }
        )

        # Routing statistics
        self._routing_count = 0
        self._thompson_selections = 0

    def route(
        self,
        required_capability: AgentCapability,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        top_k: int = 1,
        exclude_agents: Optional[List[str]] = None
    ) -> RoutingDecision:
        """
        Route to the best agent for a capability.

        Args:
            required_capability: Required capability
            strategy: Selection strategy
            top_k: Number of alternatives to return (for ensemble)
            exclude_agents: Agent IDs to exclude

        Returns:
            RoutingDecision with selected agent and alternatives
        """
        start_time = datetime.now()

        # Find candidate agents
        candidates = self._registry.find_by_capability(
            required_capability,
            only_available=True
        )

        # Filter excluded agents
        if exclude_agents:
            candidates = [a for a in candidates if a.id not in exclude_agents]

        if not candidates:
            # No candidates - return empty decision
            return RoutingDecision(
                selected_agent="",
                score=AgentScore(
                    agent_id="",
                    total_score=0.0,
                    performance_score=0.0,
                    availability_score=0.0,
                    reliability_score=0.0,
                    thompson_sample=0.0,
                    expected_reward=0.0
                ),
                routing_reason=f"No available agents with capability {required_capability.value}"
            )

        # Score all candidates
        scores = [self._score_agent(agent, strategy) for agent in candidates]

        # Sort by total score (descending)
        scores.sort(reverse=True)

        # Select top agent
        selected = scores[0]
        alternatives = scores[1:top_k] if top_k > 1 else []

        # Track statistics
        self._routing_count += 1
        if strategy == SelectionStrategy.THOMPSON:
            self._thompson_selections += 1

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        return RoutingDecision(
            selected_agent=selected.agent_id,
            score=selected,
            alternatives=alternatives,
            routing_reason=self._build_reason(selected, strategy, required_capability),
            used_thompson=strategy == SelectionStrategy.THOMPSON or (
                self._enable_thompson and selected.thompson_sample > 0.5
            ),
            routing_time_ms=elapsed_ms
        )

    def _score_agent(
        self,
        agent: AgentInfo,
        strategy: SelectionStrategy
    ) -> AgentScore:
        """Score an agent based on the strategy."""
        weights = self.STRATEGY_WEIGHTS[strategy]

        # Calculate component scores
        performance_score = self._calc_performance_score(agent)
        availability_score = self._calc_availability_score(agent)
        reliability_score = self._calc_reliability_score(agent)
        thompson_sample = self._calc_thompson_sample(agent.id)

        # Get expected reward from prior
        prior = self._priors[agent.id]
        expected_reward = prior.expected_value()

        # Combine scores based on weights
        if strategy == SelectionStrategy.RANDOM:
            total_score = random.random()
        elif strategy == SelectionStrategy.THOMPSON:
            total_score = thompson_sample
        else:
            total_score = (
                weights["performance"] * performance_score +
                weights["availability"] * availability_score +
                weights["reliability"] * reliability_score +
                weights["thompson"] * thompson_sample
            )

            # Add exploration bonus for less-explored agents
            if self._enable_thompson:
                obs = prior.total_observations()
                if obs < 10:
                    exploration_factor = (10 - obs) / 10 * self._exploration_bonus
                    total_score += exploration_factor

        return AgentScore(
            agent_id=agent.id,
            total_score=total_score,
            performance_score=performance_score,
            availability_score=availability_score,
            reliability_score=reliability_score,
            thompson_sample=thompson_sample,
            expected_reward=expected_reward
        )

    def _calc_performance_score(self, agent: AgentInfo) -> float:
        """Calculate performance score from agent metrics."""
        perf = self._performance[agent.id]

        total = perf["successes"] + perf["failures"]
        if total == 0:
            return 0.5  # Neutral for new agents

        success_rate = perf["successes"] / total
        avg_confidence = perf["total_confidence"] / total if total > 0 else 0.5

        return success_rate * avg_confidence

    def _calc_availability_score(self, agent: AgentInfo) -> float:
        """Calculate availability score from agent status."""
        status_scores = {
            AgentStatus.IDLE: 1.0,
            AgentStatus.BUSY: 0.5,
            AgentStatus.WAITING: 0.7,
            AgentStatus.ERROR: 0.1,
            AgentStatus.OFFLINE: 0.0,
        }
        return status_scores.get(agent.status, 0.5)

    def _calc_reliability_score(self, agent: AgentInfo) -> float:
        """Calculate reliability score from success/failure ratio."""
        perf = self._performance[agent.id]
        total = perf["successes"] + perf["failures"]

        if total == 0:
            # Use agent's reported success rate for new agents
            return agent.success_rate if agent.success_rate > 0 else 0.5

        # Laplace smoothing: (successes + 1) / (total + 2)
        return (perf["successes"] + 1) / (total + 2)

    def _calc_thompson_sample(self, agent_id: str) -> float:
        """Sample from Thompson prior for agent."""
        if not self._enable_thompson:
            return 0.5

        prior = self._priors[agent_id]
        return prior.sample()

    def _build_reason(
        self,
        score: AgentScore,
        strategy: SelectionStrategy,
        capability: AgentCapability
    ) -> str:
        """Build human-readable routing reason."""
        parts = [
            f"Selected {score.agent_id} for {capability.value}",
            f"Strategy: {strategy.value}",
            f"Score: {score.total_score:.3f}",
        ]

        if strategy == SelectionStrategy.THOMPSON:
            parts.append(f"Thompson sample: {score.thompson_sample:.3f}")
            parts.append(f"Expected: {score.expected_reward:.3f}")

        return " | ".join(parts)

    def update_performance(
        self,
        agent_id: str,
        success: bool,
        confidence: float = 0.5,
        latency_ms: float = 0.0
    ) -> None:
        """
        Update agent performance after task completion.

        Args:
            agent_id: ID of the agent
            success: Whether task succeeded
            confidence: Confidence level (0.0-1.0)
            latency_ms: Task latency in milliseconds
        """
        perf = self._performance[agent_id]
        prior = self._priors[agent_id]

        if success:
            perf["successes"] += 1
            prior.update_success(confidence)
        else:
            perf["failures"] += 1
            prior.update_failure(confidence)

        perf["total_confidence"] += confidence
        perf["total_latency_ms"] += latency_ms
        perf["last_used"] = datetime.now().isoformat()

    def get_agent_prior(self, agent_id: str) -> Dict[str, float]:
        """Get Thompson prior for an agent."""
        prior = self._priors[agent_id]
        return {
            "alpha": prior.alpha,
            "beta": prior.beta,
            "expected_value": prior.expected_value(),
            "total_observations": prior.total_observations()
        }

    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance history for an agent."""
        perf = self._performance[agent_id]
        total = perf["successes"] + perf["failures"]

        return {
            "successes": perf["successes"],
            "failures": perf["failures"],
            "success_rate": perf["successes"] / total if total > 0 else 0.0,
            "avg_confidence": perf["total_confidence"] / total if total > 0 else 0.0,
            "avg_latency_ms": perf["total_latency_ms"] / total if total > 0 else 0.0,
            "total_tasks": total,
            "last_used": perf["last_used"]
        }

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get overall routing statistics."""
        all_performances = {
            agent_id: self.get_agent_performance(agent_id)
            for agent_id in self._performance.keys()
        }

        total_tasks = sum(p["total_tasks"] for p in all_performances.values())

        return {
            "total_routings": self._routing_count,
            "thompson_selections": self._thompson_selections,
            "thompson_rate": (
                self._thompson_selections / self._routing_count
                if self._routing_count > 0 else 0.0
            ),
            "agents_tracked": len(self._performance),
            "total_tasks_completed": total_tasks,
            "agent_performances": all_performances
        }

    def reset_agent_history(self, agent_id: str) -> None:
        """Reset performance history and prior for an agent."""
        self._priors[agent_id] = ThompsonPrior()
        self._performance[agent_id] = {
            "successes": 0,
            "failures": 0,
            "total_confidence": 0.0,
            "total_latency_ms": 0.0,
            "last_used": None
        }

    def get_top_agents(
        self,
        capability: AgentCapability,
        k: int = 3,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED
    ) -> List[Tuple[str, float]]:
        """
        Get top K agents for a capability.

        Args:
            capability: Required capability
            k: Number of agents to return
            strategy: Scoring strategy

        Returns:
            List of (agent_id, score) tuples
        """
        decision = self.route(capability, strategy, top_k=k)

        results = [(decision.selected_agent, decision.score.total_score)]
        results.extend([
            (alt.agent_id, alt.total_score)
            for alt in decision.alternatives
        ])

        return results[:k]


# Convenience function
def create_expert_router(
    registry: AgentRegistry,
    enable_thompson: bool = True
) -> ExpertRouter:
    """
    Create an expert router for agent selection.

    Args:
        registry: Agent registry
        enable_thompson: Enable Thompson Sampling

    Returns:
        Configured ExpertRouter
    """
    return ExpertRouter(registry, enable_thompson)
