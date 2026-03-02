"""
Outcome-Based Learning for Context Packing - Phase 6.4
=======================================================

Uses Thompson Sampling to learn optimal MI budgets from query outcomes.

Key Features:
- Track query outcomes (confidence, user feedback)
- Update MI thresholds based on success rates
- Thompson Sampling for budget allocation
- Persistence for learned state

Author: Claude Code
Date: 2025-12-09
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math
import random
import json
import os
from datetime import datetime


# ============================================================================
# Data Types
# ============================================================================

class QueryComplexity(Enum):
    """Query complexity levels matching Phase 6.1."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    RESEARCH = "research"


@dataclass
class BudgetOutcome:
    """Outcome of a query with a specific budget."""
    complexity: QueryComplexity
    budget_used: float
    quality_score: float  # 0.0-1.0
    confidence: float     # Model confidence 0.0-1.0
    feedback: Optional[float] = None  # User feedback 0.0-1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetRecommendation:
    """Recommended budget from Thompson Sampling."""
    complexity: QueryComplexity
    recommended_budget: float
    confidence: float  # Confidence in recommendation
    expected_quality: float  # Expected quality at this budget
    reasoning: str
    alternatives: List[Tuple[float, float]] = field(default_factory=list)  # [(budget, expected_quality)]


# ============================================================================
# Thompson Sampler
# ============================================================================

class ThompsonSampler:
    """
    Thompson Sampler for a single budget level.

    Uses Beta(alpha, beta) posterior for success rate estimation.
    Success = quality_score >= threshold.
    """

    def __init__(
        self,
        base_budget: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        quality_threshold: float = 0.7,
    ):
        """
        Initialize Thompson Sampler.

        Args:
            base_budget: Default MI budget for this level
            alpha: Initial successes (Beta prior)
            beta: Initial failures (Beta prior)
            quality_threshold: Quality score above this is "success"
        """
        self.base_budget = base_budget
        self.alpha = alpha
        self.beta = beta
        self.quality_threshold = quality_threshold

        # Track outcomes
        self._outcomes: List[BudgetOutcome] = []
        self._total_queries = 0
        self._total_successes = 0

    def sample(self) -> float:
        """
        Sample from posterior to get budget recommendation.

        Uses Beta distribution to get success probability,
        then scales budget based on sampled success rate.
        """
        # Sample success probability from Beta posterior
        sampled_rate = random.betavariate(self.alpha, self.beta)

        # Scale budget based on success rate
        # High success rate → can use lower budget (more aggressive)
        # Low success rate → need higher budget (more conservative)
        budget_multiplier = 1.0 + (0.5 - sampled_rate)  # Range: 0.5 to 1.5

        return self.base_budget * budget_multiplier

    def update(self, outcome: BudgetOutcome) -> None:
        """
        Update posterior based on outcome.

        Args:
            outcome: Query outcome with quality score
        """
        self._outcomes.append(outcome)
        self._total_queries += 1

        # Compute effective quality (blend confidence + feedback if available)
        effective_quality = outcome.quality_score
        if outcome.feedback is not None:
            # Weight user feedback more heavily (70% feedback, 30% model)
            effective_quality = 0.7 * outcome.feedback + 0.3 * outcome.quality_score

        # Update Beta posterior
        if effective_quality >= self.quality_threshold:
            self.alpha += effective_quality  # Weighted success
            self._total_successes += 1
        else:
            self.beta += (1.0 - effective_quality)  # Weighted failure

    def expected_quality(self) -> float:
        """Get expected quality (mean of Beta posterior)."""
        return self.alpha / (self.alpha + self.beta)

    def confidence(self) -> float:
        """
        Get confidence in our estimate (inverse variance).

        More observations → higher confidence.
        """
        total = self.alpha + self.beta
        if total < 2:
            return 0.1  # Very low confidence with few samples

        # Variance of Beta: alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
        variance = (self.alpha * self.beta) / ((total ** 2) * (total + 1))

        # Convert to confidence (inverse, capped at 1.0)
        # Max variance for Beta is 0.25 (when alpha=beta=1)
        confidence = 1.0 - min(variance * 4, 1.0)
        return confidence

    def success_rate(self) -> float:
        """Get observed success rate."""
        if self._total_queries == 0:
            return 0.5  # Prior
        return self._total_successes / self._total_queries

    def get_statistics(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            "base_budget": self.base_budget,
            "alpha": self.alpha,
            "beta": self.beta,
            "expected_quality": self.expected_quality(),
            "confidence": self.confidence(),
            "success_rate": self.success_rate(),
            "total_queries": self._total_queries,
            "total_successes": self._total_successes,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "base_budget": self.base_budget,
            "alpha": self.alpha,
            "beta": self.beta,
            "quality_threshold": self.quality_threshold,
            "total_queries": self._total_queries,
            "total_successes": self._total_successes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThompsonSampler":
        """Deserialize from dictionary."""
        sampler = cls(
            base_budget=data["base_budget"],
            alpha=data["alpha"],
            beta=data["beta"],
            quality_threshold=data.get("quality_threshold", 0.7),
        )
        sampler._total_queries = data.get("total_queries", 0)
        sampler._total_successes = data.get("total_successes", 0)
        return sampler


# ============================================================================
# Adaptive Budget Learner
# ============================================================================

class AdaptiveBudgetLearner:
    """
    Learns optimal MI budgets per complexity level using Thompson Sampling.

    Maintains separate Thompson Samplers for each complexity level,
    updates based on query outcomes, and provides budget recommendations.

    Example:
        learner = AdaptiveBudgetLearner()

        # Get budget recommendation
        budget = learner.get_budget(QueryComplexity.MODERATE)

        # After query execution, update with outcome
        learner.update(
            complexity=QueryComplexity.MODERATE,
            budget_used=budget,
            quality_score=0.85,
            confidence=0.9,
            feedback=0.9  # Optional user feedback
        )
    """

    # Default MI budgets per complexity (from Phase 6.1)
    DEFAULT_BUDGETS = {
        QueryComplexity.TRIVIAL: 2.0,
        QueryComplexity.SIMPLE: 3.0,
        QueryComplexity.MODERATE: 5.0,
        QueryComplexity.COMPLEX: 8.0,
        QueryComplexity.RESEARCH: 15.0,
    }

    # Budget bounds (prevent extreme values)
    MIN_BUDGET = 1.0
    MAX_BUDGET = 20.0

    def __init__(
        self,
        quality_threshold: float = 0.7,
        enable_exploration: bool = True,
        exploration_rate: float = 0.1,
    ):
        """
        Initialize adaptive budget learner.

        Args:
            quality_threshold: Quality score above this is "success"
            enable_exploration: Whether to explore suboptimal budgets
            exploration_rate: Probability of random exploration
        """
        self.quality_threshold = quality_threshold
        self.enable_exploration = enable_exploration
        self.exploration_rate = exploration_rate

        # Create Thompson Samplers for each complexity level
        self.samplers: Dict[QueryComplexity, ThompsonSampler] = {}
        for complexity, base_budget in self.DEFAULT_BUDGETS.items():
            self.samplers[complexity] = ThompsonSampler(
                base_budget=base_budget,
                quality_threshold=quality_threshold,
            )

        # Global statistics
        self._total_updates = 0
        self._outcome_history: List[BudgetOutcome] = []

    def get_budget(self, complexity: QueryComplexity) -> float:
        """
        Get recommended MI budget for complexity level.

        Uses Thompson Sampling to balance exploration/exploitation.

        Args:
            complexity: Query complexity level

        Returns:
            Recommended MI budget (bits)
        """
        sampler = self.samplers.get(complexity)
        if sampler is None:
            return self.DEFAULT_BUDGETS.get(complexity, 5.0)

        # Epsilon-greedy exploration
        if self.enable_exploration and random.random() < self.exploration_rate:
            # Explore: random budget within bounds
            base = self.DEFAULT_BUDGETS[complexity]
            budget = base * random.uniform(0.5, 1.5)
        else:
            # Exploit: Thompson Sampling
            budget = sampler.sample()

        # Clamp to bounds
        return max(self.MIN_BUDGET, min(self.MAX_BUDGET, budget))

    def get_recommendation(self, complexity: QueryComplexity) -> BudgetRecommendation:
        """
        Get detailed budget recommendation with reasoning.

        Args:
            complexity: Query complexity level

        Returns:
            BudgetRecommendation with budget, confidence, and reasoning
        """
        sampler = self.samplers.get(complexity)
        if sampler is None:
            default = self.DEFAULT_BUDGETS.get(complexity, 5.0)
            return BudgetRecommendation(
                complexity=complexity,
                recommended_budget=default,
                confidence=0.1,
                expected_quality=0.5,
                reasoning="No sampler available, using default budget",
            )

        # Get recommendation from sampler
        budget = self.get_budget(complexity)

        # Build reasoning
        stats = sampler.get_statistics()
        if stats["total_queries"] < 5:
            reasoning = f"Few observations ({stats['total_queries']}), relying on prior"
        elif stats["success_rate"] >= 0.8:
            reasoning = f"High success rate ({stats['success_rate']:.0%}), can use lower budget"
        elif stats["success_rate"] < 0.5:
            reasoning = f"Low success rate ({stats['success_rate']:.0%}), recommending higher budget"
        else:
            reasoning = f"Balanced success rate ({stats['success_rate']:.0%})"

        # Generate alternatives
        alternatives = []
        for mult in [0.7, 0.85, 1.0, 1.15, 1.3]:
            alt_budget = self.DEFAULT_BUDGETS[complexity] * mult
            alt_budget = max(self.MIN_BUDGET, min(self.MAX_BUDGET, alt_budget))
            # Estimate quality (linear interpolation as approximation)
            est_quality = min(1.0, 0.5 + (mult - 0.7) * 0.5)
            alternatives.append((alt_budget, est_quality))

        return BudgetRecommendation(
            complexity=complexity,
            recommended_budget=budget,
            confidence=sampler.confidence(),
            expected_quality=sampler.expected_quality(),
            reasoning=reasoning,
            alternatives=alternatives,
        )

    def update(
        self,
        complexity: QueryComplexity,
        budget_used: float,
        quality_score: float,
        confidence: float = 1.0,
        feedback: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update learner with query outcome.

        Args:
            complexity: Query complexity level
            budget_used: MI budget that was used
            quality_score: Quality of the result (0.0-1.0)
            confidence: Model confidence in the result (0.0-1.0)
            feedback: Optional user feedback (0.0-1.0)
            metadata: Optional additional metadata
        """
        outcome = BudgetOutcome(
            complexity=complexity,
            budget_used=budget_used,
            quality_score=quality_score,
            confidence=confidence,
            feedback=feedback,
            metadata=metadata or {},
        )

        # Update sampler
        sampler = self.samplers.get(complexity)
        if sampler:
            sampler.update(outcome)

        # Track globally
        self._total_updates += 1
        self._outcome_history.append(outcome)

        # Prune old history (keep last 1000)
        if len(self._outcome_history) > 1000:
            self._outcome_history = self._outcome_history[-1000:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        stats = {
            "total_updates": self._total_updates,
            "quality_threshold": self.quality_threshold,
            "exploration_rate": self.exploration_rate,
            "by_complexity": {},
        }

        for complexity, sampler in self.samplers.items():
            stats["by_complexity"][complexity.value] = sampler.get_statistics()

        # Compute overall metrics
        if self._outcome_history:
            recent = self._outcome_history[-100:]  # Last 100
            stats["recent_avg_quality"] = sum(o.quality_score for o in recent) / len(recent)
            stats["recent_success_rate"] = sum(
                1 for o in recent if o.quality_score >= self.quality_threshold
            ) / len(recent)
        else:
            stats["recent_avg_quality"] = 0.0
            stats["recent_success_rate"] = 0.0

        return stats

    def save(self, path: str) -> None:
        """
        Save learning state to file.

        Args:
            path: File path for JSON state
        """
        state = {
            "version": "1.0",
            "quality_threshold": self.quality_threshold,
            "exploration_rate": self.exploration_rate,
            "total_updates": self._total_updates,
            "samplers": {
                complexity.value: sampler.to_dict()
                for complexity, sampler in self.samplers.items()
            },
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load learning state from file.

        Args:
            path: File path for JSON state
        """
        if not os.path.exists(path):
            return  # No state to load

        with open(path, "r") as f:
            state = json.load(f)

        self.quality_threshold = state.get("quality_threshold", 0.7)
        self.exploration_rate = state.get("exploration_rate", 0.1)
        self._total_updates = state.get("total_updates", 0)

        # Restore samplers
        for complexity_str, sampler_data in state.get("samplers", {}).items():
            complexity = QueryComplexity(complexity_str)
            self.samplers[complexity] = ThompsonSampler.from_dict(sampler_data)


# ============================================================================
# Convenience Functions
# ============================================================================

# Global learner instance
_global_learner: Optional[AdaptiveBudgetLearner] = None


def get_learner() -> AdaptiveBudgetLearner:
    """Get or create global adaptive budget learner."""
    global _global_learner
    if _global_learner is None:
        _global_learner = AdaptiveBudgetLearner()
    return _global_learner


def get_adaptive_budget(complexity: str) -> float:
    """
    Get adaptive MI budget for complexity level.

    Convenience function using global learner.

    Args:
        complexity: Complexity level string (trivial/simple/moderate/complex/research)

    Returns:
        Recommended MI budget (bits)
    """
    learner = get_learner()
    complexity_enum = QueryComplexity(complexity.lower())
    return learner.get_budget(complexity_enum)


def record_outcome(
    complexity: str,
    budget_used: float,
    quality_score: float,
    confidence: float = 1.0,
    feedback: Optional[float] = None,
) -> None:
    """
    Record query outcome for learning.

    Convenience function using global learner.

    Args:
        complexity: Complexity level string
        budget_used: MI budget that was used
        quality_score: Quality of the result (0.0-1.0)
        confidence: Model confidence (0.0-1.0)
        feedback: Optional user feedback (0.0-1.0)
    """
    learner = get_learner()
    complexity_enum = QueryComplexity(complexity.lower())
    learner.update(
        complexity=complexity_enum,
        budget_used=budget_used,
        quality_score=quality_score,
        confidence=confidence,
        feedback=feedback,
    )


def get_learning_statistics() -> Dict[str, Any]:
    """Get global learning statistics."""
    return get_learner().get_statistics()
