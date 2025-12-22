"""
Learning Effectiveness Metrics
==============================

Metrics for evaluating how well the learning loop actually improves over time.

Validates that Thompson Sampling and pattern learning are effective.

Metrics Implemented:
- Improvement Rate: Quality delta after N interactions
- Thompson Sampling Regret: Cumulative regret vs optimal
- Pattern Discovery Rate: New patterns found per 100 queries
- Convergence Speed: Queries to reach 90% of max quality
- Retention Score: Performance on old topics after learning new

Integration Points:
- Tracks Thompson Sampling priors (α, β)
- Monitors hot pattern feedback effectiveness
- Measures recursive learning improvements

Author: HoloLoom Architecture Team
Date: December 2025
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LearningMetrics:
    """
    Complete learning effectiveness metrics.

    Attributes:
        improvement_rate: Quality improvement per interaction (0-1)
        thompson_sampling_regret: Cumulative regret vs optimal strategy
        pattern_discovery_rate: New patterns per 100 queries
        convergence_speed: Number of queries to reach 90% max quality
        retention_score: Performance retention on old topics (0-1)
        total_interactions: Number of learning interactions
        metadata: Additional evaluation metadata
    """
    improvement_rate: float = 0.0
    thompson_sampling_regret: float = 0.0
    pattern_discovery_rate: float = 0.0
    convergence_speed: int = 0
    retention_score: float = 0.0
    total_interactions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "improvement_rate": self.improvement_rate,
            "thompson_sampling_regret": self.thompson_sampling_regret,
            "pattern_discovery_rate": self.pattern_discovery_rate,
            "convergence_speed": self.convergence_speed,
            "retention_score": self.retention_score,
            "total_interactions": self.total_interactions,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Human-readable summary of learning metrics."""
        lines = [
            f"Learning Metrics ({self.total_interactions} interactions)",
            f"  Improvement Rate: {self.improvement_rate:.4f}",
            f"  Thompson Regret: {self.thompson_sampling_regret:.4f}",
            f"  Pattern Discovery: {self.pattern_discovery_rate:.2f} per 100 queries",
            f"  Convergence Speed: {self.convergence_speed} queries",
            f"  Retention Score: {self.retention_score:.4f}",
        ]
        return "\n".join(lines)

    def is_effective(
        self,
        min_improvement: float = 0.1,
        max_regret: float = 0.5,
        min_retention: float = 0.8,
    ) -> bool:
        """Check if learning is effective."""
        return (
            self.improvement_rate >= min_improvement and
            self.thompson_sampling_regret <= max_regret and
            self.retention_score >= min_retention
        )


@dataclass
class LearningInteraction:
    """A single learning interaction with quality feedback."""
    query_id: str
    quality_score: float  # 0-1 quality of response
    confidence: float  # System confidence
    topic: Optional[str] = None  # Topic for retention tracking
    patterns_discovered: int = 0  # New patterns found
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThompsonSamplingState:
    """State of Thompson Sampling for a single arm/action."""
    arm_id: str
    alpha: float  # Success count + prior
    beta: float  # Failure count + prior
    pulls: int = 0  # Total times this arm was pulled

    @property
    def expected_reward(self) -> float:
        """Expected reward: α / (α + β)."""
        return self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.5

    @property
    def variance(self) -> float:
        """Variance of Beta distribution."""
        total = self.alpha + self.beta
        if total == 0:
            return 0.25
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))


# =============================================================================
# Individual Metric Functions
# =============================================================================

def improvement_rate(
    interactions: List[LearningInteraction],
    window_size: int = 100,
) -> float:
    """
    Calculate quality improvement rate over time.

    Compares average quality in first window vs last window.

    Args:
        interactions: List of learning interactions ordered by time
        window_size: Number of interactions per window

    Returns:
        Improvement rate (positive = improving, negative = degrading)
    """
    if len(interactions) < 2 * window_size:
        # Not enough data for comparison
        if len(interactions) < 2:
            return 0.0
        # Use halves
        mid = len(interactions) // 2
        first_scores = [i.quality_score for i in interactions[:mid]]
        last_scores = [i.quality_score for i in interactions[mid:]]
    else:
        first_scores = [i.quality_score for i in interactions[:window_size]]
        last_scores = [i.quality_score for i in interactions[-window_size:]]

    first_avg = sum(first_scores) / len(first_scores)
    last_avg = sum(last_scores) / len(last_scores)

    return last_avg - first_avg


def thompson_sampling_regret(
    states: List[ThompsonSamplingState],
    actions_taken: List[str],  # Sequence of arm_ids chosen
    rewards: List[float],  # Rewards received
) -> float:
    """
    Calculate cumulative regret for Thompson Sampling.

    Regret = Σ (optimal_reward - actual_reward)

    Uses hindsight optimal (best arm in retrospect).

    Args:
        states: Current Thompson Sampling state for each arm
        actions_taken: Sequence of actions taken
        rewards: Rewards received for each action

    Returns:
        Cumulative regret (lower is better, 0 = optimal)
    """
    if not states or not actions_taken or not rewards:
        return 0.0

    # Find optimal arm (best expected reward)
    optimal_reward = max(s.expected_reward for s in states)

    # Calculate cumulative regret
    regret = 0.0
    for reward in rewards:
        regret += optimal_reward - reward

    return regret


def thompson_sampling_regret_per_round(
    optimal_rewards: List[float],  # Optimal reward at each round
    actual_rewards: List[float],  # Actual reward at each round
) -> List[float]:
    """
    Calculate per-round regret for Thompson Sampling.

    Useful for plotting regret over time.

    Args:
        optimal_rewards: Optimal reward at each round
        actual_rewards: Actual reward at each round

    Returns:
        List of per-round regrets
    """
    if len(optimal_rewards) != len(actual_rewards):
        raise ValueError("optimal_rewards and actual_rewards must have same length")

    return [opt - act for opt, act in zip(optimal_rewards, actual_rewards)]


def pattern_discovery_rate(
    interactions: List[LearningInteraction],
    normalize_per: int = 100,
) -> float:
    """
    Calculate pattern discovery rate.

    Args:
        interactions: List of learning interactions
        normalize_per: Normalize to discoveries per N queries

    Returns:
        Patterns discovered per normalize_per queries
    """
    if not interactions:
        return 0.0

    total_patterns = sum(i.patterns_discovered for i in interactions)
    rate = total_patterns / len(interactions) * normalize_per

    return rate


def convergence_speed(
    interactions: List[LearningInteraction],
    target_quality: Optional[float] = None,
    target_percentile: float = 0.9,
) -> int:
    """
    Calculate number of interactions to reach target quality.

    Args:
        interactions: List of learning interactions ordered by time
        target_quality: Target quality score (if None, use 90% of max)
        target_percentile: Percentile of max quality to target

    Returns:
        Number of queries to reach target quality (0 if never reached)
    """
    if not interactions:
        return 0

    # Determine target quality
    max_quality = max(i.quality_score for i in interactions)
    if target_quality is None:
        target_quality = max_quality * target_percentile

    # Find first interaction that reaches target
    for idx, interaction in enumerate(interactions):
        if interaction.quality_score >= target_quality:
            return idx + 1

    return 0  # Never reached


def retention_score(
    old_topic_interactions: List[LearningInteraction],
    new_topic_interactions: List[LearningInteraction],
) -> float:
    """
    Calculate retention score for old topics after learning new ones.

    Measures how well the system maintains performance on old topics
    while learning new ones.

    Args:
        old_topic_interactions: Interactions on old topics after new learning
        new_topic_interactions: Interactions on new topics (for baseline)

    Returns:
        Retention score (0-1, higher = better retention)
    """
    if not old_topic_interactions:
        return 1.0  # No old topics to forget

    # Average quality on old topics
    old_avg = sum(i.quality_score for i in old_topic_interactions) / len(old_topic_interactions)

    # Compare to new topic performance
    if new_topic_interactions:
        new_avg = sum(i.quality_score for i in new_topic_interactions) / len(new_topic_interactions)
        # Retention = old_quality / new_quality (capped at 1.0)
        return min(old_avg / new_avg, 1.0) if new_avg > 0 else old_avg
    else:
        # No new topics, return old performance directly
        return old_avg


def catastrophic_forgetting_score(
    before_learning: Dict[str, float],  # topic -> quality before
    after_learning: Dict[str, float],  # topic -> quality after
) -> float:
    """
    Measure catastrophic forgetting across topics.

    Args:
        before_learning: Quality scores per topic before new learning
        after_learning: Quality scores per topic after new learning

    Returns:
        Forgetting score (0 = no forgetting, 1 = complete forgetting)
    """
    common_topics = set(before_learning.keys()) & set(after_learning.keys())
    if not common_topics:
        return 0.0

    total_loss = 0.0
    for topic in common_topics:
        before = before_learning[topic]
        after = after_learning[topic]
        # Loss = max(0, before - after) / before
        if before > 0:
            loss = max(0, before - after) / before
            total_loss += loss

    return total_loss / len(common_topics)


# =============================================================================
# Evaluator Class
# =============================================================================

class LearningEvaluator:
    """
    Comprehensive learning effectiveness evaluator.

    Supports:
    - Improvement rate tracking
    - Thompson Sampling regret analysis
    - Pattern discovery monitoring
    - Convergence speed measurement
    - Retention/forgetting detection
    """

    def __init__(
        self,
        window_size: int = 100,
        target_quality_percentile: float = 0.9,
    ):
        """
        Initialize learning evaluator.

        Args:
            window_size: Window size for rolling metrics
            target_quality_percentile: Target percentile for convergence
        """
        self.window_size = window_size
        self.target_quality_percentile = target_quality_percentile

    def evaluate(
        self,
        interactions: List[LearningInteraction],
        thompson_states: Optional[List[ThompsonSamplingState]] = None,
        actions_taken: Optional[List[str]] = None,
        rewards: Optional[List[float]] = None,
    ) -> LearningMetrics:
        """
        Evaluate learning effectiveness across all dimensions.

        Args:
            interactions: List of learning interactions
            thompson_states: Thompson Sampling state for regret calculation
            actions_taken: Actions taken for regret calculation
            rewards: Rewards received for regret calculation

        Returns:
            LearningMetrics with all computed metrics
        """
        if not interactions:
            return LearningMetrics()

        # Group by topic for retention analysis
        old_topic_interactions = [i for i in interactions if i.topic == "old"]
        new_topic_interactions = [i for i in interactions if i.topic == "new"]

        # Calculate Thompson regret if data available
        ts_regret = 0.0
        if thompson_states and actions_taken and rewards:
            ts_regret = thompson_sampling_regret(thompson_states, actions_taken, rewards)

        return LearningMetrics(
            improvement_rate=improvement_rate(interactions, self.window_size),
            thompson_sampling_regret=ts_regret,
            pattern_discovery_rate=pattern_discovery_rate(interactions),
            convergence_speed=convergence_speed(
                interactions,
                target_percentile=self.target_quality_percentile
            ),
            retention_score=retention_score(old_topic_interactions, new_topic_interactions),
            total_interactions=len(interactions),
            metadata={
                "window_size": self.window_size,
                "target_quality_percentile": self.target_quality_percentile,
            }
        )

    def evaluate_learn_query_verify_cycle(
        self,
        store_interactions: List[LearningInteraction],
        query_interactions: List[LearningInteraction],
    ) -> Dict[str, Any]:
        """
        Evaluate Learn-Query-Verify cycle effectiveness.

        Tests: Store knowledge → Query → Verify retrieval improved

        Args:
            store_interactions: Interactions when storing new knowledge
            query_interactions: Interactions when querying stored knowledge

        Returns:
            Dict with cycle effectiveness metrics
        """
        if not store_interactions or not query_interactions:
            return {"effective": False, "reason": "insufficient_data"}

        # Check if query quality improves after storing
        pre_store_queries = [
            q for q in query_interactions
            if q.timestamp < min(s.timestamp for s in store_interactions)
        ]
        post_store_queries = [
            q for q in query_interactions
            if q.timestamp > max(s.timestamp for s in store_interactions)
        ]

        if not pre_store_queries or not post_store_queries:
            return {
                "effective": True,
                "reason": "cannot_compare",
                "note": "No pre/post comparison available"
            }

        pre_avg = sum(q.quality_score for q in pre_store_queries) / len(pre_store_queries)
        post_avg = sum(q.quality_score for q in post_store_queries) / len(post_store_queries)

        improvement = post_avg - pre_avg

        return {
            "effective": improvement > 0.05,
            "improvement": improvement,
            "pre_store_quality": pre_avg,
            "post_store_quality": post_avg,
            "pre_count": len(pre_store_queries),
            "post_count": len(post_store_queries),
        }

    def evaluate_feedback_loop(
        self,
        positive_feedback_interactions: List[LearningInteraction],
        negative_feedback_interactions: List[LearningInteraction],
    ) -> Dict[str, Any]:
        """
        Evaluate if feedback loop works (positive → weight increase).

        Args:
            positive_feedback_interactions: Interactions with positive feedback
            negative_feedback_interactions: Interactions with negative feedback

        Returns:
            Dict with feedback loop effectiveness metrics
        """
        if not positive_feedback_interactions:
            return {"effective": False, "reason": "no_positive_feedback"}

        # Check if positive feedback leads to higher subsequent quality
        pos_avg = sum(i.quality_score for i in positive_feedback_interactions) / len(positive_feedback_interactions)

        if negative_feedback_interactions:
            neg_avg = sum(i.quality_score for i in negative_feedback_interactions) / len(negative_feedback_interactions)
        else:
            neg_avg = 0.5  # Baseline

        differential = pos_avg - neg_avg

        return {
            "effective": differential > 0.1,
            "positive_avg_quality": pos_avg,
            "negative_avg_quality": neg_avg,
            "differential": differential,
            "positive_count": len(positive_feedback_interactions),
            "negative_count": len(negative_feedback_interactions),
        }

    def evaluate_hot_pattern_effectiveness(
        self,
        hot_pattern_queries: List[LearningInteraction],
        cold_pattern_queries: List[LearningInteraction],
    ) -> Dict[str, Any]:
        """
        Evaluate if hot pattern boosting is effective.

        Args:
            hot_pattern_queries: Queries using hot patterns
            cold_pattern_queries: Queries using cold patterns

        Returns:
            Dict with hot pattern effectiveness metrics
        """
        if not hot_pattern_queries:
            return {"effective": False, "reason": "no_hot_patterns"}

        hot_avg = sum(i.quality_score for i in hot_pattern_queries) / len(hot_pattern_queries)

        if cold_pattern_queries:
            cold_avg = sum(i.quality_score for i in cold_pattern_queries) / len(cold_pattern_queries)
        else:
            cold_avg = 0.5

        boost = hot_avg - cold_avg

        return {
            "effective": boost > 0.05,
            "hot_avg_quality": hot_avg,
            "cold_avg_quality": cold_avg,
            "boost": boost,
            "hot_count": len(hot_pattern_queries),
            "cold_count": len(cold_pattern_queries),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_learning_interaction(
    query_id: str,
    quality_score: float,
    confidence: float,
    topic: Optional[str] = None,
    patterns_discovered: int = 0,
    **metadata,
) -> LearningInteraction:
    """
    Create a LearningInteraction object.

    Args:
        query_id: Unique query identifier
        quality_score: Quality of response (0-1)
        confidence: System confidence (0-1)
        topic: Optional topic for retention tracking
        patterns_discovered: Number of new patterns discovered
        **metadata: Additional metadata

    Returns:
        LearningInteraction instance
    """
    import time
    return LearningInteraction(
        query_id=query_id,
        quality_score=quality_score,
        confidence=confidence,
        topic=topic,
        patterns_discovered=patterns_discovered,
        timestamp=time.time(),
        metadata=metadata,
    )


def evaluate_simple(
    quality_scores: List[float],
    window_size: int = 100,
) -> LearningMetrics:
    """
    Simple learning evaluation from quality scores only.

    Args:
        quality_scores: List of quality scores over time
        window_size: Window size for rolling metrics

    Returns:
        LearningMetrics
    """
    interactions = [
        LearningInteraction(
            query_id=f"q_{i}",
            quality_score=score,
            confidence=score,  # Use quality as confidence proxy
            timestamp=float(i),
        )
        for i, score in enumerate(quality_scores)
    ]

    evaluator = LearningEvaluator(window_size=window_size)
    return evaluator.evaluate(interactions)


def is_learning_effective(
    quality_scores: List[float],
    min_improvement: float = 0.1,
) -> bool:
    """
    Quick check if learning is effective.

    Args:
        quality_scores: List of quality scores over time
        min_improvement: Minimum improvement threshold

    Returns:
        True if learning is effective
    """
    if len(quality_scores) < 4:
        return False

    mid = len(quality_scores) // 2
    first_half_avg = sum(quality_scores[:mid]) / mid
    second_half_avg = sum(quality_scores[mid:]) / (len(quality_scores) - mid)

    return second_half_avg - first_half_avg >= min_improvement
