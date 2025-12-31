"""
HoloLoom Memory - Shard Contribution Tracking (Phase 3)
=========================================================
Tracks which memory shards contribute to successful query outcomes,
enabling feedback-based retrieval optimization.

Architecture:
    ShardContributionTracker: Records shard-query-outcome relationships
    ContributionRecord: Single query's shard contributions
    ShardStats: Aggregated statistics for each shard
    RetrievalBooster: Applies learned boosts during retrieval

Philosophy:
    "Memory that helps should be remembered, memory that doesn't should fade."

    We track which shards were retrieved for each query and what outcome
    resulted. Shards consistently associated with high-confidence outcomes
    get boosted in future retrieval. Shards that don't contribute fade.

Integration Points:
    1. After retrieval: record_retrieval(query_id, shard_ids)
    2. After weaving: record_outcome(query_id, confidence, feedback)
    3. During retrieval: get_boost_factor(shard_id) -> float

Author: Claude Code
Date: December 2025 (Phase 3 - Outcome→Retrieval Loop)
"""

import logging
import time
import math
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class OutcomeQuality(Enum):
    """Quality tier of query outcome."""
    EXCELLENT = "excellent"  # confidence >= 0.9
    GOOD = "good"           # confidence >= 0.75
    FAIR = "fair"           # confidence >= 0.5
    POOR = "poor"           # confidence < 0.5


@dataclass
class ContributionRecord:
    """
    Record of shards contributing to a single query.

    Captures the relationship between retrieved shards and query outcome.
    """
    query_id: str
    query_text: str
    shard_ids: List[str]  # Shards retrieved for this query
    retrieval_scores: Dict[str, float]  # shard_id -> retrieval score
    timestamp: float = field(default_factory=time.time)

    # Outcome (filled in after weaving completes)
    outcome_confidence: Optional[float] = None
    outcome_quality: Optional[OutcomeQuality] = None
    user_feedback: Optional[float] = None  # Optional explicit feedback (-1 to 1)
    tool_used: Optional[str] = None

    @property
    def has_outcome(self) -> bool:
        """Check if outcome has been recorded."""
        return self.outcome_confidence is not None

    def record_outcome(
        self,
        confidence: float,
        feedback: Optional[float] = None,
        tool_used: Optional[str] = None
    ):
        """Record the query outcome."""
        self.outcome_confidence = confidence
        self.outcome_quality = self._classify_quality(confidence)
        self.user_feedback = feedback
        self.tool_used = tool_used

    @staticmethod
    def _classify_quality(confidence: float) -> OutcomeQuality:
        """Classify confidence into quality tier."""
        if confidence >= 0.9:
            return OutcomeQuality.EXCELLENT
        elif confidence >= 0.75:
            return OutcomeQuality.GOOD
        elif confidence >= 0.5:
            return OutcomeQuality.FAIR
        else:
            return OutcomeQuality.POOR

    def get_contribution_score(self) -> float:
        """
        Calculate contribution score for this record.

        Combines:
        - Outcome confidence (primary signal)
        - User feedback (if available, weighted heavily)
        - Recency decay

        Returns:
            Score in range [0, 1]
        """
        if not self.has_outcome:
            return 0.0

        # Base score from confidence
        base_score = self.outcome_confidence or 0.0

        # User feedback adjustment (if available)
        if self.user_feedback is not None:
            # Feedback in [-1, 1], scale to [0, 1]
            feedback_score = (self.user_feedback + 1) / 2
            # Heavy weight on explicit feedback (70% feedback, 30% confidence)
            base_score = 0.7 * feedback_score + 0.3 * base_score

        return base_score


@dataclass
class ShardStats:
    """
    Aggregated statistics for a single shard's contribution history.

    Tracks how often a shard contributes to successful outcomes.
    """
    shard_id: str

    # Counters
    retrieval_count: int = 0  # Times retrieved
    success_count: int = 0    # Times in high-quality outcomes (>= GOOD)
    excellent_count: int = 0  # Times in excellent outcomes
    poor_count: int = 0       # Times in poor outcomes

    # Aggregated scores
    total_contribution: float = 0.0  # Sum of contribution scores
    avg_contribution: float = 0.0     # Average contribution score
    total_retrieval_score: float = 0.0  # Sum of retrieval scores

    # Temporal tracking
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Decay tracking
    decayed_contribution: float = 0.0  # Contribution with time decay applied

    @property
    def success_rate(self) -> float:
        """Success rate (proportion of good+ outcomes)."""
        if self.retrieval_count == 0:
            return 0.0
        return self.success_count / self.retrieval_count

    @property
    def excellent_rate(self) -> float:
        """Excellent outcome rate."""
        if self.retrieval_count == 0:
            return 0.0
        return self.excellent_count / self.retrieval_count

    @property
    def poor_rate(self) -> float:
        """Poor outcome rate."""
        if self.retrieval_count == 0:
            return 0.0
        return self.poor_count / self.retrieval_count

    @property
    def age_hours(self) -> float:
        """Hours since last seen."""
        return (time.time() - self.last_seen) / 3600

    def update(
        self,
        contribution_score: float,
        retrieval_score: float,
        quality: OutcomeQuality
    ):
        """Update statistics with new contribution."""
        self.retrieval_count += 1
        self.last_seen = time.time()

        # Update contribution scores
        self.total_contribution += contribution_score
        self.avg_contribution = self.total_contribution / self.retrieval_count
        self.total_retrieval_score += retrieval_score

        # Update quality counters
        if quality in (OutcomeQuality.EXCELLENT, OutcomeQuality.GOOD):
            self.success_count += 1
        if quality == OutcomeQuality.EXCELLENT:
            self.excellent_count += 1
        if quality == OutcomeQuality.POOR:
            self.poor_count += 1

    def apply_decay(self, decay_rate: float = 0.95, hours_elapsed: float = 1.0):
        """
        Apply time-based decay to contribution score.

        Args:
            decay_rate: Decay multiplier per hour (0.95 = 5% loss per hour)
            hours_elapsed: Hours since last decay
        """
        decay_factor = decay_rate ** hours_elapsed
        self.decayed_contribution = self.total_contribution * decay_factor

    def get_boost_factor(self) -> float:
        """
        Calculate retrieval boost factor for this shard.

        Boost formula:
            boost = 1.0 + (success_rate - 0.5) * 2.0 * confidence_factor

        Where confidence_factor increases with sample size (more data = more confidence).

        Returns:
            Boost factor (< 1.0 = penalty, > 1.0 = boost, 1.0 = neutral)
            Range: [0.5, 2.0]
        """
        if self.retrieval_count == 0:
            return 1.0  # No data = neutral

        # Confidence increases with sample size (asymptotes to 1.0)
        # After 10 retrievals, confidence ~= 0.9
        confidence_factor = 1 - math.exp(-self.retrieval_count / 10)

        # Boost based on success rate relative to baseline (0.5)
        # success_rate=1.0 -> boost up to 2.0
        # success_rate=0.5 -> boost = 1.0 (neutral)
        # success_rate=0.0 -> boost down to 0.5
        base_boost = 1.0 + (self.success_rate - 0.5) * 2.0 * confidence_factor

        # Extra boost for consistently excellent results
        if self.excellent_rate > 0.5 and self.retrieval_count >= 5:
            base_boost += 0.2

        # Penalty for consistently poor results
        if self.poor_rate > 0.5 and self.retrieval_count >= 5:
            base_boost -= 0.3

        # Clamp to valid range
        return max(0.5, min(2.0, base_boost))


# ============================================================================
# Shard Contribution Tracker
# ============================================================================

@dataclass
class ShardContributionConfig:
    """Configuration for shard contribution tracking."""
    # Retention
    max_records: int = 10000  # Maximum contribution records to keep
    max_age_days: int = 30    # Maximum age of records before pruning

    # Decay
    enable_decay: bool = True
    decay_rate: float = 0.95  # Per-hour decay rate
    decay_interval_hours: float = 1.0  # Apply decay every N hours

    # Boost calculation
    min_retrievals_for_boost: int = 3  # Minimum retrievals before applying boost
    boost_floor: float = 0.5   # Minimum boost factor
    boost_ceiling: float = 2.0  # Maximum boost factor

    # Quality thresholds
    success_threshold: float = 0.75  # Confidence >= this is "success"
    excellent_threshold: float = 0.9  # Confidence >= this is "excellent"


class ShardContributionTracker:
    """
    Tracks which memory shards contribute to successful query outcomes.

    Creates a feedback loop between retrieval and outcomes:
    1. Record which shards were retrieved for a query
    2. Record the query outcome (confidence, feedback)
    3. Aggregate shard statistics over time
    4. Provide boost factors to improve future retrieval

    Usage:
        tracker = ShardContributionTracker()

        # After retrieval
        query_id = tracker.record_retrieval(
            query_text="What is Thompson Sampling?",
            shard_ids=["shard_1", "shard_2"],
            retrieval_scores={"shard_1": 0.95, "shard_2": 0.82}
        )

        # After weaving
        tracker.record_outcome(
            query_id=query_id,
            confidence=0.88,
            feedback=None  # Optional user feedback
        )

        # During future retrieval
        boost = tracker.get_boost_factor("shard_1")  # e.g., 1.15
        boosted_score = original_score * boost
    """

    def __init__(self, config: Optional[ShardContributionConfig] = None):
        """
        Initialize the shard contribution tracker.

        Args:
            config: Configuration for tracking behavior
        """
        self.config = config or ShardContributionConfig()

        # Records by query_id
        self.records: Dict[str, ContributionRecord] = {}

        # Shard statistics (aggregated)
        self.shard_stats: Dict[str, ShardStats] = {}

        # Recent query IDs (for LRU eviction)
        self.recent_queries: List[str] = []

        # Tracking
        self.last_decay_time = time.time()
        self.total_queries = 0
        self.total_outcomes = 0

        self.logger = logging.getLogger(f"{__name__}.ShardContributionTracker")

    def record_retrieval(
        self,
        query_text: str,
        shard_ids: List[str],
        retrieval_scores: Dict[str, float],
        query_id: Optional[str] = None
    ) -> str:
        """
        Record which shards were retrieved for a query.

        Call this after retrieval but before weaving.

        Args:
            query_text: The query text
            shard_ids: IDs of shards retrieved
            retrieval_scores: Retrieval scores per shard
            query_id: Optional explicit query ID (auto-generated if not provided)

        Returns:
            Query ID for tracking outcome
        """
        # Generate query ID if not provided
        if query_id is None:
            query_id = f"q_{int(time.time() * 1000)}_{self.total_queries}"

        # Create contribution record
        record = ContributionRecord(
            query_id=query_id,
            query_text=query_text,
            shard_ids=shard_ids,
            retrieval_scores=retrieval_scores,
            timestamp=time.time()
        )

        # Store record
        self.records[query_id] = record
        self.recent_queries.append(query_id)
        self.total_queries += 1

        # Prune if over capacity
        self._prune_old_records()

        self.logger.debug(
            f"Recorded retrieval: query_id={query_id}, "
            f"shards={len(shard_ids)}"
        )

        return query_id

    def record_outcome(
        self,
        query_id: str,
        confidence: float,
        feedback: Optional[float] = None,
        tool_used: Optional[str] = None
    ) -> bool:
        """
        Record the outcome of a query.

        Call this after weaving completes.

        Args:
            query_id: The query ID from record_retrieval
            confidence: Outcome confidence (0-1)
            feedback: Optional explicit user feedback (-1 to 1)
            tool_used: Optional tool that was used

        Returns:
            True if outcome was recorded, False if query_id not found
        """
        record = self.records.get(query_id)
        if record is None:
            self.logger.warning(f"Query ID not found: {query_id}")
            return False

        # Record outcome
        record.record_outcome(
            confidence=confidence,
            feedback=feedback,
            tool_used=tool_used
        )

        # Update shard statistics
        contribution_score = record.get_contribution_score()
        quality = record.outcome_quality

        for shard_id in record.shard_ids:
            self._update_shard_stats(
                shard_id=shard_id,
                contribution_score=contribution_score,
                retrieval_score=record.retrieval_scores.get(shard_id, 0.0),
                quality=quality
            )

        self.total_outcomes += 1

        # Apply decay if needed
        self._apply_decay_if_needed()

        self.logger.debug(
            f"Recorded outcome: query_id={query_id}, "
            f"confidence={confidence:.2f}, "
            f"quality={quality.value if quality else 'N/A'}"
        )

        return True

    def get_boost_factor(self, shard_id: str) -> float:
        """
        Get retrieval boost factor for a shard.

        Args:
            shard_id: The shard ID

        Returns:
            Boost factor (< 1.0 = penalty, > 1.0 = boost, 1.0 = neutral)
        """
        stats = self.shard_stats.get(shard_id)
        if stats is None:
            return 1.0  # No data = neutral

        # Check minimum retrievals
        if stats.retrieval_count < self.config.min_retrievals_for_boost:
            return 1.0  # Not enough data

        boost = stats.get_boost_factor()

        # Apply config bounds
        boost = max(self.config.boost_floor, min(self.config.boost_ceiling, boost))

        return boost

    def get_boost_factors_batch(
        self,
        shard_ids: List[str]
    ) -> Dict[str, float]:
        """
        Get boost factors for multiple shards.

        Args:
            shard_ids: List of shard IDs

        Returns:
            Dictionary of shard_id -> boost_factor
        """
        return {
            shard_id: self.get_boost_factor(shard_id)
            for shard_id in shard_ids
        }

    def apply_boosts_to_scores(
        self,
        scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply boost factors to retrieval scores.

        Args:
            scores: Dictionary of shard_id -> retrieval_score

        Returns:
            Boosted scores
        """
        boosted = {}
        for shard_id, score in scores.items():
            boost = self.get_boost_factor(shard_id)
            boosted[shard_id] = score * boost
        return boosted

    def get_shard_stats(self, shard_id: str) -> Optional[ShardStats]:
        """Get statistics for a shard."""
        return self.shard_stats.get(shard_id)

    def get_top_contributors(
        self,
        k: int = 10,
        min_retrievals: int = 5
    ) -> List[Tuple[str, ShardStats]]:
        """
        Get top contributing shards by success rate.

        Args:
            k: Number of top contributors to return
            min_retrievals: Minimum retrievals to be considered

        Returns:
            List of (shard_id, stats) tuples sorted by success rate
        """
        eligible = [
            (sid, stats)
            for sid, stats in self.shard_stats.items()
            if stats.retrieval_count >= min_retrievals
        ]

        # Sort by success rate descending
        eligible.sort(key=lambda x: x[1].success_rate, reverse=True)

        return eligible[:k]

    def get_poor_performers(
        self,
        k: int = 10,
        min_retrievals: int = 5
    ) -> List[Tuple[str, ShardStats]]:
        """
        Get worst performing shards (candidates for forgetting).

        Args:
            k: Number of poor performers to return
            min_retrievals: Minimum retrievals to be considered

        Returns:
            List of (shard_id, stats) tuples sorted by success rate ascending
        """
        eligible = [
            (sid, stats)
            for sid, stats in self.shard_stats.items()
            if stats.retrieval_count >= min_retrievals
        ]

        # Sort by success rate ascending (worst first)
        eligible.sort(key=lambda x: x[1].success_rate)

        return eligible[:k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall tracking statistics."""
        if not self.shard_stats:
            return {
                "total_queries": self.total_queries,
                "total_outcomes": self.total_outcomes,
                "unique_shards": 0,
                "avg_success_rate": 0.0,
                "records_count": len(self.records)
            }

        success_rates = [
            s.success_rate for s in self.shard_stats.values()
            if s.retrieval_count >= 3
        ]

        return {
            "total_queries": self.total_queries,
            "total_outcomes": self.total_outcomes,
            "unique_shards": len(self.shard_stats),
            "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
            "records_count": len(self.records),
            "top_shard_success_rate": max(s.success_rate for s in self.shard_stats.values()) if self.shard_stats else 0.0,
            "lowest_shard_success_rate": min(s.success_rate for s in self.shard_stats.values()) if self.shard_stats else 0.0
        }

    def _update_shard_stats(
        self,
        shard_id: str,
        contribution_score: float,
        retrieval_score: float,
        quality: Optional[OutcomeQuality]
    ):
        """Update statistics for a single shard."""
        if shard_id not in self.shard_stats:
            self.shard_stats[shard_id] = ShardStats(shard_id=shard_id)

        if quality is not None:
            self.shard_stats[shard_id].update(
                contribution_score=contribution_score,
                retrieval_score=retrieval_score,
                quality=quality
            )

    def _apply_decay_if_needed(self):
        """Apply time-based decay if interval has elapsed."""
        if not self.config.enable_decay:
            return

        hours_elapsed = (time.time() - self.last_decay_time) / 3600

        if hours_elapsed >= self.config.decay_interval_hours:
            for stats in self.shard_stats.values():
                stats.apply_decay(
                    decay_rate=self.config.decay_rate,
                    hours_elapsed=hours_elapsed
                )
            self.last_decay_time = time.time()
            self.logger.debug(f"Applied decay after {hours_elapsed:.1f} hours")

    def _prune_old_records(self):
        """Prune old records to stay within capacity."""
        # Prune by count
        while len(self.records) > self.config.max_records:
            if self.recent_queries:
                oldest_id = self.recent_queries.pop(0)
                self.records.pop(oldest_id, None)

        # Prune by age
        max_age_seconds = self.config.max_age_days * 24 * 3600
        cutoff = time.time() - max_age_seconds

        expired = [
            qid for qid, record in self.records.items()
            if record.timestamp < cutoff
        ]

        for qid in expired:
            self.records.pop(qid, None)
            if qid in self.recent_queries:
                self.recent_queries.remove(qid)


# ============================================================================
# Retrieval Booster (Integration Helper)
# ============================================================================

class RetrievalBooster:
    """
    Helper class to integrate shard contribution tracking with retrieval.

    Wraps a retriever and applies boost factors to results.
    """

    def __init__(
        self,
        tracker: ShardContributionTracker,
        blend_factor: float = 0.3
    ):
        """
        Initialize the retrieval booster.

        Args:
            tracker: The shard contribution tracker
            blend_factor: How much boost to apply (0=none, 1=full)
        """
        self.tracker = tracker
        self.blend_factor = blend_factor

    def boost_results(
        self,
        results: List[Tuple[Any, float]]  # List of (shard, score)
    ) -> List[Tuple[Any, float]]:
        """
        Apply boost factors to retrieval results.

        Args:
            results: List of (shard, score) tuples from retrieval

        Returns:
            Reranked list with boost factors applied
        """
        if not results:
            return results

        boosted = []
        for shard, score in results:
            shard_id = getattr(shard, 'id', str(shard))
            boost = self.tracker.get_boost_factor(shard_id)

            # Blend original score with boosted score
            blended_boost = 1.0 + (boost - 1.0) * self.blend_factor
            boosted_score = score * blended_boost

            boosted.append((shard, boosted_score))

        # Rerank by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)

        return boosted

    def record_and_boost(
        self,
        query_text: str,
        results: List[Tuple[Any, float]]
    ) -> Tuple[str, List[Tuple[Any, float]]]:
        """
        Record retrieval and apply boosts in one call.

        Args:
            query_text: The query text
            results: Retrieval results

        Returns:
            (query_id, boosted_results)
        """
        if not results:
            return "", results

        # Extract shard info
        shard_ids = []
        retrieval_scores = {}

        for shard, score in results:
            shard_id = getattr(shard, 'id', str(shard))
            shard_ids.append(shard_id)
            retrieval_scores[shard_id] = score

        # Record retrieval
        query_id = self.tracker.record_retrieval(
            query_text=query_text,
            shard_ids=shard_ids,
            retrieval_scores=retrieval_scores
        )

        # Apply boosts
        boosted = self.boost_results(results)

        return query_id, boosted


# ============================================================================
# Factory Functions
# ============================================================================

def create_contribution_tracker(
    max_records: int = 10000,
    enable_decay: bool = True,
    decay_rate: float = 0.95
) -> ShardContributionTracker:
    """
    Create a configured shard contribution tracker.

    Args:
        max_records: Maximum records to retain
        enable_decay: Whether to apply time-based decay
        decay_rate: Decay rate per hour

    Returns:
        Configured tracker
    """
    config = ShardContributionConfig(
        max_records=max_records,
        enable_decay=enable_decay,
        decay_rate=decay_rate
    )
    return ShardContributionTracker(config)


def create_retrieval_booster(
    tracker: Optional[ShardContributionTracker] = None,
    blend_factor: float = 0.3
) -> RetrievalBooster:
    """
    Create a retrieval booster.

    Args:
        tracker: Contribution tracker (created if not provided)
        blend_factor: How much boost to apply

    Returns:
        Configured booster
    """
    if tracker is None:
        tracker = create_contribution_tracker()
    return RetrievalBooster(tracker, blend_factor)
