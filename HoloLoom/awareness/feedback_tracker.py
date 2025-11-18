#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback Tracker for Phase 6.1: USER_FEEDBACK Refinement

Tracks user feedback on refinement results and learns which strategies
work best for different query types.

Components:
1. FeedbackSignal - Captures user feedback
2. FeedbackTracker - Stores and aggregates feedback
3. Strategy learning - Adapts based on feedback

Created: 2025-11-18
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time
from collections import defaultdict


class FeedbackType(str, Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 stars
    CORRECTION = "correction"  # User edited response
    SUGGESTION = "suggestion"  # Text feedback


@dataclass
class FeedbackSignal:
    """
    Captures user feedback on a refinement result.

    Examples:
        # Thumbs up
        FeedbackSignal(
            feedback_type=FeedbackType.THUMBS_UP,
            helpful=True
        )

        # Rating with comment
        FeedbackSignal(
            feedback_type=FeedbackType.RATING,
            rating=4,
            comment="Good but needs more examples"
        )

        # Correction
        FeedbackSignal(
            feedback_type=FeedbackType.CORRECTION,
            corrected_response="Better version...",
            what_was_wrong="Completeness"
        )
    """
    feedback_type: FeedbackType
    helpful: Optional[bool] = None  # True/False for thumbs up/down
    rating: Optional[float] = None  # 1-5 stars (can be fractional)
    comment: Optional[str] = None
    corrected_response: Optional[str] = None
    what_was_wrong: Optional[str] = None  # "coherence", "completeness", "relevance"
    timestamp: float = field(default_factory=time.time)

    def get_numeric_score(self) -> float:
        """
        Convert feedback to numeric score (0.0 = bad, 1.0 = excellent).

        Returns:
            Normalized score 0.0-1.0
        """
        if self.helpful is not None:
            return 1.0 if self.helpful else 0.0

        if self.rating is not None:
            # Convert 1-5 scale to 0-1
            return (self.rating - 1.0) / 4.0

        # Default: neutral
        return 0.5


@dataclass
class StrategyPerformance:
    """
    Tracks performance of a refinement strategy.

    Learns which strategies work best through user feedback.
    """
    strategy_name: str
    total_uses: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    total_rating_score: float = 0.0
    rating_count: int = 0
    corrections: int = 0

    def get_success_rate(self) -> float:
        """Success rate based on feedback (0.0-1.0)"""
        total_feedback = self.positive_feedback + self.negative_feedback
        if total_feedback == 0:
            return 0.5  # Neutral if no feedback
        return self.positive_feedback / total_feedback

    def get_average_rating(self) -> float:
        """Average rating (0.0-1.0 scale)"""
        if self.rating_count == 0:
            return 0.5  # Neutral if no ratings
        avg_raw = self.total_rating_score / self.rating_count
        return (avg_raw - 1.0) / 4.0  # Convert 1-5 to 0-1

    def get_confidence(self) -> float:
        """
        Confidence in performance estimate (0.0-1.0).

        Based on sample size. More feedback = higher confidence.
        """
        total_feedback = self.positive_feedback + self.negative_feedback + self.rating_count
        # Sigmoid function: confident after ~20 samples
        return min(1.0, total_feedback / 20.0)

    def get_overall_score(self) -> float:
        """
        Overall performance score (0.0-1.0).

        Weighted combination of success rate and average rating.
        """
        success_rate = self.get_success_rate()
        avg_rating = self.get_average_rating()
        confidence = self.get_confidence()

        # Weight by confidence (low confidence → regress to neutral 0.5)
        score = confidence * ((success_rate + avg_rating) / 2.0) + (1 - confidence) * 0.5

        # Penalty for corrections (indicates quality issues)
        if self.total_uses > 0:
            correction_rate = self.corrections / self.total_uses
            score *= (1.0 - 0.5 * correction_rate)  # Max 50% penalty

        return score


@dataclass
class QueryTypeProfile:
    """
    Tracks which strategies work best for a query type.

    Query types classified by:
    - Domain (technical, general, creative, etc.)
    - Complexity (simple, medium, complex)
    - Expected format (explanation, list, definition, etc.)
    """
    query_type: str
    strategy_performance: Dict[str, StrategyPerformance] = field(default_factory=dict)

    def get_best_strategy(self) -> Optional[str]:
        """Get strategy with highest overall score"""
        if not self.strategy_performance:
            return None

        best_strategy = max(
            self.strategy_performance.items(),
            key=lambda x: x[1].get_overall_score()
        )

        return best_strategy[0]

    def get_strategy_ranking(self) -> List[tuple]:
        """
        Get strategies ranked by performance.

        Returns:
            List of (strategy_name, score) tuples, sorted best to worst
        """
        ranking = [
            (name, perf.get_overall_score())
            for name, perf in self.strategy_performance.items()
        ]
        return sorted(ranking, key=lambda x: x[1], reverse=True)


class FeedbackTracker:
    """
    Tracks user feedback across queries and learns from patterns.

    Maintains:
    - Strategy performance by query type
    - Overall feedback statistics
    - Learning history

    Thread-safe for concurrent feedback.
    """

    def __init__(self):
        # Strategy performance by query type
        self.query_type_profiles: Dict[str, QueryTypeProfile] = defaultdict(
            lambda: QueryTypeProfile(query_type="unknown")
        )

        # Global strategy performance (across all query types)
        self.global_strategy_performance: Dict[str, StrategyPerformance] = defaultdict(
            lambda: StrategyPerformance(strategy_name="unknown")
        )

        # Raw feedback history
        self.feedback_history: List[Dict[str, Any]] = []

        # Statistics
        self.total_feedback_count = 0
        self.positive_feedback_count = 0
        self.negative_feedback_count = 0

    def track_feedback(
        self,
        query: str,
        query_type: str,
        strategy_used: str,
        feedback: FeedbackSignal,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Track user feedback for a query.

        Args:
            query: Original query text
            query_type: Classification (e.g., "technical_explanation")
            strategy_used: Which refinement strategy was used
            feedback: User feedback signal
            metadata: Additional context (initial_quality, final_quality, etc.)
        """
        # Update global stats
        self.total_feedback_count += 1

        # Get or create query type profile
        profile = self.query_type_profiles[query_type]
        profile.query_type = query_type

        # Get or create strategy performance tracker
        if strategy_used not in profile.strategy_performance:
            profile.strategy_performance[strategy_used] = StrategyPerformance(
                strategy_name=strategy_used
            )

        strategy_perf = profile.strategy_performance[strategy_used]
        global_perf = self.global_strategy_performance[strategy_used]
        global_perf.strategy_name = strategy_used

        # Update usage count
        strategy_perf.total_uses += 1
        global_perf.total_uses += 1

        # Update based on feedback type
        if feedback.helpful is not None:
            if feedback.helpful:
                strategy_perf.positive_feedback += 1
                global_perf.positive_feedback += 1
                self.positive_feedback_count += 1
            else:
                strategy_perf.negative_feedback += 1
                global_perf.negative_feedback += 1
                self.negative_feedback_count += 1

        if feedback.rating is not None:
            strategy_perf.total_rating_score += feedback.rating
            strategy_perf.rating_count += 1
            global_perf.total_rating_score += feedback.rating
            global_perf.rating_count += 1

        if feedback.corrected_response is not None:
            strategy_perf.corrections += 1
            global_perf.corrections += 1

        # Store in history
        self.feedback_history.append({
            "query": query,
            "query_type": query_type,
            "strategy_used": strategy_used,
            "feedback": feedback,
            "metadata": metadata or {},
            "timestamp": time.time()
        })

    def get_recommended_strategy(
        self,
        query_type: str,
        fallback: str = "adaptive"
    ) -> str:
        """
        Get recommended strategy for a query type based on feedback.

        Args:
            query_type: Classification of query
            fallback: Strategy to use if no feedback available

        Returns:
            Strategy name (e.g., "depth_first", "breadth_first")
        """
        profile = self.query_type_profiles.get(query_type)

        if profile:
            best_strategy = profile.get_best_strategy()
            if best_strategy:
                return best_strategy

        return fallback

    def get_strategy_performance(
        self,
        strategy: str,
        query_type: Optional[str] = None
    ) -> Optional[StrategyPerformance]:
        """
        Get performance metrics for a strategy.

        Args:
            strategy: Strategy name
            query_type: Optional query type filter

        Returns:
            StrategyPerformance or None if not found
        """
        if query_type:
            profile = self.query_type_profiles.get(query_type)
            if profile:
                return profile.strategy_performance.get(strategy)
            return None
        else:
            return self.global_strategy_performance.get(strategy)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall feedback statistics.

        Returns:
            Dict with keys: total_feedback, positive_rate, avg_rating, etc.
        """
        total_feedback = self.positive_feedback_count + self.negative_feedback_count

        positive_rate = 0.0
        if total_feedback > 0:
            positive_rate = self.positive_feedback_count / total_feedback

        # Calculate average rating across all strategies
        total_rating_sum = sum(
            perf.total_rating_score
            for perf in self.global_strategy_performance.values()
        )
        total_rating_count = sum(
            perf.rating_count
            for perf in self.global_strategy_performance.values()
        )

        avg_rating = 0.0
        if total_rating_count > 0:
            avg_rating = total_rating_sum / total_rating_count

        return {
            "total_feedback": self.total_feedback_count,
            "positive_feedback": self.positive_feedback_count,
            "negative_feedback": self.negative_feedback_count,
            "positive_rate": positive_rate,
            "avg_rating": avg_rating,
            "total_corrections": sum(
                perf.corrections
                for perf in self.global_strategy_performance.values()
            ),
            "unique_query_types": len(self.query_type_profiles),
            "strategies_used": len(self.global_strategy_performance)
        }

    def get_learning_insights(self) -> List[str]:
        """
        Get human-readable insights from feedback learning.

        Returns:
            List of insight strings
        """
        insights = []

        # Overall feedback quality
        stats = self.get_statistics()
        if stats["total_feedback"] > 0:
            insights.append(
                f"Received {stats['total_feedback']} feedback signals "
                f"({stats['positive_rate']:.1%} positive)"
            )

        # Best performing strategies
        strategy_scores = [
            (name, perf.get_overall_score())
            for name, perf in self.global_strategy_performance.items()
            if perf.total_uses >= 5  # Require minimum sample size
        ]

        if strategy_scores:
            strategy_scores.sort(key=lambda x: x[1], reverse=True)
            best_strategy, best_score = strategy_scores[0]
            insights.append(
                f"Best overall strategy: {best_strategy} (score: {best_score:.2f})"
            )

        # Query type specific insights
        for query_type, profile in self.query_type_profiles.items():
            best_strategy = profile.get_best_strategy()
            if best_strategy:
                perf = profile.strategy_performance[best_strategy]
                if perf.total_uses >= 3:  # Require minimum sample
                    insights.append(
                        f"For {query_type} queries: {best_strategy} works best "
                        f"({perf.get_success_rate():.1%} success)"
                    )

        # Correction patterns
        if stats["total_corrections"] > 0:
            insights.append(
                f"⚠️ {stats['total_corrections']} responses needed user corrections"
            )

        return insights

    def reset_statistics(self):
        """Reset all feedback tracking (for testing)"""
        self.query_type_profiles.clear()
        self.global_strategy_performance.clear()
        self.feedback_history.clear()
        self.total_feedback_count = 0
        self.positive_feedback_count = 0
        self.negative_feedback_count = 0
