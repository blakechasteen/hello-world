#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Phase 6.1: USER_FEEDBACK Refinement

Tests:
1. FeedbackTracker
   - Feedback signal creation and scoring
   - Strategy performance tracking
   - Query type profiling
   - Recommendation generation
   - Statistics and insights

2. UserFeedbackRefiner
   - Query type classification
   - Feedback-based strategy selection
   - Learning from feedback
   - Integration with Phase 5

3. Edge Cases
   - No feedback available
   - Conflicting feedback
   - Sparse data (low confidence)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass

# Import Phase 6.1 components
from HoloLoom.awareness.feedback_tracker import (
    FeedbackTracker,
    FeedbackSignal,
    FeedbackType,
    StrategyPerformance,
    QueryTypeProfile
)

from HoloLoom.awareness.user_feedback_refiner import (
    UserFeedbackRefiner,
    FeedbackAwareResult
)

# Import Phase 5 for mocks
from HoloLoom.awareness.refinement_engine import RefinementStrategy


# ============================================================================
# Mock Objects
# ============================================================================

@dataclass
class MockQualityScore:
    """Mock quality score"""
    coherence: float = 0.8
    completeness: float = 0.8
    relevance: float = 0.8
    overall: float = 0.8


@dataclass
class MockPackedGeneration:
    """Mock packed generation result"""
    response: str = "Mock response"
    quality_score: MockQualityScore = None

    def __post_init__(self):
        if self.quality_score is None:
            self.quality_score = MockQualityScore()


class MockLLMContextPacker:
    """Mock LLMContextPacker"""

    def __init__(self, quality_progression=None):
        self.quality_progression = quality_progression or [0.7, 0.85]
        self.call_count = 0
        self.enable_compression = True
        self.min_importance = 0.7
        self.budget = Mock(total=5000)

    async def pack_and_generate(self, *args, **kwargs):
        quality = self.quality_progression[min(self.call_count, len(self.quality_progression) - 1)]
        self.call_count += 1

        return MockPackedGeneration(
            quality_score=MockQualityScore(
                coherence=quality,
                completeness=quality,
                relevance=quality,
                overall=quality
            )
        )


# ============================================================================
# Test FeedbackSignal
# ============================================================================

def test_feedback_signal_thumbs_up():
    """Test thumbs up feedback"""
    feedback = FeedbackSignal(
        feedback_type=FeedbackType.THUMBS_UP,
        helpful=True
    )

    assert feedback.get_numeric_score() == 1.0


def test_feedback_signal_thumbs_down():
    """Test thumbs down feedback"""
    feedback = FeedbackSignal(
        feedback_type=FeedbackType.THUMBS_DOWN,
        helpful=False
    )

    assert feedback.get_numeric_score() == 0.0


def test_feedback_signal_rating():
    """Test rating feedback"""
    # 5-star rating
    feedback_5 = FeedbackSignal(
        feedback_type=FeedbackType.RATING,
        rating=5.0
    )
    assert feedback_5.get_numeric_score() == 1.0

    # 1-star rating
    feedback_1 = FeedbackSignal(
        feedback_type=FeedbackType.RATING,
        rating=1.0
    )
    assert feedback_1.get_numeric_score() == 0.0

    # 3-star rating (neutral)
    feedback_3 = FeedbackSignal(
        feedback_type=FeedbackType.RATING,
        rating=3.0
    )
    assert feedback_3.get_numeric_score() == 0.5


def test_feedback_signal_correction():
    """Test correction feedback"""
    feedback = FeedbackSignal(
        feedback_type=FeedbackType.CORRECTION,
        corrected_response="Better version",
        what_was_wrong="completeness"
    )

    assert feedback.corrected_response == "Better version"
    assert feedback.what_was_wrong == "completeness"


# ============================================================================
# Test StrategyPerformance
# ============================================================================

def test_strategy_performance_success_rate():
    """Test success rate calculation"""
    perf = StrategyPerformance(strategy_name="depth_first")

    # No feedback yet
    assert perf.get_success_rate() == 0.5  # Neutral

    # Add positive feedback
    perf.positive_feedback = 8
    perf.negative_feedback = 2

    assert perf.get_success_rate() == 0.8  # 80% success


def test_strategy_performance_average_rating():
    """Test average rating calculation"""
    perf = StrategyPerformance(strategy_name="breadth_first")

    # No ratings yet
    assert perf.get_average_rating() == 0.5  # Neutral

    # Add ratings (1-5 scale)
    perf.total_rating_score = 20.0  # 4 ratings of 5 stars each
    perf.rating_count = 4

    assert perf.get_average_rating() == 1.0  # Perfect rating


def test_strategy_performance_confidence():
    """Test confidence calculation"""
    perf = StrategyPerformance(strategy_name="focused")

    # No feedback
    assert perf.get_confidence() == 0.0

    # Few samples (low confidence)
    perf.positive_feedback = 3
    perf.negative_feedback = 2
    assert perf.get_confidence() == 0.25  # 5/20 samples

    # Many samples (high confidence)
    perf.positive_feedback = 15
    perf.negative_feedback = 5
    assert perf.get_confidence() == 1.0  # 20/20 samples


def test_strategy_performance_overall_score():
    """Test overall score calculation"""
    perf = StrategyPerformance(strategy_name="adaptive")

    # Perfect performance, high confidence
    perf.positive_feedback = 18
    perf.negative_feedback = 2
    perf.total_rating_score = 20.0  # 4×5 stars
    perf.rating_count = 4
    perf.total_uses = 20

    score = perf.get_overall_score()
    assert score > 0.8  # Should be high

    # Add corrections (penalty)
    perf.corrections = 5
    score_with_corrections = perf.get_overall_score()
    assert score_with_corrections < score  # Penalty applied


# ============================================================================
# Test QueryTypeProfile
# ============================================================================

def test_query_type_profile_best_strategy():
    """Test best strategy selection"""
    profile = QueryTypeProfile(query_type="technical_explanation")

    # Add strategy performance
    profile.strategy_performance["depth_first"] = StrategyPerformance(
        strategy_name="depth_first",
        positive_feedback=8,
        negative_feedback=2,
        total_uses=10
    )

    profile.strategy_performance["breadth_first"] = StrategyPerformance(
        strategy_name="breadth_first",
        positive_feedback=5,
        negative_feedback=5,
        total_uses=10
    )

    # depth_first should be best (80% vs 50% success)
    best = profile.get_best_strategy()
    assert best == "depth_first"


def test_query_type_profile_strategy_ranking():
    """Test strategy ranking"""
    profile = QueryTypeProfile(query_type="list")

    # Add multiple strategies
    profile.strategy_performance["depth_first"] = StrategyPerformance(
        strategy_name="depth_first",
        positive_feedback=5,
        total_uses=10
    )

    profile.strategy_performance["breadth_first"] = StrategyPerformance(
        strategy_name="breadth_first",
        positive_feedback=9,
        total_uses=10
    )

    profile.strategy_performance["focused"] = StrategyPerformance(
        strategy_name="focused",
        positive_feedback=7,
        total_uses=10
    )

    ranking = profile.get_strategy_ranking()

    # Should be sorted by score
    assert len(ranking) == 3
    assert ranking[0][0] == "breadth_first"  # Best
    assert ranking[2][0] == "depth_first"  # Worst


# ============================================================================
# Test FeedbackTracker
# ============================================================================

def test_feedback_tracker_basic():
    """Test basic feedback tracking"""
    tracker = FeedbackTracker()

    # Track feedback
    tracker.track_feedback(
        query="What is Thompson Sampling?",
        query_type="technical_explanation",
        strategy_used="depth_first",
        feedback=FeedbackSignal(
            feedback_type=FeedbackType.THUMBS_UP,
            helpful=True
        )
    )

    # Check statistics
    stats = tracker.get_statistics()
    assert stats["total_feedback"] == 1
    assert stats["positive_feedback"] == 1
    assert stats["positive_rate"] == 1.0


def test_feedback_tracker_multiple_queries():
    """Test tracking multiple queries"""
    tracker = FeedbackTracker()

    # Track 10 queries with different feedback
    for i in range(10):
        tracker.track_feedback(
            query=f"Query {i}",
            query_type="general",
            strategy_used="adaptive",
            feedback=FeedbackSignal(
                feedback_type=FeedbackType.THUMBS_UP if i % 2 == 0 else FeedbackType.THUMBS_DOWN,
                helpful=i % 2 == 0
            )
        )

    stats = tracker.get_statistics()
    assert stats["total_feedback"] == 10
    assert stats["positive_feedback"] == 5
    assert stats["negative_feedback"] == 5
    assert stats["positive_rate"] == 0.5


def test_feedback_tracker_strategy_recommendation():
    """Test strategy recommendation based on feedback"""
    tracker = FeedbackTracker()

    # Give depth_first good feedback for technical queries
    for i in range(10):
        tracker.track_feedback(
            query=f"Technical query {i}",
            query_type="technical_explanation",
            strategy_used="depth_first",
            feedback=FeedbackSignal(feedback_type=FeedbackType.THUMBS_UP, helpful=True)
        )

    # Give breadth_first poor feedback for technical queries
    for i in range(5):
        tracker.track_feedback(
            query=f"Another technical query {i}",
            query_type="technical_explanation",
            strategy_used="breadth_first",
            feedback=FeedbackSignal(feedback_type=FeedbackType.THUMBS_DOWN, helpful=False)
        )

    # Should recommend depth_first for technical_explanation
    recommended = tracker.get_recommended_strategy("technical_explanation")
    assert recommended == "depth_first"


def test_feedback_tracker_learning_insights():
    """Test learning insights generation"""
    tracker = FeedbackTracker()

    # Add feedback
    for i in range(20):
        tracker.track_feedback(
            query=f"Query {i}",
            query_type="general",
            strategy_used="adaptive",
            feedback=FeedbackSignal(
                feedback_type=FeedbackType.RATING,
                rating=4.0 if i < 15 else 2.0  # Mostly good, some bad
            )
        )

    insights = tracker.get_learning_insights()

    # Should have insights about feedback received
    assert len(insights) > 0
    assert any("20 feedback signals" in insight for insight in insights)


# ============================================================================
# Test UserFeedbackRefiner - Query Classification
# ============================================================================

@pytest.mark.asyncio
async def test_query_type_classification():
    """Test query type classification"""
    packer = MockLLMContextPacker()
    refiner = UserFeedbackRefiner(packer=packer)

    # Technical explanation
    assert refiner.classify_query_type("Explain how the algorithm works") == "technical_explanation"

    # Technical example
    assert refiner.classify_query_type("Show me code example") == "technical_example"

    # Creative
    assert refiner.classify_query_type("Write a poem") == "creative"

    # Analytical
    assert refiner.classify_query_type("Compare Python and Java") == "analytical"

    # Factual
    assert refiner.classify_query_type("What is machine learning?") == "factual_definition"

    # List
    assert refiner.classify_query_type("List the main features") == "list"

    # How-to
    assert refiner.classify_query_type("How to install Python?") == "howto"

    # General
    assert refiner.classify_query_type("Tell me about it") == "general"


# ============================================================================
# Test UserFeedbackRefiner - Learning
# ============================================================================

@pytest.mark.asyncio
async def test_user_feedback_refiner_basic():
    """Test basic refinement with feedback learning"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])
    refiner = UserFeedbackRefiner(
        packer=packer,
        quality_threshold=0.80,
        enable_feedback_learning=True
    )

    # Execute refinement
    result = await refiner.refine(
        query="Explain Thompson Sampling",
        awareness_ctx=Mock(),
        memory_results=[]
    )

    # Should be FeedbackAwareResult
    assert isinstance(result, FeedbackAwareResult)
    assert result.query_type is not None
    assert result.final_quality >= result.initial_quality


@pytest.mark.asyncio
async def test_feedback_learning_strategy_selection():
    """Test strategy selection based on feedback"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    # Create refiner with shared tracker
    tracker = FeedbackTracker()

    refiner = UserFeedbackRefiner(
        packer=packer,
        quality_threshold=0.80,
        feedback_tracker=tracker,
        enable_feedback_learning=True
    )

    # First: Train feedback for technical_explanation queries
    # Give depth_first good feedback
    for i in range(10):
        tracker.track_feedback(
            query=f"Tech query {i}",
            query_type="technical_explanation",
            strategy_used="depth_first",
            feedback=FeedbackSignal(feedback_type=FeedbackType.THUMBS_UP, helpful=True)
        )

    # Reset packer call count
    packer.call_count = 0

    # Second: Refine a technical_explanation query
    result = await refiner.refine(
        query="Explain how Thompson Sampling works technically",
        awareness_ctx=Mock(),
        memory_results=[]
    )

    # Should have used learned preference (depth_first)
    # Note: Since strategy is ADAPTIVE, it chooses based on feedback
    assert result.query_type == "technical_explanation"


@pytest.mark.asyncio
async def test_track_feedback():
    """Test tracking feedback on results"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])
    refiner = UserFeedbackRefiner(
        packer=packer,
        quality_threshold=0.80,
        enable_feedback_learning=True
    )

    # Refine
    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[]
    )

    # Track feedback
    feedback = FeedbackSignal(
        feedback_type=FeedbackType.THUMBS_UP,
        helpful=True
    )

    await refiner.track_feedback(result=result, feedback=feedback)

    # Should be stored
    assert result.feedback_signal == feedback

    # Should be in tracker
    stats = refiner.get_feedback_statistics()
    assert stats["total_feedback"] == 1


@pytest.mark.asyncio
async def test_feedback_statistics():
    """Test feedback statistics retrieval"""
    packer = MockLLMContextPacker()
    refiner = UserFeedbackRefiner(
        packer=packer,
        enable_feedback_learning=True
    )

    # Add some feedback
    for i in range(5):
        refiner.feedback_tracker.track_feedback(
            query=f"Query {i}",
            query_type="general",
            strategy_used="adaptive",
            feedback=FeedbackSignal(
                feedback_type=FeedbackType.RATING,
                rating=4.0
            )
        )

    stats = refiner.get_feedback_statistics()

    assert "total_feedback" in stats
    assert "insights" in stats
    assert "strategy_recommendations" in stats


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_no_feedback_available():
    """Test behavior when no feedback available"""
    packer = MockLLMContextPacker()
    refiner = UserFeedbackRefiner(
        packer=packer,
        enable_feedback_learning=True
    )

    # Refine without any feedback history
    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[]
    )

    # Should use default strategy (ADAPTIVE)
    assert result is not None
    assert result.query_type is not None


@pytest.mark.asyncio
async def test_feedback_learning_disabled():
    """Test with feedback learning disabled"""
    packer = MockLLMContextPacker()
    refiner = UserFeedbackRefiner(
        packer=packer,
        enable_feedback_learning=False
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[]
    )

    # Should still work, just no learning
    assert result is not None

    # Tracking feedback should do nothing
    await refiner.track_feedback(
        result=result,
        feedback=FeedbackSignal(feedback_type=FeedbackType.THUMBS_UP, helpful=True)
    )


def test_reset_feedback():
    """Test resetting feedback statistics"""
    tracker = FeedbackTracker()

    # Add feedback
    for i in range(10):
        tracker.track_feedback(
            query=f"Query {i}",
            query_type="general",
            strategy_used="adaptive",
            feedback=FeedbackSignal(feedback_type=FeedbackType.THUMBS_UP, helpful=True)
        )

    assert tracker.total_feedback_count == 10

    # Reset
    tracker.reset_statistics()

    assert tracker.total_feedback_count == 0
    assert len(tracker.feedback_history) == 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
