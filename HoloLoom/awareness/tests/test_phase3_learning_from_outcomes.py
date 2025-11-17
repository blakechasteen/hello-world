#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Phase 3: Learning from Outcomes

Tests:
1. OutcomeTracker
2. AdaptiveThresholdTuner
3. PerformanceAnalyzer
4. Integration with LLMContextPacker
"""

import pytest
from dataclasses import dataclass

# Import components to test
from HoloLoom.awareness.outcome_tracker import (
    OutcomeTracker,
    Outcome,
    CompressionLevel,
    ThresholdRecommendation
)

from HoloLoom.awareness.adaptive_threshold_tuner import (
    AdaptiveThresholdTuner,
    TuningStrategy,
    ThresholdSnapshot,
    TuningEvent
)

from HoloLoom.awareness.performance_analyzer import (
    PerformanceAnalyzer,
    Bottleneck,
    Anomaly,
    BottleneckSeverity,
    AnomalyType
)


# ============================================================================
# Test OutcomeTracker
# ============================================================================

@pytest.mark.asyncio
async def test_outcome_tracker_basic():
    """Test basic outcome tracking"""
    tracker = OutcomeTracker()

    # Track an outcome
    outcome = await tracker.track(
        query="What is Python?",
        query_complexity="SIMPLE",
        budget_allocated=4000,
        budget_used=2500,
        quality_overall=0.92,
        quality_coherence=0.90,
        quality_completeness=0.95,
        quality_relevance=0.91,
        compression_level=CompressionLevel.SUMMARY,
        importance_threshold=0.3
    )

    assert outcome.query == "What is Python?"
    assert outcome.budget_efficiency > 0
    assert len(tracker.outcomes) == 1


@pytest.mark.asyncio
async def test_outcome_tracker_statistics():
    """Test outcome statistics"""
    tracker = OutcomeTracker()

    # Track multiple outcomes
    for i in range(5):
        await tracker.track(
            query=f"Query {i}",
            query_complexity="SIMPLE",
            budget_allocated=4000,
            budget_used=3000,
            quality_overall=0.8 + (i * 0.02),  # Increasing quality
            compression_level=CompressionLevel.SUMMARY,
            importance_threshold=0.3
        )

    stats = tracker.get_statistics()
    assert stats["total_outcomes"] == 5
    assert "avg_quality" in stats
    assert 0.8 <= stats["avg_quality"] <= 0.9


@pytest.mark.asyncio
async def test_optimal_importance_threshold():
    """Test optimal importance threshold detection"""
    tracker = OutcomeTracker(min_samples_for_recommendation=5)

    # Track outcomes with different thresholds
    for threshold in [0.2, 0.3, 0.4, 0.5]:
        for _ in range(3):
            quality = 0.7 if threshold == 0.3 else 0.6  # 0.3 is best
            await tracker.track(
                query="Test",
                query_complexity="SIMPLE",
                budget_allocated=4000,
                budget_used=3000,
                quality_overall=quality,
                compression_level=CompressionLevel.SUMMARY,
                importance_threshold=threshold
            )

    # Should detect 0.3 as optimal
    optimal = tracker.get_optimal_importance_threshold()
    assert optimal is not None
    threshold, quality = optimal
    assert threshold == 0.3
    assert quality > 0.6


@pytest.mark.asyncio
async def test_optimal_compression_level():
    """Test optimal compression level detection"""
    tracker = OutcomeTracker(min_samples_for_recommendation=5)

    # Track outcomes with different compression levels
    for level in [CompressionLevel.MINIMAL, CompressionLevel.SUMMARY, CompressionLevel.DETAILED]:
        for _ in range(3):
            quality = 0.85 if level == CompressionLevel.DETAILED else 0.75
            await tracker.track(
                query="Test",
                query_complexity="SIMPLE",
                budget_allocated=4000,
                budget_used=3000,
                quality_overall=quality,
                compression_level=level,
                importance_threshold=0.3
            )

    # Should detect DETAILED as optimal
    optimal = tracker.get_optimal_compression_level()
    assert optimal is not None
    level, quality = optimal
    assert level == CompressionLevel.DETAILED
    assert quality > 0.8


@pytest.mark.asyncio
async def test_budget_efficiency_insights():
    """Test budget efficiency analysis"""
    tracker = OutcomeTracker(min_samples_for_recommendation=5)

    # Track outcomes with varying budgets
    for budget in [2000, 4000, 6000, 8000, 10000]:
        for _ in range(3):
            await tracker.track(
                query="Test",
                query_complexity="SIMPLE",
                budget_allocated=budget,
                budget_used=int(budget * 0.8),
                quality_overall=0.85,
                compression_level=CompressionLevel.SUMMARY,
                importance_threshold=0.3
            )

    insights = tracker.get_budget_efficiency_insights()
    assert "avg_efficiency" in insights
    assert "efficiency_by_budget_range" in insights


@pytest.mark.asyncio
async def test_recommendations():
    """Test threshold recommendations"""
    tracker = OutcomeTracker(min_samples_for_recommendation=5)

    # Track 15 outcomes with threshold 0.4 performing better than 0.3
    for threshold in [0.3, 0.4]:
        for _ in range(8):
            quality = 0.9 if threshold == 0.4 else 0.7
            await tracker.track(
                query="Test",
                query_complexity="SIMPLE",
                budget_allocated=4000,
                budget_used=3000,
                quality_overall=quality,
                compression_level=CompressionLevel.SUMMARY,
                importance_threshold=threshold
            )

    # Get recommendations (current threshold is 0.3)
    recs = tracker.get_recommendations(
        complexity="SIMPLE",
        current_importance_threshold=0.3,
        current_compression_level=CompressionLevel.SUMMARY
    )

    assert len(recs) > 0
    # Should recommend increasing threshold to 0.4
    threshold_rec = [r for r in recs if r.parameter == "importance_threshold"][0]
    assert threshold_rec.recommended_value == 0.4


# ============================================================================
# Test AdaptiveThresholdTuner
# ============================================================================

@pytest.mark.asyncio
async def test_threshold_tuner_init():
    """Test threshold tuner initialization"""
    tracker = OutcomeTracker()
    tuner = AdaptiveThresholdTuner(
        outcome_tracker=tracker,
        enable_auto_tuning=True,
        tuning_strategy=TuningStrategy.BALANCED
    )

    assert tuner.enable_auto_tuning is True
    assert tuner.strategy == TuningStrategy.BALANCED
    assert "SIMPLE" in tuner.current_thresholds


@pytest.mark.asyncio
async def test_get_thresholds():
    """Test getting current thresholds"""
    tracker = OutcomeTracker()
    tuner = AdaptiveThresholdTuner(outcome_tracker=tracker)

    thresholds = tuner.get_thresholds("SIMPLE")
    assert "importance_threshold" in thresholds
    assert "compression_level" in thresholds
    assert isinstance(thresholds["compression_level"], CompressionLevel)


@pytest.mark.asyncio
async def test_threshold_update():
    """Test threshold update based on recommendations"""
    tracker = OutcomeTracker(min_samples_for_recommendation=5)

    # Populate tracker with data showing 0.4 is better than 0.3
    for _ in range(10):
        await tracker.track(
            query="Test",
            query_complexity="SIMPLE",
            budget_allocated=4000,
            budget_used=3000,
            quality_overall=0.9,
            compression_level=CompressionLevel.SUMMARY,
            importance_threshold=0.4
        )

    tuner = AdaptiveThresholdTuner(
        outcome_tracker=tracker,
        enable_auto_tuning=True,
        tuning_strategy=TuningStrategy.AGGRESSIVE  # Lower confidence threshold
    )

    # Update should apply recommendation
    events = await tuner.update(complexity="SIMPLE")

    # Should have updated threshold
    assert len(events) > 0
    current = tuner.get_thresholds("SIMPLE")
    # Note: May or may not update depending on recommendation confidence


@pytest.mark.asyncio
async def test_regression_detection():
    """Test regression detection and rollback"""
    tracker = OutcomeTracker()
    tuner = AdaptiveThresholdTuner(outcome_tracker=tracker, regression_threshold=0.1)

    # Track initial good performance
    for _ in range(10):
        await tracker.track(
            query="Test",
            query_complexity="SIMPLE",
            budget_allocated=4000,
            budget_used=3000,
            quality_overall=0.9,
            compression_level=CompressionLevel.SUMMARY,
            importance_threshold=0.3
        )

    # Create snapshot
    await tuner._create_snapshot("SIMPLE")

    # Simulate quality drop
    for _ in range(10):
        await tracker.track(
            query="Test",
            query_complexity="SIMPLE",
            budget_allocated=4000,
            budget_used=3000,
            quality_overall=0.7,  # Dropped 0.2
            compression_level=CompressionLevel.SUMMARY,
            importance_threshold=0.3
        )

    # Check for regression
    snapshot = await tuner.check_for_regression("SIMPLE")
    assert snapshot is not None  # Regression detected


@pytest.mark.asyncio
async def test_tuning_history():
    """Test tuning history tracking"""
    tracker = OutcomeTracker()
    tuner = AdaptiveThresholdTuner(outcome_tracker=tracker)

    # Should start with no events
    history = tuner.get_tuning_history()
    assert len(history) == 0


# ============================================================================
# Test PerformanceAnalyzer
# ============================================================================

@pytest.mark.asyncio
async def test_performance_analyzer_basic():
    """Test basic performance analysis"""
    tracker = OutcomeTracker()

    # Track some outcomes
    for _ in range(10):
        await tracker.track(
            query="Test",
            query_complexity="SIMPLE",
            budget_allocated=4000,
            budget_used=3000,
            quality_overall=0.85,
            latency_ms=150.0,
            latency_packing_ms=50.0,
            latency_llm_ms=90.0,
            compression_level=CompressionLevel.SUMMARY,
            importance_threshold=0.3
        )

    analyzer = PerformanceAnalyzer(outcome_tracker=tracker)
    analysis = analyzer.analyze()

    assert "latency_breakdown" in analysis
    assert "budget_analysis" in analysis
    assert "quality_trends" in analysis


@pytest.mark.asyncio
async def test_bottleneck_detection():
    """Test bottleneck detection"""
    tracker = OutcomeTracker()

    # Create outcomes with slow packing (bottleneck)
    for _ in range(10):
        await tracker.track(
            query="Test",
            query_complexity="SIMPLE",
            budget_allocated=4000,
            budget_used=3000,
            quality_overall=0.85,
            latency_ms=200.0,
            latency_packing_ms=150.0,  # 75% of total (bottleneck!)
            latency_llm_ms=40.0,
            compression_level=CompressionLevel.SUMMARY,
            importance_threshold=0.3
        )

    analyzer = PerformanceAnalyzer(outcome_tracker=tracker)
    bottlenecks = analyzer.get_bottlenecks()

    # Should detect packing as bottleneck
    assert len(bottlenecks) > 0
    packing_bottleneck = [b for b in bottlenecks if b.component == "packing"]
    assert len(packing_bottleneck) > 0


@pytest.mark.asyncio
async def test_anomaly_detection():
    """Test anomaly detection"""
    tracker = OutcomeTracker()

    # Track baseline outcomes (good quality)
    for _ in range(20):
        await tracker.track(
            query="Test",
            query_complexity="SIMPLE",
            budget_allocated=4000,
            budget_used=3000,
            quality_overall=0.9,
            latency_ms=150.0,
            compression_level=CompressionLevel.SUMMARY,
            importance_threshold=0.3
        )

    # Track recent outcomes (quality dropped)
    for _ in range(6):
        await tracker.track(
            query="Test",
            query_complexity="SIMPLE",
            budget_allocated=4000,
            budget_used=3000,
            quality_overall=0.6,  # Dropped significantly
            latency_ms=150.0,
            compression_level=CompressionLevel.SUMMARY,
            importance_threshold=0.3
        )

    analyzer = PerformanceAnalyzer(outcome_tracker=tracker)
    anomalies = analyzer.detect_anomalies()

    # Should detect quality drop
    assert len(anomalies) > 0
    quality_anomalies = [a for a in anomalies if a.type == AnomalyType.QUALITY_DROP]
    assert len(quality_anomalies) > 0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_integration():
    """Test full integration: tracker + tuner + analyzer"""
    tracker = OutcomeTracker(min_samples_for_recommendation=5)
    tuner = AdaptiveThresholdTuner(
        outcome_tracker=tracker,
        enable_auto_tuning=True,
        tuning_strategy=TuningStrategy.BALANCED
    )
    analyzer = PerformanceAnalyzer(outcome_tracker=tracker)

    # Simulate 20 queries with varying performance
    for i in range(20):
        threshold = 0.3 if i < 10 else 0.4  # Change threshold halfway
        quality = 0.7 if i < 10 else 0.9    # Quality improves with threshold change

        await tracker.track(
            query=f"Query {i}",
            query_complexity="SIMPLE",
            budget_allocated=4000,
            budget_used=3000,
            quality_overall=quality,
            latency_ms=150.0,
            latency_packing_ms=50.0,
            latency_llm_ms=90.0,
            compression_level=CompressionLevel.SUMMARY,
            importance_threshold=threshold
        )

    # Get statistics
    stats = tracker.get_statistics()
    assert stats["total_outcomes"] == 20

    # Check for tuning recommendations
    recs = tracker.get_recommendations(complexity="SIMPLE", current_importance_threshold=0.3)
    # May or may not have recommendations depending on data

    # Analyze performance
    analysis = analyzer.analyze()
    assert "sample_size" in analysis
    assert analysis["sample_size"] == 20

    # Get bottlenecks (should be minimal with good data)
    bottlenecks = analyzer.get_bottlenecks()
    # May or may not have bottlenecks


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
