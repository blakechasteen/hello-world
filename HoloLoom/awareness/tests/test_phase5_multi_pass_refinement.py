#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Phase 5: Multi-Pass Refinement

Tests:
1. RefinementEngine
   - Basic refinement (quality improvement)
   - Strategy selection (ADAPTIVE, DEPTH_FIRST, BREADTH_FIRST, FOCUSED)
   - Stopping criteria (quality threshold, max passes, diminishing returns, time limit)
   - Configuration adjustments per strategy
   - Statistics tracking

2. Integration with LLMContextPacker
   - pack_and_generate_with_refinement wrapper
   - Graceful fallback when refinement unavailable
   - Quality improvement across passes

3. Edge Cases
   - Already high quality (skip refinement)
   - No improvement (diminishing returns)
   - Time limit exceeded
   - Force refinement
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional, List, Any

# Import components to test
from HoloLoom.awareness.refinement_engine import (
    RefinementEngine,
    RefinementStrategy,
    StoppingCriterion,
    RefinementPass,
    RefinementResult
)


# ============================================================================
# Mock Objects for Testing
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
    """Mock LLMContextPacker for testing"""

    def __init__(self, quality_progression: Optional[List[float]] = None):
        """
        Args:
            quality_progression: List of quality scores to return on successive calls
                               Default: [0.7, 0.8, 0.9] (improving quality)
        """
        self.quality_progression = quality_progression or [0.7, 0.8, 0.9]
        self.call_count = 0

        # Configuration that refinement might modify
        self.enable_compression = True
        self.min_importance = 0.7
        self.budget = Mock(total=5000)

    async def pack_and_generate(
        self,
        query: str,
        awareness_context: Any,
        memory_results: Optional[List[Any]] = None,
        max_memories: int = 10
    ) -> MockPackedGeneration:
        """Mock pack and generate"""
        # Get quality for this call
        quality = self.quality_progression[min(self.call_count, len(self.quality_progression) - 1)]
        self.call_count += 1

        # Create quality score
        quality_score = MockQualityScore(
            coherence=quality,
            completeness=quality,
            relevance=quality,
            overall=quality
        )

        return MockPackedGeneration(
            response=f"Response with quality {quality:.2f}",
            quality_score=quality_score
        )


# ============================================================================
# Test RefinementEngine - Basic Functionality
# ============================================================================

@pytest.mark.asyncio
async def test_refinement_basic_quality_improvement():
    """Test basic refinement improves quality"""
    # Create packer that improves quality: 0.7 → 0.8 → 0.9
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.8, 0.9])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.85,
        max_passes=3
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should have improved quality
    assert result.initial_quality == 0.7
    assert result.final_quality > result.initial_quality

    # Should have executed refinement passes
    assert result.passes_executed > 0
    assert len(result.passes) > 0


@pytest.mark.asyncio
async def test_skip_refinement_high_quality():
    """Test refinement skipped when quality already high"""
    # Create packer with high quality from start
    packer = MockLLMContextPacker(quality_progression=[0.95, 0.95, 0.95])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.85,
        max_passes=3
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should skip refinement
    assert result.passes_executed == 0
    assert len(result.passes) == 0
    assert result.stopping_criterion == StoppingCriterion.QUALITY_THRESHOLD
    assert result.initial_quality == result.final_quality


@pytest.mark.asyncio
async def test_force_refinement():
    """Test force_refinement bypasses quality check"""
    # Create packer with high quality
    packer = MockLLMContextPacker(quality_progression=[0.95, 0.96, 0.97])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.85,
        max_passes=2
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10,
        force_refinement=True  # Force refinement even though quality is high
    )

    # Should execute refinement passes despite high initial quality
    assert result.passes_executed > 0
    assert len(result.passes) > 0


# ============================================================================
# Test RefinementEngine - Stopping Criteria
# ============================================================================

@pytest.mark.asyncio
async def test_stopping_quality_threshold():
    """Test stopping when quality threshold reached"""
    # Quality: 0.7 → 0.9 (exceeds threshold of 0.85)
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.9, 0.95])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.85,
        max_passes=5
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should stop when quality threshold reached
    assert result.stopping_criterion == StoppingCriterion.QUALITY_THRESHOLD
    assert result.final_quality >= 0.85
    assert result.passes_executed < 5  # Stopped early


@pytest.mark.asyncio
async def test_stopping_max_passes():
    """Test stopping when max passes reached"""
    # Quality never reaches threshold: 0.7 → 0.75 → 0.78 → 0.80
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.75, 0.78, 0.80])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.90,  # High threshold, won't be reached
        max_passes=3
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should stop at max passes
    assert result.stopping_criterion == StoppingCriterion.MAX_PASSES
    assert result.passes_executed == 3


@pytest.mark.asyncio
async def test_stopping_diminishing_returns():
    """Test stopping when improvement too small"""
    # Quality improves minimally: 0.7 → 0.71 → 0.711 (improvement < 2%)
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.71, 0.711, 0.712])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.90,
        max_passes=5,
        min_improvement_per_pass=0.02  # Require 2% improvement
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should stop due to diminishing returns
    assert result.stopping_criterion == StoppingCriterion.DIMINISHING_RETURNS
    assert result.passes_executed < 5  # Stopped early


@pytest.mark.asyncio
async def test_stopping_time_limit():
    """Test stopping when time limit exceeded"""
    # Create packer with slow responses that show improvement
    async def slow_pack_and_generate(*args, **kwargs):
        await asyncio.sleep(0.3)  # 300ms per call
        # Return improving quality to avoid diminishing returns
        return MockPackedGeneration(
            quality_score=MockQualityScore(
                overall=slow_pack_and_generate.counter * 0.05 + 0.7
            )
        )

    slow_pack_and_generate.counter = 0

    # Wrap to track counter
    original_pack = slow_pack_and_generate

    async def tracked_pack(*args, **kwargs):
        result = await original_pack(*args, **kwargs)
        slow_pack_and_generate.counter += 1
        return result

    packer = MockLLMContextPacker()
    packer.pack_and_generate = tracked_pack

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.95,  # High threshold, won't be reached
        max_passes=10,
        time_limit_seconds=0.8,  # 800ms limit (allows ~2 passes at 300ms each)
        min_improvement_per_pass=0.01  # Low threshold to avoid diminishing returns
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should stop due to time limit
    assert result.stopping_criterion == StoppingCriterion.TIME_LIMIT
    assert result.passes_executed < 10  # Stopped early


# ============================================================================
# Test RefinementEngine - Strategy Selection
# ============================================================================

@pytest.mark.asyncio
async def test_strategy_depth_first():
    """Test DEPTH_FIRST strategy"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=2,
        strategy=RefinementStrategy.DEPTH_FIRST
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should use DEPTH_FIRST strategy
    assert all(p.strategy_used == RefinementStrategy.DEPTH_FIRST for p in result.passes)


@pytest.mark.asyncio
async def test_strategy_breadth_first():
    """Test BREADTH_FIRST strategy"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=2,
        strategy=RefinementStrategy.BREADTH_FIRST
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should use BREADTH_FIRST strategy
    assert all(p.strategy_used == RefinementStrategy.BREADTH_FIRST for p in result.passes)


@pytest.mark.asyncio
async def test_strategy_focused():
    """Test FOCUSED strategy"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=2,
        strategy=RefinementStrategy.FOCUSED
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should use FOCUSED strategy
    assert all(p.strategy_used == RefinementStrategy.FOCUSED for p in result.passes)


@pytest.mark.asyncio
async def test_strategy_adaptive():
    """Test ADAPTIVE strategy selection"""
    # Create packer with low completeness
    class MockPackerWeakCompleteness(MockLLMContextPacker):
        async def pack_and_generate(self, *args, **kwargs):
            gen = await super().pack_and_generate(*args, **kwargs)
            # Make completeness weak
            gen.quality_score.completeness = 0.5
            gen.quality_score.coherence = 0.8
            gen.quality_score.relevance = 0.8
            return gen

    packer = MockPackerWeakCompleteness(quality_progression=[0.7, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=2,
        strategy=RefinementStrategy.ADAPTIVE
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should select BREADTH_FIRST for low completeness
    if len(result.passes) > 0:
        assert result.passes[0].strategy_used == RefinementStrategy.BREADTH_FIRST


# ============================================================================
# Test RefinementEngine - Configuration Adjustments
# ============================================================================

@pytest.mark.asyncio
async def test_config_adjustments_depth_first():
    """Test configuration adjustments for DEPTH_FIRST"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=1,
        strategy=RefinementStrategy.DEPTH_FIRST,
        enable_compression_disable=True,
        enable_budget_increase=False  # Disable budget increase for simpler test
    )

    # Store original config
    original_compression = packer.enable_compression
    original_threshold = packer.min_importance

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Config should be restored after refinement
    assert packer.enable_compression == original_compression
    assert packer.min_importance == original_threshold


@pytest.mark.asyncio
async def test_config_adjustments_restored():
    """Test configuration restored after refinement"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    # Store original values
    original_compression = packer.enable_compression
    original_importance = packer.min_importance

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=2,
        strategy=RefinementStrategy.DEPTH_FIRST,
        enable_budget_increase=False  # Disable for simpler test
    )

    await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # All config should be restored
    assert packer.enable_compression == original_compression
    assert packer.min_importance == original_importance


# ============================================================================
# Test RefinementEngine - Pass Tracking
# ============================================================================

@pytest.mark.asyncio
async def test_pass_tracking():
    """Test refinement pass tracking"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.8, 0.9])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.85,
        max_passes=3
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should track all passes
    assert len(result.passes) == result.passes_executed

    # Each pass should have quality improvement
    for i, pass_result in enumerate(result.passes):
        assert pass_result.pass_number == i + 1
        assert pass_result.quality_after > pass_result.quality_before
        assert pass_result.quality_improvement > 0
        assert pass_result.latency_ms > 0


@pytest.mark.asyncio
async def test_best_pass_tracking():
    """Test tracking of best pass"""
    # Quality improves then degrades: 0.7 → 0.9 → 0.85
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.9, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.95,  # Won't be reached
        max_passes=3
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Best pass should be pass 1 (quality 0.9)
    assert result.best_pass_number == 1
    assert result.final_quality == 0.9  # Best quality


# ============================================================================
# Test RefinementEngine - Statistics
# ============================================================================

@pytest.mark.asyncio
async def test_statistics_tracking():
    """Test refinement statistics tracking"""
    # Use low initial quality to ensure refinement happens both times
    packer1 = MockLLMContextPacker(quality_progression=[0.7, 0.85])
    packer2 = MockLLMContextPacker(quality_progression=[0.65, 0.80])

    refiner = RefinementEngine(
        packer=packer1,
        quality_threshold=0.90,  # High threshold ensures refinement
        max_passes=2
    )

    # Execute refinement first time
    await refiner.refine(
        query="Query 1",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Switch to second packer for second refinement
    refiner.packer = packer2

    # Execute refinement second time
    await refiner.refine(
        query="Query 2",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    stats = refiner.get_statistics()

    # Should track statistics
    assert stats["total_refinements"] == 2
    assert stats["total_passes_executed"] >= 0
    assert stats["avg_passes_per_refinement"] >= 0
    assert "avg_quality_improvement" in stats


@pytest.mark.asyncio
async def test_reset_statistics():
    """Test resetting refinement statistics"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=2
    )

    # Execute refinement
    await refiner.refine(
        query="Query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Reset statistics
    refiner.reset_statistics()

    stats = refiner.get_statistics()

    # All counters should be zero
    assert stats["total_refinements"] == 0
    assert stats["total_passes_executed"] == 0
    assert stats["avg_quality_improvement"] == 0.0


# ============================================================================
# Test RefinementResult
# ============================================================================

@pytest.mark.asyncio
async def test_refinement_result_summary():
    """Test refinement result summary"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=2
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    summary = result.summary()

    # Should contain key information
    assert "Quality:" in summary
    assert "Passes:" in summary
    assert "Stopped:" in summary


@pytest.mark.asyncio
async def test_refinement_result_metrics():
    """Test refinement result metrics"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=2
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Check all metrics present
    assert result.query == "Test query"
    assert result.initial_generation is not None
    assert result.final_generation is not None
    assert result.initial_quality >= 0.0
    assert result.final_quality >= 0.0
    assert result.total_improvement >= 0.0
    assert result.total_latency_ms > 0
    assert result.avg_latency_per_pass_ms >= 0


# ============================================================================
# Test Integration with LLMContextPacker
# ============================================================================

@pytest.mark.asyncio
async def test_integration_wrapper_method():
    """Test pack_and_generate_with_refinement wrapper"""
    # This test requires importing the actual LLMContextPacker
    # For now, we'll test the logic with mocks

    # Mock the wrapper behavior
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    # Simulate the wrapper creating a RefinementEngine
    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=3,
        strategy=RefinementStrategy.ADAPTIVE
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=None,
        max_memories=10
    )

    # Should return RefinementResult
    assert isinstance(result, RefinementResult)
    assert result.final_quality > result.initial_quality


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_edge_case_no_improvement():
    """Test when refinement provides no improvement"""
    # Quality stays same: 0.7 → 0.7 → 0.7
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.7, 0.7])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.90,
        max_passes=3,
        min_improvement_per_pass=0.01
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should stop due to diminishing returns
    assert result.stopping_criterion == StoppingCriterion.DIMINISHING_RETURNS
    assert result.total_improvement == 0.0


@pytest.mark.asyncio
async def test_edge_case_quality_degradation():
    """Test when quality degrades (still keeps best)"""
    # Quality: 0.7 → 0.9 → 0.6 (degrades on last pass)
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.9, 0.6])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.95,  # Won't be reached
        max_passes=3
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should keep best result (0.9 from pass 1)
    assert result.final_quality == 0.9
    assert result.best_pass_number == 1


@pytest.mark.asyncio
async def test_edge_case_single_pass():
    """Test refinement with max_passes=1"""
    packer = MockLLMContextPacker(quality_progression=[0.7, 0.85])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.80,
        max_passes=1  # Only one refinement pass
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should execute exactly 1 pass
    assert result.passes_executed == 1
    assert len(result.passes) == 1


@pytest.mark.asyncio
async def test_edge_case_zero_threshold():
    """Test with quality_threshold=0.0 (always acceptable)"""
    packer = MockLLMContextPacker(quality_progression=[0.5, 0.6])

    refiner = RefinementEngine(
        packer=packer,
        quality_threshold=0.0,  # Any quality is acceptable
        max_passes=3
    )

    result = await refiner.refine(
        query="Test query",
        awareness_ctx=Mock(),
        memory_results=[],
        max_memories=10
    )

    # Should skip refinement (quality already acceptable)
    assert result.passes_executed == 0
    assert result.stopping_criterion == StoppingCriterion.QUALITY_THRESHOLD


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
