"""
Unit tests for Phase 6.2: CONSENSUS Refinement

Tests parallel strategy execution with ensemble voting.

Created: November 2025
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import List, Optional, Any

# Import Phase 6.2 components
from HoloLoom.awareness.consensus_refiner import (
    ConsensusRefiner,
    VotingMethod,
    StrategyResult,
    DisagreementPoint,
    ConsensusResult,
    RefinementStrategy
)


# ============================================================================
# Mock Classes
# ============================================================================

@dataclass
class MockRefinementResult:
    """Mock refinement result for testing"""
    query: str
    final_quality: float
    passes_executed: int = 1
    total_latency_ms: float = 100.0

    @property
    def passes(self):
        """Mock passes property"""
        return [Mock(strategy_used=RefinementStrategy.ADAPTIVE) for _ in range(self.passes_executed)]


class MockLLMContextPacker:
    """Mock context packer for testing"""

    def __init__(self, base_quality: float = 0.8):
        self.base_quality = base_quality
        self.pack_count = 0

    async def pack_and_generate(self, query, awareness_context, memory_results=None, max_memories=10):
        """Mock packing"""
        self.pack_count += 1
        await asyncio.sleep(0.001)  # Simulate async work
        return MockRefinementResult(
            query=query,
            final_quality=self.base_quality
        )


# ============================================================================
# Test Enums and Basic Data Structures
# ============================================================================

def test_voting_method_enum():
    """Test VotingMethod enum values"""
    assert VotingMethod.BEST_OF_N.value == "best_of_n"
    assert VotingMethod.QUALITY_WEIGHTED.value == "quality_weighted"
    assert VotingMethod.DIVERSITY.value == "diversity"
    assert VotingMethod.UNANIMOUS.value == "unanimous"


def test_strategy_result_creation():
    """Test StrategyResult creation and is_success"""
    # Successful result
    result = StrategyResult(
        strategy=RefinementStrategy.DEPTH_FIRST,
        result=MockRefinementResult(query="test", final_quality=0.9),
        quality=0.9,
        latency_ms=150.0
    )

    assert result.is_success() is True
    assert result.strategy == RefinementStrategy.DEPTH_FIRST
    assert result.quality == 0.9

    # Failed result
    failed_result = StrategyResult(
        strategy=RefinementStrategy.BREADTH_FIRST,
        result=None,
        quality=0.0,
        latency_ms=50.0,
        error="Timeout"
    )

    assert failed_result.is_success() is False
    assert failed_result.error == "Timeout"


def test_disagreement_point_creation():
    """Test DisagreementPoint creation"""
    dp = DisagreementPoint(
        dimension="quality",
        strategies=[RefinementStrategy.DEPTH_FIRST, RefinementStrategy.BREADTH_FIRST],
        values=[0.9, 0.7],
        severity=0.8,
        description="Quality differs by 0.2"
    )

    assert dp.dimension == "quality"
    assert len(dp.strategies) == 2
    assert dp.severity == 0.8


# ============================================================================
# Test ConsensusResult
# ============================================================================

def test_consensus_result_basic():
    """Test ConsensusResult basic properties"""
    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.9),
            quality=0.9,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.85),
            quality=0.85,
            latency_ms=120.0
        )
    ]

    consensus = ConsensusResult(
        selected_result=strategy_results[0].result,
        selected_strategy=RefinementStrategy.DEPTH_FIRST,
        consensus_confidence=0.92,
        agreement_level=0.88,
        voting_method=VotingMethod.QUALITY_WEIGHTED,
        strategy_results=strategy_results,
        successful_strategies=2,
        failed_strategies=0,
        disagreement_points=[],
        has_major_disagreement=False,
        total_latency_ms=120.0,
        parallel_speedup=1.8,
        vote_distribution={
            RefinementStrategy.DEPTH_FIRST: 0.6,
            RefinementStrategy.BREADTH_FIRST: 0.4
        }
    )

    assert consensus.consensus_confidence == 0.92
    assert consensus.agreement_level == 0.88
    assert consensus.successful_strategies == 2
    assert consensus.failed_strategies == 0


def test_consensus_result_quality_range():
    """Test ConsensusResult quality range calculation"""
    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.9),
            quality=0.9,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.7),
            quality=0.7,
            latency_ms=120.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.FOCUSED,
            result=MockRefinementResult(query="test", final_quality=0.85),
            quality=0.85,
            latency_ms=110.0
        )
    ]

    consensus = ConsensusResult(
        selected_result=strategy_results[0].result,
        selected_strategy=RefinementStrategy.DEPTH_FIRST,
        consensus_confidence=0.88,
        agreement_level=0.75,
        voting_method=VotingMethod.BEST_OF_N,
        strategy_results=strategy_results,
        successful_strategies=3,
        failed_strategies=0,
        disagreement_points=[],
        has_major_disagreement=False,
        total_latency_ms=120.0,
        parallel_speedup=2.8,
        vote_distribution={}
    )

    min_q, max_q = consensus.get_quality_range()
    assert min_q == 0.7
    assert max_q == 0.9


def test_consensus_result_summary():
    """Test ConsensusResult summary generation"""
    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.9),
            quality=0.9,
            latency_ms=100.0
        )
    ]

    consensus = ConsensusResult(
        selected_result=strategy_results[0].result,
        selected_strategy=RefinementStrategy.DEPTH_FIRST,
        consensus_confidence=0.92,
        agreement_level=1.0,
        voting_method=VotingMethod.BEST_OF_N,
        strategy_results=strategy_results,
        successful_strategies=1,
        failed_strategies=0,
        disagreement_points=[],
        has_major_disagreement=False,
        total_latency_ms=100.0,
        parallel_speedup=1.0,
        vote_distribution={RefinementStrategy.DEPTH_FIRST: 1.0}
    )

    summary = consensus.get_summary()
    assert "depth_first" in summary
    assert "0.92" in summary  # Consensus confidence
    assert "1.0x" in summary  # Speedup


# ============================================================================
# Test ConsensusRefiner Initialization
# ============================================================================

def test_consensus_refiner_initialization():
    """Test ConsensusRefiner initialization with defaults"""
    packer = MockLLMContextPacker()

    refiner = ConsensusRefiner(
        packer=packer,
        voting_method=VotingMethod.QUALITY_WEIGHTED
    )

    assert refiner.packer == packer
    assert refiner.voting_method == VotingMethod.QUALITY_WEIGHTED
    assert len(refiner.strategies) == 3  # Default strategies
    assert RefinementStrategy.DEPTH_FIRST in refiner.strategies
    assert RefinementStrategy.BREADTH_FIRST in refiner.strategies
    assert RefinementStrategy.FOCUSED in refiner.strategies


def test_consensus_refiner_custom_strategies():
    """Test ConsensusRefiner with custom strategies"""
    packer = MockLLMContextPacker()

    # Use only strategies that exist in both the fallback and real enum
    custom_strategies = [
        RefinementStrategy.DEPTH_FIRST,
        RefinementStrategy.ADAPTIVE
    ]

    refiner = ConsensusRefiner(
        packer=packer,
        strategies=custom_strategies,
        voting_method=VotingMethod.DIVERSITY
    )

    assert len(refiner.strategies) == 2
    assert RefinementStrategy.DEPTH_FIRST in refiner.strategies
    assert RefinementStrategy.ADAPTIVE in refiner.strategies


# ============================================================================
# Test Voting Methods
# ============================================================================

def test_vote_best_of_n():
    """Test BEST_OF_N voting (simple maximum)"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(packer=packer, voting_method=VotingMethod.BEST_OF_N)

    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.9),
            quality=0.9,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.75),
            quality=0.75,
            latency_ms=120.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.FOCUSED,
            result=MockRefinementResult(query="test", final_quality=0.85),
            quality=0.85,
            latency_ms=110.0
        )
    ]

    selected, vote_dist = refiner._vote_best_of_n(strategy_results)

    assert selected.strategy == RefinementStrategy.DEPTH_FIRST
    assert selected.quality == 0.9
    assert vote_dist[RefinementStrategy.DEPTH_FIRST] == 1.0
    assert vote_dist[RefinementStrategy.BREADTH_FIRST] == 0.0
    assert vote_dist[RefinementStrategy.FOCUSED] == 0.0


def test_vote_quality_weighted():
    """Test QUALITY_WEIGHTED voting"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(packer=packer, voting_method=VotingMethod.QUALITY_WEIGHTED)

    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.9),
            quality=0.9,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.6),
            quality=0.6,
            latency_ms=120.0
        )
    ]

    selected, vote_dist = refiner._vote_quality_weighted(strategy_results)

    # Total quality: 0.9 + 0.6 = 1.5
    # DEPTH_FIRST weight: 0.9 / 1.5 = 0.6
    # BREADTH_FIRST weight: 0.6 / 1.5 = 0.4

    assert selected.strategy == RefinementStrategy.DEPTH_FIRST
    assert abs(vote_dist[RefinementStrategy.DEPTH_FIRST] - 0.6) < 0.01
    assert abs(vote_dist[RefinementStrategy.BREADTH_FIRST] - 0.4) < 0.01


def test_vote_diversity():
    """Test DIVERSITY voting (prefers diverse perspectives)"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(packer=packer, voting_method=VotingMethod.DIVERSITY)

    # High variance (diverse) results
    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.9),
            quality=0.9,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.5),
            quality=0.5,
            latency_ms=120.0
        )
    ]

    selected, vote_dist = refiner._vote_diversity(strategy_results)

    # Should select best, but with diversity bonus considered
    assert selected.strategy in [RefinementStrategy.DEPTH_FIRST, RefinementStrategy.BREADTH_FIRST]


def test_vote_unanimous():
    """Test UNANIMOUS voting (requires agreement)"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(
        packer=packer,
        voting_method=VotingMethod.UNANIMOUS,
        unanimity_threshold=0.8
    )

    # High agreement (all similar quality)
    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.88),
            quality=0.88,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.90),
            quality=0.90,
            latency_ms=120.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.FOCUSED,
            result=MockRefinementResult(query="test", final_quality=0.85),
            quality=0.85,
            latency_ms=110.0
        )
    ]

    selected, vote_dist = refiner._vote_unanimous(strategy_results)

    # Should select best among agreeing strategies
    assert selected.quality >= 0.85


# ============================================================================
# Test Agreement Calculation
# ============================================================================

def test_calculate_agreement_high():
    """Test agreement calculation with high agreement"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(packer=packer)

    # High agreement (similar quality)
    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.88),
            quality=0.88,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.90),
            quality=0.90,
            latency_ms=120.0
        )
    ]

    agreement = refiner._calculate_agreement(strategy_results)

    assert agreement > 0.9  # High agreement


def test_calculate_agreement_low():
    """Test agreement calculation with low agreement"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(packer=packer)

    # Low agreement (very different quality)
    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.9),
            quality=0.9,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.3),
            quality=0.3,
            latency_ms=120.0
        )
    ]

    agreement = refiner._calculate_agreement(strategy_results)

    assert agreement < 0.7  # Low agreement


# ============================================================================
# Test Disagreement Detection
# ============================================================================

def test_detect_disagreements_quality():
    """Test disagreement detection based on quality"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(packer=packer, enable_disagreement_detection=True)

    # Significant quality disagreement (>15% range)
    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.9),
            quality=0.9,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.65),
            quality=0.65,
            latency_ms=120.0
        )
    ]

    disagreements = refiner._detect_disagreements(strategy_results)

    assert len(disagreements) > 0
    quality_disagreement = [d for d in disagreements if d.dimension == "quality"]
    assert len(quality_disagreement) > 0
    assert quality_disagreement[0].severity > 0


def test_detect_disagreements_passes():
    """Test disagreement detection based on passes"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(packer=packer, enable_disagreement_detection=True)

    # Different pass counts
    strategy_results = [
        StrategyResult(
            strategy=RefinementStrategy.DEPTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.9, passes_executed=3),
            quality=0.9,
            latency_ms=100.0
        ),
        StrategyResult(
            strategy=RefinementStrategy.BREADTH_FIRST,
            result=MockRefinementResult(query="test", final_quality=0.85, passes_executed=1),
            quality=0.85,
            latency_ms=120.0
        )
    ]

    disagreements = refiner._detect_disagreements(strategy_results)

    passes_disagreement = [d for d in disagreements if d.dimension == "passes"]
    assert len(passes_disagreement) > 0


# ============================================================================
# Test Consensus Confidence Calculation
# ============================================================================

def test_calculate_consensus_confidence():
    """Test consensus confidence calculation"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(packer=packer)

    selected_result = StrategyResult(
        strategy=RefinementStrategy.DEPTH_FIRST,
        result=MockRefinementResult(query="test", final_quality=0.9),
        quality=0.9,
        latency_ms=100.0
    )

    agreement_level = 0.95
    vote_distribution = {
        RefinementStrategy.DEPTH_FIRST: 0.7,
        RefinementStrategy.BREADTH_FIRST: 0.3
    }

    confidence = refiner._calculate_consensus_confidence(
        selected_result=selected_result,
        agreement_level=agreement_level,
        vote_distribution=vote_distribution
    )

    # Confidence = 0.9 * 0.5 (quality) + 0.95 * 0.3 (agreement) + 0.7 * 0.2 (vote)
    #            = 0.45 + 0.285 + 0.14 = 0.875

    assert confidence > 0.8
    assert confidence <= 1.0


# ============================================================================
# Test Parallel Execution
# ============================================================================

@pytest.mark.asyncio
async def test_parallel_execution():
    """Test parallel strategy execution"""
    packer = MockLLMContextPacker(base_quality=0.85)

    refiner = ConsensusRefiner(
        packer=packer,
        strategies=[
            RefinementStrategy.DEPTH_FIRST,
            RefinementStrategy.BREADTH_FIRST
        ],
        voting_method=VotingMethod.BEST_OF_N
    )

    # Mock _execute_single_strategy to return StrategyResult
    async def mock_execute(strategy, query, awareness_ctx, memory_results, **kwargs):
        await asyncio.sleep(0.01)  # Simulate work
        return StrategyResult(
            strategy=strategy,
            result=MockRefinementResult(query=query, final_quality=0.85),
            quality=0.85,
            latency_ms=10.0
        )

    refiner._execute_single_strategy = mock_execute

    # Execute in parallel
    strategy_results = await refiner._execute_strategies_parallel(
        query="test query",
        awareness_ctx=Mock(),
        memory_results=None
    )

    assert len(strategy_results) == 2
    assert all(sr.is_success() for sr in strategy_results)


# ============================================================================
# Test Full Refinement Flow
# ============================================================================

@pytest.mark.asyncio
async def test_full_consensus_refinement():
    """Test full consensus refinement flow"""
    packer = MockLLMContextPacker(base_quality=0.85)

    refiner = ConsensusRefiner(
        packer=packer,
        strategies=[
            RefinementStrategy.DEPTH_FIRST,
            RefinementStrategy.BREADTH_FIRST
        ],
        voting_method=VotingMethod.QUALITY_WEIGHTED
    )

    # Mock _execute_single_strategy
    async def mock_execute(strategy, query, awareness_ctx, memory_results, **kwargs):
        quality = 0.9 if strategy == RefinementStrategy.DEPTH_FIRST else 0.8
        return StrategyResult(
            strategy=strategy,
            result=MockRefinementResult(query=query, final_quality=quality),
            quality=quality,
            latency_ms=10.0
        )

    refiner._execute_single_strategy = mock_execute

    # Execute refinement
    result = await refiner.refine(
        query="Test consensus query",
        awareness_ctx=Mock(),
        memory_results=None
    )

    assert isinstance(result, ConsensusResult)
    assert result.selected_strategy == RefinementStrategy.DEPTH_FIRST  # Higher quality
    assert result.consensus_confidence > 0.0
    assert result.agreement_level > 0.0
    assert result.successful_strategies == 2
    assert result.failed_strategies == 0


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_all_strategies_fail():
    """Test handling when all strategies fail"""
    packer = MockLLMContextPacker()

    refiner = ConsensusRefiner(
        packer=packer,
        strategies=[RefinementStrategy.DEPTH_FIRST, RefinementStrategy.BREADTH_FIRST]
    )

    # Mock all strategies to fail
    async def mock_execute_fail(strategy, query, awareness_ctx, memory_results, **kwargs):
        return StrategyResult(
            strategy=strategy,
            result=None,
            quality=0.0,
            latency_ms=0.0,
            error="Failed"
        )

    refiner._execute_single_strategy = mock_execute_fail

    # Should raise ValueError
    with pytest.raises(ValueError, match="All strategies failed"):
        await refiner.refine(
            query="Test query",
            awareness_ctx=Mock(),
            memory_results=None
        )


def test_statistics_tracking():
    """Test statistics tracking"""
    packer = MockLLMContextPacker()
    refiner = ConsensusRefiner(packer=packer)

    # Initially zero
    stats = refiner.get_statistics()
    assert stats['total_refinements'] == 0
    assert stats['avg_parallel_time_ms'] == 0.0

    # Simulate some refinements
    refiner.total_refinements = 5
    refiner.total_parallel_time_ms = 500.0
    refiner.total_sequential_time_ms = 1500.0

    stats = refiner.get_statistics()
    assert stats['total_refinements'] == 5
    assert stats['avg_parallel_time_ms'] == 100.0  # 500 / 5
    assert stats['avg_sequential_time_ms'] == 300.0  # 1500 / 5
    assert stats['avg_parallel_speedup'] == 3.0  # 300 / 100
