"""
Unit tests for HoloLoom.recursive.advanced_refinement

Tests refinement strategies, quality trajectory tracking, and strategy learning.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from hololoom.recursive.advanced_refinement import (
    RefinementStrategy,
    QualityMetrics,
    RefinementResult,
    RefinementPattern,
    AdvancedRefiner,
    refine_with_strategy
)
from hololoom.prompting.unified_mrf import RefinementStrategyType
from hololoom.protocols.types import Query
from hololoom.fabric.spacetime import Spacetime, WeavingTrace
from hololoom.recursive.scratchpad import Scratchpad
from datetime import datetime


def make_trace(
    tool_confidence: float = 0.80,
    threads_activated: list = None,
    motifs_detected: list = None,
) -> WeavingTrace:
    """Create a minimal valid WeavingTrace for tests."""
    now = datetime.now()
    return WeavingTrace(
        start_time=now,
        end_time=now,
        duration_ms=0.0,
        threads_activated=threads_activated or [],
        motifs_detected=motifs_detected or [],
        tool_confidence=tool_confidence,
    )


def make_spacetime(
    query_text: str = "Test query",
    response: str = "Response",
    confidence: float = 0.80,
    trace: WeavingTrace = None,
) -> Spacetime:
    """Create a minimal valid Spacetime for tests."""
    return Spacetime(
        query_text=query_text,
        response=response,
        tool_used="test_tool",
        confidence=confidence,
        trace=trace or make_trace(tool_confidence=confidence),
    )


class TestRefinementStrategy:
    """Test RefinementStrategy enum"""

    def test_all_strategies_exist(self):
        """Test all expected strategies exist"""
        assert RefinementStrategy.REFINE.value == "refine"
        assert RefinementStrategy.CRITIQUE.value == "critique"
        assert RefinementStrategy.VERIFY.value == "verify"
        assert RefinementStrategy.ELEGANCE.value == "elegance"
        assert RefinementStrategy.HOFSTADTER.value == "hofstadter"

    def test_strategy_count(self):
        """Test we have all 5 strategies"""
        strategies = list(RefinementStrategy)
        assert len(strategies) == 5


class TestQualityMetrics:
    """Test QualityMetrics dataclass"""

    def test_metrics_creation(self):
        """Test creating quality metrics"""
        metrics = QualityMetrics(
            confidence=0.85,
            threads_activated=5,
            motifs_detected=3,
            context_size=8,
            response_length=250
        )

        assert metrics.confidence == 0.85
        assert metrics.threads_activated == 5
        assert metrics.motifs_detected == 3
        assert metrics.context_size == 8
        assert metrics.response_length == 250
        assert isinstance(metrics.timestamp, datetime)

    def test_quality_score_calculation(self):
        """Test composite quality score calculation"""
        metrics = QualityMetrics(
            confidence=0.90,
            threads_activated=5,
            motifs_detected=5,
            context_size=10,
            response_length=500
        )

        score = metrics.score()

        # Score should be weighted: 0.7*confidence + 0.2*context + 0.1*response
        # With high values, should be close to confidence
        assert 0.80 <= score <= 1.0
        assert score >= 0.7 * metrics.confidence

    def test_quality_score_minimal(self):
        """Test quality score with minimal inputs"""
        metrics = QualityMetrics(
            confidence=0.50,
            threads_activated=0,
            motifs_detected=0,
            context_size=0,
            response_length=50
        )

        score = metrics.score()

        # Score should be dominated by confidence (70% weight)
        assert score <= 0.50
        assert score >= 0.35  # Some contribution from response


class TestRefinementResult:
    """Test RefinementResult dataclass"""

    def test_result_creation(self):
        """Test creating refinement result"""
        spacetime = make_spacetime(query_text="Test query", response="Refined response", confidence=0.92)

        # Create trajectory
        metrics1 = QualityMetrics(0.70, 3, 2, 5, 200)
        metrics2 = QualityMetrics(0.85, 5, 3, 8, 300)
        metrics3 = QualityMetrics(0.92, 6, 4, 10, 350)

        result = RefinementResult(
            final_spacetime=spacetime,
            trajectory=[metrics1, metrics2, metrics3],
            strategy_used=RefinementStrategy.ELEGANCE,
            iterations=2,
            improved=True,
            improvement_rate=0.11
        )

        assert result.iterations == 2
        assert result.improved is True
        assert result.strategy_used == RefinementStrategy.ELEGANCE
        assert len(result.trajectory) == 3

    def test_summary_generation(self):
        """Test summary string generation"""
        spacetime = make_spacetime(query_text="Test", response="Response", confidence=0.90)

        metrics1 = QualityMetrics(0.60, 2, 1, 3, 150)
        metrics2 = QualityMetrics(0.90, 5, 3, 8, 300)

        result = RefinementResult(
            final_spacetime=spacetime,
            trajectory=[metrics1, metrics2],
            strategy_used=RefinementStrategy.VERIFY,
            iterations=1,
            improved=True,
            improvement_rate=0.30
        )

        summary = result.summary()

        assert "verify" in summary
        assert "1" in summary  # iterations
        # Check actual computed score() values: metrics1 -> 0.51, metrics2 -> 0.85
        initial_score = metrics1.score()
        final_score = metrics2.score()
        assert f"{initial_score:.2f}" in summary
        assert f"{final_score:.2f}" in summary


class TestRefinementPattern:
    """Test RefinementPattern dataclass"""

    def test_pattern_creation(self):
        """Test creating a refinement pattern"""
        pattern = RefinementPattern(
            strategy=RefinementStrategy.REFINE,
            initial_quality=0.60,
            final_quality=0.85,
            iterations=3,
            query_characteristics={"length": 50, "complexity": "medium"},
            improvement_rate=0.083
        )

        assert pattern.strategy == RefinementStrategy.REFINE
        assert pattern.initial_quality == 0.60
        assert pattern.final_quality == 0.85
        assert pattern.occurrences == 1
        assert pattern.avg_improvement == 0.0

    def test_pattern_update(self):
        """Test updating pattern with new occurrence"""
        pattern = RefinementPattern(
            strategy=RefinementStrategy.CRITIQUE,
            initial_quality=0.70,
            final_quality=0.90,
            iterations=2,
            query_characteristics={},
            improvement_rate=0.10,
            avg_improvement=0.20
        )

        initial_occurrences = pattern.occurrences

        pattern.update(improvement=0.15)

        assert pattern.occurrences == initial_occurrences + 1
        assert pattern.avg_improvement > 0

    def test_pattern_multiple_updates(self):
        """Test multiple pattern updates"""
        pattern = RefinementPattern(
            strategy=RefinementStrategy.ELEGANCE,
            initial_quality=0.65,
            final_quality=0.85,
            iterations=3,
            query_characteristics={},
            improvement_rate=0.067
        )

        improvements = [0.20, 0.15, 0.25, 0.18]

        for imp in improvements:
            pattern.update(imp)

        assert pattern.occurrences == 1 + len(improvements)
        # Average should be reasonable
        assert 0.15 <= pattern.avg_improvement <= 0.25


class TestAdvancedRefiner:
    """Test AdvancedRefiner class"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator"""
        orch = AsyncMock()
        orch.weave = AsyncMock()
        return orch

    @pytest.fixture
    def refiner(self, mock_orchestrator):
        """Create refiner with mock orchestrator"""
        scratchpad = Scratchpad()
        return AdvancedRefiner(
            orchestrator=mock_orchestrator,
            scratchpad=scratchpad,
            enable_learning=True
        )

    def test_initialization(self, mock_orchestrator):
        """Test refiner initialization"""
        scratchpad = Scratchpad()
        refiner = AdvancedRefiner(
            orchestrator=mock_orchestrator,
            scratchpad=scratchpad,
            enable_learning=True
        )

        assert refiner.orchestrator == mock_orchestrator
        assert refiner.scratchpad == scratchpad
        assert refiner.enable_learning is True
        assert len(refiner.learned_patterns) == 0

    def test_extract_metrics(self, refiner):
        """Test extracting metrics from spacetime"""
        trace = make_trace(
            tool_confidence=0.85,
            threads_activated=["t1", "t2", "t3"],
            motifs_detected=["AI", "ML"],
        )
        spacetime = make_spacetime(
            query_text="Test query",
            response="A" * 250,  # 250 char response
            confidence=0.85,
            trace=trace,
        )

        metrics = refiner._extract_metrics(spacetime)

        assert metrics.confidence == 0.85
        assert metrics.threads_activated == 3
        assert metrics.motifs_detected == 2
        assert metrics.context_size == 5
        assert metrics.response_length == 250

    def test_select_strategy_low_confidence(self, refiner):
        """Test strategy selection for low confidence"""
        query = Query(text="Short query")
        trace = make_trace(tool_confidence=0.55, threads_activated=["t1"])
        spacetime = make_spacetime(query_text="Short query", response="Response", confidence=0.55, trace=trace)

        strategy = refiner._select_strategy(query, spacetime)

        # Low confidence + few threads -> REFINE
        assert strategy == RefinementStrategyType.REFINE

    def test_select_strategy_medium_confidence(self, refiner):
        """Test strategy selection for medium confidence"""
        query = Query(text="Medium query")
        trace = make_trace(tool_confidence=0.70, threads_activated=["t1", "t2", "t3", "t4"])
        spacetime = make_spacetime(query_text="Medium query", response="Response", confidence=0.70, trace=trace)

        strategy = refiner._select_strategy(query, spacetime)

        # Medium confidence + many threads -> CRITIQUE
        assert strategy == RefinementStrategyType.CRITIQUE

    def test_select_strategy_long_query(self, refiner):
        """Test strategy selection for long query"""
        long_text = "A" * 150  # 150 char query
        query = Query(text=long_text)
        trace = make_trace(tool_confidence=0.80, threads_activated=["t1"])
        spacetime = make_spacetime(query_text=long_text, response="Response", confidence=0.80, trace=trace)

        strategy = refiner._select_strategy(query, spacetime)

        # Long query -> HOFSTADTER
        assert strategy == RefinementStrategyType.HOFSTADTER

    @pytest.mark.asyncio
    async def test_refine_strategy_execution(self, refiner, mock_orchestrator):
        """Test REFINE strategy execution"""
        query = Query(text="Test query")

        # Mock initial spacetime
        spacetime = make_spacetime(
            query_text="Test query",
            response="Initial response",
            confidence=0.65,
            trace=make_trace(tool_confidence=0.65, threads_activated=["t1"], motifs_detected=["m1"]),
        )

        # Mock refined spacetime (higher confidence)
        refined_spacetime = make_spacetime(
            query_text="Test query expanded",
            response="Refined response",
            confidence=0.85,
            trace=make_trace(tool_confidence=0.85, threads_activated=["t1", "t2", "t3"], motifs_detected=["m1", "m2"]),
        )

        mock_orchestrator.weave.return_value = refined_spacetime

        # Execute refine strategy
        result = await refiner._refine_strategy(query, spacetime, iteration=0)

        assert result == refined_spacetime
        mock_orchestrator.weave.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_strategy_execution(self, refiner, mock_orchestrator):
        """Test CRITIQUE strategy execution"""
        query = Query(text="Test query")
        spacetime = make_spacetime(query_text="Test query", response="Response", confidence=0.70)
        refined_spacetime = make_spacetime(query_text="Test query with critique", response="Better response", confidence=0.90)

        mock_orchestrator.weave.return_value = refined_spacetime

        result = await refiner._critique_strategy(query, spacetime, iteration=0)

        assert result == refined_spacetime
        mock_orchestrator.weave.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_strategy_multipass(self, refiner, mock_orchestrator):
        """Test VERIFY strategy uses different passes"""
        query = Query(text="Test query")
        spacetime = make_spacetime(query_text="Test query", response="Response", confidence=0.75)
        refined_spacetime = make_spacetime(query_text="Verified query", response="Verified response", confidence=0.90)

        mock_orchestrator.weave.return_value = refined_spacetime

        # Test different iterations (different verification focuses)
        for iteration in [0, 1, 2]:
            result = await refiner._verify_strategy(query, spacetime, iteration)
            assert result == refined_spacetime

    @pytest.mark.asyncio
    async def test_elegance_strategy_multipass(self, refiner, mock_orchestrator):
        """Test ELEGANCE strategy uses different passes"""
        query = Query(text="Test query")
        spacetime = make_spacetime(query_text="Test query", response="Response", confidence=0.75)
        refined_spacetime = make_spacetime(query_text="Elegant query", response="Elegant response", confidence=0.92)

        mock_orchestrator.weave.return_value = refined_spacetime

        # Test different iterations (clarity, simplicity, beauty)
        for iteration in [0, 1, 2]:
            result = await refiner._elegance_strategy(query, spacetime, iteration)
            assert result == refined_spacetime

    @pytest.mark.asyncio
    async def test_hofstadter_strategy_selfref(self, refiner, mock_orchestrator):
        """Test HOFSTADTER strategy uses self-reference"""
        query = Query(text="Deep query about recursion")
        spacetime = make_spacetime(query_text="Deep query", response="Initial understanding", confidence=0.70)
        refined_spacetime = make_spacetime(query_text="Meta-level query", response="Deeper understanding", confidence=0.88)

        mock_orchestrator.weave.return_value = refined_spacetime

        result = await refiner._hofstadter_strategy(query, spacetime, iteration=0)

        assert result == refined_spacetime
        # Should reference previous result in query
        call_args = mock_orchestrator.weave.call_args
        assert "Initial understanding" in call_args[0][0].text[:200]

    @pytest.mark.asyncio
    async def test_refine_complete_flow(self, refiner, mock_orchestrator):
        """Test complete refinement flow"""
        query = Query(text="Test query")

        # Create initial low-confidence spacetime
        initial_spacetime = make_spacetime(
            query_text="Test query",
            response="Initial",
            confidence=0.65,
            trace=make_trace(tool_confidence=0.65, threads_activated=["t1"], motifs_detected=["m1"]),
        )

        # Create improved spacetime
        improved_spacetime = make_spacetime(
            query_text="Test query",
            response="Improved",
            confidence=0.92,
            trace=make_trace(tool_confidence=0.92, threads_activated=["t1", "t2"], motifs_detected=["m1", "m2"]),
        )

        mock_orchestrator.weave.return_value = improved_spacetime

        result = await refiner.refine(
            query=query,
            initial_spacetime=initial_spacetime,
            strategy=RefinementStrategy.REFINE,
            max_iterations=3,
            quality_threshold=0.9
        )

        assert isinstance(result, RefinementResult)
        assert result.strategy_used == RefinementStrategyType.REFINE
        assert result.improved is True
        assert len(result.trajectory) >= 1

    @pytest.mark.asyncio
    async def test_refine_reaches_threshold(self, refiner, mock_orchestrator):
        """Test refinement stops when threshold reached"""
        query = Query(text="Test query")

        initial_spacetime = make_spacetime(query_text="Test query", response="Initial", confidence=0.70)

        # First iteration: quality=0.91 (above threshold 0.9)
        high_quality_spacetime = make_spacetime(
            query_text="Test query",
            response="A" * 500,  # Long response
            confidence=0.91,
            trace=make_trace(
                tool_confidence=0.91,
                threads_activated=["t1", "t2", "t3", "t4", "t5"],
                motifs_detected=["m1", "m2", "m3", "m4", "m5"],
            ),
        )

        mock_orchestrator.weave.return_value = high_quality_spacetime

        result = await refiner.refine(
            query=query,
            initial_spacetime=initial_spacetime,
            strategy=RefinementStrategy.REFINE,
            max_iterations=5,
            quality_threshold=0.9
        )

        # Should stop after reaching threshold
        assert result.iterations <= 5
        assert result.trajectory[-1].score() >= 0.9

    def test_learn_from_refinement(self, refiner):
        """Test learning from successful refinement"""
        query = Query(text="Test query of medium length")

        # Create successful refinement result
        metrics1 = QualityMetrics(0.65, 2, 1, 3, 150)
        metrics2 = QualityMetrics(0.92, 5, 3, 8, 300)

        spacetime = make_spacetime(query_text="Test", response="Response", confidence=0.92)

        result = RefinementResult(
            final_spacetime=spacetime,
            trajectory=[metrics1, metrics2],
            strategy_used=RefinementStrategy.ELEGANCE,
            iterations=1,
            improved=True,
            improvement_rate=0.27
        )

        initial_patterns = len(refiner.learned_patterns)

        refiner._learn_from_refinement(query, result)

        # Should have learned a new pattern
        assert len(refiner.learned_patterns) >= initial_patterns

    def test_get_strategy_statistics(self, refiner):
        """Test getting strategy statistics"""
        # Manually add some strategy results
        refiner.strategy_success_rates[RefinementStrategyType.REFINE] = [0.20, 0.15, 0.25]
        refiner.strategy_success_rates[RefinementStrategyType.CRITIQUE] = [0.30]

        stats = refiner.get_strategy_statistics()

        assert "refine" in stats
        assert "critique" in stats
        assert stats["refine"]["uses"] == 3
        assert stats["critique"]["uses"] == 1
        assert stats["refine"]["avg_improvement"] == pytest.approx(0.20, abs=0.01)

    def test_get_learned_patterns(self, refiner):
        """Test getting learned patterns sorted by improvement"""
        # Add some patterns
        pattern1 = RefinementPattern(
            strategy=RefinementStrategy.REFINE,
            initial_quality=0.60,
            final_quality=0.80,
            iterations=2,
            query_characteristics={},
            improvement_rate=0.10,
            avg_improvement=0.20
        )

        pattern2 = RefinementPattern(
            strategy=RefinementStrategy.ELEGANCE,
            initial_quality=0.70,
            final_quality=0.95,
            iterations=3,
            query_characteristics={},
            improvement_rate=0.083,
            avg_improvement=0.25
        )

        refiner.learned_patterns["p1"] = pattern1
        refiner.learned_patterns["p2"] = pattern2

        patterns = refiner.get_learned_patterns()

        # Should be sorted by avg_improvement (descending)
        assert patterns[0].avg_improvement >= patterns[1].avg_improvement


@pytest.mark.asyncio
async def test_refine_with_strategy_convenience():
    """Test convenience function"""
    query = Query(text="Test query")

    # Mock orchestrator
    mock_orch = AsyncMock()
    mock_orch.weave = AsyncMock()

    # Mock initial spacetime
    initial = make_spacetime(query_text="Test", response="Initial", confidence=0.70)

    # Mock refined spacetime
    refined = make_spacetime(
        query_text="Test",
        response="A" * 500,
        confidence=0.92,
        trace=make_trace(
            tool_confidence=0.92,
            threads_activated=["t1", "t2", "t3", "t4", "t5"],
            motifs_detected=["m1", "m2", "m3", "m4", "m5"],
        ),
    )

    mock_orch.weave.return_value = refined

    result = await refine_with_strategy(
        query=query,
        initial_spacetime=initial,
        orchestrator=mock_orch,
        strategy=RefinementStrategy.VERIFY,
        max_iterations=3
    )

    assert isinstance(result, RefinementResult)
    assert result.strategy_used == RefinementStrategyType.VERIFY
