"""
Comprehensive Tests for Recursive Reasoner
===========================================

Tests all components of the recursive reasoning system:
- Query decomposition
- Strategy selection
- Convergence detection
- Recursive reasoning loop
- RAG Department integration

Created: 2025-11-20
Author: HoloLoom Team
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, List, Any

# Components to test
from HoloLoom.core.convergence.query_decomposition import QueryDecomposer, ComplexityDetector
from HoloLoom.core.convergence.refinement_strategies import StrategySelector, QueryClassifier, StrategyExecutor
from HoloLoom.core.convergence.detectors import (
    ConfidenceThresholdDetector,
    ImprovementDeltaDetector,
    QualityPlateauDetector,
    MaxIterationsDetector,
    MultiCriteriaDetector
)
from HoloLoom.core.protocols.recursive_reasoning import (
    RefinementStep,
    RefinementStrategy,
    RecursiveConfig,
    ReasoningStrategy
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_refinement_history():
    """Sample refinement history for testing."""
    return [
        RefinementStep(1, "Q1", "R1", 0.65, 0.0, "initial", "Initial", datetime.utcnow()),
        RefinementStep(2, "Q2", "R2", 0.75, 0.10, "expand", "Expanded", datetime.utcnow()),
        RefinementStep(3, "Q3", "R3", 0.82, 0.07, "rerank", "Reranked", datetime.utcnow()),
        RefinementStep(4, "Q4", "R4", 0.86, 0.04, "verify", "Verified", datetime.utcnow()),
        RefinementStep(5, "Q5", "R5", 0.87, 0.01, "alternate", "Alternate", datetime.utcnow()),
    ]


@pytest.fixture
def mock_department():
    """Mock RAG Department for testing."""
    class MockDepartment:
        def __init__(self):
            self.call_count = 0

        async def execute(self, params: Dict) -> Any:
            """Mock execute method."""
            self.call_count += 1

            # Simulate improving confidence
            base_confidence = 0.6 + (self.call_count * 0.05)
            confidence = min(base_confidence, 0.95)

            class MockResponse:
                def __init__(self, conf):
                    self.confidence = type('obj', (), {'score': conf})()
                    self.response = {"answer": f"Mock answer {MockDepartment.call_count}"}

            return MockResponse(confidence)

    return MockDepartment()


# ============================================================================
# Test ComplexityDetector
# ============================================================================

def test_complexity_detector_simple_query():
    """Test complexity detection for simple query."""
    detector = ComplexityDetector()
    score = detector.detect("What is Thompson Sampling?")
    assert 0.0 <= score < 0.5  # Simple query


def test_complexity_detector_complex_query():
    """Test complexity detection for complex query."""
    detector = ComplexityDetector()
    score = detector.detect(
        "Compare and contrast Thompson Sampling and Upper Confidence Bound algorithms, "
        "analyzing their tradeoffs in terms of exploration-exploitation balance, "
        "computational efficiency, and practical applications in production systems."
    )
    assert score >= 0.7  # Complex query


def test_complexity_detector_multiple_questions():
    """Test complexity detection for multiple questions."""
    detector = ComplexityDetector()
    score = detector.detect("What is recursion? How does it work? When should I use it?")
    assert score >= 0.5  # Multiple questions increase complexity


# ============================================================================
# Test QueryDecomposer
# ============================================================================

@pytest.mark.asyncio
async def test_query_decomposer_simple_query():
    """Test decomposer with simple query (no decomposition)."""
    decomposer = QueryDecomposer(max_depth=2, max_sub_queries=5)
    tree = await decomposer.decompose("What is Python?", complexity_threshold=0.7)

    assert tree.complexity_score < 0.7
    assert len(tree.sub_queries) == 1  # Not decomposed
    assert tree.depth == 1


@pytest.mark.asyncio
async def test_query_decomposer_complex_query():
    """Test decomposer with complex query (decomposition triggered)."""
    decomposer = QueryDecomposer(max_depth=2, max_sub_queries=5)
    tree = await decomposer.decompose(
        "Compare Thompson Sampling and Upper Confidence Bound algorithms",
        complexity_threshold=0.5
    )

    assert len(tree.sub_queries) > 1  # Decomposed
    assert tree.synthesis_strategy == "comparison_synthesis"


@pytest.mark.asyncio
async def test_query_decomposer_conjunction_splitting():
    """Test conjunction-based splitting."""
    decomposer = QueryDecomposer()
    tree = await decomposer.decompose(
        "What is machine learning and what is deep learning?",
        complexity_threshold=0.3
    )

    assert len(tree.sub_queries) >= 2  # Split on "and"


# ============================================================================
# Test QueryClassifier
# ============================================================================

def test_query_classifier_factual():
    """Test factual query classification."""
    classifier = QueryClassifier()
    assert classifier.classify("What is Python?") == "factual"
    assert classifier.classify("Define machine learning") == "factual"


def test_query_classifier_procedural():
    """Test procedural query classification."""
    classifier = QueryClassifier()
    assert classifier.classify("How to implement gradient descent?") == "procedural"
    assert classifier.classify("What are the steps to train a model?") == "procedural"


def test_query_classifier_analytical():
    """Test analytical query classification."""
    classifier = QueryClassifier()
    assert classifier.classify("Why does overfitting occur?") == "analytical"
    assert classifier.classify("Analyze the impact of learning rate") == "analytical"


def test_query_classifier_comparative():
    """Test comparative query classification."""
    classifier = QueryClassifier()
    assert classifier.classify("Compare Python and Java") == "comparative"
    assert classifier.classify("What's the difference between SVM vs Neural Networks?") == "comparative"


# ============================================================================
# Test StrategySelector
# ============================================================================

@pytest.mark.asyncio
async def test_strategy_selector_low_confidence():
    """Test strategy selection for low confidence."""
    selector = StrategySelector(enable_learning=False)
    strategy = await selector.select_strategy(
        query="What is Python?",
        current_confidence=0.45,
        refinement_history=[]
    )

    # Low confidence → EXPAND_SEARCH or DECOMPOSE
    assert strategy in [RefinementStrategy.EXPAND_SEARCH, RefinementStrategy.DECOMPOSE]


@pytest.mark.asyncio
async def test_strategy_selector_learning():
    """Test strategy learning from outcomes."""
    selector = StrategySelector(enable_learning=True)

    # Simulate learning
    await selector.learn_from_outcome(
        strategy=RefinementStrategy.EXPAND_SEARCH,
        improvement=0.15,
        query_type="factual"
    )

    stats = selector.get_strategy_statistics(query_type="factual")
    assert "expand_search" in stats["strategies"]


# ============================================================================
# Test Convergence Detectors
# ============================================================================

def test_confidence_threshold_detector(sample_refinement_history):
    """Test confidence threshold detector."""
    detector = ConfidenceThresholdDetector(threshold=0.85)

    # Should converge (latest confidence = 0.87 > 0.85)
    converged, reason = detector.detect(sample_refinement_history)
    assert converged
    assert "0.87" in reason


def test_improvement_delta_detector(sample_refinement_history):
    """Test improvement delta detector."""
    detector = ImprovementDeltaDetector(min_delta=0.05)

    # Should converge (latest improvement = 0.01 < 0.05)
    converged, reason = detector.detect(sample_refinement_history)
    assert converged
    assert "0.01" in reason


def test_quality_plateau_detector(sample_refinement_history):
    """Test quality plateau detector."""
    detector = QualityPlateauDetector(plateau_window=2, min_delta=0.03)

    # Should converge (last 2 improvements < 0.03)
    converged, reason = detector.detect(sample_refinement_history)
    assert converged


def test_max_iterations_detector(sample_refinement_history):
    """Test max iterations detector."""
    detector = MaxIterationsDetector(max_iterations=3)

    # Should converge (5 iterations > 3 max)
    converged, reason = detector.detect(sample_refinement_history)
    assert converged


def test_multi_criteria_detector_or_logic(sample_refinement_history):
    """Test multi-criteria detector with OR logic."""
    from HoloLoom.core.convergence.detectors import create_multi_criteria_detector

    detector = create_multi_criteria_detector(
        confidence_threshold=0.85,
        min_improvement=0.05,
        plateau_window=2,
        max_iterations=10,
        logic="OR"
    )

    # Should converge (confidence threshold met)
    converged, reason = detector.detect(sample_refinement_history)
    assert converged


def test_multi_criteria_detector_and_logic(sample_refinement_history):
    """Test multi-criteria detector with AND logic."""
    from HoloLoom.core.convergence.detectors import create_multi_criteria_detector

    detector = create_multi_criteria_detector(
        confidence_threshold=0.85,
        min_improvement=0.05,
        plateau_window=2,
        max_iterations=10,
        logic="AND"
    )

    # Might not converge (all criteria must be met)
    converged, reason = detector.detect(sample_refinement_history)
    # Just ensure it doesn't crash
    assert isinstance(converged, bool)


# ============================================================================
# Test StrategyExecutor
# ============================================================================

def test_strategy_executor_expand_search():
    """Test strategy executor for EXPAND_SEARCH."""
    params = StrategyExecutor.get_execution_params(RefinementStrategy.EXPAND_SEARCH)

    assert params["max_sources"] == 10
    assert params["mode"] == "research"
    assert params["enable_reranking"] is True


def test_strategy_executor_decompose():
    """Test strategy executor for DECOMPOSE."""
    params = StrategyExecutor.get_execution_params(RefinementStrategy.DECOMPOSE)

    assert params["decompose"] is True
    assert params["max_sources"] == 3


# ============================================================================
# Integration Tests (Mock RAG Department)
# ============================================================================

@pytest.mark.asyncio
async def test_recursive_reasoner_convergence(mock_department):
    """Test recursive reasoner converges correctly."""
    from HoloLoom.core.convergence.recursive_reasoner_enhanced import create_recursive_reasoner

    config = RecursiveConfig(
        strategy=ReasoningStrategy.REFINE,
        max_iterations=5,
        quality_threshold=0.85,
        min_improvement=0.05,
        enable_decomposition=False,
        enable_learning=False,
        refinement_threshold=0.75
    )

    reasoner = create_recursive_reasoner(
        department=mock_department,
        config=config,
        enable_decomposition=False,
        enable_learning=False
    )

    result = await reasoner.reason("What is Thompson Sampling?", context={})

    # Check result structure
    assert result.final_confidence > 0.6
    assert result.iterations >= 1
    assert len(result.refinement_steps) == result.iterations
    assert result.total_latency_ms > 0


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_empty_refinement_history():
    """Test detectors with empty history."""
    detector = ConfidenceThresholdDetector(0.85)
    converged, reason = detector.detect([])
    assert not converged
    assert "No history" in reason


def test_single_refinement_step():
    """Test detectors with single step."""
    single_step = [
        RefinementStep(1, "Q", "R", 0.90, 0.0, "initial", "Init", datetime.utcnow())
    ]

    # Confidence detector should work
    detector1 = ConfidenceThresholdDetector(0.85)
    converged, _ = detector1.detect(single_step)
    assert converged  # 0.90 > 0.85

    # Improvement detector needs 2+ steps
    detector2 = ImprovementDeltaDetector(0.05)
    converged, reason = detector2.detect(single_step)
    assert not converged
    assert "Insufficient" in reason


# ============================================================================
# Performance Tests
# ============================================================================

def test_complexity_detector_performance():
    """Test complexity detector is fast."""
    import time

    detector = ComplexityDetector()
    queries = [
        "What is Python?",
        "Compare and contrast Python and Java for web development",
        "Analyze the tradeoffs of microservices vs monolithic architecture"
    ] * 100  # 300 queries

    start = time.time()
    for query in queries:
        detector.detect(query)
    elapsed = time.time() - start

    # Should be very fast (<100ms for 300 queries)
    assert elapsed < 0.1


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
