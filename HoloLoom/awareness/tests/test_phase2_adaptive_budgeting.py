#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Phase 2: Adaptive Budgeting

Tests:
1. QueryComplexityAnalyzer
2. AdaptiveBudgetCalculator
3. MODEL_REGISTRY
4. Integration with LLMContextPacker
"""

import pytest
from dataclasses import dataclass

# Import components to test
from HoloLoom.awareness.query_complexity import (
    QueryComplexityAnalyzer,
    ComplexityLevel,
    ComplexityAnalysis
)

from HoloLoom.awareness.adaptive_budget import (
    AdaptiveBudgetCalculator,
    AdaptiveBudget,
    ModelInfo,
    MODEL_REGISTRY,
    get_model_info
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_awareness_context():
    """Mock awareness context with uncertainty"""
    @dataclass
    class MockConfidence:
        uncertainty_level: float = 0.2
        query_cache_status: str = "miss"
        knowledge_gap_detected: bool = False

    @dataclass
    class MockAwarenessContext:
        confidence: MockConfidence = MockConfidence()

    return MockAwarenessContext()


@pytest.fixture
def high_uncertainty_context():
    """Mock awareness context with high uncertainty"""
    @dataclass
    class MockConfidence:
        uncertainty_level: float = 0.85
        query_cache_status: str = "miss"
        knowledge_gap_detected: bool = True

    @dataclass
    class MockAwarenessContext:
        confidence: MockConfidence = MockConfidence()

    return MockAwarenessContext()


# ============================================================================
# Test QueryComplexityAnalyzer
# ============================================================================

def test_simple_query_complexity():
    """Test simple query classification"""
    analyzer = QueryComplexityAnalyzer()

    # Simple what question
    result = analyzer.analyze("What is Python?")

    assert result.level == ComplexityLevel.SIMPLE
    assert result.score < 0.4
    assert 2000 <= result.recommended_budget <= 4000


def test_moderate_query_complexity():
    """Test moderate query classification"""
    analyzer = QueryComplexityAnalyzer()

    # How question with some detail
    result = analyzer.analyze("How does Thompson Sampling work in practice?")

    assert result.level == ComplexityLevel.MODERATE
    assert 0.3 <= result.score < 0.6  # MODERATE range: 0.3-0.6
    assert 4000 <= result.recommended_budget <= 8000


def test_complex_query_complexity():
    """Test complex query classification"""
    analyzer = QueryComplexityAnalyzer()

    # Compare question with multiple technical terms
    result = analyzer.analyze(
        "Compare Thompson Sampling, UCB1, and epsilon-greedy algorithms "
        "for multi-armed bandit optimization. Analyze their theoretical "
        "guarantees and practical performance trade-offs."
    )

    assert result.level in (ComplexityLevel.COMPLEX, ComplexityLevel.RESEARCH)
    assert result.score >= 0.6  # COMPLEX or RESEARCH
    assert result.recommended_budget >= 8000


def test_research_query_complexity():
    """Test research-level query classification"""
    analyzer = QueryComplexityAnalyzer()

    # Compare question with comprehensive detail request
    result = analyzer.analyze(
        "Compare and contrast Thompson Sampling, UCB1, and epsilon-greedy "
        "strategies for contextual bandits comprehensively. Analyze in depth "
        "their theoretical foundations, implementation details, performance "
        "characteristics, and provide detailed examples with complete explanations."
    )

    assert result.level in (ComplexityLevel.COMPLEX, ComplexityLevel.RESEARCH)
    assert result.score >= 0.6  # High complexity
    assert result.recommended_budget >= 8000


def test_complexity_with_uncertainty():
    """Test complexity analysis with awareness context"""
    analyzer = QueryComplexityAnalyzer()

    # Same query, with/without uncertainty
    query = "How does machine learning work?"

    # Without context
    result_no_ctx = analyzer.analyze(query)

    # With high uncertainty - use simple class instead of dataclass
    class MockConfidence:
        def __init__(self):
            self.uncertainty_level = 0.9

    class MockAwarenessContext:
        def __init__(self):
            self.confidence = MockConfidence()

    result_with_uncertainty = analyzer.analyze(query, MockAwarenessContext())

    # High uncertainty should increase complexity score
    assert result_with_uncertainty.score > result_no_ctx.score


def test_batch_analysis():
    """Test batch analysis of queries"""
    analyzer = QueryComplexityAnalyzer()

    queries = [
        "What is Python?",
        "How does it work?",
        "Why use Python over Java?",
        "Compare Python, Java, and C++ comprehensively in detail with examples."
    ]

    results = analyzer.analyze_batch(queries)

    assert len(results) == 4
    # First should be simple
    assert results[0].level == ComplexityLevel.SIMPLE
    # Last should be more complex (at least moderate)
    assert results[-1].level in (ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX, ComplexityLevel.RESEARCH)
    # Scores should generally increase
    assert results[-1].score > results[0].score


def test_complexity_statistics():
    """Test complexity statistics"""
    analyzer = QueryComplexityAnalyzer()

    queries = [
        "What is X?" for _ in range(10)
    ] + [
        "Compare X and Y comprehensively in detail with examples." for _ in range(5)
    ]

    results = analyzer.analyze_batch(queries)
    stats = analyzer.get_statistics(results)

    assert stats["total_queries"] == 15
    assert "level_distribution" in stats
    assert stats["level_distribution"]["simple"] == 10  # All "What is X?" should be simple
    assert "avg_recommended_budget" in stats
    assert "budget_range" in stats


# ============================================================================
# Test MODEL_REGISTRY
# ============================================================================

def test_model_registry_anthropic():
    """Test Anthropic model info"""
    model = get_model_info("claude-3-5-sonnet-20241022")

    assert model.provider == "anthropic"
    assert model.context_window == 200_000
    assert model.recommended_max == 180_000
    assert model.cost_per_1m_input == 3.0


def test_model_registry_openai():
    """Test OpenAI model info"""
    model = get_model_info("gpt-4-turbo-preview")

    assert model.provider == "openai"
    assert model.context_window == 128_000
    assert model.recommended_max == 120_000


def test_model_registry_ollama():
    """Test Ollama model info (free)"""
    model = get_model_info("llama3.2:3b")

    assert model.provider == "ollama"
    assert model.cost_per_1m_input == 0.0
    assert model.cost_per_1m_output == 0.0


def test_model_registry_fallback():
    """Test fallback for unknown models"""
    # Unknown Claude model - falls back to default Claude
    model = get_model_info("claude-unknown")
    assert model.provider == "anthropic"
    assert model.context_window == 200_000  # Uses default Claude specs

    # Unknown GPT-4 model - falls back to GPT-4 Turbo
    model = get_model_info("gpt-4-unknown")
    assert model.provider == "openai"
    assert model.context_window == 128_000

    # Completely unknown - creates conservative defaults
    model = get_model_info("totally-unknown-model")
    assert model.provider == "unknown"
    assert model.name == "totally-unknown-model"  # Preserves original name
    assert model.context_window == 8_192  # Conservative default


# ============================================================================
# Test AdaptiveBudgetCalculator
# ============================================================================

def test_adaptive_budget_simple_query():
    """Test adaptive budget for simple query"""
    calc = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022",
        min_budget=2_000,
        max_budget=32_000
    )

    budget = calc.calculate_budget("What is Python?")

    assert budget.query_complexity == ComplexityLevel.SIMPLE
    assert 2_000 <= budget.total <= 10_000  # Base × adjustments
    assert len(budget.reasoning) > 0


def test_adaptive_budget_complex_query():
    """Test adaptive budget for complex query"""
    calc = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022",
        min_budget=2_000,
        max_budget=32_000
    )

    budget = calc.calculate_budget(
        "Compare Thompson Sampling, UCB1, and epsilon-greedy comprehensively. "
        "Analyze their theoretical foundations, implementation complexity, and "
        "practical performance trade-offs in detail with complete examples."
    )

    # Should be at least MODERATE, possibly COMPLEX/RESEARCH
    assert budget.query_complexity in (ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX, ComplexityLevel.RESEARCH)
    # With large model multiplier (1.5x), should get decent budget
    assert budget.total >= 6_000  # Moderate+ with 1.5x multiplier


def test_adaptive_budget_model_capacity_adjustment():
    """Test model capacity adjustment"""
    # Large model (Claude)
    calc_large = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022"
    )

    # Small model (GPT-3.5)
    calc_small = AdaptiveBudgetCalculator(
        model_name="gpt-3.5-turbo"
    )

    query = "What is machine learning?"

    budget_large = calc_large.calculate_budget(query)
    budget_small = calc_small.calculate_budget(query)

    # Large model should get 1.5x multiplier
    assert budget_large.total > budget_small.total


def test_adaptive_budget_uncertainty_adjustment():
    """Test uncertainty adjustment"""
    calc = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022"
    )

    query = "What is quantum computing?"

    # Low uncertainty - use simple classes
    class LowConfidence:
        def __init__(self):
            self.uncertainty_level = 0.2

    class LowUncertaintyCtx:
        def __init__(self):
            self.confidence = LowConfidence()

    budget_low = calc.calculate_budget(query, LowUncertaintyCtx())

    # High uncertainty
    class HighConfidence:
        def __init__(self):
            self.uncertainty_level = 0.85

    class HighUncertaintyCtx:
        def __init__(self):
            self.confidence = HighConfidence()

    budget_high = calc.calculate_budget(query, HighUncertaintyCtx())

    # High uncertainty should increase budget by 1.3x
    assert budget_high.total > budget_low.total


def test_adaptive_budget_memory_adjustment():
    """Test memory count adjustment"""
    calc = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022"
    )

    query = "What is Python?"

    # Few memories
    budget_few = calc.calculate_budget(query, available_memories=5)

    # Many memories
    budget_many = calc.calculate_budget(query, available_memories=30)

    # Many memories should increase budget by 1.2x
    assert budget_many.total > budget_few.total


def test_adaptive_budget_clamping():
    """Test budget clamping to min/max"""
    calc = AdaptiveBudgetCalculator(
        model_name="gpt-3.5-turbo",  # Small model, no 1.5x multiplier
        min_budget=2_000,
        max_budget=3_000
    )

    # Simple query (base ~2-3k)
    budget_simple = calc.calculate_budget("Hi")
    assert budget_simple.total >= 2_000  # Should be at/above min
    assert budget_simple.total <= 3_000  # Should not exceed max

    # Complex query with high uncertainty (would normally be > 3k with multipliers)
    class HighConfidence:
        def __init__(self):
            self.uncertainty_level = 0.95

    class HighUncertaintyCtx:
        def __init__(self):
            self.confidence = HighConfidence()

    budget_high = calc.calculate_budget(
        "Compare and analyze comprehensively in extreme detail with complete examples",
        awareness_context=HighUncertaintyCtx(),
        available_memories=50
    )
    assert budget_high.total == 3_000  # Clamped to max


def test_adaptive_budget_learning():
    """Test learning from outcomes"""
    calc = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022",
        enable_learning=True
    )

    # Create budgets
    budget_3k = AdaptiveBudget(
        total=3_000,
        reserved_for_query=300,
        reserved_for_response=500,
        model_max=200_000,
        recommended_max=180_000,
        reasoning=[],
        adjustments={},
        query_complexity=ComplexityLevel.SIMPLE,
        complexity_score=0.3
    )

    budget_10k = AdaptiveBudget(
        total=10_000,
        reserved_for_query=500,
        reserved_for_response=1000,
        model_max=200_000,
        recommended_max=180_000,
        reasoning=[],
        adjustments={},
        query_complexity=ComplexityLevel.MODERATE,
        complexity_score=0.5
    )

    # Feed outcomes: 3k → low quality, 10k → high quality
    for _ in range(10):
        calc.learn_from_outcome(budget_3k, quality_score=0.5)
        calc.learn_from_outcome(budget_10k, quality_score=0.95)

    stats = calc.get_statistics()
    assert stats["learning_enabled"] is True
    assert stats["budgets_tracked"] >= 2
    assert stats["total_samples"] == 20


def test_adaptive_budget_optimal():
    """Test optimal budget selection from learning"""
    calc = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022",
        enable_learning=True
    )

    # Not enough data initially
    optimal = calc.get_optimal_budget(ComplexityLevel.SIMPLE)
    assert optimal is None

    # Feed enough samples
    budget = AdaptiveBudget(
        total=5_000,
        reserved_for_query=300,
        reserved_for_response=500,
        model_max=200_000,
        recommended_max=180_000,
        reasoning=[],
        adjustments={},
        query_complexity=ComplexityLevel.SIMPLE,
        complexity_score=0.3
    )

    for _ in range(10):
        calc.learn_from_outcome(budget, quality_score=0.95)

    # Now should have optimal budget
    optimal = calc.get_optimal_budget(ComplexityLevel.SIMPLE)
    assert optimal is not None
    assert 4_000 <= optimal <= 6_000  # Rounded to nearest 1000


def test_adaptive_budget_to_token_budget():
    """Test conversion to TokenBudget"""
    calc = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022"
    )

    adaptive = calc.calculate_budget("What is Python?")
    token_budget = adaptive.to_token_budget()

    assert token_budget.total == adaptive.total
    assert token_budget.reserved_for_query == adaptive.reserved_for_query
    assert token_budget.reserved_for_response == adaptive.reserved_for_response
    assert token_budget.available_for_context == adaptive.available_for_context


def test_adaptive_budget_reasoning():
    """Test budget reasoning generation"""
    calc = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022"
    )

    budget = calc.calculate_budget(
        "What is quantum computing?",
        available_memories=25
    )

    # Should have reasoning for:
    # - Base budget from complexity
    # - Large context window adjustment
    # - Memory count adjustment
    assert len(budget.reasoning) >= 3
    assert any("Base budget" in r for r in budget.reasoning)
    assert any("context window" in r for r in budget.reasoning)
    assert any("memories" in r for r in budget.reasoning)


# ============================================================================
# Integration Tests (would require LLMContextPacker)
# ============================================================================

def test_integration_placeholder():
    """
    Placeholder for integration tests with LLMContextPacker.

    These would test:
    - Adaptive budgeting enabled vs disabled
    - Budget override behavior
    - Learning feedback loop
    - Cost savings measurement

    Requires full LLMContextPacker setup with mocked LLM.
    """
    # TODO: Add integration tests once LLMContextPacker is fully integrated
    pass


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
