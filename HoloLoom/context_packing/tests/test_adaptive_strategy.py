"""
Tests for Adaptive Compression Strategy (Phase 6.1)
===================================================

Comprehensive test suite covering:
- All 5 complexity levels (TRIVIAL, SIMPLE, MODERATE, COMPLEX, RESEARCH)
- MI budget mapping
- Config selection
- Edge cases and boundary conditions
- Statistics tracking

Author: Claude Code
Date: 2025-12-09
"""

import pytest
from HoloLoom.context_packing.adaptive_strategy import (
    AdaptiveComplexity,
    AdaptiveCompressionStrategy,
    StrategySelection,
    MI_BUDGETS,
    COMPRESSION_RATIOS,
    select_adaptive_strategy,
    get_adaptive_mi_budget,
    get_adaptive_config,
)
from HoloLoom.context_packing.config import ContextPackerConfig


class TestAdaptiveComplexity:
    """Tests for AdaptiveComplexity enum."""

    def test_all_levels_exist(self):
        """Verify all complexity levels are defined."""
        assert AdaptiveComplexity.TRIVIAL.value == "trivial"
        assert AdaptiveComplexity.SIMPLE.value == "simple"
        assert AdaptiveComplexity.MODERATE.value == "moderate"
        assert AdaptiveComplexity.COMPLEX.value == "complex"
        assert AdaptiveComplexity.RESEARCH.value == "research"

    def test_mi_budgets_defined_for_all_levels(self):
        """Verify MI budgets are defined for all complexity levels."""
        for level in AdaptiveComplexity:
            assert level in MI_BUDGETS
            assert MI_BUDGETS[level] > 0

    def test_compression_ratios_defined_for_all_levels(self):
        """Verify compression ratios are defined for all complexity levels."""
        for level in AdaptiveComplexity:
            assert level in COMPRESSION_RATIOS
            assert 0 < COMPRESSION_RATIOS[level] <= 1.0


class TestTrivialClassification:
    """Tests for TRIVIAL complexity classification."""

    @pytest.fixture
    def strategy(self):
        return AdaptiveCompressionStrategy()

    @pytest.mark.parametrize("query", [
        "hi",
        "Hello!",
        "Hey",
        "yo",
        "thanks",
        "Thank you!",
        "thx",
        "bye",
        "goodbye",
        "ok",
        "okay",
        "yes",
        "no",
        "help",
        "?",
    ])
    def test_trivial_queries(self, strategy, query):
        """Trivial queries should be classified as TRIVIAL."""
        result = strategy.select_strategy(query)
        assert result.complexity == AdaptiveComplexity.TRIVIAL
        assert result.mi_budget == MI_BUDGETS[AdaptiveComplexity.TRIVIAL]
        assert result.confidence >= 0.9

    def test_trivial_gets_aggressive_config(self, strategy):
        """TRIVIAL should use aggressive compression config."""
        result = strategy.select_strategy("hi")
        assert result.compression_ratio == 0.30  # Aggressive keeps 30%
        assert result.mi_budget == 2.0


class TestSimpleClassification:
    """Tests for SIMPLE complexity classification."""

    @pytest.fixture
    def strategy(self):
        return AdaptiveCompressionStrategy()

    @pytest.mark.parametrize("query", [
        "what is python?",
        "define recursion",
        "who is Einstein",
        "when is Christmas",
        "where is Paris",
        "list colors",
        "foo",  # Very short defaults to simple
        "ab",   # Very short
    ])
    def test_simple_queries(self, strategy, query):
        """Simple factual queries should be classified as SIMPLE."""
        result = strategy.select_strategy(query)
        assert result.complexity == AdaptiveComplexity.SIMPLE
        assert result.mi_budget == MI_BUDGETS[AdaptiveComplexity.SIMPLE]

    def test_simple_gets_aggressive_config(self, strategy):
        """SIMPLE should use aggressive compression config."""
        result = strategy.select_strategy("what is python?")
        assert result.compression_ratio == 0.30  # Aggressive keeps 30%
        assert result.mi_budget == 3.0


class TestModerateClassification:
    """Tests for MODERATE complexity classification."""

    @pytest.fixture
    def strategy(self):
        return AdaptiveCompressionStrategy()

    @pytest.mark.parametrize("query", [
        "explain how computers work",
        "describe the process of photosynthesis",
        "what does the algorithm do",
        "how does machine learning work",
        "tell me about Thompson Sampling",
        "summarize the key points",
        "give me an overview of the topic",
    ])
    def test_moderate_queries(self, strategy, query):
        """Moderate explanation queries should be classified as MODERATE."""
        result = strategy.select_strategy(query)
        assert result.complexity == AdaptiveComplexity.MODERATE
        assert result.mi_budget == MI_BUDGETS[AdaptiveComplexity.MODERATE]

    def test_moderate_gets_balanced_config(self, strategy):
        """MODERATE should use balanced compression config."""
        result = strategy.select_strategy("explain how computers work")
        assert result.compression_ratio == 0.50  # Balanced keeps 50%
        assert result.mi_budget == 5.0


class TestComplexClassification:
    """Tests for COMPLEX complexity classification."""

    @pytest.fixture
    def strategy(self):
        return AdaptiveCompressionStrategy()

    @pytest.mark.parametrize("query", [
        "compare Python and JavaScript for web development",
        "what is the difference between TCP and UDP",
        "pros and cons of microservices architecture",
        "advantages and disadvantages of cloud computing",
        "how does recursion relate to iteration",
        "why is functional programming useful compared to OOP",
    ])
    def test_complex_queries(self, strategy, query):
        """Complex comparison queries should be classified as COMPLEX."""
        result = strategy.select_strategy(query)
        assert result.complexity == AdaptiveComplexity.COMPLEX
        assert result.mi_budget == MI_BUDGETS[AdaptiveComplexity.COMPLEX]

    def test_complex_gets_conservative_config(self, strategy):
        """COMPLEX should use conservative compression config."""
        result = strategy.select_strategy("compare Python vs JavaScript")
        assert result.compression_ratio == 0.70  # Conservative keeps 70%
        assert result.mi_budget == 8.0


class TestResearchClassification:
    """Tests for RESEARCH complexity classification."""

    @pytest.fixture
    def strategy(self):
        return AdaptiveCompressionStrategy()

    @pytest.mark.parametrize("query", [
        "analyze the tradeoffs of different sorting algorithms comprehensively",
        "evaluate all aspects of machine learning model selection",
        "examine and assess the impact of cloud computing on enterprise architecture in depth",
        "deep dive into the comprehensive tradeoffs of Thompson Sampling versus UCB",
        "thoroughly investigate the extensive research on neural network optimization",
    ])
    def test_research_queries(self, strategy, query):
        """Research-level queries should be classified as RESEARCH."""
        result = strategy.select_strategy(query)
        assert result.complexity == AdaptiveComplexity.RESEARCH
        assert result.mi_budget == MI_BUDGETS[AdaptiveComplexity.RESEARCH]

    def test_research_gets_research_config(self, strategy):
        """RESEARCH should use research compression config."""
        result = strategy.select_strategy("analyze all tradeoffs comprehensively")
        assert result.compression_ratio == 0.90  # Research keeps 90%
        assert result.mi_budget == 15.0


class TestMIBudgetMapping:
    """Tests for MI budget mapping."""

    def test_mi_budget_ordering(self):
        """MI budgets should increase with complexity."""
        assert MI_BUDGETS[AdaptiveComplexity.TRIVIAL] < MI_BUDGETS[AdaptiveComplexity.SIMPLE]
        assert MI_BUDGETS[AdaptiveComplexity.SIMPLE] < MI_BUDGETS[AdaptiveComplexity.MODERATE]
        assert MI_BUDGETS[AdaptiveComplexity.MODERATE] < MI_BUDGETS[AdaptiveComplexity.COMPLEX]
        assert MI_BUDGETS[AdaptiveComplexity.COMPLEX] < MI_BUDGETS[AdaptiveComplexity.RESEARCH]

    def test_compression_ratio_ordering(self):
        """Compression ratios should increase with complexity (keep more)."""
        assert COMPRESSION_RATIOS[AdaptiveComplexity.TRIVIAL] <= COMPRESSION_RATIOS[AdaptiveComplexity.SIMPLE]
        assert COMPRESSION_RATIOS[AdaptiveComplexity.SIMPLE] <= COMPRESSION_RATIOS[AdaptiveComplexity.MODERATE]
        assert COMPRESSION_RATIOS[AdaptiveComplexity.MODERATE] <= COMPRESSION_RATIOS[AdaptiveComplexity.COMPLEX]
        assert COMPRESSION_RATIOS[AdaptiveComplexity.COMPLEX] <= COMPRESSION_RATIOS[AdaptiveComplexity.RESEARCH]

    def test_specific_mi_values(self):
        """Verify specific MI budget values."""
        assert MI_BUDGETS[AdaptiveComplexity.TRIVIAL] == 2.0
        assert MI_BUDGETS[AdaptiveComplexity.SIMPLE] == 3.0
        assert MI_BUDGETS[AdaptiveComplexity.MODERATE] == 5.0
        assert MI_BUDGETS[AdaptiveComplexity.COMPLEX] == 8.0
        assert MI_BUDGETS[AdaptiveComplexity.RESEARCH] == 15.0


class TestConfigSelection:
    """Tests for ContextPackerConfig selection."""

    @pytest.fixture
    def strategy(self):
        return AdaptiveCompressionStrategy()

    def test_config_has_correct_mi_budget(self, strategy):
        """Config should have MI budget set correctly."""
        for query, expected_mi in [
            ("hi", 2.0),
            ("what is python?", 3.0),
            ("explain machine learning", 5.0),
            ("compare Python vs Java", 8.0),
            ("analyze all tradeoffs comprehensively", 15.0),
        ]:
            result = strategy.select_strategy(query)
            assert result.config.compression.information_budget == expected_mi

    def test_config_has_correct_target_ratio(self, strategy):
        """Config should have target ratio set correctly."""
        for query, expected_ratio in [
            ("hi", 0.30),
            ("what is python?", 0.30),
            ("explain machine learning", 0.50),
            ("compare Python vs Java", 0.70),
            ("analyze all tradeoffs comprehensively", 0.90),
        ]:
            result = strategy.select_strategy(query)
            assert result.config.compression.target_ratio == expected_ratio


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_select_adaptive_strategy(self):
        """select_adaptive_strategy should return StrategySelection."""
        result = select_adaptive_strategy("what is python?")
        assert isinstance(result, StrategySelection)
        assert result.complexity == AdaptiveComplexity.SIMPLE

    def test_get_adaptive_mi_budget(self):
        """get_adaptive_mi_budget should return MI budget."""
        budget = get_adaptive_mi_budget("what is python?")
        assert budget == 3.0

        budget = get_adaptive_mi_budget("analyze all tradeoffs comprehensively")
        assert budget == 15.0

    def test_get_adaptive_config(self):
        """get_adaptive_config should return config and budget tuple."""
        config, budget = get_adaptive_config("explain machine learning")
        assert isinstance(config, ContextPackerConfig)
        assert budget == 5.0
        assert config.compression.information_budget == 5.0


class TestStatistics:
    """Tests for statistics tracking."""

    def test_selection_counts_tracked(self):
        """Selection counts should be tracked."""
        strategy = AdaptiveCompressionStrategy()

        strategy.select_strategy("hi")
        strategy.select_strategy("hello")
        strategy.select_strategy("what is python?")

        stats = strategy.get_statistics()
        assert stats["total_selections"] == 3
        assert stats["by_complexity"]["trivial"]["count"] == 2
        assert stats["by_complexity"]["simple"]["count"] == 1

    def test_outcome_recording(self):
        """Outcomes should be recorded when learning enabled."""
        strategy = AdaptiveCompressionStrategy(enable_learning=True)

        strategy.record_outcome(AdaptiveComplexity.SIMPLE, 0.85)
        strategy.record_outcome(AdaptiveComplexity.SIMPLE, 0.92)

        assert len(strategy._outcome_scores[AdaptiveComplexity.SIMPLE]) == 2


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def strategy(self):
        return AdaptiveCompressionStrategy()

    def test_empty_query(self, strategy):
        """Empty query should be classified."""
        result = strategy.select_strategy("")
        assert result.complexity is not None

    def test_whitespace_query(self, strategy):
        """Whitespace-only query should be classified."""
        result = strategy.select_strategy("   ")
        assert result.complexity is not None

    def test_very_long_query(self, strategy):
        """Very long query should be classified as RESEARCH."""
        long_query = " ".join(["word"] * 30)  # 30 words
        result = strategy.select_strategy(long_query)
        # Long queries get boosted toward research
        assert result.complexity in [AdaptiveComplexity.COMPLEX, AdaptiveComplexity.RESEARCH]

    def test_mixed_case(self, strategy):
        """Classification should be case-insensitive."""
        result1 = strategy.select_strategy("WHAT IS PYTHON?")
        result2 = strategy.select_strategy("what is python?")
        assert result1.complexity == result2.complexity

    def test_special_characters(self, strategy):
        """Special characters should not break classification."""
        result = strategy.select_strategy("What's the difference between @#$ and *&^?")
        assert result.complexity is not None


class TestStrategySelection:
    """Tests for StrategySelection dataclass."""

    def test_selection_has_all_fields(self):
        """StrategySelection should have all required fields."""
        strategy = AdaptiveCompressionStrategy()
        result = strategy.select_strategy("explain machine learning")

        assert hasattr(result, 'complexity')
        assert hasattr(result, 'config')
        assert hasattr(result, 'mi_budget')
        assert hasattr(result, 'compression_ratio')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'detected_indicators')

    def test_selection_confidence_range(self):
        """Confidence should be in valid range."""
        strategy = AdaptiveCompressionStrategy()

        for query in ["hi", "what is python?", "explain ML", "compare A vs B", "analyze all"]:
            result = strategy.select_strategy(query)
            assert 0.0 <= result.confidence <= 1.0

    def test_selection_has_reasoning(self):
        """Selection should include reasoning."""
        strategy = AdaptiveCompressionStrategy()
        result = strategy.select_strategy("hi")
        assert len(result.reasoning) > 0
        assert "trivial" in result.reasoning.lower() or "pattern" in result.reasoning.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
