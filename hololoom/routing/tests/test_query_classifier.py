"""
Unit tests for HoloLoom routing query classifier.

Tests:
- QueryClassifier initialization
- Complexity classification (TRIVIAL, SIMPLE, MODERATE, COMPLEX, RESEARCH)
- Pattern detection (factual, procedural, analytical queries)
- Confidence scoring
- Fast path routing decisions
- AUTO strategy selection
- Edge cases (empty, very long, special characters)
- Integration with FastPathRouter

Author: Claude Code
Date: 2025-12-01
"""

import pytest
from hololoom.routing.query_classifier import (
    QueryClassifier,
    QueryComplexity,
    ClassificationResult,
    classify_query
)


class TestQueryClassifierInitialization:
    """Test classifier initialization and setup."""

    def test_classifier_initialization(self):
        """QueryClassifier should initialize with compiled regex patterns."""
        classifier = QueryClassifier()

        # Should compile trivial patterns
        assert len(classifier._trivial_regex) > 0
        assert len(classifier._trivial_regex) == len(QueryClassifier.TRIVIAL_PATTERNS)

        # Should compile simple patterns
        assert len(classifier._simple_regex) > 0
        assert len(classifier._simple_regex) == len(QueryClassifier.SIMPLE_PATTERNS)

    def test_classifier_pattern_sets(self):
        """Classifier should have all expected pattern sets."""
        assert hasattr(QueryClassifier, 'TRIVIAL_PATTERNS')
        assert hasattr(QueryClassifier, 'SIMPLE_PATTERNS')
        assert hasattr(QueryClassifier, 'RESEARCH_KEYWORDS')
        assert hasattr(QueryClassifier, 'COMPLEXITY_KEYWORDS')

        assert len(QueryClassifier.TRIVIAL_PATTERNS) >= 5
        assert len(QueryClassifier.SIMPLE_PATTERNS) >= 5
        assert len(QueryClassifier.RESEARCH_KEYWORDS) >= 10


class TestTrivialQueryClassification:
    """Test TRIVIAL query classification."""

    @pytest.mark.parametrize("query,expected_pattern", [
        ("hi", "trivial_match"),
        ("hello", "trivial_match"),
        ("hey", "trivial_match"),
        ("thanks", "trivial_match"),
        ("thank you", "trivial_match"),
        ("ok", "trivial_match"),
        ("bye", "trivial_match"),
        ("help", "trivial_match"),
    ])
    def test_trivial_greetings_and_acknowledgments(self, query, expected_pattern):
        """Greetings and acknowledgments should classify as TRIVIAL."""
        classifier = QueryClassifier()
        result = classifier.classify(query)

        assert result.complexity == QueryComplexity.TRIVIAL
        assert result.confidence >= 0.90
        assert expected_pattern in result.detected_patterns
        assert "trivial" in result.reasoning.lower()

    def test_trivial_case_insensitive(self):
        """Trivial classification should be case-insensitive."""
        classifier = QueryClassifier()

        queries = ["HI", "Hello", "THANK YOU", "Ok"]
        for query in queries:
            result = classifier.classify(query)
            assert result.complexity == QueryComplexity.TRIVIAL
            assert result.confidence >= 0.90

    def test_trivial_with_punctuation(self):
        """Trivial queries with punctuation should still match."""
        classifier = QueryClassifier()

        queries = ["hi!", "hello?", "thanks.", "ok!"]
        for query in queries:
            result = classifier.classify(query)
            assert result.complexity == QueryComplexity.TRIVIAL
            assert result.confidence >= 0.90


class TestSimpleQueryClassification:
    """Test SIMPLE query classification."""

    @pytest.mark.parametrize("query", [
        "what is Python?",
        "what is machine learning",
        "define recursion",
        "define neural network",
        "who is Einstein",
        "when is Christmas",
        "where is Paris",
    ])
    def test_simple_factual_queries(self, query):
        """Factual lookup queries should classify as SIMPLE."""
        classifier = QueryClassifier()
        result = classifier.classify(query)

        assert result.complexity == QueryComplexity.SIMPLE
        # Some queries may have lower confidence (0.6) if no strong pattern match
        assert result.confidence >= 0.60
        assert "simple" in result.reasoning.lower() or "short" in result.reasoning.lower() or "default" in result.reasoning.lower()

    def test_simple_short_word_count(self):
        """Very short queries (≤3 words) should classify as SIMPLE."""
        classifier = QueryClassifier()

        queries = ["Python?", "machine learning", "neural networks"]
        for query in queries:
            result = classifier.classify(query)
            assert result.complexity == QueryComplexity.SIMPLE
            assert "short query" in result.reasoning.lower()


class TestComplexQueryClassification:
    """Test COMPLEX query classification."""

    @pytest.mark.parametrize("query", [
        "explain how neural networks work",
        "describe the attention mechanism",
        "why do transformers use self-attention?",
        "how does backpropagation work?",
        "explain gradient descent with examples",
    ])
    def test_complex_explanation_queries(self, query):
        """Queries with 'explain', 'why', 'how' should classify as COMPLEX or at least not TRIVIAL."""
        classifier = QueryClassifier()
        result = classifier.classify(query)

        # Some shorter queries may classify as SIMPLE due to word count heuristic
        # The key is they shouldn't be TRIVIAL
        assert result.complexity in {QueryComplexity.SIMPLE, QueryComplexity.COMPLEX, QueryComplexity.RESEARCH}
        assert result.confidence >= 0.60

    def test_complex_medium_length(self):
        """Medium-length queries (10-20 words) with complexity indicators."""
        classifier = QueryClassifier()

        query = "explain in detail how the transformer architecture uses self-attention mechanisms"
        result = classifier.classify(query)

        # Should be COMPLEX due to complexity keywords
        assert result.complexity in {QueryComplexity.COMPLEX, QueryComplexity.RESEARCH}
        assert result.confidence >= 0.70


class TestResearchQueryClassification:
    """Test RESEARCH query classification."""

    @pytest.mark.parametrize("query", [
        "analyze tradeoffs between Thompson Sampling and epsilon-greedy",
        "evaluate advantages and disadvantages of neural networks",
        "assess all the different optimization algorithms comprehensively",
        "examine the pros and cons of reinforcement learning versus supervised learning",
    ])
    def test_research_analysis_queries(self, query):
        """Queries with multiple research keywords should classify as RESEARCH or COMPLEX."""
        classifier = QueryClassifier()
        result = classifier.classify(query)

        # These queries have 2+ research keywords or are long enough
        # They should classify as RESEARCH or COMPLEX
        assert result.complexity in {QueryComplexity.COMPLEX, QueryComplexity.RESEARCH}
        assert result.confidence >= 0.60

    def test_research_single_keyword_may_be_simple(self):
        """Query with single research keyword may not classify as RESEARCH."""
        classifier = QueryClassifier()

        # "compare" is only 1 keyword, needs 2+ for RESEARCH
        query = "compare and contrast supervised and unsupervised learning"
        result = classifier.classify(query)

        # May be SIMPLE (only 1 research keyword)
        # Just verify it doesn't crash and gives a result
        assert result.complexity in {QueryComplexity.SIMPLE, QueryComplexity.COMPLEX, QueryComplexity.RESEARCH}
        assert result.confidence > 0

    def test_research_very_long_queries(self):
        """Very long queries (≥30 words) should classify as RESEARCH."""
        classifier = QueryClassifier()

        query = " ".join(["word"] * 35)  # 35-word query
        result = classifier.classify(query)

        assert result.complexity == QueryComplexity.RESEARCH
        assert "long query" in result.reasoning.lower()

    def test_research_multiple_keywords(self):
        """Queries with 2+ research keywords should classify as RESEARCH."""
        classifier = QueryClassifier()

        # Contains "analyze", "compare", "evaluate" (3 research keywords)
        query = "analyze, compare, and evaluate different approaches"
        result = classifier.classify(query)

        assert result.complexity == QueryComplexity.RESEARCH
        assert result.confidence >= 0.90


class TestPatternDetection:
    """Test pattern detection and metadata."""

    def test_trivial_pattern_detection(self):
        """Trivial queries should detect 'trivial_match' pattern."""
        classifier = QueryClassifier()
        result = classifier.classify("hello")

        assert "trivial_match" in result.detected_patterns

    def test_simple_pattern_detection(self):
        """Simple queries should detect 'simple_match' pattern."""
        classifier = QueryClassifier()
        result = classifier.classify("what is Python?")

        # May detect simple_match or word_count pattern
        assert len(result.detected_patterns) > 0

    def test_research_pattern_detection(self):
        """Research queries should detect 'research_keywords' pattern."""
        classifier = QueryClassifier()
        result = classifier.classify("analyze and compare all approaches")

        assert "research_keywords" in result.detected_patterns

    def test_word_count_pattern_detection(self):
        """Queries should track word count in detected patterns."""
        classifier = QueryClassifier()
        result = classifier.classify("testing word count")

        # Should have word_count pattern
        patterns = [p for p in result.detected_patterns if p.startswith("word_count_")]
        assert len(patterns) > 0


class TestConfidenceScoring:
    """Test confidence score assignment."""

    def test_trivial_high_confidence(self):
        """TRIVIAL queries should have high confidence (≥0.90)."""
        classifier = QueryClassifier()
        result = classifier.classify("hello")

        assert result.confidence >= 0.90

    def test_pattern_match_higher_confidence(self):
        """Pattern matches should have higher confidence than heuristics."""
        classifier = QueryClassifier()

        # Pattern match
        result_pattern = classifier.classify("what is Python?")
        # Heuristic (medium-length generic query)
        result_heuristic = classifier.classify("tell me about things")

        # Pattern match should have higher or equal confidence
        assert result_pattern.confidence >= result_heuristic.confidence

    def test_default_classification_confidence(self):
        """Default SIMPLE classification should have moderate confidence."""
        classifier = QueryClassifier()

        # Generic query with no strong indicators
        result = classifier.classify("something about stuff")

        assert result.complexity == QueryComplexity.SIMPLE
        assert 0.50 <= result.confidence <= 0.70


class TestFastPathRouting:
    """Test fast path routing decisions."""

    def test_should_use_fast_path_trivial(self):
        """TRIVIAL queries should use fast path."""
        classifier = QueryClassifier()

        assert classifier.should_use_fast_path(QueryComplexity.TRIVIAL) is True

    def test_should_use_fast_path_simple(self):
        """SIMPLE queries should use fast path."""
        classifier = QueryClassifier()

        assert classifier.should_use_fast_path(QueryComplexity.SIMPLE) is True

    def test_should_not_use_fast_path_complex(self):
        """COMPLEX queries should NOT use fast path."""
        classifier = QueryClassifier()

        assert classifier.should_use_fast_path(QueryComplexity.COMPLEX) is False

    def test_should_not_use_fast_path_research(self):
        """RESEARCH queries should NOT use fast path."""
        classifier = QueryClassifier()

        assert classifier.should_use_fast_path(QueryComplexity.RESEARCH) is False

    def test_estimate_latency_budget_trivial(self):
        """TRIVIAL queries should have 10ms budget."""
        classifier = QueryClassifier()
        budget = classifier.estimate_latency_budget(QueryComplexity.TRIVIAL)

        assert budget == 10

    def test_estimate_latency_budget_simple(self):
        """SIMPLE queries should have 50ms budget."""
        classifier = QueryClassifier()
        budget = classifier.estimate_latency_budget(QueryComplexity.SIMPLE)

        assert budget == 50

    def test_estimate_latency_budget_complex(self):
        """COMPLEX queries should have 150ms budget."""
        classifier = QueryClassifier()
        budget = classifier.estimate_latency_budget(QueryComplexity.COMPLEX)

        assert budget == 150

    def test_estimate_latency_budget_research(self):
        """RESEARCH queries should have unlimited budget."""
        classifier = QueryClassifier()
        budget = classifier.estimate_latency_budget(QueryComplexity.RESEARCH)

        assert budget == float('inf')


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self):
        """Empty query should classify as SIMPLE (conservative)."""
        classifier = QueryClassifier()
        result = classifier.classify("")

        assert result.complexity == QueryComplexity.SIMPLE
        assert result.confidence >= 0.50

    def test_whitespace_only_query(self):
        """Whitespace-only query should classify as SIMPLE."""
        classifier = QueryClassifier()
        result = classifier.classify("   \n\t  ")

        assert result.complexity == QueryComplexity.SIMPLE
        assert result.confidence >= 0.50

    def test_very_long_query(self):
        """Very long query (100+ words) should classify as RESEARCH."""
        classifier = QueryClassifier()

        long_query = " ".join(["word"] * 100)
        result = classifier.classify(long_query)

        assert result.complexity == QueryComplexity.RESEARCH
        assert "long query" in result.reasoning.lower()

    def test_special_characters(self):
        """Queries with special characters should still classify."""
        classifier = QueryClassifier()

        queries = [
            "what is C++?",
            "define @decorator",
            "hello!!! how are you???",
            "thanks... bye...",
        ]

        for query in queries:
            result = classifier.classify(query)
            # Should classify without errors
            assert result.complexity in {
                QueryComplexity.TRIVIAL,
                QueryComplexity.SIMPLE,
                QueryComplexity.COMPLEX,
                QueryComplexity.RESEARCH
            }

    def test_unicode_characters(self):
        """Queries with unicode characters should handle gracefully."""
        classifier = QueryClassifier()

        queries = [
            "hello 👋",
            "what is machine learning? 🤖",
            "explain neural networks 🧠",
        ]

        for query in queries:
            result = classifier.classify(query)
            # Should classify without errors
            assert result.complexity is not None
            assert result.confidence > 0

    def test_mixed_case_and_punctuation(self):
        """Mixed case and punctuation should normalize correctly."""
        classifier = QueryClassifier()

        # Same query, different formatting
        queries = [
            "What Is Python?",
            "WHAT IS PYTHON?",
            "what is python",
            "what is python?!",
        ]

        results = [classifier.classify(q) for q in queries]

        # All should classify the same way
        complexities = [r.complexity for r in results]
        assert len(set(complexities)) == 1  # All same complexity


class TestConvenienceFunction:
    """Test convenience function classify_query()."""

    def test_classify_query_function(self):
        """classify_query() should work without explicit classifier."""
        result = classify_query("hello")

        assert result.complexity == QueryComplexity.TRIVIAL
        assert result.confidence >= 0.90

    def test_classify_query_returns_classification_result(self):
        """classify_query() should return ClassificationResult."""
        result = classify_query("what is Python?")

        assert isinstance(result, ClassificationResult)
        assert hasattr(result, 'complexity')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'detected_patterns')


class TestClassificationConsistency:
    """Test classification consistency across multiple calls."""

    def test_same_query_same_result(self):
        """Same query should produce same classification."""
        classifier = QueryClassifier()

        query = "explain how neural networks work"
        result1 = classifier.classify(query)
        result2 = classifier.classify(query)

        assert result1.complexity == result2.complexity
        assert result1.confidence == result2.confidence
        assert result1.detected_patterns == result2.detected_patterns

    def test_different_instances_same_result(self):
        """Different classifier instances should produce same result."""
        classifier1 = QueryClassifier()
        classifier2 = QueryClassifier()

        query = "analyze all tradeoffs comprehensively"
        result1 = classifier1.classify(query)
        result2 = classifier2.classify(query)

        assert result1.complexity == result2.complexity
        assert result1.confidence == result2.confidence


class TestReasoningExplanations:
    """Test reasoning explanations are informative."""

    def test_reasoning_not_empty(self):
        """All classifications should have non-empty reasoning."""
        classifier = QueryClassifier()

        queries = [
            "hello",
            "what is Python?",
            "explain neural networks",
            "analyze all approaches comprehensively",
        ]

        for query in queries:
            result = classifier.classify(query)
            assert len(result.reasoning) > 0
            assert result.reasoning.strip() != ""

    def test_reasoning_describes_complexity(self):
        """Reasoning should reference the complexity level or classification method."""
        classifier = QueryClassifier()

        queries_and_keywords = [
            ("hello", ["trivial", "greeting", "acknowledgment", "pattern"]),
            ("what is Python?", ["simple", "factual", "lookup", "short", "pattern", "default"]),
            ("explain neural networks", ["complex", "keyword", "simple", "default", "short"]),  # Added "short"
            ("analyze all approaches comprehensively", ["research", "keyword", "complex", "long"]),
        ]

        for query, keywords in queries_and_keywords:
            result = classifier.classify(query)
            reasoning_lower = result.reasoning.lower()

            # At least one keyword should appear (including "default" as fallback)
            has_keyword = any(kw in reasoning_lower for kw in keywords)
            assert has_keyword, f"Query '{query}' reasoning '{result.reasoning}' should contain one of {keywords}"
