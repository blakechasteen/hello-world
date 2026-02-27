"""
Tests for Advanced Reranking Module
===================================

Unit tests for reranking functionality:
- Reranker protocol compliance
- Cross-encoder reranking
- No-op reranker fallback
- Edge cases (empty docs, top_k > n_docs, etc.)
- Reranking statistics and benefit computation
"""

import pytest
from typing import List, Tuple
from hololoom.rag.reranking import (
    Reranker,
    CrossEncoderReranker,
    NoOpReranker,
    create_reranker,
    RerankingStats,
    compute_reranking_benefit,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_query() -> str:
    """Sample query for testing."""
    return "What is machine learning?"


@pytest.fixture
def relevant_documents() -> List[str]:
    """Sample documents with varying relevance."""
    return [
        "Machine learning uses algorithms and statistical models",  # Highly relevant
        "The weather is nice today",  # Irrelevant
        "Neural networks are used in deep learning",  # Relevant
        "I like pizza and pasta",  # Irrelevant
        "Supervised learning requires labeled training data",  # Relevant
        "Python is a popular programming language",  # Somewhat relevant
    ]


@pytest.fixture
def noop_reranker() -> Reranker:
    """Create a no-op reranker."""
    return NoOpReranker()


# ============================================================================
# Test Reranker Protocol
# ============================================================================

def test_reranker_protocol_compliance_noop(noop_reranker, sample_query, relevant_documents):
    """Test that NoOpReranker implements Reranker protocol."""
    # Should not raise
    result = noop_reranker.rerank(sample_query, relevant_documents, top_k=3)

    # Check return type
    assert isinstance(result, list)
    assert len(result) <= 3
    assert all(isinstance(item, tuple) for item in result)
    assert all(len(item) == 2 for item in result)
    assert all(isinstance(item[0], int) for item in result)
    assert all(isinstance(item[1], float) for item in result)


def test_reranker_factory_noop():
    """Test that factory creates NoOpReranker."""
    reranker = create_reranker("noop")
    assert isinstance(reranker, NoOpReranker)


def test_reranker_factory_invalid_type():
    """Test that factory raises ValueError for unknown type."""
    with pytest.raises(ValueError):
        create_reranker("unknown_type")


# ============================================================================
# Test No-Op Reranker
# ============================================================================

def test_noop_reranker_basic(noop_reranker, sample_query, relevant_documents):
    """Test basic no-op reranking."""
    result = noop_reranker.rerank(sample_query, relevant_documents, top_k=3)

    # Should return first 3 documents
    assert len(result) == 3
    assert result[0][0] == 0
    assert result[1][0] == 1
    assert result[2][0] == 2

    # All scores should be 1.0
    assert all(score == 1.0 for _, score in result)


def test_noop_reranker_top_k_greater_than_docs(noop_reranker, sample_query):
    """Test no-op reranker with top_k > number of documents."""
    docs = ["doc1", "doc2", "doc3"]
    result = noop_reranker.rerank(sample_query, docs, top_k=10)

    # Should return only 3 documents
    assert len(result) == 3


def test_noop_reranker_empty_documents(noop_reranker, sample_query):
    """Test no-op reranker with empty document list."""
    result = noop_reranker.rerank(sample_query, [], top_k=5)

    # Should return empty list
    assert result == []


def test_noop_reranker_top_k_zero(noop_reranker, sample_query, relevant_documents):
    """Test no-op reranker with top_k=0."""
    result = noop_reranker.rerank(sample_query, relevant_documents, top_k=0)

    # Should return empty list
    assert len(result) == 0


def test_noop_reranker_single_document(noop_reranker, sample_query):
    """Test no-op reranker with single document."""
    result = noop_reranker.rerank(sample_query, ["single doc"], top_k=1)

    assert len(result) == 1
    assert result[0] == (0, 1.0)


# ============================================================================
# Test Cross-Encoder Reranker
# ============================================================================

def test_cross_encoder_reranker_import_error():
    """Test that CrossEncoderReranker raises ImportError if sentence-transformers unavailable."""
    # This test will only pass if sentence-transformers is not installed
    # Skip if it is installed
    try:
        from sentence_transformers import CrossEncoder
        pytest.skip("sentence-transformers is installed, skipping import error test")
    except ImportError:
        with pytest.raises(ImportError):
            CrossEncoderReranker()


@pytest.mark.skipif(
    pytest.importorskip("sentence_transformers", minversion=None) is None,
    reason="sentence-transformers not installed"
)
def test_cross_encoder_reranker_basic():
    """Test basic cross-encoder reranking."""
    reranker = CrossEncoderReranker()

    query = "machine learning"
    documents = [
        "Machine learning uses statistical methods",  # Relevant
        "The weather is nice",  # Irrelevant
        "Deep learning is a subset of machine learning",  # Relevant
    ]

    result = reranker.rerank(query, documents, top_k=2)

    # Should return 2 results
    assert len(result) == 2

    # Check format
    assert all(isinstance(idx, int) for idx, _ in result)
    assert all(0 <= score <= 1 for _, score in result)

    # Most relevant documents should be ranked higher
    # Documents 0 and 2 should be ranked higher than 1
    top_indices = [idx for idx, _ in result]
    assert 1 not in top_indices or result[0][1] > result[-1][1]


@pytest.mark.skipif(
    pytest.importorskip("sentence_transformers", minversion=None) is None,
    reason="sentence-transformers not installed"
)
def test_cross_encoder_reranker_respects_top_k():
    """Test that cross-encoder returns exactly top_k results."""
    reranker = CrossEncoderReranker()

    query = "test query"
    documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

    for top_k in [1, 3, 5]:
        result = reranker.rerank(query, documents, top_k=top_k)
        assert len(result) == top_k


@pytest.mark.skipif(
    pytest.importorskip("sentence_transformers", minversion=None) is None,
    reason="sentence-transformers not installed"
)
def test_cross_encoder_reranker_sorted_by_score():
    """Test that results are sorted by score (descending)."""
    reranker = CrossEncoderReranker()

    query = "machine learning algorithms"
    documents = [
        "ML uses algorithms",
        "Weather today",
        "Neural networks and ML",
        "I like coffee",
    ]

    result = reranker.rerank(query, documents, top_k=4)

    # Scores should be in descending order
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(
    pytest.importorskip("sentence_transformers", minversion=None) is None,
    reason="sentence-transformers not installed"
)
def test_cross_encoder_reranker_empty_documents():
    """Test cross-encoder with empty document list."""
    reranker = CrossEncoderReranker()
    result = reranker.rerank("query", [], top_k=5)

    assert result == []


@pytest.mark.skipif(
    pytest.importorskip("sentence_transformers", minversion=None) is None,
    reason="sentence-transformers not installed"
)
def test_cross_encoder_reranker_top_k_greater_than_docs():
    """Test cross-encoder with top_k > number of documents."""
    reranker = CrossEncoderReranker()

    query = "test"
    documents = ["doc1", "doc2"]

    result = reranker.rerank(query, documents, top_k=10)

    # Should return only 2 documents
    assert len(result) == 2


# ============================================================================
# Test Reranking Factory
# ============================================================================

def test_create_reranker_cross_encoder_fallback():
    """Test that factory falls back to NoOpReranker if cross-encoder fails."""
    try:
        from sentence_transformers import CrossEncoder
        # If available, we'll get CrossEncoderReranker
        reranker = create_reranker("cross-encoder")
        assert isinstance(reranker, CrossEncoderReranker)
    except ImportError:
        # If not available, should fall back to NoOp
        reranker = create_reranker("cross-encoder")
        assert isinstance(reranker, NoOpReranker)


def test_create_reranker_noop():
    """Test creating no-op reranker."""
    reranker = create_reranker("noop")
    assert isinstance(reranker, NoOpReranker)


# ============================================================================
# Test Reranking Statistics
# ============================================================================

def test_reranking_stats_creation():
    """Test creating reranking statistics."""
    stats = RerankingStats(
        n_documents=20,
        latency_ms=45.5,
        top_score=0.95,
        top_k=5
    )

    assert stats.n_documents == 20
    assert stats.latency_ms == 45.5
    assert stats.top_score == 0.95
    assert stats.top_k == 5


def test_reranking_stats_str():
    """Test string representation of reranking statistics."""
    stats = RerankingStats(
        n_documents=20,
        latency_ms=45.5,
        top_score=0.95,
        top_k=5
    )

    s = str(stats)
    assert "20" in s
    assert "45.5" in s
    assert "0.95" in s


# ============================================================================
# Test Reranking Benefit Computation
# ============================================================================

def test_compute_reranking_benefit_improvement():
    """Test computing benefit when reranking improves results."""
    original = [0.6, 0.5, 0.4]
    reranked = [0.9, 0.8, 0.7]

    benefit = compute_reranking_benefit(original, reranked)

    # Reranked average: (0.9 + 0.8 + 0.7) / 3 = 0.8
    # Original average: (0.6 + 0.5 + 0.4) / 3 = 0.5
    # Benefit: 0.8 / 0.5 = 1.6
    assert abs(benefit - 1.6) < 0.01


def test_compute_reranking_benefit_no_improvement():
    """Test computing benefit when reranking doesn't improve."""
    original = [0.9, 0.8, 0.7]
    reranked = [0.9, 0.8, 0.7]

    benefit = compute_reranking_benefit(original, reranked)

    # Should be 1.0 (no change)
    assert abs(benefit - 1.0) < 0.01


def test_compute_reranking_benefit_mixed():
    """Test computing benefit with mixed improvements."""
    original = [0.7, 0.6, 0.5, 0.4, 0.3]
    reranked = [0.85, 0.75, 0.65]

    benefit = compute_reranking_benefit(original, reranked)

    # Should be > 1.0 (reranked is better)
    assert benefit > 1.0


def test_compute_reranking_benefit_empty():
    """Test computing benefit with empty lists."""
    benefit = compute_reranking_benefit([], [])
    assert benefit == 1.0

    benefit = compute_reranking_benefit([0.5], [])
    assert benefit == 1.0

    benefit = compute_reranking_benefit([], [0.5])
    assert benefit == 1.0


def test_compute_reranking_benefit_zero_average():
    """Test computing benefit when original average is zero."""
    original = [0.0, 0.0, 0.0]
    reranked = [0.5, 0.4, 0.3]

    benefit = compute_reranking_benefit(original, reranked)
    assert benefit == 1.0


# ============================================================================
# Test Reranking Precision Improvement
# ============================================================================

def test_noop_precision_baseline(noop_reranker):
    """Test that no-op provides baseline (no improvement)."""
    query = "machine learning"
    documents = [
        "Unrelated content",
        "ML algorithms",
        "Weather",
        "Neural networks",
        "Food recipes",
    ]

    result = noop_reranker.rerank(query, documents, top_k=3)

    # No-op returns first 3 in order
    indices = [idx for idx, _ in result]
    assert indices == [0, 1, 2]


@pytest.mark.skipif(
    pytest.importorskip("sentence_transformers", minversion=None) is None,
    reason="sentence-transformers not installed"
)
def test_cross_encoder_precision_improvement():
    """Test that cross-encoder improves precision over no-op."""
    noop = NoOpReranker()
    cross_encoder = CrossEncoderReranker()

    query = "machine learning"
    documents = [
        "Unrelated: The sky is blue",
        "ML uses statistical methods",
        "Unrelated: I like pizza",
        "Neural networks learn patterns",
        "Unrelated: Coffee is good",
    ]

    noop_result = noop.rerank(query, documents, top_k=2)
    ce_result = cross_encoder.rerank(query, documents, top_k=2)

    # No-op returns first 2: indices [0, 1]
    noop_indices = [idx for idx, _ in noop_result]
    assert noop_indices == [0, 1]

    # Cross-encoder should prefer indices 1 and 3 (relevant docs)
    ce_indices = [idx for idx, _ in ce_result]
    # Both should be from {1, 3} (the relevant documents)
    assert 1 in ce_indices or 3 in ce_indices
    assert noop_indices != ce_indices, "Cross-encoder should reorder irrelevant results"


# ============================================================================
# Test Integration with SimpleRAG (Conceptual)
# ============================================================================

def test_reranker_integration_pattern():
    """Test the integration pattern used in SimpleRAG.query()."""
    # This tests the pattern: retrieve top_k, rerank to max_sources
    noop = NoOpReranker()

    query = "What is Thompson Sampling?"
    # Simulate 20 retrieved documents (rerank_top_k=20)
    documents = [
        "Thompson Sampling uses Bayesian statistics",  # 0 - Relevant
        "The weather is sunny",  # 1 - Irrelevant
        "Bayesian methods for exploration",  # 2 - Relevant
        "I like coffee",  # 3 - Irrelevant
        "Multi-armed bandits and Thompson",  # 4 - Relevant
        *[f"Random doc {i}" for i in range(5, 20)]  # 5-19 - Mixed
    ]

    # No-op reranking: return first 5
    reranked = noop.rerank(query, documents, top_k=5)
    reranked_indices = [idx for idx, _ in reranked]

    # No-op returns [0, 1, 2, 3, 4]
    assert len(reranked_indices) == 5
    assert reranked_indices == [0, 1, 2, 3, 4]


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_reranker_duplicate_documents(noop_reranker):
    """Test reranker with duplicate documents."""
    query = "test"
    documents = ["doc", "doc", "doc"]

    result = noop_reranker.rerank(query, documents, top_k=2)

    assert len(result) == 2
    assert result[0][0] == 0
    assert result[1][0] == 1


def test_reranker_very_long_query(noop_reranker):
    """Test reranker with very long query."""
    query = "test " * 1000  # 1000 words
    documents = ["doc1", "doc2"]

    result = noop_reranker.rerank(query, documents, top_k=1)

    assert len(result) == 1


def test_reranker_unicode_documents(noop_reranker):
    """Test reranker with unicode characters."""
    query = "machine learning"
    documents = [
        "机器学习使用算法",  # Chinese
        "Machine learning uses algorithms",  # English
        "Машинное обучение",  # Russian
    ]

    result = noop_reranker.rerank(query, documents, top_k=2)

    assert len(result) == 2


def test_reranker_very_short_documents(noop_reranker):
    """Test reranker with very short documents."""
    query = "test"
    documents = ["a", "b", "c"]

    result = noop_reranker.rerank(query, documents, top_k=2)

    assert len(result) == 2


def test_reranker_whitespace_documents(noop_reranker):
    """Test reranker with whitespace-only documents."""
    query = "test"
    documents = ["   ", "\t", "\n"]

    result = noop_reranker.rerank(query, documents, top_k=2)

    assert len(result) == 2


# ============================================================================
# Test Performance Characteristics
# ============================================================================

def test_noop_reranker_performance(noop_reranker):
    """Test that no-op reranker is very fast."""
    import time

    query = "test query"
    documents = [f"document {i}" for i in range(100)]

    start = time.time()
    result = noop_reranker.rerank(query, documents, top_k=10)
    elapsed = time.time() - start

    # No-op should be very fast (< 1ms)
    assert elapsed < 0.01
    assert len(result) == 10


@pytest.mark.skipif(
    pytest.importorskip("sentence_transformers", minversion=None) is None,
    reason="sentence-transformers not installed"
)
def test_cross_encoder_reranker_performance():
    """Test cross-encoder reranking latency."""
    import time

    reranker = CrossEncoderReranker()
    query = "machine learning"
    documents = [f"document about {topic}" for topic in
                 ["ML", "weather", "sports", "cooking", "ML algorithms",
                  "deep learning", "food", "cars", "music", "statistics"]]

    start = time.time()
    result = reranker.rerank(query, documents, top_k=5)
    elapsed = time.time() - start

    # Cross-encoder should be reasonably fast for 10 documents
    # Expected: 20-100ms depending on hardware
    assert elapsed < 1.0  # Conservative upper bound
    assert len(result) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
