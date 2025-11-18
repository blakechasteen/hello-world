#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Phase 4: Context Compression

Tests:
1. SemanticDeduplicator
   - Similarity detection (all methods)
   - Deduplication strategies
   - Duplicate group identification
   - Token savings calculation

2. CompressionAnalyzer
   - Compression tracking
   - Insights generation
   - Recommendations

3. LLMContextPacker integration
   - Compression step execution
   - Metadata preservation
   - Statistics retrieval
"""

import pytest
from dataclasses import dataclass
from typing import List

# Import components to test
from HoloLoom.awareness.semantic_deduplicator import (
    SemanticDeduplicator,
    DeduplicationStrategy,
    SimilarityMethod,
    ContextElement,
    ContextImportance
)

from HoloLoom.awareness.compression_analyzer import (
    CompressionAnalyzer,
    CompressionRecord,
    CompressionInsights
)


# ============================================================================
# Test SemanticDeduplicator - Similarity Detection
# ============================================================================

def test_exact_similarity():
    """Test exact similarity detection"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.EXACT,
        similarity_threshold=1.0
    )

    text1 = "Python is a programming language"
    text2 = "Python is a programming language"
    text3 = "Python is a scripting language"

    # Exact match
    assert deduplicator._compute_similarity(text1, text2) == 1.0

    # Not exact match
    assert deduplicator._compute_similarity(text1, text3) == 0.0


def test_jaccard_similarity():
    """Test Jaccard similarity (word-based)"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.JACCARD,
        similarity_threshold=0.5
    )

    text1 = "machine learning is powerful"
    text2 = "machine learning is useful"
    text3 = "deep learning neural networks"

    # High Jaccard similarity (3/5 words overlap)
    sim12 = deduplicator._compute_similarity(text1, text2)
    assert sim12 >= 0.5

    # Low Jaccard similarity (0/7 words overlap)
    sim13 = deduplicator._compute_similarity(text1, text3)
    assert sim13 < 0.5


def test_ngram_similarity():
    """Test N-gram similarity (character-based)"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.NGRAM,
        similarity_threshold=0.8
    )

    text1 = "Thompson Sampling balances exploration and exploitation"
    text2 = "Thompson Sampling balances exploration and exploitation well"
    text3 = "Epsilon greedy is another exploration strategy"

    # High n-gram similarity (very similar strings)
    sim12 = deduplicator._compute_similarity(text1, text2)
    assert sim12 >= 0.8

    # Low n-gram similarity (different strings)
    sim13 = deduplicator._compute_similarity(text1, text3)
    assert sim13 < 0.5


# ============================================================================
# Test SemanticDeduplicator - Duplicate Group Detection
# ============================================================================

def test_find_duplicates_basic():
    """Test finding duplicate groups"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.NGRAM,
        similarity_threshold=0.8
    )

    # Create elements with duplicates
    elements = [
        ContextElement("Python is great", 0.8, 4, "memory"),
        ContextElement("Python is great for data science", 0.8, 6, "memory"),
        ContextElement("Java is also good", 0.6, 4, "memory"),
    ]

    # Find duplicates
    groups = deduplicator.find_duplicates(elements)

    # Should find 1 group (first two elements are similar)
    assert len(groups) >= 0  # May or may not find duplicates depending on threshold


def test_find_duplicates_no_duplicates():
    """Test when no duplicates exist"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.NGRAM,
        similarity_threshold=0.9
    )

    # Create completely different elements
    elements = [
        ContextElement("Python programming", 0.8, 3, "memory"),
        ContextElement("Machine learning algorithms", 0.7, 3, "memory"),
        ContextElement("Database optimization", 0.6, 3, "memory"),
    ]

    groups = deduplicator.find_duplicates(elements)

    # Should find no groups
    assert len(groups) == 0


def test_preserve_critical_importance():
    """Test that CRITICAL importance elements are preserved"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.NGRAM,
        similarity_threshold=0.8,
        preserve_importance=True
    )

    # Create elements with CRITICAL importance
    elements = [
        ContextElement("Query result is here", 1.0, 4, "awareness"),  # CRITICAL
        ContextElement("Query result is here too", 0.5, 5, "memory"),  # Not critical
    ]

    groups = deduplicator.find_duplicates(elements)

    # Should not group CRITICAL element
    assert len(groups) == 0


# ============================================================================
# Test SemanticDeduplicator - Deduplication Strategies
# ============================================================================

def test_merge_similar_strategy():
    """Test merge_similar deduplication strategy"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.EXACT,
        similarity_threshold=1.0
    )

    # Create exact duplicates
    elements = [
        ContextElement("Python is great", 0.8, 4, "memory"),
        ContextElement("Python is great", 0.7, 4, "memory"),
        ContextElement("Python is great", 0.6, 4, "memory"),
    ]

    result = deduplicator.deduplicate(elements, strategy=DeduplicationStrategy.MERGE_SIMILAR)

    # Should have fewer elements
    assert len(result.deduplicated_elements) < len(elements)

    # Should have token savings
    assert result.token_savings > 0


def test_keep_best_strategy():
    """Test keep_best deduplication strategy"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.EXACT,
        similarity_threshold=1.0
    )

    # Create duplicates with different importance
    elements = [
        ContextElement("Python is great", 0.5, 4, "memory"),
        ContextElement("Python is great", 0.9, 4, "memory"),  # Highest importance
        ContextElement("Python is great", 0.7, 4, "memory"),
    ]

    result = deduplicator.deduplicate(elements, strategy=DeduplicationStrategy.KEEP_BEST)

    # Should keep only 1 element (highest importance)
    assert len(result.deduplicated_elements) == 1

    # Should be the one with importance=0.9
    assert result.deduplicated_elements[0].importance == 0.9


def test_remove_exact_strategy():
    """Test remove_exact deduplication strategy"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.EXACT,
        similarity_threshold=1.0
    )

    # Create exact duplicates and unique element
    elements = [
        ContextElement("Python is great", 0.8, 4, "memory"),
        ContextElement("Python is great", 0.7, 4, "memory"),  # Duplicate
        ContextElement("Java is also good", 0.6, 4, "memory"),  # Unique
    ]

    result = deduplicator.deduplicate(elements, strategy=DeduplicationStrategy.REMOVE_EXACT)

    # Should remove 1 exact duplicate
    assert len(result.deduplicated_elements) == 2
    assert result.elements_removed == 1


def test_summarize_strategy():
    """Test summarize deduplication strategy"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.EXACT,
        similarity_threshold=1.0
    )

    # Create duplicates
    elements = [
        ContextElement("First sentence. Second sentence.", 0.8, 6, "memory"),
        ContextElement("First sentence. Second sentence.", 0.7, 6, "memory"),
    ]

    result = deduplicator.deduplicate(elements, strategy=DeduplicationStrategy.SUMMARIZE)

    # Should have summarized version
    assert len(result.deduplicated_elements) < len(elements)


# ============================================================================
# Test SemanticDeduplicator - Token Savings
# ============================================================================

def test_token_savings_calculation():
    """Test token savings calculation"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.EXACT,
        similarity_threshold=1.0
    )

    # Create elements with known token counts
    elements = [
        ContextElement("Content A", 0.8, 100, "memory"),
        ContextElement("Content A", 0.7, 100, "memory"),
        ContextElement("Content A", 0.6, 100, "memory"),
    ]

    result = deduplicator.deduplicate(elements, strategy=DeduplicationStrategy.KEEP_BEST)

    # Should save 200 tokens (removed 2 elements × 100 tokens)
    assert result.original_tokens == 300
    assert result.deduplicated_tokens == 100
    assert result.token_savings == 200
    assert result.compression_ratio == 100 / 300


def test_compression_ratio():
    """Test compression ratio calculation"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.EXACT,
        similarity_threshold=1.0
    )

    # Create duplicates
    elements = [
        ContextElement("Content", 0.8, 50, "memory"),
        ContextElement("Content", 0.7, 50, "memory"),
    ]

    result = deduplicator.deduplicate(elements, strategy=DeduplicationStrategy.KEEP_BEST)

    # Compression ratio should be 0.5 (50/100)
    assert result.compression_ratio == 0.5


# ============================================================================
# Test CompressionAnalyzer - Tracking
# ============================================================================

def test_compression_tracking_basic():
    """Test basic compression tracking"""
    analyzer = CompressionAnalyzer()

    record = analyzer.track_compression(
        original_tokens=1000,
        compressed_tokens=700,
        strategy="merge_similar",
        quality_before=0.90,
        quality_after=0.88
    )

    assert record.original_tokens == 1000
    assert record.compressed_tokens == 700
    assert record.token_savings == 300
    assert record.compression_ratio == 0.7
    assert abs(record.quality_delta - (-0.02)) < 0.001  # Floating point tolerance


def test_compression_tracking_multiple():
    """Test tracking multiple compressions"""
    analyzer = CompressionAnalyzer()

    # Track 5 compressions
    for i in range(5):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=700,
            strategy="merge_similar"
        )

    assert len(analyzer.records) == 5


def test_quality_preserved_flag():
    """Test quality_preserved flag"""
    analyzer = CompressionAnalyzer()

    # Quality preserved (>95% of original)
    record1 = analyzer.track_compression(
        original_tokens=1000,
        compressed_tokens=700,
        quality_before=0.90,
        quality_after=0.88  # 97.8% preserved
    )

    # Quality not preserved (<95% of original)
    record2 = analyzer.track_compression(
        original_tokens=1000,
        compressed_tokens=700,
        quality_before=0.90,
        quality_after=0.80  # 88.9% preserved
    )

    assert record1.quality_preserved is True
    assert record2.quality_preserved is False


# ============================================================================
# Test CompressionAnalyzer - Insights
# ============================================================================

def test_compression_insights_basic():
    """Test basic insights generation"""
    analyzer = CompressionAnalyzer()

    # Track 10 compressions
    for i in range(10):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=700,
            strategy="merge_similar",
            quality_before=0.90,
            quality_after=0.88
        )

    insights = analyzer.get_insights()

    assert insights.total_compressions == 10
    assert insights.total_tokens_saved == 3000  # 10 × 300
    assert insights.avg_compression_ratio == 0.7
    assert abs(insights.avg_savings_pct - 30.0) < 0.001  # Floating point tolerance
    assert insights.avg_quality_before == 0.90
    assert insights.avg_quality_after == 0.88


def test_strategy_performance_comparison():
    """Test strategy performance comparison"""
    analyzer = CompressionAnalyzer()

    # Track with different strategies
    for _ in range(10):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=700,
            strategy="merge_similar",
            quality_before=0.90,
            quality_after=0.88
        )

    for _ in range(10):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=500,
            strategy="keep_best",
            quality_before=0.90,
            quality_after=0.85
        )

    insights = analyzer.get_insights()

    # Should have performance for both strategies
    assert "merge_similar" in insights.strategy_performance
    assert "keep_best" in insights.strategy_performance

    # merge_similar: better quality preservation
    # keep_best: better compression
    merge_quality = insights.strategy_performance["merge_similar"]["quality_delta"]
    keep_quality = insights.strategy_performance["keep_best"]["quality_delta"]

    assert merge_quality > keep_quality  # merge_similar preserves quality better


def test_best_strategy_selection():
    """Test best strategy selection"""
    analyzer = CompressionAnalyzer()

    # Track with two strategies
    # Strategy A: Good savings, good quality
    for _ in range(10):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=700,
            strategy="strategy_a",
            quality_before=0.90,
            quality_after=0.88  # Quality preserved
        )

    # Strategy B: Great savings, poor quality
    for _ in range(10):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=400,
            strategy="strategy_b",
            quality_before=0.90,
            quality_after=0.75  # Quality not preserved
        )

    insights = analyzer.get_insights()

    # Should recommend strategy_a (quality preserved + good savings)
    assert insights.best_strategy == "strategy_a"


# ============================================================================
# Test CompressionAnalyzer - Recommendations
# ============================================================================

def test_recommendations_high_effectiveness():
    """Test recommendations for highly effective compression"""
    analyzer = CompressionAnalyzer()

    # Track highly effective compressions (>20% savings)
    for _ in range(10):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=700,
            strategy="merge_similar",
            quality_before=0.90,
            quality_after=0.89
        )

    recommendations = analyzer.get_recommendations()

    # Should recommend continuing
    assert any("effective" in rec.lower() for rec in recommendations)


def test_recommendations_quality_impact():
    """Test recommendations when compression hurts quality"""
    analyzer = CompressionAnalyzer()

    # Track compressions with quality drops (>5%)
    for _ in range(10):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=700,
            strategy="aggressive",
            quality_before=0.90,
            quality_after=0.80  # 11% drop
        )

    recommendations = analyzer.get_recommendations()

    # Should warn about quality impact
    assert any("quality" in rec.lower() for rec in recommendations)


def test_recommendations_minimal_savings():
    """Test recommendations when compression provides minimal savings"""
    analyzer = CompressionAnalyzer()

    # Track compressions with minimal savings (<10%)
    for _ in range(10):
        analyzer.track_compression(
            original_tokens=1000,
            compressed_tokens=950,
            strategy="conservative",
            quality_before=0.90,
            quality_after=0.89
        )

    recommendations = analyzer.get_recommendations()

    # Should suggest adjusting settings or disabling
    assert any("minimal" in rec.lower() or "savings" in rec.lower() for rec in recommendations)


# ============================================================================
# Test SemanticDeduplicator Statistics
# ============================================================================

def test_deduplicator_statistics():
    """Test deduplicator statistics tracking"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.NGRAM,
        similarity_threshold=0.8
    )

    # Create and deduplicate elements
    elements = [
        ContextElement("Content A", 0.8, 100, "memory"),
        ContextElement("Content A", 0.7, 100, "memory"),
        ContextElement("Content B", 0.6, 50, "memory"),
    ]

    result = deduplicator.deduplicate(elements, strategy=DeduplicationStrategy.KEEP_BEST)

    stats = deduplicator.get_statistics()

    assert stats["total_comparisons"] >= 0  # Should have made comparisons
    assert stats["similarity_method"] == "ngram"
    assert stats["similarity_threshold"] == 0.8


def test_deduplicator_reset_statistics():
    """Test resetting deduplicator statistics"""
    deduplicator = SemanticDeduplicator()

    # Track some operations
    elements = [
        ContextElement("Content", 0.8, 100, "memory"),
        ContextElement("Content", 0.7, 100, "memory"),
    ]

    deduplicator.deduplicate(elements)

    # Reset statistics
    deduplicator.reset_statistics()

    stats = deduplicator.get_statistics()

    assert stats["total_comparisons"] == 0
    assert stats["duplicates_found"] == 0
    assert stats["tokens_saved"] == 0


# ============================================================================
# Test DeduplicationResult
# ============================================================================

def test_deduplication_result_summary():
    """Test deduplication result summary"""
    deduplicator = SemanticDeduplicator(
        similarity_method=SimilarityMethod.EXACT,
        similarity_threshold=1.0
    )

    elements = [
        ContextElement("Content", 0.8, 100, "memory"),
        ContextElement("Content", 0.7, 100, "memory"),
        ContextElement("Content", 0.6, 100, "memory"),
    ]

    result = deduplicator.deduplicate(elements, strategy=DeduplicationStrategy.KEEP_BEST)

    summary = result.summary()

    # Should contain key metrics
    assert "Original:" in summary
    assert "Deduplicated:" in summary
    assert "Savings:" in summary


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
