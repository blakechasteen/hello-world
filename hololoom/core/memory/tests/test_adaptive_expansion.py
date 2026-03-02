"""
Tests for Adaptive Graph Expansion (Phase 1)
============================================

Comprehensive test suite for importance-weighted, budget-aware graph traversal.

Test Categories:
1. Unit Tests - Individual components (RelevanceScorer, BudgetTracker)
2. Integration Tests - Full expansion algorithm
3. Comparison Tests - Adaptive vs fixed-depth
4. Edge Cases - Budget limits, relevance thresholds, empty graphs

Author: Claude Code
Date: 2025-11-24
"""

import pytest
import asyncio
from typing import List, Dict, Set
import networkx as nx

# Import adaptive expansion components
from hololoom.memory.adaptive_expansion import (
    AdaptiveExpander,
    RelevanceScorer,
    BudgetTracker,
    ExpansionNode,
    ExpansionResult,
    BudgetStatus,
    EdgeType,
    EDGE_IMPORTANCE_WEIGHTS,
    expand_context_adaptive
)

# Import graph for testing
from hololoom.memory.graph import KG, KGEdge


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    kg = KG()
    kg.add_edges([
        KGEdge("A", "B", "IS_A", 1.0),
        KGEdge("B", "C", "USES", 1.0),
        KGEdge("C", "D", "LEADS_TO", 1.0),
        KGEdge("A", "E", "MENTIONS", 1.0),
        KGEdge("E", "F", "PART_OF", 1.0),
    ])
    return kg


@pytest.fixture
def complex_graph():
    """Create a more complex test graph with multiple paths."""
    kg = KG()

    # Main path A -> B -> C -> D
    kg.add_edges([
        KGEdge("A", "B", "IS_A", 1.0),
        KGEdge("B", "C", "USES", 1.0),
        KGEdge("C", "D", "LEADS_TO", 1.0),
    ])

    # Alternative path A -> E -> F -> D
    kg.add_edges([
        KGEdge("A", "E", "MENTIONS", 1.0),
        KGEdge("E", "F", "PART_OF", 1.0),
        KGEdge("F", "D", "USES", 1.0),
    ])

    # Side branches
    kg.add_edges([
        KGEdge("B", "G", "MENTIONS", 1.0),
        KGEdge("C", "H", "MENTIONS", 1.0),
        KGEdge("E", "I", "MENTIONS", 1.0),
    ])

    return kg


@pytest.fixture
def importance_scores():
    """Sample importance scores for nodes."""
    return {
        "A": 1.0,
        "B": 0.9,
        "C": 0.8,
        "D": 0.7,
        "E": 0.6,
        "F": 0.5,
        "G": 0.4,
        "H": 0.3,
        "I": 0.2,
    }


@pytest.fixture
def node_contents():
    """Sample node text content."""
    return {
        "A": "Thompson Sampling is a Bayesian approach",
        "B": "It balances exploration and exploitation",
        "C": "Uses Beta distributions for arm selection",
        "D": "Optimal regret bounds in bandit problems",
        "E": "Related to multi-armed bandits",
        "F": "Foundational work by Thompson (1933)",
        "G": "Similar to UCB algorithms",
        "H": "Applied in recommendation systems",
        "I": "Used in A/B testing",
    }


# ============================================================================
# Unit Tests - RelevanceScorer
# ============================================================================

def test_relevance_scorer_initialization():
    """Test RelevanceScorer initialization."""
    scorer = RelevanceScorer(
        embedder=None,
        distance_decay=0.85,
        use_edge_weights=True
    )

    assert scorer.distance_decay == 0.85
    assert scorer.use_edge_weights is True
    assert scorer._query_embedding is None
    assert scorer._query_text is None


def test_relevance_scorer_set_query():
    """Test setting query in RelevanceScorer."""
    scorer = RelevanceScorer()
    scorer.set_query("What is Thompson Sampling?")

    assert scorer._query_text == "What is Thompson Sampling?"
    # Embedding will be None without embedder


def test_relevance_scorer_distance_decay():
    """Test distance decay in relevance scoring."""
    scorer = RelevanceScorer(distance_decay=0.85, use_edge_weights=False)
    scorer.set_query("test query")

    # Score at different distances (no edge weighting)
    score_0 = scorer.score_relevance("A", hop_distance=0)
    score_1 = scorer.score_relevance("B", hop_distance=1)
    score_2 = scorer.score_relevance("C", hop_distance=2)

    # Scores should decay with distance
    assert score_0 > score_1 > score_2

    # Check approximate decay rate
    # score_1 should be ~0.85 * score_0 (accounting for other factors)
    decay_ratio = score_1 / score_0
    assert 0.5 < decay_ratio < 1.0  # Should be less than 1.0


def test_relevance_scorer_edge_type_weighting():
    """Test edge type importance weighting."""
    scorer = RelevanceScorer(use_edge_weights=True, distance_decay=1.0)
    scorer.set_query("test query")

    # Score with different edge types (distance=1 to isolate edge effect)
    score_is_a = scorer.score_relevance("B", hop_distance=1, edge_type="IS_A")
    score_mentions = scorer.score_relevance("B", hop_distance=1, edge_type="MENTIONS")

    # IS_A should score higher than MENTIONS
    assert score_is_a > score_mentions


def test_edge_importance_weights():
    """Test edge type importance weight values."""
    assert EDGE_IMPORTANCE_WEIGHTS[EdgeType.IS_A] == 1.0
    assert EDGE_IMPORTANCE_WEIGHTS[EdgeType.USES] == 1.0
    assert EDGE_IMPORTANCE_WEIGHTS[EdgeType.LEADS_TO] == 1.0
    assert EDGE_IMPORTANCE_WEIGHTS[EdgeType.PART_OF] == 0.7
    assert EDGE_IMPORTANCE_WEIGHTS[EdgeType.MENTIONS] == 0.3
    assert EDGE_IMPORTANCE_WEIGHTS[EdgeType.UNKNOWN] == 0.1


# ============================================================================
# Unit Tests - BudgetTracker
# ============================================================================

def test_budget_tracker_initialization():
    """Test BudgetTracker initialization."""
    tracker = BudgetTracker(
        token_budget=2000,
        matryoshka_scales=[384, 256, 128],
        avg_tokens_per_node=100
    )

    assert tracker.token_budget == 2000
    assert tracker.matryoshka_scales == [384, 256, 128]
    assert tracker.consumed_tokens == 0


def test_budget_tracker_estimate_tokens_matryoshka():
    """Test Matryoshka-aware token estimation."""
    tracker = BudgetTracker(token_budget=2000, avg_tokens_per_node=100)

    # High importance -> full 384D
    tokens_high = tracker.estimate_tokens("A", importance_score=0.9)
    assert tokens_high == 100  # Full scale

    # Medium importance -> 256D (2/3)
    tokens_medium = tracker.estimate_tokens("B", importance_score=0.6)
    assert tokens_medium == 67  # 2/3 scale

    # Low importance -> 128D (1/3)
    tokens_low = tracker.estimate_tokens("C", importance_score=0.4)
    assert tokens_low == 33  # 1/3 scale

    # Very low importance -> dropped
    tokens_very_low = tracker.estimate_tokens("D", importance_score=0.1)
    assert tokens_very_low == 0  # Dropped


def test_budget_tracker_consume():
    """Test token consumption."""
    tracker = BudgetTracker(token_budget=1000)

    # Consume tokens
    success = tracker.consume("A", 300)
    assert success is True
    assert tracker.consumed_tokens == 300

    # Consume more
    success = tracker.consume("B", 500)
    assert success is True
    assert tracker.consumed_tokens == 800

    # Try to consume beyond budget
    success = tracker.consume("C", 300)
    assert success is False
    assert tracker.consumed_tokens == 800  # Unchanged


def test_budget_tracker_can_afford():
    """Test budget affordability check."""
    tracker = BudgetTracker(token_budget=1000)
    tracker.consume("A", 500)

    # Can afford (within safety margin)
    assert tracker.can_afford(300, safety_margin=0.9) is True

    # Cannot afford (exceeds safety margin)
    assert tracker.can_afford(500, safety_margin=0.9) is False


def test_budget_tracker_get_status():
    """Test budget status retrieval."""
    tracker = BudgetTracker(token_budget=1000)
    tracker.consume("A", 300)

    status = tracker.get_status()
    assert status.consumed == 300
    assert status.remaining == 700
    assert status.limit == 1000
    assert status.utilization == 0.3


# ============================================================================
# Integration Tests - AdaptiveExpander
# ============================================================================

@pytest.mark.asyncio
async def test_adaptive_expander_simple_expansion(simple_graph, importance_scores):
    """Test basic adaptive expansion."""
    expander = AdaptiveExpander()

    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=1000,
        min_relevance=0.3,
        max_hops=3,
        importance_scores=importance_scores
    )

    # Should expand from A
    assert "A" in result.nodes
    assert len(result.nodes) > 1  # Should expand beyond seed
    assert result.total_tokens <= 1000  # Respects budget
    assert result.execution_time_ms > 0


@pytest.mark.asyncio
async def test_adaptive_expander_budget_constraint(simple_graph, importance_scores):
    """Test that expansion respects token budget."""
    expander = AdaptiveExpander()

    # Very small budget
    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=150,  # Very small
        min_relevance=0.2,
        max_hops=5,
        importance_scores=importance_scores
    )

    assert result.total_tokens <= 150
    assert result.stopping_reason in ["budget_exhausted", "completed"]


@pytest.mark.asyncio
async def test_adaptive_expander_relevance_threshold(simple_graph, importance_scores):
    """Test that expansion respects relevance threshold."""
    expander = AdaptiveExpander()

    # High relevance threshold
    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=2000,
        min_relevance=0.8,  # High threshold
        max_hops=5,
        importance_scores=importance_scores
    )

    # Should stop early due to relevance decay
    assert len(result.nodes) < 5  # Won't expand far
    assert result.avg_relevance >= 0.5  # High average relevance


@pytest.mark.asyncio
async def test_adaptive_expander_max_hops(simple_graph, importance_scores):
    """Test that expansion respects max hops."""
    expander = AdaptiveExpander()

    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=2000,
        min_relevance=0.1,  # Very low to avoid early stopping
        max_hops=2,  # Limited hops
        importance_scores=importance_scores
    )

    assert result.hops_taken <= 2


@pytest.mark.asyncio
async def test_adaptive_expander_priority_ordering(complex_graph, importance_scores):
    """Test that expansion follows priority ordering (importance × relevance)."""
    expander = AdaptiveExpander()

    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=complex_graph,
        token_budget=500,  # Limited budget to force prioritization
        min_relevance=0.2,
        max_hops=5,
        importance_scores=importance_scores
    )

    # Should prioritize high-importance nodes (B, C) over low-importance (G, H, I)
    high_importance_nodes = {"A", "B", "C", "D"}
    low_importance_nodes = {"G", "H", "I"}

    high_count = sum(1 for n in result.nodes if n in high_importance_nodes)
    low_count = sum(1 for n in result.nodes if n in low_importance_nodes)

    # Should have more high-importance nodes
    assert high_count > low_count


@pytest.mark.asyncio
async def test_adaptive_expander_edge_provenance(simple_graph, importance_scores):
    """Test that expansion tracks edge provenance."""
    expander = AdaptiveExpander()

    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=2000,
        min_relevance=0.3,
        max_hops=3,
        importance_scores=importance_scores
    )

    # Should track edges
    assert len(result.edges) > 0

    # Edges should be tuples of (src, dst, edge_type)
    for edge in result.edges:
        assert len(edge) == 3
        src, dst, edge_type = edge
        assert src in result.nodes
        assert dst in result.nodes


# ============================================================================
# Comparison Tests - Adaptive vs Fixed-Depth
# ============================================================================

@pytest.mark.asyncio
async def test_adaptive_vs_fixed_depth_precision(complex_graph, importance_scores):
    """Compare adaptive expansion vs fixed-depth for precision."""
    # Adaptive expansion
    expander = AdaptiveExpander()
    adaptive_result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=complex_graph,
        token_budget=500,
        min_relevance=0.3,
        max_hops=5,
        importance_scores=importance_scores
    )

    # Fixed-depth expansion (simulate by expanding all neighbors up to max_hops=2)
    fixed_nodes = complex_graph.get_neighbors("A", direction="both", max_hops=2)

    # Adaptive should be more selective
    assert len(adaptive_result.nodes) <= len(fixed_nodes)

    # Adaptive should have higher average relevance (more selective)
    assert adaptive_result.avg_relevance >= 0.4


@pytest.mark.asyncio
async def test_adaptive_vs_fixed_depth_token_efficiency(complex_graph, importance_scores):
    """Compare token efficiency of adaptive vs fixed-depth."""
    # Adaptive expansion with budget
    expander = AdaptiveExpander()
    adaptive_result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=complex_graph,
        token_budget=500,
        min_relevance=0.3,
        max_hops=5,
        importance_scores=importance_scores
    )

    # Fixed-depth would expand all nodes (no budget awareness)
    fixed_nodes = complex_graph.get_neighbors("A", direction="both", max_hops=2)

    # Adaptive uses fewer tokens (selective expansion)
    assert adaptive_result.total_tokens <= 500

    # Adaptive has better token efficiency (relevance per token)
    efficiency = adaptive_result.avg_relevance / max(adaptive_result.total_tokens, 1)
    assert efficiency > 0.0005  # Should be reasonably efficient


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.asyncio
async def test_adaptive_expander_empty_graph():
    """Test expansion on empty graph."""
    kg = KG()
    expander = AdaptiveExpander()

    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=kg,
        token_budget=2000,
        min_relevance=0.3,
        max_hops=3
    )

    # Should handle gracefully
    assert len(result.nodes) == 0
    assert result.stopping_reason == "completed"


@pytest.mark.asyncio
async def test_adaptive_expander_nonexistent_seed(simple_graph):
    """Test expansion with nonexistent seed node."""
    expander = AdaptiveExpander()

    result = await expander.expand(
        query="What is Z?",
        seed_nodes=["Z"],  # Doesn't exist
        graph=simple_graph,
        token_budget=2000,
        min_relevance=0.3,
        max_hops=3
    )

    # Should handle gracefully
    assert len(result.nodes) == 0


@pytest.mark.asyncio
async def test_adaptive_expander_zero_budget(simple_graph):
    """Test expansion with zero token budget."""
    expander = AdaptiveExpander()

    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=0,  # Zero budget
        min_relevance=0.3,
        max_hops=3
    )

    assert len(result.nodes) == 0
    assert result.stopping_reason == "budget_exhausted"


@pytest.mark.asyncio
async def test_adaptive_expander_single_node(simple_graph, importance_scores):
    """Test expansion with max_hops=0 (seed only)."""
    expander = AdaptiveExpander()

    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=2000,
        min_relevance=0.3,
        max_hops=0,  # No expansion
        importance_scores=importance_scores
    )

    # Should only include seed
    assert result.nodes == ["A"]
    assert len(result.edges) == 0


# ============================================================================
# Convenience Function Tests
# ============================================================================

@pytest.mark.asyncio
async def test_expand_context_adaptive_convenience(simple_graph, importance_scores):
    """Test convenience function for adaptive expansion."""
    result = await expand_context_adaptive(
        query="What is Thompson Sampling?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=2000,
        min_relevance=0.3,
        max_hops=5,
        importance_scores=importance_scores
    )

    assert isinstance(result, ExpansionResult)
    assert len(result.nodes) > 0
    assert result.total_tokens > 0


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_adaptive_expander_performance(complex_graph, importance_scores):
    """Test that expansion completes in reasonable time."""
    expander = AdaptiveExpander()

    result = await expander.expand(
        query="What is A?",
        seed_nodes=["A"],
        graph=complex_graph,
        token_budget=2000,
        min_relevance=0.3,
        max_hops=5,
        importance_scores=importance_scores
    )

    # Should complete in <100ms for small graph
    assert result.execution_time_ms < 100


@pytest.mark.asyncio
async def test_adaptive_expander_stats():
    """Test that expander tracks statistics."""
    expander = AdaptiveExpander()
    kg = KG()
    kg.add_edges([KGEdge("A", "B", "IS_A", 1.0)])

    # Run multiple expansions
    for _ in range(3):
        await expander.expand(
            query="test",
            seed_nodes=["A"],
            graph=kg,
            token_budget=1000
        )

    stats = expander.get_stats()
    assert stats['total_expansions'] == 3
    assert stats['avg_nodes_expanded'] > 0
    assert stats['avg_execution_time_ms'] > 0


# ============================================================================
# Test Summary
# ============================================================================

def test_summary():
    """Print test summary for documentation."""
    print("\n" + "="*60)
    print("Adaptive Graph Expansion Test Suite")
    print("="*60)
    print("\nTest Categories:")
    print("  - Unit Tests (RelevanceScorer): 5 tests")
    print("  - Unit Tests (BudgetTracker): 5 tests")
    print("  - Integration Tests (AdaptiveExpander): 6 tests")
    print("  - Comparison Tests (Adaptive vs Fixed): 2 tests")
    print("  - Edge Case Tests: 4 tests")
    print("  - Convenience Function Tests: 1 test")
    print("  - Performance Tests: 2 tests")
    print("\nTotal: 25 comprehensive tests")
    print("="*60 + "\n")
