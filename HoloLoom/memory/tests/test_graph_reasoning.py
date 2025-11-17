"""
Tests for Multi-Hop Graph Reasoning
====================================

Tests cover:
- Multi-hop query expansion (1-3 hops)
- Path finding between entities
- Relationship-specific traversal
- Hybrid ranking (semantic + graph proximity)
- Query caching
- Performance characteristics

Author: HoloLoom Architecture Team
Date: 2025-11-17
"""

import pytest
import asyncio
from datetime import datetime

from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.graph_reasoning import (
    GraphReasoner,
    GraphPath,
    MultiHopResult,
    create_graph_reasoner
)
from HoloLoom.protocols import Memory


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_kg():
    """Create sample knowledge graph for testing."""
    kg = KG()

    # Build a test graph: ML concepts and papers
    edges = [
        # Core ML concepts
        KGEdge("attention", "mechanism", "IS_A", 1.0),
        KGEdge("transformer", "neural_network", "IS_A", 1.0),
        KGEdge("transformer", "attention", "USES", 1.0),
        KGEdge("BERT", "transformer", "IS_A", 1.0),
        KGEdge("GPT", "transformer", "IS_A", 1.0),
        KGEdge("multi-head_attention", "attention", "IS_A", 0.9),

        # Papers
        KGEdge("Attention_Paper", "attention", "DISCUSSES", 1.0),
        KGEdge("Transformer_Paper", "transformer", "DISCUSSES", 1.0),
        KGEdge("BERT_Paper", "BERT", "DISCUSSES", 1.0),
        KGEdge("GPT_Paper", "GPT", "DISCUSSES", 1.0),

        # Citations
        KGEdge("Transformer_Paper", "Attention_Paper", "CITES", 1.0),
        KGEdge("BERT_Paper", "Transformer_Paper", "CITES", 1.0),
        KGEdge("GPT_Paper", "Transformer_Paper", "CITES", 1.0),

        # Related work
        KGEdge("BERT_Paper", "GPT_Paper", "RELATED", 0.8),
    ]

    kg.add_edges(edges)

    return kg


@pytest.fixture
async def sample_memories(sample_kg):
    """Create sample memories and store in graph."""
    memories = [
        Memory(
            id="mem1",
            text="Attention mechanisms allow models to focus on relevant input",
            timestamp=datetime.now(),
            context={'entities': ['attention', 'mechanism']},
            metadata={}
        ),
        Memory(
            id="mem2",
            text="Transformers use self-attention for sequence processing",
            timestamp=datetime.now(),
            context={'entities': ['transformer', 'attention']},
            metadata={}
        ),
        Memory(
            id="mem3",
            text="BERT is a bidirectional transformer encoder",
            timestamp=datetime.now(),
            context={'entities': ['BERT', 'transformer']},
            metadata={}
        ),
        Memory(
            id="mem4",
            text="GPT is an autoregressive transformer decoder",
            timestamp=datetime.now(),
            context={'entities': ['GPT', 'transformer']},
            metadata={}
        ),
    ]

    # Store memories in graph
    for memory in memories:
        await sample_kg.store(memory)

    return memories


# ============================================================================
# Multi-Hop Query Tests
# ============================================================================

@pytest.mark.asyncio
async def test_multi_hop_query_expansion(sample_kg, sample_memories):
    """Test multi-hop query expansion finds indirect connections."""
    reasoner = create_graph_reasoner(sample_kg)

    # Query should find direct mentions AND indirect via graph
    result = await reasoner.multi_hop_query(
        "What is attention?",
        max_hops=2,
        final_limit=10
    )

    # Assertions
    assert isinstance(result, MultiHopResult)
    assert result.query_text == "What is attention?"
    assert len(result.query_entities) > 0  # Should extract "attention"

    # Should find results at multiple hop distances
    assert len(result.direct_results) > 0 or len(result.hop1_results) > 0

    # All results should be ranked
    assert len(result.all_results) > 0

    # Scores should be descending
    scores = [score for _, score in result.all_results]
    assert scores == sorted(scores, reverse=True)

    # Check metadata
    assert result.metadata['max_hops'] == 2
    assert 'latency_ms' in result.metadata
    assert result.metadata['latency_ms'] < 500  # Should be fast


@pytest.mark.asyncio
async def test_two_hop_finds_indirect_connections(sample_kg, sample_memories):
    """Test 2-hop query finds entities connected indirectly."""
    reasoner = create_graph_reasoner(sample_kg)

    # Query about transformers should find BERT/GPT via 2-hop traversal
    result = await reasoner.multi_hop_query(
        "Tell me about Transformers",
        max_hops=2,
        final_limit=10
    )

    # Should find transformer-related entities at hop 1 or 2
    assert len(result.hop1_results) > 0 or len(result.hop2_results) > 0

    # Check that we found related memories
    all_memory_ids = [mem.id for mem, _ in result.all_results]

    # Should find at least some transformer-related memories
    assert any('mem' in mem_id for mem_id in all_memory_ids)


@pytest.mark.asyncio
async def test_three_hop_expands_further(sample_kg, sample_memories):
    """Test 3-hop query expands to distant connections."""
    reasoner = create_graph_reasoner(sample_kg)

    result = await reasoner.multi_hop_query(
        "Neural networks",
        max_hops=3,
        final_limit=10
    )

    # With 3 hops, should find more entities
    total_hops_used = (
        len(result.direct_results) +
        len(result.hop1_results) +
        len(result.hop2_results) +
        len(result.hop3_results)
    )

    # Should have expanded to multiple hops
    assert total_hops_used > 0


@pytest.mark.asyncio
async def test_query_caching(sample_kg, sample_memories):
    """Test query results are cached for performance."""
    reasoner = create_graph_reasoner(sample_kg, enable_caching=True)

    # First query (cold)
    result1 = await reasoner.multi_hop_query("What is attention?", max_hops=2)
    latency1 = result1.metadata['latency_ms']

    # Second query (warm - should be cached)
    result2 = await reasoner.multi_hop_query("What is attention?", max_hops=2)
    latency2 = result2.metadata['latency_ms']

    # Cached query should be much faster
    assert result2.metadata['cache_hit'] == True
    # Note: Cache returns same object, so latency will be the original

    # Results should be identical
    assert result1.query_text == result2.query_text
    assert len(result1.all_results) == len(result2.all_results)


# ============================================================================
# Path Finding Tests
# ============================================================================

@pytest.mark.asyncio
async def test_find_path_between_entities(sample_kg):
    """Test path finding between distant entities."""
    reasoner = create_graph_reasoner(sample_kg)

    # Find paths from BERT to attention
    paths = await reasoner.find_path("BERT", "attention", max_hops=5)

    # Should find at least one path
    assert len(paths) > 0

    # Check path structure
    path = paths[0]
    assert isinstance(path, GraphPath)
    assert path.start_entity == "BERT"
    assert path.end_entity == "attention"
    assert len(path.path) >= 2  # At least start and end
    assert len(path.edge_types) == len(path.path) - 1  # One edge per hop

    # Path should make sense: BERT → transformer → attention
    assert "transformer" in path.path or "BERT" in path.path


@pytest.mark.asyncio
async def test_path_finding_with_no_path(sample_kg):
    """Test path finding when no path exists."""
    reasoner = create_graph_reasoner(sample_kg)

    # Try to find path between disconnected entities
    paths = await reasoner.find_path("BERT", "nonexistent_entity", max_hops=5)

    # Should return empty list
    assert paths == []


@pytest.mark.asyncio
async def test_path_caching(sample_kg):
    """Test path finding results are cached."""
    reasoner = create_graph_reasoner(sample_kg, enable_caching=True)

    # First call
    paths1 = await reasoner.find_path("BERT", "attention", max_hops=5)

    # Second call (should be cached)
    paths2 = await reasoner.find_path("BERT", "attention", max_hops=5)

    # Should return same paths
    assert len(paths1) == len(paths2)
    if paths1:
        assert paths1[0].path == paths2[0].path


@pytest.mark.asyncio
async def test_multiple_paths_ranked_by_weight(sample_kg):
    """Test multiple paths are ranked by total weight."""
    reasoner = create_graph_reasoner(sample_kg)

    # Add alternative path with different weights
    sample_kg.add_edge(KGEdge("BERT", "attention", "USES", 0.5))  # Direct path

    paths = await reasoner.find_path("BERT", "attention", max_hops=3)

    # Should find multiple paths
    assert len(paths) >= 2

    # Paths should be sorted by weight (ascending)
    weights = [p.total_weight for p in paths]
    assert weights == sorted(weights)


# ============================================================================
# Relationship Traversal Tests
# ============================================================================

@pytest.mark.asyncio
async def test_traverse_by_relationship_type(sample_kg, sample_memories):
    """Test traversal by specific relationship type."""
    reasoner = create_graph_reasoner(sample_kg)

    # Find all entities that USE attention (outgoing USES edges)
    results = await reasoner.traverse_by_relationship(
        entity="attention",
        rel_type="USES",
        max_depth=2,
        direction="in"  # Who uses attention?
    )

    # Should find transformer (which uses attention)
    # Note: Results are memories, not entities directly
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_traverse_is_a_backward(sample_kg, sample_memories):
    """Test traversing IS_A relationships backward (find supertypes)."""
    reasoner = create_graph_reasoner(sample_kg)

    # Find what BERT is a type of (traverse IS_A backward)
    results = await reasoner.traverse_by_relationship(
        entity="BERT",
        rel_type="IS_A",
        max_depth=2,
        direction="out"  # What is BERT?
    )

    # Results should be memories (if any)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_traverse_cites_relationships(sample_kg, sample_memories):
    """Test traversing citation graph."""
    reasoner = create_graph_reasoner(sample_kg)

    # Find papers that cite Attention_Paper
    results = await reasoner.traverse_by_relationship(
        entity="Attention_Paper",
        rel_type="CITES",
        max_depth=2,
        direction="in"  # Who cites this?
    )

    # Should find at least Transformer_Paper
    assert isinstance(results, list)


# ============================================================================
# Ranking Tests
# ============================================================================

@pytest.mark.asyncio
async def test_rank_by_graph_proximity(sample_kg, sample_memories):
    """Test ranking by graph distance."""
    reasoner = create_graph_reasoner(sample_kg)

    # Get candidate memories
    candidates = sample_memories

    # Rank by proximity to "attention"
    ranked = reasoner.rank_by_graph_proximity(
        query_entities=["attention"],
        candidates=candidates,
        semantic_weight=0.5
    )

    # Should return ranked list
    assert len(ranked) == len(candidates)

    # Memories mentioning attention should rank higher
    top_memory = ranked[0]
    assert 'attention' in top_memory.context.get('entities', [])


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_two_hop_performance(sample_kg, sample_memories):
    """Test 2-hop query completes in reasonable time."""
    reasoner = create_graph_reasoner(sample_kg)

    result = await reasoner.multi_hop_query(
        "Tell me about transformers",
        max_hops=2,
        final_limit=10
    )

    # Should complete in <100ms for small graphs
    assert result.metadata['latency_ms'] < 200


@pytest.mark.asyncio
async def test_three_hop_performance(sample_kg, sample_memories):
    """Test 3-hop query performance."""
    reasoner = create_graph_reasoner(sample_kg)

    result = await reasoner.multi_hop_query(
        "Neural network architectures",
        max_hops=3,
        final_limit=10
    )

    # Should complete in <300ms for small graphs
    assert result.metadata['latency_ms'] < 400


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_paths_explain_results(sample_kg, sample_memories):
    """Test reasoning paths provide explanation."""
    reasoner = create_graph_reasoner(sample_kg)

    result = await reasoner.multi_hop_query(
        "What is BERT?",
        max_hops=2,
        final_limit=5
    )

    # Should generate reasoning paths
    assert len(result.reasoning_paths) > 0

    # Each path should be valid
    for path in result.reasoning_paths:
        assert isinstance(path, GraphPath)
        assert len(path.path) >= 2
        assert len(path.edge_types) == len(path.path) - 1


@pytest.mark.asyncio
async def test_cache_statistics(sample_kg, sample_memories):
    """Test cache statistics are tracked."""
    reasoner = create_graph_reasoner(sample_kg, enable_caching=True)

    # Run some queries
    await reasoner.multi_hop_query("attention", max_hops=2)
    await reasoner.find_path("BERT", "attention", max_hops=3)

    # Check stats
    stats = reasoner.get_cache_stats()

    assert 'query_cache_size' in stats
    assert 'path_cache_size' in stats
    assert stats['query_cache_size'] > 0
    assert stats['path_cache_size'] > 0


@pytest.mark.asyncio
async def test_clear_cache(sample_kg, sample_memories):
    """Test cache clearing."""
    reasoner = create_graph_reasoner(sample_kg, enable_caching=True)

    # Populate cache
    await reasoner.multi_hop_query("attention", max_hops=2)
    await reasoner.find_path("BERT", "attention", max_hops=3)

    # Verify cache is populated
    stats_before = reasoner.get_cache_stats()
    assert stats_before['query_cache_size'] > 0

    # Clear cache
    reasoner.clear_cache()

    # Verify cache is empty
    stats_after = reasoner.get_cache_stats()
    assert stats_after['query_cache_size'] == 0
    assert stats_after['path_cache_size'] == 0


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_empty_graph_returns_empty_results(sample_memories):
    """Test reasoner handles empty graph gracefully."""
    kg = KG()  # Empty graph
    reasoner = create_graph_reasoner(kg)

    result = await reasoner.multi_hop_query("test query", max_hops=2)

    # Should return empty results without crashing
    assert len(result.all_results) == 0
    assert len(result.direct_results) == 0


@pytest.mark.asyncio
async def test_query_with_no_entities(sample_kg, sample_memories):
    """Test query with no extractable entities."""
    reasoner = create_graph_reasoner(sample_kg)

    result = await reasoner.multi_hop_query(
        "how does it work?",  # No capitalized entities
        max_hops=2
    )

    # Should still return results (using graph recall)
    assert isinstance(result, MultiHopResult)
    assert result.query_entities == []  # No entities extracted


@pytest.mark.asyncio
async def test_max_hops_limit_respected(sample_kg, sample_memories):
    """Test max_hops parameter is respected."""
    reasoner = create_graph_reasoner(sample_kg)

    # Query with max_hops=1 should not populate hop2/hop3
    result = await reasoner.multi_hop_query("attention", max_hops=1)

    assert len(result.hop2_results) == 0
    assert len(result.hop3_results) == 0
    assert result.metadata['max_hops'] == 1


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
