"""
Tests for Streaming Context Builder (Phase 2)
==============================================

Validates progressive context expansion with async iteration.

Test Coverage:
- Unit tests: ContextChunk, ChunkYieldStrategy
- Streaming tests: Basic streaming, hop boundaries, token thresholds
- Integration tests: With adaptive expansion, real graphs
- Edge cases: Empty graphs, zero budget, interruption
- Performance tests: Latency, throughput

Author: Claude Code
Date: 2025-11-24
"""

import pytest
import asyncio
from typing import Dict, List

from hololoom.memory.streaming_expansion import (
    StreamingContextBuilder,
    ContextChunk,
    ChunkYieldStrategy,
    stream_context_expansion
)
from hololoom.memory.graph import KG, KGEdge


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_graph() -> KG:
    """Create simple test graph: A → B → C → D."""
    kg = KG()
    kg.add_edges([
        KGEdge("A", "B", "LEADS_TO", 1.0),
        KGEdge("B", "C", "LEADS_TO", 1.0),
        KGEdge("C", "D", "LEADS_TO", 1.0),
    ])
    return kg


@pytest.fixture
def branching_graph() -> KG:
    """
    Create branching graph:
         A
        /|\
       B C D
       |   |
       E   F
    """
    kg = KG()
    kg.add_edges([
        KGEdge("A", "B", "LEADS_TO", 1.0),
        KGEdge("A", "C", "LEADS_TO", 1.0),
        KGEdge("A", "D", "LEADS_TO", 1.0),
        KGEdge("B", "E", "LEADS_TO", 1.0),
        KGEdge("D", "F", "LEADS_TO", 1.0),
    ])
    return kg


@pytest.fixture
def importance_scores() -> Dict[str, float]:
    """Realistic importance scores."""
    return {
        "A": 1.0,
        "B": 0.9,
        "C": 0.8,
        "D": 0.7,
        "E": 0.6,
        "F": 0.5,
    }


@pytest.fixture
def node_contents() -> Dict[str, str]:
    """Node content for semantic scoring."""
    return {
        "A": "Node A is the root concept with high importance",
        "B": "Node B is a related concept",
        "C": "Node C is another related concept",
        "D": "Node D is a downstream concept",
        "E": "Node E is a leaf concept",
        "F": "Node F is another leaf concept",
    }


# ============================================================================
# Unit Tests: ContextChunk
# ============================================================================

def test_context_chunk_creation():
    """Test ContextChunk creation and properties."""
    chunk = ContextChunk(
        nodes=["A", "B", "C"],
        contents={"A": "Content A", "B": "Content B"},
        relevance_scores={"A": 1.0, "B": 0.9, "C": 0.8},
        hop_distance=1,
        token_count=300,
        cumulative_tokens=500,
        chunk_index=2,
        is_final=False
    )

    assert chunk.node_count == 3
    assert chunk.avg_relevance == pytest.approx((1.0 + 0.9 + 0.8) / 3)
    assert chunk.token_count == 300
    assert chunk.cumulative_tokens == 500
    assert not chunk.is_final


def test_context_chunk_empty_relevance():
    """Test ContextChunk with no relevance scores."""
    chunk = ContextChunk(
        nodes=["A"],
        contents={},
        relevance_scores={},
        hop_distance=0,
        token_count=100,
        cumulative_tokens=100,
        chunk_index=0,
        is_final=True
    )

    assert chunk.avg_relevance == 0.0


# ============================================================================
# Streaming Tests: Basic Functionality
# ============================================================================

@pytest.mark.asyncio
async def test_basic_streaming(simple_graph, importance_scores, node_contents):
    """Test basic streaming expansion."""
    builder = StreamingContextBuilder()

    chunks = []
    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=1000,
        chunk_size=200,
        min_relevance=0.2,
        max_hops=3,
        importance_scores=importance_scores,
        node_contents=node_contents
    ):
        chunks.append(chunk)

    # Should yield at least seed chunk
    assert len(chunks) > 0

    # First chunk should be seed nodes (hop 0)
    assert chunks[0].hop_distance == 0
    assert "A" in chunks[0].nodes

    # Last chunk should be marked final
    assert chunks[-1].is_final

    # Cumulative tokens should increase
    for i in range(1, len(chunks)):
        assert chunks[i].cumulative_tokens >= chunks[i-1].cumulative_tokens


@pytest.mark.asyncio
async def test_streaming_hop_boundaries(branching_graph, importance_scores, node_contents):
    """Test that chunks are yielded at hop boundaries."""
    builder = StreamingContextBuilder()

    chunks = []
    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=branching_graph,
        token_budget=2000,
        chunk_size=1000,  # Large chunk size (won't hit token threshold)
        min_relevance=0.2,
        max_hops=3,
        importance_scores=importance_scores,
        node_contents=node_contents,
        yield_strategy=ChunkYieldStrategy.HOP_BOUNDARY
    ):
        chunks.append(chunk)

    # Should yield at each hop
    hop_distances = [c.hop_distance for c in chunks]

    # Hop 0 (seed) should be present
    assert 0 in hop_distances

    # Should have multiple hops
    assert len(set(hop_distances)) > 1


@pytest.mark.asyncio
async def test_streaming_token_threshold(simple_graph, importance_scores, node_contents):
    """Test that chunks are yielded when token threshold reached."""
    builder = StreamingContextBuilder()

    chunk_size = 150
    chunks = []

    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=1000,
        chunk_size=chunk_size,
        min_relevance=0.2,
        max_hops=3,
        importance_scores=importance_scores,
        node_contents=node_contents,
        yield_strategy=ChunkYieldStrategy.TOKEN_THRESHOLD
    ):
        chunks.append(chunk)

    # Most chunks should be close to chunk_size (except final)
    for chunk in chunks[:-1]:
        # Allow some variance (±50 tokens)
        # Some chunks may be smaller if hop boundary reached first
        assert chunk.token_count <= chunk_size + 100


@pytest.mark.asyncio
async def test_streaming_hybrid_strategy(branching_graph, importance_scores, node_contents):
    """Test hybrid yield strategy (both token and hop boundaries)."""
    builder = StreamingContextBuilder()

    chunks = []
    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=branching_graph,
        token_budget=2000,
        chunk_size=200,
        min_relevance=0.2,
        max_hops=3,
        importance_scores=importance_scores,
        node_contents=node_contents,
        yield_strategy=ChunkYieldStrategy.HYBRID
    ):
        chunks.append(chunk)

    # Should yield multiple chunks (both conditions)
    assert len(chunks) > 1

    # Check yield reasons in metadata
    yield_reasons = [c.metadata.get("yield_reason", "") for c in chunks]
    # Should have mix of reasons (seed, token, hop, final)
    unique_reasons = set(yield_reasons)
    assert len(unique_reasons) > 1


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_streaming_respects_budget(simple_graph, importance_scores, node_contents):
    """Test that streaming respects token budget."""
    builder = StreamingContextBuilder()

    token_budget = 250
    total_tokens = 0

    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=token_budget,
        chunk_size=100,
        min_relevance=0.2,
        max_hops=5,
        importance_scores=importance_scores,
        node_contents=node_contents
    ):
        total_tokens = chunk.cumulative_tokens

    # Should not exceed budget
    assert total_tokens <= token_budget


@pytest.mark.asyncio
async def test_streaming_stops_on_low_relevance(simple_graph, importance_scores, node_contents):
    """Test that streaming stops when relevance too low."""
    builder = StreamingContextBuilder()

    chunks = []
    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=2000,
        chunk_size=200,
        min_relevance=0.95,  # Very high threshold
        max_hops=5,
        importance_scores=importance_scores,
        node_contents=node_contents
    ):
        chunks.append(chunk)

    # Should stop early due to low relevance
    # Only seed nodes should have relevance ≥ 0.95
    result = builder.get_last_result()
    assert result.total_nodes <= 2  # Seed + maybe 1 high-relevance neighbor


@pytest.mark.asyncio
async def test_streaming_progressive_nodes(branching_graph, importance_scores, node_contents):
    """Test that nodes are discovered progressively (not all at once)."""
    builder = StreamingContextBuilder()

    discovered_nodes = []
    chunks = []

    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=branching_graph,
        token_budget=2000,
        chunk_size=200,
        min_relevance=0.2,
        max_hops=3,
        importance_scores=importance_scores,
        node_contents=node_contents
    ):
        discovered_nodes.extend(chunk.nodes)
        chunks.append(chunk)

    # Should discover nodes progressively
    # First chunk should have seed only
    assert chunks[0].nodes == ["A"]

    # Later chunks should have more nodes
    if len(chunks) > 1:
        assert len(chunks[1].nodes) > 0


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_streaming_empty_graph():
    """Test streaming with empty graph."""
    kg = KG()
    builder = StreamingContextBuilder()

    chunks = []
    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=kg,
        token_budget=1000,
        chunk_size=200
    ):
        chunks.append(chunk)

    # Should yield seed chunk + empty final chunk (EOF marker)
    assert len(chunks) >= 1
    assert chunks[0].nodes == ["A"]
    assert chunks[-1].is_final  # Last chunk is always final


@pytest.mark.asyncio
async def test_streaming_zero_budget():
    """Test streaming with zero budget."""
    kg = KG()
    kg.add_edges([KGEdge("A", "B", "LEADS_TO", 1.0)])
    builder = StreamingContextBuilder()

    chunks = []
    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=kg,
        token_budget=0,  # Zero budget
        chunk_size=200
    ):
        chunks.append(chunk)

    # Should yield empty final chunk (EOF marker, even with zero budget)
    assert len(chunks) == 1
    assert chunks[0].nodes == []
    assert chunks[0].is_final


@pytest.mark.asyncio
async def test_streaming_nonexistent_seed():
    """Test streaming with nonexistent seed node."""
    kg = KG()
    kg.add_edges([KGEdge("A", "B", "LEADS_TO", 1.0)])
    builder = StreamingContextBuilder()

    chunks = []
    async for chunk in builder.stream_expansion(
        query="What is Z?",
        seed_nodes=["Z"],  # Doesn't exist
        graph=kg,
        token_budget=1000,
        chunk_size=200
    ):
        chunks.append(chunk)

    # Should yield seed chunk + empty final chunk (EOF marker)
    assert len(chunks) >= 1
    assert chunks[0].nodes == ["Z"]
    assert chunks[-1].is_final  # Last chunk is always final


@pytest.mark.asyncio
async def test_streaming_interruption(branching_graph, importance_scores, node_contents):
    """Test that streaming can be interrupted mid-expansion."""
    builder = StreamingContextBuilder()

    chunks = []
    chunk_limit = 2

    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=branching_graph,
        token_budget=2000,
        chunk_size=100,
        min_relevance=0.2,
        max_hops=5,
        importance_scores=importance_scores,
        node_contents=node_contents
    ):
        chunks.append(chunk)

        if len(chunks) >= chunk_limit:
            break  # Interrupt expansion

    # Should have stopped early
    assert len(chunks) == chunk_limit

    # Last chunk should NOT be marked final (interrupted)
    assert not chunks[-1].is_final


# ============================================================================
# Convenience Function Tests
# ============================================================================

@pytest.mark.asyncio
async def test_convenience_function(simple_graph, importance_scores, node_contents):
    """Test convenience function stream_context_expansion()."""
    chunks = []

    async for chunk in stream_context_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=1000,
        chunk_size=200,
        importance_scores=importance_scores,
        node_contents=node_contents
    ):
        chunks.append(chunk)

    # Should work identically to StreamingContextBuilder
    assert len(chunks) > 0
    assert chunks[0].nodes == ["A"]
    assert chunks[-1].is_final


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_streaming_latency_to_first_chunk(branching_graph, importance_scores, node_contents):
    """Test that first chunk arrives quickly (low latency)."""
    import time

    builder = StreamingContextBuilder()

    start_time = time.time()
    first_chunk_time = None

    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=branching_graph,
        token_budget=2000,
        chunk_size=200,
        importance_scores=importance_scores,
        node_contents=node_contents
    ):
        if first_chunk_time is None:
            first_chunk_time = time.time() - start_time
        break  # Get first chunk only

    # First chunk should arrive in <10ms
    assert first_chunk_time < 0.01  # 10ms


@pytest.mark.asyncio
async def test_streaming_result_summary(branching_graph, importance_scores, node_contents):
    """Test that result summary is accurate."""
    builder = StreamingContextBuilder()

    chunks = []
    async for chunk in builder.stream_expansion(
        query="What is A?",
        seed_nodes=["A"],
        graph=branching_graph,
        token_budget=2000,
        chunk_size=200,
        importance_scores=importance_scores,
        node_contents=node_contents
    ):
        chunks.append(chunk)

    result = builder.get_last_result()

    # Check summary matches actual chunks
    assert result.total_chunks == len(chunks)
    assert result.total_tokens == chunks[-1].cumulative_tokens
    assert result.total_nodes == sum(len(c.nodes) for c in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
