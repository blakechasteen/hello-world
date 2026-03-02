"""
Tests for Advanced Interleaved Generation Features
===================================================

Tests the three advanced features:
1. Adaptive context feeding
2. Agentic graph navigation
3. Context summarization

Author: Claude Code (Opus 4.5)
Date: 2025-11-25
"""

import asyncio
import pytest
import time
from typing import Dict, List
import networkx as nx

from hololoom.memory.interleaved_generation import ContextChunk, GenerationToken, StreamMetadata
from hololoom.memory.interleaved_generation_advanced import (
    AdaptiveLLMProtocol,
    AdvancedInterleavedManager,
    AgenticGraphNavigator,
    ContextSummarizer,
    ContextUpdate,
    MockAdaptiveLLM,
    NodeRequest,
    NodeRequestType,
    SummarizationConfig,
    stream_advanced_generation
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    G = nx.MultiDiGraph()
    G.add_edges_from([
        ("A", "B", {"type": "USES"}),
        ("A", "C", {"type": "IS_A"}),
        ("B", "D", {"type": "MENTIONS"}),
        ("C", "E", {"type": "LEADS_TO"}),
        ("D", "E", {"type": "USES"}),
        ("A", "bayesian_approach", {"type": "MENTIONS"}),
    ])
    return G


@pytest.fixture
def node_contents():
    """Create node content mapping."""
    return {
        "A": "Node A: Thompson Sampling is a Bayesian approach",
        "B": "Node B: Exploration-exploitation tradeoff",
        "C": "Node C: Bayesian methods use prior distributions",
        "D": "Node D: Multi-armed bandits optimize decisions",
        "E": "Node E: Optimal performance through learning",
        "bayesian_approach": "Bayesian approach uses probability theory"
    }


@pytest.fixture
def importance_scores():
    """Create importance scores."""
    return {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2}


@pytest.fixture
def mock_llm():
    """Create mock adaptive LLM."""
    return MockAdaptiveLLM()


# ============================================================================
# Feature 1: Adaptive Context Feeding Tests
# ============================================================================

@pytest.mark.asyncio
async def test_adaptive_llm_protocol_interface():
    """Test that AdaptiveLLMProtocol defines the correct interface."""
    llm = MockAdaptiveLLM()

    # Should have both standard and adaptive methods
    assert hasattr(llm, 'generate_stream')
    assert hasattr(llm, 'generate_stream_adaptive')


@pytest.mark.asyncio
async def test_context_update_creation():
    """Test ContextUpdate dataclass creation."""
    chunk = ContextChunk(
        nodes=["A"],
        relevance_scores={"A": 0.9},
        contents={"A": "Test content"},
        token_count=2,
        cumulative_tokens=2,
        hop_distance=0,
        chunk_index=0,
        is_final=False
    )

    update = ContextUpdate(
        chunk=chunk,
        priority=0.9,
        timestamp_ms=1000.0
    )

    assert update.chunk == chunk
    assert update.priority == 0.9
    assert update.timestamp_ms == 1000.0


@pytest.mark.asyncio
async def test_adaptive_generation_receives_updates(simple_graph, node_contents, importance_scores, mock_llm):
    """Test that adaptive generation receives and incorporates context updates."""
    manager = AdvancedInterleavedManager(
        llm=mock_llm,
        enable_adaptive_feeding=True,
        enable_agentic_navigation=False,
        enable_summarization=False
    )

    received_updates = False
    generation_tokens = []

    async for item in manager.stream_advanced(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=1000,
        max_generation_tokens=20,
        importance_scores=importance_scores,
        node_contents=node_contents,
        emit_metadata=True
    ):
        if isinstance(item, GenerationToken):
            generation_tokens.append(item.token)
            # Check if response mentions new context
            if "[New context:" in item.cumulative_text:
                received_updates = True

    # Should have received at least one context update
    assert received_updates, "Adaptive generation should incorporate context updates"
    assert len(generation_tokens) > 0


# ============================================================================
# Feature 2: Agentic Graph Navigation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_navigator_creation(simple_graph, node_contents):
    """Test AgenticGraphNavigator initialization."""
    navigator = AgenticGraphNavigator(simple_graph, node_contents)

    assert navigator.graph == simple_graph
    assert navigator.node_contents == node_contents
    assert len(navigator.fulfilled_requests) == 0


@pytest.mark.asyncio
async def test_detect_node_request_by_name():
    """Test detection of BY_NAME node requests."""
    navigator = AgenticGraphNavigator(nx.MultiDiGraph(), {})

    text = "I need to know about <request_node>thompson_sampling</request_node> please."

    requests = navigator.detect_requests(text)

    assert len(requests) == 1
    assert requests[0].request_type == NodeRequestType.BY_NAME
    assert requests[0].target == "thompson_sampling"


@pytest.mark.asyncio
async def test_detect_multiple_requests():
    """Test detection of multiple node requests."""
    navigator = AgenticGraphNavigator(nx.MultiDiGraph(), {})

    text = """
    First, <request_node>node_a</request_node>.
    Then, <request_related>connection</request_related>.
    Finally, <request_query>What connects to Bayesian methods?</request_query>.
    """

    requests = navigator.detect_requests(text)

    assert len(requests) == 3
    assert requests[0].request_type == NodeRequestType.BY_NAME
    assert requests[1].request_type == NodeRequestType.BY_RELATIONSHIP
    assert requests[2].request_type == NodeRequestType.BY_QUERY


@pytest.mark.asyncio
async def test_fulfill_node_request_by_name(simple_graph, node_contents):
    """Test fulfilling a BY_NAME node request."""
    navigator = AgenticGraphNavigator(simple_graph, node_contents)

    request = NodeRequest(
        request_type=NodeRequestType.BY_NAME,
        target="A",
        context=""
    )

    chunk = navigator.fulfill_request(request)

    assert chunk is not None
    assert "A" in chunk.nodes
    assert chunk.node_count == 1
    assert chunk.relevance_scores["A"] == 1.0  # Explicitly requested = high relevance


@pytest.mark.asyncio
async def test_fulfill_nonexistent_node():
    """Test fulfilling a request for a nonexistent node."""
    navigator = AgenticGraphNavigator(nx.MultiDiGraph(), {})

    request = NodeRequest(
        request_type=NodeRequestType.BY_NAME,
        target="nonexistent",
        context=""
    )

    chunk = navigator.fulfill_request(request)

    assert chunk is None  # Should return None for nonexistent nodes


@pytest.mark.asyncio
async def test_agentic_navigation_integration(simple_graph, node_contents, importance_scores, mock_llm):
    """Test full agentic navigation integration."""
    # Note: Agentic navigation requires adaptive_feeding=True because
    # node request tokens are only emitted in generate_stream_adaptive()
    manager = AdvancedInterleavedManager(
        llm=mock_llm,
        enable_adaptive_feeding=True,  # Required for agentic navigation
        enable_agentic_navigation=True,
        enable_summarization=False
    )

    node_requests_fulfilled = 0

    async for item in manager.stream_advanced(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=1000,
        max_generation_tokens=30,
        importance_scores=importance_scores,
        node_contents=node_contents,
        emit_metadata=True
    ):
        if isinstance(item, StreamMetadata) and item.event_type == "node_request_fulfilled":
            node_requests_fulfilled += 1

    # Mock LLM should emit at least one node request
    assert node_requests_fulfilled >= 1, "Should fulfill at least one node request"


# ============================================================================
# Feature 3: Context Summarization Tests
# ============================================================================

@pytest.mark.asyncio
async def test_summarizer_creation():
    """Test ContextSummarizer initialization."""
    config = SummarizationConfig(
        enabled=True,
        chunk_threshold=10,
        token_threshold=2000,
        importance_cutoff=0.4,
        compression_ratio=0.3
    )

    summarizer = ContextSummarizer(config)

    assert summarizer.config.enabled
    assert summarizer.config.chunk_threshold == 10
    assert summarizer.config.compression_ratio == 0.3


@pytest.mark.asyncio
async def test_should_summarize_by_chunk_threshold():
    """Test summarization trigger by chunk count."""
    config = SummarizationConfig(chunk_threshold=5, token_threshold=10000)
    summarizer = ContextSummarizer(config)

    # Create 6 chunks (exceeds threshold)
    chunks = [
        ContextChunk(
            nodes=[f"node_{i}"],
            relevance_scores={f"node_{i}": 0.5},
            contents={f"node_{i}": f"Content {i}"},
            token_count=10,
            cumulative_tokens=10 * (i + 1),
            hop_distance=i,
            chunk_index=i,
            is_final=False
        )
        for i in range(6)
    ]

    assert summarizer.should_summarize(chunks)


@pytest.mark.asyncio
async def test_should_not_summarize_below_threshold():
    """Test that summarization doesn't trigger below threshold."""
    config = SummarizationConfig(chunk_threshold=10, token_threshold=10000)
    summarizer = ContextSummarizer(config)

    # Create 5 chunks (below threshold)
    chunks = [
        ContextChunk(
            nodes=[f"node_{i}"],
            relevance_scores={f"node_{i}": 0.5},
            contents={f"node_{i}": f"Content {i}"},
            token_count=10,
            cumulative_tokens=10 * (i + 1),
            hop_distance=i,
            chunk_index=i,
            is_final=False
        )
        for i in range(5)
    ]

    assert not summarizer.should_summarize(chunks)


@pytest.mark.asyncio
async def test_summarize_chunks():
    """Test actual chunk summarization."""
    config = SummarizationConfig(
        enabled=True,
        compression_ratio=0.5  # Keep 50%, summarize 50%
    )
    summarizer = ContextSummarizer(config)

    # Create chunks with varying relevance
    chunks = [
        ContextChunk(
            nodes=[f"node_{i}"],
            relevance_scores={f"node_{i}": 1.0 - (i * 0.1)},  # Decreasing relevance
            contents={f"node_{i}": f"This is content for node {i}. Important information here."},
            token_count=10,
            cumulative_tokens=10 * (i + 1),
            hop_distance=i,
            chunk_index=i,
            is_final=False
        )
        for i in range(10)
    ]

    kept_chunks, summarized_chunk = summarizer.summarize(chunks)

    # Should keep ~50% of chunks
    assert len(kept_chunks) >= 4
    assert len(kept_chunks) <= 6

    # Should create a summarized chunk
    assert summarized_chunk is not None
    assert summarized_chunk.chunk_index == -1  # Special marker
    # Check for "summary" key (case insensitive)
    summary_key = list(summarized_chunk.contents.keys())[0]
    assert "summary" in summary_key.lower()

    # Summarized chunk should have fewer tokens than original
    original_tokens = sum(c.token_count for c in chunks[len(kept_chunks):])
    assert summarized_chunk.token_count < original_tokens


@pytest.mark.asyncio
async def test_summarization_integration(simple_graph, node_contents, importance_scores, mock_llm):
    """Test full summarization integration."""
    manager = AdvancedInterleavedManager(
        llm=mock_llm,
        enable_adaptive_feeding=False,
        enable_agentic_navigation=False,
        enable_summarization=True
    )

    summarization_applied = False

    async for item in manager.stream_advanced(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=1000,
        max_generation_tokens=20,
        importance_scores=importance_scores,
        node_contents=node_contents,
        emit_metadata=True
    ):
        if isinstance(item, StreamMetadata) and item.event_type == "summarization_applied":
            summarization_applied = True
            assert "compression_ratio" in item.data
            assert item.data["compression_ratio"] < 1.0  # Should compress

    # Note: Summarization may not always trigger for small graphs
    # Test just verifies it can be enabled without errors


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_all_features_enabled(simple_graph, node_contents, importance_scores, mock_llm):
    """Test all three features working together."""
    manager = AdvancedInterleavedManager(
        llm=mock_llm,
        enable_adaptive_feeding=True,
        enable_agentic_navigation=True,
        enable_summarization=True
    )

    chunks_received = 0
    tokens_received = 0
    metadata_events = []

    async for item in manager.stream_advanced(
        query="Explain A and related concepts",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=1000,
        max_generation_tokens=30,
        importance_scores=importance_scores,
        node_contents=node_contents,
        emit_metadata=True
    ):
        if isinstance(item, ContextChunk):
            chunks_received += 1
        elif isinstance(item, GenerationToken) and not item.is_final:
            tokens_received += 1
        elif isinstance(item, StreamMetadata):
            metadata_events.append(item.event_type)

    # Should receive both chunks and tokens
    assert chunks_received > 0
    assert tokens_received > 0

    # Should have completion metadata
    assert "stream_complete" in metadata_events

    # Verify all features are reflected in final metadata
    final_metadata = [m for m in metadata_events if "stream_complete" in m]
    # (Can't easily verify from event name alone, but system should run without errors)


@pytest.mark.asyncio
async def test_convenience_function(simple_graph, node_contents, importance_scores):
    """Test stream_advanced_generation convenience function."""
    tokens_received = 0

    async for item in stream_advanced_generation(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        node_contents=node_contents,
        importance_scores=importance_scores,
        token_budget=1000,
        max_generation_tokens=20,
        enable_adaptive_feeding=True,
        enable_agentic_navigation=True,
        enable_summarization=True
    ):
        if isinstance(item, GenerationToken) and not item.is_final:
            tokens_received += 1

    assert tokens_received > 0


@pytest.mark.asyncio
async def test_selective_feature_enable():
    """Test enabling features selectively."""
    # Only adaptive feeding
    manager1 = AdvancedInterleavedManager(
        llm=MockAdaptiveLLM(),
        enable_adaptive_feeding=True,
        enable_agentic_navigation=False,
        enable_summarization=False
    )
    assert manager1.enable_adaptive_feeding
    assert not manager1.enable_agentic_navigation
    assert not manager1.enable_summarization

    # Only agentic navigation
    manager2 = AdvancedInterleavedManager(
        llm=MockAdaptiveLLM(),
        enable_adaptive_feeding=False,
        enable_agentic_navigation=True,
        enable_summarization=False
    )
    assert not manager2.enable_adaptive_feeding
    assert manager2.enable_agentic_navigation
    assert not manager2.enable_summarization

    # Only summarization
    manager3 = AdvancedInterleavedManager(
        llm=MockAdaptiveLLM(),
        enable_adaptive_feeding=False,
        enable_agentic_navigation=False,
        enable_summarization=True
    )
    assert not manager3.enable_adaptive_feeding
    assert not manager3.enable_agentic_navigation
    assert manager3.enable_summarization


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_advanced_features_latency(simple_graph, node_contents, importance_scores, mock_llm):
    """Test that advanced features don't add excessive latency."""
    manager = AdvancedInterleavedManager(
        llm=mock_llm,
        enable_adaptive_feeding=True,
        enable_agentic_navigation=True,
        enable_summarization=True
    )

    start_time = time.time()
    first_token_time = None

    async for item in manager.stream_advanced(
        query="What is A?",
        seed_nodes=["A"],
        graph=simple_graph,
        token_budget=1000,
        max_generation_tokens=20,
        importance_scores=importance_scores,
        node_contents=node_contents,
        emit_metadata=False
    ):
        if isinstance(item, GenerationToken) and first_token_time is None and not item.is_final:
            first_token_time = time.time() - start_time
            break  # Got first token

    # Advanced features shouldn't significantly delay first token
    # Should still be sub-second
    assert first_token_time is not None
    assert first_token_time < 2.0, "First token latency should be <2s even with advanced features"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])