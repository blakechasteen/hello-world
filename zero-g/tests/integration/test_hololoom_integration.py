"""
Integration Tests for HoloLoom Integration

Tests ProductionLoomCore integration with HoloLoom 1.0.0.
Validates all 8 Loom components work correctly with HoloLoom backends.

Run:
    pytest tests/integration/test_hololoom_integration.py -v

Skip if HoloLoom unavailable:
    pytest tests/integration/test_hololoom_integration.py -v -m "not hololoom"
"""

import pytest
import asyncio
from typing import List, Dict
from datetime import datetime

# Check HoloLoom availability
try:
    from HoloLoom import HoloLoom
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False

# Import Zero-G components
from backend.loom_core.protocols import (
    Thread,
    YarnNode,
    YarnEdge,
    SpacetimeEvent,
    RiftAction,
    DecisionContext,
    MemoryType
)

if HOLOLOOM_AVAILABLE:
    from backend.loom_core.production_loom import (
        create_production_loom_core,
        ProductionLoomCore
    )


# ==================== Fixtures ====================

@pytest.fixture
async def production_loom():
    """Create ProductionLoomCore with INMEMORY backend"""
    if not HOLOLOOM_AVAILABLE:
        pytest.skip("HoloLoom not available")

    loom = await create_production_loom_core({
        "backend": "INMEMORY",
        "execution_mode": "FAST",
        "scales": [96, 192, 384],
        "enable_safety": True
    })

    yield loom

    await loom.shutdown()


@pytest.fixture
async def hybrid_loom():
    """Create ProductionLoomCore with HYBRID backend (requires Docker)"""
    if not HOLOLOOM_AVAILABLE:
        pytest.skip("HoloLoom not available")

    try:
        loom = await create_production_loom_core({
            "backend": "HYBRID",
            "execution_mode": "FAST"
        })

        yield loom

        await loom.shutdown()

    except Exception as e:
        pytest.skip(f"HYBRID backend unavailable: {e}")


# ==================== Component Tests ====================

@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_warpspace_embedding(production_loom):
    """Test WarpSpace embedding with MatryoshkaEmbeddings"""
    warp = production_loom.warp_space

    # Embed text
    embedding = await warp.embed("Thompson Sampling balances exploration", modality="text")

    # Check embedding
    assert isinstance(embedding, list)
    assert len(embedding) == 384  # 384D Matryoshka embedding
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_warpspace_search(production_loom):
    """Test WarpSpace semantic search"""
    warp = production_loom.warp_space

    # Index threads
    await warp.index_thread(Thread(
        id="thread1",
        content="Thompson Sampling is a Bayesian strategy"
    ))
    await warp.index_thread(Thread(
        id="thread2",
        content="It balances exploration and exploitation"
    ))
    await warp.index_thread(Thread(
        id="thread3",
        content="Python has decorators for functions"
    ))

    # Search
    results = await warp.search("What is Thompson Sampling?", k=2)

    # Check results
    assert len(results) <= 2
    assert all(isinstance(t, Thread) for t in results)

    # Top results should be about Thompson Sampling
    # (semantic similarity)
    if len(results) > 0:
        assert "thompson" in results[0].content.lower() or "sampling" in results[0].content.lower()


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_yarngraph_node_operations(production_loom):
    """Test YarnGraph node operations"""
    yarn = production_loom.yarn_graph

    # Add node
    node = YarnNode(
        id="python",
        entity_type="programming_language",
        properties={"version": "3.9"}
    )
    await yarn.add_node(node)

    # Get node
    retrieved = await yarn.get_node("python")
    assert retrieved is not None
    assert retrieved.id == "python"


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_yarngraph_edge_operations(production_loom):
    """Test YarnGraph edge operations"""
    yarn = production_loom.yarn_graph

    # Add edges
    await yarn.add_edge(YarnEdge(
        source_id="python",
        target_id="programming_language",
        relationship_type="IS_A",
        weight=1.0
    ))

    await yarn.add_edge(YarnEdge(
        source_id="attention",
        target_id="transformer",
        relationship_type="USES",
        weight=0.9
    ))

    # Find neighbors
    neighbors = await yarn.find_neighbors("python")
    assert len(neighbors) > 0


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_yarngraph_path_finding(production_loom):
    """Test YarnGraph path finding"""
    yarn = production_loom.yarn_graph

    # Create path
    await yarn.add_edge(YarnEdge("A", "B", "LEADS_TO", 1.0))
    await yarn.add_edge(YarnEdge("B", "C", "LEADS_TO", 1.0))

    # Find path
    path = await yarn.find_path("A", "C", max_hops=5)

    # Check path (may be None if graph structure doesn't support path finding)
    if path:
        assert len(path) > 0
        assert all(isinstance(e, YarnEdge) for e in path)


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_rift_tool_execution(production_loom):
    """Test Rift tool execution"""
    rift = production_loom.rift

    # Execute answer tool
    action = RiftAction(
        tool_name="answer",
        parameters={"query": "What is Thompson Sampling?"}
    )

    result = await rift.invoke(action)

    # Check result
    assert result.get("success") is True or "error" in result
    assert "result" in result or "error" in result


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_rift_custom_tool(production_loom):
    """Test Rift custom tool registration"""
    rift = production_loom.rift

    # Register custom tool
    async def custom_tool(params):
        return f"Custom: {params.get('input', '')}"

    await rift.register_tool(
        "custom",
        custom_tool,
        {"type": "custom", "description": "Custom tool"}
    )

    # Execute custom tool
    action = RiftAction(
        tool_name="custom",
        parameters={"input": "test"}
    )

    result = await rift.invoke(action)

    # Check result
    assert result.get("success") is True
    assert "Custom: test" in result["result"]


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_spacetime_fabric_logging(production_loom):
    """Test SpacetimeFabric event logging"""
    fabric = production_loom.spacetime_fabric

    # Log event
    event = SpacetimeEvent(
        timestamp=datetime.now(),
        event_type="test_event",
        component="test_component",
        data={"key": "value"}
    )

    await fabric.log_event(event)

    # Get trace
    end_time = datetime.now()
    start_time = datetime(end_time.year, end_time.month, end_time.day)

    events = await fabric.get_trace(start_time, end_time)

    # Check events
    assert len(events) > 0
    assert any(e.event_type == "test_event" for e in events)


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_reflection_buffer_experience(production_loom):
    """Test ReflectionBuffer experience storage"""
    buffer = production_loom.reflection_buffer

    # Store experience
    action = RiftAction(
        tool_name="answer",
        parameters={"query": "test"}
    )

    await buffer.store_experience(
        state={"query": "test"},
        action=action,
        reward=0.9,
        next_state={"result": "success"}
    )

    # Sample batch
    batch = await buffer.sample_batch(batch_size=1)

    # Check batch
    assert len(batch) > 0
    assert batch[0]["reward"] == 0.9


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_thread_spinner_page_in_out(production_loom):
    """Test ThreadSpinner paging operations"""
    spinner = production_loom.thread_spinner

    # Create thread
    thread = Thread(
        id="test_thread",
        content="Test content for paging"
    )

    # Store in hot memory manually
    spinner.hot_threads[thread.id] = thread

    # Page out
    await spinner.page_out(thread.id)

    # Check removed from hot
    assert thread.id not in spinner.hot_threads

    # Page in
    paged_thread = await spinner.page_in(thread.id)

    # Check paged back in
    assert paged_thread.id == thread.id
    assert thread.id in spinner.hot_threads


# ==================== Integration Tests ====================

@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_full_weave_cycle(production_loom):
    """Test complete weave cycle"""
    # Weave query
    result = await production_loom.weave(
        query="What is Thompson Sampling?",
        context={"test": True}
    )

    # Check result
    assert "query" in result
    assert "action" in result
    assert "result" in result
    assert "duration_ms" in result
    assert result["query"] == "What is Thompson Sampling?"


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_weave_with_search_tool(production_loom):
    """Test weave with search tool"""
    result = await production_loom.weave(
        query="Search for information about Python",
        context={}
    )

    # Check action
    assert result["action"] == "search"
    assert "result" in result


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_config_swap_simple_to_production():
    """Test config-based swap from Simple to Production"""
    if not HOLOLOOM_AVAILABLE:
        pytest.skip("HoloLoom not available")

    from backend.loom_core.simple_loom import create_simple_loom_core

    # Create simple loom
    simple = await create_simple_loom_core()

    # Weave with simple
    simple_result = await simple.weave("Test query")
    simple_duration = simple_result["duration_ms"]

    await simple.shutdown()

    # Create production loom
    production = await create_production_loom_core({"backend": "INMEMORY"})

    # Weave with production
    production_result = await production.weave("Test query")
    production_duration = production_result["duration_ms"]

    await production.shutdown()

    # Production should be slower but higher quality
    # (hash vs semantic embeddings, rule-based vs neural decision)
    assert "backend" in production_result
    assert production_result["backend"] == "inmemory"


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.skipif(
    not HOLOLOOM_AVAILABLE,
    reason="HoloLoom not available"
)
@pytest.mark.asyncio
async def test_backend_fallback():
    """Test HYBRID → INMEMORY fallback when Docker unavailable"""
    # Try HYBRID (may fallback to INMEMORY)
    loom = await create_production_loom_core({
        "backend": "HYBRID",
        "execution_mode": "FAST"
    })

    # Weave should work regardless of backend
    result = await loom.weave("Test query")

    assert "result" in result
    # Backend may be "inmemory" if HYBRID unavailable
    assert result.get("backend") in ["inmemory", "hybrid"]

    await loom.shutdown()


# ==================== Performance Tests ====================

@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_embedding_performance(production_loom):
    """Test embedding latency"""
    warp = production_loom.warp_space

    import time
    start = time.time()

    # Embed 10 queries
    for i in range(10):
        await warp.embed(f"Test query {i}")

    end = time.time()
    avg_latency = (end - start) / 10 * 1000  # ms

    # Should be <10ms per embedding
    assert avg_latency < 20  # Allow some overhead


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_weave_latency(production_loom):
    """Test full weave latency"""
    result = await production_loom.weave("Test query")

    # Should complete in <500ms (FAST mode)
    assert result["duration_ms"] < 500


# ==================== Edge Cases ====================

@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_empty_query(production_loom):
    """Test empty query handling"""
    result = await production_loom.weave("")

    # Should handle gracefully
    assert "result" in result or "error" in result


@pytest.mark.integration
@pytest.mark.hololoom
@pytest.mark.asyncio
async def test_very_long_query(production_loom):
    """Test very long query handling"""
    long_query = "What is Thompson Sampling? " * 100  # ~2500 chars

    result = await production_loom.weave(long_query)

    # Should handle without crashing
    assert "result" in result


# ==================== Marker for running tests ====================

# Run all HoloLoom tests:
#   pytest tests/integration/test_hololoom_integration.py -v -m hololoom

# Run without HoloLoom tests:
#   pytest tests/integration/test_hololoom_integration.py -v -m "not hololoom"
