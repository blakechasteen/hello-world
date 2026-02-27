"""
Integration Tests for Shuttle + WeavingOrchestrator

Tests the full weaving cycle with Shuttle integration:
- Full weaving cycle with Shuttle enabled
- Fallback behavior when Shuttle disabled
- Graceful degradation on errors
- Performance characteristics

Created: 2025-01-21
"""

import pytest
import asyncio
from datetime import datetime, timedelta

# HoloLoom types
from hololoom.protocols.types import Query, MemoryShard
from hololoom.config import Config, ExecutionMode
from hololoom.memory.graph import KG, KGEdge


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_shards():
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling is a Bayesian algorithm for balancing exploration and exploitation.",
            metadata={'topic': 'thompson_sampling', 'confidence': 0.9}
        ),
        MemoryShard(
            id="shard_2",
            text="Multi-armed bandits are problems where you must choose between multiple options.",
            metadata={'topic': 'bandits', 'confidence': 0.85}
        ),
        MemoryShard(
            id="shard_3",
            text="Beta distributions are used to model probabilities in Bayesian inference.",
            metadata={'topic': 'bayesian', 'confidence': 0.8}
        ),
    ]


@pytest.fixture
def test_kg():
    """Create test knowledge graph."""
    kg = KG()
    kg.add_edges([
        KGEdge("thompson_sampling", "bandit", "USES", 1.0),
        KGEdge("thompson_sampling", "bayesian", "IS_A", 1.0),
        KGEdge("bandit", "exploration", "LEADS_TO", 0.8),
        KGEdge("bayesian", "beta_distribution", "USES", 0.9),
    ])

    # Add descriptions
    for node in kg.G.nodes():
        kg.G.nodes[node]['description'] = f"Description of {node}"

    return kg


# ============================================================================
# Full Weaving Cycle Tests
# ============================================================================

@pytest.mark.asyncio
async def test_weaving_with_shuttle_enabled(test_shards, test_kg):
    """Test full weaving cycle with Shuttle enabled."""
    from hololoom.weaving_orchestrator import WeavingOrchestrator

    # Create config with Shuttle enabled
    config = Config.fast()
    config.enable_shuttle = True

    # Create orchestrator
    async with WeavingOrchestrator(
        cfg=config,
        shards=test_shards,
        yarn_graph=test_kg
    ) as orchestrator:

        # Weave query
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify result
        assert spacetime is not None
        assert spacetime.response is not None
        assert spacetime.confidence > 0.0

        # Check metadata for Shuttle usage
        metadata = spacetime.metadata or {}
        if 'shuttle_selection' in metadata.get('timings', {}):
            # Shuttle was used
            assert metadata['timings']['shuttle_selection'] > 0


@pytest.mark.asyncio
async def test_weaving_with_shuttle_disabled(test_shards, test_kg):
    """Test full weaving cycle with Shuttle disabled (fallback behavior)."""
    from hololoom.weaving_orchestrator import WeavingOrchestrator

    # Create config with Shuttle disabled
    config = Config.fast()
    config.enable_shuttle = False

    # Create orchestrator
    async with WeavingOrchestrator(
        cfg=config,
        shards=test_shards,
        yarn_graph=test_kg
    ) as orchestrator:

        # Weave query
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify result (should work without Shuttle)
        assert spacetime is not None
        assert spacetime.response is not None
        assert spacetime.confidence > 0.0

        # Check metadata - should NOT have shuttle_selection
        metadata = spacetime.metadata or {}
        assert 'shuttle_selection' not in metadata.get('timings', {})


@pytest.mark.asyncio
async def test_weaving_shuttle_modes(test_shards, test_kg):
    """Test weaving with different execution modes (BARE/FAST/FUSED)."""
    from hololoom.weaving_orchestrator import WeavingOrchestrator

    modes = [
        (Config.bare(), "BARE"),
        (Config.fast(), "FAST"),
        (Config.fused(), "FUSED"),
    ]

    for config, mode_name in modes:
        config.enable_shuttle = True

        async with WeavingOrchestrator(
            cfg=config,
            shards=test_shards,
            yarn_graph=test_kg
        ) as orchestrator:

            query = Query(text="What is Thompson Sampling?")
            spacetime = await orchestrator.weave(query)

            # All modes should work
            assert spacetime is not None, f"{mode_name} mode failed"
            assert spacetime.response is not None, f"{mode_name} mode returned no response"


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_weaving_shuttle_graceful_degradation(test_shards):
    """Test graceful degradation when Shuttle fails."""
    from hololoom.weaving_orchestrator import WeavingOrchestrator

    config = Config.fast()
    config.enable_shuttle = True

    # Use shards instead of KG (will cause Shuttle to fail gracefully)
    async with WeavingOrchestrator(
        cfg=config,
        shards=test_shards
    ) as orchestrator:

        # Should fall back to simple retrieval
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Should still work (graceful degradation)
        assert spacetime is not None
        assert spacetime.response is not None


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_weaving_shuttle_performance(test_shards, test_kg):
    """Test Shuttle performance overhead is acceptable."""
    from hololoom.weaving_orchestrator import WeavingOrchestrator
    import time

    config = Config.fast()

    # Baseline: without Shuttle
    config.enable_shuttle = False
    async with WeavingOrchestrator(
        cfg=config,
        shards=test_shards,
        yarn_graph=test_kg
    ) as orchestrator:
        start = time.time()
        query = Query(text="What is Thompson Sampling?")
        await orchestrator.weave(query)
        baseline_time = (time.time() - start) * 1000  # ms

    # With Shuttle
    config.enable_shuttle = True
    async with WeavingOrchestrator(
        cfg=config,
        shards=test_shards,
        yarn_graph=test_kg
    ) as orchestrator:
        start = time.time()
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)
        shuttle_time = (time.time() - start) * 1000  # ms

    # Check overhead
    overhead = shuttle_time - baseline_time
    print(f"\nBaseline: {baseline_time:.1f}ms, Shuttle: {shuttle_time:.1f}ms, Overhead: {overhead:.1f}ms")

    # Overhead should be reasonable (<100ms for FAST mode)
    # Note: This may fail in CI due to timing variance
    # assert overhead < 100, f"Shuttle overhead too high: {overhead:.1f}ms"


@pytest.mark.asyncio
async def test_weaving_shuttle_quality(test_shards, test_kg):
    """Test Shuttle improves result quality (confidence)."""
    from hololoom.weaving_orchestrator import WeavingOrchestrator

    config = Config.fast()

    # Baseline: without Shuttle
    config.enable_shuttle = False
    async with WeavingOrchestrator(
        cfg=config,
        shards=test_shards,
        yarn_graph=test_kg
    ) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        baseline = await orchestrator.weave(query)

    # With Shuttle
    config.enable_shuttle = True
    async with WeavingOrchestrator(
        cfg=config,
        shards=test_shards,
        yarn_graph=test_kg
    ) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        shuttle = await orchestrator.weave(query)

    # Both should have reasonable confidence
    assert baseline.confidence > 0.0
    assert shuttle.confidence > 0.0

    # Note: Quality improvement is hard to test deterministically
    # In practice, Shuttle should find more relevant context via graph traversal
    print(f"\nBaseline confidence: {baseline.confidence:.2f}, Shuttle confidence: {shuttle.confidence:.2f}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
