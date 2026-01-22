"""
End-to-End Test: Full HoloLoom Weaving Pipeline
================================================
Tests the complete weaving cycle from query to response.
"""

import asyncio
import pytest
from HoloLoom.config import Config
from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.protocols.types import Query
from HoloLoom.memory.graph import KG, KGEdge


def create_test_memory():
    """Create a minimal KG for testing."""
    kg = KG()
    kg.add_edges([
        KGEdge("HoloLoom", "neural_system", "IS_A", 1.0),
        KGEdge("semantic_learning", "learning_method", "IS_A", 1.0),
        KGEdge("HoloLoom", "semantic_learning", "USES", 0.9),
        KGEdge("244D_space", "semantic_space", "IS_A", 1.0),
    ])
    return kg


@pytest.mark.asyncio
async def test_full_weaving_cycle_bare():
    """Test complete weaving with BARE mode (fastest)."""
    config = Config.bare()
    kg = create_test_memory()

    async with WeavingShuttle(cfg=config, yarn_graph=kg) as shuttle:
        query = Query(text="What is HoloLoom?")
        result = await shuttle.weave(query)

        assert result is not None
        assert result.response is not None


@pytest.mark.asyncio
async def test_full_weaving_cycle_fast():
    """Test complete weaving with FAST mode (balanced)."""
    config = Config.fast()
    kg = create_test_memory()

    async with WeavingShuttle(cfg=config, yarn_graph=kg) as shuttle:
        query = Query(text="How does semantic learning work?")
        result = await shuttle.weave(query)

        assert result is not None
        assert result.response is not None


@pytest.mark.asyncio
async def test_full_weaving_cycle_fused():
    """Test complete weaving with FUSED mode (full features)."""
    config = Config.fused()
    kg = create_test_memory()

    async with WeavingShuttle(cfg=config, yarn_graph=kg) as shuttle:
        query = Query(text="Explain the 244D semantic space")
        result = await shuttle.weave(query)

        assert result is not None
        assert result.response is not None


@pytest.mark.asyncio
async def test_multiple_queries_sequential():
    """Test handling multiple queries in sequence."""
    config = Config.fast()
    kg = create_test_memory()

    async with WeavingShuttle(cfg=config, yarn_graph=kg) as shuttle:
        queries = [
            Query(text="First question"),
            Query(text="Second question"),
            Query(text="Third question"),
        ]

        results = []
        for query in queries:
            result = await shuttle.weave(query)
            results.append(result)

        assert len(results) == 3
        assert all(r.response is not None for r in results)


@pytest.mark.asyncio
async def test_error_handling():
    """Test graceful error handling."""
    config = Config.bare()
    kg = create_test_memory()

    async with WeavingShuttle(cfg=config, yarn_graph=kg) as shuttle:
        # Empty query should still work (not crash)
        query = Query(text="")
        result = await shuttle.weave(query)

        # Should return something, even if it's an error message
        assert result is not None


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_full_weaving_cycle_bare())
    asyncio.run(test_full_weaving_cycle_fast())
    print("E2E tests passed!")
