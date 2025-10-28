"""
End-to-End Test: Full HoloLoom Weaving Pipeline
================================================
Tests the complete weaving cycle from query to response.
"""

import asyncio
import pytest
from HoloLoom.config import Config
from HoloLoom.weaving_shuttle import WeavingShuttle
from HoloLoom.documentation.types import Query


@pytest.mark.asyncio
async def test_full_weaving_cycle_bare():
    """Test complete weaving with BARE mode (fastest)."""
    config = Config.bare()

    async with WeavingShuttle(cfg=config) as shuttle:
        query = Query(content="What is HoloLoom?")
        result = await shuttle.weave(query)

        assert result is not None
        assert result.response is not None


@pytest.mark.asyncio
async def test_full_weaving_cycle_fast():
    """Test complete weaving with FAST mode (balanced)."""
    config = Config.fast()

    async with WeavingShuttle(cfg=config) as shuttle:
        query = Query(content="How does semantic learning work?")
        result = await shuttle.weave(query)

        assert result is not None
        assert result.response is not None


@pytest.mark.asyncio
async def test_full_weaving_cycle_fused():
    """Test complete weaving with FUSED mode (full features)."""
    config = Config.fused()

    async with WeavingShuttle(cfg=config) as shuttle:
        query = Query(content="Explain the 244D semantic space")
        result = await shuttle.weave(query)

        assert result is not None
        assert result.response is not None


@pytest.mark.asyncio
async def test_multiple_queries_sequential():
    """Test handling multiple queries in sequence."""
    config = Config.fast()

    async with WeavingShuttle(cfg=config) as shuttle:
        queries = [
            Query(content="First question"),
            Query(content="Second question"),
            Query(content="Third question"),
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

    async with WeavingShuttle(cfg=config) as shuttle:
        # Empty query should still work (not crash)
        query = Query(content="")
        result = await shuttle.weave(query)

        # Should return something, even if it's an error message
        assert result is not None


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_full_weaving_cycle_bare())
    asyncio.run(test_full_weaving_cycle_fast())
    print("E2E tests passed!")
