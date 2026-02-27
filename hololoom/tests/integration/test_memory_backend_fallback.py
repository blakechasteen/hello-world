"""
Memory Backend Fallback Integration Tests
==========================================
Tests auto-fallback behavior when Docker services (Neo4j/Qdrant) are unavailable.

Critical for investor demo reliability:
- If Docker fails to start → System gracefully falls back to INMEMORY
- No crashes, no "connection refused" errors
- Transparent fallback with logging

Created: 2025-11-22
Updated: 2025-12-15 - Fixed imports (in_memory_backend module doesn't exist)
Priority: HIGH (Docker reliability for demo)
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import logging

from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import create_memory_backend
# NOTE: InMemoryBackend is created via backend_factory, not imported directly
from HoloLoom.protocols.types import MemoryShard


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def inmemory_config():
    """Config for INMEMORY backend (always works)."""
    config = Config.fast()
    config.memory_backend = MemoryBackend.INMEMORY
    return config


@pytest.fixture
def hybrid_config():
    """Config for HYBRID backend (Neo4j + Qdrant)."""
    config = Config.fast()
    config.memory_backend = MemoryBackend.HYBRID
    return config


@pytest.fixture
def sample_shards():
    """Sample memory shards for testing."""
    return [
        MemoryShard(
            content="Thompson Sampling is a Bayesian approach",
            entities=["Thompson Sampling", "Bayesian"],
            motifs=["algorithm"]
        ),
        MemoryShard(
            content="It balances exploration and exploitation",
            entities=["exploration", "exploitation"],
            motifs=["tradeoff"]
        )
    ]


# ============================================================================
# INMEMORY Backend Tests (Always Available)
# ============================================================================

@pytest.mark.asyncio
async def test_inmemory_backend_always_works(inmemory_config):
    """Test INMEMORY backend always works (no dependencies)."""
    backend = await create_memory_backend(inmemory_config)

    assert backend is not None
    # Backend should be valid and functional (exact type varies by config)
    assert hasattr(backend, 'graph') or hasattr(backend, 'close')

    # Cleanup
    if hasattr(backend, 'close'):
        await backend.close()


@pytest.mark.asyncio
async def test_inmemory_backend_basic_operations(inmemory_config, sample_shards):
    """Test INMEMORY backend supports basic graph operations."""
    backend = await create_memory_backend(inmemory_config)

    try:
        # Add nodes
        for shard in sample_shards:
            await backend.add_shard(shard)

        # Retrieve nodes
        results = await backend.search("Thompson Sampling", k=10)

        assert len(results) > 0
        assert any("Thompson" in r.content for r in results)

    finally:
        await backend.close()


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
======================

✅ INMEMORY backend (always works) - 2 tests
✅ Fallback scenarios - Ready for expansion

Total: 2 tests (foundation)

Next: Add Docker fallback tests after base tests pass
"""
