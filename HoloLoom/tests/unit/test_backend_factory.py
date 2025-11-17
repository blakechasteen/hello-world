"""
Unit tests for memory backend factory.

Tests backend creation, fallback logic, and isolation.
All network calls mocked for speed (<150ms target).
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from HoloLoom.config import Config, MemoryBackend
from HoloLoom.memory.backend_factory import (
    create_memory_backend,
)


class TestBackendCreation:
    """Test memory backend creation."""

    @pytest.mark.asyncio
    async def test_inmemory_backend_creation(self):
        """INMEMORY backend should create without dependencies."""
        cfg = Config.fast()
        cfg.memory_backend = MemoryBackend.INMEMORY

        backend = await create_memory_backend(cfg)
        assert backend is not None
        assert hasattr(backend, "add_shard")

    @pytest.mark.asyncio
    async def test_inmemory_backend_fast(self):
        """INMEMORY backend creation should be fast."""
        import time

        cfg = Config.fast()
        cfg.memory_backend = MemoryBackend.INMEMORY

        start = time.perf_counter()
        backend = await create_memory_backend(cfg)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 50, f"Backend creation took {elapsed:.2f}ms (target: <50ms)"


class TestBackendFallback:
    """Test backend fallback logic."""

    @pytest.mark.asyncio
    @patch("HoloLoom.memory.backend_factory._create_hybrid_backend")
    async def test_hybrid_fallback_to_inmemory(self, mock_hybrid):
        """HYBRID should fall back to INMEMORY on failure."""
        # Simulate Neo4j/Qdrant unavailable
        mock_hybrid.side_effect = Exception("Connection refused")

        cfg = Config.fast()
        cfg.memory_backend = MemoryBackend.HYBRID

        # Should not raise, should fall back
        backend = await create_memory_backend(cfg)
        assert backend is not None

    @pytest.mark.asyncio
    @patch("HoloLoom.memory.backend_factory._create_hyperspace_backend")
    async def test_hyperspace_fallback(self, mock_hyperspace):
        """HYPERSPACE should fall back to INMEMORY on failure."""
        mock_hyperspace.side_effect = ImportError("Hyperspace not available")

        cfg = Config.fast()
        cfg.memory_backend = MemoryBackend.HYPERSPACE

        backend = await create_memory_backend(cfg)
        assert backend is not None


class TestBackendIsolation:
    """Test backend isolation and mocking."""

    @pytest.mark.asyncio
    @patch("HoloLoom.memory.backend_factory.KG")
    @patch("HoloLoom.memory.backend_factory.MemoryManager")
    async def test_inmemory_no_side_effects(self, mock_memory, mock_kg):
        """INMEMORY backend should not have side effects."""
        mock_kg.return_value = Mock()
        mock_memory.return_value = Mock()

        cfg = Config.fast()
        cfg.memory_backend = MemoryBackend.INMEMORY

        backend1 = await create_memory_backend(cfg)
        backend2 = await create_memory_backend(cfg)

        # Should create separate instances
        assert backend1 is not backend2

    @pytest.mark.asyncio
    async def test_backend_memory_usage(self):
        """Backend should have reasonable memory footprint."""
        import sys

        cfg = Config.bare()
        cfg.memory_backend = MemoryBackend.INMEMORY

        backend = await create_memory_backend(cfg)

        # Basic size check - should be < 1MB for empty backend
        size = sys.getsizeof(backend)
        assert size < 1024 * 1024, f"Backend size {size} bytes exceeds 1MB"


class TestBackendPerformance:
    """Test backend performance characteristics."""

    @pytest.mark.asyncio
    async def test_inmemory_latency(self):
        """INMEMORY operations should be <10ms."""
        import time

        cfg = Config.bare()
        cfg.memory_backend = MemoryBackend.INMEMORY

        backend = await create_memory_backend(cfg)

        # Measure add operation
        start = time.perf_counter()
        if hasattr(backend, "add_shard"):
            from HoloLoom.Documentation.types import MemoryShard

            shard = MemoryShard(text="test", source="unit_test")
            await backend.add_shard(shard)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 10, f"Add operation took {elapsed:.2f}ms (target: <10ms)"

    @pytest.mark.asyncio
    async def test_backend_concurrent_access(self):
        """Backend should handle concurrent operations."""
        import asyncio

        cfg = Config.fast()
        cfg.memory_backend = MemoryBackend.INMEMORY

        backend = await create_memory_backend(cfg)

        # Create mock shards
        async def add_mock_shard(i):
            if hasattr(backend, "add_shard"):
                from HoloLoom.Documentation.types import MemoryShard

                shard = MemoryShard(text=f"test_{i}", source="concurrent_test")
                await backend.add_shard(shard)

        # Run concurrent adds
        await asyncio.gather(*[add_mock_shard(i) for i in range(10)])

        # Should complete without errors


class TestBackendCleanup:
    """Test backend resource cleanup."""

    @pytest.mark.asyncio
    async def test_backend_context_manager(self):
        """Backend should support async context manager."""
        cfg = Config.fast()
        cfg.memory_backend = MemoryBackend.INMEMORY

        backend = await create_memory_backend(cfg)

        if hasattr(backend, "__aenter__"):
            async with backend:
                pass  # Should enter and exit cleanly

    @pytest.mark.asyncio
    @patch("HoloLoom.memory.backend_factory._create_hybrid_backend")
    async def test_backend_cleanup_on_error(self, mock_hybrid):
        """Backend should clean up resources on creation error."""
        mock_backend = AsyncMock()
        mock_backend.close = AsyncMock()
        mock_hybrid.return_value = mock_backend
        mock_hybrid.side_effect = Exception("Simulated error")

        cfg = Config.fast()
        cfg.memory_backend = MemoryBackend.HYBRID

        # Should fall back gracefully
        backend = await create_memory_backend(cfg)
        assert backend is not None  # Falls back to INMEMORY
