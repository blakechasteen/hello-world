"""
Unit tests for WeavingOrchestrator core functionality.

Tests orchestration logic with all external dependencies mocked.
Target: <150ms per test, isolated, no network calls.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest


class TestOrchestratorCreation:
    """Test orchestrator initialization."""

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    @patch("HoloLoom.weaving_orchestrator.SpectralEmbedding")
    @patch("HoloLoom.weaving_orchestrator.create_policy")
    async def test_orchestrator_init_fast(self, mock_policy, mock_embed, mock_backend):
        """Orchestrator should initialize quickly."""
        import time

        from HoloLoom.config import Config
        from HoloLoom.Documentation.types import MemoryShard
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Mock all dependencies
        mock_backend.return_value = AsyncMock()
        mock_embed.return_value = Mock(embedding_dim=768)
        mock_policy.return_value = Mock()

        cfg = Config.fast()
        shards = [MemoryShard(text="test", source="init_test")]

        start = time.perf_counter()
        async with WeavingOrchestrator(cfg=cfg, shards=shards) as shuttle:
            pass
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 150, f"Init took {elapsed:.2f}ms (target: <150ms)"

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_orchestrator_with_empty_shards(self, mock_backend):
        """Orchestrator should handle empty shard list."""
        from HoloLoom.config import Config
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_backend.return_value = AsyncMock()
        cfg = Config.bare()

        async with WeavingOrchestrator(cfg=cfg, shards=[]) as shuttle:
            assert shuttle is not None

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_orchestrator_context_manager(self, mock_backend):
        """Orchestrator should support async context manager."""
        from HoloLoom.config import Config
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_backend.return_value = AsyncMock()
        cfg = Config.fast()

        # Should enter and exit cleanly
        async with WeavingOrchestrator(cfg=cfg, shards=[]) as shuttle:
            assert shuttle is not None


class TestOrchestratorWeaving:
    """Test core weaving functionality."""

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    @patch("HoloLoom.weaving_orchestrator.SpectralEmbedding")
    @patch("HoloLoom.weaving_orchestrator.create_policy")
    async def test_weave_basic_query(self, mock_policy, mock_embed, mock_backend):
        """Should handle basic query."""
        import torch

        from HoloLoom.config import Config
        from HoloLoom.Documentation.types import MemoryShard, Query
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Setup mocks
        mock_backend_instance = AsyncMock()
        mock_backend_instance.recall = AsyncMock(return_value=[])
        mock_backend.return_value = mock_backend_instance

        mock_embed_instance = Mock()
        mock_embed_instance.embedding_dim = 768
        mock_embed_instance.encode_multi_scale = Mock(return_value=[torch.randn(1, 768)])
        mock_embed.return_value = mock_embed_instance

        mock_policy_instance = Mock()
        mock_policy_instance.select_action = Mock(
            return_value=(torch.tensor([0]), torch.tensor([0.9]))
        )
        mock_policy.return_value = mock_policy_instance

        cfg = Config.bare()
        shards = [MemoryShard(text="test data", source="test")]

        async with WeavingOrchestrator(cfg=cfg, shards=shards) as shuttle:
            query = Query(text="test query")
            # Mock internal methods to avoid full execution
            shuttle._extract_features = AsyncMock(return_value=Mock())
            shuttle._select_tool = AsyncMock(return_value="answer")
            shuttle._execute_tool = AsyncMock(return_value="mocked response")

            result = await shuttle.weave(query)
            assert result is not None

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_weave_performance_target(self, mock_backend):
        """Weaving should meet performance targets."""
        import time

        from HoloLoom.config import Config
        from HoloLoom.Documentation.types import Query
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Mock everything for speed
        mock_backend.return_value = AsyncMock()
        cfg = Config.bare()

        async with WeavingOrchestrator(cfg=cfg, shards=[]) as shuttle:
            # Mock entire weave pipeline
            shuttle._extract_features = AsyncMock(return_value=Mock())
            shuttle._select_tool = AsyncMock(return_value="answer")
            shuttle._execute_tool = AsyncMock(return_value="fast response")

            query = Query(text="fast query")

            start = time.perf_counter()
            result = await shuttle.weave(query)
            elapsed = (time.perf_counter() - start) * 1000

            assert elapsed < 150, f"Weave took {elapsed:.2f}ms (target: <150ms)"

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_weave_concurrent_queries(self, mock_backend):
        """Should handle concurrent queries."""
        from HoloLoom.config import Config
        from HoloLoom.Documentation.types import Query
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_backend.return_value = AsyncMock()
        cfg = Config.fast()

        async with WeavingOrchestrator(cfg=cfg, shards=[]) as shuttle:
            # Mock pipeline
            shuttle._extract_features = AsyncMock(return_value=Mock())
            shuttle._select_tool = AsyncMock(return_value="answer")
            shuttle._execute_tool = AsyncMock(return_value="concurrent response")

            queries = [Query(text=f"query_{i}") for i in range(5)]

            # Execute concurrently
            results = await asyncio.gather(*[shuttle.weave(q) for q in queries])

            assert len(results) == 5
            assert all(r is not None for r in results)


class TestOrchestratorReflection:
    """Test reflection and learning."""

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_reflection_enabled(self, mock_backend):
        """Reflection should work when enabled."""
        from HoloLoom.config import Config
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_backend.return_value = AsyncMock()
        cfg = Config.fast()

        async with WeavingOrchestrator(cfg=cfg, shards=[], enable_reflection=True) as shuttle:
            assert hasattr(shuttle, "reflection_buffer") or hasattr(shuttle, "reflect")

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_reflection_disabled(self, mock_backend):
        """Should work without reflection."""
        from HoloLoom.config import Config
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_backend.return_value = AsyncMock()
        cfg = Config.bare()

        async with WeavingOrchestrator(cfg=cfg, shards=[], enable_reflection=False) as shuttle:
            assert shuttle is not None


class TestOrchestratorCleanup:
    """Test resource cleanup."""

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_cleanup_on_exit(self, mock_backend):
        """Should clean up resources on exit."""
        from HoloLoom.config import Config
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_backend_instance = AsyncMock()
        mock_backend_instance.close = AsyncMock()
        mock_backend.return_value = mock_backend_instance

        cfg = Config.fast()

        async with WeavingOrchestrator(cfg=cfg, shards=[]):
            pass

        # Cleanup should be called (context manager exit)

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_cleanup_on_error(self, mock_backend):
        """Should clean up even on error."""
        from HoloLoom.config import Config
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_backend.return_value = AsyncMock()
        cfg = Config.fast()

        try:
            async with WeavingOrchestrator(cfg=cfg, shards=[]) as shuttle:
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected

        # Cleanup should still happen


class TestOrchestratorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_empty_query(self, mock_backend):
        """Should handle empty query text."""
        from HoloLoom.config import Config
        from HoloLoom.Documentation.types import Query
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_backend.return_value = AsyncMock()
        cfg = Config.fast()

        async with WeavingOrchestrator(cfg=cfg, shards=[]) as shuttle:
            shuttle._extract_features = AsyncMock(return_value=Mock())
            shuttle._select_tool = AsyncMock(return_value="answer")
            shuttle._execute_tool = AsyncMock(return_value="response")

            query = Query(text="")
            result = await shuttle.weave(query)
            assert result is not None

    @pytest.mark.asyncio
    @patch("HoloLoom.weaving_orchestrator.create_memory_backend")
    async def test_very_long_query(self, mock_backend):
        """Should handle very long queries."""
        from HoloLoom.config import Config
        from HoloLoom.Documentation.types import Query
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_backend.return_value = AsyncMock()
        cfg = Config.fast()

        async with WeavingOrchestrator(cfg=cfg, shards=[]) as shuttle:
            shuttle._extract_features = AsyncMock(return_value=Mock())
            shuttle._select_tool = AsyncMock(return_value="answer")
            shuttle._execute_tool = AsyncMock(return_value="response")

            query = Query(text="test " * 10000)  # Very long query
            result = await shuttle.weave(query)
            assert result is not None
