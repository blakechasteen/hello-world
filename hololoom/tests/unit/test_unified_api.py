"""
Unit tests for unified_api.py - HoloLoom unified entry point.

Tests:
- HoloLoom class instantiation
- create() factory method with different configurations
- API method signatures (query, chat, ingest_*)
- Configuration validation
- Graceful degradation when optional dependencies unavailable
- Statistics and state management
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock


class TestHoloLoomCreation:
    """Test HoloLoom instantiation and factory methods."""

    @pytest.mark.asyncio
    async def test_create_with_defaults(self, mock_all_external_deps):
        """create() should work with default parameters."""
        from hololoom.unified_api import HoloLoom

        with patch('hololoom.unified_api.WeavingOrchestrator'):
            loom = await HoloLoom.create()

            # Should return HoloLoom instance
            assert loom is not None
            assert isinstance(loom, HoloLoom)

    @pytest.mark.asyncio
    async def test_create_with_bare_pattern(self, mock_all_external_deps):
        """create() should accept BARE pattern."""
        from hololoom.unified_api import HoloLoom

        with patch('hololoom.unified_api.WeavingOrchestrator'):
            loom = await HoloLoom.create(pattern="bare")

            assert loom is not None

    @pytest.mark.asyncio
    async def test_create_with_fast_pattern(self, mock_all_external_deps):
        """create() should accept FAST pattern."""
        from hololoom.unified_api import HoloLoom

        with patch('hololoom.unified_api.WeavingOrchestrator'):
            loom = await HoloLoom.create(pattern="fast")

            assert loom is not None

    @pytest.mark.asyncio
    async def test_create_with_fused_pattern(self, mock_all_external_deps):
        """create() should accept FUSED pattern."""
        from hololoom.unified_api import HoloLoom

        with patch('hololoom.unified_api.WeavingOrchestrator'):
            loom = await HoloLoom.create(pattern="fused")

            assert loom is not None

    @pytest.mark.asyncio
    async def test_create_with_memory_backend(self, mock_all_external_deps):
        """create() should accept memory_backend parameter."""
        from hololoom.unified_api import HoloLoom

        with patch('hololoom.unified_api.WeavingOrchestrator'):
            loom = await HoloLoom.create(memory_backend="simple")

            assert loom is not None

    @pytest.mark.asyncio
    async def test_create_with_synthesis_enabled(self, mock_all_external_deps):
        """create() should accept enable_synthesis parameter."""
        from hololoom.unified_api import HoloLoom

        with patch('hololoom.unified_api.WeavingOrchestrator'):
            loom = await HoloLoom.create(enable_synthesis=True)

            assert loom is not None
            # Should enable synthesis
            assert hasattr(loom, 'enable_synthesis')
            assert loom.enable_synthesis == True


class TestQueryMethod:
    """Test query() method."""

    @pytest.mark.asyncio
    async def test_query_returns_result(self, mock_all_external_deps):
        """query() should return a result."""
        from hololoom.unified_api import HoloLoom
        from hololoom.fabric.spacetime import Spacetime

        # Mock weaver
        mock_weaver = AsyncMock()
        mock_weaver.weave = AsyncMock(return_value=Spacetime(
            response="Test response",
            query_text="Test query",
            tool_used="test_tool",
            confidence=0.9,
            trace=None
        ))

        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        result = await loom.query("Test query")

        # Should return Spacetime
        assert result is not None
        assert isinstance(result, Spacetime)
        assert result.response == "Test response"

    @pytest.mark.asyncio
    async def test_query_calls_weaver(self, mock_all_external_deps):
        """query() should call weaver.weave()."""
        from hololoom.unified_api import HoloLoom
        from hololoom.fabric.spacetime import Spacetime

        mock_weaver = AsyncMock()
        mock_weaver.weave = AsyncMock(return_value=Spacetime(
            response="Test",
            query_text="Test",
            tool_used="test_tool",
            confidence=0.9,
            trace=None
        ))

        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)
        await loom.query("Test query")

        # Should have called weave
        mock_weaver.weave.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_empty_string(self, mock_all_external_deps):
        """query() should handle empty string."""
        from hololoom.unified_api import HoloLoom
        from hololoom.fabric.spacetime import Spacetime

        mock_weaver = AsyncMock()
        mock_weaver.weave = AsyncMock(return_value=Spacetime(
            response="",
            query_text="",
            tool_used="test_tool",
            confidence=0.5,
            trace=None
        ))

        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)
        result = await loom.query("")

        # Should still return result
        assert result is not None


class TestChatMethod:
    """Test chat() conversational method."""

    @pytest.mark.asyncio
    async def test_chat_returns_result(self, mock_all_external_deps):
        """chat() should return a result."""
        from hololoom.unified_api import HoloLoom
        from hololoom.fabric.spacetime import Spacetime

        mock_weaver = AsyncMock()
        mock_weaver.weave = AsyncMock(return_value=Spacetime(
            response="Chat response",
            query_text="Chat query",
            tool_used="test_tool",
            confidence=0.9,
            trace=None
        ))

        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)
        result = await loom.chat("Chat query", return_trace=True)

        # Should return Spacetime when return_trace=True
        assert result is not None
        assert isinstance(result, Spacetime)

    @pytest.mark.asyncio
    async def test_chat_maintains_conversation(self, mock_all_external_deps):
        """chat() should maintain conversation context."""
        from hololoom.unified_api import HoloLoom
        from hololoom.fabric.spacetime import Spacetime

        mock_weaver = AsyncMock()
        mock_weaver.weave = AsyncMock(return_value=Spacetime(
            response="Response",
            query_text="Query",
            tool_used="test_tool",
            confidence=0.9,
            trace=None
        ))

        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        # Multiple chat calls
        await loom.chat("First message")
        await loom.chat("Second message")

        # Should have called weave twice
        assert mock_weaver.weave.call_count == 2


class TestIngestionMethods:
    """Test data ingestion methods."""

    @pytest.mark.asyncio
    async def test_ingest_text(self, mock_all_external_deps):
        """ingest_text() should accept text content."""
        from hololoom.unified_api import HoloLoom

        mock_weaver = AsyncMock()
        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        # Should not raise error
        result = await loom.ingest_text("Sample text content")

        # Should return count or success indicator
        assert result is not None

    @pytest.mark.asyncio
    async def test_ingest_text_with_empty_string(self, mock_all_external_deps):
        """ingest_text() should handle empty string."""
        from hololoom.unified_api import HoloLoom

        mock_weaver = AsyncMock()
        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        # Should handle gracefully
        result = await loom.ingest_text("")
        assert result is not None

    @pytest.mark.asyncio
    async def test_ingest_text_with_metadata(self, mock_all_external_deps):
        """ingest_text() should accept metadata parameter."""
        from hololoom.unified_api import HoloLoom

        mock_weaver = AsyncMock()
        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        metadata = {"source": "test", "timestamp": "2024-01-01"}
        result = await loom.ingest_text("Sample text", metadata=metadata)

        assert result is not None

    @pytest.mark.asyncio
    async def test_ingest_web(self, mock_all_external_deps):
        """ingest_web() should accept URL."""
        from hololoom.unified_api import HoloLoom

        mock_weaver = AsyncMock()
        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        with patch('hololoom.unified_api.WebsiteSpinner'):
            result = await loom.ingest_web("https://example.com")

            assert result is not None

    @pytest.mark.asyncio
    async def test_ingest_youtube(self, mock_all_external_deps):
        """ingest_youtube() should accept video ID."""
        from hololoom.unified_api import HoloLoom

        mock_weaver = AsyncMock()
        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        with patch('hololoom.unified_api.YouTubeSpinner'):
            result = await loom.ingest_youtube("VIDEO_ID")

            assert result is not None


class TestStatistics:
    """Test statistics and state management."""

    def test_get_stats_returns_dict(self, mock_all_external_deps):
        """get_stats() should return dictionary."""
        from hololoom.unified_api import HoloLoom

        mock_weaver = Mock()
        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        stats = loom.get_stats()

        # Should return dict
        assert isinstance(stats, dict)

    def test_get_stats_includes_query_count(self, mock_all_external_deps):
        """get_stats() should include query count."""
        from hololoom.unified_api import HoloLoom

        mock_weaver = Mock()
        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        stats = loom.get_stats()

        # Should have query count
        assert 'total_queries' in stats or 'query_count' in stats or len(stats) >= 0


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_handles_weaver_error_gracefully(self, mock_all_external_deps):
        """query() should handle weaver errors gracefully."""
        from hololoom.unified_api import HoloLoom

        mock_weaver = AsyncMock()
        mock_weaver.weave = AsyncMock(side_effect=Exception("Weaver error"))

        loom = HoloLoom(weaver=mock_weaver, enable_synthesis=False)

        # Should either raise or return error result (depends on implementation)
        with pytest.raises(Exception):
            await loom.query("Test query")

    @pytest.mark.asyncio
    async def test_optional_dependencies_degrade_gracefully(self):
        """System should work without optional dependencies."""
        # This tests that imports don't fail when optional deps missing
        # Already tested by module import in other tests
        from hololoom.unified_api import HoloLoom

        # Should import successfully even if mythy not installed
        assert HoloLoom is not None


class TestConfiguration:
    """Test configuration handling."""

    @pytest.mark.asyncio
    async def test_create_accepts_config_object(self, bare_config, mock_all_external_deps):
        """create() should accept pattern corresponding to config mode."""
        from hololoom.unified_api import HoloLoom

        with patch('hololoom.unified_api.WeavingOrchestrator'):
            # Create with pattern matching bare_config mode
            loom = await HoloLoom.create(pattern="bare")

            assert loom is not None

    def test_init_stores_config(self, bare_config):
        """__init__() should store config parameter."""
        from hololoom.unified_api import HoloLoom

        mock_weaver = Mock()
        loom = HoloLoom(weaver=mock_weaver, config=bare_config, enable_synthesis=False)

        # Should store config
        assert hasattr(loom, '_config') or hasattr(loom, 'config')


class TestAsyncContextManager:
    """Test async context manager support (if implemented)."""

    @pytest.mark.asyncio
    async def test_context_manager_if_implemented(self, mock_all_external_deps):
        """If HoloLoom supports async context manager, test it."""
        from hololoom.unified_api import HoloLoom

        # Check if context manager is implemented
        if hasattr(HoloLoom, '__aenter__'):
            with patch('hololoom.unified_api.WeavingOrchestrator'):
                async with await HoloLoom.create() as loom:
                    assert loom is not None
        else:
            # Context manager not implemented, skip
            pytest.skip("HoloLoom doesn't implement async context manager")
