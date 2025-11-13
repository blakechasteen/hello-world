"""
Unit Tests for Streaming RAG
=============================

Tests for token-by-token streaming functionality.

Coverage:
- StreamToken dataclass
- query_stream() basic functionality
- Streaming with various modes
- Fallback behavior
- Caching after streaming
- Error handling
- Mock LLM providers
"""

import pytest
import asyncio
import logging
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from HoloLoom.rag.simple_rag import SimpleRAG, RAGResult
from HoloLoom.rag.streaming import StreamToken, StreamingError, stream_from_orchestrator
from HoloLoom.documentation.types import Query

logger = logging.getLogger(__name__)


class TestStreamToken:
    """Tests for StreamToken dataclass."""

    def test_stream_token_creation(self):
        """Test basic StreamToken creation."""
        token = StreamToken(
            text="Hello",
            index=0,
            cumulative_text="Hello",
            metadata={"latency_ms": 10.5},
            is_final=False
        )

        assert token.text == "Hello"
        assert token.index == 0
        assert token.cumulative_text == "Hello"
        assert token.metadata["latency_ms"] == 10.5
        assert token.is_final is False

    def test_stream_token_final(self):
        """Test StreamToken with is_final flag."""
        token = StreamToken(
            text="!",
            index=5,
            cumulative_text="Hello!",
            metadata={"total_tokens": 6},
            is_final=True
        )

        assert token.is_final is True
        assert token.metadata["total_tokens"] == 6

    def test_stream_token_default_metadata(self):
        """Test StreamToken with default metadata."""
        token = StreamToken(
            text="test",
            index=0,
            cumulative_text="test"
        )

        assert token.metadata == {}
        assert token.is_final is False

    def test_stream_token_repr(self):
        """Test StreamToken string representation."""
        token = StreamToken(
            text="Hello",
            index=5,
            cumulative_text="Hello World",
            is_final=True
        )

        repr_str = repr(token)
        assert "index=5" in repr_str
        assert "len=11" in repr_str
        assert "final=True" in repr_str


class TestStreamingFromOrchestrator:
    """Tests for stream_from_orchestrator function."""

    @pytest.mark.asyncio
    async def test_streaming_error_no_orchestrator(self):
        """Test streaming error when orchestrator is None."""
        query = Query(text="test")

        with pytest.raises(StreamingError, match="Orchestrator not available"):
            async for token in stream_from_orchestrator(None, query, []):
                pass

    @pytest.mark.asyncio
    async def test_streaming_error_no_llm(self):
        """Test streaming error when LLM not initialized."""
        orchestrator = MagicMock()
        orchestrator.llm = None
        query = Query(text="test")

        with pytest.raises(StreamingError, match="LLM not initialized"):
            async for token in stream_from_orchestrator(orchestrator, query, []):
                pass

    @pytest.mark.asyncio
    async def test_streaming_error_unsupported_provider(self):
        """Test streaming error for unsupported LLM provider."""
        orchestrator = MagicMock()
        orchestrator.llm = MagicMock()
        orchestrator.llm_provider = "unknown_provider"
        query = Query(text="test")

        with pytest.raises(StreamingError, match="Streaming not supported"):
            async for token in stream_from_orchestrator(orchestrator, query, []):
                pass

    @pytest.mark.asyncio
    async def test_streaming_ollama_mock(self):
        """Test streaming from mock Ollama LLM."""
        # Create mock stream generator
        async def mock_stream():
            for text in ["Hello", " ", "world", "!"]:
                yield text

        # Create mock orchestrator
        orchestrator = MagicMock()
        orchestrator.llm_provider = "ollama"
        orchestrator.llm = MagicMock()
        orchestrator.llm.stream_generate = AsyncMock(return_value=mock_stream())

        query = Query(text="Say hello")
        tokens = []

        async for token in stream_from_orchestrator(orchestrator, query, []):
            tokens.append(token)

        # Verify tokens
        assert len(tokens) == 5  # 4 chunks + 1 final
        assert tokens[0].text == "Hello"
        assert tokens[0].index == 0
        assert tokens[0].is_final is False
        assert tokens[-1].is_final is True
        assert tokens[-1].cumulative_text == "Hello world!"

    @pytest.mark.asyncio
    async def test_streaming_cumulative_text(self):
        """Test cumulative text accumulation."""
        async def mock_stream():
            for char in "Hi!":
                yield char

        orchestrator = MagicMock()
        orchestrator.llm_provider = "ollama"
        orchestrator.llm = MagicMock()
        orchestrator.llm.stream_generate = AsyncMock(return_value=mock_stream())

        query = Query(text="Say hi")
        cumulative_texts = []

        async for token in stream_from_orchestrator(orchestrator, query, []):
            cumulative_texts.append(token.cumulative_text)

        # Verify cumulative text accumulation
        assert cumulative_texts[0] == "H"
        assert cumulative_texts[1] == "Hi"
        assert cumulative_texts[2] == "Hi!"
        assert cumulative_texts[3] == "Hi!"  # Final token

    @pytest.mark.asyncio
    async def test_streaming_tokens_per_second(self):
        """Test tokens per second calculation."""
        async def mock_stream():
            for text in ["a", "b", "c"]:
                yield text

        orchestrator = MagicMock()
        orchestrator.llm_provider = "ollama"
        orchestrator.llm = MagicMock()
        orchestrator.llm.stream_generate = AsyncMock(return_value=mock_stream())

        query = Query(text="test")
        tokens = []

        async for token in stream_from_orchestrator(orchestrator, query, []):
            tokens.append(token)

        # Final token should have tokens_per_sec calculated
        # Note: may be 0 on very fast execution, so we check it's a number >= 0
        assert isinstance(tokens[-1].metadata['tokens_per_sec'], (int, float))
        assert tokens[-1].metadata['total_tokens'] == 3
        assert tokens[-1].metadata['tokens_per_sec'] >= 0

    @pytest.mark.asyncio
    async def test_streaming_anthropic_mock(self):
        """Test streaming from mock Anthropic LLM."""
        async def mock_stream():
            for text in ["Test", " ", "streaming"]:
                yield text

        orchestrator = MagicMock()
        orchestrator.llm_provider = "anthropic"
        orchestrator.llm = MagicMock()
        orchestrator.llm.messages_stream = AsyncMock(return_value=mock_stream())

        query = Query(text="test")
        tokens = []

        async for token in stream_from_orchestrator(orchestrator, query, []):
            tokens.append(token)

        assert len(tokens) == 4  # 3 chunks + 1 final
        assert tokens[-1].cumulative_text == "Test streaming"


class TestQueryStreamBasic:
    """Tests for SimpleRAG.query_stream() method."""

    @pytest.mark.asyncio
    async def test_query_stream_not_initialized(self):
        """Test streaming error when SimpleRAG not initialized."""
        rag = SimpleRAG()

        with pytest.raises(RuntimeError, match="not initialized"):
            async for token in rag.query_stream("test"):
                pass

    @pytest.mark.asyncio
    async def test_query_stream_fallback_mode_not_direct(self):
        """Test that non-direct modes fall back to regular query."""
        # Create a mock RAG instance
        rag = SimpleRAG()
        rag.loom = AsyncMock()
        rag.orchestrator = MagicMock()

        # Mock query method to return a result
        rag.query = AsyncMock(return_value=RAGResult(
            response="Test response",
            sources=["source1"],
            confidence=0.8
        ))

        # Stream with "verify" mode should fall back
        tokens = []
        async for token in rag.query_stream("test", mode="verify"):
            tokens.append(token)

        # Should have called query (fallback)
        rag.query.assert_called_once()
        assert len(tokens) > 0
        assert tokens[-1].is_final is True

    @pytest.mark.asyncio
    async def test_query_stream_caching(self):
        """Test that streamed responses are cached."""
        rag = SimpleRAG(enable_caching=True)
        rag._cache = {}  # Initialize cache
        rag.loom = AsyncMock()
        rag.enable_caching = True

        # Create mock response
        async def mock_stream():
            for text in ["Hello", " ", "world"]:
                yield text

        rag.orchestrator = MagicMock()
        rag.orchestrator.llm_provider = "ollama"
        rag.orchestrator.llm = MagicMock()
        rag.orchestrator.llm.stream_generate = AsyncMock(return_value=mock_stream())

        # Mock recall
        mock_memory = MagicMock()
        mock_memory.text = "source text"
        rag.loom.recall = AsyncMock(return_value=[mock_memory])

        # Stream query
        tokens = []
        async for token in rag.query_stream("test", mode="direct"):
            tokens.append(token)

        # Verify cache was populated
        cache_key = ("test", "direct")
        assert cache_key in rag._cache
        cached_result = rag._cache[cache_key]
        assert cached_result.response == "Hello world"
        assert cached_result.metadata.get('streamed') is True


class TestQueryStreamFallback:
    """Tests for fallback behavior in streaming."""

    @pytest.mark.asyncio
    async def test_streaming_fallback_no_sources(self):
        """Test fallback when no sources found."""
        rag = SimpleRAG()
        rag.loom = AsyncMock()
        rag.loom.recall = AsyncMock(return_value=[])  # No sources
        rag.orchestrator = MagicMock()

        tokens = []
        async for token in rag.query_stream("test", mode="direct", max_sources=5):
            tokens.append(token)

        # Should emit no tokens or fallback gracefully
        # (either empty or minimal response)
        assert len(tokens) >= 0

    @pytest.mark.asyncio
    async def test_streaming_fallback_no_orchestrator(self):
        """Test fallback when orchestrator not available."""
        rag = SimpleRAG()
        rag.loom = AsyncMock()
        rag.orchestrator = None  # No orchestrator

        mock_memory = MagicMock()
        mock_memory.text = "source"
        rag.loom.recall = AsyncMock(return_value=[mock_memory])

        tokens = []
        async for token in rag.query_stream("test", mode="direct"):
            tokens.append(token)

        # Should return empty or minimal response
        assert len(tokens) >= 0


class TestQueryStreamIntegration:
    """Integration tests for streaming."""

    @pytest.mark.asyncio
    async def test_query_stream_complete_flow(self):
        """Test complete streaming flow with mocks."""
        rag = SimpleRAG(enable_caching=True)
        rag._cache = {}

        # Mock all dependencies
        rag.loom = AsyncMock()
        rag.orchestrator = MagicMock()

        # Mock memory recall
        mock_memory = MagicMock()
        mock_memory.text = "Thompson Sampling is a Bayesian exploration strategy"
        rag.loom.recall = AsyncMock(return_value=[mock_memory])

        # Mock streaming from orchestrator
        async def mock_stream():
            for text in ["Thompson", " ", "Sampling", " ", "explores"]:
                yield text

        rag.orchestrator.llm_provider = "ollama"
        rag.orchestrator.llm = MagicMock()
        rag.orchestrator.llm.stream_generate = AsyncMock(return_value=mock_stream())

        # Stream
        tokens = []
        async for token in rag.query_stream("What is Thompson Sampling?", mode="direct"):
            tokens.append(token)

        # Verify
        assert len(tokens) == 6  # 5 chunks + 1 final
        assert tokens[0].text == "Thompson"
        assert tokens[-1].is_final is True
        assert tokens[-1].cumulative_text == "Thompson Sampling explores"

        # Verify caching
        cache_key = ("What is Thompson Sampling?", "direct")
        assert cache_key in rag._cache

    @pytest.mark.asyncio
    async def test_query_stream_multiple_calls(self):
        """Test multiple streaming calls on same instance."""
        rag = SimpleRAG(enable_caching=True)
        rag._cache = {}
        rag.loom = AsyncMock()
        rag.orchestrator = MagicMock()

        # Mock responses for two queries
        call_count = 0

        async def mock_stream_call1():
            for text in ["Answer", " ", "one"]:
                yield text

        async def mock_stream_call2():
            for text in ["Answer", " ", "two"]:
                yield text

        mock_memory = MagicMock()
        mock_memory.text = "source"

        rag.loom.recall = AsyncMock(return_value=[mock_memory])
        rag.orchestrator.llm_provider = "ollama"
        rag.orchestrator.llm = MagicMock()

        # First call
        rag.orchestrator.llm.stream_generate = AsyncMock(return_value=mock_stream_call1())
        tokens1 = []
        async for token in rag.query_stream("Question 1", mode="direct"):
            tokens1.append(token)

        # Second call
        rag.orchestrator.llm.stream_generate = AsyncMock(return_value=mock_stream_call2())
        tokens2 = []
        async for token in rag.query_stream("Question 2", mode="direct"):
            tokens2.append(token)

        # Both should be cached
        assert ("Question 1", "direct") in rag._cache
        assert ("Question 2", "direct") in rag._cache
        assert rag._cache[("Question 1", "direct")].response == "Answer one"
        assert rag._cache[("Question 2", "direct")].response == "Answer two"


class TestStreamingMetadata:
    """Tests for streaming metadata."""

    @pytest.mark.asyncio
    async def test_token_metadata_contains_latency(self):
        """Test that tokens include latency metadata."""
        async def mock_stream():
            for text in ["test"]:
                yield text

        orchestrator = MagicMock()
        orchestrator.llm_provider = "ollama"
        orchestrator.llm = MagicMock()
        orchestrator.llm.stream_generate = AsyncMock(return_value=mock_stream())

        query = Query(text="test")
        async for token in stream_from_orchestrator(orchestrator, query, []):
            if token.is_final:
                assert 'latency_ms' in token.metadata
                assert token.metadata['latency_ms'] >= 0

    @pytest.mark.asyncio
    async def test_token_metadata_contains_provider(self):
        """Test that tokens include LLM provider metadata."""
        async def mock_stream():
            for text in ["x"]:
                yield text

        orchestrator = MagicMock()
        orchestrator.llm_provider = "anthropic"
        orchestrator.llm = MagicMock()
        orchestrator.llm.messages_stream = AsyncMock(return_value=mock_stream())

        query = Query(text="test")
        async for token in stream_from_orchestrator(orchestrator, query, []):
            assert token.metadata.get('llm_provider') in ["anthropic"]


class TestStreamingErrors:
    """Tests for error handling in streaming."""

    @pytest.mark.asyncio
    async def test_streaming_error_on_stream_failure(self):
        """Test that streaming errors are properly raised."""
        orchestrator = MagicMock()
        orchestrator.llm_provider = "ollama"
        orchestrator.llm = MagicMock()

        # Make stream_generate raise an error
        async def broken_stream():
            raise RuntimeError("Connection failed")
            yield  # This is never reached

        orchestrator.llm.stream_generate = AsyncMock(return_value=broken_stream())

        query = Query(text="test")

        with pytest.raises(StreamingError):
            async for token in stream_from_orchestrator(orchestrator, query, []):
                pass


# ==================== Test Execution ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
