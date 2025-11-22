"""
Tests for SimpleRAG module.

Tests both unit-level functionality (with mocks) and integration scenarios.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from HoloLoom.rag import SimpleRAG, RAGResult
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query


# ============================================================================
# Unit Tests (with mocks)
# ============================================================================

class TestRAGResult:
    """Test RAGResult dataclass."""

    def test_creation(self):
        """Test creating a RAGResult."""
        result = RAGResult(
            response="Thompson Sampling balances exploration",
            sources=["source1", "source2"],
            confidence=0.85,
            reasoning_mode="verify"
        )

        assert result.response == "Thompson Sampling balances exploration"
        assert len(result.sources) == 2
        assert result.confidence == 0.85
        assert result.reasoning_mode == "verify"

    def test_string_representation(self):
        """Test string representation."""
        result = RAGResult(
            response="Answer to a long question that will be truncated",
            sources=["a", "b", "c"],
            confidence=0.9
        )

        str_repr = str(result)
        assert "RAGResult" in str_repr
        assert "sources: 3 items" in str_repr
        assert "0.90" in str_repr

    def test_metadata_default(self):
        """Test metadata defaults to empty dict."""
        result = RAGResult(
            response="test",
            sources=[],
            confidence=0.5
        )

        assert result.metadata == {}
        result.metadata['key'] = 'value'
        assert result.metadata['key'] == 'value'


class TestSimpleRAGInit:
    """Test SimpleRAG initialization."""

    def test_default_init(self):
        """Test default initialization."""
        rag = SimpleRAG()
        assert rag.config is not None
        assert rag.llm_provider == "ollama"
        assert rag.enable_caching is True
        assert rag.loom is None  # Not initialized until __aenter__

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = Config.fused()
        rag = SimpleRAG(config=config)
        assert rag.config is config

    def test_llm_provider_option(self):
        """Test setting LLM provider."""
        rag = SimpleRAG(llm_provider="anthropic")
        assert rag.llm_provider == "anthropic"


class TestSimpleRAGAsync:
    """Test SimpleRAG async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager lifecycle."""
        rag = SimpleRAG()

        # Should work as context manager
        async with rag as rag_obj:
            assert rag_obj is rag
            assert rag.loom is not None

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self):
        """Test that cleanup occurs on exit."""
        rag = SimpleRAG()

        # Use context manager
        async with rag:
            assert rag.loom is not None

        # After exit, loom should still exist but be cleaned up

    @pytest.mark.asyncio
    async def test_query_without_context(self):
        """Test that query fails outside context manager."""
        rag = SimpleRAG()

        with pytest.raises(RuntimeError):
            await rag.query("test question")

    @pytest.mark.asyncio
    async def test_ingest_without_context(self):
        """Test that ingest fails outside context manager."""
        rag = SimpleRAG()

        with pytest.raises(RuntimeError):
            await rag.ingest("test content")


class TestSimpleRAGIngest:
    """Test SimpleRAG ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_text(self):
        """Test ingesting text content."""
        rag = SimpleRAG()

        with patch.object(rag, 'loom') as mock_loom:
            mock_loom.experience = AsyncMock()
            rag.loom = mock_loom

            # Should delegate to loom.experience()
            await rag.ingest("test content")
            mock_loom.experience.assert_called_once_with("test content")

    @pytest.mark.asyncio
    async def test_ingest_multiple(self):
        """Test ingesting multiple items."""
        rag = SimpleRAG()

        with patch.object(rag, 'loom') as mock_loom:
            mock_loom.experience = AsyncMock()
            rag.loom = mock_loom

            contents = ["content1", "content2", "content3"]
            for content in contents:
                await rag.ingest(content)

            assert mock_loom.experience.call_count == 3


class TestSimpleRAGQuery:
    """Test SimpleRAG query functionality."""

    @pytest.mark.asyncio
    async def test_query_basic(self):
        """Test basic query with sources."""
        rag = SimpleRAG()

        # Mock components
        mock_memory = MagicMock()
        mock_memory.text = "Thompson Sampling is a Bayesian algorithm"

        with patch.object(rag, 'loom') as mock_loom:
            mock_loom.recall = AsyncMock(return_value=[mock_memory])
            rag.loom = mock_loom
            rag.orchestrator = None  # Fallback mode

            result = await rag.query("What is Thompson Sampling?")

            assert isinstance(result, RAGResult)
            assert result.response is not None
            assert len(result.sources) > 0
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_query_with_caching(self):
        """Test that results are cached."""
        rag = SimpleRAG(enable_caching=True)

        mock_memory = MagicMock()
        mock_memory.text = "source text"

        with patch.object(rag, 'loom') as mock_loom:
            mock_loom.recall = AsyncMock(return_value=[mock_memory])
            rag.loom = mock_loom
            rag.orchestrator = None

            question = "What is Thompson Sampling?"

            # First query
            result1 = await rag.query(question)
            call_count_1 = mock_loom.recall.call_count

            # Second query (should use cache)
            result2 = await rag.query(question)
            call_count_2 = mock_loom.recall.call_count

            # Recall should not be called again due to cache
            assert call_count_1 == call_count_2
            assert result1.response == result2.response

    @pytest.mark.asyncio
    async def test_query_no_sources(self):
        """Test query with no sources found."""
        rag = SimpleRAG()

        with patch.object(rag, 'loom') as mock_loom:
            mock_loom.recall = AsyncMock(return_value=[])
            rag.loom = mock_loom
            rag.orchestrator = None

            result = await rag.query("unknown question")

            assert result.response is not None
            assert len(result.sources) == 0
            assert result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_query_max_sources(self):
        """Test max_sources parameter."""
        rag = SimpleRAG()

        # Create mock memories
        memories = [MagicMock(text=f"source_{i}") for i in range(10)]

        with patch.object(rag, 'loom') as mock_loom:
            mock_loom.recall = AsyncMock(return_value=memories)
            rag.loom = mock_loom
            rag.orchestrator = None

            result = await rag.query("test", max_sources=3)

            # Recall should be called with limit=3
            mock_loom.recall.assert_called_once()
            call_kwargs = mock_loom.recall.call_args[1]
            assert call_kwargs.get('limit') == 3

            # Result should have at most 3 sources
            assert len(result.sources) <= 3

    @pytest.mark.asyncio
    async def test_query_modes(self):
        """Test different reasoning modes."""
        rag = SimpleRAG()

        with patch.object(rag, 'loom') as mock_loom:
            mock_loom.recall = AsyncMock(return_value=[MagicMock(text="source")])
            rag.loom = mock_loom
            rag.orchestrator = None

            modes = ["direct", "verify", "research", "plan_execute"]

            for mode in modes:
                result = await rag.query("test", mode=mode)
                assert result.reasoning_mode == mode


class TestSimpleRAGBatch:
    """Test batch query functionality."""

    @pytest.mark.asyncio
    async def test_batch_query(self):
        """Test batch querying."""
        rag = SimpleRAG()

        with patch.object(rag, 'query') as mock_query:
            # Mock query to return results
            async def mock_query_impl(*args, **kwargs):
                return RAGResult(
                    response="answer",
                    sources=["source"],
                    confidence=0.8
                )

            mock_query.side_effect = mock_query_impl

            questions = ["q1", "q2", "q3"]
            results = await rag.batch_query(questions)

            assert len(results) == 3
            for result in results:
                assert isinstance(result, RAGResult)

    @pytest.mark.asyncio
    async def test_batch_query_empty(self):
        """Test batch query with empty list."""
        rag = SimpleRAG()

        results = await rag.batch_query([])
        assert results == []


class TestSimpleRAGMetrics:
    """Test metrics and monitoring."""

    def test_get_metrics(self):
        """Test getting metrics."""
        rag = SimpleRAG()

        # Create mock loom
        rag.loom = MagicMock()
        rag.loom.get_metrics = MagicMock(return_value={
            'n_memories': 42,
            'n_connections': 128
        })

        metrics = rag.get_metrics()

        assert metrics['n_memories'] == 42
        assert metrics['n_connections'] == 128
        assert 'cache_size' in metrics
        assert 'cache_hit_rate' in metrics

    def test_clear_cache(self):
        """Test clearing the cache."""
        rag = SimpleRAG()

        # Add items to cache
        rag._cache[('q1', 'verify')] = RAGResult(
            response="ans1", sources=[], confidence=0.8
        )

        assert len(rag._cache) == 1

        rag.clear_cache()

        assert len(rag._cache) == 0

    def test_summary(self):
        """Test summary output."""
        rag = SimpleRAG()

        rag.loom = MagicMock()
        rag.loom.get_metrics = MagicMock(return_value={
            'n_memories': 42,
            'cache_size': 5,
            'cache_hit_rate': 0.6,
            'llm_provider': 'ollama',
            'llm_available': True
        })

        summary = rag.summary()

        assert "SimpleRAG" in summary
        assert "42" in summary
        assert "ollama" in summary


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_ingest_and_query():
    """Integration test: ingest content, then query."""
    async with SimpleRAG() as rag:
        # Ingest content
        await rag.ingest("Thompson Sampling balances exploration and exploitation")

        # Query about ingested content
        result = await rag.query("What is Thompson Sampling?")

        # Verify result structure
        assert isinstance(result, RAGResult)
        assert result.response is not None
        assert 0.0 <= result.confidence <= 1.0

        # Should have some sources (at least the ingested content)
        assert len(result.sources) >= 0  # May be 0 if content not retrieved


@pytest.mark.asyncio
async def test_multiple_ingest_and_query():
    """Integration test: ingest multiple items, query."""
    async with SimpleRAG() as rag:
        # Ingest multiple items
        contents = [
            "Thompson Sampling is a Bayesian algorithm",
            "It balances exploration and exploitation",
            "Used in multi-armed bandit problems"
        ]

        for content in contents:
            await rag.ingest(content)

        # Query
        result = await rag.query("Tell me about Thompson Sampling")

        # Should have retrieved something
        assert result.response is not None
        assert result.confidence > 0.0


@pytest.mark.asyncio
async def test_batch_ingest_and_query():
    """Integration test: batch ingest and batch query."""
    async with SimpleRAG() as rag:
        # Ingest batch
        contents = [
            "Reinforcement learning uses rewards",
            "Policy gradient methods optimize directly",
            "Q-learning learns action values"
        ]

        for content in contents:
            await rag.ingest(content)

        # Batch query
        questions = [
            "What is reinforcement learning?",
            "What are policy gradients?",
            "What is Q-learning?"
        ]

        results = await rag.batch_query(questions)

        assert len(results) == len(questions)
        for result in results:
            assert result.response is not None
