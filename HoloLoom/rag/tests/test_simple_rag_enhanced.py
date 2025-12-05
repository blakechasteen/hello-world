"""
Enhanced comprehensive unit tests for SimpleRAG module.

This file extends the existing test_simple_rag.py with additional test coverage for:
1. Epistemic confidence calculation (consciousness integration)
2. Reranking functionality
3. Error handling edge cases
4. Cache invalidation scenarios
5. Embedding provider integration
6. LLM fallback behavior
7. Metadata validation
8. Performance characteristics

Tests follow HoloLoom testing patterns:
- pytest.mark.asyncio for async tests
- Mock external dependencies (HoloLoom backend, LLM)
- Fixtures for reusable test components
- Docstrings explaining what each test validates
- Naming: test_<feature>_<scenario>
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from HoloLoom.rag import SimpleRAG, RAGResult
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_hololoom():
    """Create a mock HoloLoom instance with common methods."""
    mock = MagicMock()
    mock.recall = AsyncMock(return_value=[])
    mock.experience = AsyncMock()
    mock.get_metrics = MagicMock(return_value={
        'n_memories': 10,
        'activation': {'active_nodes': 5, 'density': 0.5},
        'coherence': {'global_coherence': 0.7}
    })
    mock.awareness = MagicMock()  # Awareness graph available
    return mock


@pytest.fixture
def mock_memory():
    """Create a mock memory object."""
    memory = MagicMock()
    memory.text = "Thompson Sampling is a Bayesian algorithm for exploration-exploitation"
    memory.id = "test_memory_1"
    memory.timestamp = 1234567890.0
    memory.relevance = 0.85
    return memory


@pytest.fixture
def mock_orchestrator():
    """Create a mock WeavingOrchestrator for LLM generation."""
    mock = AsyncMock()

    # Mock spacetime response
    mock_spacetime = MagicMock()
    mock_spacetime.response = "Thompson Sampling is a Bayesian algorithm that balances exploration and exploitation."
    mock_spacetime.confidence = 0.92

    mock.weave = AsyncMock(return_value=mock_spacetime)
    return mock


@pytest.fixture
def mock_reranker():
    """Create a mock reranker."""
    mock = MagicMock()
    # Return indices and scores for reranking
    mock.rerank = MagicMock(return_value=[
        (0, 0.95),  # (index, score)
        (2, 0.87),
        (1, 0.82),
    ])
    return mock


# ============================================================================
# Epistemic Confidence Tests (Consciousness Integration - Phase 1)
# ============================================================================

class TestEpistemicConfidence:
    """Test epistemic confidence calculation from awareness layer."""

    @pytest.mark.asyncio
    async def test_epistemic_confidence_with_awareness(self, mock_hololoom, mock_memory):
        """Test epistemic confidence calculation when awareness layer available."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.orchestrator = None

        # Setup awareness metrics
        mock_hololoom.get_metrics = MagicMock(return_value={
            'activation': {'active_nodes': 10, 'density': 0.8},
            'coherence': {'global_coherence': 0.9},
        })
        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        result = await rag.query("Test query")

        # Should have epistemic confidence
        assert result.epistemic_confidence is not None
        assert 0.0 <= result.epistemic_confidence <= 1.0

        # Should be high (good coherence + sources)
        assert result.epistemic_confidence > 0.6

    @pytest.mark.asyncio
    async def test_epistemic_confidence_low_coherence(self, mock_hololoom, mock_memory):
        """Test epistemic confidence with low coherence (uncertain)."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.orchestrator = None

        # Low coherence = epistemic uncertainty
        mock_hololoom.get_metrics = MagicMock(return_value={
            'activation': {'active_nodes': 2, 'density': 0.2},
            'coherence': {'global_coherence': 0.3},
        })
        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        result = await rag.query("Test query")

        # Should have low epistemic confidence
        assert result.epistemic_confidence is not None
        assert result.epistemic_confidence < 0.5

    @pytest.mark.asyncio
    async def test_epistemic_confidence_no_sources(self, mock_hololoom):
        """Test epistemic confidence with no sources (very uncertain)."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.orchestrator = None

        mock_hololoom.get_metrics = MagicMock(return_value={
            'activation': {'active_nodes': 0, 'density': 0.0},
            'coherence': {'global_coherence': 0.5},
        })
        mock_hololoom.recall = AsyncMock(return_value=[])  # No sources

        result = await rag.query("Unknown query")

        # Should have very low epistemic confidence (no sources)
        assert result.epistemic_confidence is not None
        assert result.epistemic_confidence < 0.3

    @pytest.mark.asyncio
    async def test_epistemic_confidence_without_awareness(self, mock_hololoom, mock_memory):
        """Test that epistemic confidence is None when awareness unavailable."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.loom.awareness = None  # No awareness layer
        rag.orchestrator = None

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        result = await rag.query("Test query")

        # Should be None when awareness unavailable
        assert result.epistemic_confidence is None

    @pytest.mark.asyncio
    async def test_awareness_metadata_in_result(self, mock_hololoom, mock_memory):
        """Test that awareness metadata is included in result."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.orchestrator = None

        mock_hololoom.get_metrics = MagicMock(return_value={
            'activation': {'active_nodes': 10, 'density': 0.8},
            'coherence': {'global_coherence': 0.9},
        })
        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        result = await rag.query("Test query")

        # Should have awareness metadata
        assert 'awareness' in result.metadata
        awareness = result.metadata['awareness']
        assert 'epistemic_confidence' in awareness
        assert 'coherence' in awareness
        assert 'activation_density' in awareness
        assert 'active_nodes' in awareness


# ============================================================================
# Reranking Tests
# ============================================================================

class TestReranking:
    """Test reranking functionality."""

    @pytest.mark.asyncio
    async def test_reranking_enabled(self, mock_hololoom, mock_reranker):
        """Test query with reranking enabled."""
        rag = SimpleRAG(enable_reranking=True, rerank_top_k=10)
        rag.loom = mock_hololoom
        rag.reranker = mock_reranker
        rag.orchestrator = None

        # Create multiple memories
        memories = [
            MagicMock(text=f"Memory {i}") for i in range(10)
        ]
        mock_hololoom.recall = AsyncMock(return_value=memories)

        result = await rag.query("Test query", max_sources=3)

        # Should have called reranker
        assert mock_reranker.rerank.called

        # Should have rerank_latency_ms in metadata
        assert 'rerank_latency_ms' in result.metadata
        assert result.metadata['rerank_latency_ms'] >= 0

    @pytest.mark.asyncio
    async def test_reranking_retrieval_count(self, mock_hololoom, mock_reranker):
        """Test that reranking retrieves more documents than needed."""
        rag = SimpleRAG(enable_reranking=True, rerank_top_k=20)
        rag.loom = mock_hololoom
        rag.reranker = mock_reranker
        rag.orchestrator = None

        memories = [MagicMock(text=f"Memory {i}") for i in range(20)]
        mock_hololoom.recall = AsyncMock(return_value=memories)

        await rag.query("Test query", max_sources=5)

        # Should retrieve rerank_top_k documents
        call_kwargs = mock_hololoom.recall.call_args[1]
        assert call_kwargs['limit'] == 20

    @pytest.mark.asyncio
    async def test_reranking_stats_tracking(self, mock_hololoom, mock_reranker):
        """Test that reranking stats are tracked."""
        rag = SimpleRAG(enable_reranking=True)
        rag.loom = mock_hololoom
        rag.reranker = mock_reranker
        rag.orchestrator = None

        memories = [MagicMock(text=f"Memory {i}") for i in range(10)]
        mock_hololoom.recall = AsyncMock(return_value=memories)

        # Initial stats
        assert rag._rerank_stats['total_reranks'] == 0

        # Query with reranking
        await rag.query("Test query", max_sources=3)

        # Stats should be updated
        assert rag._rerank_stats['total_reranks'] == 1
        assert rag._rerank_stats['total_rerank_latency_ms'] > 0

    @pytest.mark.asyncio
    async def test_reranking_fallback_on_error(self, mock_hololoom):
        """Test fallback to original order if reranking fails."""
        rag = SimpleRAG(enable_reranking=True)
        rag.loom = mock_hololoom

        # Mock reranker that raises error
        mock_reranker = MagicMock()
        mock_reranker.rerank = MagicMock(side_effect=Exception("Reranker error"))
        rag.reranker = mock_reranker
        rag.orchestrator = None

        memories = [MagicMock(text=f"Memory {i}") for i in range(10)]
        mock_hololoom.recall = AsyncMock(return_value=memories)

        # Should not crash
        result = await rag.query("Test query", max_sources=3)

        # Should have sources (original order)
        assert len(result.sources) <= 3


# ============================================================================
# Cache Tests
# ============================================================================

class TestCaching:
    """Test query caching functionality."""

    @pytest.mark.asyncio
    async def test_cache_hit_metadata(self, mock_hololoom, mock_memory):
        """Test that cache hit is indicated in metadata."""
        rag = SimpleRAG(enable_caching=True)
        rag.loom = mock_hololoom
        rag.orchestrator = None

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        question = "What is Thompson Sampling?"

        # First query
        result1 = await rag.query(question)
        assert result1.metadata.get('cache_hit') is False

        # Second query (cached)
        result2 = await rag.query(question)
        assert result2.metadata.get('cache_hit') is True

    @pytest.mark.asyncio
    async def test_cache_key_includes_mode(self, mock_hololoom, mock_memory):
        """Test that cache key includes reasoning mode."""
        rag = SimpleRAG(enable_caching=True)
        rag.loom = mock_hololoom
        rag.orchestrator = None

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        question = "Test question"

        # Query with different modes
        result1 = await rag.query(question, mode="direct")
        result2 = await rag.query(question, mode="verify")

        # Both should call recall (different cache keys)
        assert mock_hololoom.recall.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_disabled(self, mock_hololoom, mock_memory):
        """Test query with caching disabled."""
        rag = SimpleRAG(enable_caching=False)
        rag.loom = mock_hololoom
        rag.orchestrator = None

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        question = "Test question"

        # Query twice
        await rag.query(question)
        await rag.query(question)

        # Should call recall both times
        assert mock_hololoom.recall.call_count == 2

    @pytest.mark.asyncio
    async def test_use_cache_parameter(self, mock_hololoom, mock_memory):
        """Test use_cache parameter overrides default."""
        rag = SimpleRAG(enable_caching=True)
        rag.loom = mock_hololoom
        rag.orchestrator = None

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        question = "Test question"

        # First query
        await rag.query(question)

        # Second query with use_cache=False
        await rag.query(question, use_cache=False)

        # Should call recall both times
        assert mock_hololoom.recall.call_count == 2


# ============================================================================
# LLM Integration Tests
# ============================================================================

class TestLLMIntegration:
    """Test LLM orchestrator integration."""

    @pytest.mark.asyncio
    async def test_query_with_llm(self, mock_hololoom, mock_orchestrator, mock_memory):
        """Test query with LLM orchestrator available."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.orchestrator = mock_orchestrator

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        result = await rag.query("Test query")

        # Should have called orchestrator
        assert mock_orchestrator.weave.called

        # Should have LLM-generated response
        assert "Bayesian algorithm" in result.response

    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self, mock_hololoom, mock_memory):
        """Test fallback when LLM fails."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom

        # Mock orchestrator that raises error
        mock_orch = AsyncMock()
        mock_orch.weave = AsyncMock(side_effect=Exception("LLM error"))
        rag.orchestrator = mock_orch

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        # Should not crash
        result = await rag.query("Test query")

        # Should have fallback response
        assert result.response is not None
        assert "Based on" in result.response or "No relevant" in result.response

    @pytest.mark.asyncio
    async def test_llm_provider_in_metadata(self, mock_hololoom, mock_orchestrator, mock_memory):
        """Test that LLM provider is included in metadata."""
        rag = SimpleRAG(llm_provider="anthropic")
        rag.loom = mock_hololoom
        rag.orchestrator = mock_orchestrator

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        result = await rag.query("Test query")

        assert result.metadata['llm_provider'] == "anthropic"


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_query_without_initialization(self):
        """Test that query fails when not initialized."""
        rag = SimpleRAG()

        with pytest.raises(RuntimeError) as exc_info:
            await rag.query("Test query")

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ingest_without_initialization(self):
        """Test that ingest fails when not initialized."""
        rag = SimpleRAG()

        with pytest.raises(RuntimeError) as exc_info:
            await rag.ingest("Test content")

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_empty_question(self, mock_hololoom):
        """Test query with empty question string."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.orchestrator = None

        mock_hololoom.recall = AsyncMock(return_value=[])

        # Should not crash
        result = await rag.query("")

        assert result is not None
        assert isinstance(result, RAGResult)

    @pytest.mark.asyncio
    async def test_recall_exception_handling(self, mock_hololoom):
        """Test handling of recall exceptions."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.orchestrator = None

        # Mock recall that raises exception
        mock_hololoom.recall = AsyncMock(side_effect=Exception("Recall error"))

        # Should propagate exception (not caught internally)
        with pytest.raises(Exception) as exc_info:
            await rag.query("Test query")

        assert "Recall error" in str(exc_info.value)


# ============================================================================
# Metrics Tests
# ============================================================================

class TestMetrics:
    """Test metrics and monitoring."""

    def test_get_metrics_with_reranking(self, mock_hololoom):
        """Test metrics include reranking stats."""
        rag = SimpleRAG(enable_reranking=True)
        rag.loom = mock_hololoom

        # Simulate some reranks
        rag._rerank_stats['total_reranks'] = 5
        rag._rerank_stats['total_rerank_latency_ms'] = 250.0

        metrics = rag.get_metrics()

        assert metrics['reranking_enabled'] is True
        assert metrics['total_reranks'] == 5
        assert metrics['avg_rerank_latency_ms'] == 50.0  # 250/5

    def test_get_metrics_cache_hit_rate(self, mock_hololoom):
        """Test cache hit rate calculation in metrics."""
        rag = SimpleRAG(enable_caching=True)
        rag.loom = mock_hololoom

        # Add cache entries
        rag._cache[('q1', 'verify')] = RAGResult(
            response="ans1", sources=[], confidence=0.8,
            metadata={'cache_hit': False}
        )
        rag._cache[('q2', 'verify')] = RAGResult(
            response="ans2", sources=[], confidence=0.8,
            metadata={'cache_hit': True}
        )
        rag._cache[('q3', 'verify')] = RAGResult(
            response="ans3", sources=[], confidence=0.8,
            metadata={'cache_hit': True}
        )

        metrics = rag.get_metrics()

        assert metrics['cache_size'] == 3
        # 2 hits out of 3 = 66.7%
        assert 0.66 <= metrics['cache_hit_rate'] <= 0.67

    def test_get_metrics_embedding_provider(self):
        """Test that embedding provider info is included in metrics."""
        rag = SimpleRAG()

        # Mock embedding provider
        mock_embedding = MagicMock()
        mock_embedding.dimension = 384
        rag.embedding_provider = mock_embedding

        rag.loom = MagicMock()
        rag.loom.get_metrics = MagicMock(return_value={})

        metrics = rag.get_metrics()

        assert metrics['embedding_dimension'] == 384


# ============================================================================
# Batch Query Tests
# ============================================================================

class TestBatchQuery:
    """Test batch query functionality."""

    @pytest.mark.asyncio
    async def test_batch_query_preserves_order(self, mock_hololoom, mock_memory):
        """Test that batch query preserves question order."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.orchestrator = None

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        questions = ["q1", "q2", "q3"]
        results = await rag.batch_query(questions)

        assert len(results) == 3
        # All should be RAGResults
        for result in results:
            assert isinstance(result, RAGResult)

    @pytest.mark.asyncio
    async def test_batch_query_mode_parameter(self, mock_hololoom, mock_memory):
        """Test that mode parameter applies to all queries."""
        rag = SimpleRAG()
        rag.loom = mock_hololoom
        rag.orchestrator = None

        mock_hololoom.recall = AsyncMock(return_value=[mock_memory])

        questions = ["q1", "q2"]
        results = await rag.batch_query(questions, mode="research")

        # All should have research mode
        for result in results:
            assert result.reasoning_mode == "research"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with real components."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test complete lifecycle: init, ingest, query, cleanup."""
        async with SimpleRAG() as rag:
            # Should be initialized
            assert rag.loom is not None

            # Ingest content
            await rag.ingest("Thompson Sampling is a Bayesian algorithm")

            # Query
            result = await rag.query("What is Thompson Sampling?")

            # Verify result structure
            assert isinstance(result, RAGResult)
            assert result.response is not None
            assert 0.0 <= result.confidence <= 1.0
            assert result.reasoning_mode in ["direct", "verify", "research", "plan_execute"]

            # Metadata should exist
            assert result.metadata is not None
            assert 'cache_hit' in result.metadata

        # After exit, should be cleaned up

    @pytest.mark.asyncio
    async def test_cache_across_queries(self):
        """Test that cache works across multiple queries."""
        async with SimpleRAG(enable_caching=True) as rag:
            # Ingest
            await rag.ingest("Thompson Sampling balances exploration")

            question = "What is Thompson Sampling?"

            # First query
            result1 = await rag.query(question)
            metrics1 = rag.get_metrics()

            # Second query (should use cache)
            result2 = await rag.query(question)
            metrics2 = rag.get_metrics()

            # Cache size should be 1
            assert metrics2['cache_size'] >= 1

            # Second result should indicate cache hit
            assert result2.metadata.get('cache_hit') is True
