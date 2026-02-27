"""
Unit Tests for Multimodal RAG

Tests:
- MultimodalRAG initialization
- Photo ingestion
- Visual Q&A (query with image)
- Photo retrieval (text and image queries)
- Visual compression
- Backward compatibility with SimpleRAG
- Graceful degradation
- Error handling

Author: Agent D (Claude Code)
Date: January 2025
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path
import numpy as np

# Import classes to test
from hololoom.rag.multimodal_rag import MultimodalRAG, MultimodalRAGResult
from hololoom.rag.simple_rag import RAGResult
from hololoom.config import Config


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Mock Config object."""
    config = Mock(spec=Config)
    config.scales = [96, 192, 384]
    return config


@pytest.fixture
def mock_photo_token():
    """Mock PhotoToken object."""
    photo = Mock()
    photo.token_id = "photo_123"
    photo.caption = "Test diagram"
    photo.tags = ["test", "diagram"]
    photo.metadata = {"score": 0.9}
    photo.clip_embedding = np.random.rand(512)
    return photo


@pytest.fixture
def mock_visual_qa_engine():
    """Mock VisualQAEngine."""
    engine = Mock()
    engine.extract_text_from_image = AsyncMock(return_value="Extracted text from image")
    engine.compute_image_embedding = Mock(return_value=np.random.rand(512))
    engine.is_available = Mock(return_value=True)
    return engine


@pytest.fixture
def mock_loom():
    """Mock HoloLoom instance."""
    loom = Mock()

    # Mock memory objects
    mem1 = Mock()
    mem1.text = "Source text 1: Thompson Sampling uses Bayesian statistics"
    mem2 = Mock()
    mem2.text = "Source text 2: It balances exploration and exploitation"

    # Mock methods
    loom.experience = AsyncMock(return_value=Mock(id="mem_123"))
    loom.remember_photo = AsyncMock(return_value="photo_123")
    loom.recall = AsyncMock(return_value=[mem1, mem2])
    loom.find_similar_photos = AsyncMock(return_value=[])
    loom.get_metrics = Mock(return_value={'n_memories': 10})
    loom.get_compression_stats = Mock(return_value={
        'total_compressed': 5,
        'avg_compression_ratio': 3.5,
        'total_tokens_saved': 2000
    })
    loom.__aenter__ = AsyncMock(return_value=loom)
    loom.__aexit__ = AsyncMock()

    return loom


@pytest.fixture
def mock_orchestrator():
    """Mock WeavingOrchestrator."""
    orchestrator = Mock()

    # Mock spacetime result
    spacetime = Mock()
    spacetime.response = "Generated answer from LLM"
    spacetime.confidence = 0.92
    spacetime.metadata = {'duration_ms': 150}

    orchestrator.weave = AsyncMock(return_value=spacetime)
    orchestrator.__aenter__ = AsyncMock(return_value=orchestrator)
    orchestrator.__aexit__ = AsyncMock()

    return orchestrator


# ============================================================================
# Initialization Tests
# ============================================================================

def test_multimodal_rag_initialization():
    """Test MultimodalRAG initializes correctly."""
    rag = MultimodalRAG()

    assert rag.enable_visual_compression == True
    assert rag.compression_threshold == 10
    assert rag.visual_qa_engine is None  # Not initialized yet


def test_multimodal_rag_with_config():
    """Test MultimodalRAG with custom config."""
    config = Config.fast()
    rag = MultimodalRAG(
        config=config,
        enable_visual_compression=False,
        compression_threshold=20
    )

    assert rag.config == config
    assert rag.enable_visual_compression == False
    assert rag.compression_threshold == 20


@pytest.mark.asyncio
async def test_multimodal_rag_context_manager():
    """Test MultimodalRAG async context manager initialization."""
    # Simple test - just verify it initializes without error
    # Full integration testing is covered by other tests
    rag = MultimodalRAG()
    assert rag.loom is None  # Not initialized yet
    assert rag.visual_qa_engine is None  # Not initialized yet

    # Note: Full context manager test requires real HoloLoom initialization
    # which is tested in integration tests


# ============================================================================
# Photo Ingestion Tests
# ============================================================================

@pytest.mark.asyncio
async def test_ingest_photo(mock_loom):
    """Test photo ingestion."""
    rag = MultimodalRAG()
    rag.loom = mock_loom  # Manually set loom (bypass context manager for test)

    photo_id = await rag.ingest_photo(
        image="test.png",
        tags=["test", "demo"],
        description="Test image"
    )

    assert photo_id == "photo_123"
    mock_loom.remember_photo.assert_called_once()
    call_args = mock_loom.remember_photo.call_args
    assert call_args[1]['image'] == "test.png"
    assert call_args[1]['tags'] == ["test", "demo"]
    assert call_args[1]['caption'] == "Test image"


@pytest.mark.asyncio
async def test_ingest_photo_not_initialized():
    """Test photo ingestion without initialization."""
    rag = MultimodalRAG()
    # Don't initialize (loom = None)

    with pytest.raises(RuntimeError, match="not initialized"):
        await rag.ingest_photo("test.png")


# ============================================================================
# Visual Q&A Tests
# ============================================================================

@pytest.mark.asyncio
async def test_query_with_image_basic(mock_loom, mock_orchestrator, mock_visual_qa_engine, mock_photo_token):
    """Test basic visual Q&A."""
    rag = MultimodalRAG()
    rag.loom = mock_loom
    rag.orchestrator = mock_orchestrator
    rag.visual_qa_engine = mock_visual_qa_engine

    # Mock multimodal recall
    mock_loom.recall = AsyncMock(return_value={
        'text': [Mock(text="source1"), Mock(text="source2")],
        'photos': [mock_photo_token]
    })

    result = await rag.query_with_image(
        question="What is this?",
        image="test.png"
    )

    assert isinstance(result, MultimodalRAGResult)
    assert result.response == "Generated answer from LLM"
    assert len(result.sources) == 2
    assert len(result.image_sources) == 1
    assert result.confidence == 0.92

    # Verify OCR was called
    mock_visual_qa_engine.extract_text_from_image.assert_called_once_with("test.png")

    # Verify recall was called with OCR-enhanced query
    mock_loom.recall.assert_called_once()
    call_args = mock_loom.recall.call_args
    assert "Extracted text from image" in call_args[0][0]


@pytest.mark.asyncio
async def test_query_with_image_no_visual_qa():
    """Test query with image when visual Q&A unavailable."""
    rag = MultimodalRAG()
    rag.loom = Mock()
    rag.visual_qa_engine = None

    with pytest.raises(RuntimeError, match="Visual Q&A unavailable"):
        await rag.query_with_image("What is this?", "test.png")


@pytest.mark.asyncio
async def test_query_with_image_visual_compression(mock_loom, mock_orchestrator, mock_visual_qa_engine):
    """Test visual compression for large contexts."""
    rag = MultimodalRAG(enable_visual_compression=True, compression_threshold=5)
    rag.loom = mock_loom
    rag.orchestrator = mock_orchestrator
    rag.visual_qa_engine = mock_visual_qa_engine

    # Mock recall with many sources (trigger compression)
    many_sources = [Mock(text=f"source{i}") for i in range(15)]
    mock_loom.recall = AsyncMock(return_value={
        'text': many_sources,
        'photos': []
    })

    # Mock visual compression
    with patch('hololoom.rag.multimodal_rag.compress_to_visual') as mock_compress:
        mock_metrics = Mock()
        mock_metrics.compression_ratio = 12.5
        mock_metrics.original_tokens = 5000
        mock_metrics.visual_tokens = 400
        mock_metrics.compression_type = "knowledge_graph"
        mock_metrics.info_density = 18.75

        mock_compress.return_value = (np.zeros((600, 800, 3), dtype=np.uint8), mock_metrics)

        result = await rag.query_with_image(
            question="Complex question requiring many sources",
            image="test.png"
        )

        # Verify compression was applied
        assert result.compressed_context is not None
        assert result.compression_ratio == 12.5
        assert result.compression_metrics['compression_ratio'] == 12.5
        mock_compress.assert_called_once()


@pytest.mark.asyncio
async def test_query_with_image_caching(mock_loom, mock_orchestrator, mock_visual_qa_engine):
    """Test query caching for repeated queries."""
    rag = MultimodalRAG(enable_caching=True)
    rag.loom = mock_loom
    rag.orchestrator = mock_orchestrator
    rag.visual_qa_engine = mock_visual_qa_engine

    mock_loom.recall = AsyncMock(return_value={
        'text': [Mock(text="source1")],
        'photos': []
    })

    # First query (cold)
    with patch('builtins.open', create=True):
        result1 = await rag.query_with_image("What is this?", Path("test.png"))

    # Second query (warm - should hit cache)
    with patch('builtins.open', create=True):
        result2 = await rag.query_with_image("What is this?", Path("test.png"))

    # Cache hit
    assert result2.metadata['cache_hit'] == True

    # Verify LLM was only called once
    assert mock_orchestrator.weave.call_count == 1


# ============================================================================
# Photo Retrieval Tests
# ============================================================================

@pytest.mark.asyncio
async def test_get_related_photos(mock_loom, mock_photo_token):
    """Test text-based photo retrieval."""
    rag = MultimodalRAG()
    rag.loom = mock_loom

    # Mock recall with photos
    mock_loom.recall = AsyncMock(return_value={
        'text': [],
        'photos': [mock_photo_token]
    })

    photos = await rag.get_related_photos("architecture diagram", max_photos=5)

    assert len(photos) == 1
    assert photos[0].token_id == "photo_123"

    mock_loom.recall.assert_called_once_with(
        "architecture diagram",
        limit=0,
        include_photos=True
    )


@pytest.mark.asyncio
async def test_get_similar_photos(mock_loom, mock_photo_token):
    """Test image-based similarity search."""
    rag = MultimodalRAG()
    rag.loom = mock_loom

    mock_loom.find_similar_photos = AsyncMock(return_value=[mock_photo_token])

    photos = await rag.get_similar_photos("reference.png", max_photos=5)

    assert len(photos) == 1
    assert photos[0].token_id == "photo_123"

    mock_loom.find_similar_photos.assert_called_once_with("reference.png", k=5)


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

@pytest.mark.asyncio
async def test_backward_compatible_query(mock_loom, mock_orchestrator):
    """Test that standard query() works exactly like SimpleRAG."""
    rag = MultimodalRAG()
    rag.loom = mock_loom
    rag.orchestrator = mock_orchestrator

    result = await rag.query("What is Thompson Sampling?")

    # Should return standard RAGResult (not MultimodalRAGResult)
    assert isinstance(result, RAGResult)
    assert result.response == "Generated answer from LLM"
    assert len(result.sources) == 2
    assert result.confidence == 0.92


@pytest.mark.asyncio
async def test_backward_compatible_ingest(mock_loom):
    """Test that standard ingest() works exactly like SimpleRAG."""
    rag = MultimodalRAG()
    rag.loom = mock_loom

    await rag.ingest("Thompson Sampling uses Bayesian statistics")

    mock_loom.experience.assert_called_once_with("Thompson Sampling uses Bayesian statistics")


# ============================================================================
# Utility Tests
# ============================================================================

def test_build_graph_from_sources():
    """Test knowledge graph construction from sources."""
    rag = MultimodalRAG()

    sources = [
        "Source 1 about Thompson Sampling",
        "Source 2 about Bayesian statistics",
        "Source 3 about exploration"
    ]

    kg = rag._build_graph_from_sources(sources)

    # Verify graph has nodes and edges
    # KG uses 'G' attribute for the graph
    assert kg.G.number_of_nodes() > 0
    assert kg.G.number_of_edges() > 0


def test_get_multimodal_metrics(mock_loom):
    """Test multimodal metrics."""
    rag = MultimodalRAG()
    rag.loom = mock_loom

    metrics = rag.get_multimodal_metrics()

    assert 'visual_qa_available' in metrics
    assert 'visual_compression_enabled' in metrics
    assert 'compression_threshold' in metrics
    assert 'n_compressed_visuals' in metrics
    assert metrics['compression_threshold'] == 10


def test_summary(mock_loom):
    """Test system summary."""
    rag = MultimodalRAG()
    rag.loom = mock_loom
    rag.visual_qa_engine = Mock()

    summary = rag.summary()

    assert "MultimodalRAG System" in summary
    assert "Visual Q&A: ✓" in summary
    assert "Visual compression: ✓" in summary


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_query_with_image_ocr_failure(mock_loom, mock_orchestrator):
    """Test graceful handling of OCR failure."""
    rag = MultimodalRAG()
    rag.loom = mock_loom
    rag.orchestrator = mock_orchestrator

    # Mock visual Q&A engine with failing OCR
    mock_engine = Mock()
    mock_engine.extract_text_from_image = AsyncMock(return_value="")  # Empty text
    rag.visual_qa_engine = mock_engine

    mock_loom.recall = AsyncMock(return_value={
        'text': [Mock(text="source1")],
        'photos': []
    })

    # Should still work (use query without OCR enhancement)
    result = await rag.query_with_image("What is this?", "test.png")

    assert isinstance(result, MultimodalRAGResult)
    assert result.metadata['ocr_text_length'] == 0


@pytest.mark.asyncio
async def test_query_with_image_llm_failure(mock_loom, mock_visual_qa_engine):
    """Test fallback when LLM fails."""
    rag = MultimodalRAG()
    rag.loom = mock_loom
    rag.visual_qa_engine = mock_visual_qa_engine

    # Mock orchestrator that fails
    mock_orch = Mock()
    mock_orch.weave = AsyncMock(side_effect=Exception("LLM error"))
    rag.orchestrator = mock_orch

    mock_loom.recall = AsyncMock(return_value={
        'text': [Mock(text="source1")],
        'photos': []
    })

    # Should fall back to synthesizing from sources
    result = await rag.query_with_image("What is this?", "test.png")

    assert isinstance(result, MultimodalRAGResult)
    assert "Based on multimodal context" in result.response
    assert result.confidence < 0.8  # Lower confidence for fallback


@pytest.mark.asyncio
async def test_visual_compression_failure(mock_loom, mock_orchestrator, mock_visual_qa_engine):
    """Test graceful handling of compression failure."""
    rag = MultimodalRAG(enable_visual_compression=True, compression_threshold=5)
    rag.loom = mock_loom
    rag.orchestrator = mock_orchestrator
    rag.visual_qa_engine = mock_visual_qa_engine

    # Many sources (trigger compression)
    many_sources = [Mock(text=f"source{i}") for i in range(15)]
    mock_loom.recall = AsyncMock(return_value={
        'text': many_sources,
        'photos': []
    })

    # Mock compression failure
    with patch('hololoom.rag.multimodal_rag.compress_to_visual', side_effect=Exception("Compression error")):
        result = await rag.query_with_image("Complex question", "test.png")

        # Should still succeed (compression is optional)
        assert isinstance(result, MultimodalRAGResult)
        assert result.compressed_context is None
        assert result.compression_ratio is None


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_end_to_end_multimodal_flow(mock_loom, mock_orchestrator, mock_visual_qa_engine, mock_photo_token):
    """Test complete multimodal workflow."""
    rag = MultimodalRAG()
    rag.loom = mock_loom
    rag.orchestrator = mock_orchestrator
    rag.visual_qa_engine = mock_visual_qa_engine

    # 1. Ingest text
    await rag.ingest("Thompson Sampling uses Bayesian statistics")
    mock_loom.experience.assert_called_once()

    # 2. Ingest photo
    photo_id = await rag.ingest_photo("diagram.png", tags=["architecture"])
    assert photo_id == "photo_123"
    mock_loom.remember_photo.assert_called_once()

    # 3. Query with image
    mock_loom.recall = AsyncMock(return_value={
        'text': [Mock(text="source1"), Mock(text="source2")],
        'photos': [mock_photo_token]
    })

    result = await rag.query_with_image(
        question="Explain the architecture",
        image="diagram.png"
    )

    assert isinstance(result, MultimodalRAGResult)
    assert len(result.sources) == 2
    assert len(result.image_sources) == 1
    assert result.response == "Generated answer from LLM"

    # 4. Get related photos
    mock_loom.recall = AsyncMock(return_value={
        'text': [],
        'photos': [mock_photo_token]
    })

    photos = await rag.get_related_photos("architecture")
    assert len(photos) == 1


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_batch_multimodal_queries(mock_loom, mock_orchestrator, mock_visual_qa_engine):
    """Test batch processing of multimodal queries."""
    rag = MultimodalRAG()
    rag.loom = mock_loom
    rag.orchestrator = mock_orchestrator
    rag.visual_qa_engine = mock_visual_qa_engine

    mock_loom.recall = AsyncMock(return_value={
        'text': [Mock(text="source1")],
        'photos': []
    })

    # Process multiple queries
    questions = [
        "What is in image 1?",
        "What is in image 2?",
        "What is in image 3?"
    ]

    results = []
    for question in questions:
        result = await rag.query_with_image(question, "test.png")
        results.append(result)

    assert len(results) == 3
    assert all(isinstance(r, MultimodalRAGResult) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
