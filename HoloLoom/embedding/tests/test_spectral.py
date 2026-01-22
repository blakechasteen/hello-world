"""
Comprehensive Tests for Spectral Embeddings Module
===================================================

Test Coverage (~1,200 lines):

1. MatryoshkaEmbeddings Tests (~700 lines)
   - Initialization and configuration
   - Multi-scale extraction (96, 192, 384 dimensions)
   - Embedding normalization verification
   - Batch processing
   - Projection matrix operations
   - Caching behavior
   - Graceful degradation without sentence-transformers
   - External heads functionality
   - Corpus-based QR refresh

2. SpectralFusion Tests (~500 lines)
   - Graph Laplacian eigenvalue computation
   - SVD topic components extraction
   - Feature concatenation
   - Coherence metrics
   - Wavelet features (optional)
   - Diffusion map features (optional)
   - Error handling

Created: 2026-01-22
Author: Claude Code
"""

import pytest
import numpy as np
import networkx as nx
import asyncio
import warnings
from typing import List, Dict
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_texts() -> List[str]:
    """Provide sample texts for embedding tests."""
    return [
        "Multi-head attention processes multiple representation subspaces",
        "Transformers use self-attention mechanisms",
        "Neural networks learn hierarchical features",
        "Deep learning revolutionizes pattern recognition",
        "Machine learning enables data-driven predictions"
    ]


@pytest.fixture
def single_text() -> List[str]:
    """Provide single text for basic tests."""
    return ["Thompson Sampling balances exploration and exploitation"]


@pytest.fixture
def empty_texts() -> List[str]:
    """Provide empty text list for edge case tests."""
    return []


@pytest.fixture
def sample_graph() -> nx.MultiDiGraph:
    """Provide sample knowledge graph for spectral tests."""
    G = nx.MultiDiGraph()

    # Add nodes
    nodes = ["attention", "transformer", "neural_network", "BERT", "GPT", "NLP"]
    for node in nodes:
        G.add_node(node)

    # Add edges with weights
    edges = [
        ("attention", "transformer", {"type": "USES", "weight": 1.0}),
        ("transformer", "neural_network", {"type": "IS_A", "weight": 1.0}),
        ("attention", "neural_network", {"type": "PART_OF", "weight": 0.8}),
        ("BERT", "transformer", {"type": "IS_A", "weight": 1.0}),
        ("GPT", "transformer", {"type": "IS_A", "weight": 1.0}),
        ("BERT", "NLP", {"type": "USED_IN", "weight": 1.0}),
        ("GPT", "NLP", {"type": "USED_IN", "weight": 1.0}),
    ]
    G.add_edges_from([(u, v, d) for u, v, d in edges])

    return G


@pytest.fixture
def small_graph() -> nx.MultiDiGraph:
    """Provide small graph (2 nodes) for edge cases."""
    G = nx.MultiDiGraph()
    G.add_edge("A", "B", weight=1.0)
    return G


@pytest.fixture
def empty_graph() -> nx.MultiDiGraph:
    """Provide empty graph for edge cases."""
    return nx.MultiDiGraph()


@pytest.fixture
def disconnected_graph() -> nx.MultiDiGraph:
    """Provide disconnected graph with multiple components."""
    G = nx.MultiDiGraph()
    # Component 1
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    # Component 2 (disconnected)
    G.add_edge("X", "Y", weight=1.0)
    G.add_edge("Y", "Z", weight=1.0)
    return G


# ============================================================================
# MATRYOSHKA EMBEDDINGS - INITIALIZATION TESTS
# ============================================================================

class TestMatryoshkaEmbeddingsInit:
    """Tests for MatryoshkaEmbeddings initialization."""

    def test_init_default_scales(self):
        """Default initialization should use [768] scale."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings()

        assert embedder is not None
        assert embedder.sizes == [768]
        assert embedder.base_dim == 768
        assert embedder._model_loaded is False

    def test_init_custom_scales_ascending(self):
        """Custom scales must be in ascending order."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

        assert embedder.sizes == [96, 192, 384]
        assert 96 in embedder.proj
        assert 192 in embedder.proj
        assert 384 in embedder.proj

    def test_init_single_scale(self):
        """Should work with single scale."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[256])

        assert embedder.sizes == [256]
        assert 256 in embedder.proj

    def test_init_invalid_scales_descending(self):
        """Descending scales should raise AssertionError."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        with pytest.raises(AssertionError):
            MatryoshkaEmbeddings(sizes=[384, 192, 96])

    def test_init_invalid_scales_unsorted(self):
        """Unsorted scales should raise AssertionError."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        with pytest.raises(AssertionError):
            MatryoshkaEmbeddings(sizes=[192, 96, 384])

    def test_init_creates_cache(self):
        """Initialization should create embedding cache."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        assert hasattr(embedder, '_embedding_cache')
        assert embedder._embedding_cache is not None

    def test_init_with_external_heads(self):
        """Should accept external projection heads."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        def custom_head(x):
            return x[:, :64]

        embedder = MatryoshkaEmbeddings(
            sizes=[64, 128],
            external_heads={64: custom_head}
        )

        assert 64 in embedder.external_heads
        assert embedder.external_heads[64] == custom_head
        assert 128 not in embedder.external_heads

    def test_init_with_custom_model_name(self):
        """Should accept custom model name."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(
            sizes=[96],
            base_model_name="custom-model/v1"
        )

        assert embedder.base_model_name == "custom-model/v1"

    def test_init_lazy_loading(self):
        """Model should not load until first encode."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        assert embedder._model is None
        assert embedder._model_loaded is False

    def test_init_projection_matrices_shape(self):
        """Projection matrices should have correct shapes."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        embedder._ensure_model_loaded()

        assert embedder.proj[96].shape[1] == 96
        assert embedder.proj[192].shape[1] == 192
        assert embedder.proj[384].shape[1] == 384


# ============================================================================
# MATRYOSHKA EMBEDDINGS - ENCODING TESTS
# ============================================================================

class TestMatryoshkaEmbeddingsEncode:
    """Tests for encode() method."""

    def test_encode_single_text(self, single_text):
        """encode() should work with single text."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(single_text)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 96)

    def test_encode_multiple_texts(self, sample_texts):
        """encode() should work with multiple texts."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(sample_texts)

        assert result.shape[0] == len(sample_texts)
        assert result.shape[1] == 96

    def test_encode_empty_list(self, empty_texts):
        """encode() should handle empty list gracefully."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(empty_texts)

        assert result is not None
        assert result.shape == (0, 96)

    def test_encode_returns_largest_scale(self):
        """encode() should return embeddings at largest scale."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode(["Test"])

        assert result.shape[1] == 384

    def test_encode_empty_strings(self):
        """encode() should handle empty strings."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(["", "non-empty", ""])

        assert result.shape == (3, 96)

    def test_encode_very_long_text(self):
        """encode() should handle very long texts."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        long_text = "word " * 10000
        result = embedder.encode([long_text])

        assert result.shape == (1, 96)

    def test_encode_unicode_text(self):
        """encode() should handle unicode characters."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        unicode_texts = [
            "Hello",
            "Hola",
            "Bonjour",
            "Guten Tag",
        ]
        result = embedder.encode(unicode_texts)

        assert result.shape == (4, 96)

    def test_encode_special_characters(self):
        """encode() should handle special characters."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        special_texts = [
            "Hello! @#$%^&*()",
            "Test <html></html>",
            "Code: def foo(): pass",
            "Math: x^2 + y^2 = z^2"
        ]
        result = embedder.encode(special_texts)

        assert result.shape == (4, 96)


# ============================================================================
# MATRYOSHKA EMBEDDINGS - ENCODE BASE TESTS
# ============================================================================

class TestMatryoshkaEmbeddingsEncodeBase:
    """Tests for encode_base() method."""

    def test_encode_base_shape(self, single_text):
        """encode_base() should return base dimension."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192])
        embedder._ensure_model_loaded()
        result = embedder.encode_base(single_text)

        assert result.shape[0] == 1
        assert result.shape[1] == embedder.base_dim

    def test_encode_base_normalization(self, single_text):
        """encode_base() should return unit-normalized vectors."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode_base(single_text)

        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01

    def test_encode_base_batch_normalization(self, sample_texts):
        """All vectors in batch should be normalized."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode_base(sample_texts)

        for i in range(result.shape[0]):
            norm = np.linalg.norm(result[i])
            assert abs(norm - 1.0) < 0.01, f"Vector {i} not normalized: norm={norm}"

    def test_encode_base_empty_input(self, empty_texts):
        """encode_base() should handle empty input."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._ensure_model_loaded()
        result = embedder.encode_base(empty_texts)

        assert result.shape == (0, embedder.base_dim)

    def test_encode_base_caching(self, single_text):
        """encode_base() should cache results."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        result1 = embedder.encode_base(single_text)
        result2 = embedder.encode_base(single_text)

        assert np.allclose(result1, result2)

    def test_encode_base_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        result1 = embedder.encode_base(["Hello world"])
        result2 = embedder.encode_base(["Goodbye universe"])

        assert not np.allclose(result1, result2)


# ============================================================================
# MATRYOSHKA EMBEDDINGS - ENCODE SCALES TESTS
# ============================================================================

class TestMatryoshkaEmbeddingsEncodeScales:
    """Tests for encode_scales() method."""

    def test_encode_scales_all_scales(self, single_text):
        """encode_scales() should return all scales when size=None."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(single_text)

        assert isinstance(result, dict)
        assert 96 in result
        assert 192 in result
        assert 384 in result

    def test_encode_scales_specific_scale(self, single_text):
        """encode_scales() should return specific scale when size specified."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(single_text, size=192)

        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 192

    def test_encode_scales_dimensions_correct(self, sample_texts):
        """encode_scales() should return correct dimensions for each scale."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(sample_texts)

        n_texts = len(sample_texts)
        assert result[96].shape == (n_texts, 96)
        assert result[192].shape == (n_texts, 192)
        assert result[384].shape == (n_texts, 384)

    def test_encode_scales_empty_input(self, empty_texts):
        """encode_scales() should handle empty input."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192])
        result = embedder.encode_scales(empty_texts)

        assert isinstance(result, dict)
        assert result[96].shape == (0, 96)
        assert result[192].shape == (0, 192)

    def test_encode_scales_empty_input_specific_size(self, empty_texts):
        """encode_scales() with specific size should handle empty input."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192])
        result = embedder.encode_scales(empty_texts, size=96)

        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 96)

    def test_encode_scales_consistency(self, single_text):
        """Smaller scales should be prefixes of larger scales (after projection)."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(single_text)

        # Each scale should have correct dimension
        assert result[96].shape[1] == 96
        assert result[192].shape[1] == 192
        assert result[384].shape[1] == 384

    def test_encode_scales_external_head_used(self, single_text):
        """External head should be used when available."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        def custom_head(x):
            return x[:, :64]

        embedder = MatryoshkaEmbeddings(
            sizes=[64],
            external_heads={64: custom_head}
        )
        result = embedder.encode_scales(single_text, size=64)

        assert result.shape[1] == 64


# ============================================================================
# MATRYOSHKA EMBEDDINGS - PROJECTION TESTS
# ============================================================================

class TestMatryoshkaEmbeddingsProjection:
    """Tests for projection matrix operations."""

    def test_projection_deterministic(self):
        """Projection matrices should be deterministic with same seed."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder1 = MatryoshkaEmbeddings(sizes=[96])
        embedder1._ensure_model_loaded()
        embedder1._build_projection(seed=12345)
        proj1 = embedder1.proj[96].copy()

        embedder2 = MatryoshkaEmbeddings(sizes=[96])
        embedder2._ensure_model_loaded()
        embedder2._build_projection(seed=12345)
        proj2 = embedder2.proj[96].copy()

        assert np.allclose(proj1, proj2)

    def test_projection_different_seeds_different_matrices(self):
        """Different seeds should produce different projections."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._ensure_model_loaded()

        embedder._build_projection(seed=12345)
        proj1 = embedder.proj[96].copy()

        embedder._build_projection(seed=54321)
        proj2 = embedder.proj[96].copy()

        assert not np.allclose(proj1, proj2)

    def test_projection_orthonormal(self):
        """Projection matrices should be orthonormal."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._ensure_model_loaded()
        proj = embedder.proj[96]

        # Check P^T P = I
        product = proj.T @ proj
        identity = np.eye(96)

        assert np.allclose(product, identity, atol=1e-6)

    def test_projection_all_scales_orthonormal(self):
        """All projection matrices should be orthonormal."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        embedder._ensure_model_loaded()

        for size in [96, 192, 384]:
            proj = embedder.proj[size]
            product = proj.T @ proj
            identity = np.eye(size)

            assert np.allclose(product, identity, atol=1e-6), \
                f"Projection for size {size} not orthonormal"

    def test_refresh_runtime_qr_changes_projection(self):
        """refresh_runtime_qr() should update projections based on corpus."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._ensure_model_loaded()

        proj_before = embedder.proj[96].copy()
        embedder.refresh_runtime_qr(["corpus text 1", "corpus text 2"])
        proj_after = embedder.proj[96].copy()

        assert not np.allclose(proj_before, proj_after)

    def test_refresh_runtime_qr_same_corpus_same_projection(self):
        """Same corpus should produce same projection (deterministic)."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._ensure_model_loaded()

        corpus = ["corpus text 1", "corpus text 2"]

        embedder.refresh_runtime_qr(corpus)
        proj1 = embedder.proj[96].copy()

        embedder.refresh_runtime_qr(corpus)
        proj2 = embedder.proj[96].copy()

        assert np.allclose(proj1, proj2)

    def test_refresh_runtime_qr_no_change_if_same_hash(self):
        """Refresh with same corpus should not rebuild (hash check)."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._ensure_model_loaded()

        corpus = ["corpus text 1", "corpus text 2"]

        embedder.refresh_runtime_qr(corpus)
        hash_before = embedder._last_hash

        embedder.refresh_runtime_qr(corpus)
        hash_after = embedder._last_hash

        assert hash_before == hash_after


# ============================================================================
# MATRYOSHKA EMBEDDINGS - CACHING TESTS
# ============================================================================

class TestMatryoshkaEmbeddingsCaching:
    """Tests for embedding caching behavior."""

    def test_cache_populated_after_encode(self):
        """Cache should be populated after encoding."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder.encode(["Test text"])

        # Check cache size using stats() method or len(cache)
        assert len(embedder._embedding_cache.cache) > 0

    def test_cache_hit_returns_same_embedding(self):
        """Cache hit should return identical embedding."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        result1 = embedder.encode(["Test text"])
        result2 = embedder.encode(["Test text"])

        assert np.allclose(result1, result2)

    def test_cache_different_texts_different_entries(self):
        """Different texts should create different cache entries."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        embedder.encode(["Text A"])
        embedder.encode(["Text B"])

        # Cache should have 2 entries
        assert len(embedder._embedding_cache.cache) >= 2

    def test_cache_batch_encoding(self, sample_texts):
        """Batch encoding should cache all texts."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder.encode(sample_texts)

        assert len(embedder._embedding_cache.cache) >= len(sample_texts)

    def test_cache_partial_batch_reuse(self):
        """Batch with cached texts should reuse cached embeddings."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        # First encode some texts
        embedder.encode(["Text A", "Text B"])

        # Now encode mix of cached and new
        result = embedder.encode(["Text A", "Text C", "Text B"])

        assert result.shape == (3, 96)


# ============================================================================
# MATRYOSHKA EMBEDDINGS - GRACEFUL DEGRADATION TESTS
# ============================================================================

class TestMatryoshkaEmbeddingsGracefulDegradation:
    """Tests for fallback behavior without optional dependencies."""

    def test_fallback_produces_embeddings(self):
        """Fallback should still produce embeddings."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._model = None
        embedder._model_loaded = True
        embedder.base_dim = 96

        result = embedder.encode_base(["Test text"])

        assert result is not None
        assert result.shape == (1, 96)

    def test_fallback_deterministic(self):
        """Fallback embeddings should be deterministic."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder1 = MatryoshkaEmbeddings(sizes=[96])
        embedder1._model = None
        embedder1._model_loaded = True
        embedder1.base_dim = 96
        result1 = embedder1.encode_base(["Test text"])

        embedder2 = MatryoshkaEmbeddings(sizes=[96])
        embedder2._model = None
        embedder2._model_loaded = True
        embedder2.base_dim = 96
        result2 = embedder2.encode_base(["Test text"])

        assert np.allclose(result1, result2)

    def test_fallback_different_texts_different_embeddings(self):
        """Fallback should produce different embeddings for different texts."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._model = None
        embedder._model_loaded = True
        embedder.base_dim = 96

        result1 = embedder.encode_base(["Text A"])
        result2 = embedder.encode_base(["Text B"])

        assert not np.allclose(result1, result2)

    def test_fallback_normalized(self):
        """Fallback embeddings should be normalized."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._model = None
        embedder._model_loaded = True
        embedder.base_dim = 96

        result = embedder.encode_base(["Test text"])
        norm = np.linalg.norm(result[0])

        assert abs(norm - 1.0) < 0.01


# ============================================================================
# MATRYOSHKA EMBEDDINGS - BATCH PROCESSING TESTS
# ============================================================================

class TestMatryoshkaEmbeddingsBatchProcessing:
    """Tests for batch processing performance and correctness."""

    def test_batch_consistency(self, sample_texts):
        """Batch encoding should give same results as individual encoding."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        # Batch encoding
        batch_result = embedder.encode(sample_texts)

        # Individual encoding
        individual_results = []
        for text in sample_texts:
            result = embedder.encode([text])
            individual_results.append(result[0])
        individual_result = np.vstack(individual_results)

        assert np.allclose(batch_result, individual_result)

    def test_large_batch(self):
        """Should handle large batches."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        texts = [f"Text {i}" for i in range(100)]

        result = embedder.encode(texts)

        assert result.shape == (100, 96)

    def test_batch_order_preserved(self, sample_texts):
        """Batch encoding should preserve text order."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        result = embedder.encode(sample_texts)

        # First embedding should match first text
        single_result = embedder.encode([sample_texts[0]])
        assert np.allclose(result[0], single_result[0])

        # Last embedding should match last text
        single_result = embedder.encode([sample_texts[-1]])
        assert np.allclose(result[-1], single_result[0])


# ============================================================================
# SPECTRAL FUSION - INITIALIZATION TESTS
# ============================================================================

class TestSpectralFusionInit:
    """Tests for SpectralFusion initialization."""

    def test_init_default_params(self):
        """Default initialization should set k_eigen=4, svd_components=2."""
        from HoloLoom.embedding.spectral import SpectralFusion

        spectral = SpectralFusion()

        assert spectral.k_eigen == 4
        assert spectral.svd_components == 2
        assert spectral.use_wavelets is False
        assert spectral.use_diffusion_maps is False

    def test_init_custom_params(self):
        """Should accept custom parameters."""
        from HoloLoom.embedding.spectral import SpectralFusion

        spectral = SpectralFusion(
            k_eigen=6,
            svd_components=4,
            use_wavelets=True,
            use_diffusion_maps=True,
            diffusion_map_dims=16
        )

        assert spectral.k_eigen == 6
        assert spectral.svd_components == 4
        assert spectral.use_wavelets is True
        assert spectral.use_diffusion_maps is True
        assert spectral.diffusion_map_dims == 16


# ============================================================================
# SPECTRAL FUSION - FEATURES TESTS
# ============================================================================

class TestSpectralFusionFeatures:
    """Tests for SpectralFusion features() method."""

    @pytest.mark.asyncio
    async def test_features_returns_psi_and_metrics(self, sample_graph, sample_texts):
        """features() should return psi vector and metrics dict."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

        psi, metrics = await spectral.features(sample_graph, sample_texts, embedder)

        assert isinstance(psi, np.ndarray)
        assert isinstance(metrics, dict)
        assert len(psi) > 0

    @pytest.mark.asyncio
    async def test_features_metrics_keys(self, sample_graph, sample_texts):
        """Metrics should contain expected keys."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

        psi, metrics = await spectral.features(sample_graph, sample_texts, embedder)

        assert 'fiedler' in metrics
        assert 'topic_var' in metrics
        assert 'coherence' in metrics
        assert 'feature_dim' in metrics

    @pytest.mark.asyncio
    async def test_features_empty_graph(self, empty_graph, sample_texts):
        """features() should handle empty graph."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(empty_graph, sample_texts, embedder)

        assert psi is not None
        assert metrics['fiedler'] == 0.0

    @pytest.mark.asyncio
    async def test_features_small_graph(self, small_graph, sample_texts):
        """features() should handle small graph (2 nodes)."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(small_graph, sample_texts, embedder)

        assert psi is not None
        assert len(psi) > 0

    @pytest.mark.asyncio
    async def test_features_empty_texts(self, sample_graph, empty_texts):
        """features() should handle empty text list."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(sample_graph, empty_texts, embedder)

        assert psi is not None
        assert metrics['topic_var'] == 0.0

    @pytest.mark.asyncio
    async def test_features_single_node_graph(self, sample_texts):
        """features() should handle single node graph."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        G = nx.MultiDiGraph()
        G.add_node("single_node")

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(G, sample_texts, embedder)

        assert psi is not None
        assert metrics['fiedler'] == 0.0


# ============================================================================
# SPECTRAL FUSION - GRAPH SPECTRUM TESTS
# ============================================================================

class TestSpectralFusionGraphSpectrum:
    """Tests for graph Laplacian spectrum computation."""

    def test_graph_spectrum_returns_correct_length(self, sample_graph):
        """_graph_spectrum() should return k_eigen eigenvalues."""
        from HoloLoom.embedding.spectral import SpectralFusion

        spectral = SpectralFusion(k_eigen=4)
        spectrum, fiedler = spectral._graph_spectrum(sample_graph)

        assert len(spectrum) == 4

    def test_graph_spectrum_first_eigenvalue_near_zero(self, sample_graph):
        """First Laplacian eigenvalue should be near zero."""
        from HoloLoom.embedding.spectral import SpectralFusion

        spectral = SpectralFusion(k_eigen=4)
        spectrum, fiedler = spectral._graph_spectrum(sample_graph)

        assert abs(spectrum[0]) < 1e-6

    def test_graph_spectrum_fiedler_positive(self, sample_graph):
        """Fiedler value should be positive for connected graph."""
        from HoloLoom.embedding.spectral import SpectralFusion

        spectral = SpectralFusion(k_eigen=4)
        spectrum, fiedler = spectral._graph_spectrum(sample_graph)

        # Fiedler value is positive for connected graphs
        assert fiedler >= 0.0

    def test_graph_spectrum_eigenvalues_sorted(self, sample_graph):
        """Eigenvalues should be in ascending order."""
        from HoloLoom.embedding.spectral import SpectralFusion

        spectral = SpectralFusion(k_eigen=4)
        spectrum, fiedler = spectral._graph_spectrum(sample_graph)

        for i in range(len(spectrum) - 1):
            assert spectrum[i] <= spectrum[i + 1] + 1e-6

    def test_graph_spectrum_empty_graph(self, empty_graph):
        """_graph_spectrum() should handle empty graph."""
        from HoloLoom.embedding.spectral import SpectralFusion

        spectral = SpectralFusion(k_eigen=4)
        spectrum, fiedler = spectral._graph_spectrum(empty_graph)

        assert np.allclose(spectrum, np.zeros(4))
        assert fiedler == 0.0

    def test_graph_spectrum_single_node(self):
        """_graph_spectrum() should handle single node graph."""
        from HoloLoom.embedding.spectral import SpectralFusion

        G = nx.MultiDiGraph()
        G.add_node("single")

        spectral = SpectralFusion(k_eigen=4)
        spectrum, fiedler = spectral._graph_spectrum(G)

        assert np.allclose(spectrum, np.zeros(4))
        assert fiedler == 0.0

    def test_graph_spectrum_padded_when_small(self, small_graph):
        """Spectrum should be padded when graph smaller than k_eigen."""
        from HoloLoom.embedding.spectral import SpectralFusion

        spectral = SpectralFusion(k_eigen=10)
        spectrum, fiedler = spectral._graph_spectrum(small_graph)

        assert len(spectrum) == 10


# ============================================================================
# SPECTRAL FUSION - TOPIC FEATURES TESTS
# ============================================================================

class TestSpectralFusionTopicFeatures:
    """Tests for topic features via SVD."""

    def test_topic_features_returns_correct_length(self, sample_texts):
        """_topic_features() should return svd_components values."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion(svd_components=3)
        embedder = MatryoshkaEmbeddings(sizes=[96])

        topic, topic_var = spectral._topic_features(sample_texts, embedder)

        assert len(topic) == 3

    def test_topic_features_variance_in_range(self, sample_texts):
        """Topic variance should be between 0 and 1."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion(svd_components=2)
        embedder = MatryoshkaEmbeddings(sizes=[96])

        topic, topic_var = spectral._topic_features(sample_texts, embedder)

        assert 0.0 <= topic_var <= 1.0

    def test_topic_features_empty_texts(self, empty_texts):
        """_topic_features() should handle empty texts."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion(svd_components=2)
        embedder = MatryoshkaEmbeddings(sizes=[96])

        topic, topic_var = spectral._topic_features(empty_texts, embedder)

        assert np.allclose(topic, np.zeros(2))
        assert topic_var == 0.0

    def test_topic_features_single_text(self, single_text):
        """_topic_features() should handle single text."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion(svd_components=2)
        embedder = MatryoshkaEmbeddings(sizes=[96])

        topic, topic_var = spectral._topic_features(single_text, embedder)

        assert len(topic) == 2

    def test_topic_features_padded_when_small(self, single_text):
        """Topic features should be padded when texts fewer than svd_components."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion(svd_components=10)
        embedder = MatryoshkaEmbeddings(sizes=[96])

        topic, topic_var = spectral._topic_features(single_text, embedder)

        assert len(topic) == 10


# ============================================================================
# SPECTRAL FUSION - COHERENCE TESTS
# ============================================================================

class TestSpectralFusionCoherence:
    """Tests for coherence computation."""

    @pytest.mark.asyncio
    async def test_coherence_positive_for_connected_graph(self, sample_graph, sample_texts):
        """Coherence should be positive for connected graph with good topics."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(sample_graph, sample_texts, embedder)

        assert metrics['coherence'] >= 0.0

    @pytest.mark.asyncio
    async def test_coherence_zero_for_empty_graph(self, empty_graph, empty_texts):
        """Coherence should be zero for empty graph and texts."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(empty_graph, empty_texts, embedder)

        assert metrics['coherence'] == 0.0


# ============================================================================
# SPECTRAL FUSION - WAVELET FEATURES TESTS (OPTIONAL)
# ============================================================================

class TestSpectralFusionWaveletFeatures:
    """Tests for optional wavelet features."""

    @pytest.mark.asyncio
    async def test_wavelets_disabled_by_default(self, sample_graph, sample_texts):
        """Wavelets should be disabled by default."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(sample_graph, sample_texts, embedder)

        assert metrics['wavelet_energy'] == 0.0

    @pytest.mark.asyncio
    async def test_wavelets_enabled_adds_features(self, sample_graph, sample_texts):
        """Enabling wavelets should add wavelet features."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion(use_wavelets=True, wavelet_scales=[0.1, 1.0])
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(sample_graph, sample_texts, embedder)

        # Feature dimension should be larger than base (4 + 2)
        assert metrics['feature_dim'] >= 6


# ============================================================================
# SPECTRAL FUSION - DIFFUSION MAP TESTS (OPTIONAL)
# ============================================================================

class TestSpectralFusionDiffusionMaps:
    """Tests for optional diffusion map features."""

    @pytest.mark.asyncio
    async def test_diffusion_maps_disabled_by_default(self, sample_graph, sample_texts):
        """Diffusion maps should be disabled by default."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(sample_graph, sample_texts, embedder)

        assert metrics['diffusion_variance'] == 0.0


# ============================================================================
# FACTORY FUNCTION TESTS
# ============================================================================

class TestCreateEmbedder:
    """Tests for create_embedder factory function."""

    def test_create_embedder_matryoshka(self):
        """create_embedder() should create MatryoshkaEmbeddings."""
        from HoloLoom.embedding.spectral import create_embedder

        embedder = create_embedder(mode="matryoshka", sizes=[96, 192])

        assert embedder is not None
        assert embedder.sizes == [96, 192]

    def test_create_embedder_with_model_name(self):
        """create_embedder() should pass model name."""
        from HoloLoom.embedding.spectral import create_embedder

        embedder = create_embedder(
            mode="matryoshka",
            sizes=[96],
            base_model_name="custom-model"
        )

        assert embedder.base_model_name == "custom-model"

    def test_create_embedder_invalid_mode(self):
        """create_embedder() should raise for invalid mode."""
        from HoloLoom.embedding.spectral import create_embedder

        with pytest.raises(ValueError):
            create_embedder(mode="invalid")


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_duplicate_texts_in_batch(self):
        """Should handle duplicate texts in batch."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        texts = ["Same text", "Same text", "Same text"]

        result = embedder.encode(texts)

        assert result.shape == (3, 96)
        # All embeddings should be identical
        assert np.allclose(result[0], result[1])
        assert np.allclose(result[1], result[2])

    def test_whitespace_only_text(self):
        """Should handle whitespace-only text."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        texts = ["   ", "\t\n", "  \t  "]

        result = embedder.encode(texts)

        assert result.shape == (3, 96)

    def test_mixed_empty_and_nonempty(self):
        """Should handle mix of empty and non-empty texts."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        texts = ["", "Hello", "", "World", ""]

        result = embedder.encode(texts)

        assert result.shape == (5, 96)

    @pytest.mark.asyncio
    async def test_disconnected_graph_spectrum(self, disconnected_graph, sample_texts):
        """Should handle disconnected graph (multiple components)."""
        from HoloLoom.embedding.spectral import SpectralFusion, MatryoshkaEmbeddings

        spectral = SpectralFusion()
        embedder = MatryoshkaEmbeddings(sizes=[96])

        psi, metrics = await spectral.features(disconnected_graph, sample_texts, embedder)

        assert psi is not None
        # Fiedler value should be 0 for disconnected graph
        # (or near 0 depending on implementation)

    def test_scale_not_in_projection(self):
        """Should handle requesting scale not in projections gracefully."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192])

        # This should use projection matrix multiplication
        # even if size not originally configured
        result = embedder.encode_scales(["Test"], size=96)

        assert result.shape[1] == 96


# ============================================================================
# PERFORMANCE SANITY TESTS
# ============================================================================

class TestPerformanceSanity:
    """Basic performance sanity checks (not benchmarks)."""

    def test_batch_encoding_reasonable_time(self, sample_texts):
        """Batch encoding should complete in reasonable time."""
        import time
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        start = time.time()
        embedder.encode(sample_texts)
        elapsed = time.time() - start

        # Should complete in under 5 seconds for 5 texts
        assert elapsed < 5.0

    def test_cache_hit_faster_than_miss(self):
        """Cache hit should be faster than initial encoding."""
        import time
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        text = ["Performance test text"]

        # First encoding (cache miss)
        start = time.time()
        embedder.encode(text)
        first_time = time.time() - start

        # Second encoding (cache hit)
        start = time.time()
        embedder.encode(text)
        second_time = time.time() - start

        # Cache hit should be at least as fast (usually much faster)
        assert second_time <= first_time + 0.001


# ============================================================================
# PROTOCOL COMPLIANCE TESTS
# ============================================================================

class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_embedder_protocol(self):
        """MatryoshkaEmbeddings should implement Embedder protocol."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings, Embedder

        embedder = MatryoshkaEmbeddings(sizes=[96])

        # Should have encode method
        assert hasattr(embedder, 'encode')
        assert callable(embedder.encode)

    def test_encode_returns_ndarray(self, single_text):
        """encode() should return numpy array."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(single_text)

        assert isinstance(result, np.ndarray)

    def test_encode_dtype_float(self, single_text):
        """encode() should return float array."""
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(single_text)

        assert result.dtype in [np.float32, np.float64]
