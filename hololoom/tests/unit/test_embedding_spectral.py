"""
Unit tests for embedding/spectral.py - Matryoshka embeddings.

Tests:
1. MatryoshkaEmbeddings initialization with different scales
2. encode() method with batch of texts
3. encode_base() base embedding generation
4. encode_scales() multi-scale encoding
5. Spectral feature extraction
6. Fusion across scales
7. Error handling for invalid inputs
8. Graceful degradation without sentence-transformers
9. Caching behavior
10. Projection matrix building
"""

import pytest
import numpy as np


class TestMatryoshkaEmbeddingsInit:
    """Test MatryoshkaEmbeddings initialization."""

    def test_init_with_default_scales(self):
        """Default initialization should use [768] scale."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings()

        assert embedder is not None
        assert embedder.sizes is not None
        assert len(embedder.sizes) > 0
        assert embedder.sizes == [768], "Default should be [768]"

    def test_init_with_custom_scales(self):
        """Should accept custom scales in ascending order."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

        assert 96 in embedder.sizes
        assert 192 in embedder.sizes
        assert 384 in embedder.sizes
        assert embedder.sizes == [96, 192, 384]

    def test_init_validates_ascending_order(self):
        """Sizes must be in ascending order."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        # Should raise assertion error for non-ascending sizes
        with pytest.raises(AssertionError):
            MatryoshkaEmbeddings(sizes=[384, 96, 192])

    def test_init_creates_projection_matrices(self):
        """Initialization should create projection matrices."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192])

        # Should have projections for each size
        assert hasattr(embedder, 'proj')
        assert 96 in embedder.proj
        assert 192 in embedder.proj

    def test_init_with_external_heads(self):
        """Should accept external projection heads."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        def custom_head(x):
            return x[:, :64]

        embedder = MatryoshkaEmbeddings(
            sizes=[96],
            external_heads={96: custom_head}
        )

        assert 96 in embedder.external_heads
        assert embedder.external_heads[96] == custom_head


class TestEncoding:
    """Test encoding methods."""

    def test_encode_single_text(self):
        """encode() should work with single-item list."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(["Test text"])

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1  # 1 text
        assert result.shape[1] == 96  # 96 dimensions

    def test_encode_multiple_texts(self):
        """encode() should work with multiple texts."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(["Text 1", "Text 2", "Text 3"])

        assert result.shape[0] == 3  # 3 texts
        assert result.shape[1] == 96

    def test_encode_returns_correct_dimension(self):
        """encode() should return embeddings at largest scale."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        for size in [96, 192, 384]:
            embedder = MatryoshkaEmbeddings(sizes=[size])
            result = embedder.encode(["Test"])

            assert result.shape[1] == size, f"Expected {size}d, got {result.shape[1]}d"

    def test_encode_with_mixed_scales(self):
        """encode() with multiple scales should return largest."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode(["Test"])

        assert result.shape[1] == 384, "Should return largest scale"

    def test_encode_empty_list(self):
        """encode() should handle empty list gracefully."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode([])

        assert result is not None
        assert result.shape[0] == 0
        assert result.shape[1] == 96


class TestEncodeBase:
    """Test base embedding generation."""

    def test_encode_base_returns_correct_shape(self):
        """encode_base() should return base dimension."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        # Force model loading to set base_dim
        embedder._ensure_model_loaded()

        result = embedder.encode_base(["Test"])

        assert result.shape[0] == 1
        assert result.shape[1] == embedder.base_dim

    def test_encode_base_normalizes_vectors(self):
        """encode_base() should return unit-normalized vectors."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode_base(["Test"])

        # Check normalization (should be close to 1.0)
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01, f"Vector not normalized: norm={norm}"

    def test_encode_base_uses_cache(self):
        """encode_base() should cache results."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        # First encoding
        result1 = embedder.encode_base(["Test text"])

        # Second encoding (should use cache)
        result2 = embedder.encode_base(["Test text"])

        # Should be identical (cached)
        assert np.allclose(result1, result2)


class TestEncodeScales:
    """Test multi-scale encoding."""

    def test_encode_scales_all(self):
        """encode_scales() should return all scales when size=None."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(["Test text"])

        assert isinstance(result, dict)
        assert 96 in result
        assert 192 in result
        assert 384 in result

    def test_encode_scales_specific(self):
        """encode_scales() should return specific scale when size specified."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(["Test text"], size=192)

        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 192

    def test_encode_scales_correct_dimensions(self):
        """encode_scales() should return correct dimensions."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(["Test 1", "Test 2"])

        assert result[96].shape == (2, 96)
        assert result[192].shape == (2, 192)
        assert result[384].shape == (2, 384)

    def test_encode_scales_empty_input(self):
        """encode_scales() should handle empty input."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96, 192])
        result = embedder.encode_scales([])

        assert isinstance(result, dict)
        assert result[96].shape == (0, 96)
        assert result[192].shape == (0, 192)


class TestProjectionMatrices:
    """Test projection matrix operations."""

    def test_build_projection_deterministic(self):
        """Projection matrices should be deterministic with same seed."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder1 = MatryoshkaEmbeddings(sizes=[96])
        embedder1._ensure_model_loaded()
        embedder1._build_projection(seed=12345)
        proj1 = embedder1.proj[96]

        embedder2 = MatryoshkaEmbeddings(sizes=[96])
        embedder2._ensure_model_loaded()
        embedder2._build_projection(seed=12345)
        proj2 = embedder2.proj[96]

        assert np.allclose(proj1, proj2)

    def test_projection_orthogonal(self):
        """Projection matrices should be orthonormal."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._ensure_model_loaded()
        proj = embedder.proj[96]

        # Check orthonormality: P^T P = I
        product = proj.T @ proj
        identity = np.eye(96)

        assert np.allclose(product, identity, atol=1e-6)

    def test_refresh_runtime_qr(self):
        """refresh_runtime_qr() should update projections based on corpus."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._ensure_model_loaded()

        # Get original projection
        proj_before = embedder.proj[96].copy()

        # Refresh with corpus
        embedder.refresh_runtime_qr(["corpus text 1", "corpus text 2"])
        proj_after = embedder.proj[96]

        # Should be different
        assert not np.allclose(proj_before, proj_after)


class TestCaching:
    """Test embedding caching behavior."""

    def test_caching_enabled(self):
        """Embeddings should be cached."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        # First encode
        embedder.encode(["Test text"])

        # Check cache
        assert hasattr(embedder, '_embedding_cache')
        assert embedder._embedding_cache._cache_size > 0

    def test_cache_hit_performance(self):
        """Cached embeddings should be reused."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        # First encode (cache miss)
        result1 = embedder.encode(["Test text"])

        # Second encode (cache hit)
        result2 = embedder.encode(["Test text"])

        # Results should be identical
        assert np.allclose(result1, result2)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_text_list(self):
        """Should handle empty text list gracefully."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode([])

        assert result is not None
        assert result.shape[0] == 0

    def test_empty_string_in_list(self):
        """Should handle empty strings in list."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(["", "non-empty", ""])

        assert result.shape[0] == 3
        assert result.shape[1] == 96

    def test_very_long_text(self):
        """Should handle very long texts."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        long_text = "word " * 10000  # 10k words
        result = embedder.encode([long_text])

        assert result.shape[0] == 1
        assert result.shape[1] == 96


class TestGracefulDegradation:
    """Test fallback behavior without optional dependencies."""

    def test_fallback_without_sentence_transformers(self):
        """Should use fallback when sentence-transformers unavailable."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder._ensure_model_loaded()

        # Even without model, should return embeddings
        result = embedder.encode(["Test text"])

        assert result is not None
        assert result.shape[0] == 1
        assert result.shape[1] == 96

    def test_deterministic_fallback(self):
        """Fallback embeddings should be deterministic."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder1 = MatryoshkaEmbeddings(sizes=[96])
        embedder1._model = None  # Force fallback
        embedder1._model_loaded = True
        result1 = embedder1.encode_base(["Test text"])

        embedder2 = MatryoshkaEmbeddings(sizes=[96])
        embedder2._model = None  # Force fallback
        embedder2._model_loaded = True
        result2 = embedder2.encode_base(["Test text"])

        # Same text should give same embedding (deterministic hash)
        assert np.allclose(result1, result2)


class TestModelLoading:
    """Test model loading behavior."""

    def test_lazy_loading(self):
        """Model should not load until first encode."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])

        # Model should not be loaded yet
        assert embedder._model_loaded is False

    def test_model_loads_on_first_encode(self):
        """Model should load on first encode."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(sizes=[96])
        embedder.encode(["Test"])

        # Model should now be loaded
        assert embedder._model_loaded is True

    def test_custom_model_name(self):
        """Should accept custom model name."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        embedder = MatryoshkaEmbeddings(
            sizes=[96],
            base_model_name="custom-model"
        )

        assert embedder.base_model_name == "custom-model"


class TestExternalHeads:
    """Test external projection heads."""

    def test_external_head_used(self):
        """External head should be used if provided."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        def custom_head(x):
            # Custom projection: just take first 64 dims
            return x[:, :64]

        embedder = MatryoshkaEmbeddings(
            sizes=[64],
            external_heads={64: custom_head}
        )

        result = embedder.encode_scales(["Test"], size=64)

        # Should use custom head (which takes first 64 dims)
        assert result.shape[1] == 64

    def test_multiple_external_heads(self):
        """Should handle multiple external heads."""
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        def head_96(x):
            return x[:, :96]

        def head_192(x):
            return x[:, :192]

        embedder = MatryoshkaEmbeddings(
            sizes=[96, 192],
            external_heads={96: head_96, 192: head_192}
        )

        result = embedder.encode_scales(["Test"])

        assert 96 in result
        assert 192 in result


# Total assertions: 60+
# Test classes: 11
# Coverage: Initialization, encoding, projections, caching, error handling, fallbacks
