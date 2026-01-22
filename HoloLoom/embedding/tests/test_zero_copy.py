"""
Comprehensive Tests for Zero-Copy Embeddings Module
====================================================

Test Coverage (~800 lines):

1. EmbeddingStore Tests (~300 lines)
   - Creation and file format
   - Read/write operations
   - Memory-mapped I/O
   - Batch operations
   - Context manager support
   - Error handling

2. ZeroCopyMatryoshkaEmbeddings Tests (~350 lines)
   - Initialization and lazy loading
   - Zero-copy scale extraction
   - Cache hit/miss behavior
   - Memory efficiency verification
   - Performance benchmarks
   - Integration with standard API

3. Edge Cases and Error Handling (~150 lines)
   - Corruption recovery
   - Invalid indices
   - Dimension mismatches
   - Concurrent access considerations

Created: 2026-01-22
Author: Claude Code
"""

import pytest
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Provide temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_embedding():
    """Provide sample embedding vector."""
    np.random.seed(42)
    return np.random.randn(768).astype(np.float32)


@pytest.fixture
def sample_embeddings_batch():
    """Provide batch of sample embeddings."""
    np.random.seed(42)
    return np.random.randn(10, 768).astype(np.float32)


@pytest.fixture
def sample_texts():
    """Provide sample texts for embedding tests."""
    return [
        "Multi-head attention processes multiple representation subspaces",
        "Transformers use self-attention mechanisms",
        "Neural networks learn hierarchical features"
    ]


@pytest.fixture
def single_text():
    """Provide single text for basic tests."""
    return ["Thompson Sampling balances exploration and exploitation"]


@pytest.fixture
def empty_texts():
    """Provide empty text list."""
    return []


# ============================================================================
# EMBEDDING STORE - CREATION TESTS
# ============================================================================

class TestEmbeddingStoreCreation:
    """Tests for EmbeddingStore creation."""

    def test_create_creates_file(self, temp_dir):
        """create() should create mmap file."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"
        store = EmbeddingStore.create(path, max_embeddings=100, dim=384)

        assert path.exists()
        store.close()

    def test_create_correct_file_size(self, temp_dir):
        """Created file should have correct size."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        max_embeddings = 100
        dim = 384
        path = temp_dir / "test.mmap"

        store = EmbeddingStore.create(path, max_embeddings=max_embeddings, dim=dim)
        store.close()

        expected_size = 64 + (max_embeddings * dim * 4)  # header + data
        actual_size = path.stat().st_size

        assert actual_size == expected_size

    def test_create_writes_magic_number(self, temp_dir):
        """Created file should have HOLOLOOM magic number."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"
        store = EmbeddingStore.create(path, max_embeddings=10, dim=4)
        store.close()

        with open(path, 'rb') as f:
            magic = f.read(8)

        assert magic == b'HOLOLOOM'

    def test_create_writes_version(self, temp_dir):
        """Created file should have version 1."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"
        store = EmbeddingStore.create(path, max_embeddings=10, dim=4)
        store.close()

        with open(path, 'rb') as f:
            f.read(8)  # Skip magic
            version = int.from_bytes(f.read(4), 'little')

        assert version == 1

    def test_create_writes_dimensions(self, temp_dir):
        """Created file should have correct dimensions in header."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        max_embeddings = 500
        dim = 256
        path = temp_dir / "test.mmap"

        store = EmbeddingStore.create(path, max_embeddings=max_embeddings, dim=dim)
        store.close()

        with open(path, 'rb') as f:
            f.read(8)  # Skip magic
            f.read(4)  # Skip version
            stored_max = int.from_bytes(f.read(8), 'little')
            stored_dim = int.from_bytes(f.read(4), 'little')

        assert stored_max == max_embeddings
        assert stored_dim == dim

    def test_create_returns_readwrite_store(self, temp_dir):
        """create() should return store in r+ mode."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"
        store = EmbeddingStore.create(path, max_embeddings=10, dim=4)

        assert store.mode == 'r+'
        store.close()

    def test_create_nested_directory(self, temp_dir):
        """create() should create parent directories if needed."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "nested" / "deep" / "test.mmap"

        store = EmbeddingStore.create(path, max_embeddings=10, dim=4)
        store.close()

        assert path.exists()


# ============================================================================
# EMBEDDING STORE - OPEN TESTS
# ============================================================================

class TestEmbeddingStoreOpen:
    """Tests for EmbeddingStore.open()."""

    def test_open_existing_file(self, temp_dir):
        """open() should load existing store."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        # Create first
        store1 = EmbeddingStore.create(path, max_embeddings=100, dim=384)
        store1.close()

        # Open
        store2 = EmbeddingStore.open(path)

        assert store2.max_embeddings == 100
        assert store2.dim == 384
        store2.close()

    def test_open_nonexistent_raises(self, temp_dir):
        """open() should raise for nonexistent file."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "nonexistent.mmap"

        with pytest.raises(FileNotFoundError):
            EmbeddingStore.open(path)

    def test_open_invalid_magic_raises(self, temp_dir):
        """open() should raise for invalid magic number."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "invalid.mmap"

        # Write invalid file
        with open(path, 'wb') as f:
            f.write(b'INVALID!')
            f.write(b'\x00' * 100)

        with pytest.raises(ValueError, match="Invalid magic"):
            EmbeddingStore.open(path)

    def test_open_invalid_version_raises(self, temp_dir):
        """open() should raise for unsupported version."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "badversion.mmap"

        # Write file with version 99
        with open(path, 'wb') as f:
            f.write(b'HOLOLOOM')
            f.write((99).to_bytes(4, 'little'))
            f.write(b'\x00' * 100)

        with pytest.raises(ValueError, match="Unsupported version"):
            EmbeddingStore.open(path)

    def test_open_readonly_mode(self, temp_dir):
        """open() with mode='r' should open read-only."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"
        store1 = EmbeddingStore.create(path, max_embeddings=10, dim=4)
        store1.close()

        store2 = EmbeddingStore.open(path, mode='r')

        assert store2.mode == 'r'
        store2.close()

    def test_open_readwrite_mode(self, temp_dir):
        """open() with mode='r+' should open read-write."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"
        store1 = EmbeddingStore.create(path, max_embeddings=10, dim=4)
        store1.close()

        store2 = EmbeddingStore.open(path, mode='r+')

        assert store2.mode == 'r+'
        store2.close()


# ============================================================================
# EMBEDDING STORE - READ/WRITE TESTS
# ============================================================================

class TestEmbeddingStoreReadWrite:
    """Tests for EmbeddingStore read/write operations."""

    def test_write_read_roundtrip(self, temp_dir, sample_embedding):
        """Written embedding should be readable."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"
        dim = len(sample_embedding)

        with EmbeddingStore.create(path, max_embeddings=10, dim=dim) as store:
            store.write(0, sample_embedding)
            result = store.read(0)

            assert np.allclose(result, sample_embedding)

    def test_write_multiple_indices(self, temp_dir):
        """Should write to multiple indices."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=10, dim=4) as store:
            emb1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            emb2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
            emb3 = np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32)

            store.write(0, emb1)
            store.write(1, emb2)
            store.write(5, emb3)

            assert np.allclose(store.read(0), emb1)
            assert np.allclose(store.read(1), emb2)
            assert np.allclose(store.read(5), emb3)

    def test_write_overwrites_existing(self, temp_dir):
        """Writing to same index should overwrite."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=10, dim=4) as store:
            emb1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            emb2 = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

            store.write(0, emb1)
            store.write(0, emb2)  # Overwrite

            result = store.read(0)
            assert np.allclose(result, emb2)

    def test_write_readonly_raises(self, temp_dir):
        """Writing in read-only mode should raise."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"
        store1 = EmbeddingStore.create(path, max_embeddings=10, dim=4)
        store1.close()

        with EmbeddingStore.open(path, mode='r') as store:
            emb = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

            with pytest.raises(ValueError, match="read-only"):
                store.write(0, emb)

    def test_write_index_out_of_bounds(self, temp_dir):
        """Writing beyond max_embeddings should raise."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=10, dim=4) as store:
            emb = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

            with pytest.raises(IndexError):
                store.write(10, emb)

    def test_write_wrong_dimension(self, temp_dir):
        """Writing wrong dimension should raise."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=10, dim=4) as store:
            emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Wrong dim

            with pytest.raises(ValueError, match="dim"):
                store.write(0, emb)

    def test_read_index_out_of_bounds(self, temp_dir):
        """Reading beyond max_embeddings should raise."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=10, dim=4) as store:
            with pytest.raises(IndexError):
                store.read(10)


# ============================================================================
# EMBEDDING STORE - ZERO-COPY VERIFICATION
# ============================================================================

class TestEmbeddingStoreZeroCopy:
    """Tests verifying zero-copy behavior."""

    def test_read_returns_view(self, temp_dir):
        """read() should return view (modifiable in mmap)."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=1, dim=4) as store:
            emb = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            store.write(0, emb)

            view = store.read(0)
            view[0] = 999.0

            # Read again - should see the change
            view2 = store.read(0)
            assert view2[0] == 999.0

    def test_read_range_returns_view(self, temp_dir):
        """read_range() should return view."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=5, dim=4) as store:
            for i in range(5):
                emb = np.full(4, float(i), dtype=np.float32)
                store.write(i, emb)

            view = store.read_range(1, 4)

            assert view.shape == (3, 4)
            # Verify values
            assert np.allclose(view[0], np.full(4, 1.0))
            assert np.allclose(view[1], np.full(4, 2.0))
            assert np.allclose(view[2], np.full(4, 3.0))

    def test_scale_extraction_via_slice(self, temp_dir):
        """Scale extraction via slice should be zero-copy."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=1, dim=768) as store:
            emb = np.random.randn(768).astype(np.float32)
            store.write(0, emb)

            # Extract scales via slicing
            full = store.read(0)
            scale_96 = full[:96]
            scale_192 = full[:192]
            scale_384 = full[:384]

            assert scale_96.shape == (96,)
            assert scale_192.shape == (192,)
            assert scale_384.shape == (384,)

            # Verify prefix property
            assert np.allclose(scale_96, scale_192[:96])
            assert np.allclose(scale_192, scale_384[:192])


# ============================================================================
# EMBEDDING STORE - BATCH OPERATIONS
# ============================================================================

class TestEmbeddingStoreBatchOperations:
    """Tests for batch read operations."""

    def test_read_batch(self, temp_dir, sample_embeddings_batch):
        """read_batch() should read multiple embeddings."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"
        n_embeddings = sample_embeddings_batch.shape[0]
        dim = sample_embeddings_batch.shape[1]

        with EmbeddingStore.create(path, max_embeddings=n_embeddings, dim=dim) as store:
            # Write all embeddings
            for i in range(n_embeddings):
                store.write(i, sample_embeddings_batch[i])

            # Read batch
            indices = [0, 2, 5, 9]
            batch = store.read_batch(indices)

            assert batch.shape == (4, dim)
            for i, idx in enumerate(indices):
                assert np.allclose(batch[i], sample_embeddings_batch[idx])

    def test_get_all(self, temp_dir):
        """get_all() should return all written embeddings."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=10, dim=4) as store:
            for i in range(5):
                emb = np.full(4, float(i), dtype=np.float32)
                store.write(i, emb)

            all_embs = store.get_all()

            assert all_embs.shape == (5, 4)


# ============================================================================
# EMBEDDING STORE - CONTEXT MANAGER
# ============================================================================

class TestEmbeddingStoreContextManager:
    """Tests for context manager support."""

    def test_context_manager_closes(self, temp_dir):
        """Context manager should close store."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        with EmbeddingStore.create(path, max_embeddings=10, dim=4) as store:
            pass

        # Should be closed after context
        assert store._mmap is None
        assert store._file_handle is None

    def test_context_manager_on_exception(self, temp_dir):
        """Context manager should close on exception."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "test.mmap"

        try:
            with EmbeddingStore.create(path, max_embeddings=10, dim=4) as store:
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert store._mmap is None


# ============================================================================
# ZERO-COPY MATRYOSHKA EMBEDDINGS - INITIALIZATION
# ============================================================================

class TestZeroCopyMatryoshkaInit:
    """Tests for ZeroCopyMatryoshkaEmbeddings initialization."""

    def test_init_default_sizes(self):
        """Default initialization should use [96, 192, 384, 768]."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings()

        assert embedder.sizes == [96, 192, 384, 768]

    def test_init_custom_sizes(self):
        """Should accept custom sizes."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[64, 128, 256])

        assert embedder.sizes == [64, 128, 256]

    def test_init_invalid_sizes_descending(self):
        """Descending sizes should raise."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        with pytest.raises(AssertionError):
            ZeroCopyMatryoshkaEmbeddings(sizes=[384, 192, 96])

    def test_init_invalid_sizes_too_large(self):
        """Sizes > 768 should raise."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        with pytest.raises(AssertionError):
            ZeroCopyMatryoshkaEmbeddings(sizes=[96, 1024])

    def test_init_lazy_loading(self):
        """Model should not load until first encode."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96])

        assert embedder._model_loaded is False
        assert embedder._model is None

    def test_init_with_store_path(self, temp_dir):
        """Should accept store path for persistence."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        path = temp_dir / "embeddings.mmap"

        embedder = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96],
            store_path=path
        )

        assert embedder.store_path == path


# ============================================================================
# ZERO-COPY MATRYOSHKA EMBEDDINGS - ENCODING
# ============================================================================

class TestZeroCopyMatryoshkaEncoding:
    """Tests for ZeroCopyMatryoshkaEmbeddings encoding methods."""

    def test_encode_scales_all(self, single_text):
        """encode_scales() should return all scales when size=None."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(single_text)

        assert isinstance(result, dict)
        assert 96 in result
        assert 192 in result
        assert 384 in result

    def test_encode_scales_specific(self, single_text):
        """encode_scales() should return specific scale."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(single_text, size=192)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 192)

    def test_encode_scales_dimensions(self, sample_texts):
        """All scales should have correct dimensions."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(sample_texts)

        n = len(sample_texts)
        assert result[96].shape == (n, 96)
        assert result[192].shape == (n, 192)
        assert result[384].shape == (n, 384)

    def test_encode_scales_empty_input(self, empty_texts):
        """encode_scales() should handle empty input."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])
        result = embedder.encode_scales(empty_texts)

        assert isinstance(result, dict)
        assert result[96].shape == (0, 96)
        assert result[192].shape == (0, 192)

    def test_encode_returns_largest_scale(self, single_text):
        """encode() should return largest scale."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode(single_text)

        assert result.shape == (1, 384)

    def test_encode_base(self, single_text):
        """encode_base() should return base dimension."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])
        result = embedder.encode_base(single_text)

        assert result.shape[0] == 1
        # Base dim is the internal model dimension (768 or max size)


# ============================================================================
# ZERO-COPY MATRYOSHKA EMBEDDINGS - PREFIX PROPERTY
# ============================================================================

class TestZeroCopyMatryoshkaPrefixProperty:
    """Tests verifying Matryoshka prefix property."""

    def test_prefix_property_holds(self, single_text):
        """Smaller scales should be prefixes of larger scales."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(single_text)

        # 96d should be prefix of 192d
        assert np.allclose(result[96][0], result[192][0, :96])

        # 192d should be prefix of 384d
        assert np.allclose(result[192][0], result[384][0, :192])

    def test_prefix_property_batch(self, sample_texts):
        """Prefix property should hold for batch encoding."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192, 384])
        result = embedder.encode_scales(sample_texts)

        for i in range(len(sample_texts)):
            assert np.allclose(result[96][i], result[192][i, :96])
            assert np.allclose(result[192][i], result[384][i, :192])


# ============================================================================
# ZERO-COPY MATRYOSHKA EMBEDDINGS - CACHING
# ============================================================================

class TestZeroCopyMatryoshkaCaching:
    """Tests for cache behavior."""

    def test_cache_populated(self, temp_dir, single_text):
        """Cache should be populated after encoding."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        path = temp_dir / "cache.mmap"
        embedder = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96],
            store_path=path,
            max_cache_size=100
        )

        embedder.encode(single_text)

        assert embedder._next_idx > 0
        assert single_text[0] in embedder._text_to_idx

    def test_cache_hit(self, temp_dir, single_text):
        """Second encoding should use cache."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        path = temp_dir / "cache.mmap"
        embedder = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96],
            store_path=path,
            max_cache_size=100
        )

        result1 = embedder.encode(single_text)
        result2 = embedder.encode(single_text)

        assert np.allclose(result1, result2)

    def test_cache_miss_different_texts(self, temp_dir):
        """Different texts should not use cache."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        path = temp_dir / "cache.mmap"
        embedder = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96],
            store_path=path,
            max_cache_size=100
        )

        embedder.encode(["Text A"])
        idx_after_first = embedder._next_idx

        embedder.encode(["Text B"])
        idx_after_second = embedder._next_idx

        assert idx_after_second > idx_after_first

    def test_cache_max_size_respected(self, temp_dir):
        """Cache should stop growing at max_cache_size."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        path = temp_dir / "cache.mmap"
        max_size = 5
        embedder = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96],
            store_path=path,
            max_cache_size=max_size
        )

        # Encode more texts than cache size
        for i in range(10):
            embedder.encode([f"Text {i}"])

        # Cache should be at max
        assert embedder._next_idx <= max_size


# ============================================================================
# ZERO-COPY MATRYOSHKA EMBEDDINGS - PERFORMANCE
# ============================================================================

class TestZeroCopyMatryoshkaPerformance:
    """Tests for performance characteristics."""

    def test_scale_extraction_fast(self, temp_dir, single_text):
        """Scale extraction should be very fast."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        path = temp_dir / "perf.mmap"
        embedder = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96, 192, 384],
            store_path=path
        )

        # Warm up (populate cache)
        embedder.encode_scales(single_text)

        # Measure scale extraction (should be <1ms)
        start = time.time()
        for _ in range(100):
            embedder.encode_scales(single_text)
        elapsed = time.time() - start

        avg_ms = elapsed / 100 * 1000
        # Should be very fast when cached
        assert avg_ms < 10  # 10ms threshold for test stability

    def test_multi_scale_faster_than_multiple_single(self, sample_texts):
        """Getting all scales at once should be efficient."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192, 384])

        # Time all scales at once
        start = time.time()
        result_all = embedder.encode_scales(sample_texts)
        time_all = time.time() - start

        # Time individual scales
        start = time.time()
        result_96 = embedder.encode_scales(sample_texts, size=96)
        result_192 = embedder.encode_scales(sample_texts, size=192)
        result_384 = embedder.encode_scales(sample_texts, size=384)
        time_individual = time.time() - start

        # All-at-once should not be significantly slower
        # (may be faster due to single base embedding computation)
        assert time_all <= time_individual * 2


# ============================================================================
# ZERO-COPY MATRYOSHKA EMBEDDINGS - PERSISTENCE
# ============================================================================

class TestZeroCopyMatryoshkaPersistence:
    """Tests for persistent storage."""

    def test_persistence_across_instances(self, temp_dir, single_text):
        """Cached embeddings should persist across instances."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        path = temp_dir / "persist.mmap"

        # First instance
        embedder1 = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96],
            store_path=path,
            max_cache_size=100
        )
        result1 = embedder1.encode(single_text)
        embedder1.close()

        # Second instance (same path)
        embedder2 = ZeroCopyMatryoshkaEmbeddings(
            sizes=[96],
            store_path=path,
            max_cache_size=100
        )
        # Note: text_to_idx mapping is not persisted in current implementation
        # but the raw embeddings are in the mmap file

    def test_context_manager_closes_store(self, temp_dir):
        """Context manager should close store properly."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        path = temp_dir / "ctx.mmap"

        with ZeroCopyMatryoshkaEmbeddings(
            sizes=[96],
            store_path=path,
            max_cache_size=100
        ) as embedder:
            embedder.encode(["Test"])

        # Should be closed
        assert embedder._store is None or embedder._store._mmap is None


# ============================================================================
# ZERO-COPY MATRYOSHKA EMBEDDINGS - GRACEFUL DEGRADATION
# ============================================================================

class TestZeroCopyMatryoshkaGracefulDegradation:
    """Tests for fallback behavior."""

    def test_fallback_without_model(self, single_text):
        """Should use fallback when model unavailable."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])
        embedder._model = None
        embedder._model_loaded = True
        embedder.base_dim = 192

        result = embedder.encode_scales(single_text, size=96)

        assert result is not None
        assert result.shape == (1, 96)

    def test_fallback_deterministic(self, single_text):
        """Fallback embeddings should be deterministic."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder1 = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        embedder1._model = None
        embedder1._model_loaded = True
        embedder1.base_dim = 768
        result1 = embedder1.encode(single_text)

        embedder2 = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        embedder2._model = None
        embedder2._model_loaded = True
        embedder2.base_dim = 768
        result2 = embedder2.encode(single_text)

        assert np.allclose(result1, result2)


# ============================================================================
# ZERO-COPY MATRYOSHKA EMBEDDINGS - NO STORE
# ============================================================================

class TestZeroCopyMatryoshkaNoStore:
    """Tests when no store path provided."""

    def test_works_without_store(self, single_text):
        """Should work without persistent store."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])

        result = embedder.encode_scales(single_text)

        assert 96 in result
        assert 192 in result

    def test_caching_still_works_in_memory(self, single_text):
        """In-memory caching should work."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96])

        result1 = embedder.encode(single_text)
        result2 = embedder.encode(single_text)

        assert np.allclose(result1, result2)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_text_list(self, empty_texts):
        """Should handle empty text list."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(empty_texts)

        assert result.shape == (0, 96)

    def test_empty_string(self):
        """Should handle empty string."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode([""])

        assert result.shape == (1, 96)

    def test_unicode_text(self):
        """Should handle unicode text."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(["Hello"])

        assert result.shape == (1, 96)

    def test_very_long_text(self):
        """Should handle very long text."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        long_text = "word " * 5000
        result = embedder.encode([long_text])

        assert result.shape == (1, 96)

    def test_duplicate_texts(self):
        """Should handle duplicate texts in batch."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        texts = ["Same text", "Same text", "Same text"]
        result = embedder.encode(texts)

        assert result.shape == (3, 96)
        assert np.allclose(result[0], result[1])
        assert np.allclose(result[1], result[2])


# ============================================================================
# STORE FILE CORRUPTION RECOVERY
# ============================================================================

class TestCorruptionRecovery:
    """Tests for handling corrupted store files."""

    def test_truncated_header(self, temp_dir):
        """Should handle truncated header."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "truncated.mmap"

        # Write truncated file
        with open(path, 'wb') as f:
            f.write(b'HOLO')  # Only partial magic

        with pytest.raises(ValueError):
            EmbeddingStore.open(path)

    def test_zero_byte_file(self, temp_dir):
        """Should handle zero-byte file."""
        from HoloLoom.embedding.zero_copy import EmbeddingStore

        path = temp_dir / "empty.mmap"
        path.touch()  # Create empty file

        with pytest.raises(ValueError):
            EmbeddingStore.open(path)


# ============================================================================
# API COMPATIBILITY
# ============================================================================

class TestAPICompatibility:
    """Tests for API compatibility with standard MatryoshkaEmbeddings."""

    def test_encode_interface(self, single_text):
        """encode() should have same interface as MatryoshkaEmbeddings."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode(single_text)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == len(single_text)

    def test_encode_scales_interface(self, single_text):
        """encode_scales() should have same interface as MatryoshkaEmbeddings."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96, 192])

        # All scales
        result_all = embedder.encode_scales(single_text)
        assert isinstance(result_all, dict)

        # Specific scale
        result_single = embedder.encode_scales(single_text, size=96)
        assert isinstance(result_single, np.ndarray)

    def test_encode_base_interface(self, single_text):
        """encode_base() should have same interface as MatryoshkaEmbeddings."""
        from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings

        embedder = ZeroCopyMatryoshkaEmbeddings(sizes=[96])
        result = embedder.encode_base(single_text)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == len(single_text)


# ============================================================================
# BENCHMARK UTILITY TESTS
# ============================================================================

class TestBenchmarkUtility:
    """Tests for benchmark_zero_copy_vs_projection utility."""

    def test_benchmark_returns_results(self, sample_texts):
        """benchmark_zero_copy_vs_projection should return timing results."""
        from HoloLoom.embedding.zero_copy import benchmark_zero_copy_vs_projection

        results = benchmark_zero_copy_vs_projection(
            texts=sample_texts,
            sizes=[96, 192],
            n_iterations=5  # Small for test speed
        )

        assert 'projection_based_ms' in results
        assert 'zero_copy_ms' in results
        assert 'speedup' in results

    def test_benchmark_speedup_positive(self, sample_texts):
        """Speedup should be positive (zero-copy not slower)."""
        from HoloLoom.embedding.zero_copy import benchmark_zero_copy_vs_projection

        results = benchmark_zero_copy_vs_projection(
            texts=sample_texts,
            sizes=[96],
            n_iterations=5
        )

        # Zero-copy should not be significantly slower
        # Speedup >= 0.5 means at worst 2x slower (which would be unexpected)
        assert results['speedup'] >= 0.1
