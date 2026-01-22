"""
Tests for Semantic Calculus Performance Utilities
=================================================

Comprehensive test suite for caching and optimization utilities:
- EmbeddingCache: LRU cache for word embeddings
- ProjectionCache: Cache for matrix-vector projections
- SparseSemanticVector: Sparse representation
- LazyArray: Lazy evaluation wrapper
- Vectorized computation functions
- JIT-compiled functions (with fallback)

Run with:
    pytest HoloLoom/semantic_calculus/tests/test_performance.py -v

Author: HoloLoom Team
Date: 2025-01-22
"""

import pytest
import numpy as np
from typing import List
import time
import io
import sys


# ============================================================================
# Test Imports
# ============================================================================

class TestPerformanceImports:
    """Test that all performance utilities can be imported."""

    def test_import_embedding_cache(self):
        """Test EmbeddingCache import."""
        from HoloLoom.semantic_calculus.performance import EmbeddingCache
        assert EmbeddingCache is not None

    def test_import_projection_cache(self):
        """Test ProjectionCache import."""
        from HoloLoom.semantic_calculus.performance import ProjectionCache
        assert ProjectionCache is not None

    def test_import_timer(self):
        """Test timer decorator import."""
        from HoloLoom.semantic_calculus.performance import timer
        assert timer is not None

    def test_import_batch_iterator(self):
        """Test batch_iterator import."""
        from HoloLoom.semantic_calculus.performance import batch_iterator
        assert batch_iterator is not None

    def test_import_sparse_vector(self):
        """Test SparseSemanticVector import."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector
        assert SparseSemanticVector is not None

    def test_import_lazy_array(self):
        """Test LazyArray import."""
        from HoloLoom.semantic_calculus.performance import LazyArray
        assert LazyArray is not None

    def test_import_vectorized_functions(self):
        """Test vectorized function imports."""
        from HoloLoom.semantic_calculus.performance import (
            compute_finite_difference_vectorized,
            compute_curvature_vectorized,
        )
        assert compute_finite_difference_vectorized is not None
        assert compute_curvature_vectorized is not None

    def test_import_jit_functions(self):
        """Test JIT function imports (may be fallback)."""
        from HoloLoom.semantic_calculus.performance import (
            stormer_verlet_step_jit,
            compute_gradient_batch_jit,
            HAS_NUMBA,
        )
        assert stormer_verlet_step_jit is not None
        assert compute_gradient_batch_jit is not None
        assert isinstance(HAS_NUMBA, bool)

    def test_import_all_exports(self):
        """Test all exports are available."""
        from HoloLoom.semantic_calculus.performance import (
            EmbeddingCache,
            ProjectionCache,
            timer,
            batch_iterator,
            SparseSemanticVector,
            LazyArray,
            compute_finite_difference_vectorized,
            compute_curvature_vectorized,
            stormer_verlet_step_jit,
            compute_gradient_batch_jit,
            HAS_NUMBA,
        )
        # All should be importable
        assert True


# ============================================================================
# EmbeddingCache Tests
# ============================================================================

class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    @pytest.fixture
    def mock_embed_fn(self):
        """Create a simple mock embedding function."""
        def embed(words: List[str]) -> np.ndarray:
            # Generate deterministic embeddings based on word hash
            embeddings = []
            for word in words:
                np.random.seed(hash(word) % 2**31)
                emb = np.random.randn(384)
                emb = emb / np.linalg.norm(emb)
                embeddings.append(emb)
            return np.array(embeddings)
        return embed

    @pytest.fixture
    def cache(self, mock_embed_fn):
        """Create EmbeddingCache instance."""
        from HoloLoom.semantic_calculus.performance import EmbeddingCache
        return EmbeddingCache(mock_embed_fn, max_size=100)

    def test_initialization(self, mock_embed_fn):
        """Test EmbeddingCache initialization."""
        from HoloLoom.semantic_calculus.performance import EmbeddingCache

        cache = EmbeddingCache(mock_embed_fn, max_size=500)

        assert cache.max_size == 500
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_get_single_word_miss(self, cache):
        """Test getting a single word (cache miss)."""
        embedding = cache.get("hello")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert cache.misses == 1
        assert cache.hits == 0
        assert len(cache.cache) == 1

    def test_get_single_word_hit(self, cache):
        """Test getting a cached word (cache hit)."""
        # First call - cache miss
        embedding1 = cache.get("hello")

        # Second call - cache hit
        embedding2 = cache.get("hello")

        assert np.array_equal(embedding1, embedding2)
        assert cache.hits == 1
        assert cache.misses == 1

    def test_get_batch_all_miss(self, cache):
        """Test batch get with all cache misses."""
        words = ["apple", "banana", "cherry"]
        embeddings = cache.get_batch(words)

        assert embeddings.shape == (3, 384)
        assert cache.misses == 3
        assert cache.hits == 0
        assert len(cache.cache) == 3

    def test_get_batch_all_hit(self, cache):
        """Test batch get with all cache hits."""
        words = ["apple", "banana", "cherry"]

        # First call - all misses
        cache.get_batch(words)

        # Second call - all hits
        embeddings = cache.get_batch(words)

        assert embeddings.shape == (3, 384)
        assert cache.misses == 3
        assert cache.hits == 3

    def test_get_batch_mixed(self, cache):
        """Test batch get with mixed hits/misses."""
        # Pre-cache some words
        cache.get("apple")
        cache.get("banana")

        # Request mix of cached and new words
        words = ["apple", "cherry", "banana", "date"]
        embeddings = cache.get_batch(words)

        assert embeddings.shape == (4, 384)
        assert cache.hits == 2  # apple, banana
        assert cache.misses == 4  # apple, banana (pre-cache), cherry, date

    def test_lru_eviction(self, mock_embed_fn):
        """Test LRU eviction when cache is full."""
        from HoloLoom.semantic_calculus.performance import EmbeddingCache

        cache = EmbeddingCache(mock_embed_fn, max_size=3)

        # Fill cache
        cache.get("word1")
        cache.get("word2")
        cache.get("word3")

        assert len(cache.cache) == 3
        assert "word1" in cache.cache

        # Add one more - should evict word1 (least recently used)
        cache.get("word4")

        assert len(cache.cache) == 3
        assert "word1" not in cache.cache
        assert "word4" in cache.cache

    def test_lru_access_updates_order(self, mock_embed_fn):
        """Test that accessing a cached item updates its recency."""
        from HoloLoom.semantic_calculus.performance import EmbeddingCache

        cache = EmbeddingCache(mock_embed_fn, max_size=3)

        # Fill cache
        cache.get("word1")
        cache.get("word2")
        cache.get("word3")

        # Access word1 to make it most recently used
        cache.get("word1")

        # Add word4 - should evict word2 (now least recently used)
        cache.get("word4")

        assert "word1" in cache.cache
        assert "word2" not in cache.cache
        assert "word3" in cache.cache
        assert "word4" in cache.cache

    def test_get_stats(self, cache):
        """Test cache statistics."""
        cache.get("word1")
        cache.get("word2")
        cache.get("word1")  # hit

        stats = cache.get_stats()

        assert stats['size'] == 2
        assert stats['max_size'] == 100
        assert stats['hits'] == 1
        assert stats['misses'] == 2
        assert abs(stats['hit_rate'] - 1/3) < 0.01

    def test_get_stats_empty_cache(self, cache):
        """Test statistics on empty cache."""
        stats = cache.get_stats()

        assert stats['size'] == 0
        assert stats['hit_rate'] == 0.0

    def test_clear(self, cache):
        """Test cache clearing."""
        cache.get("word1")
        cache.get("word2")

        cache.clear()

        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_embedding_consistency(self, cache):
        """Test that same word always returns same embedding."""
        emb1 = cache.get("consistent")
        cache.clear()
        emb2 = cache.get("consistent")

        # Same word should produce same embedding (deterministic mock)
        assert np.allclose(emb1, emb2)


# ============================================================================
# ProjectionCache Tests
# ============================================================================

class TestProjectionCache:
    """Tests for ProjectionCache class."""

    @pytest.fixture
    def projection_matrix(self):
        """Create a projection matrix."""
        np.random.seed(42)
        P = np.random.randn(16, 384)
        # Orthonormalize rows
        P, _ = np.linalg.qr(P.T)
        return P.T[:16, :]

    @pytest.fixture
    def cache(self):
        """Create ProjectionCache instance."""
        from HoloLoom.semantic_calculus.performance import ProjectionCache
        return ProjectionCache(max_size=100)

    def test_initialization(self):
        """Test ProjectionCache initialization."""
        from HoloLoom.semantic_calculus.performance import ProjectionCache

        cache = ProjectionCache(max_size=500)

        assert cache.max_size == 500
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_project_miss(self, cache, projection_matrix):
        """Test projection with cache miss."""
        q = np.random.randn(384)

        result = cache.project(projection_matrix, q)

        assert result.shape == (16,)
        assert cache.misses == 1
        assert cache.hits == 0

    def test_project_hit(self, cache, projection_matrix):
        """Test projection with cache hit."""
        q = np.random.randn(384)

        # First call - miss
        result1 = cache.project(projection_matrix, q)

        # Second call - hit
        result2 = cache.project(projection_matrix, q)

        assert np.array_equal(result1, result2)
        assert cache.hits == 1
        assert cache.misses == 1

    def test_project_different_vectors(self, cache, projection_matrix):
        """Test projection with different vectors."""
        q1 = np.random.randn(384)
        q2 = np.random.randn(384)

        result1 = cache.project(projection_matrix, q1)
        result2 = cache.project(projection_matrix, q2)

        assert result1.shape == result2.shape
        assert not np.array_equal(result1, result2)
        assert cache.misses == 2

    def test_project_batch(self, cache, projection_matrix):
        """Test batch projection."""
        Q = np.random.randn(10, 384)

        result = cache.project_batch(projection_matrix, Q)

        assert result.shape == (10, 16)

    def test_project_batch_correctness(self, cache, projection_matrix):
        """Test batch projection produces correct results."""
        Q = np.random.randn(5, 384)

        batch_result = cache.project_batch(projection_matrix, Q)

        # Compare with individual projections
        for i in range(5):
            individual = projection_matrix @ Q[i]
            assert np.allclose(batch_result[i], individual)

    def test_lru_eviction_projection(self):
        """Test LRU eviction in projection cache."""
        from HoloLoom.semantic_calculus.performance import ProjectionCache

        cache = ProjectionCache(max_size=2)
        np.random.seed(42)
        P = np.random.randn(16, 384)

        q1 = np.array([1.0] * 384)
        q2 = np.array([2.0] * 384)
        q3 = np.array([3.0] * 384)

        cache.project(P, q1)
        cache.project(P, q2)

        assert len(cache.cache) == 2

        # Add third - should evict first
        cache.project(P, q3)

        assert len(cache.cache) == 2


# ============================================================================
# Timer Decorator Tests
# ============================================================================

class TestTimerDecorator:
    """Tests for timer decorator."""

    def test_timer_executes_function(self):
        """Test that timer executes the wrapped function."""
        from HoloLoom.semantic_calculus.performance import timer

        @timer
        def add(a, b):
            return a + b

        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured

        result = add(2, 3)

        sys.stdout = sys.__stdout__

        assert result == 5

    def test_timer_prints_timing(self):
        """Test that timer prints timing information."""
        from HoloLoom.semantic_calculus.performance import timer

        @timer
        def slow_function():
            time.sleep(0.01)
            return "done"

        # Capture stdout
        captured = io.StringIO()
        sys.stdout = captured

        slow_function()

        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "[TIMER]" in output
        assert "slow_function" in output
        assert "ms" in output

    def test_timer_preserves_function_name(self):
        """Test that timer preserves function metadata."""
        from HoloLoom.semantic_calculus.performance import timer

        @timer
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_timer_with_args_and_kwargs(self):
        """Test timer with various argument types."""
        from HoloLoom.semantic_calculus.performance import timer

        @timer
        def complex_function(a, b, c=10, d=20):
            return a + b + c + d

        # Capture stdout to suppress output
        captured = io.StringIO()
        sys.stdout = captured

        result = complex_function(1, 2, c=3, d=4)

        sys.stdout = sys.__stdout__

        assert result == 10


# ============================================================================
# Batch Iterator Tests
# ============================================================================

class TestBatchIterator:
    """Tests for batch_iterator function."""

    def test_batch_iterator_basic(self):
        """Test basic batch iteration."""
        from HoloLoom.semantic_calculus.performance import batch_iterator

        items = list(range(10))
        batches = list(batch_iterator(items, batch_size=3))

        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]

    def test_batch_iterator_exact_division(self):
        """Test when items divide evenly into batches."""
        from HoloLoom.semantic_calculus.performance import batch_iterator

        items = list(range(9))
        batches = list(batch_iterator(items, batch_size=3))

        assert len(batches) == 3
        assert all(len(b) == 3 for b in batches)

    def test_batch_iterator_empty_list(self):
        """Test batch iteration on empty list."""
        from HoloLoom.semantic_calculus.performance import batch_iterator

        batches = list(batch_iterator([], batch_size=3))

        assert len(batches) == 0

    def test_batch_iterator_single_item(self):
        """Test batch iteration with single item."""
        from HoloLoom.semantic_calculus.performance import batch_iterator

        batches = list(batch_iterator([42], batch_size=3))

        assert len(batches) == 1
        assert batches[0] == [42]

    def test_batch_iterator_batch_larger_than_list(self):
        """Test when batch size is larger than list."""
        from HoloLoom.semantic_calculus.performance import batch_iterator

        items = [1, 2, 3]
        batches = list(batch_iterator(items, batch_size=10))

        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]

    def test_batch_iterator_generator_behavior(self):
        """Test that batch_iterator is a generator."""
        from HoloLoom.semantic_calculus.performance import batch_iterator

        items = list(range(100))
        gen = batch_iterator(items, batch_size=10)

        # Should be a generator, not a list
        assert hasattr(gen, '__next__')

        # Can iterate
        first_batch = next(gen)
        assert first_batch == list(range(10))


# ============================================================================
# SparseSemanticVector Tests
# ============================================================================

class TestSparseSemanticVector:
    """Tests for SparseSemanticVector class."""

    def test_initialization(self):
        """Test SparseSemanticVector initialization."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        values = np.array([0.5, 0.001, -0.3, 0.0, 0.2])
        sparse = SparseSemanticVector(values, threshold=0.01)

        assert sparse.n_dims == 5

    def test_sparsification(self):
        """Test that small values are treated as zero."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        values = np.array([0.5, 0.005, -0.3, 0.0, 0.2])
        sparse = SparseSemanticVector(values, threshold=0.01)

        # Only values > 0.01 should be stored
        assert len(sparse.active_indices) == 3
        assert 0 in sparse.active_indices  # 0.5
        assert 2 in sparse.active_indices  # -0.3
        assert 4 in sparse.active_indices  # 0.2

    def test_to_dense(self):
        """Test conversion back to dense vector."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        values = np.array([0.5, 0.001, -0.3, 0.0, 0.2])
        sparse = SparseSemanticVector(values, threshold=0.01)

        dense = sparse.to_dense()

        assert dense.shape == (5,)
        assert dense[0] == 0.5
        assert dense[1] == 0.0  # Below threshold
        assert dense[2] == -0.3
        assert dense[3] == 0.0
        assert dense[4] == 0.2

    def test_sparsity_property(self):
        """Test sparsity calculation."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        # 3 non-zero out of 10
        values = np.zeros(10)
        values[0] = 0.5
        values[3] = -0.2
        values[7] = 0.8

        sparse = SparseSemanticVector(values, threshold=0.01)

        assert abs(sparse.sparsity - 0.7) < 0.01  # 70% zero

    def test_all_zero_vector(self):
        """Test vector with all zeros."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        values = np.zeros(100)
        sparse = SparseSemanticVector(values, threshold=0.01)

        assert len(sparse.active_indices) == 0
        assert sparse.sparsity == 1.0

    def test_all_nonzero_vector(self):
        """Test vector with all non-zero values."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        values = np.ones(100) * 0.5
        sparse = SparseSemanticVector(values, threshold=0.01)

        assert len(sparse.active_indices) == 100
        assert sparse.sparsity == 0.0

    def test_repr(self):
        """Test string representation."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        values = np.array([0.5, 0.001, -0.3])
        sparse = SparseSemanticVector(values, threshold=0.01)

        repr_str = repr(sparse)

        assert "SparseSemanticVector" in repr_str
        assert "active=" in repr_str
        assert "sparsity=" in repr_str

    def test_threshold_edge_case(self):
        """Test threshold boundary case."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        values = np.array([0.01, 0.011, 0.009])
        sparse = SparseSemanticVector(values, threshold=0.01)

        # Exactly at threshold should not be included (> not >=)
        assert len(sparse.active_indices) == 1  # Only 0.011

    def test_negative_values(self):
        """Test handling of negative values."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        values = np.array([0.5, -0.5, 0.001, -0.001])
        sparse = SparseSemanticVector(values, threshold=0.01)

        # Both positive and negative large values should be kept
        assert len(sparse.active_indices) == 2

        dense = sparse.to_dense()
        assert dense[0] == 0.5
        assert dense[1] == -0.5


# ============================================================================
# LazyArray Tests
# ============================================================================

class TestLazyArray:
    """Tests for LazyArray class."""

    def test_initialization(self):
        """Test LazyArray initialization."""
        from HoloLoom.semantic_calculus.performance import LazyArray

        def compute():
            return np.array([1, 2, 3])

        lazy = LazyArray(compute)

        assert not lazy.is_computed
        assert lazy._value is None

    def test_lazy_evaluation(self):
        """Test that computation is deferred."""
        from HoloLoom.semantic_calculus.performance import LazyArray

        call_count = [0]

        def compute():
            call_count[0] += 1
            return np.array([1, 2, 3])

        lazy = LazyArray(compute)

        # Not computed yet
        assert call_count[0] == 0

        # First access triggers computation
        result = lazy.get()

        assert call_count[0] == 1
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_caching(self):
        """Test that result is cached."""
        from HoloLoom.semantic_calculus.performance import LazyArray

        call_count = [0]

        def compute():
            call_count[0] += 1
            return np.random.randn(10)

        lazy = LazyArray(compute)

        # Multiple accesses
        result1 = lazy.get()
        result2 = lazy.get()
        result3 = lazy.get()

        # Should only compute once
        assert call_count[0] == 1
        assert np.array_equal(result1, result2)
        assert np.array_equal(result2, result3)

    def test_is_computed_property(self):
        """Test is_computed property."""
        from HoloLoom.semantic_calculus.performance import LazyArray

        lazy = LazyArray(lambda: np.array([1]))

        assert not lazy.is_computed

        lazy.get()

        assert lazy.is_computed

    def test_clear(self):
        """Test clearing cached value."""
        from HoloLoom.semantic_calculus.performance import LazyArray

        call_count = [0]

        def compute():
            call_count[0] += 1
            return np.array([call_count[0]])

        lazy = LazyArray(compute)

        result1 = lazy.get()
        assert lazy.is_computed
        assert result1[0] == 1

        # Clear and recompute
        lazy.clear()
        assert not lazy.is_computed

        result2 = lazy.get()
        assert result2[0] == 2
        assert call_count[0] == 2

    def test_with_expensive_computation(self):
        """Test with simulated expensive computation."""
        from HoloLoom.semantic_calculus.performance import LazyArray

        def expensive():
            time.sleep(0.01)
            return np.random.randn(1000)

        lazy = LazyArray(expensive)

        # First access - should take time
        start = time.time()
        result1 = lazy.get()
        first_time = time.time() - start

        # Second access - should be fast (cached)
        start = time.time()
        result2 = lazy.get()
        second_time = time.time() - start

        assert first_time > 0.005  # First took time
        assert second_time < 0.001  # Second was instant
        assert np.array_equal(result1, result2)


# ============================================================================
# Vectorized Function Tests
# ============================================================================

class TestComputeFiniteDifferenceVectorized:
    """Tests for compute_finite_difference_vectorized function."""

    def test_basic_computation(self):
        """Test basic finite difference computation."""
        from HoloLoom.semantic_calculus.performance import compute_finite_difference_vectorized

        # Linear trajectory: position = t
        positions = np.array([[0], [1], [2], [3], [4]], dtype=float)
        dt = 1.0

        velocities = compute_finite_difference_vectorized(positions, dt)

        # Velocity should be approximately 1 everywhere
        assert velocities.shape == positions.shape
        assert np.allclose(velocities, 1.0, atol=0.01)

    def test_forward_difference_first(self):
        """Test that first point uses forward difference."""
        from HoloLoom.semantic_calculus.performance import compute_finite_difference_vectorized

        positions = np.array([[0], [2], [3], [3.5]], dtype=float)
        dt = 1.0

        velocities = compute_finite_difference_vectorized(positions, dt)

        # First point: (2 - 0) / 1 = 2
        assert abs(velocities[0, 0] - 2.0) < 0.01

    def test_backward_difference_last(self):
        """Test that last point uses backward difference."""
        from HoloLoom.semantic_calculus.performance import compute_finite_difference_vectorized

        positions = np.array([[0], [1], [3], [6]], dtype=float)
        dt = 1.0

        velocities = compute_finite_difference_vectorized(positions, dt)

        # Last point: (6 - 3) / 1 = 3
        assert abs(velocities[-1, 0] - 3.0) < 0.01

    def test_central_difference_middle(self):
        """Test that middle points use central difference."""
        from HoloLoom.semantic_calculus.performance import compute_finite_difference_vectorized

        positions = np.array([[0], [1], [3], [6]], dtype=float)
        dt = 1.0

        velocities = compute_finite_difference_vectorized(positions, dt)

        # Middle point at index 1: (3 - 0) / 2 = 1.5
        assert abs(velocities[1, 0] - 1.5) < 0.01

        # Middle point at index 2: (6 - 1) / 2 = 2.5
        assert abs(velocities[2, 0] - 2.5) < 0.01

    def test_multidimensional(self):
        """Test with multi-dimensional positions."""
        from HoloLoom.semantic_calculus.performance import compute_finite_difference_vectorized

        positions = np.array([
            [0, 0],
            [1, 2],
            [2, 4],
            [3, 6],
        ], dtype=float)
        dt = 1.0

        velocities = compute_finite_difference_vectorized(positions, dt)

        assert velocities.shape == (4, 2)
        # First dimension: velocity ~1
        # Second dimension: velocity ~2
        assert np.allclose(velocities[:, 0], 1.0, atol=0.1)
        assert np.allclose(velocities[:, 1], 2.0, atol=0.1)

    def test_different_dt(self):
        """Test with different time step."""
        from HoloLoom.semantic_calculus.performance import compute_finite_difference_vectorized

        positions = np.array([[0], [0.5], [1], [1.5]], dtype=float)
        dt = 0.5

        velocities = compute_finite_difference_vectorized(positions, dt)

        # Velocity should be 1 (position changes by 0.5 per dt=0.5)
        assert np.allclose(velocities, 1.0, atol=0.01)


class TestComputeCurvatureVectorized:
    """Tests for compute_curvature_vectorized function."""

    def test_straight_line_zero_curvature(self):
        """Test that straight line has zero curvature."""
        from HoloLoom.semantic_calculus.performance import compute_curvature_vectorized

        n_steps = 10
        positions = np.zeros((n_steps, 3))
        positions[:, 0] = np.linspace(0, 1, n_steps)  # Linear x

        velocities = np.zeros((n_steps, 3))
        velocities[:, 0] = 1.0  # Constant velocity in x

        accelerations = np.zeros((n_steps, 3))  # Zero acceleration

        curvature = compute_curvature_vectorized(positions, velocities, accelerations)

        assert curvature.shape == (n_steps,)
        assert np.allclose(curvature, 0.0, atol=1e-6)

    def test_output_shape(self):
        """Test output shape."""
        from HoloLoom.semantic_calculus.performance import compute_curvature_vectorized

        n_steps = 20
        dim = 384

        positions = np.random.randn(n_steps, dim)
        velocities = np.random.randn(n_steps, dim)
        accelerations = np.random.randn(n_steps, dim)

        curvature = compute_curvature_vectorized(positions, velocities, accelerations)

        assert curvature.shape == (n_steps,)

    def test_curvature_non_negative(self):
        """Test that curvature is always non-negative."""
        from HoloLoom.semantic_calculus.performance import compute_curvature_vectorized

        np.random.seed(42)
        n_steps = 50

        positions = np.random.randn(n_steps, 10)
        velocities = np.random.randn(n_steps, 10)
        accelerations = np.random.randn(n_steps, 10)

        curvature = compute_curvature_vectorized(positions, velocities, accelerations)

        assert np.all(curvature >= 0)

    def test_zero_velocity_handling(self):
        """Test handling of zero velocity (should not produce NaN/Inf)."""
        from HoloLoom.semantic_calculus.performance import compute_curvature_vectorized

        n_steps = 5
        positions = np.zeros((n_steps, 3))
        velocities = np.zeros((n_steps, 3))  # Zero velocity
        accelerations = np.random.randn(n_steps, 3)

        curvature = compute_curvature_vectorized(positions, velocities, accelerations)

        assert not np.any(np.isnan(curvature))
        assert not np.any(np.isinf(curvature))
        assert np.allclose(curvature, 0.0)


# ============================================================================
# JIT Function Tests
# ============================================================================

class TestStormerVerletStepJit:
    """Tests for stormer_verlet_step_jit function."""

    def test_basic_step(self):
        """Test basic integration step."""
        from HoloLoom.semantic_calculus.performance import stormer_verlet_step_jit

        q = np.array([1.0, 0.0, 0.0])
        p = np.array([0.0, 1.0, 0.0])
        grad_q = np.array([0.1, 0.0, 0.0])
        grad_q_new = np.array([0.1, 0.0, 0.0])
        dt = 0.1
        mass = 1.0

        q_new, p_new = stormer_verlet_step_jit(q, p, grad_q, grad_q_new, dt, mass)

        assert q_new.shape == q.shape
        assert p_new.shape == p.shape

    def test_momentum_updates(self):
        """Test that momentum is updated correctly."""
        from HoloLoom.semantic_calculus.performance import stormer_verlet_step_jit

        q = np.array([0.0, 0.0])
        p = np.array([1.0, 0.0])
        grad_q = np.array([0.0, 0.0])
        grad_q_new = np.array([0.0, 0.0])
        dt = 0.1
        mass = 1.0

        q_new, p_new = stormer_verlet_step_jit(q, p, grad_q, grad_q_new, dt, mass)

        # With no gradient, momentum should stay the same
        assert np.allclose(p_new, p)

    def test_position_updates(self):
        """Test that position is updated correctly."""
        from HoloLoom.semantic_calculus.performance import stormer_verlet_step_jit

        q = np.array([0.0])
        p = np.array([1.0])
        grad_q = np.array([0.0])
        grad_q_new = np.array([0.0])
        dt = 0.1
        mass = 1.0

        q_new, p_new = stormer_verlet_step_jit(q, p, grad_q, grad_q_new, dt, mass)

        # q_new = q + dt * p / mass = 0 + 0.1 * 1.0 / 1.0 = 0.1
        assert abs(q_new[0] - 0.1) < 0.01

    def test_mass_scaling(self):
        """Test that mass affects position update."""
        from HoloLoom.semantic_calculus.performance import stormer_verlet_step_jit

        q = np.array([0.0])
        p = np.array([1.0])
        grad_q = np.array([0.0])
        grad_q_new = np.array([0.0])
        dt = 0.1

        # Mass = 1
        q1, _ = stormer_verlet_step_jit(q, p, grad_q, grad_q_new, dt, mass=1.0)

        # Mass = 2 (slower)
        q2, _ = stormer_verlet_step_jit(q, p, grad_q, grad_q_new, dt, mass=2.0)

        assert abs(q2[0]) < abs(q1[0])  # Higher mass = smaller position change

    def test_gradient_effect(self):
        """Test that gradient affects momentum."""
        from HoloLoom.semantic_calculus.performance import stormer_verlet_step_jit

        q = np.array([0.0])
        p = np.array([0.0])
        grad_q = np.array([1.0])
        grad_q_new = np.array([1.0])
        dt = 0.1
        mass = 1.0

        q_new, p_new = stormer_verlet_step_jit(q, p, grad_q, grad_q_new, dt, mass)

        # Momentum should decrease (gradient is positive)
        assert p_new[0] < 0


class TestComputeGradientBatchJit:
    """Tests for compute_gradient_batch_jit function."""

    def test_basic_computation(self):
        """Test basic gradient computation."""
        from HoloLoom.semantic_calculus.performance import compute_gradient_batch_jit

        Q = np.array([[1.0, 2.0]])  # 1 point, 2 dims
        coeffs = np.array([
            [0.0, 0.0],  # constant term
            [1.0, 1.0],  # linear term
        ])

        gradients = compute_gradient_batch_jit(Q, coeffs)

        # grad of c1*x = c1 = 1.0
        assert gradients.shape == (1, 2)
        assert np.allclose(gradients, 1.0)

    def test_output_shape(self):
        """Test output shape."""
        from HoloLoom.semantic_calculus.performance import compute_gradient_batch_jit

        n_batch = 10
        n_dims = 5
        n_terms = 3

        Q = np.random.randn(n_batch, n_dims)
        coeffs = np.random.randn(n_terms, n_dims)

        gradients = compute_gradient_batch_jit(Q, coeffs)

        assert gradients.shape == (n_batch, n_dims)

    def test_quadratic_gradient(self):
        """Test gradient of quadratic term."""
        from HoloLoom.semantic_calculus.performance import compute_gradient_batch_jit

        Q = np.array([[2.0]])  # x = 2
        coeffs = np.array([
            [0.0],  # constant term (k=0)
            [0.0],  # linear term (k=1)
            [1.0],  # quadratic term (k=2)
        ])

        # V(x) = x^2, grad V = 2x = 4
        gradients = compute_gradient_batch_jit(Q, coeffs)

        assert abs(gradients[0, 0] - 4.0) < 0.01

    def test_zero_coefficients(self):
        """Test with zero coefficients."""
        from HoloLoom.semantic_calculus.performance import compute_gradient_batch_jit

        Q = np.random.randn(5, 3)
        coeffs = np.zeros((4, 3))

        gradients = compute_gradient_batch_jit(Q, coeffs)

        assert np.allclose(gradients, 0.0)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformanceOptimizations:
    """Tests verifying performance characteristics."""

    def test_embedding_cache_speedup(self):
        """Test that cache provides speedup for repeated lookups."""
        from HoloLoom.semantic_calculus.performance import EmbeddingCache

        def slow_embed(words):
            time.sleep(0.001 * len(words))  # Simulate slow embedding
            return np.random.randn(len(words), 384)

        cache = EmbeddingCache(slow_embed, max_size=100)
        words = ["word_" + str(i) for i in range(10)]

        # First pass - cold cache
        start = time.time()
        for word in words:
            cache.get(word)
        cold_time = time.time() - start

        # Second pass - warm cache
        start = time.time()
        for word in words:
            cache.get(word)
        warm_time = time.time() - start

        # Warm should be much faster (at least 5x)
        assert warm_time < cold_time / 5

    def test_sparse_vector_memory_efficiency(self):
        """Test that sparse vectors use less memory for sparse data."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        # 99% sparse vector
        values = np.zeros(10000)
        values[np.random.choice(10000, 100, replace=False)] = np.random.randn(100)

        sparse = SparseSemanticVector(values, threshold=0.01)

        # Sparse should store much less
        sparse_size = len(sparse.active_indices) + len(sparse.active_values)
        dense_size = len(values)

        assert sparse_size < dense_size / 10

    def test_lazy_array_deferred_computation(self):
        """Test that LazyArray doesn't compute until needed."""
        from HoloLoom.semantic_calculus.performance import LazyArray

        computation_time = [0.0]

        def expensive():
            start = time.time()
            result = np.random.randn(10000, 1000)  # Large computation
            computation_time[0] = time.time() - start
            return result

        lazy = LazyArray(expensive)

        # No computation yet
        assert computation_time[0] == 0.0

        # Trigger computation
        lazy.get()

        assert computation_time[0] > 0

    def test_vectorized_vs_loop_speedup(self):
        """Test that vectorized operations are faster than loops."""
        from HoloLoom.semantic_calculus.performance import compute_finite_difference_vectorized

        n_steps = 1000
        dim = 100
        positions = np.random.randn(n_steps, dim)
        dt = 1.0

        # Vectorized
        start = time.time()
        for _ in range(10):
            compute_finite_difference_vectorized(positions, dt)
        vectorized_time = time.time() - start

        # Loop-based (naive implementation)
        def naive_fd(positions, dt):
            velocities = np.zeros_like(positions)
            for i in range(len(positions)):
                if i == 0:
                    velocities[i] = (positions[1] - positions[0]) / dt
                elif i == len(positions) - 1:
                    velocities[i] = (positions[-1] - positions[-2]) / dt
                else:
                    velocities[i] = (positions[i+1] - positions[i-1]) / (2 * dt)
            return velocities

        start = time.time()
        for _ in range(10):
            naive_fd(positions, dt)
        loop_time = time.time() - start

        # Vectorized should be faster
        assert vectorized_time < loop_time


# ============================================================================
# Edge Cases and Numerical Stability
# ============================================================================

class TestEdgeCasesAndNumericalStability:
    """Tests for edge cases and numerical stability."""

    def test_very_large_cache(self):
        """Test cache with many entries."""
        from HoloLoom.semantic_calculus.performance import EmbeddingCache

        def embed(words):
            return np.random.randn(len(words), 384)

        cache = EmbeddingCache(embed, max_size=10000)

        # Add many entries
        for i in range(5000):
            cache.get(f"word_{i}")

        assert len(cache.cache) == 5000

        stats = cache.get_stats()
        assert stats['size'] == 5000

    def test_sparse_vector_high_dimension(self):
        """Test sparse vector with very high dimensions."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector

        values = np.zeros(100000)
        values[np.random.choice(100000, 100, replace=False)] = np.random.randn(100)

        sparse = SparseSemanticVector(values, threshold=0.01)

        assert sparse.n_dims == 100000
        assert len(sparse.active_indices) == 100
        assert abs(sparse.sparsity - 0.999) < 0.001

        # Conversion back should work
        dense = sparse.to_dense()
        assert dense.shape == (100000,)

    def test_finite_difference_short_trajectory(self):
        """Test finite difference with minimum length trajectory."""
        from HoloLoom.semantic_calculus.performance import compute_finite_difference_vectorized

        # Minimum: 2 points
        positions = np.array([[0.0], [1.0]])
        dt = 1.0

        velocities = compute_finite_difference_vectorized(positions, dt)

        assert velocities.shape == (2, 1)
        assert np.allclose(velocities, 1.0)

    def test_curvature_parallel_va(self):
        """Test curvature when velocity and acceleration are parallel."""
        from HoloLoom.semantic_calculus.performance import compute_curvature_vectorized

        n_steps = 5
        positions = np.zeros((n_steps, 3))

        # Velocity and acceleration both in x direction (parallel)
        velocities = np.zeros((n_steps, 3))
        velocities[:, 0] = 1.0

        accelerations = np.zeros((n_steps, 3))
        accelerations[:, 0] = 0.5

        curvature = compute_curvature_vectorized(positions, velocities, accelerations)

        # Parallel v and a means zero curvature (no turning)
        assert np.allclose(curvature, 0.0, atol=1e-6)

    def test_jit_with_empty_arrays(self):
        """Test JIT functions don't fail with edge dimensions."""
        from HoloLoom.semantic_calculus.performance import (
            stormer_verlet_step_jit,
            compute_gradient_batch_jit,
        )

        # Single dimension
        q = np.array([1.0])
        p = np.array([0.5])
        grad = np.array([0.1])

        q_new, p_new = stormer_verlet_step_jit(q, p, grad, grad, 0.1, 1.0)
        assert q_new.shape == (1,)

        # Single batch
        Q = np.array([[1.0, 2.0]])
        coeffs = np.array([[1.0, 1.0]])

        gradients = compute_gradient_batch_jit(Q, coeffs)
        assert gradients.shape == (1, 2)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for performance utilities."""

    def test_cache_with_projection(self):
        """Test EmbeddingCache and ProjectionCache working together."""
        from HoloLoom.semantic_calculus.performance import EmbeddingCache, ProjectionCache

        def embed(words):
            result = []
            for word in words:
                np.random.seed(hash(word) % 2**31)
                result.append(np.random.randn(384))
            return np.array(result)

        emb_cache = EmbeddingCache(embed, max_size=100)
        proj_cache = ProjectionCache(max_size=100)

        np.random.seed(42)
        P = np.random.randn(16, 384)

        # Get embedding and project
        embedding = emb_cache.get("test_word")
        projection = proj_cache.project(P, embedding)

        assert embedding.shape == (384,)
        assert projection.shape == (16,)

        # Second time should use caches
        embedding2 = emb_cache.get("test_word")
        projection2 = proj_cache.project(P, embedding2)

        assert emb_cache.hits == 1
        assert proj_cache.hits == 1

    def test_sparse_vector_with_lazy_array(self):
        """Test SparseSemanticVector with LazyArray."""
        from HoloLoom.semantic_calculus.performance import SparseSemanticVector, LazyArray

        def compute_semantic_vector():
            # Simulate expensive computation
            values = np.zeros(244)
            # Set only a few dimensions
            for i in [0, 5, 10, 15]:
                values[i] = np.random.randn()
            return values

        lazy = LazyArray(compute_semantic_vector)

        # Not computed yet
        assert not lazy.is_computed

        # Get and convert to sparse
        dense = lazy.get()
        sparse = SparseSemanticVector(dense, threshold=0.01)

        assert lazy.is_computed
        assert sparse.sparsity > 0.9  # Mostly zero

    def test_full_workflow(self):
        """Test complete performance-optimized workflow."""
        from HoloLoom.semantic_calculus.performance import (
            EmbeddingCache,
            ProjectionCache,
            SparseSemanticVector,
            batch_iterator,
        )

        # Setup
        def embed(words):
            result = []
            for word in words:
                np.random.seed(hash(word) % 2**31)
                vec = np.random.randn(384)
                vec = vec / np.linalg.norm(vec)
                result.append(vec)
            return np.array(result)

        emb_cache = EmbeddingCache(embed, max_size=1000)
        proj_cache = ProjectionCache(max_size=1000)

        np.random.seed(42)
        P = np.random.randn(16, 384)

        # Process many words in batches
        words = [f"word_{i}" for i in range(100)]
        projections = []

        for batch in batch_iterator(words, batch_size=10):
            embeddings = emb_cache.get_batch(batch)

            for emb in embeddings:
                proj = proj_cache.project(P, emb)
                sparse = SparseSemanticVector(proj, threshold=0.01)
                projections.append(sparse)

        assert len(projections) == 100

        # Second pass should be faster (cached)
        emb_stats = emb_cache.get_stats()
        assert emb_stats['misses'] == 100  # First pass was all misses

        emb_cache.get_batch(words[:10])
        emb_stats = emb_cache.get_stats()
        assert emb_stats['hits'] == 10  # Second pass all hits


# ============================================================================
# HAS_NUMBA Flag Tests
# ============================================================================

class TestNumbaFlag:
    """Tests for HAS_NUMBA flag behavior."""

    def test_has_numba_is_boolean(self):
        """Test that HAS_NUMBA is a boolean."""
        from HoloLoom.semantic_calculus.performance import HAS_NUMBA

        assert isinstance(HAS_NUMBA, bool)

    def test_jit_functions_work_regardless_of_numba(self):
        """Test that JIT functions work with or without numba."""
        from HoloLoom.semantic_calculus.performance import (
            stormer_verlet_step_jit,
            compute_gradient_batch_jit,
            HAS_NUMBA,
        )

        # These should work regardless of HAS_NUMBA
        q = np.array([1.0, 2.0, 3.0])
        p = np.array([0.1, 0.2, 0.3])
        grad = np.array([0.01, 0.02, 0.03])

        q_new, p_new = stormer_verlet_step_jit(q, p, grad, grad, 0.1, 1.0)
        assert q_new.shape == q.shape

        Q = np.random.randn(5, 3)
        coeffs = np.random.randn(3, 3)
        gradients = compute_gradient_batch_jit(Q, coeffs)
        assert gradients.shape == Q.shape

        # Report numba status for debugging
        print(f"\nHAS_NUMBA: {HAS_NUMBA}")
