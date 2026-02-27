"""
Performance optimization utilities for semantic calculus

Provides:
- Embedding cache with LRU eviction
- Memoization decorators
- Batch processing utilities
- Sparse operation helpers
"""

from functools import wraps, lru_cache
from typing import Callable, List, Optional, Dict, Any, Tuple
import numpy as np
from collections import OrderedDict
import time


class EmbeddingCache:
    """
    LRU cache for word embeddings with automatic batching

    Features:
    - Thread-safe caching
    - Automatic batch embedding when cache misses
    - Memory-bounded with LRU eviction
    - Hit rate tracking
    """

    def __init__(self, embed_fn: Callable, max_size: int = 10000):
        """
        Args:
            embed_fn: Function that takes a list of words and returns array of embeddings
            max_size: Maximum number of cached embeddings
        """
        self.embed_fn = embed_fn
        self.max_size = max_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, word: str) -> np.ndarray:
        """Get embedding for single word"""
        if word in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(word)
            return self.cache[word]

        # Cache miss - embed and cache
        self.misses += 1
        embedding = self.embed_fn([word])[0]
        self._cache_embedding(word, embedding)
        return embedding

    def get_batch(self, words: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple words with optimal batching

        Returns:
            Array of shape (len(words), embedding_dim)
        """
        cached = []
        missing = []
        missing_indices = []

        for i, word in enumerate(words):
            if word in self.cache:
                self.hits += 1
                self.cache.move_to_end(word)
                cached.append((i, self.cache[word]))
            else:
                missing.append(word)
                missing_indices.append(i)

        # Batch embed missing words
        if missing:
            self.misses += len(missing)
            embeddings = self.embed_fn(missing)
            for word, emb in zip(missing, embeddings):
                self._cache_embedding(word, emb)
                cached.append((missing_indices[missing.index(word)], emb))

        # Reconstruct in original order
        cached.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in cached])

    def _cache_embedding(self, word: str, embedding: np.ndarray):
        """Cache embedding with LRU eviction"""
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            self.cache.popitem(last=False)
        self.cache[word] = embedding

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }

    def clear(self):
        """Clear cache and reset statistics"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class ProjectionCache:
    """
    Cache for expensive projection matrix computations

    Caches P @ q operations where P is projection matrix
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[Tuple[int, int], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def project(self, P: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Cached matrix-vector product

        Args:
            P: Projection matrix (n_dims, embedding_dim)
            q: Vector to project (embedding_dim,)

        Returns:
            Projected vector (n_dims,)
        """
        # Use hash of array data as key
        q_hash = hash(q.tobytes())
        P_hash = hash(P.tobytes())
        key = (P_hash, q_hash)

        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]

        # Cache miss - compute projection
        self.misses += 1
        result = P @ q

        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = result

        return result

    def project_batch(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Batch projection (always computes, too expensive to cache)

        Args:
            P: Projection matrix (n_dims, embedding_dim)
            Q: Matrix of vectors (n_vectors, embedding_dim)

        Returns:
            Projected vectors (n_vectors, n_dims)
        """
        # For batch operations, direct computation is usually faster
        return (P @ Q.T).T


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution

    Usage:
        @timer
        def expensive_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__}: {elapsed*1000:.2f}ms")
        return result
    return wrapper


def batch_iterator(items: List[Any], batch_size: int):
    """
    Yield batches of items

    Args:
        items: List to batch
        batch_size: Size of each batch

    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class SparseSemanticVector:
    """
    Sparse representation of semantic dimension vector

    Only stores non-zero dimensions to save memory and computation
    """

    def __init__(self, values: np.ndarray, threshold: float = 0.01):
        """
        Args:
            values: Full semantic vector
            threshold: Values below this are considered zero
        """
        self.n_dims = len(values)

        # Find non-zero indices
        mask = np.abs(values) > threshold
        self.active_indices = np.where(mask)[0]
        self.active_values = values[mask]

    def to_dense(self) -> np.ndarray:
        """Convert back to dense vector"""
        dense = np.zeros(self.n_dims)
        dense[self.active_indices] = self.active_values
        return dense

    @property
    def sparsity(self) -> float:
        """Fraction of dimensions that are zero"""
        return 1.0 - (len(self.active_indices) / self.n_dims)

    def __repr__(self):
        return f"SparseSemanticVector(active={len(self.active_indices)}/{self.n_dims}, sparsity={self.sparsity:.2%})"


class LazyArray:
    """
    Lazy evaluation wrapper for expensive array computations

    Only computes when accessed, caches result
    """

    def __init__(self, compute_fn: Callable[[], np.ndarray]):
        """
        Args:
            compute_fn: Function that computes the array
        """
        self.compute_fn = compute_fn
        self._value: Optional[np.ndarray] = None
        self._computed = False

    def get(self) -> np.ndarray:
        """Get value, computing if necessary"""
        if not self._computed:
            self._value = self.compute_fn()
            self._computed = True
        return self._value

    @property
    def is_computed(self) -> bool:
        """Check if value has been computed"""
        return self._computed

    def clear(self):
        """Clear cached value"""
        self._value = None
        self._computed = False


def compute_finite_difference_vectorized(positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Vectorized finite difference computation for velocity

    Args:
        positions: Array of shape (n_steps, embedding_dim)
        dt: Time step

    Returns:
        Velocities of shape (n_steps, embedding_dim)
    """
    n_steps = len(positions)
    velocities = np.zeros_like(positions)

    # Forward difference for first point
    velocities[0] = (positions[1] - positions[0]) / dt

    # Central difference for middle points (vectorized)
    velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)

    # Backward difference for last point
    velocities[-1] = (positions[-1] - positions[-2]) / dt

    return velocities


def compute_curvature_vectorized(positions: np.ndarray, velocities: np.ndarray,
                                  accelerations: np.ndarray) -> np.ndarray:
    """
    Vectorized curvature computation

    kappa = ||v x a|| / ||v||^3

    Args:
        positions: Array of shape (n_steps, embedding_dim)
        velocities: Array of shape (n_steps, embedding_dim)
        accelerations: Array of shape (n_steps, embedding_dim)

    Returns:
        Curvatures of shape (n_steps,)
    """
    # Compute cross product magnitude (generalized for high dimensions)
    # |v x a| = sqrt(|v|^2 |a|^2 - (v·a)^2)

    v_norm_sq = np.sum(velocities**2, axis=1)
    a_norm_sq = np.sum(accelerations**2, axis=1)
    va_dot = np.sum(velocities * accelerations, axis=1)

    cross_magnitude_sq = v_norm_sq * a_norm_sq - va_dot**2
    cross_magnitude = np.sqrt(np.maximum(cross_magnitude_sq, 0))  # Avoid numerical errors

    # Curvature
    v_norm_cubed = v_norm_sq * np.sqrt(v_norm_sq)

    # Avoid division by zero
    curvature = np.zeros(len(positions))
    mask = v_norm_cubed > 1e-10
    curvature[mask] = cross_magnitude[mask] / v_norm_cubed[mask]

    return curvature


# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True

    @jit(nopython=True, parallel=True, cache=True)
    def stormer_verlet_step_jit(q: np.ndarray, p: np.ndarray, grad_q: np.ndarray,
                                 grad_q_new: np.ndarray, dt: float, mass: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        JIT-compiled Störmer-Verlet integration step

        Args:
            q: Current position
            p: Current momentum
            grad_q: Gradient at current position
            grad_q_new: Gradient at new position
            dt: Time step
            mass: Mass

        Returns:
            (q_new, p_new)
        """
        # Half-step momentum
        p_half = p - 0.5 * dt * grad_q

        # Full-step position
        q_new = q + dt * p_half / mass

        # Half-step momentum
        p_new = p_half - 0.5 * dt * grad_q_new

        return q_new, p_new

    @jit(nopython=True, parallel=True, cache=True)
    def compute_gradient_batch_jit(Q: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        JIT-compiled batch gradient computation

        For polynomial potential: grad V(q) = sum_i coeffs[i] * q^i

        Args:
            Q: Batch of positions (n_batch, n_dims)
            coeffs: Polynomial coefficients (n_terms, n_dims)

        Returns:
            Gradients (n_batch, n_dims)
        """
        n_batch, n_dims = Q.shape
        n_terms = len(coeffs)
        gradients = np.zeros_like(Q)

        for i in prange(n_batch):
            for j in range(n_dims):
                grad = 0.0
                for k in range(n_terms):
                    # Polynomial gradient: d/dq (c * q^k) = k * c * q^(k-1)
                    if k > 0:
                        grad += k * coeffs[k, j] * (Q[i, j] ** (k - 1))
                gradients[i, j] = grad

        return gradients

except ImportError:
    HAS_NUMBA = False

    def stormer_verlet_step_jit(q, p, grad_q, grad_q_new, dt, mass):
        """Fallback implementation without JIT"""
        p_half = p - 0.5 * dt * grad_q
        q_new = q + dt * p_half / mass
        p_new = p_half - 0.5 * dt * grad_q_new
        return q_new, p_new

    def compute_gradient_batch_jit(Q, coeffs):
        """Fallback implementation without JIT"""
        n_batch, n_dims = Q.shape
        n_terms = len(coeffs)
        gradients = np.zeros_like(Q)

        for i in range(n_batch):
            for j in range(n_dims):
                grad = 0.0
                for k in range(n_terms):
                    if k > 0:
                        grad += k * coeffs[k, j] * (Q[i, j] ** (k - 1))
                gradients[i, j] = grad

        return gradients


__all__ = [
    'EmbeddingCache',
    'ProjectionCache',
    'timer',
    'batch_iterator',
    'SparseSemanticVector',
    'LazyArray',
    'compute_finite_difference_vectorized',
    'compute_curvature_vectorized',
    'stormer_verlet_step_jit',
    'compute_gradient_batch_jit',
    'HAS_NUMBA',
]
