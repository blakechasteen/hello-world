"""
Rust Operations Wrapper - High-Performance Clustering with NumPy Fallback

Provides a unified interface for clustering and similarity operations that uses
Rust implementations when available (30-50x speedup) with automatic fallback
to NumPy when Rust bindings are not installed.

The Rust implementation provides:
- SIMD-accelerated cosine similarity (AVX2/AVX-512)
- Parallel k-means clustering
- Optimized silhouette score computation

Usage:
    from HoloLoom.eggroll.rust_ops import (
        cosine_similarity_batch,
        normalize_batch,
        kmeans_cluster,
        silhouette_score,
        RUST_AVAILABLE
    )

    # Functions automatically use Rust if available, NumPy otherwise
    similarities = cosine_similarity_batch(vectors, centroid)
    labels, centroids, inertia = kmeans_cluster(data, k=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  RUST AVAILABILITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

RUST_AVAILABLE = False
_RUST_VERSION = None

try:
    from hololoom_rust import (
        cosine_similarity_batch as _rust_cosine_similarity,
        normalize_batch as _rust_normalize,
        kmeans as _rust_kmeans,
        silhouette_score as _rust_silhouette,
        __version__ as _rust_version,
    )
    RUST_AVAILABLE = True
    _RUST_VERSION = _rust_version
    logger.info(f"Rust clustering backend available (v{_rust_version}) - 30-50x speedup enabled")
except ImportError:
    logger.info("Rust clustering backend not available - using NumPy fallback")


def get_backend_info() -> dict:
    """Get information about the active backend.

    Returns:
        Dictionary with backend status and version info
    """
    return {
        "rust_available": RUST_AVAILABLE,
        "rust_version": _RUST_VERSION,
        "backend": "rust" if RUST_AVAILABLE else "numpy",
        "estimated_speedup": "30-50x" if RUST_AVAILABLE else "1x",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NUMPY FALLBACK IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def _numpy_cosine_similarity_batch(
    vectors: np.ndarray,
    centroid: np.ndarray
) -> np.ndarray:
    """NumPy fallback for batch cosine similarity.

    Args:
        vectors: 2D array of shape (n_vectors, dim)
        centroid: 1D array of shape (dim,)

    Returns:
        1D array of cosine similarities, shape (n_vectors,)
    """
    # Normalize vectors
    vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = np.where(vectors_norm > 0, vectors_norm, 1.0)  # Avoid division by zero
    vectors_normalized = vectors / vectors_norm

    # Normalize centroid
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid_normalized = centroid / centroid_norm
    else:
        centroid_normalized = centroid

    # Compute cosine similarity as dot product of normalized vectors
    similarities = vectors_normalized @ centroid_normalized

    return similarities.astype(np.float32)


def _numpy_normalize_batch(vectors: np.ndarray) -> np.ndarray:
    """NumPy fallback for batch vector normalization.

    Args:
        vectors: 2D array of shape (n_vectors, dim)

    Returns:
        2D array of normalized vectors, shape (n_vectors, dim)
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
    return (vectors / norms).astype(np.float32)


def _numpy_kmeans(
    data: np.ndarray,
    k: int,
    max_iterations: int = 300,
    tolerance: float = 1e-4,
    n_init: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """NumPy fallback for k-means clustering.

    Uses k-means++ initialization and Lloyd's algorithm.

    Args:
        data: 2D array of shape (n_samples, dim)
        k: Number of clusters
        max_iterations: Maximum iterations per initialization
        tolerance: Convergence tolerance
        n_init: Number of initializations (best result is returned)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (labels, centroids, inertia, n_iterations)
    """
    rng = np.random.default_rng(seed)
    n_samples, dim = data.shape

    best_labels = None
    best_centroids = None
    best_inertia = np.inf
    best_n_iter = 0

    for init_idx in range(n_init):
        # K-means++ initialization
        centroids = _kmeans_plusplus_init(data, k, rng)

        prev_inertia = np.inf
        for iteration in range(max_iterations):
            # Assignment step
            distances = _compute_distances(data, centroids)
            labels = np.argmin(distances, axis=1)

            # Compute inertia (sum of squared distances to assigned centroid)
            inertia = np.sum(np.min(distances, axis=1))

            # Check convergence
            if abs(prev_inertia - inertia) < tolerance:
                break
            prev_inertia = inertia

            # Update step
            new_centroids = np.zeros((k, dim), dtype=np.float32)
            for cluster_idx in range(k):
                mask = labels == cluster_idx
                if np.any(mask):
                    new_centroids[cluster_idx] = data[mask].mean(axis=0)
                else:
                    # Empty cluster - reinitialize randomly
                    new_centroids[cluster_idx] = data[rng.integers(0, n_samples)]

            centroids = new_centroids

        # Keep best result across initializations
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids
            best_n_iter = iteration + 1

    return best_labels, best_centroids, best_inertia, best_n_iter


def _kmeans_plusplus_init(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator
) -> np.ndarray:
    """K-means++ initialization.

    Selects initial centroids with probability proportional to squared distance
    from nearest existing centroid.
    """
    n_samples, dim = data.shape
    centroids = np.zeros((k, dim), dtype=np.float32)

    # First centroid: random sample
    centroids[0] = data[rng.integers(0, n_samples)]

    for i in range(1, k):
        # Compute distances to nearest centroid
        distances = _compute_distances(data, centroids[:i])
        min_distances = np.min(distances, axis=1)

        # Sample proportional to squared distance
        probs = min_distances / (min_distances.sum() + 1e-10)
        centroids[i] = data[rng.choice(n_samples, p=probs)]

    return centroids


def _compute_distances(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distances from data points to centroids.

    Args:
        data: (n_samples, dim)
        centroids: (k, dim)

    Returns:
        Distance matrix (n_samples, k)
    """
    # Efficient computation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    data_sq = np.sum(data ** 2, axis=1, keepdims=True)  # (n_samples, 1)
    centroids_sq = np.sum(centroids ** 2, axis=1)  # (k,)
    cross = data @ centroids.T  # (n_samples, k)

    distances = data_sq + centroids_sq - 2 * cross
    distances = np.maximum(distances, 0)  # Numerical stability

    return distances


def _numpy_silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """NumPy fallback for silhouette score computation.

    Args:
        data: 2D array of shape (n_samples, dim)
        labels: 1D array of cluster assignments

    Returns:
        Mean silhouette coefficient [-1, 1]
    """
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return 0.0

    # Compute pairwise distances
    # Using memory-efficient approach for large datasets
    silhouette_values = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        sample = data[i]
        sample_label = labels[i]

        # Intra-cluster distance (a)
        same_cluster = labels == sample_label
        if same_cluster.sum() > 1:
            a = np.mean(np.sqrt(np.sum((data[same_cluster] - sample) ** 2, axis=1)))
        else:
            a = 0.0

        # Inter-cluster distances (b)
        b = np.inf
        for cluster_label in unique_labels:
            if cluster_label == sample_label:
                continue
            other_cluster = labels == cluster_label
            if other_cluster.sum() > 0:
                mean_dist = np.mean(np.sqrt(np.sum((data[other_cluster] - sample) ** 2, axis=1)))
                b = min(b, mean_dist)

        if b == np.inf:
            b = 0.0

        # Silhouette coefficient
        max_ab = max(a, b)
        if max_ab > 0:
            silhouette_values[i] = (b - a) / max_ab
        else:
            silhouette_values[i] = 0.0

    return float(np.mean(silhouette_values))


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API - UNIFIED INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════


def cosine_similarity_batch(
    vectors: np.ndarray,
    centroid: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between multiple vectors and a centroid.

    Uses Rust implementation if available (30-50x faster), otherwise NumPy.

    Args:
        vectors: 2D array of shape (n_vectors, dim), dtype float32
        centroid: 1D array of shape (dim,), dtype float32

    Returns:
        1D array of cosine similarities in range [-1, 1], shape (n_vectors,)

    Example:
        >>> vectors = np.random.randn(1000, 384).astype(np.float32)
        >>> centroid = np.random.randn(384).astype(np.float32)
        >>> similarities = cosine_similarity_batch(vectors, centroid)
    """
    # Ensure proper types
    vectors = np.asarray(vectors, dtype=np.float32)
    centroid = np.asarray(centroid, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
    if centroid.ndim != 1:
        raise ValueError(f"centroid must be 1D, got shape {centroid.shape}")
    if vectors.shape[1] != centroid.shape[0]:
        raise ValueError(
            f"Dimension mismatch: vectors have dim={vectors.shape[1]}, "
            f"centroid has dim={centroid.shape[0]}"
        )

    if RUST_AVAILABLE:
        return _rust_cosine_similarity(vectors, centroid)
    else:
        return _numpy_cosine_similarity_batch(vectors, centroid)


def normalize_batch(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length.

    Uses Rust implementation if available (30-50x faster), otherwise NumPy.

    Args:
        vectors: 2D array of shape (n_vectors, dim), dtype float32

    Returns:
        2D array of normalized vectors, shape (n_vectors, dim)

    Example:
        >>> vectors = np.random.randn(1000, 384).astype(np.float32)
        >>> normalized = normalize_batch(vectors)
        >>> norms = np.linalg.norm(normalized, axis=1)  # All ~1.0
    """
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")

    if RUST_AVAILABLE:
        return _rust_normalize(vectors)
    else:
        return _numpy_normalize_batch(vectors)


@dataclass
class KMeansResult:
    """Result of k-means clustering."""
    labels: np.ndarray  # Cluster assignments (n_samples,)
    centroids: np.ndarray  # Cluster centers (k, dim)
    inertia: float  # Sum of squared distances
    n_iterations: int  # Number of iterations


def kmeans_cluster(
    data: np.ndarray,
    k: int,
    max_iterations: int = 300,
    tolerance: float = 1e-4,
    n_init: int = 10,
    seed: int = 42,
) -> KMeansResult:
    """K-means clustering with k-means++ initialization.

    Uses Rust implementation if available (30-50x faster), otherwise NumPy.

    Args:
        data: 2D array of shape (n_samples, dim), dtype float32
        k: Number of clusters
        max_iterations: Maximum iterations per initialization
        tolerance: Convergence tolerance
        n_init: Number of initializations (best result is returned)
        seed: Random seed for reproducibility

    Returns:
        KMeansResult with labels, centroids, inertia, and n_iterations

    Example:
        >>> data = np.random.randn(1000, 384).astype(np.float32)
        >>> result = kmeans_cluster(data, k=5)
        >>> print(f"Inertia: {result.inertia:.2f}")
    """
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > data.shape[0]:
        raise ValueError(f"k ({k}) cannot exceed n_samples ({data.shape[0]})")

    if RUST_AVAILABLE:
        labels, centroids, inertia, n_iter = _rust_kmeans(
            data, k, max_iterations, tolerance, n_init, seed
        )
        labels = np.asarray(labels, dtype=np.int64)
        centroids = np.asarray(centroids, dtype=np.float32)
    else:
        labels, centroids, inertia, n_iter = _numpy_kmeans(
            data, k, max_iterations, tolerance, n_init, seed
        )

    return KMeansResult(
        labels=labels,
        centroids=centroids,
        inertia=inertia,
        n_iterations=n_iter,
    )


def silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score for clustering quality.

    Uses Rust implementation if available (30-50x faster), otherwise NumPy.

    Args:
        data: 2D array of shape (n_samples, dim)
        labels: 1D array of cluster assignments (n_samples,)

    Returns:
        Mean silhouette coefficient in range [-1, 1]
        Higher is better (1 = perfect, -1 = wrong clustering)

    Example:
        >>> data = np.random.randn(1000, 384).astype(np.float32)
        >>> result = kmeans_cluster(data, k=5)
        >>> score = silhouette_score(data, result.labels)
        >>> print(f"Silhouette: {score:.3f}")
    """
    data = np.asarray(data, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if len(labels) != data.shape[0]:
        raise ValueError(
            f"Shape mismatch: data has {data.shape[0]} samples, "
            f"labels has {len(labels)} elements"
        )

    if RUST_AVAILABLE:
        return _rust_silhouette(data, labels)
    else:
        return _numpy_silhouette_score(data, labels)


# ═══════════════════════════════════════════════════════════════════════════════
#  EGGROLL-SPECIFIC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def cluster_population(
    embeddings: np.ndarray,
    n_clusters: int = 5,
    seed: int = 42,
) -> KMeansResult:
    """Cluster a population of embeddings for evolutionary diversity analysis.

    This is a convenience wrapper for k-means clustering tailored to Eggroll's
    population-based optimization. Uses adaptive cluster count if population
    is too small for requested clusters.

    Args:
        embeddings: Population embeddings (n_population, dim)
        n_clusters: Desired number of clusters (auto-adjusted if needed)
        seed: Random seed

    Returns:
        KMeansResult with cluster assignments and centroids
    """
    n_samples = embeddings.shape[0]

    # Adjust cluster count if population too small
    effective_k = min(n_clusters, max(2, n_samples // 2))
    if effective_k != n_clusters:
        logger.debug(f"Adjusted clusters from {n_clusters} to {effective_k} for population size {n_samples}")

    return kmeans_cluster(embeddings, k=effective_k, seed=seed)


def compute_warp_similarity(
    candidate_embeddings: np.ndarray,
    target_embedding: np.ndarray,
) -> np.ndarray:
    """Compute Warp-space similarity scores for fitness evaluation.

    Computes cosine similarity between candidate solutions and target,
    returning scores suitable for evolutionary selection.

    Args:
        candidate_embeddings: Candidate embeddings (n_candidates, dim)
        target_embedding: Target embedding (dim,)

    Returns:
        Similarity scores in range [-1, 1], higher is better
    """
    return cosine_similarity_batch(candidate_embeddings, target_embedding)


def diversity_bonus(
    population_embeddings: np.ndarray,
    n_clusters: int = 5,
    seed: int = 42,
) -> float:
    """Compute population diversity bonus using clustering quality.

    Returns a bonus value based on how well-separated the population
    clusters are. Higher values indicate more diverse populations,
    which is beneficial for exploration in evolutionary strategies.

    Args:
        population_embeddings: Population embeddings
        n_clusters: Number of clusters for diversity measurement
        seed: Random seed

    Returns:
        Diversity score in range [0, 1] (mapped from silhouette [-1, 1])
    """
    if population_embeddings.shape[0] < 4:
        # Too few samples for meaningful clustering
        return 0.5

    result = cluster_population(population_embeddings, n_clusters, seed)
    score = silhouette_score(population_embeddings, result.labels)

    # Map from [-1, 1] to [0, 1]
    return (score + 1.0) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Availability flag
    "RUST_AVAILABLE",
    "get_backend_info",
    # Core functions
    "cosine_similarity_batch",
    "normalize_batch",
    "kmeans_cluster",
    "silhouette_score",
    # Result types
    "KMeansResult",
    # Eggroll utilities
    "cluster_population",
    "compute_warp_similarity",
    "diversity_bonus",
]
