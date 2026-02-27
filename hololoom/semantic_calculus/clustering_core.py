"""
Smart Clustering: One-line API for intelligent clustering.

WASM + EXO + HoloLoom Magic = Easy Clustering

Usage:
    from HoloLoom.clustering import cluster

    results = cluster(texts)
    # [
    #   {"label": "Machine Learning", "items": [...], "confidence": 0.92},
    #   {"label": "Web Development", "items": [...], "confidence": 0.87},
    # ]

What makes it smart:
- Auto-detects optimal number of clusters (no guessing k)
- Semantic labels (not "Cluster 0, 1, 2")
- Matryoshka multi-scale (fast → accurate)
- Thompson Sampling for k exploration
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import warnings

# High-performance Rust implementations (30-50x faster)
try:
    from hololoom_rust import (
        cosine_similarity_batch as _rust_cosine_similarity_batch,
        normalize_batch as _rust_normalize_batch,
        kmeans as _rust_kmeans,
        silhouette_score as _rust_silhouette_score,
    )
    _HAVE_RUST = True
except ImportError:
    _HAVE_RUST = False
    # Rust module not available - will fall back to sklearn

# Optional sklearn for clustering algorithms (fallback when Rust not available)
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score as _sklearn_silhouette_score
    from sklearn.metrics import calinski_harabasz_score
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False
    warnings.warn("sklearn not available. Install with: pip install scikit-learn")


def _silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score using Rust (fast) or sklearn (fallback)."""
    if _HAVE_RUST:
        # Rust expects float32
        embeddings_f32 = embeddings.astype(np.float32)
        labels_i64 = labels.astype(np.int64)
        return float(_rust_silhouette_score(embeddings_f32, labels_i64))
    elif _HAVE_SKLEARN:
        return float(_sklearn_silhouette_score(embeddings, labels))
    else:
        raise ImportError("Neither hololoom_rust nor sklearn available for silhouette score")


def _cosine_similarity_batch(vectors: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Compute cosine similarities using Rust (fast) or NumPy (fallback)."""
    if _HAVE_RUST:
        # Rust expects float32
        vectors_f32 = vectors.astype(np.float32)
        centroid_f32 = centroid.astype(np.float32)
        return _rust_cosine_similarity_batch(vectors_f32, centroid_f32)
    else:
        # NumPy fallback
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(centroid) + 1e-9
        return np.dot(vectors, centroid) / norms


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    label: str
    items: List[Any]
    indices: List[int]
    confidence: float
    centroid: Optional[np.ndarray] = None
    size: int = 0

    def __post_init__(self):
        self.size = len(self.items)


@dataclass
class ClusteringOutput:
    """Complete clustering output with metadata."""
    clusters: List[ClusterResult]
    n_clusters: int
    silhouette: float
    embeddings: Optional[np.ndarray] = None
    scale_used: int = 384
    method: str = "auto"

    def __iter__(self):
        return iter(self.clusters)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, idx):
        return self.clusters[idx]

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert to simple dict format."""
        return [
            {
                "label": c.label,
                "items": c.items,
                "confidence": c.confidence,
                "size": c.size
            }
            for c in self.clusters
        ]


def cluster(
    data: Union[List[str], List[Dict], np.ndarray],
    *,
    k: Optional[int] = None,
    min_k: int = 2,
    max_k: int = 10,
    method: str = "auto",
    scale: Optional[int] = None,
    label_clusters: bool = True,
    return_embeddings: bool = False,
    embedder: Optional[Any] = None,
    use_thompson: bool = True,
    thompson_budget: int = 15,
) -> ClusteringOutput:
    """
    Cluster data with automatic optimization.

    Args:
        data: Input data - texts, dicts with 'text' field, or pre-computed embeddings
        k: Number of clusters (None = auto-detect)
        min_k: Minimum clusters to try (default: 2)
        max_k: Maximum clusters to try (default: 10)
        method: Clustering method - "auto", "kmeans", "hierarchical"
        scale: Matryoshka scale - 96 (fast), 192 (balanced), 384 (accurate), None (auto)
        label_clusters: Generate semantic labels (default: True)
        return_embeddings: Include embeddings in output (default: False)
        embedder: Custom embedder (default: MatryoshkaEmbeddings)
        use_thompson: Use Thompson Sampling for k selection (default: True)
        thompson_budget: Max iterations for Thompson Sampling (default: 15)

    Returns:
        ClusteringOutput with labeled clusters

    Example:
        >>> results = cluster(["AI is cool", "Python rocks", "Neural networks learn"])
        >>> print(results[0].label)  # "Artificial Intelligence"
        >>> print(results[0].items)  # ["AI is cool", "Neural networks learn"]
    """
    if not _HAVE_SKLEARN:
        raise ImportError("sklearn required for clustering. Install with: pip install scikit-learn")

    # Extract texts and get embeddings
    texts, embeddings = _prepare_data(data, scale=scale, embedder=embedder)

    # Auto-select k if not specified
    if k is None:
        if use_thompson:
            k, silhouette = _find_optimal_k_thompson(
                embeddings, min_k=min_k, max_k=max_k, method=method, budget=thompson_budget
            )
        else:
            k, silhouette = _find_optimal_k(embeddings, min_k=min_k, max_k=max_k, method=method)
    else:
        silhouette = 0.0

    # Ensure k doesn't exceed data size
    k = min(k, len(texts))

    # Perform clustering
    labels, centroids = _do_clustering(embeddings, k=k, method=method)

    # Calculate silhouette if not already computed
    if silhouette == 0.0 and k > 1 and k < len(texts):
        silhouette = _silhouette_score(embeddings, labels)

    # Build cluster results
    clusters = _build_clusters(texts, embeddings, labels, centroids)

    # Generate semantic labels
    if label_clusters:
        clusters = _label_clusters(clusters, embeddings, texts)

    # Determine scale used
    scale_used = embeddings.shape[1] if embeddings is not None else 384

    return ClusteringOutput(
        clusters=clusters,
        n_clusters=k,
        silhouette=silhouette,
        embeddings=embeddings if return_embeddings else None,
        scale_used=scale_used,
        method=method
    )


def _prepare_data(
    data: Union[List[str], List[Dict], np.ndarray],
    scale: Optional[int] = None,
    embedder: Optional[Any] = None
) -> tuple:
    """Prepare data for clustering - extract texts and compute embeddings."""

    # Handle pre-computed embeddings
    if isinstance(data, np.ndarray):
        return [f"item_{i}" for i in range(len(data))], data

    # Extract texts from dicts if needed
    if data and isinstance(data[0], dict):
        texts = [d.get("text", d.get("content", str(d))) for d in data]
    else:
        texts = list(data)

    # Get embeddings
    if embedder is None:
        embedder = _get_default_embedder(scale)

    if scale is not None and hasattr(embedder, 'encode_at_scale'):
        embeddings = embedder.encode_at_scale(texts, scale)
    elif hasattr(embedder, 'encode'):
        embeddings = embedder.encode(texts)
    else:
        embeddings = embedder(texts)

    return texts, np.array(embeddings)


def _get_default_embedder(scale: Optional[int] = None):
    """Get default Matryoshka embedder."""
    try:
        from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
        sizes = [96, 192, 384, 768] if scale is None else [scale]
        return MatryoshkaEmbeddings(sizes=sizes)
    except ImportError:
        # Fallback to simple embedder
        return _FallbackEmbedder()


class _FallbackEmbedder:
    """Fallback embedder when MatryoshkaEmbeddings not available."""

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate deterministic embeddings from text hashes."""
        embeddings = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.normal(0, 1, 384)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            embeddings.append(vec)
        return np.array(embeddings)


def _find_optimal_k(
    embeddings: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    method: str = "auto"
) -> tuple:
    """
    Find optimal number of clusters using silhouette analysis + Thompson Sampling.

    Returns:
        (optimal_k, silhouette_score)
    """
    n_samples = len(embeddings)
    max_k = min(max_k, n_samples - 1)  # Can't have more clusters than samples

    if max_k < min_k:
        return min_k, 0.0

    # Try different k values and track scores
    scores = {}

    for k in range(min_k, max_k + 1):
        labels, _ = _do_clustering(embeddings, k=k, method=method)

        # Calculate silhouette score
        if len(set(labels)) > 1:  # Need at least 2 clusters
            score = _silhouette_score(embeddings, labels)
        else:
            score = -1.0

        scores[k] = score

    # Find k with best silhouette
    best_k = max(scores, key=scores.get)
    best_score = scores[best_k]

    return best_k, best_score


def _find_optimal_k_thompson(
    embeddings: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    method: str = "auto",
    budget: int = 15
) -> tuple:
    """
    Find optimal k using Thompson Sampling - smarter than grid search.

    Instead of trying every k, we use Bayesian exploration to quickly
    converge on the best k with fewer evaluations.

    Returns:
        (optimal_k, silhouette_score)
    """
    from .clustering_thompson import ThompsonClusterSampler

    n_samples = len(embeddings)
    max_k = min(max_k, n_samples - 1)

    if max_k < min_k:
        return min_k, 0.0

    sampler = ThompsonClusterSampler(min_k=min_k, max_k=max_k)

    # Track best seen
    best_k = min_k
    best_score = -1.0

    for _ in range(budget):
        # Sample next k to try
        k = sampler.sample_k()

        # Cluster and evaluate
        labels, _ = _do_clustering(embeddings, k=k, method=method)

        if len(set(labels)) > 1:
            score = _silhouette_score(embeddings, labels)
        else:
            score = -1.0

        # Update sampler
        sampler.update(k, score)

        # Track best
        if score > best_score:
            best_score = score
            best_k = k

        # Early stopping if we're confident
        if not sampler.should_explore_more(threshold=0.85):
            break

    return best_k, best_score


def _do_clustering(
    embeddings: np.ndarray,
    k: int,
    method: str = "auto"
) -> tuple:
    """
    Perform clustering with specified method.

    Uses Rust implementations when available (20-30x faster for k-means).

    Returns:
        (labels, centroids)
    """
    n_samples = len(embeddings)

    if method == "auto":
        # Use hierarchical for small datasets, kmeans for larger
        method = "hierarchical" if n_samples < 100 else "kmeans"

    if method == "kmeans":
        if _HAVE_RUST:
            # Use high-performance Rust implementation (20-30x faster)
            embeddings_f32 = embeddings.astype(np.float32)
            labels, centroids, inertia, n_iter = _rust_kmeans(
                embeddings_f32, k,
                max_iterations=300,
                tolerance=1e-4,
                n_init=10,
                seed=42
            )
            labels = np.array(labels)
        elif _HAVE_SKLEARN:
            # Fallback to sklearn
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(embeddings)
            centroids = model.cluster_centers_
        else:
            raise ImportError("Neither hololoom_rust nor sklearn available for k-means")

    elif method == "hierarchical":
        if not _HAVE_SKLEARN:
            raise ImportError("sklearn required for hierarchical clustering")
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(embeddings)
        # Compute centroids manually
        centroids = np.array([
            embeddings[labels == i].mean(axis=0)
            for i in range(k)
        ])

    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'kmeans', or 'hierarchical'")

    return labels, centroids


def _build_clusters(
    texts: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray
) -> List[ClusterResult]:
    """Build ClusterResult objects from clustering output."""

    clusters = []
    unique_labels = sorted(set(labels))

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        indices = np.where(mask)[0].tolist()
        items = [texts[i] for i in indices]

        # Calculate confidence based on cluster cohesion
        cluster_embeddings = embeddings[mask]
        centroid = centroids[cluster_id]

        # Cohesion = average cosine similarity to centroid
        # Uses Rust implementation when available (30-50x faster)
        similarities = _cosine_similarity_batch(cluster_embeddings, centroid)
        confidence = float(np.mean(similarities))

        clusters.append(ClusterResult(
            label=f"Cluster {cluster_id}",  # Will be replaced by semantic label
            items=items,
            indices=indices,
            confidence=confidence,
            centroid=centroid
        ))

    return clusters


def _label_clusters(
    clusters: List[ClusterResult],
    embeddings: np.ndarray,
    texts: List[str]
) -> List[ClusterResult]:
    """Generate semantic labels for clusters."""
    try:
        from .clustering_labeler import label_clusters as semantic_label
        return semantic_label(clusters, embeddings, texts)
    except ImportError:
        # Fallback: use most common words
        return _simple_label_clusters(clusters, texts)


def _simple_label_clusters(
    clusters: List[ClusterResult],
    texts: List[str]
) -> List[ClusterResult]:
    """Simple labeling fallback using common words."""
    import re
    from collections import Counter

    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below',
                 'between', 'under', 'again', 'further', 'then', 'once',
                 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                 'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                 'very', 'just', 'also', 'now', 'here', 'there', 'when',
                 'where', 'why', 'how', 'all', 'each', 'every', 'both',
                 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                 'any', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
                 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she',
                 'her', 'it', 'its', 'they', 'them', 'their', 'what',
                 'which', 'who', 'whom', 'whose'}

    for cluster in clusters:
        # Get all words from cluster items
        all_text = ' '.join(cluster.items).lower()
        words = re.findall(r'\b[a-z]{3,}\b', all_text)

        # Filter stopwords and count
        filtered = [w for w in words if w not in stopwords]
        counter = Counter(filtered)

        # Get top 2-3 words as label
        top_words = [word for word, _ in counter.most_common(3)]
        if top_words:
            cluster.label = ' '.join(word.title() for word in top_words[:2])
        else:
            cluster.label = f"Group {clusters.index(cluster) + 1}"

    return clusters


# Convenience aliases
smart_cluster = cluster
auto_cluster = cluster
