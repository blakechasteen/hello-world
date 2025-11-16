"""
Cluster Engine - Intelligent Memory Organization
=================================================
Multi-algorithm clustering for organizing embeddings into semantic groups.

Philosophy:
Memory is not flat - similar concepts cluster together. The Cluster Engine
discovers these natural groupings, enabling efficient retrieval, topic discovery,
and adaptive resource allocation (e.g., BARE for dense clusters, FUSED for sparse).
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    logger.warning("hdbscan not installed. HDBSCAN clustering unavailable.")
    HDBSCAN_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. K-means and metrics unavailable.")
    SKLEARN_AVAILABLE = False
    KMeans = None
    SpectralClustering = None

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    logger.warning("umap-learn not installed. UMAP visualization unavailable.")
    UMAP_AVAILABLE = False


class ClusterAlgorithm(Enum):
    """Supported clustering algorithms."""
    HDBSCAN = "hdbscan"      # Density-based, handles noise
    KMEANS = "kmeans"        # Centroid-based, fast
    SPECTRAL = "spectral"    # Graph-based, complex shapes


@dataclass
class ClusterConfig:
    """Configuration for clustering."""
    algorithm: ClusterAlgorithm = ClusterAlgorithm.HDBSCAN

    # HDBSCAN params
    min_cluster_size: int = 5
    min_samples: int = 3

    # K-means params
    n_clusters: int = 10

    # Spectral params
    n_neighbors: int = 10

    # General
    metric: str = "euclidean"
    random_state: int = 42


@dataclass
class ClusterResult:
    """Result from clustering operation."""
    labels: np.ndarray  # Cluster assignments (-1 for noise)
    n_clusters: int  # Number of clusters found
    cluster_centers: Optional[np.ndarray] = None  # Cluster centroids
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    noise_count: int = 0
    algorithm: str = "unknown"


class ClusterEngine:
    """
    Clustering engine for embedding organization.

    Provides multiple clustering algorithms with quality metrics
    and automatic parameter tuning.

    Usage:
        engine = ClusterEngine(ClusterConfig(algorithm=ClusterAlgorithm.HDBSCAN))

        embeddings = np.array([...])  # N x D
        result = engine.cluster(embeddings)

        print(f"Found {result.n_clusters} clusters")
        print(f"Silhouette score: {result.quality_metrics['silhouette']:.3f}")

        # Get cluster assignments
        for idx, label in enumerate(result.labels):
            print(f"Item {idx} → Cluster {label}")
    """

    def __init__(self, config: ClusterConfig):
        """
        Initialize cluster engine.

        Args:
            config: Clustering configuration
        """
        self.config = config

        # Cache for clustering models
        self.model = None
        self.last_result: Optional[ClusterResult] = None

        logger.info(f"ClusterEngine initialized (algorithm={config.algorithm.value})")

    def cluster(self, embeddings: np.ndarray) -> ClusterResult:
        """
        Cluster embeddings.

        Args:
            embeddings: Array of shape (n_samples, n_features)

        Returns:
            ClusterResult with assignments and metrics
        """
        if len(embeddings) < self.config.min_cluster_size:
            logger.warning(f"Too few embeddings ({len(embeddings)}) for clustering")
            # Return single cluster
            return ClusterResult(
                labels=np.zeros(len(embeddings), dtype=int),
                n_clusters=1,
                algorithm="fallback",
                cluster_sizes={0: len(embeddings)}
            )

        # Route to appropriate algorithm
        if self.config.algorithm == ClusterAlgorithm.HDBSCAN:
            result = self._cluster_hdbscan(embeddings)
        elif self.config.algorithm == ClusterAlgorithm.KMEANS:
            result = self._cluster_kmeans(embeddings)
        elif self.config.algorithm == ClusterAlgorithm.SPECTRAL:
            result = self._cluster_spectral(embeddings)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        # Compute quality metrics
        if result.n_clusters > 1 and len(np.unique(result.labels)) > 1:
            result.quality_metrics = self._compute_metrics(embeddings, result.labels)

        # Cache result
        self.last_result = result

        logger.info(f"Clustering complete: {result.n_clusters} clusters, "
                   f"{result.noise_count} noise points")

        return result

    def _cluster_hdbscan(self, embeddings: np.ndarray) -> ClusterResult:
        """HDBSCAN clustering (density-based)."""
        if not HDBSCAN_AVAILABLE:
            logger.error("HDBSCAN not available, falling back to simple assignment")
            return ClusterResult(
                labels=np.zeros(len(embeddings), dtype=int),
                n_clusters=1,
                algorithm="fallback"
            )

        logger.debug("Running HDBSCAN clustering...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric=self.config.metric
        )

        labels = clusterer.fit_predict(embeddings)
        self.model = clusterer

        # Count clusters (excluding noise label -1)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        noise_count = np.sum(labels == -1)

        # Compute cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = int(np.sum(labels == label))

        # Compute centroids for valid clusters
        cluster_centers = []
        for label in range(n_clusters):
            mask = labels == label
            if np.any(mask):
                centroid = np.mean(embeddings[mask], axis=0)
                cluster_centers.append(centroid)

        cluster_centers = np.array(cluster_centers) if cluster_centers else None

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=cluster_centers,
            cluster_sizes=cluster_sizes,
            noise_count=noise_count,
            algorithm="hdbscan"
        )

    def _cluster_kmeans(self, embeddings: np.ndarray) -> ClusterResult:
        """K-means clustering (centroid-based)."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available")
            return ClusterResult(
                labels=np.zeros(len(embeddings), dtype=int),
                n_clusters=1,
                algorithm="fallback"
            )

        logger.debug("Running K-means clustering...")

        # Adjust n_clusters if needed
        n_clusters = min(self.config.n_clusters, len(embeddings))

        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init=10
        )

        labels = clusterer.fit_predict(embeddings)
        self.model = clusterer

        # Cluster sizes
        cluster_sizes = {}
        for label in range(n_clusters):
            cluster_sizes[label] = int(np.sum(labels == label))

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=clusterer.cluster_centers_,
            cluster_sizes=cluster_sizes,
            noise_count=0,
            algorithm="kmeans"
        )

    def _cluster_spectral(self, embeddings: np.ndarray) -> ClusterResult:
        """Spectral clustering (graph-based)."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available")
            return ClusterResult(
                labels=np.zeros(len(embeddings), dtype=int),
                n_clusters=1,
                algorithm="fallback"
            )

        logger.debug("Running Spectral clustering...")

        # Adjust n_clusters if needed
        n_clusters = min(self.config.n_clusters, len(embeddings))

        clusterer = SpectralClustering(
            n_clusters=n_clusters,
            n_neighbors=self.config.n_neighbors,
            random_state=self.config.random_state,
            affinity='nearest_neighbors'
        )

        labels = clusterer.fit_predict(embeddings)
        self.model = clusterer

        # Cluster sizes
        cluster_sizes = {}
        for label in range(n_clusters):
            cluster_sizes[label] = int(np.sum(labels == label))

        # Compute centroids
        cluster_centers = []
        for label in range(n_clusters):
            mask = labels == label
            if np.any(mask):
                centroid = np.mean(embeddings[mask], axis=0)
                cluster_centers.append(centroid)

        cluster_centers = np.array(cluster_centers) if cluster_centers else None

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=cluster_centers,
            cluster_sizes=cluster_sizes,
            noise_count=0,
            algorithm="spectral"
        )

    def _compute_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute cluster quality metrics."""
        if not SKLEARN_AVAILABLE:
            return {}

        metrics = {}

        try:
            # Filter out noise points for metrics
            mask = labels >= 0
            if np.sum(mask) < 2:
                return metrics

            filtered_embeddings = embeddings[mask]
            filtered_labels = labels[mask]

            # Silhouette score (higher is better, range [-1, 1])
            if len(np.unique(filtered_labels)) > 1:
                silhouette = silhouette_score(
                    filtered_embeddings,
                    filtered_labels,
                    metric=self.config.metric
                )
                metrics['silhouette'] = float(silhouette)

                # Davies-Bouldin index (lower is better)
                db_index = davies_bouldin_score(
                    filtered_embeddings,
                    filtered_labels
                )
                metrics['davies_bouldin'] = float(db_index)

        except Exception as e:
            logger.warning(f"Failed to compute metrics: {e}")

        return metrics

    def predict_cluster(self, embedding: np.ndarray) -> int:
        """
        Predict cluster for new embedding.

        Args:
            embedding: Single embedding vector

        Returns:
            Cluster label
        """
        if self.model is None:
            logger.warning("No model fitted yet")
            return -1

        # K-means has predict method
        if self.config.algorithm == ClusterAlgorithm.KMEANS and hasattr(self.model, 'predict'):
            label = self.model.predict([embedding])[0]
            return int(label)

        # For HDBSCAN/Spectral, find nearest cluster center
        if self.last_result and self.last_result.cluster_centers is not None:
            distances = np.linalg.norm(
                self.last_result.cluster_centers - embedding,
                axis=1
            )
            return int(np.argmin(distances))

        return -1

    def get_cluster_members(
        self,
        cluster_id: int,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Get indices of members in a cluster.

        Args:
            cluster_id: Cluster identifier
            embeddings: Original embeddings

        Returns:
            Array of indices
        """
        if self.last_result is None:
            logger.warning("No clustering result available")
            return np.array([])

        mask = self.last_result.labels == cluster_id
        return np.where(mask)[0]

    def get_cluster_summary(self, cluster_id: int) -> Dict[str, Any]:
        """
        Get summary statistics for a cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Dict with cluster statistics
        """
        if self.last_result is None:
            return {}

        size = self.last_result.cluster_sizes.get(cluster_id, 0)
        centroid = None

        if self.last_result.cluster_centers is not None and cluster_id >= 0:
            if cluster_id < len(self.last_result.cluster_centers):
                centroid = self.last_result.cluster_centers[cluster_id].tolist()

        return {
            'cluster_id': cluster_id,
            'size': size,
            'centroid': centroid,
            'is_noise': cluster_id == -1
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Cluster Engine Demo")
    print("="*80 + "\n")

    # Create synthetic data (3 clusters)
    np.random.seed(42)

    cluster1 = np.random.randn(50, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(50, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(50, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
    noise = np.random.randn(10, 10) * 5

    embeddings = np.vstack([cluster1, cluster2, cluster3, noise])

    print(f"Dataset: {len(embeddings)} embeddings, 10 dimensions\n")

    # Test each algorithm
    algorithms = [
        (ClusterAlgorithm.KMEANS, "K-Means (fast, needs k)"),
        (ClusterAlgorithm.HDBSCAN, "HDBSCAN (density, finds k)"),
    ]

    for algo, desc in algorithms:
        print(f"\n{desc}")
        print("-" * 40)

        try:
            config = ClusterConfig(
                algorithm=algo,
                n_clusters=3,
                min_cluster_size=5
            )

            engine = ClusterEngine(config)
            result = engine.cluster(embeddings)

            print(f"  Clusters found: {result.n_clusters}")
            print(f"  Noise points: {result.noise_count}")

            if result.quality_metrics:
                print(f"  Silhouette score: {result.quality_metrics.get('silhouette', 0):.3f}")
                print(f"  Davies-Bouldin index: {result.quality_metrics.get('davies_bouldin', 0):.3f}")

            print(f"  Cluster sizes: {result.cluster_sizes}")

            # Test prediction
            new_point = np.random.randn(10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            predicted_cluster = engine.predict_cluster(new_point)
            print(f"  New point predicted cluster: {predicted_cluster}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n✓ Demo complete!")
