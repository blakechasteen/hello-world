"""
Spectral Graph Theory - Laplacians and Clustering
==================================================

Mathematical foundations for graph-based learning and clustering.

Classes:
    GraphLaplacian: Compute various Laplacian matrices
    SpectralClustering: Cluster graphs via spectral embedding
    KnowledgeGraphClusterer: HoloLoom integration

Key Concepts:
    Laplacian L = D - A where D is degree matrix, A is adjacency

    Eigenvalues reveal graph structure:
        - λ₁ = 0 (always, for connected component)
        - λ₂ = "algebraic connectivity" (Fiedler value)
        - Multiplicity of 0 = number of connected components

    Fiedler vector (eigenvector of λ₂) gives optimal 2-way partition

Applications:
    - Knowledge graph topic clustering
    - Memory deduplication (cluster similar memories)
    - Community detection in entity graphs
    - Dimensionality reduction via spectral embedding

Author: Claude Code
Date: December 2025 (Math Module Expansion)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class LaplacianType(Enum):
    """Types of graph Laplacian matrices."""
    UNNORMALIZED = "unnormalized"     # L = D - A
    SYMMETRIC = "symmetric"            # L_sym = D^(-1/2) L D^(-1/2)
    RANDOM_WALK = "random_walk"        # L_rw = D^(-1) L


@dataclass
class SpectralEmbedding:
    """Result of spectral embedding."""
    coordinates: np.ndarray      # n_nodes x k embedding
    eigenvalues: np.ndarray      # k eigenvalues
    eigenvectors: np.ndarray     # n_nodes x k eigenvectors
    n_components: int            # Number of connected components

    @property
    def algebraic_connectivity(self) -> float:
        """Fiedler value (second smallest eigenvalue)."""
        if len(self.eigenvalues) > 1:
            return float(self.eigenvalues[1])
        return 0.0

    @property
    def fiedler_vector(self) -> np.ndarray:
        """Eigenvector corresponding to algebraic connectivity."""
        if self.eigenvectors.shape[1] > 1:
            return self.eigenvectors[:, 1]
        return self.eigenvectors[:, 0]


@dataclass
class ClusterResult:
    """Result of spectral clustering."""
    labels: np.ndarray           # Cluster assignment per node
    n_clusters: int              # Number of clusters found
    embedding: SpectralEmbedding # Underlying spectral embedding
    inertia: float               # k-means inertia
    silhouette: Optional[float] = None  # Silhouette score if computed


class GraphLaplacian:
    """
    Compute graph Laplacian matrices.

    The Laplacian encodes graph structure in a matrix form
    suitable for spectral analysis.

    Three variants:
        1. Unnormalized: L = D - A
        2. Symmetric normalized: L_sym = I - D^(-1/2) A D^(-1/2)
        3. Random walk normalized: L_rw = I - D^(-1) A

    All have the same eigenvalues for connected graphs, but
    different numerical properties.
    """

    @staticmethod
    def from_adjacency(
        A: np.ndarray,
        laplacian_type: LaplacianType = LaplacianType.SYMMETRIC
    ) -> np.ndarray:
        """
        Compute Laplacian from adjacency matrix.

        Args:
            A: Adjacency matrix (n x n), can be weighted
            laplacian_type: Type of normalization

        Returns:
            Laplacian matrix (n x n)
        """
        A = np.asarray(A, dtype=float)
        n = A.shape[0]

        # Degree matrix
        degrees = A.sum(axis=1)
        D = np.diag(degrees)

        if laplacian_type == LaplacianType.UNNORMALIZED:
            return D - A

        elif laplacian_type == LaplacianType.SYMMETRIC:
            # L_sym = I - D^(-1/2) A D^(-1/2)
            # Equivalent to D^(-1/2) L D^(-1/2)
            d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
            return np.eye(n) - d_inv_sqrt @ A @ d_inv_sqrt

        else:  # RANDOM_WALK
            # L_rw = I - D^(-1) A
            d_inv = np.diag(1.0 / np.maximum(degrees, 1e-10))
            return np.eye(n) - d_inv @ A

    @staticmethod
    def from_similarity(
        S: np.ndarray,
        laplacian_type: LaplacianType = LaplacianType.SYMMETRIC
    ) -> np.ndarray:
        """
        Compute Laplacian from similarity matrix.

        Similarity is treated as edge weights (higher = more connected).
        """
        # Zero diagonal (no self-loops)
        A = S.copy()
        np.fill_diagonal(A, 0)
        return GraphLaplacian.from_adjacency(A, laplacian_type)

    @staticmethod
    def from_distance(
        D: np.ndarray,
        sigma: float = 1.0,
        laplacian_type: LaplacianType = LaplacianType.SYMMETRIC
    ) -> np.ndarray:
        """
        Compute Laplacian from distance matrix using Gaussian kernel.

        Args:
            D: Distance matrix
            sigma: Bandwidth for Gaussian kernel
            laplacian_type: Type of normalization

        Returns:
            Laplacian matrix
        """
        # Gaussian kernel: W_ij = exp(-d_ij^2 / (2*sigma^2))
        W = np.exp(-D**2 / (2 * sigma**2))
        np.fill_diagonal(W, 0)  # No self-loops
        return GraphLaplacian.from_adjacency(W, laplacian_type)


def compute_laplacian(
    graph_data: Union[np.ndarray, Any],
    laplacian_type: LaplacianType = LaplacianType.SYMMETRIC,
    data_type: str = "adjacency"
) -> np.ndarray:
    """
    Convenience function to compute Laplacian.

    Args:
        graph_data: Adjacency, similarity, or distance matrix (or NetworkX graph)
        laplacian_type: Type of normalization
        data_type: "adjacency", "similarity", "distance", or "networkx"

    Returns:
        Laplacian matrix
    """
    if data_type == "networkx":
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for graph input")
        A = nx.to_numpy_array(graph_data)
        return GraphLaplacian.from_adjacency(A, laplacian_type)

    elif data_type == "similarity":
        return GraphLaplacian.from_similarity(graph_data, laplacian_type)

    elif data_type == "distance":
        return GraphLaplacian.from_distance(graph_data, laplacian_type=laplacian_type)

    else:  # adjacency
        return GraphLaplacian.from_adjacency(graph_data, laplacian_type)


def spectral_embed(
    L: np.ndarray,
    k: int = 8,
    return_eigenvalues: bool = True
) -> SpectralEmbedding:
    """
    Compute spectral embedding from Laplacian.

    Projects graph nodes into k-dimensional space using
    the k smallest eigenvectors of the Laplacian.

    Args:
        L: Laplacian matrix
        k: Embedding dimension (number of eigenvectors)
        return_eigenvalues: Whether to return eigenvalues

    Returns:
        SpectralEmbedding with coordinates and spectral info
    """
    n = L.shape[0]
    k = min(k, n)

    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Sort by eigenvalue (should already be sorted, but ensure)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Count connected components (number of zero eigenvalues)
    n_components = np.sum(eigenvalues < 1e-10)

    # Use first k eigenvectors as embedding
    # Skip first (constant) eigenvector for clustering
    if k > 1:
        coordinates = eigenvectors[:, 1:k]
        eigenvalues_used = eigenvalues[:k]
    else:
        coordinates = eigenvectors[:, :k]
        eigenvalues_used = eigenvalues[:k]

    return SpectralEmbedding(
        coordinates=coordinates,
        eigenvalues=eigenvalues_used,
        eigenvectors=eigenvectors[:, :k],
        n_components=n_components
    )


def fiedler_vector(L: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute Fiedler vector and algebraic connectivity.

    The Fiedler vector is the eigenvector corresponding to the
    second smallest eigenvalue (algebraic connectivity).

    It gives the optimal 2-way partition:
        - Positive entries → cluster 1
        - Negative entries → cluster 2

    Args:
        L: Laplacian matrix

    Returns:
        (fiedler_vector, algebraic_connectivity)
    """
    embedding = spectral_embed(L, k=2)
    return embedding.fiedler_vector, embedding.algebraic_connectivity


def algebraic_connectivity(L: np.ndarray) -> float:
    """
    Compute algebraic connectivity (second smallest eigenvalue).

    Measures how well-connected the graph is:
        - 0 = disconnected
        - Higher = more connected
        - For complete graph: n
    """
    _, ac = fiedler_vector(L)
    return ac


class SpectralClustering:
    """
    Spectral clustering algorithm.

    Steps:
        1. Compute graph Laplacian
        2. Compute spectral embedding (k smallest eigenvectors)
        3. Normalize rows (for symmetric Laplacian)
        4. Cluster in embedding space using k-means

    Properties:
        - Finds non-convex clusters
        - Works with any similarity/distance measure
        - Captures graph structure
    """

    def __init__(
        self,
        n_clusters: int = 2,
        laplacian_type: LaplacianType = LaplacianType.SYMMETRIC,
        n_components: Optional[int] = None,
        normalize_rows: bool = True
    ):
        """
        Args:
            n_clusters: Number of clusters
            laplacian_type: Type of Laplacian normalization
            n_components: Embedding dimension (default: n_clusters)
            normalize_rows: Normalize embedding rows before k-means
        """
        self.n_clusters = n_clusters
        self.laplacian_type = laplacian_type
        self.n_components = n_components or n_clusters
        self.normalize_rows = normalize_rows

    def fit(self, A: np.ndarray) -> ClusterResult:
        """
        Perform spectral clustering.

        Args:
            A: Adjacency or similarity matrix

        Returns:
            ClusterResult with labels and embedding
        """
        # Compute Laplacian
        L = GraphLaplacian.from_adjacency(A, self.laplacian_type)

        # Spectral embedding
        embedding = spectral_embed(L, k=self.n_components + 1)

        # Get coordinates (skip first constant eigenvector)
        coords = embedding.coordinates

        # Normalize rows
        if self.normalize_rows:
            norms = np.linalg.norm(coords, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            coords = coords / norms

        # K-means clustering
        labels, inertia = self._kmeans(coords, self.n_clusters)

        return ClusterResult(
            labels=labels,
            n_clusters=self.n_clusters,
            embedding=embedding,
            inertia=inertia
        )

    def fit_from_embeddings(
        self,
        embeddings: np.ndarray,
        sigma: float = 1.0
    ) -> ClusterResult:
        """
        Cluster from embedding vectors using RBF similarity.

        Args:
            embeddings: n_samples x d embedding vectors
            sigma: RBF bandwidth

        Returns:
            ClusterResult
        """
        # Compute pairwise distances
        diff = embeddings[:, np.newaxis] - embeddings[np.newaxis, :]
        D = np.sqrt(np.sum(diff**2, axis=2))

        # RBF kernel
        A = np.exp(-D**2 / (2 * sigma**2))
        np.fill_diagonal(A, 0)

        return self.fit(A)

    def _kmeans(
        self,
        X: np.ndarray,
        k: int,
        max_iters: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        Simple k-means implementation with k-means++ initialization.

        Args:
            X: Data points
            k: Number of clusters
            max_iters: Maximum iterations

        Returns:
            (labels, inertia)
        """
        n = X.shape[0]

        # K-means++ initialization: pick diverse centroids
        centroids = np.zeros((k, X.shape[1]))

        # First centroid: random
        first_idx = np.random.randint(n)
        centroids[0] = X[first_idx].copy()

        # Remaining centroids: weighted by distance squared
        for i in range(1, k):
            # Compute distance to nearest existing centroid
            distances = np.zeros(n)
            for j in range(n):
                min_dist = np.inf
                for c in range(i):
                    d = np.linalg.norm(X[j] - centroids[c])
                    if d < min_dist:
                        min_dist = d
                distances[j] = min_dist ** 2

            # Sample proportional to distance squared
            probs = distances / (distances.sum() + 1e-10)
            next_idx = np.random.choice(n, p=probs)
            centroids[i] = X[next_idx].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iters):
            # Assign to nearest centroid
            distances = np.zeros((n, k))
            for j in range(k):
                distances[:, j] = np.linalg.norm(X - centroids[j], axis=1)

            new_labels = np.argmin(distances, axis=1)

            # Check convergence
            if np.all(new_labels == labels):
                break

            labels = new_labels

            # Update centroids
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centroids[j] = X[mask].mean(axis=0)

        # Compute inertia
        inertia = 0.0
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                inertia += np.sum((X[mask] - centroids[j])**2)

        return labels, inertia


def cluster_graph(
    A: np.ndarray,
    n_clusters: int = 2,
    laplacian_type: LaplacianType = LaplacianType.SYMMETRIC
) -> ClusterResult:
    """
    Convenience function for spectral clustering.

    Args:
        A: Adjacency matrix
        n_clusters: Number of clusters
        laplacian_type: Laplacian normalization

    Returns:
        ClusterResult
    """
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        laplacian_type=laplacian_type
    )
    return clusterer.fit(A)


# HoloLoom Integration
class KnowledgeGraphClusterer:
    """
    Spectral clustering for HoloLoom knowledge graphs.

    Applications:
        - Topic clustering (find semantic groups)
        - Memory deduplication (cluster similar memories)
        - Community detection (find entity clusters)
    """

    def __init__(
        self,
        n_clusters: int = 5,
        laplacian_type: LaplacianType = LaplacianType.SYMMETRIC,
        min_cluster_size: int = 2
    ):
        """
        Args:
            n_clusters: Target number of clusters
            laplacian_type: Laplacian normalization
            min_cluster_size: Minimum cluster size
        """
        self.n_clusters = n_clusters
        self.laplacian_type = laplacian_type
        self.min_cluster_size = min_cluster_size
        self.clusterer = SpectralClustering(
            n_clusters=n_clusters,
            laplacian_type=laplacian_type
        )

    def cluster_from_networkx(self, G) -> Dict[str, Any]:
        """
        Cluster a NetworkX graph.

        Args:
            G: NetworkX graph (from hololoom.core.memory.graph.KG)

        Returns:
            Dict with clusters, node assignments, and metrics
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required")

        # Get nodes in stable order
        nodes = list(G.nodes())
        n = len(nodes)

        if n < self.n_clusters:
            # Fallback: each node is own cluster
            return {
                "clusters": {i: [node] for i, node in enumerate(nodes)},
                "node_to_cluster": {node: i for i, node in enumerate(nodes)},
                "n_clusters": n,
                "algebraic_connectivity": 0.0
            }

        # Build adjacency matrix
        A = nx.to_numpy_array(G, nodelist=nodes)

        # Handle weighted edges
        if not np.any(A):
            # Disconnected: each node is own cluster
            return {
                "clusters": {i: [node] for i, node in enumerate(nodes)},
                "node_to_cluster": {node: i for i, node in enumerate(nodes)},
                "n_clusters": n,
                "algebraic_connectivity": 0.0
            }

        # Cluster
        result = self.clusterer.fit(A)

        # Build cluster dict
        clusters = {}
        node_to_cluster = {}

        for i, label in enumerate(result.labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(nodes[i])
            node_to_cluster[nodes[i]] = label

        return {
            "clusters": clusters,
            "node_to_cluster": node_to_cluster,
            "n_clusters": result.n_clusters,
            "algebraic_connectivity": result.embedding.algebraic_connectivity,
            "inertia": result.inertia
        }

    def cluster_from_embeddings(
        self,
        node_ids: List[str],
        embeddings: np.ndarray,
        sigma: float = 1.0
    ) -> Dict[str, Any]:
        """
        Cluster nodes by embedding similarity.

        Args:
            node_ids: List of node identifiers
            embeddings: n_nodes x embedding_dim array
            sigma: RBF bandwidth

        Returns:
            Cluster assignments and metrics
        """
        result = self.clusterer.fit_from_embeddings(embeddings, sigma)

        clusters = {}
        node_to_cluster = {}

        for i, label in enumerate(result.labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node_ids[i])
            node_to_cluster[node_ids[i]] = label

        return {
            "clusters": clusters,
            "node_to_cluster": node_to_cluster,
            "n_clusters": result.n_clusters,
            "algebraic_connectivity": result.embedding.algebraic_connectivity,
            "inertia": result.inertia
        }

    def find_similar_clusters(
        self,
        cluster_result: Dict[str, Any],
        similarity_threshold: float = 0.8
    ) -> List[Tuple[int, int, float]]:
        """
        Find pairs of clusters that might be merged.

        Returns:
            List of (cluster1, cluster2, similarity) tuples
        """
        # This would require embeddings - placeholder for extension
        return []


if __name__ == "__main__":
    print("Spectral Clustering Module Demo")
    print("=" * 50)

    np.random.seed(42)

    # Create a graph with two communities
    # Community 1: nodes 0-4 (fully connected)
    # Community 2: nodes 5-9 (fully connected)
    # Weak connection between communities

    n = 10
    A = np.zeros((n, n))

    # Community 1
    for i in range(5):
        for j in range(i+1, 5):
            A[i, j] = A[j, i] = 1.0

    # Community 2
    for i in range(5, 10):
        for j in range(i+1, 10):
            A[i, j] = A[j, i] = 1.0

    # Weak bridge
    A[4, 5] = A[5, 4] = 0.1

    print("\nGraph with 2 communities (0-4, 5-9) and weak bridge:")

    # Compute Laplacian
    L = compute_laplacian(A, LaplacianType.SYMMETRIC)

    # Get Fiedler vector
    fv, ac = fiedler_vector(L)
    print(f"\nAlgebraic connectivity: {ac:.4f}")
    print(f"Fiedler vector: {fv.round(3)}")
    print(f"Partition: {['C1' if x > 0 else 'C2' for x in fv]}")

    # Spectral clustering
    result = cluster_graph(A, n_clusters=2)
    print(f"\nSpectral clustering labels: {result.labels}")
    print(f"Inertia: {result.inertia:.4f}")

    # NetworkX integration
    if HAS_NETWORKX:
        G = nx.Graph()
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] > 0:
                    G.add_edge(i, j, weight=A[i, j])

        clusterer = KnowledgeGraphClusterer(n_clusters=2)
        kg_result = clusterer.cluster_from_networkx(G)

        print(f"\nKnowledge graph clustering:")
        for cluster_id, members in kg_result["clusters"].items():
            print(f"  Cluster {cluster_id}: {members}")

    print("\n" + "=" * 50)
    print("Spectral clustering module ready!")
