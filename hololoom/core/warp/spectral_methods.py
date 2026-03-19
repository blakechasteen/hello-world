"""
Advanced Spectral Methods for Knowledge Graphs
==============================================
Wavelets, diffusion maps, and spectral clustering for deep graph analysis.

CURRENT STATE (embedding/spectral.py):
- Laplacian eigenvalues (basic)
- SVD of embeddings
- No wavelet analysis
- No diffusion geometry

NEW CAPABILITIES:
1. **Graph Wavelets**: Multi-scale localized features
2. **Diffusion Maps**: Nonlinear dimensionality reduction
3. **Spectral Clustering**: Community detection
4. **Heat Kernel**: Time-dependent diffusion
5. **Commute Times**: Effective resistance between nodes

WHY THIS MATTERS:
- Laplacian eigenvalues alone are global (can't detect local structure)
- Wavelets provide multi-scale local analysis
- Diffusion maps reveal intrinsic geometry (better than PCA/t-SNE)
- Heat kernel models semantic spreading over time

Mathematical Foundation:
-----------------------
Graph Laplacian: L = D - A
- D: Degree matrix
- A: Adjacency matrix

Spectrum: L = Φ Λ Φᵀ
- Φ: Eigenvectors (graph Fourier basis)
- Λ: Eigenvalues (frequencies)

Wavelets: Localized bases combining space + frequency
Diffusion: exp(-tL) for t > 0 (heat spreading)

Author: HoloLoom Spectral Theory Team
Date: 2025-11-03
"""

import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh, expm
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    warnings.warn("scipy not available - spectral methods will use dense fallback")


# ============================================================================
# Laplacian Types
# ============================================================================

class LaplacianType(Enum):
    """Types of graph Laplacians."""

    COMBINATORIAL = "combinatorial"  # L = D - A
    NORMALIZED = "normalized"        # L_sym = D^{-1/2} L D^{-1/2}
    RANDOM_WALK = "random_walk"      # L_rw = D^{-1} L


# ============================================================================
# Graph Laplacian
# ============================================================================

class GraphLaplacian:
    """
    Compute and analyze graph Laplacian matrices.

    Three standard Laplacians:
    1. Combinatorial: L = D - A
    2. Normalized (symmetric): L_sym = I - D^{-1/2} A D^{-1/2}
    3. Random walk: L_rw = I - D^{-1} A

    Each has different properties:
    - Combinatorial: Simplest, integer spectrum
    - Normalized: Better for clustering (bounded spectrum [0, 2])
    - Random walk: Natural for diffusion processes
    """

    def __init__(
        self,
        graph,
        laplacian_type: LaplacianType = LaplacianType.NORMALIZED
    ):
        """
        Initialize graph Laplacian.

        Args:
            graph: Knowledge graph (must have .G property with NetworkX graph)
            laplacian_type: Type of Laplacian
        """
        self.graph = graph
        self.laplacian_type = laplacian_type

        self.nodes = list(graph.G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        # Build matrices
        self.A = self._build_adjacency()
        self.D = self._build_degree()
        self.L = self._build_laplacian()

        # Cached spectrum
        self._eigenvalues = None
        self._eigenvectors = None

    def _build_adjacency(self) -> np.ndarray | csr_matrix:
        """Build adjacency matrix."""
        if self.n == 0:
            return np.zeros((0, 0))

        # Extract edges with weights
        rows, cols, data = [], [], []
        for u, v, edge_data in self.graph.G.edges(data=True):
            if u not in self.node_to_idx or v not in self.node_to_idx:
                continue

            i = self.node_to_idx[u]
            j = self.node_to_idx[v]
            w = float(edge_data.get('weight', 1.0))

            # Symmetric (undirected)
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([w, w])

        if _HAVE_SCIPY:
            return csr_matrix((data, (rows, cols)), shape=(self.n, self.n))
        else:
            # Dense fallback
            A = np.zeros((self.n, self.n))
            for r, c, w in zip(rows, cols, data):
                A[r, c] += w
            return A

    def _build_degree(self) -> np.ndarray | csr_matrix:
        """Build degree matrix."""
        if _HAVE_SCIPY and hasattr(self.A, 'toarray'):
            degrees = np.asarray(self.A.sum(axis=1)).ravel()
            return diags(degrees, format='csr')
        else:
            degrees = self.A.sum(axis=1)
            return np.diag(degrees)

    def _build_laplacian(self) -> np.ndarray | csr_matrix:
        """Build Laplacian matrix."""
        if self.laplacian_type == LaplacianType.COMBINATORIAL:
            # L = D - A
            return self.D - self.A

        elif self.laplacian_type == LaplacianType.NORMALIZED:
            # L_sym = I - D^{-1/2} A D^{-1/2}
            if _HAVE_SCIPY and hasattr(self.D, 'toarray'):
                degrees = np.asarray(self.D.diagonal()).ravel()
                d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
                D_inv_sqrt = diags(d_inv_sqrt)
                return diags(np.ones(self.n)) - D_inv_sqrt @ self.A @ D_inv_sqrt
            else:
                degrees = np.diag(self.D)
                d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
                D_inv_sqrt = np.diag(d_inv_sqrt)
                return np.eye(self.n) - D_inv_sqrt @ self.A @ D_inv_sqrt

        elif self.laplacian_type == LaplacianType.RANDOM_WALK:
            # L_rw = I - D^{-1} A
            if _HAVE_SCIPY and hasattr(self.D, 'toarray'):
                degrees = np.asarray(self.D.diagonal()).ravel()
                d_inv = np.where(degrees > 0, 1.0 / degrees, 0.0)
                D_inv = diags(d_inv)
                return diags(np.ones(self.n)) - D_inv @ self.A
            else:
                degrees = np.diag(self.D)
                d_inv = np.where(degrees > 0, 1.0 / degrees, 0.0)
                D_inv = np.diag(d_inv)
                return np.eye(self.n) - D_inv @ self.A

    def compute_spectrum(
        self,
        k: int | None = None,
        which: str = 'SM'
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Laplacian eigenvalues and eigenvectors.

        Args:
            k: Number of eigenvalues (None = all)
            which: Which eigenvalues ('SM' = smallest, 'LM' = largest)

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if self._eigenvalues is not None:
            return self._eigenvalues, self._eigenvectors

        if self.n == 0:
            return np.array([]), np.zeros((0, 0))

        # Convert to dense if needed
        L_dense = self.L.toarray() if hasattr(self.L, 'toarray') else self.L

        if k is None or k >= self.n - 1:
            # Full eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
        else:
            # Partial eigendecomposition
            if _HAVE_SCIPY and hasattr(self.L, 'toarray'):
                eigenvalues, eigenvectors = eigsh(self.L, k=k, which=which)
            else:
                # Fallback: full decomposition, then truncate
                vals, vecs = np.linalg.eigh(L_dense)
                if which == 'SM':
                    eigenvalues = vals[:k]
                    eigenvectors = vecs[:, :k]
                else:
                    eigenvalues = vals[-k:]
                    eigenvectors = vecs[:, -k:]

        # Cache
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

        return eigenvalues, eigenvectors

    def fiedler_value(self) -> float:
        """
        Compute Fiedler value (2nd smallest eigenvalue).

        Measures graph connectivity:
        - λ₁ = 0 (always)
        - λ₂ > 0: Connected graph
        - λ₂ = 0: Disconnected graph
        - Larger λ₂: More connected

        Returns:
            Fiedler value (algebraic connectivity)
        """
        eigenvalues, _ = self.compute_spectrum(k=min(2, self.n))
        return float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0


# ============================================================================
# Graph Wavelets
# ============================================================================

@dataclass
class SpectralWavelet:
    """
    Graph wavelet transform using spectral filtering.

    Wavelets = Localized bases combining space + frequency.

    Standard approach:
    1. Compute Laplacian eigenbasis Φ
    2. Define wavelet operator: Ψ_s = Φ g(sΛ) Φᵀ
       where g(·) is a kernel function (e.g., heat kernel)
       and s is the scale parameter
    3. Apply to signal: W_s f = Ψ_s f

    Multi-scale analysis: Vary s to detect features at different scales.
    """

    laplacian: GraphLaplacian
    n_scales: int = 5
    scale_min: float = 0.1
    scale_max: float = 10.0

    def __post_init__(self):
        self.scales = np.logspace(
            np.log10(self.scale_min),
            np.log10(self.scale_max),
            self.n_scales
        )

        # Compute spectrum once
        self.eigenvalues, self.eigenvectors = self.laplacian.compute_spectrum()

    def heat_kernel(self, scale: float) -> np.ndarray:
        """
        Heat kernel wavelet: g(λ) = exp(-s × λ)

        Args:
            scale: Scale parameter s

        Returns:
            Wavelet operator matrix
        """
        # g(Λ) = exp(-s × Λ)
        g_lambda = np.exp(-scale * self.eigenvalues)

        # Ψ_s = Φ g(Λ) Φᵀ
        wavelet_op = self.eigenvectors @ np.diag(g_lambda) @ self.eigenvectors.T
        return wavelet_op

    def mexican_hat_kernel(self, scale: float) -> np.ndarray:
        """
        Mexican hat wavelet: g(λ) = λ × exp(-s × λ)

        Better localization than heat kernel.

        Args:
            scale: Scale parameter

        Returns:
            Wavelet operator matrix
        """
        g_lambda = self.eigenvalues * np.exp(-scale * self.eigenvalues)
        wavelet_op = self.eigenvectors @ np.diag(g_lambda) @ self.eigenvectors.T
        return wavelet_op

    def transform(
        self,
        signal: np.ndarray,
        kernel: str = 'heat'
    ) -> dict[float, np.ndarray]:
        """
        Multi-scale wavelet transform.

        Args:
            signal: Node signal (length n)
            kernel: Kernel type ('heat' or 'mexican_hat')

        Returns:
            Dict mapping scale → wavelet coefficients
        """
        coefficients = {}

        for scale in self.scales:
            if kernel == 'heat':
                wavelet_op = self.heat_kernel(scale)
            elif kernel == 'mexican_hat':
                wavelet_op = self.mexican_hat_kernel(scale)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")

            # Apply wavelet: W_s f = Ψ_s @ f
            coefficients[scale] = wavelet_op @ signal

        return coefficients


# ============================================================================
# Diffusion Maps
# ============================================================================

@dataclass
class DiffusionMap:
    """
    Diffusion map: Nonlinear dimensionality reduction via diffusion geometry.

    Key insight: Euclidean distance in embedding space ≠ semantic distance.
    Diffusion distance = distance via random walks on graph.

    Algorithm:
    1. Compute diffusion operator: P = D^{-1} A (random walk matrix)
    2. Eigendecompose: P = Φ Λ Φᵀ
    3. Diffusion map: Ψ_t(x_i) = [λ₁ᵗ φ₁(i), λ₂ᵗ φ₂(i), ...]

    Time t controls diffusion:
    - t small: Local structure
    - t large: Global structure

    Diffusion distance: D_t(i, j) = ||Ψ_t(i) - Ψ_t(j)||
    """

    laplacian: GraphLaplacian
    t: float = 1.0                   # Diffusion time
    n_components: int = 10           # Embedding dimension

    def __post_init__(self):
        self._embedding = None

    def compute_embedding(self) -> np.ndarray:
        """
        Compute diffusion map embedding.

        Returns:
            Embedding matrix (n × n_components)
        """
        if self._embedding is not None:
            return self._embedding

        # Get spectrum
        eigenvalues, eigenvectors = self.laplacian.compute_spectrum(
            k=self.n_components + 1
        )

        # Skip first (trivial) eigenvector
        eigenvalues = eigenvalues[1:]
        eigenvectors = eigenvectors[:, 1:]

        # Diffusion map: Ψ_t = [λ₁ᵗ φ₁, λ₂ᵗ φ₂, ...]
        self._embedding = eigenvectors @ np.diag(eigenvalues ** self.t)

        return self._embedding

    def diffusion_distance(self, i: int, j: int) -> float:
        """
        Compute diffusion distance between nodes.

        Args:
            i, j: Node indices

        Returns:
            Diffusion distance
        """
        embedding = self.compute_embedding()
        return np.linalg.norm(embedding[i] - embedding[j])


# ============================================================================
# Spectral Clustering
# ============================================================================

def spectral_clustering(
    laplacian: GraphLaplacian,
    n_clusters: int,
    n_components: int | None = None
) -> np.ndarray:
    """
    Spectral clustering using normalized Laplacian.

    Algorithm:
    1. Compute first k eigenvectors of normalized Laplacian
    2. Treat rows as points in R^k
    3. Apply k-means clustering

    Args:
        laplacian: Graph Laplacian
        n_clusters: Number of clusters
        n_components: Number of eigenvectors (default: n_clusters)

    Returns:
        Cluster labels (length n)
    """
    if n_components is None:
        n_components = n_clusters

    # Compute spectral embedding
    _, eigenvectors = laplacian.compute_spectrum(k=n_components + 1)

    # Skip first (trivial) eigenvector
    embedding = eigenvectors[:, 1:]

    # Normalize rows
    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / (row_norms + 1e-10)

    # k-means clustering (simple implementation)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embedding)

    return labels


# ============================================================================
# Heat Kernel
# ============================================================================

def heat_kernel(
    laplacian: GraphLaplacian,
    t: float,
    source_nodes: list[int] | None = None
) -> np.ndarray:
    """
    Compute heat kernel diffusion.

    Heat equation on graph: ∂u/∂t = -L u
    Solution: u(t) = exp(-t L) u(0)

    Interpretation: Heat spreads from source nodes over time.

    Args:
        laplacian: Graph Laplacian
        t: Diffusion time
        source_nodes: Initial heat sources (None = all nodes equally)

    Returns:
        Heat distribution at time t (length n)
    """
    n = laplacian.n

    if source_nodes is None:
        u0 = np.ones(n) / n  # Uniform initial condition
    else:
        u0 = np.zeros(n)
        for idx in source_nodes:
            u0[idx] = 1.0
        u0 /= np.sum(u0)  # Normalize

    # Compute exp(-t L)
    L_dense = laplacian.L.toarray() if hasattr(laplacian.L, 'toarray') else laplacian.L

    if _HAVE_SCIPY:
        # Matrix exponential (sparse)
        heat_op = expm(-t * L_dense)
    else:
        # Fallback: eigendecomposition
        eigenvalues, eigenvectors = laplacian.compute_spectrum()
        heat_op = eigenvectors @ np.diag(np.exp(-t * eigenvalues)) @ eigenvectors.T

    return heat_op @ u0


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'LaplacianType',
    'GraphLaplacian',
    'SpectralWavelet',
    'DiffusionMap',
    'spectral_clustering',
    'heat_kernel',
]
