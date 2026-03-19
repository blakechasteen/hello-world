"""
Manifold Learning — Dimensionality Reduction for Visualization
==============================================================

Nonlinear dimensionality reduction methods that preserve geometric
structure of high-dimensional data on low-dimensional manifolds.

Methods:
- t-SNE: Stochastic Neighbor Embedding with Student-t kernel
- UMAP (simplified): Uniform Manifold Approximation and Projection
- Isomap: Isometric Mapping via geodesic distances
- LLE: Locally Linear Embedding via reconstruction weights

Mathematical Foundation:
All methods share the pattern:
1. Build a neighborhood/affinity graph in high-D
2. Define a cost function relating high-D and low-D affinities
3. Optimize low-D coordinates to minimize cost

t-SNE: min KL(P || Q) where P = Gaussian affinities, Q = Student-t affinities
UMAP: min CE(P, Q) where P = fuzzy simplicial set, Q = low-D membership
Isomap: geodesic MDS (preserve shortest-path distances)
LLE: preserve local reconstruction weights

Imports from randomized_linear_algebra.py: RandomizedSVD

Applications to Warp Space:
- Visualizing embedding clusters in DotPlasma feature space
- Exploring Thompson Sampling arm distributions
- Debugging convergence by projecting high-D state to 2D
- Interactive memory exploration in cognitive UI

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ManifoldEmbedding:
    """Result of manifold learning.

    Attributes:
        coordinates: Low-dimensional embedding, shape (n_samples, n_components).
        stress: Goodness-of-fit measure (lower is better). Definition varies
            by method.
        trustworthiness: Fraction of true nearest neighbors preserved in
            embedding, in [0, 1] (higher is better).
        method: Name of the embedding method used.
    """
    coordinates: np.ndarray
    stress: float
    trustworthiness: float
    method: str


# ============================================================================
# UTILITIES
# ============================================================================

def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix.

    Args:
        X: Data matrix, shape (n, d).

    Returns:
        Distance matrix, shape (n, n).
    """
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    sq_norms = np.sum(X**2, axis=1)
    D_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
    return np.sqrt(np.maximum(D_sq, 0.0))


def _knn_graph(D: np.ndarray, k: int) -> np.ndarray:
    """Build k-nearest-neighbor adjacency from distance matrix.

    Args:
        D: Pairwise distance matrix, shape (n, n).
        k: Number of neighbors.

    Returns:
        kNN indices, shape (n, k).
    """
    n = D.shape[0]
    k = min(k, n - 1)
    # For each point, find k nearest (excluding self)
    indices = np.zeros((n, k), dtype=int)
    for i in range(n):
        d = D[i].copy()
        d[i] = np.inf
        indices[i] = np.argpartition(d, k)[:k]
    return indices


def trustworthiness(X_high: np.ndarray, X_low: np.ndarray, k: int = 7) -> float:
    """Compute trustworthiness of a dimensionality reduction.

    Measures how well the low-dimensional embedding preserves local
    neighborhoods from the high-dimensional space. A value of 1.0
    means all k-nearest neighbors in high-D are also k-nearest in low-D.

    T(k) = 1 - (2 / (n * k * (2n - 3k - 1))) * sum_{violations} (r_low(i,j) - k)

    Args:
        X_high: Original high-dimensional data, shape (n, d_high).
        X_low: Low-dimensional embedding, shape (n, d_low).
        k: Number of neighbors to consider.

    Returns:
        Trustworthiness score in [0, 1].
    """
    n = X_high.shape[0]
    k = min(k, n - 2)
    if k < 1:
        return 1.0

    D_high = _pairwise_distances(X_high)
    D_low = _pairwise_distances(X_low)

    # k-NN in high-D
    knn_high = _knn_graph(D_high, k)

    # Ranks in low-D
    ranks_low = np.zeros_like(D_low, dtype=int)
    for i in range(n):
        order = np.argsort(D_low[i])
        for rank, j in enumerate(order):
            ranks_low[i, j] = rank

    # Sum of rank violations
    violation_sum = 0.0
    for i in range(n):
        high_nn_set = set(knn_high[i])
        for j in range(n):
            if j != i and j not in high_nn_set:
                # j is NOT a high-D neighbor but might be a low-D neighbor
                r = ranks_low[i, j]
                if r <= k:
                    violation_sum += r - k

    # Normalize
    denom = n * k * (2 * n - 3 * k - 1)
    if denom <= 0:
        return 1.0

    return float(1.0 - (2.0 / denom) * violation_sum)


# ============================================================================
# t-SNE
# ============================================================================

class TSNE:
    """t-Distributed Stochastic Neighbor Embedding.

    Converts high-dimensional pairwise affinities (Gaussian kernel) to
    low-dimensional affinities (Student-t kernel) and minimizes the
    KL divergence between them.

    The heavy-tailed Student-t kernel in low-D alleviates the "crowding
    problem" — moderate distances in high-D need to become large distances
    in low-D to avoid overlap.
    """

    @staticmethod
    def fit_transform(
        X: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        lr: float = 200.0,
        n_iter: int = 1000,
        momentum_start: float = 0.5,
        momentum_final: float = 0.8,
        momentum_switch: int = 250,
        seed: int = 42,
    ) -> ManifoldEmbedding:
        """Compute t-SNE embedding.

        Args:
            X: Input data, shape (n_samples, n_features).
            n_components: Output dimensionality (typically 2 or 3).
            perplexity: Effective number of neighbors (5-50 typical).
            lr: Learning rate for gradient descent.
            n_iter: Number of optimization iterations.
            momentum_start: Momentum for early iterations.
            momentum_final: Momentum for later iterations.
            momentum_switch: Iteration to switch momentum.
            seed: Random seed.

        Returns:
            ManifoldEmbedding with coordinates, stress, and trustworthiness.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        rng = np.random.default_rng(seed)

        # Step 1: Compute pairwise affinities P with perplexity
        D = _pairwise_distances(X)
        P = TSNE._compute_P(D, perplexity)

        # Symmetrize and normalize
        P = (P + P.T) / (2.0 * n)
        P = np.maximum(P, 1e-12)

        # Step 2: Initialize Y randomly
        Y = rng.normal(0, 1e-4, size=(n, n_components))
        Y_prev = Y.copy()
        gains = np.ones_like(Y)

        # Early exaggeration: multiply P by 4 for first 100 iterations
        P_exag = P * 4.0

        for iteration in range(n_iter):
            P_use = P_exag if iteration < 100 else P

            # Compute Q: Student-t affinities in low-D
            D_low_sq = np.sum((Y[:, None, :] - Y[None, :, :])**2, axis=2)
            Q_num = 1.0 / (1.0 + D_low_sq)
            np.fill_diagonal(Q_num, 0.0)
            Q_sum = Q_num.sum()
            Q = Q_num / max(Q_sum, 1e-12)
            Q = np.maximum(Q, 1e-12)

            # Gradient: dC/dY_i = 4 * sum_j (P_ij - Q_ij) * (Y_i - Y_j) * (1 + ||Y_i - Y_j||^2)^{-1}
            PQ_diff = P_use - Q
            grad = np.zeros_like(Y)
            for i in range(n):
                diff = Y[i] - Y  # (n, d)
                grad[i] = 4.0 * np.sum(
                    (PQ_diff[i, :] * Q_num[i, :])[:, None] * diff, axis=0
                )

            # Adaptive gains (Jacobs 1988)
            grad_sign = np.sign(grad)
            prev_sign = np.sign(Y - Y_prev)
            gains = np.where(
                grad_sign == prev_sign,
                gains * 0.8,
                gains + 0.2,
            )
            gains = np.maximum(gains, 0.01)

            momentum = momentum_start if iteration < momentum_switch else momentum_final

            Y_new = Y - lr * gains * grad + momentum * (Y - Y_prev)
            Y_prev = Y.copy()
            Y = Y_new

            # Center
            Y -= Y.mean(axis=0)

        # Compute final stress (KL divergence)
        D_low_sq = np.sum((Y[:, None, :] - Y[None, :, :])**2, axis=2)
        Q_num = 1.0 / (1.0 + D_low_sq)
        np.fill_diagonal(Q_num, 0.0)
        Q = Q_num / max(Q_num.sum(), 1e-12)
        Q = np.maximum(Q, 1e-12)
        kl = float(np.sum(P * np.log(P / Q)))

        trust = trustworthiness(X, Y, k=min(7, n - 2))

        return ManifoldEmbedding(
            coordinates=Y,
            stress=kl,
            trustworthiness=trust,
            method="t-SNE",
        )

    @staticmethod
    def _compute_P(D: np.ndarray, perplexity: float) -> np.ndarray:
        """Compute conditional probabilities with target perplexity.

        For each point i, find sigma_i such that the entropy of the
        conditional distribution P(j|i) = exp(-D_ij^2 / 2*sigma_i^2) / Z
        equals log2(perplexity).

        Uses binary search on sigma.
        """
        n = D.shape[0]
        P = np.zeros((n, n))
        target_entropy = np.log2(perplexity)

        for i in range(n):
            # Binary search for sigma
            lo, hi = 1e-5, 1e5
            d_i = D[i].copy()
            d_i[i] = np.inf

            for _ in range(50):
                sigma = (lo + hi) / 2.0
                p_i = np.exp(-d_i**2 / (2.0 * sigma**2))
                p_i[i] = 0.0
                p_sum = p_i.sum()

                if p_sum < 1e-15:
                    lo = sigma
                    continue

                p_i /= p_sum
                # Entropy in bits
                h = -np.sum(p_i * np.log2(np.maximum(p_i, 1e-15)))

                if h > target_entropy:
                    hi = sigma
                else:
                    lo = sigma

                if abs(h - target_entropy) < 1e-5:
                    break

            P[i] = p_i

        return P


# ============================================================================
# UMAP (Simplified)
# ============================================================================

class UMAP:
    """Simplified Uniform Manifold Approximation and Projection.

    Simplified version of the UMAP algorithm:
    1. Build kNN graph with fuzzy simplicial set membership
    2. Initialize with spectral embedding (or random)
    3. Optimize cross-entropy between high-D and low-D membership strengths
    """

    @staticmethod
    def fit_transform(
        X: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_epochs: int = 200,
        lr: float = 1.0,
        seed: int = 42,
    ) -> ManifoldEmbedding:
        """Compute simplified UMAP embedding.

        Args:
            X: Input data, shape (n_samples, n_features).
            n_components: Output dimensionality.
            n_neighbors: Number of nearest neighbors for graph construction.
            min_dist: Minimum distance in embedding (controls tightness).
            n_epochs: Number of optimization epochs.
            lr: Learning rate.
            seed: Random seed.

        Returns:
            ManifoldEmbedding with coordinates, stress, and trustworthiness.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        k = min(n_neighbors, n - 1)

        # Step 1: Build fuzzy simplicial set (symmetrized kNN graph)
        D = _pairwise_distances(X)
        knn_idx = _knn_graph(D, k)

        # Compute local connectivity: sigma_i such that sum of memberships = log2(k)
        W = np.zeros((n, n))
        for i in range(n):
            nn_dists = D[i, knn_idx[i]]
            rho = nn_dists.min()  # distance to nearest neighbor

            # Binary search for sigma
            target = np.log2(k)
            lo, hi = 1e-5, 100.0
            for _ in range(50):
                sigma = (lo + hi) / 2.0
                memberships = np.exp(-(np.maximum(nn_dists - rho, 0)) / sigma)
                s = memberships.sum()
                if s > target:
                    hi = sigma
                else:
                    lo = sigma
                if abs(s - target) < 1e-3:
                    break

            for j_idx, j in enumerate(knn_idx[i]):
                W[i, j] = np.exp(-(max(D[i, j] - rho, 0)) / sigma)

        # Symmetrize: W_sym = W + W^T - W * W^T
        W_sym = W + W.T - W * W.T
        W_sym = np.maximum(W_sym, 0)

        # Step 2: Spectral initialization via graph Laplacian
        try:
            degree = W_sym.sum(axis=1)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
            L_norm = np.eye(n) - D_inv_sqrt @ W_sym @ D_inv_sqrt
            eigvals, eigvecs = np.linalg.eigh(L_norm)
            # Skip first eigenvector (constant), take next n_components
            Y = eigvecs[:, 1:n_components + 1].copy()
            # Scale
            Y *= 10.0 / (Y.max() - Y.min() + 1e-10)
        except Exception:
            Y = rng.normal(0, 1, size=(n, n_components))

        # Step 3: Optimize cross-entropy
        # Find a, b parameters from min_dist
        # Simplified: use fixed a=1.929, b=0.7915 (default UMAP for min_dist=0.1)
        a = 1.929
        b = 0.7915
        if min_dist != 0.1:
            # Approximate: scale b inversely with min_dist
            b = 0.7915 * (0.1 / max(min_dist, 0.01))

        # Build edge list from W_sym
        edges_i, edges_j = np.where(W_sym > 0)
        edge_weights = W_sym[edges_i, edges_j]

        n_edges = len(edges_i)
        neg_sample_rate = 5

        for epoch in range(n_epochs):
            alpha = lr * (1.0 - epoch / n_epochs)

            # Positive edges
            for e_idx in range(n_edges):
                i, j = edges_i[e_idx], edges_j[e_idx]
                w = edge_weights[e_idx]

                diff = Y[i] - Y[j]
                dist_sq = np.sum(diff**2)
                dist_sq = max(dist_sq, 1e-6)

                # Gradient of attractive term
                grad_coeff = (-2.0 * a * b * dist_sq**(b - 1)) / (1.0 + a * dist_sq**b)
                grad = w * grad_coeff * diff

                Y[i] += alpha * grad
                Y[j] -= alpha * grad

                # Negative sampling
                for _ in range(neg_sample_rate):
                    k_neg = rng.integers(0, n)
                    if k_neg == i:
                        continue

                    diff_neg = Y[i] - Y[k_neg]
                    dist_sq_neg = np.sum(diff_neg**2)
                    dist_sq_neg = max(dist_sq_neg, 1e-6)

                    # Repulsive gradient
                    grad_rep = (2.0 * b) / ((0.001 + dist_sq_neg) * (1.0 + a * dist_sq_neg**b))
                    Y[i] += alpha * grad_rep * diff_neg

        # Compute stress as negative log-likelihood
        D_low = _pairwise_distances(Y)
        stress = 0.0
        for e_idx in range(n_edges):
            i, j = edges_i[e_idx], edges_j[e_idx]
            d = max(D_low[i, j], 1e-10)
            q = 1.0 / (1.0 + a * d**(2*b))
            stress -= edge_weights[e_idx] * np.log(max(q, 1e-15))

        trust = trustworthiness(X, Y, k=min(7, n - 2))

        return ManifoldEmbedding(
            coordinates=Y,
            stress=float(stress),
            trustworthiness=trust,
            method="UMAP",
        )


# ============================================================================
# ISOMAP
# ============================================================================

class Isomap:
    """Isometric Mapping via geodesic distances + MDS.

    1. Build kNN graph
    2. Compute geodesic distances via shortest paths (Floyd-Warshall)
    3. Apply classical MDS to geodesic distance matrix
    """

    @staticmethod
    def fit_transform(
        X: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 5,
    ) -> ManifoldEmbedding:
        """Compute Isomap embedding.

        Args:
            X: Input data, shape (n_samples, n_features).
            n_components: Output dimensionality.
            n_neighbors: Number of neighbors for graph construction.

        Returns:
            ManifoldEmbedding with coordinates, stress, and trustworthiness.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        k = min(n_neighbors, n - 1)

        # Step 1: kNN graph with Euclidean distances
        D = _pairwise_distances(X)
        knn_idx = _knn_graph(D, k)

        # Build adjacency: only keep kNN edges (symmetrized)
        G = np.full((n, n), np.inf)
        np.fill_diagonal(G, 0.0)
        for i in range(n):
            for j in knn_idx[i]:
                G[i, j] = D[i, j]
                G[j, i] = D[j, i]

        # Step 2: Shortest paths (Floyd-Warshall)
        geo = G.copy()
        for mid in range(n):
            for i in range(n):
                for j in range(n):
                    if geo[i, mid] + geo[mid, j] < geo[i, j]:
                        geo[i, j] = geo[i, mid] + geo[mid, j]

        # Handle disconnected components: replace inf with max finite distance * 2
        finite_mask = np.isfinite(geo)
        if not finite_mask.all():
            max_dist = geo[finite_mask].max() if finite_mask.any() else 1.0
            geo[~finite_mask] = max_dist * 2.0

        # Step 3: Classical MDS on geodesic distances
        Y, stress = Isomap._classical_mds(geo, n_components)

        trust = trustworthiness(X, Y, k=min(7, n - 2))

        return ManifoldEmbedding(
            coordinates=Y,
            stress=stress,
            trustworthiness=trust,
            method="Isomap",
        )

    @staticmethod
    def _classical_mds(D: np.ndarray, n_components: int) -> tuple[np.ndarray, float]:
        """Classical (metric) multidimensional scaling.

        Given distance matrix D, find coordinates Y such that
        ||Y_i - Y_j|| ≈ D_ij.

        Uses double centering: B = -0.5 * H * D^2 * H where H = I - 1/n * 11^T
        Then Y = top eigenvectors of B scaled by sqrt(eigenvalues).
        """
        n = D.shape[0]
        D_sq = D**2

        # Double centering
        row_mean = D_sq.mean(axis=1, keepdims=True)
        col_mean = D_sq.mean(axis=0, keepdims=True)
        grand_mean = D_sq.mean()
        B = -0.5 * (D_sq - row_mean - col_mean + grand_mean)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(B)

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take top n_components
        eigvals_k = eigvals[:n_components]
        eigvecs_k = eigvecs[:, :n_components]

        # Ensure non-negative eigenvalues
        eigvals_k = np.maximum(eigvals_k, 0.0)

        Y = eigvecs_k * np.sqrt(eigvals_k)[None, :]

        # Stress: normalized difference between input and output distances
        D_low = _pairwise_distances(Y)
        mask = np.triu_indices(n, k=1)
        residuals = D[mask] - D_low[mask]
        denom = np.sum(D[mask]**2)
        stress = float(np.sqrt(np.sum(residuals**2) / max(denom, 1e-15)))

        return Y, stress


# ============================================================================
# LOCALLY LINEAR EMBEDDING
# ============================================================================

class LocallyLinearEmbedding:
    """Locally Linear Embedding (LLE).

    1. Find k nearest neighbors for each point
    2. Compute reconstruction weights: x_i ≈ sum_j W_ij * x_j
    3. Find low-D coordinates that preserve these weights:
       min sum_i ||Y_i - sum_j W_ij Y_j||^2

    LLE is especially good at unfolding manifolds (e.g., Swiss roll).
    """

    @staticmethod
    def fit_transform(
        X: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 5,
        reg: float = 1e-3,
    ) -> ManifoldEmbedding:
        """Compute LLE embedding.

        Args:
            X: Input data, shape (n_samples, n_features).
            n_components: Output dimensionality.
            n_neighbors: Number of neighbors for weight computation.
            reg: Regularization parameter for weight computation.

        Returns:
            ManifoldEmbedding with coordinates, stress, and trustworthiness.
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        k = min(n_neighbors, n - 1)

        # Step 1: Find kNN
        D = _pairwise_distances(X)
        knn_idx = _knn_graph(D, k)

        # Step 2: Compute reconstruction weights
        W = np.zeros((n, n))
        for i in range(n):
            neighbors = knn_idx[i]
            # Z = X[neighbors] - X[i]  (center at x_i)
            Z = X[neighbors] - X[i]  # (k, d)

            # Local covariance: C = Z @ Z^T
            C = Z @ Z.T  # (k, k)

            # Regularize
            C += reg * np.eye(k) * np.trace(C)

            # Solve C w = 1 for weights
            try:
                w = np.linalg.solve(C, np.ones(k))
            except np.linalg.LinAlgError:
                w = np.ones(k)

            # Normalize
            w /= w.sum() + 1e-15

            W[i, neighbors] = w

        # Step 3: Eigendecomposition of (I - W)^T (I - W)
        M = np.eye(n) - W
        M = M.T @ M

        eigvals, eigvecs = np.linalg.eigh(M)

        # Skip first eigenvector (near-zero eigenvalue, constant vector)
        # Take next n_components eigenvectors
        Y = eigvecs[:, 1:n_components + 1]

        # Stress: reconstruction error in low-D
        residual = 0.0
        for i in range(n):
            reconstruction = W[i] @ Y
            residual += np.sum((Y[i] - reconstruction)**2)
        stress = float(residual / n)

        trust = trustworthiness(X, Y, k=min(7, n - 2))

        return ManifoldEmbedding(
            coordinates=Y,
            stress=stress,
            trustworthiness=trust,
            method="LLE",
        )


if __name__ == "__main__":
    print("Manifold Learning Module Demo")
    print("=" * 50)

    rng = np.random.default_rng(42)

    # Generate Swiss roll (classic manifold learning test)
    n = 200
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    x = t * np.cos(t)
    y = 21 * rng.random(n)
    z = t * np.sin(t)
    X = np.column_stack([x, y, z])

    print(f"Swiss roll: {X.shape}")

    # t-SNE
    result_tsne = TSNE.fit_transform(X, n_components=2, perplexity=20, n_iter=500)
    print(f"\nt-SNE: stress={result_tsne.stress:.4f}, trust={result_tsne.trustworthiness:.4f}")

    # Isomap
    result_iso = Isomap.fit_transform(X, n_components=2, n_neighbors=10)
    print(f"Isomap: stress={result_iso.stress:.4f}, trust={result_iso.trustworthiness:.4f}")

    # LLE
    result_lle = LocallyLinearEmbedding.fit_transform(X, n_components=2, n_neighbors=10)
    print(f"LLE: stress={result_lle.stress:.6f}, trust={result_lle.trustworthiness:.4f}")

    # UMAP (simplified)
    result_umap = UMAP.fit_transform(X, n_components=2, n_neighbors=15, n_epochs=100)
    print(f"UMAP: stress={result_umap.stress:.4f}, trust={result_umap.trustworthiness:.4f}")

    # Trustworthiness on simple data
    X_simple = rng.normal(0, 1, (50, 5))
    result_simple = TSNE.fit_transform(X_simple, n_components=2, n_iter=300)
    trust_val = trustworthiness(X_simple, result_simple.coordinates, k=5)
    print(f"\nGaussian blob trustworthiness: {trust_val:.4f}")

    print("\n" + "=" * 50)
    print("Manifold learning module ready!")
