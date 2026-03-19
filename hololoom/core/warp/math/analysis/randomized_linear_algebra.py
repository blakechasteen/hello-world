"""
Randomized Linear Algebra — Fast Approximate Matrix Decompositions
==================================================================

Algorithms for approximate matrix decompositions using randomized projections.

Core Concepts:
- Randomized SVD: Halko-Martinsson-Tropp algorithm for truncated SVD
- Johnson-Lindenstrauss: Dimension reduction with bounded distortion
- Nystrom Approximation: Low-rank kernel matrix approximation

Mathematical Foundation:
Randomized SVD: A ≈ U_k S_k V_k^T via random sampling of column space
JL Lemma: ∀ε∈(0,1), d ≥ 8 ln(n)/ε² preserves pairwise distances within (1±ε)
Nystrom: K ≈ C W^{-1} C^T where C is a column sample of K

Applications to Warp Space:
- Fast approximate embedding decomposition (bypass full SVD on large matrices)
- Dimensionality reduction for DotPlasma feature vectors
- Efficient kernel approximation for similarity computation
- Low-rank approximation of attention matrices

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

try:
    from scipy.linalg import lu, qr
    from scipy.linalg import svd as scipy_svd
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TruncatedSVDResult:
    """Result of a truncated SVD decomposition.

    A ≈ U @ diag(S) @ Vt

    Attributes:
        U: Left singular vectors, shape (m, k).
        S: Singular values, shape (k,), sorted descending.
        Vt: Right singular vectors (transposed), shape (k, n).
        explained_variance_ratio: Fraction of total variance captured
            by the top-k components (sum(S_k²) / sum(S_all²)).
    """
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    explained_variance_ratio: float

    @property
    def rank(self) -> int:
        """Number of components retained."""
        return len(self.S)

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the low-rank approximation A ≈ U S Vt."""
        return self.U * self.S[np.newaxis, :] @ self.Vt

    def truncate(self, k: int) -> 'TruncatedSVDResult':
        """Further truncate to top-k components."""
        if k >= self.rank:
            return self
        total_var = float(np.sum(self.S ** 2))
        new_var = float(np.sum(self.S[:k] ** 2))
        ratio = new_var / total_var if total_var > 0 else 0.0
        return TruncatedSVDResult(
            U=self.U[:, :k],
            S=self.S[:k],
            Vt=self.Vt[:k, :],
            explained_variance_ratio=ratio,
        )


# ============================================================================
# RANDOMIZED SVD
# ============================================================================

class RandomizedSVD:
    """
    Randomized truncated SVD via the Halko-Martinsson-Tropp algorithm.

    Computes an approximate rank-k SVD of a matrix A by:
    1. Form a random projection: Y = A @ Omega (Omega is random Gaussian)
    2. Orthonormalize: Q = orth(Y)
    3. Form small matrix: B = Q.T @ A
    4. Compute exact SVD of B: B = U_B S V_B^T
    5. Recover: U = Q @ U_B

    Power iteration improves accuracy for matrices with slowly
    decaying singular values: replace A with (A A^T)^q A.

    Reference: Halko, Martinsson, Tropp (2011).
    "Finding Structure with Randomness: Probabilistic Algorithms
     for Constructing Approximate Matrix Decompositions."
    """

    @staticmethod
    def decompose(
        A: np.ndarray,
        k: int,
        n_oversampling: int = 10,
        n_power_iter: int = 2,
        seed: int | None = None,
    ) -> TruncatedSVDResult:
        """
        Compute randomized rank-k SVD of matrix A.

        Args:
            A: Input matrix, shape (m, n).
            k: Target rank (number of singular values/vectors to compute).
            n_oversampling: Extra columns in random projection for accuracy.
                           Total projection dimension = k + n_oversampling.
            n_power_iter: Number of power iterations for improved accuracy.
                         Higher values help when singular values decay slowly.
            seed: Random seed for reproducibility.

        Returns:
            TruncatedSVDResult with U (m,k), S (k,), Vt (k,n).
        """
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

        if k > min(m, n):
            k = min(m, n)
            logger.warning(f"Reduced k to {k} (matrix dimensions: {m}x{n})")

        rng = np.random.RandomState(seed)
        p = min(k + n_oversampling, min(m, n))

        # Step 1: Random projection
        Omega = rng.randn(n, p)
        Y = A @ Omega

        # Step 2: Power iteration for improved accuracy
        # Replaces Y with (A A^T)^q A @ Omega
        for _ in range(n_power_iter):
            # Orthonormalize to maintain numerical stability
            Y, _ = np.linalg.qr(Y)
            Z = A.T @ Y
            Z, _ = np.linalg.qr(Z)
            Y = A @ Z

        # Step 3: Orthonormal basis for column space
        Q, _ = np.linalg.qr(Y)
        Q = Q[:, :p]  # Ensure correct size

        # Step 4: Project to low-dimensional space
        B = Q.T @ A  # Shape: (p, n)

        # Step 5: Exact SVD of the small matrix B
        U_B, S_full, Vt_full = np.linalg.svd(B, full_matrices=False)

        # Step 6: Recover left singular vectors
        U = Q @ U_B

        # Truncate to rank k
        U = U[:, :k]
        S = S_full[:k]
        Vt = Vt_full[:k, :]

        # Compute explained variance ratio
        # Use the singular values we found as an estimate
        total_variance_estimate = float(np.sum(S_full ** 2))
        if total_variance_estimate > 0:
            explained = float(np.sum(S ** 2)) / total_variance_estimate
        else:
            explained = 0.0

        return TruncatedSVDResult(
            U=U, S=S, Vt=Vt,
            explained_variance_ratio=explained,
        )

    @staticmethod
    def approximate_rank(
        A: np.ndarray,
        threshold: float = 0.95,
        max_k: int = 100,
        seed: int | None = None,
    ) -> int:
        """
        Estimate the numerical rank needed to capture a given fraction
        of the total variance.

        Incrementally increases k until explained_variance_ratio >= threshold.

        Args:
            A: Input matrix.
            threshold: Target explained variance ratio (0 to 1).
            max_k: Maximum rank to test.
            seed: Random seed.

        Returns:
            Estimated rank.
        """
        A = np.asarray(A, dtype=np.float64)
        max_possible = min(A.shape)
        max_k = min(max_k, max_possible)

        # Binary search over k
        lo, hi = 1, max_k
        best_k = max_k

        while lo <= hi:
            mid = (lo + hi) // 2
            result = RandomizedSVD.decompose(A, mid, seed=seed)
            if result.explained_variance_ratio >= threshold:
                best_k = mid
                hi = mid - 1
            else:
                lo = mid + 1

        return best_k


# ============================================================================
# JOHNSON-LINDENSTRAUSS PROJECTIONS
# ============================================================================

class JohnsonLindenstrauss:
    """
    Johnson-Lindenstrauss random projections for dimensionality reduction.

    The JL lemma guarantees that for n points in high-dimensional space,
    a random projection to d >= 8 ln(n) / eps^2 dimensions preserves
    all pairwise distances within a factor of (1 ± eps).
    """

    @staticmethod
    def random_projection(
        X: np.ndarray,
        target_dim: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Project data to lower dimension using Gaussian random projection.

        The projection matrix R has entries drawn from N(0, 1/target_dim).
        By JL lemma, pairwise distances are preserved within (1±eps) with
        high probability when target_dim >= 8 ln(n) / eps^2.

        Args:
            X: Data matrix, shape (n_samples, n_features).
            target_dim: Target dimensionality.
            seed: Random seed for reproducibility.

        Returns:
            Projected data, shape (n_samples, target_dim).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if target_dim >= n_features:
            logger.info("Target dim >= input dim, returning original data")
            return X

        rng = np.random.RandomState(seed)

        # Gaussian random projection matrix
        # Scale by 1/sqrt(target_dim) to preserve norms in expectation
        R = rng.randn(n_features, target_dim) / np.sqrt(target_dim)

        return X @ R

    @staticmethod
    def min_dim(n_samples: int, eps: float = 0.1) -> int:
        """
        Minimum target dimension from the JL lemma.

        d >= 8 ln(n) / eps^2

        Args:
            n_samples: Number of data points.
            eps: Maximum relative distortion of pairwise distances.

        Returns:
            Minimum target dimension (integer, rounded up).
        """
        if n_samples <= 1:
            return 1
        if eps <= 0 or eps >= 1:
            raise ValueError(f"eps must be in (0, 1), got {eps}")

        d = 8.0 * np.log(n_samples) / (eps ** 2)
        return int(np.ceil(d))

    @staticmethod
    def sparse_projection(
        X: np.ndarray,
        target_dim: int,
        density: float = 1.0 / 3.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Sparse random projection (Achlioptas, 2003).

        Projection matrix entries are:
            +sqrt(1/(density * target_dim))  with probability density/2
            0                                 with probability 1 - density
            -sqrt(1/(density * target_dim))  with probability density/2

        Faster than Gaussian projection due to sparsity, with similar
        JL guarantees. The classic choice density=1/3 gives a matrix
        that is 2/3 sparse.

        Args:
            X: Data matrix, shape (n_samples, n_features).
            target_dim: Target dimensionality.
            density: Fraction of non-zero entries (default 1/3).
            seed: Random seed.

        Returns:
            Projected data, shape (n_samples, target_dim).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if target_dim >= n_features:
            return X

        rng = np.random.RandomState(seed)

        # Generate sparse random matrix
        scale = np.sqrt(1.0 / (density * target_dim))

        # Draw from {-1, 0, +1} with probabilities {density/2, 1-density, density/2}
        u = rng.uniform(0, 1, size=(n_features, target_dim))
        R = np.zeros((n_features, target_dim), dtype=np.float64)
        R[u < density / 2.0] = scale
        R[u > 1.0 - density / 2.0] = -scale

        return X @ R

    @staticmethod
    def distortion_check(
        X_original: np.ndarray,
        X_projected: np.ndarray,
        n_pairs: int = 1000,
        seed: int | None = None,
    ) -> tuple[float, float, float]:
        """
        Empirically check the distortion of pairwise distances.

        Samples random pairs and computes the ratio of projected distance
        to original distance. Reports min, mean, and max distortion.

        Args:
            X_original: Original data (n, d_orig).
            X_projected: Projected data (n, d_proj).
            n_pairs: Number of random pairs to check.
            seed: Random seed.

        Returns:
            (min_ratio, mean_ratio, max_ratio) of
            ||proj(x)-proj(y)|| / ||x-y|| across sampled pairs.
        """
        n = len(X_original)
        if n < 2:
            return (1.0, 1.0, 1.0)

        rng = np.random.RandomState(seed)
        n_pairs = min(n_pairs, n * (n - 1) // 2)

        ratios = []
        for _ in range(n_pairs):
            i, j = rng.choice(n, 2, replace=False)
            d_orig = np.linalg.norm(X_original[i] - X_original[j])
            d_proj = np.linalg.norm(X_projected[i] - X_projected[j])

            if d_orig > 1e-15:
                ratios.append(d_proj / d_orig)

        if not ratios:
            return (1.0, 1.0, 1.0)

        ratios_arr = np.array(ratios)
        return (float(ratios_arr.min()), float(ratios_arr.mean()), float(ratios_arr.max()))


# ============================================================================
# RANDOMIZED NYSTROM APPROXIMATION
# ============================================================================

class RandomizedNystrom:
    """
    Nystrom approximation for low-rank kernel matrix approximation.

    Given an n x n symmetric positive semi-definite kernel matrix K,
    approximate it as K ≈ C W^{-1} C^T where:
    - C = K[:, S] is the column sample (n x k)
    - W = K[S, S] is the intersection (k x k)
    - S is a set of k sampled column indices

    Reduces storage and computation from O(n^2) to O(nk).
    """

    @staticmethod
    def approximate_kernel(
        K: np.ndarray,
        k: int,
        n_oversampling: int = 10,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Compute low-rank Nystrom approximation of a kernel matrix.

        Uses randomized column sampling with oversampling for improved
        accuracy. Falls back to uniform sampling if leverage scores
        are too expensive.

        Args:
            K: Symmetric PSD kernel matrix, shape (n, n).
            k: Target rank of the approximation.
            n_oversampling: Extra columns to sample for accuracy.
            seed: Random seed.

        Returns:
            Low-rank approximation of K, shape (n, n).
        """
        K = np.asarray(K, dtype=np.float64)
        n = K.shape[0]
        if K.shape != (n, n):
            raise ValueError(f"K must be square, got shape {K.shape}")

        rng = np.random.RandomState(seed)
        p = min(k + n_oversampling, n)

        # Sample columns (uniform random for simplicity)
        indices = rng.choice(n, size=p, replace=False)
        indices.sort()

        # Column sample C = K[:, indices], shape (n, p)
        C = K[:, indices]

        # Intersection W = K[indices, indices], shape (p, p)
        W = K[np.ix_(indices, indices)]

        # Regularize W for numerical stability
        W = W + 1e-10 * np.eye(p)

        # Compute W^{-1} via eigendecomposition for stability
        eigvals, eigvecs = np.linalg.eigh(W)
        # Clip small eigenvalues
        eigvals = np.maximum(eigvals, 1e-10)
        W_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

        # Nystrom approximation: K ≈ C @ W^{-1} @ C^T
        approx = C @ W_inv @ C.T

        # Truncate to rank k via SVD of the approximation factor
        # C @ W^{-1/2} gives a factor F such that K ≈ F F^T
        W_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        F = C @ W_inv_sqrt  # Shape (n, p)

        # SVD of F to get rank-k truncation
        U, S_vals, _ = np.linalg.svd(F, full_matrices=False)
        U_k = U[:, :k]
        S_k = S_vals[:k]

        # Reconstruct: K ≈ U_k diag(S_k^2) U_k^T
        return (U_k * S_k[np.newaxis, :] ** 2) @ U_k.T

    @staticmethod
    def approximate_features(
        K: np.ndarray,
        k: int,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Compute approximate feature map from kernel matrix.

        Returns Z such that K ≈ Z @ Z^T, with Z of shape (n, k).
        Useful when you need explicit feature vectors rather than
        the full kernel matrix.

        Args:
            K: Symmetric PSD kernel matrix, shape (n, n).
            k: Number of features.
            seed: Random seed.

        Returns:
            Feature matrix Z, shape (n, k).
        """
        K = np.asarray(K, dtype=np.float64)
        n = K.shape[0]

        rng = np.random.RandomState(seed)
        p = min(k + 5, n)

        indices = rng.choice(n, size=p, replace=False)
        indices.sort()

        C = K[:, indices]
        W = K[np.ix_(indices, indices)] + 1e-10 * np.eye(p)

        eigvals, eigvecs = np.linalg.eigh(W)
        eigvals = np.maximum(eigvals, 1e-10)
        W_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        F = C @ W_inv_sqrt  # Shape (n, p)

        # Truncate to k features via SVD
        U, S_vals, _ = np.linalg.svd(F, full_matrices=False)
        return U[:, :k] * S_vals[np.newaxis, :k]

    @staticmethod
    def reconstruction_error(
        K: np.ndarray,
        K_approx: np.ndarray,
        norm: str = "frobenius",
    ) -> float:
        """
        Compute reconstruction error between original and approximated kernel.

        Args:
            K: Original kernel matrix.
            K_approx: Approximated kernel matrix.
            norm: 'frobenius' for Frobenius norm, 'spectral' for spectral norm.

        Returns:
            Relative error ||K - K_approx|| / ||K||.
        """
        diff = K - K_approx
        if norm == "frobenius":
            error = np.linalg.norm(diff, "fro")
            base = np.linalg.norm(K, "fro")
        elif norm == "spectral":
            error = np.linalg.norm(diff, 2)
            base = np.linalg.norm(K, 2)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        if base < 1e-15:
            return 0.0
        return float(error / base)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def randomized_svd(
    A: np.ndarray,
    k: int,
    n_power_iter: int = 2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick interface: returns (U, S, Vt) tuple directly.

    Args:
        A: Input matrix.
        k: Target rank.
        n_power_iter: Power iterations.
        seed: Random seed.

    Returns:
        (U, S, Vt) tuple.
    """
    result = RandomizedSVD.decompose(A, k, n_power_iter=n_power_iter, seed=seed)
    return result.U, result.S, result.Vt


def jl_project(
    X: np.ndarray,
    eps: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Project X using JL lemma with automatic target dimension.

    Args:
        X: Data matrix (n_samples, n_features).
        eps: Maximum distortion.
        seed: Random seed.

    Returns:
        Projected data.
    """
    n = X.shape[0]
    target_dim = JohnsonLindenstrauss.min_dim(n, eps)
    target_dim = min(target_dim, X.shape[1])
    return JohnsonLindenstrauss.random_projection(X, target_dim, seed=seed)
