"""
Wasserstein Distances — Optimal Transport and Earth Mover's Distance
=====================================================================

Distances between probability distributions based on the cost of
transporting mass. Fundamental in distribution comparison,
generative modeling, and fairness metrics.

Core Concepts:
- Wasserstein Distance: Minimum-cost transport between distributions
- Sinkhorn Divergence: Entropy-regularized approximation (fast, differentiable)
- Wasserstein Barycenter: Frechet mean in Wasserstein space
- Sliced Wasserstein: Fast approximation via random 1D projections

Mathematical Foundation:
W_p(μ,ν) = inf_{γ∈Γ(μ,ν)} (∫ c(x,y)ᵖ dγ(x,y))^{1/p}
Sinkhorn: min_P ⟨C,P⟩ + ε H(P),  solved by alternating row/col scaling
Barycenter: argmin_ν Σ wᵢ W²(μᵢ, ν)
Sliced: SW(μ,ν) = E_θ[W₁(proj_θ μ, proj_θ ν)]

Applications to Warp Space:
- Comparing embedding distributions across weaving cycles
- Fairness metrics for Thompson Sampling exploration
- Distribution shift detection in memory systems

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TransportResult:
    """Result of optimal transport computation.

    Attributes:
        distance: Wasserstein distance value
        transport_plan: Optimal transport matrix (n, m)
        cost: Total transport cost (same as distance for W₁)
        n_iterations: Number of iterations (for iterative methods)
    """
    distance: float
    transport_plan: np.ndarray | None = None
    cost: float = 0.0
    n_iterations: int = 0


# ============================================================================
# WASSERSTEIN DISTANCE (Exact via LP)
# ============================================================================

class WassersteinDistance:
    """Exact Wasserstein distance via linear programming.

    Solves the optimal transport problem:
    min_P ⟨C, P⟩ subject to P1 = p, Pᵀ1 = q, P ≥ 0

    Example:
        >>> wd = WassersteinDistance()
        >>> p = np.array([0.5, 0.5])
        >>> q = np.array([0.3, 0.7])
        >>> C = np.array([[0, 1], [1, 0]])
        >>> result = wd.compute(p, q, C)
        >>> result.distance  # 0.2
    """

    def compute(
        self,
        p: np.ndarray,
        q: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> TransportResult:
        """Compute exact Wasserstein distance.

        Args:
            p: Source distribution (n,), must sum to 1
            q: Target distribution (m,), must sum to 1
            cost_matrix: Cost matrix C[i,j] (n, m)

        Returns:
            TransportResult with distance and transport plan
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        C = np.asarray(cost_matrix, dtype=float)

        n, m = C.shape

        # Normalize
        p = p / (p.sum() + 1e-15)
        q = q / (q.sum() + 1e-15)

        if HAS_SCIPY:
            return self._solve_lp(p, q, C)
        else:
            # Fall back to Sinkhorn with small regularization
            logger.debug("scipy unavailable, using Sinkhorn approximation for Wasserstein")
            sinkhorn = SinkhornDivergence()
            return sinkhorn.compute(p, q, C, epsilon=1e-3)

    def _solve_lp(
        self,
        p: np.ndarray,
        q: np.ndarray,
        C: np.ndarray,
    ) -> TransportResult:
        """Solve via linear programming (scipy)."""
        n, m = C.shape

        # Flatten cost matrix
        c = C.ravel()

        # Equality constraints: row sums = p, column sums = q
        # Row constraints: Σ_j P[i,j] = p[i] for each i
        A_eq_rows = np.zeros((n, n * m))
        for i in range(n):
            A_eq_rows[i, i * m:(i + 1) * m] = 1.0

        # Column constraints: Σ_i P[i,j] = q[j] for each j
        A_eq_cols = np.zeros((m, n * m))
        for j in range(m):
            for i in range(n):
                A_eq_cols[j, i * m + j] = 1.0

        A_eq = np.vstack([A_eq_rows, A_eq_cols])
        b_eq = np.concatenate([p, q])

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (n * m),
                         method='highs')

        if result.success:
            P = result.x.reshape(n, m)
            distance = result.fun
        else:
            logger.warning("LP solver failed, falling back to Sinkhorn")
            sinkhorn = SinkhornDivergence()
            return sinkhorn.compute(p, q, C, epsilon=1e-3)

        return TransportResult(
            distance=float(distance),
            transport_plan=P,
            cost=float(distance),
        )


# ============================================================================
# SINKHORN DIVERGENCE (Entropic Regularization)
# ============================================================================

class SinkhornDivergence:
    """Sinkhorn divergence: entropy-regularized optimal transport.

    Adds entropic penalty: min_P ⟨C, P⟩ + ε H(P)
    Solved efficiently by alternating Sinkhorn scaling iterations.

    Example:
        >>> sd = SinkhornDivergence()
        >>> p = np.array([0.4, 0.6])
        >>> q = np.array([0.5, 0.5])
        >>> C = np.array([[0, 1], [1, 0]])
        >>> result = sd.compute(p, q, C, epsilon=0.1)
    """

    def compute(
        self,
        p: np.ndarray,
        q: np.ndarray,
        cost_matrix: np.ndarray,
        epsilon: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> TransportResult:
        """Compute Sinkhorn divergence.

        Args:
            p: Source distribution (n,)
            q: Target distribution (m,)
            cost_matrix: Cost matrix (n, m)
            epsilon: Regularization strength (smaller = closer to exact)
            max_iter: Maximum Sinkhorn iterations
            tol: Convergence tolerance

        Returns:
            TransportResult with regularized distance
        """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        C = np.asarray(cost_matrix, dtype=float)

        p = p / (p.sum() + 1e-15)
        q = q / (q.sum() + 1e-15)

        n, m = C.shape

        # Gibbs kernel
        K = np.exp(-C / epsilon)

        # Sinkhorn iterations
        u = np.ones(n)
        v = np.ones(m)

        for iteration in range(max_iter):
            u_prev = u.copy()

            u = p / (K @ v + 1e-300)
            v = q / (K.T @ u + 1e-300)

            # Convergence check
            if np.max(np.abs(u - u_prev)) < tol:
                break

        # Transport plan
        P = np.diag(u) @ K @ np.diag(v)

        # Transport cost
        distance = float(np.sum(P * C))

        return TransportResult(
            distance=distance,
            transport_plan=P,
            cost=distance,
            n_iterations=iteration + 1,
        )


# ============================================================================
# WASSERSTEIN BARYCENTER
# ============================================================================

class WassersteinBarycenter:
    """Wasserstein barycenter: Frechet mean in Wasserstein space.

    Given distributions {μᵢ} with weights {wᵢ}, find
    ν* = argmin_ν Σᵢ wᵢ W²(μᵢ, ν)

    Solved via iterated Sinkhorn projections.

    Example:
        >>> wb = WassersteinBarycenter()
        >>> dists = [np.array([1, 0, 0]), np.array([0, 0, 1])]
        >>> weights = [0.5, 0.5]
        >>> C = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        >>> bary = wb.compute(dists, weights, C)
        >>> bary[1]  # Should have mass near center
    """

    def compute(
        self,
        distributions: list[np.ndarray],
        weights: list[float],
        cost_matrix: np.ndarray,
        epsilon: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Compute Wasserstein barycenter via Sinkhorn.

        Args:
            distributions: List of distributions, each (n,)
            weights: Barycenter weights (sum to 1)
            cost_matrix: Cost matrix (n, n), same support for all
            epsilon: Sinkhorn regularization
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Barycenter distribution (n,)
        """
        dists = [np.asarray(d, dtype=float) for d in distributions]
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        C = np.asarray(cost_matrix, dtype=float)

        n = len(dists[0])
        K_num = len(dists)

        # Gibbs kernel
        K = np.exp(-C / epsilon)

        # Initialize barycenter as uniform
        bary = np.ones(n) / n

        # Dual variables
        v_list = [np.ones(n) for _ in range(K_num)]

        for iteration in range(max_iter):
            bary_prev = bary.copy()

            # Update dual variables and barycenter
            log_bary = np.zeros(n)
            for k in range(K_num):
                u_k = dists[k] / (K @ v_list[k] + 1e-300)
                v_list[k] = bary / (K.T @ u_k + 1e-300)
                log_bary += weights[k] * np.log(K.T @ u_k + 1e-300)

            log_bary += np.sum(weights[:, None] * np.log(
                np.array(v_list) + 1e-300
            ), axis=0)

            bary = np.exp(log_bary)
            bary = bary / (bary.sum() + 1e-300)

            if np.max(np.abs(bary - bary_prev)) < tol:
                break

        return bary


# ============================================================================
# SLICED WASSERSTEIN DISTANCE
# ============================================================================

class SlicedWasserstein:
    """Sliced Wasserstein distance via random 1D projections.

    SW(μ, ν) = E_θ [W₁(proj_θ μ, proj_θ ν)]

    Much faster than exact Wasserstein in high dimensions.
    1D Wasserstein has closed form via sorted samples.

    Example:
        >>> sw = SlicedWasserstein()
        >>> X = np.random.randn(100, 5)
        >>> Y = np.random.randn(100, 5) + 1
        >>> dist = sw.compute(X, Y, n_projections=50)
        >>> dist > 0
        True
    """

    def compute(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_projections: int = 50,
    ) -> float:
        """Compute sliced Wasserstein distance.

        Args:
            X: First sample (n, d)
            Y: Second sample (m, d)
            n_projections: Number of random 1D projections

        Returns:
            Sliced Wasserstein distance (scalar)
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]

        d = X.shape[1]

        # Random projection directions on unit sphere
        directions = np.random.randn(n_projections, d)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        total = 0.0
        for theta in directions:
            # Project to 1D
            proj_x = X @ theta
            proj_y = Y @ theta

            # 1D Wasserstein = integral of |F_X⁻¹(t) - F_Y⁻¹(t)| dt
            # For empirical distributions: sort and compare
            total += self._wasserstein_1d(proj_x, proj_y)

        return total / n_projections

    @staticmethod
    def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
        """1D Wasserstein distance between empirical distributions.

        Uses the quantile function representation:
        W₁ = ∫₀¹ |F⁻¹_X(t) - F⁻¹_Y(t)| dt

        For equal-size samples: sort both and take mean absolute difference.
        For unequal sizes: interpolate quantiles.
        """
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)

        n, m = len(x_sorted), len(y_sorted)

        if n == m:
            return float(np.mean(np.abs(x_sorted - y_sorted)))

        # Interpolate to common grid
        grid = np.linspace(0, 1, max(n, m))
        x_quantiles = np.interp(grid, np.linspace(0, 1, n), x_sorted)
        y_quantiles = np.interp(grid, np.linspace(0, 1, m), y_sorted)

        return float(np.mean(np.abs(x_quantiles - y_quantiles)))
