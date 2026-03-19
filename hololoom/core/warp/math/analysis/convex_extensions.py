"""
Convex Extensions — Fenchel Conjugate, Moreau Envelope, Proximal Operators
==========================================================================

Core tools from convex analysis and proximal optimization theory.
These are the building blocks for modern optimization algorithms
(ADMM, proximal gradient, etc.).

Core Concepts:
- Fenchel Conjugate: f*(y) = sup_x {<x,y> - f(x)}. Dual representation.
- Moreau Envelope: M_λf(x) = inf_z {f(z) + ||x-z||²/(2λ)}. Smooth approximation.
- Proximal Operator: prox_{λf}(x) = argmin_z {f(z) + ||x-z||²/(2λ)}. Resolvent.

Mathematical Foundation:
Biconjugate: f** = f iff f is closed convex proper (Fenchel-Moreau theorem)
Moreau decomposition: x = prox_{λf}(x) + λ prox_{f*/λ}(x/λ)
Proximal gradient: x_{k+1} = prox_{λg}(x_k - λ∇f(x_k))

Applications to Warp Space:
- Proximal operators for regularized embedding optimization
- Moreau envelope smoothing for non-smooth scoring functions
- Duality theory for constrained optimization in policy learning
- Proximal gradient methods for sparse feature selection

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ConjugateResult:
    """Result of Fenchel conjugate computation.

    Attributes:
        values: f*(y) evaluated at query points.
        maximizers: x* that achieves the supremum at each y.
        domain: The y values where f* was evaluated.
    """
    values: np.ndarray
    maximizers: np.ndarray
    domain: np.ndarray


@dataclass
class ProxResult:
    """Result of proximal operator computation.

    Attributes:
        point: The proximal point prox_{λf}(x).
        objective: f(prox) + ||prox - x||²/(2λ).
        iterations: Number of iterations used (for iterative methods).
    """
    point: np.ndarray
    objective: float
    iterations: int = 0


# ============================================================================
# FENCHEL CONJUGATE
# ============================================================================

class FenchelConjugate:
    """Fenchel-Legendre conjugate (convex conjugate).

    f*(y) = sup_x { <x, y> - f(x) }

    The conjugate transforms a function into its dual representation.
    For differentiable strictly convex f, the supremum is attained at
    x* where ∇f(x*) = y.

    For closed convex proper functions, f** = f (biconjugation theorem).

    Example:
        >>> fc = FenchelConjugate()
        >>> f = lambda x: 0.5 * np.sum(x**2)  # f(x) = ||x||²/2
        >>> # f*(y) = ||y||²/2 (self-conjugate)
        >>> result = fc.compute(f, np.linspace(-3, 3, 100))
    """

    def compute(
        self,
        f: Callable[[np.ndarray], float],
        domain: np.ndarray,
        search_range: tuple[float, float] | None = None,
        n_search: int = 1000,
    ) -> ConjugateResult:
        """Compute f*(y) by grid search over x.

        For scalar functions. Evaluates sup_x {xy - f(x)} over a
        fine grid of x values.

        Args:
            f: Convex function (scalar -> scalar via numpy).
            domain: Points y at which to evaluate f*.
            search_range: Range of x values to search over. Defaults to domain range.
            n_search: Number of grid points for x search.

        Returns:
            ConjugateResult with conjugate values.
        """
        domain = np.asarray(domain, dtype=float).ravel()

        if search_range is None:
            margin = max(np.abs(domain).max(), 1.0)
            search_range = (-3.0 * margin, 3.0 * margin)

        x_grid = np.linspace(search_range[0], search_range[1], n_search)
        f_values = np.array([f(np.array([xi])) for xi in x_grid])

        values = np.empty(len(domain))
        maximizers = np.empty(len(domain))

        for i, y in enumerate(domain):
            objective = x_grid * y - f_values
            idx = np.argmax(objective)
            values[i] = objective[idx]
            maximizers[i] = x_grid[idx]

        return ConjugateResult(values=values, maximizers=maximizers, domain=domain)

    def compute_nd(
        self,
        f: Callable[[np.ndarray], float],
        y_points: np.ndarray,
        search_samples: int = 5000,
        search_radius: float = 10.0,
    ) -> ConjugateResult:
        """Compute f*(y) for multi-dimensional functions via random search.

        Args:
            f: Convex function R^d -> R.
            y_points: Points at which to evaluate f* (m, d).
            search_samples: Number of random x samples.
            search_radius: Radius of search region.

        Returns:
            ConjugateResult with conjugate values.
        """
        y_points = np.atleast_2d(y_points)
        m, d = y_points.shape

        x_samples = np.random.randn(search_samples, d) * search_radius
        f_values = np.array([f(x) for x in x_samples])

        values = np.empty(m)
        maximizers = np.empty((m, d))

        for i in range(m):
            objective = x_samples @ y_points[i] - f_values
            idx = np.argmax(objective)
            values[i] = objective[idx]
            maximizers[i] = x_samples[idx]

        return ConjugateResult(values=values, maximizers=maximizers, domain=y_points)

    @staticmethod
    def verify_biconjugate(
        f: Callable[[np.ndarray], float],
        x_test: np.ndarray,
        search_range: tuple[float, float] = (-10.0, 10.0),
        n_search: int = 1000,
        tol: float = 0.1,
    ) -> tuple[bool, float]:
        """Verify f** ≈ f (Fenchel-Moreau theorem check).

        For closed convex proper functions, the biconjugate should
        equal the original function.

        Args:
            f: Function to test.
            x_test: Points at which to check f**(x) ≈ f(x).
            search_range: Search range for intermediate computations.
            n_search: Grid density.
            tol: Tolerance for equality check.

        Returns:
            (is_biconjugate, max_error).
        """
        fc = FenchelConjugate()
        x_test = np.asarray(x_test).ravel()

        # Compute f*
        y_grid = np.linspace(search_range[0], search_range[1], n_search)
        conj = fc.compute(f, y_grid, search_range, n_search)

        # Compute f** at test points
        f_star = lambda yi: np.interp(yi, conj.domain, conj.values)

        max_error = 0.0
        for x in x_test:
            f_x = f(np.array([x]))
            # f**(x) = sup_y {xy - f*(y)}
            objective = y_grid * x - np.array([f_star(y) for y in y_grid])
            f_star_star_x = np.max(objective)
            error = abs(f_star_star_x - f_x)
            max_error = max(max_error, error)

        return max_error < tol, float(max_error)


# ============================================================================
# MOREAU ENVELOPE
# ============================================================================

class MoreauEnvelope:
    """Moreau envelope (Moreau-Yosida regularization).

    M_λf(x) = inf_z { f(z) + ||x - z||² / (2λ) }

    The Moreau envelope is a smooth (C¹) approximation to f that:
    - Is always differentiable even when f is not
    - Has gradient: ∇M_λf(x) = (x - prox_λf(x)) / λ
    - Converges: M_λf(x) → f(x) as λ → 0⁺
    - Preserves minimizers: argmin f = argmin M_λf

    Example:
        >>> me = MoreauEnvelope()
        >>> f = lambda x: np.abs(x)  # Non-smooth at 0
        >>> me.compute(f, 0.0, lambda_param=1.0)  # Smooth version
    """

    def compute(
        self,
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        lambda_param: float = 1.0,
        n_search: int = 1000,
        search_radius: float | None = None,
    ) -> float:
        """Compute Moreau envelope value at x.

        Args:
            f: Function to envelope.
            x: Evaluation point (scalar or vector).
            lambda_param: Regularization parameter (> 0).
            n_search: Grid density for minimization.
            search_radius: Search radius around x.

        Returns:
            M_λf(x).
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        d = len(x)

        if search_radius is None:
            search_radius = 3.0 * np.sqrt(lambda_param)

        if d == 1:
            z_grid = np.linspace(x[0] - search_radius, x[0] + search_radius, n_search)
            objectives = np.array([
                f(np.array([z])) + np.sum((x - z) ** 2) / (2.0 * lambda_param)
                for z in z_grid
            ])
            return float(np.min(objectives))
        else:
            # Random search for higher dimensions
            z_samples = x + np.random.randn(n_search, d) * search_radius
            objectives = np.array([
                f(z) + np.sum((x - z) ** 2) / (2.0 * lambda_param)
                for z in z_samples
            ])
            return float(np.min(objectives))

    def compute_array(
        self,
        f: Callable[[np.ndarray], float],
        x_array: np.ndarray,
        lambda_param: float = 1.0,
        n_search: int = 500,
    ) -> np.ndarray:
        """Compute Moreau envelope at multiple points.

        Args:
            f: Function to envelope.
            x_array: Evaluation points (n,) or (n, d).
            lambda_param: Regularization parameter.
            n_search: Grid density per point.

        Returns:
            Envelope values (n,).
        """
        x_array = np.atleast_2d(x_array)
        if x_array.shape[0] == 1 and len(x_array.shape) == 2:
            x_array = x_array.T

        return np.array([
            self.compute(f, x, lambda_param, n_search)
            for x in x_array
        ])

    def gradient(
        self,
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        lambda_param: float = 1.0,
        n_search: int = 1000,
    ) -> np.ndarray:
        """Gradient of Moreau envelope: ∇M_λf(x) = (x - prox_λf(x)) / λ.

        Args:
            f: Function to envelope.
            x: Evaluation point.
            lambda_param: Regularization parameter.
            n_search: Grid density for proximal computation.

        Returns:
            Gradient vector.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        prox = ProximalOperator()
        prox_x = prox.compute(f, x, lambda_param, n_search=n_search).point
        return (x - prox_x) / lambda_param


# ============================================================================
# PROXIMAL OPERATOR
# ============================================================================

class ProximalOperator:
    """Proximal operator (proximity operator, resolvent).

    prox_{λf}(x) = argmin_z { f(z) + ||x - z||² / (2λ) }

    The proximal operator generalizes projection onto convex sets
    and is the key building block for proximal algorithms.

    Special cases:
    - prox_{λ||·||₁}(x) = soft_threshold(x, λ)   (L1 norm)
    - prox_{ι_C}(x) = proj_C(x)                   (indicator of convex set)
    - prox_{λ||·||₂}(x) = (1 - λ/||x||)₊ x       (L2 norm / group lasso)

    Example:
        >>> prox = ProximalOperator()
        >>> f = lambda x: np.sum(np.abs(x))  # L1 norm
        >>> result = prox.compute(f, np.array([3.0, -2.0]), lambda_param=1.0)
        >>> # Should be [2.0, -1.0] (soft thresholding)
    """

    def compute(
        self,
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        lambda_param: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-8,
        n_search: int = 1000,
    ) -> ProxResult:
        """Compute proximal operator via gradient descent on the Moreau envelope.

        For general f, uses numerical minimization. For known special
        cases, use the specific methods below.

        Args:
            f: Convex function.
            x: Input point.
            lambda_param: Step size parameter.
            max_iter: Maximum iterations.
            tol: Convergence tolerance.
            n_search: Grid density for 1D problems.

        Returns:
            ProxResult with proximal point.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        d = len(x)

        if d == 1:
            # Grid search for 1D
            search_r = max(3.0 * np.sqrt(lambda_param), np.abs(x[0]) + 5.0)
            z_grid = np.linspace(x[0] - search_r, x[0] + search_r, n_search)
            objectives = np.array([
                f(np.array([z])) + (x[0] - z) ** 2 / (2.0 * lambda_param)
                for z in z_grid
            ])
            idx = np.argmin(objectives)
            point = np.array([z_grid[idx]])
            return ProxResult(point=point, objective=float(objectives[idx]), iterations=0)

        # Gradient descent for higher dimensions
        z = x.copy()
        step_size = lambda_param / (1.0 + lambda_param)

        for it in range(max_iter):
            # Gradient of the Moreau objective: ∇f(z) + (z - x)/λ
            # Approximate ∇f by finite differences
            grad_f = self._numerical_gradient(f, z)
            grad = grad_f + (z - x) / lambda_param

            z_new = z - step_size * grad

            if np.linalg.norm(z_new - z) < tol:
                z = z_new
                obj = f(z) + np.sum((x - z) ** 2) / (2.0 * lambda_param)
                return ProxResult(point=z, objective=float(obj), iterations=it + 1)
            z = z_new

        obj = f(z) + np.sum((x - z) ** 2) / (2.0 * lambda_param)
        return ProxResult(point=z, objective=float(obj), iterations=max_iter)

    @staticmethod
    def soft_threshold(x: np.ndarray, lambda_param: float) -> np.ndarray:
        """Proximal operator for L1 norm: prox_{λ||·||₁}(x).

        Also known as soft thresholding or shrinkage operator.
        S_λ(x)_i = sign(x_i) * max(|x_i| - λ, 0)

        Args:
            x: Input vector.
            lambda_param: Threshold parameter.

        Returns:
            Soft-thresholded vector.
        """
        return np.sign(x) * np.maximum(np.abs(x) - lambda_param, 0.0)

    @staticmethod
    def group_threshold(x: np.ndarray, lambda_param: float) -> np.ndarray:
        """Proximal operator for L2 norm: prox_{λ||·||₂}(x).

        Block soft thresholding for group lasso:
        prox(x) = (1 - λ/||x||)₊ * x

        Args:
            x: Input vector.
            lambda_param: Threshold parameter.

        Returns:
            Group-thresholded vector.
        """
        norm = np.linalg.norm(x)
        if norm <= lambda_param:
            return np.zeros_like(x)
        return (1.0 - lambda_param / norm) * x

    @staticmethod
    def projection_simplex(x: np.ndarray) -> np.ndarray:
        """Proximal operator for indicator of probability simplex.

        Equivalent to Euclidean projection onto {z: z >= 0, sum(z) = 1}.

        Args:
            x: Input vector.

        Returns:
            Projection onto simplex.
        """
        n = len(x)
        u = np.sort(x)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho = np.max(np.where(u * np.arange(1, n + 1) > cssv)[0]) + 1
        theta = cssv[rho - 1] / rho
        return np.maximum(x - theta, 0.0)

    @staticmethod
    def projection_l2_ball(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """Project onto L2 ball of given radius.

        Args:
            x: Input vector.
            radius: Ball radius.

        Returns:
            Projection onto {z: ||z|| <= radius}.
        """
        norm = np.linalg.norm(x)
        if norm <= radius:
            return x.copy()
        return radius * x / norm

    @staticmethod
    def _numerical_gradient(f: Callable, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Compute gradient by central finite differences."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * eps)
        return grad
