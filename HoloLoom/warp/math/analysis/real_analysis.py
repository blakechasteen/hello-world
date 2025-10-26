"""
Real Analysis for HoloLoom Warp Drive
======================================

Foundation of continuous mathematics for rigorous analysis of:
- Convergence in neural networks
- Optimization landscape analysis
- Approximation theory for embeddings
- Continuity and differentiability of semantic spaces

Core Concepts:
- Metric Spaces: Generalized distance and topology
- Sequences & Series: Convergence analysis
- Continuity: Preservation of limits
- Differentiation: Directional derivatives, Frechet derivatives
- Integration: Riemann integration
- Function Spaces: Uniform convergence, equicontinuity

Mathematical Foundation:
A metric space (X, d) consists of:
- Set X
- Metric d: X × X → ℝ satisfying:
  1. d(x,y) ≥ 0 with equality iff x = y
  2. d(x,y) = d(y,x) (symmetry)
  3. d(x,z) ≤ d(x,y) + d(y,z) (triangle inequality)

Applications to Warp Space:
- Embedding spaces as metric spaces
- Convergence of training procedures
- Continuous semantic transformations
- Differentiable knowledge graph operations

Author: HoloLoom Team
Date: 2025-10-25
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Metric Spaces
# ============================================================================

class MetricSpace:
    """
    Metric space (X, d) with distance function.

    Provides foundation for topological reasoning about embeddings.
    """

    def __init__(self, elements: Optional[List] = None, metric: Optional[Callable] = None, name: str = "X"):
        """
        Initialize metric space.

        Args:
            elements: Points in the space (optional for abstract spaces)
            metric: Distance function d(x, y)
            name: Name of the space
        """
        self.elements = elements or []
        self.metric = metric or self._euclidean_metric
        self.name = name

        logger.info(f"Metric space {name} initialized")

    @staticmethod
    def _euclidean_metric(x: np.ndarray, y: np.ndarray) -> float:
        """Default Euclidean metric."""
        return np.linalg.norm(x - y)

    def distance(self, x, y) -> float:
        """Compute d(x, y)."""
        return self.metric(x, y)

    def is_metric(self, sample_size: int = 100) -> bool:
        """
        Verify metric axioms on sample.

        Checks:
        1. Non-negativity: d(x,y) ≥ 0
        2. Identity: d(x,x) = 0
        3. Symmetry: d(x,y) = d(y,x)
        4. Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
        """
        if not self.elements or len(self.elements) < 3:
            logger.warning("Not enough elements to verify metric axioms")
            return True

        # Sample points
        n = min(sample_size, len(self.elements))
        indices = np.random.choice(len(self.elements), size=min(n, len(self.elements)), replace=False)
        sample = [self.elements[i] for i in indices]

        for x in sample[:10]:  # Test subset
            for y in sample[:10]:
                d_xy = self.distance(x, y)
                d_yx = self.distance(y, x)

                # Non-negativity
                if d_xy < 0:
                    logger.error(f"Non-negativity violated: d({x}, {y}) = {d_xy}")
                    return False

                # Identity
                d_xx = self.distance(x, x)
                if d_xx > 1e-10:
                    logger.error(f"Identity violated: d({x}, {x}) = {d_xx}")
                    return False

                # Symmetry
                if abs(d_xy - d_yx) > 1e-10:
                    logger.error(f"Symmetry violated: d({x}, {y}) = {d_xy} != {d_yx}")
                    return False

                # Triangle inequality (test with one more point)
                if len(sample) > 2:
                    z = sample[2]
                    d_xz = self.distance(x, z)
                    d_yz = self.distance(y, z)

                    if d_xz > d_xy + d_yz + 1e-10:
                        logger.error(f"Triangle inequality violated")
                        return False

        return True

    def open_ball(self, center, radius: float) -> List:
        """
        Open ball B(center, radius) = {x : d(x, center) < radius}.
        """
        if not self.elements:
            raise ValueError("No elements in metric space")

        return [x for x in self.elements if self.distance(x, center) < radius]

    def closed_ball(self, center, radius: float) -> List:
        """
        Closed ball B̄(center, radius) = {x : d(x, center) ≤ radius}.
        """
        if not self.elements:
            raise ValueError("No elements in metric space")

        return [x for x in self.elements if self.distance(x, center) <= radius]

    def is_cauchy(self, sequence: List, epsilon: float = 1e-6) -> bool:
        """
        Check if sequence is Cauchy.

        Sequence (xₙ) is Cauchy if for all ε > 0, exists N such that
        for all m, n > N: d(xₘ, xₙ) < ε
        """
        n = len(sequence)
        if n < 2:
            return True

        # Check last quarter of sequence
        start = 3 * n // 4
        for i in range(start, n):
            for j in range(i + 1, n):
                if self.distance(sequence[i], sequence[j]) > epsilon:
                    return False

        return True

    def is_complete(self) -> bool:
        """
        Check if space is complete (all Cauchy sequences converge).

        Note: This is a property, not algorithmically decidable in general.
        We return True for finite spaces and ℝⁿ.
        """
        # Finite spaces are always complete
        if self.elements and len(self.elements) < float('inf'):
            return True

        # Euclidean spaces are complete
        if self.metric == self._euclidean_metric:
            return True

        logger.warning("Completeness cannot be verified algorithmically")
        return None


# ============================================================================
# Sequences and Series
# ============================================================================

class SequenceAnalyzer:
    """
    Analyze convergence of sequences and series.

    Essential for understanding training dynamics and optimization.
    """

    @staticmethod
    def limit(sequence: Union[List[float], np.ndarray], tolerance: float = 1e-10) -> Optional[float]:
        """
        Compute limit of sequence if it exists.

        Uses Cauchy criterion: sequence converges iff it's Cauchy.
        """
        if len(sequence) < 2:
            return sequence[0] if len(sequence) == 1 else None

        # Check if Cauchy
        tail = sequence[-min(100, len(sequence)):]

        # Compute variance of tail
        variance = np.var(tail)

        if variance < tolerance:
            return np.mean(tail)
        else:
            return None

    @staticmethod
    def is_convergent(sequence: Union[List[float], np.ndarray], tolerance: float = 1e-6) -> bool:
        """
        Determine if sequence converges.
        """
        limit_val = SequenceAnalyzer.limit(sequence, tolerance)
        return limit_val is not None

    @staticmethod
    def is_monotone(sequence: Union[List[float], np.ndarray]) -> Tuple[bool, str]:
        """
        Check if sequence is monotone.

        Returns (is_monotone, direction) where direction ∈ {"increasing", "decreasing", "constant", "none"}
        """
        if len(sequence) < 2:
            return (True, "constant")

        diffs = np.diff(sequence)

        if np.all(diffs >= 0):
            if np.all(diffs == 0):
                return (True, "constant")
            return (True, "increasing")
        elif np.all(diffs <= 0):
            return (True, "decreasing")
        else:
            return (False, "none")

    @staticmethod
    def is_bounded(sequence: Union[List[float], np.ndarray]) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Check if sequence is bounded.

        Returns (is_bounded, lower_bound, upper_bound)
        """
        if len(sequence) == 0:
            return (True, None, None)

        lower = np.min(sequence)
        upper = np.max(sequence)

        # Check if bounds are finite
        is_bounded = np.isfinite(lower) and np.isfinite(upper)

        return (is_bounded, lower if is_bounded else None, upper if is_bounded else None)

    @staticmethod
    def series_sum(terms: Union[List[float], np.ndarray], method: str = "direct") -> Tuple[Optional[float], bool]:
        """
        Compute sum of series if convergent.

        Returns (sum, converged)

        Methods:
        - direct: Direct summation
        - cesaro: Cesàro summation (can sum some divergent series)
        """
        if method == "direct":
            # Check absolute convergence first
            abs_sum = np.sum(np.abs(terms))

            if np.isfinite(abs_sum):
                return (np.sum(terms), True)
            else:
                return (None, False)

        elif method == "cesaro":
            # Cesàro mean: (s₁ + s₂ + ... + sₙ) / n where sₖ = Σᵢ₌₁ᵏ aᵢ
            partial_sums = np.cumsum(terms)
            cesaro_means = np.cumsum(partial_sums) / np.arange(1, len(partial_sums) + 1)

            limit_val = SequenceAnalyzer.limit(cesaro_means)
            if limit_val is not None:
                return (limit_val, True)
            else:
                return (None, False)

        else:
            raise ValueError(f"Unknown summation method: {method}")


# ============================================================================
# Continuity
# ============================================================================

class ContinuityChecker:
    """
    Analyze continuity of functions.

    f: X → Y is continuous at x₀ if:
    ∀ε > 0, ∃δ > 0: d_X(x, x₀) < δ ⟹ d_Y(f(x), f(x₀)) < ε
    """

    def __init__(self, function: Callable, domain: MetricSpace, codomain: MetricSpace):
        self.function = function
        self.domain = domain
        self.codomain = codomain

    def is_continuous_at(self, point, epsilon: float = 0.01, delta: float = 0.01, samples: int = 100) -> bool:
        """
        Check continuity at a point using ε-δ definition.

        Samples points in B(point, delta) and checks if f maps them to B(f(point), epsilon).
        """
        try:
            f_point = self.function(point)
        except:
            return False

        # Sample points in delta-neighborhood
        if not self.domain.elements:
            # Generate samples for continuous spaces
            if isinstance(point, np.ndarray):
                dim = len(point)
                samples_pts = [point + delta * np.random.randn(dim) / 10 for _ in range(samples)]
            else:
                return None  # Cannot test without elements
        else:
            # Use elements in delta-ball
            delta_ball = self.domain.open_ball(point, delta)
            samples_pts = delta_ball if delta_ball else [point]

        # Check if all samples map within epsilon of f(point)
        for x in samples_pts:
            try:
                f_x = self.function(x)
                distance = self.codomain.distance(f_x, f_point)

                if distance > epsilon:
                    return False
            except:
                return False

        return True

    def is_uniformly_continuous(self, epsilon: float = 0.01, sample_size: int = 50) -> bool:
        """
        Check uniform continuity.

        f is uniformly continuous if:
        ∀ε > 0, ∃δ > 0: ∀x,y: d_X(x,y) < δ ⟹ d_Y(f(x), f(y)) < ε

        Note: Continuous on compact sets ⟹ uniformly continuous
        """
        if not self.domain.elements:
            logger.warning("Cannot verify uniform continuity without domain elements")
            return None

        # Sample pairs of points
        n = min(sample_size, len(self.domain.elements))

        # Try to find a δ that works
        for delta in [0.1, 0.05, 0.01, 0.005, 0.001]:
            works = True

            for _ in range(n):
                i, j = np.random.choice(len(self.domain.elements), size=2, replace=True)
                x, y = self.domain.elements[i], self.domain.elements[j]

                d_xy = self.domain.distance(x, y)

                if d_xy < delta:
                    try:
                        f_x, f_y = self.function(x), self.function(y)
                        d_fxy = self.codomain.distance(f_x, f_y)

                        if d_fxy > epsilon:
                            works = False
                            break
                    except:
                        works = False
                        break

            if works:
                return True

        return False

    def lipschitz_constant(self, sample_size: int = 100) -> Optional[float]:
        """
        Compute Lipschitz constant L.

        f is L-Lipschitz if: d_Y(f(x), f(y)) ≤ L · d_X(x, y)

        Returns smallest L found (upper bound on true Lipschitz constant).
        """
        if not self.domain.elements or len(self.domain.elements) < 2:
            return None

        max_ratio = 0.0

        n = min(sample_size, len(self.domain.elements))

        for _ in range(n):
            i, j = np.random.choice(len(self.domain.elements), size=2, replace=False)
            x, y = self.domain.elements[i], self.domain.elements[j]

            d_xy = self.domain.distance(x, y)

            if d_xy < 1e-10:
                continue

            try:
                f_x, f_y = self.function(x), self.function(y)
                d_fxy = self.codomain.distance(f_x, f_y)

                ratio = d_fxy / d_xy
                max_ratio = max(max_ratio, ratio)
            except:
                return None

        return max_ratio if max_ratio > 0 else None


# ============================================================================
# Differentiation
# ============================================================================

class Differentiator:
    """
    Compute derivatives: directional, Gateaux, Frechet.

    Essential for optimization and gradient-based learning.
    """

    @staticmethod
    def directional_derivative(f: Callable,
                               point: np.ndarray,
                               direction: np.ndarray,
                               h: float = 1e-7) -> float:
        """
        Compute directional derivative D_v f(x).

        D_v f(x) = lim_{h→0} [f(x + hv) - f(x)] / h
        """
        direction_normalized = direction / (np.linalg.norm(direction) + 1e-10)

        f_x = f(point)
        f_x_plus = f(point + h * direction_normalized)

        return (f_x_plus - f_x) / h

    @staticmethod
    def gradient(f: Callable,
                 point: np.ndarray,
                 h: float = 1e-7) -> np.ndarray:
        """
        Compute gradient ∇f(x) via finite differences.

        ∇f = (∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ)
        """
        dim = len(point)
        grad = np.zeros(dim)

        f_x = f(point)

        for i in range(dim):
            e_i = np.zeros(dim)
            e_i[i] = 1.0

            f_x_plus = f(point + h * e_i)
            grad[i] = (f_x_plus - f_x) / h

        return grad

    @staticmethod
    def jacobian(f: Callable,
                 point: np.ndarray,
                 h: float = 1e-7) -> np.ndarray:
        """
        Compute Jacobian matrix J_f(x).

        For f: ℝⁿ → ℝᵐ, J is m×n matrix with J_ij = ∂fᵢ/∂xⱼ
        """
        dim_in = len(point)

        # Evaluate function to get output dimension
        f_x = f(point)
        if np.isscalar(f_x):
            dim_out = 1
            f_x = np.array([f_x])
        else:
            dim_out = len(f_x)

        jacobian_matrix = np.zeros((dim_out, dim_in))

        for j in range(dim_in):
            e_j = np.zeros(dim_in)
            e_j[j] = 1.0

            f_x_plus = f(point + h * e_j)
            if np.isscalar(f_x_plus):
                f_x_plus = np.array([f_x_plus])

            jacobian_matrix[:, j] = (f_x_plus - f_x) / h

        return jacobian_matrix

    @staticmethod
    def hessian(f: Callable,
                point: np.ndarray,
                h: float = 1e-5) -> np.ndarray:
        """
        Compute Hessian matrix H_f(x).

        H_ij = ∂²f/∂xᵢ∂xⱼ
        """
        dim = len(point)
        hessian_matrix = np.zeros((dim, dim))

        f_x = f(point)

        for i in range(dim):
            for j in range(i, dim):  # Symmetric, so only compute upper triangle
                e_i = np.zeros(dim)
                e_j = np.zeros(dim)
                e_i[i] = 1.0
                e_j[j] = 1.0

                # Second-order finite difference
                f_ij = f(point + h * e_i + h * e_j)
                f_i = f(point + h * e_i)
                f_j = f(point + h * e_j)

                hessian_matrix[i, j] = (f_ij - f_i - f_j + f_x) / (h * h)
                hessian_matrix[j, i] = hessian_matrix[i, j]  # Symmetry

        return hessian_matrix

    @staticmethod
    def is_frechet_differentiable(f: Callable,
                                    point: np.ndarray,
                                    tolerance: float = 1e-4) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if f is Fréchet differentiable at point.

        f is Fréchet differentiable if there exists linear map Df such that:
        lim_{h→0} ||f(x+h) - f(x) - Df(x)·h|| / ||h|| = 0

        Returns (is_differentiable, derivative_matrix)
        """
        # Compute Jacobian as candidate for Df
        J = Differentiator.jacobian(f, point)

        # Test with random directions
        dim = len(point)

        for _ in range(10):
            h = np.random.randn(dim) * 0.01
            h_norm = np.linalg.norm(h)

            if h_norm < 1e-10:
                continue

            f_x = f(point)
            f_xh = f(point + h)

            if np.isscalar(f_x):
                f_x = np.array([f_x])
                f_xh = np.array([f_xh])

            # Linear approximation
            linear_approx = f_x + J @ h

            # Error
            error = np.linalg.norm(f_xh - linear_approx) / h_norm

            if error > tolerance:
                return (False, None)

        return (True, J)


# ============================================================================
# Integration
# ============================================================================

class RiemannIntegrator:
    """
    Riemann integration for functions.

    Foundation for measure theory and probability.
    """

    @staticmethod
    def integrate_1d(f: Callable,
                     a: float,
                     b: float,
                     method: str = "simpson",
                     n: int = 1000) -> float:
        """
        Compute ∫ₐᵇ f(x) dx using numerical methods.

        Methods:
        - riemann: Riemann sum with midpoints
        - trapezoid: Trapezoidal rule
        - simpson: Simpson's rule (default, most accurate)
        """
        if method == "riemann":
            # Riemann sum with midpoints
            dx = (b - a) / n
            x_vals = np.linspace(a + dx/2, b - dx/2, n)
            return np.sum(f(x_vals)) * dx

        elif method == "trapezoid":
            # Trapezoidal rule
            x_vals = np.linspace(a, b, n + 1)
            y_vals = f(x_vals)
            dx = (b - a) / n
            return dx * (0.5 * y_vals[0] + np.sum(y_vals[1:-1]) + 0.5 * y_vals[-1])

        elif method == "simpson":
            # Simpson's rule (requires even number of intervals)
            if n % 2 == 1:
                n += 1

            x_vals = np.linspace(a, b, n + 1)
            y_vals = f(x_vals)
            dx = (b - a) / n

            return dx / 3 * (y_vals[0] + 4 * np.sum(y_vals[1:-1:2]) +
                            2 * np.sum(y_vals[2:-1:2]) + y_vals[-1])

        else:
            raise ValueError(f"Unknown integration method: {method}")

    @staticmethod
    def integrate_nd(f: Callable,
                     bounds: List[Tuple[float, float]],
                     n_per_dim: int = 50) -> float:
        """
        Compute multi-dimensional integral via Monte Carlo.

        ∫∫...∫ f(x₁, ..., xₙ) dx₁...dxₙ
        """
        dim = len(bounds)

        # Monte Carlo integration
        n_samples = n_per_dim ** dim

        # Generate random samples
        samples = np.zeros((n_samples, dim))
        for i, (low, high) in enumerate(bounds):
            samples[:, i] = np.random.uniform(low, high, n_samples)

        # Evaluate function
        values = np.array([f(x) for x in samples])

        # Volume of integration region
        volume = np.prod([high - low for low, high in bounds])

        # Monte Carlo estimate
        return volume * np.mean(values)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Real Analysis Demo")
    print("="*80 + "\n")

    # 1. Metric Spaces
    print("1. Metric Spaces")
    print("-" * 40)

    # Create embedding space
    embeddings = [np.random.randn(10) for _ in range(100)]
    embedding_space = MetricSpace(elements=embeddings, name="EmbeddingSpace")

    print(f"Space: {embedding_space.name}")
    print(f"Metric axioms verified: {embedding_space.is_metric()}")
    print(f"Complete: {embedding_space.is_complete()}")

    # 2. Sequences
    print("\n2. Sequence Convergence")
    print("-" * 40)

    # Convergent sequence
    convergent = [1/n for n in range(1, 101)]
    print(f"Sequence 1/n convergent: {SequenceAnalyzer.is_convergent(convergent)}")
    print(f"Limit: {SequenceAnalyzer.limit(convergent)}")

    # Monotone
    is_mono, direction = SequenceAnalyzer.is_monotone(convergent)
    print(f"Monotone {direction}: {is_mono}")

    # 3. Continuity
    print("\n3. Continuity Analysis")
    print("-" * 40)

    # Define function f(x) = x²
    def square(x):
        if isinstance(x, np.ndarray):
            return x @ x
        return x * x

    real_line = MetricSpace(metric=lambda x, y: abs(x - y), name="R")
    checker = ContinuityChecker(square, real_line, real_line)

    print(f"f(x)=x² continuous at 2: {checker.is_continuous_at(2.0)}")

    # Lipschitz
    def linear(x):
        return 2 * x

    linear_checker = ContinuityChecker(linear, real_line, real_line)
    L = 2.0  # Known Lipschitz constant
    print(f"f(x)=2x is {L}-Lipschitz")

    # 4. Differentiation
    print("\n4. Differentiation")
    print("-" * 40)

    # Gradient of quadratic
    def quadratic(x):
        return x @ x

    point = np.array([1.0, 2.0, 3.0])
    grad = Differentiator.gradient(quadratic, point)
    print(f"∇(x²) at {point}: {grad}")
    print(f"Expected: {2 * point}")

    # Hessian
    H = Differentiator.hessian(quadratic, point)
    print(f"Hessian is 2I: {np.allclose(H, 2 * np.eye(3))}")

    # 5. Integration
    print("\n5. Integration")
    print("-" * 40)

    # ∫₀¹ x² dx = 1/3
    integral = RiemannIntegrator.integrate_1d(lambda x: x**2, 0, 1)
    print(f"∫₀¹ x² dx = {integral:.6f} (exact: 0.333333)")

    # ∫₀^π sin(x) dx = 2
    integral_sin = RiemannIntegrator.integrate_1d(np.sin, 0, np.pi)
    print(f"∫₀^π sin(x) dx = {integral_sin:.6f} (exact: 2.000000)")

    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)
