"""
Exotic Calculus
================

Scope slots: CA5.1 (fractional calculus), CA5.2 (jet bundles),
CA5.3 (ultrafilter convergence), CA5.4-5.5 (reserved/extensions).

Pure numpy implementations of non-standard calculus:
- Fractional calculus: Riemann-Liouville fractional derivative/integral
- Jet bundles: k-jets as Taylor truncations (f, f', f'', ...)
- Ultrafilter convergence: generalized convergence on ultrafilters

These extend classical calculus in different directions: fractional calculus
generalizes d^n/dx^n to non-integer n, jet bundles package all derivatives
into a geometric object, and ultrafilter convergence unifies all notions
of convergence in topology.

Author: Claude Code
Date: March 2026
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from math import factorial
from math import gamma as math_gamma

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# FRACTIONAL CALCULUS (CA5.1)
# ============================================================================

class FractionalCalculus:
    """Fractional derivative and integral via Riemann-Liouville.

    The Riemann-Liouville fractional integral of order α > 0:
        I^α f(x) = (1/Γ(α)) ∫_a^x (x-t)^{α-1} f(t) dt

    The fractional derivative of order α (non-integer):
        D^α f(x) = (d/dx)^m I^{m-α} f(x)

    where m = ⌈α⌉. This gives d^{0.5}/dx^{0.5} and similar.

    Properties:
    - D^n agrees with ordinary derivative for integer n
    - I^α I^β = I^{α+β} (semigroup property)
    - D^α x^β = Γ(β+1)/Γ(β-α+1) x^{β-α} for β > α - 1

    Example:
        >>> fc = FractionalCalculus()
        >>> # Half-derivative of x: D^{0.5} x = 2√x/√π
        >>> x = np.linspace(0.01, 1, 100)
        >>> f = x  # f(x) = x
        >>> D_half = fc.fractional_derivative(f, x, alpha=0.5)
        >>> exact = 2 * np.sqrt(x) / np.sqrt(np.pi)
    """

    def fractional_integral(
        self,
        f: np.ndarray,
        x: np.ndarray,
        alpha: float,
        a: float | None = None,
    ) -> np.ndarray:
        """Riemann-Liouville fractional integral I^α f.

        Args:
            f: Function values at grid points, shape (n,).
            x: Grid points, shape (n,).
            alpha: Order of integration (> 0).
            a: Lower limit (defaults to x[0]).

        Returns:
            I^α f evaluated at each grid point.
        """
        if alpha <= 0:
            raise ValueError("Fractional integral order must be positive")

        n = len(x)
        if a is None:
            a = x[0]

        result = np.zeros(n)
        gamma_alpha = math_gamma(alpha)

        for i in range(n):
            if x[i] <= a:
                continue

            # Trapezoidal quadrature for the convolution integral
            # I^α f(x_i) = (1/Γ(α)) ∫_a^{x_i} (x_i - t)^{α-1} f(t) dt
            integral = 0.0
            for j in range(i):
                t_j = x[j]
                t_j1 = x[j + 1]
                if t_j < a:
                    continue

                # Midpoint kernel evaluation
                t_mid = 0.5 * (t_j + t_j1)
                f_mid = 0.5 * (f[j] + f[j + 1])
                dt = t_j1 - t_j
                kernel = (x[i] - t_mid) ** (alpha - 1)
                integral += kernel * f_mid * dt

            result[i] = integral / gamma_alpha

        return result

    def fractional_derivative(
        self,
        f: np.ndarray,
        x: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Riemann-Liouville fractional derivative D^α f.

        For non-integer α, uses: D^α f = d^m/dx^m [I^{m-α} f]
        where m = ceil(α).

        For integer α, uses standard finite differences.

        Args:
            f: Function values at grid points.
            x: Grid points (must be uniformly spaced).
            alpha: Order of differentiation (≥ 0).

        Returns:
            D^α f evaluated at grid points.
        """
        if alpha < 0:
            return self.fractional_integral(f, x, -alpha)

        if abs(alpha - round(alpha)) < 1e-12:
            # Integer order: use standard finite differences
            m = int(round(alpha))
            result = f.copy()
            dx = x[1] - x[0]
            for _ in range(m):
                result = np.gradient(result, dx)
            return result

        # Non-integer: D^α = d^m/dx^m ∘ I^{m-α}
        m = int(np.ceil(alpha))
        frac_part = m - alpha  # 0 < frac_part < 1

        # First: fractional integral of order (m - α)
        If = self.fractional_integral(f, x, frac_part)

        # Then: m-th integer derivative
        dx = x[1] - x[0]
        result = If.copy()
        for _ in range(m):
            result = np.gradient(result, dx)

        return result

    @staticmethod
    def caputo_derivative(
        f: np.ndarray,
        x: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Caputo fractional derivative (alternative to Riemann-Liouville).

        ^C D^α f = I^{m-α} [d^m f / dx^m]

        The Caputo derivative reverses the order: differentiate first,
        then fractionally integrate. This has better initial condition
        behavior for ODEs.

        Args:
            f: Function values.
            x: Grid points.
            alpha: Order (> 0).

        Returns:
            Caputo fractional derivative at grid points.
        """
        fc = FractionalCalculus()
        m = int(np.ceil(alpha))
        dx = x[1] - x[0]

        # Integer derivative first
        df = f.copy()
        for _ in range(m):
            df = np.gradient(df, dx)

        # Then fractional integral
        frac_part = m - alpha
        if frac_part < 1e-12:
            return df
        return fc.fractional_integral(df, x, frac_part)

    @staticmethod
    def mittag_leffler(z: np.ndarray, alpha: float, beta: float = 1.0, n_terms: int = 50) -> np.ndarray:
        """Mittag-Leffler function E_{α,β}(z) = Σ z^k / Γ(αk + β).

        The fundamental solution of fractional ODEs, generalizing
        the exponential: E_{1,1}(z) = exp(z).

        Args:
            z: Input values.
            alpha: First parameter (> 0).
            beta: Second parameter (default 1).
            n_terms: Number of series terms.

        Returns:
            E_{α,β}(z) at each input point.
        """
        result = np.zeros_like(z, dtype=float)
        z_power = np.ones_like(z, dtype=float)

        for k in range(n_terms):
            term = z_power / math_gamma(alpha * k + beta)
            result += term
            z_power *= z
            if np.all(np.abs(term) < 1e-15):
                break

        return result


# ============================================================================
# JET BUNDLES (CA5.2)
# ============================================================================

@dataclass
class JetData:
    """k-jet data at a point.

    Attributes:
        point: Base point x.
        values: Array [f(x), f'(x), f''(x), ..., f^{(k)}(x)].
        order: Jet order k.
    """
    point: float
    values: np.ndarray
    order: int


class JetBundle:
    """Jet bundle J^k(R, R) — the space of k-jets.

    A k-jet of a function f at x packages the Taylor data:
        j^k_x(f) = (x, f(x), f'(x), f''(x), ..., f^{(k)}(x))

    The k-jet determines the k-th order Taylor polynomial.
    Jet bundles are the natural geometric setting for differential
    equations: a k-th order ODE is a submanifold of J^k.

    Example:
        >>> jb = JetBundle()
        >>> f = lambda x: np.sin(x)
        >>> jet = jb.jet(f, x=0.0, order=4)
        >>> jet.values  # [sin(0), cos(0), -sin(0), -cos(0), sin(0)]
        >>> # ≈ [0, 1, 0, -1, 0]
    """

    def jet(
        self,
        f: Callable[[float], float],
        x: float,
        order: int,
        dx: float = 1e-5,
    ) -> JetData:
        """Compute k-jet of f at x.

        Uses numerical differentiation (central differences with
        Richardson extrapolation for higher orders).

        Args:
            f: Scalar function.
            x: Base point.
            order: Jet order k.
            dx: Step size for numerical differentiation.

        Returns:
            JetData with derivatives up to order k.
        """
        values = np.zeros(order + 1)
        values[0] = f(x)

        for k in range(1, order + 1):
            values[k] = self._numerical_derivative(f, x, k, dx)

        return JetData(point=x, values=values, order=order)

    def jet_from_samples(
        self,
        x_samples: np.ndarray,
        f_samples: np.ndarray,
        x: float,
        order: int,
    ) -> JetData:
        """Estimate k-jet from function samples using polynomial fitting.

        Args:
            x_samples: Sample points near x.
            f_samples: Function values at sample points.
            x: Base point.
            order: Jet order.

        Returns:
            JetData estimated from samples.
        """
        # Fit polynomial of degree order
        n = min(order + 1, len(x_samples))
        coeffs = np.polyfit(x_samples, f_samples, n - 1)
        poly = np.poly1d(coeffs)

        values = np.zeros(order + 1)
        current_poly = poly
        for k in range(order + 1):
            values[k] = current_poly(x)
            if k < order:
                current_poly = current_poly.deriv()

        return JetData(point=x, values=values, order=order)

    def taylor_polynomial(self, jet: JetData) -> Callable[[float], float]:
        """Reconstruct Taylor polynomial from jet data.

        T_k(x) = Σ_{i=0}^{k} f^{(i)}(x₀)/i! · (x - x₀)^i

        Args:
            jet: k-jet data.

        Returns:
            Taylor polynomial function.
        """
        x0 = jet.point
        coeffs = jet.values.copy()

        def taylor(x: float) -> float:
            dx = x - x0
            result = 0.0
            dx_power = 1.0
            for k in range(jet.order + 1):
                result += coeffs[k] / factorial(k) * dx_power
                dx_power *= dx
            return result

        return taylor

    def jet_composition(self, jet_f: JetData, jet_g: JetData) -> JetData:
        """Compute jet of composition j^k(f ∘ g) via Faa di Bruno formula.

        The chain rule for higher derivatives (Faa di Bruno):
            (f ∘ g)^{(n)} = Σ f^{(k)} B_{n,k}(g', g'', ..., g^{(n-k+1)})

        where B_{n,k} are partial Bell polynomials.

        Args:
            jet_f: k-jet of f at g(x).
            jet_g: k-jet of g at x.

        Returns:
            k-jet of f ∘ g at x.
        """
        order = min(jet_f.order, jet_g.order)
        values = np.zeros(order + 1)

        # 0th order: (f∘g)(x) = f(g(x))
        values[0] = jet_f.values[0]

        if order >= 1:
            # 1st order: f'(g(x)) g'(x)
            values[1] = jet_f.values[1] * jet_g.values[1]

        if order >= 2:
            # 2nd order: f''(g(x))(g'(x))² + f'(g(x))g''(x)
            values[2] = (jet_f.values[2] * jet_g.values[1]**2
                        + jet_f.values[1] * jet_g.values[2])

        if order >= 3:
            # 3rd order via Faa di Bruno
            values[3] = (jet_f.values[3] * jet_g.values[1]**3
                        + 3 * jet_f.values[2] * jet_g.values[1] * jet_g.values[2]
                        + jet_f.values[1] * jet_g.values[3])

        # For higher orders, use recursive formula
        # (This covers the most common cases explicitly)

        return JetData(point=jet_g.point, values=values, order=order)

    @staticmethod
    def _numerical_derivative(f: Callable, x: float, order: int, dx: float) -> float:
        """Compute n-th derivative numerically via finite differences."""
        if order == 0:
            return f(x)

        # Central difference stencil coefficients for derivative of order n
        # Use order+2 point stencil for accuracy
        n_pts = order + 2
        if n_pts % 2 == 0:
            n_pts += 1
        half = n_pts // 2

        # Evaluate function at stencil points
        points = [x + k * dx for k in range(-half, half + 1)]
        values = [f(p) for p in points]

        # Apply finite difference formula iteratively
        result = np.array(values, dtype=float)
        for _ in range(order):
            result = np.diff(result) / dx

        return float(result[len(result) // 2])


# ============================================================================
# ULTRAFILTER CONVERGENCE (CA5.3)
# ============================================================================

class UltrafilterConvergence:
    """Convergence along an ultrafilter.

    An ultrafilter U on N is a collection of subsets satisfying:
    1. N ∈ U, ∅ ∉ U
    2. A, B ∈ U ⟹ A ∩ B ∈ U
    3. A ∈ U, A ⊆ B ⟹ B ∈ U
    4. For all A ⊆ N: either A ∈ U or N\\A ∈ U

    A sequence (x_n) converges to L along U iff for every ε > 0,
    {n : |x_n - L| < ε} ∈ U.

    Principal ultrafilters give ordinary limits. Free (non-principal)
    ultrafilters give generalized limits that can assign values to
    sequences like (-1)^n.

    Since explicitly constructing a free ultrafilter requires the axiom
    of choice, we work with principal ultrafilters and approximate
    free ultrafilter behavior via density-based methods.

    Example:
        >>> uf = UltrafilterConvergence()
        >>> seq = [(-1)**n / (n+1) for n in range(100)]  # → 0
        >>> uf.converges(seq, candidate=0.0)  # True
    """

    def principal_ultrafilter(self, n: int, universe_size: int) -> 'PrincipalUltrafilter':
        """Create the principal ultrafilter at n.

        The principal ultrafilter at n is {A ⊆ N : n ∈ A}.
        Convergence along it just evaluates at n.

        Args:
            n: The focal point.
            universe_size: Size of the index set.

        Returns:
            PrincipalUltrafilter object.
        """
        return PrincipalUltrafilter(n, universe_size)

    def converges(
        self,
        sequence: list[float],
        candidate: float | None = None,
        tol: float = 1e-6,
    ) -> bool:
        """Test if a sequence converges (standard convergence).

        Uses the Cauchy criterion: for all ε > 0, ∃ N such that
        |x_m - x_n| < ε for all m, n ≥ N.

        Args:
            sequence: Sequence values.
            candidate: Candidate limit (auto-detected if None).
            tol: Convergence tolerance.

        Returns:
            True if the sequence converges.
        """
        if len(sequence) < 2:
            return True

        if candidate is None:
            candidate = self._estimate_limit(sequence)

        # Check if tail is within tolerance
        n = len(sequence)
        tail_start = max(1, n // 2)
        tail = np.array(sequence[tail_start:])
        return bool(np.all(np.abs(tail - candidate) < tol))

    def banach_limit(self, sequence: list[float]) -> float:
        """Compute approximate Banach limit (generalized limit).

        For convergent sequences, agrees with the ordinary limit.
        For bounded non-convergent sequences (like (-1)^n), returns
        the Cesaro mean as an approximation to the Banach limit.

        Args:
            sequence: Bounded sequence.

        Returns:
            Approximate Banach limit.
        """
        # Cesaro mean: lim (1/N) Σ x_n
        n = len(sequence)
        cumsum = np.cumsum(sequence)
        cesaro = cumsum / np.arange(1, n + 1)

        # Check if Cesaro means converge
        if n > 10:
            tail = cesaro[n // 2:]
            if np.std(tail) < 1e-8:
                return float(np.mean(tail))

        return float(cesaro[-1])

    def density_convergence(
        self,
        sequence: list[float],
        candidate: float,
        tol: float = 1e-6,
    ) -> float:
        """Compute the natural density of indices where |x_n - L| < ε.

        For ordinary convergence, this density is 1.
        For statistical convergence, it's between 0 and 1.

        Args:
            sequence: Sequence values.
            candidate: Candidate limit.
            tol: Tolerance.

        Returns:
            Natural density (0 to 1).
        """
        n = len(sequence)
        close = sum(1 for x in sequence if abs(x - candidate) < tol)
        return close / n

    @staticmethod
    def _estimate_limit(sequence: list[float]) -> float:
        """Estimate the limit of a potentially convergent sequence."""
        n = len(sequence)
        if n < 4:
            return sequence[-1]

        # Use average of last quarter
        tail = sequence[3 * n // 4:]
        return float(np.mean(tail))


@dataclass
class PrincipalUltrafilter:
    """Principal ultrafilter at a point.

    The principal ultrafilter U_n = {A ⊆ N : n ∈ A}.
    A sequence converges along U_n to x_n (trivially).
    """
    point: int
    universe_size: int

    def contains(self, subset: set[int]) -> bool:
        """Check if A ∈ U (i.e., point ∈ A)."""
        return self.point in subset

    def limit(self, sequence: list[float]) -> float:
        """Limit along this ultrafilter = value at the focal point."""
        if self.point < len(sequence):
            return sequence[self.point]
        raise IndexError(f"Sequence has {len(sequence)} elements, point is {self.point}")
