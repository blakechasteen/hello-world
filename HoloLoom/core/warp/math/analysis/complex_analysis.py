"""
Complex Analysis for HoloLoom Warp Drive
=========================================

Complex-valued functions for signal processing and Fourier analysis.

Core Concepts:
- Holomorphic Functions: Complex differentiability
- Cauchy's Theorem: Contour integration
- Residue Calculus: Computing complex integrals
- Conformal Mappings: Angle-preserving transformations
- Laurent Series: Generalized Taylor series
- Analytic Continuation: Extending function domains

Mathematical Foundation:
f: ℂ → ℂ is holomorphic if f'(z) exists and is continuous.
Equivalently (Cauchy-Riemann equations):
∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x
where f(z) = u(x,y) + iv(x,y)

Applications to Warp Space:
- Fourier transforms for embeddings
- Signal processing on semantic spaces
- Conformal mappings for dimensionality reduction
- Residue theory for spectral analysis

Author: HoloLoom Team
Date: 2025-10-25
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# Complex Functions
# ============================================================================

class ComplexFunction:
    """
    Complex-valued function f: ℂ → ℂ.

    Provides analysis of holomorphic properties.
    """

    def __init__(self, f: Callable[[complex], complex], name: str = "f"):
        self.f = f
        self.name = name

    def __call__(self, z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
        """Evaluate f(z)."""
        if isinstance(z, np.ndarray):
            return np.array([self.f(zi) for zi in z.flat]).reshape(z.shape)
        return self.f(z)

    def derivative(self, z: complex, h: float = 1e-7) -> complex:
        """
        Compute complex derivative f'(z).

        f'(z) = lim_{h→0} [f(z+h) - f(z)] / h
        """
        h_complex = h + 0j
        return (self.f(z + h_complex) - self.f(z)) / h_complex

    def is_cauchy_riemann(self, z: complex, tolerance: float = 1e-5) -> bool:
        """
        Check Cauchy-Riemann equations at z.

        f(x + iy) = u(x,y) + iv(x,y)
        CR: ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x
        """
        x, y = z.real, z.imag
        h = 1e-6

        # Compute partial derivatives numerically
        f_z = self.f(z)
        f_x_plus = self.f(complex(x + h, y))
        f_y_plus = self.f(complex(x, y + h))

        # ∂f/∂x
        df_dx = (f_x_plus - f_z) / h
        u_x = df_dx.real
        v_x = df_dx.imag

        # ∂f/∂y
        df_dy = (f_y_plus - f_z) / h
        u_y = df_dy.real
        v_y = df_dy.imag

        # Check CR equations
        cr1 = abs(u_x - v_y) < tolerance
        cr2 = abs(u_y + v_x) < tolerance

        return cr1 and cr2

    def is_holomorphic_at(self, z: complex) -> bool:
        """
        Check if f is holomorphic (complex differentiable) at z.

        Uses Cauchy-Riemann equations.
        """
        return self.is_cauchy_riemann(z)


# ============================================================================
# Contour Integration
# ============================================================================

class ContourIntegrator:
    """
    Integrate complex functions over contours.

    Essential for Cauchy's theorem and residue calculus.
    """

    @staticmethod
    def integrate_contour(f: Callable[[complex], complex],
                         contour: Callable[[float], complex],
                         a: float,
                         b: float,
                         n: int = 1000) -> complex:
        """
        Compute ∫_γ f(z) dz where γ(t) is parametrized contour for t ∈ [a,b].

        ∫_γ f(z) dz = ∫_a^b f(γ(t)) · γ'(t) dt
        """
        t_vals = np.linspace(a, b, n)
        dt = (b - a) / n

        integral = 0j

        for i in range(n - 1):
            t = t_vals[i]
            t_next = t_vals[i + 1]

            # Midpoint
            t_mid = (t + t_next) / 2
            z_mid = contour(t_mid)

            # Derivative γ'(t) ≈ (γ(t+dt) - γ(t)) / dt
            gamma_prime = (contour(t_next) - contour(t)) / (t_next - t)

            # Integrand: f(γ(t)) · γ'(t)
            integrand = f(z_mid) * gamma_prime

            integral += integrand * dt

        return integral

    @staticmethod
    def integrate_circle(f: Callable[[complex], complex],
                        center: complex,
                        radius: float,
                        n: int = 1000) -> complex:
        """
        Integrate f over circle |z - center| = radius.

        Parametrization: γ(t) = center + radius·e^(it) for t ∈ [0, 2π]
        """
        def circle_contour(t):
            return center + radius * np.exp(1j * t)

        return ContourIntegrator.integrate_contour(f, circle_contour, 0, 2 * np.pi, n)


# ============================================================================
# Residue Calculus
# ============================================================================

class ResidueCalculator:
    """
    Compute residues for residue theorem.

    Residue Theorem: ∫_γ f(z) dz = 2πi · Σ Res(f, z_k)
    """

    @staticmethod
    def residue_at_pole(f: Callable[[complex], complex],
                       pole: complex,
                       order: int = 1,
                       epsilon: float = 0.01) -> complex:
        """
        Compute residue of f at pole of given order.

        Res(f, z₀) = 1/(2πi) ∫_{|z-z₀|=ε} f(z) dz

        For simple pole (order 1):
        Res(f, z₀) = lim_{z→z₀} (z - z₀)f(z)

        For pole of order m:
        Res(f, z₀) = 1/(m-1)! · lim_{z→z₀} d^(m-1)/dz^(m-1) [(z-z₀)^m f(z)]
        """
        if order == 1:
            # Simple pole: Res = lim_{z→z₀} (z - z₀)f(z)
            def g(z):
                if abs(z - pole) < 1e-10:
                    # Use L'Hospital
                    h = 1e-7
                    return ((pole + h - pole) * f(pole + h) - 0) / h
                return (z - pole) * f(z)

            # Approach from multiple directions
            h = 1e-5
            values = []
            for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                z_near = pole + h * np.exp(1j * angle)
                values.append(g(z_near))

            return np.mean(values)

        else:
            # General case: use contour integral
            def integrand(z):
                return f(z)

            residue = ContourIntegrator.integrate_circle(integrand, pole, epsilon) / (2j * np.pi)
            return residue

    @staticmethod
    def compute_integral_via_residues(f: Callable[[complex], complex],
                                     poles: List[complex],
                                     contour_encloses: Callable[[complex], bool] = None) -> complex:
        """
        Compute ∫_γ f(z) dz using residue theorem.

        ∫_γ f(z) dz = 2πi · Σ_{z_k inside γ} Res(f, z_k)
        """
        total_residue = 0j

        for pole in poles:
            if contour_encloses is None or contour_encloses(pole):
                res = ResidueCalculator.residue_at_pole(f, pole)
                total_residue += res

        return 2j * np.pi * total_residue


# ============================================================================
# Conformal Mappings
# ============================================================================

class ConformalMapper:
    """
    Conformal (angle-preserving) mappings.

    Holomorphic functions with non-zero derivative are conformal.
    """

    @staticmethod
    def mobius_transform(z: complex, a: complex, b: complex, c: complex, d: complex) -> complex:
        """
        Möbius transformation: f(z) = (az + b) / (cz + d)

        Properties:
        - Maps circles/lines to circles/lines
        - Preserves angles
        - Invertible (when ad - bc ≠ 0)
        """
        if c * z + d == 0:
            return complex(float('inf'), 0)  # Point at infinity

        return (a * z + b) / (c * z + d)

    @staticmethod
    def exponential_map(z: complex) -> complex:
        """
        Exponential map: w = e^z

        Maps horizontal lines to circles
        Maps vertical lines to rays
        """
        return np.exp(z)

    @staticmethod
    def logarithm_map(z: complex, branch: int = 0) -> complex:
        """
        Logarithm: w = log(z) (multi-valued)

        Principal branch: log(z) = log|z| + i·arg(z) where arg(z) ∈ (-π, π]
        """
        if z == 0:
            raise ValueError("Logarithm undefined at z=0")

        r = abs(z)
        theta = np.angle(z) + 2 * np.pi * branch

        return np.log(r) + 1j * theta

    @staticmethod
    def joukowski_map(z: complex) -> complex:
        """
        Joukowski transformation: w = z + 1/z

        Used in aerodynamics (airfoil design).
        Maps circles to airfoil shapes.
        """
        if z == 0:
            raise ValueError("Joukowski map undefined at z=0")

        return z + 1/z

    @staticmethod
    def schwarz_christoffel_triangle(z: complex, vertices: List[complex], angles: List[float]) -> complex:
        """
        Schwarz-Christoffel mapping for triangles.

        Maps upper half-plane to polygon.
        """
        # Simplified for demonstration
        # Full implementation requires solving for parameters

        logger.warning("Schwarz-Christoffel implementation is simplified")

        # For equilateral triangle
        return z ** (2/3)


# ============================================================================
# Series Expansions
# ============================================================================

class SeriesExpansion:
    """
    Taylor and Laurent series for complex functions.
    """

    @staticmethod
    def taylor_series(f: Callable[[complex], complex],
                     center: complex,
                     order: int = 10) -> List[complex]:
        """
        Compute Taylor series coefficients.

        f(z) = Σ_{n=0}^∞ aₙ(z - z₀)^n where aₙ = f^(n)(z₀) / n!
        """
        coefficients = []

        # Compute derivatives at center
        h = 1e-7

        for n in range(order + 1):
            if n == 0:
                coefficients.append(f(center))
            else:
                # Numerical nth derivative
                # Use finite differences
                derivative_n = SeriesExpansion._nth_derivative(f, center, n, h)
                coefficient = derivative_n / math.factorial(n)
                coefficients.append(coefficient)

        return coefficients

    @staticmethod
    def _nth_derivative(f: Callable[[complex], complex],
                       z: complex,
                       n: int,
                       h: float) -> complex:
        """Compute nth derivative numerically."""
        if n == 0:
            return f(z)
        elif n == 1:
            return (f(z + h) - f(z)) / h
        else:
            # Recursive: f^(n) = (f^(n-1))'
            def f_n_minus_1(z):
                return SeriesExpansion._nth_derivative(f, z, n - 1, h)

            return (f_n_minus_1(z + h) - f_n_minus_1(z)) / h

    @staticmethod
    def evaluate_series(coefficients: List[complex], z: complex, center: complex = 0) -> complex:
        """
        Evaluate series Σ aₙ(z - center)^n.
        """
        result = 0j
        power = 1

        for coef in coefficients:
            result += coef * power
            power *= (z - center)

        return result

    @staticmethod
    def laurent_series(f: Callable[[complex], complex],
                      center: complex,
                      inner_radius: float,
                      outer_radius: float,
                      order_positive: int = 10,
                      order_negative: int = 10) -> Tuple[List[complex], List[complex]]:
        """
        Compute Laurent series in annulus.

        f(z) = Σ_{n=-∞}^∞ aₙ(z - z₀)^n

        Returns (positive_coeffs, negative_coeffs) where:
        - positive_coeffs[n] = aₙ for n ≥ 0
        - negative_coeffs[m] = a₋ₘ for m > 0
        """
        # Positive coefficients (regular part)
        positive_coeffs = SeriesExpansion.taylor_series(f, center, order_positive)

        # Negative coefficients (principal part)
        # aₙ = 1/(2πi) ∫ f(ζ)/(ζ - z₀)^(n+1) dζ
        negative_coeffs = []

        radius = (inner_radius + outer_radius) / 2

        for n in range(1, order_negative + 1):
            def integrand(zeta):
                return f(zeta) / (zeta - center) ** (n + 1)

            coef = ContourIntegrator.integrate_circle(integrand, center, radius) / (2j * np.pi)
            negative_coeffs.append(coef)

        return positive_coeffs, negative_coeffs


# ============================================================================
# Analytic Continuation
# ============================================================================

class AnalyticContinuation:
    """
    Extend domain of analytic functions.
    """

    @staticmethod
    def power_series_continuation(series_coeffs: List[complex],
                                  old_center: complex,
                                  new_center: complex,
                                  order: int = None) -> List[complex]:
        """
        Continue power series to new center.

        Given f(z) = Σ aₙ(z - z₀)^n, compute series around z₁.
        """
        if order is None:
            order = len(series_coeffs)

        # Evaluate original series and derivatives at new center
        # Then compute new coefficients

        def f(z):
            return SeriesExpansion.evaluate_series(series_coeffs, z, old_center)

        # Compute new Taylor series
        new_coeffs = SeriesExpansion.taylor_series(f, new_center, order)

        return new_coeffs


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Complex Analysis Demo")
    print("="*80 + "\n")

    # 1. Holomorphic Functions
    print("1. Holomorphic Functions")
    print("-" * 40)

    # f(z) = z^2 is holomorphic
    f_poly = ComplexFunction(lambda z: z**2, name="z^2")
    print(f"f(z) = z^2 holomorphic at 1+i: {f_poly.is_holomorphic_at(1 + 1j)}")

    # f'(z) = 2z
    z = 2 + 3j
    f_prime = f_poly.derivative(z)
    print(f"f'(2+3i) = {f_prime} (expected: {2*z})")

    # 2. Contour Integration
    print("\n2. Contour Integration")
    print("-" * 40)

    # ∫_{|z|=1} 1/z dz = 2πi
    def one_over_z(z):
        return 1/z

    integral = ContourIntegrator.integrate_circle(one_over_z, center=0, radius=1)
    print(f"Integral of 1/z around unit circle: {integral}")
    print(f"Expected: {2j * np.pi}")

    # 3. Residue Calculus
    print("\n3. Residue Calculus")
    print("-" * 40)

    # Residue of 1/(z-1) at z=1 is 1
    def f_simple_pole(z):
        if abs(z - 1) < 1e-10:
            return float('inf')
        return 1 / (z - 1)

    residue = ResidueCalculator.residue_at_pole(f_simple_pole, pole=1+0j)
    print(f"Res(1/(z-1), z=1) = {residue.real:.6f} (expected: 1)")

    # 4. Conformal Mappings
    print("\n4. Conformal Mappings")
    print("-" * 40)

    # Exponential map
    z = 1 + 1j * np.pi / 2
    w = ConformalMapper.exponential_map(z)
    print(f"exp({z}) = {w}")

    # Mobius transform (identity)
    z = 2 + 3j
    w = ConformalMapper.mobius_transform(z, a=1, b=0, c=0, d=1)
    print(f"Mobius identity: {z} -> {w}")

    # 5. Series Expansion
    print("\n5. Taylor Series")
    print("-" * 40)

    # Taylor series of e^z at z=0
    exp_series = SeriesExpansion.taylor_series(np.exp, center=0, order=5)
    print(f"e^z Taylor coefficients: {[c.real for c in exp_series[:6]]}")
    print("Expected: [1, 1, 1/2, 1/6, 1/24, 1/120, ...]")

    # Evaluate at z=1
    result = SeriesExpansion.evaluate_series(exp_series, z=1, center=0)
    print(f"e^1 via series: {result.real:.6f} (exact: {np.e:.6f})")

    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)
