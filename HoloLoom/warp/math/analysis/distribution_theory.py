"""
Distribution Theory for HoloLoom Warp Drive
===========================================

Generalized functions (distributions) and weak derivatives.

Core Concepts:
- Test Functions: Schwartz space S(ℝ) of rapidly decreasing functions
- Distributions: Continuous linear functionals on test functions
- Tempered Distributions: Dual of Schwartz space
- Dirac Delta: δ(f) = f(0)
- Weak Derivatives: Derivatives of distributions

Mathematical Foundation:
Distribution: T: S(ℝ) → ℂ continuous linear functional
Dirac delta: ⟨δ, φ⟩ = φ(0)
Weak derivative: ⟨T', φ⟩ = -⟨T, φ'⟩
Fourier transform: ⟨F[T], φ⟩ = ⟨T, F[φ]⟩

Applications to Warp Space:
- Green's functions for PDEs
- Impulse responses in signal processing
- Weak solutions to differential equations
- Quantum mechanics (wavefunctions as distributions)

Author: HoloLoom Team
Date: 2025-10-26
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SCHWARTZ SPACE (TEST FUNCTIONS)
# ============================================================================

class SchwartzFunction:
    """
    Rapidly decreasing test function φ ∈ S(ℝ).

    φ is smooth and x^n φ^(m)(x) → 0 as |x| → ∞ for all n, m.
    """

    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        """
        Initialize Schwartz function.

        Args:
            function: The test function φ(x)
        """
        self.function = function

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate φ(x)."""
        return self.function(x)

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Compute derivative φ^(n)(x) using finite differences.
        """
        h = 1e-6

        if order == 0:
            return self.function(x)
        elif order == 1:
            return (self.function(x + h) - self.function(x - h)) / (2 * h)
        elif order == 2:
            return (self.function(x + h) - 2*self.function(x) + self.function(x - h)) / h**2
        else:
            # Higher orders via recursion
            deriv_n_minus_1 = self.derivative(x, order - 1)
            deriv_n_minus_1_plus_h = self.derivative(x + h, order - 1)
            return (deriv_n_minus_1_plus_h - deriv_n_minus_1) / h

    def is_rapidly_decreasing(self, x_max: float = 10) -> bool:
        """
        Check if φ is rapidly decreasing.

        Test if x² |φ(x)| → 0 as |x| → ∞
        """
        x_test = np.array([-x_max, x_max])
        phi_test = self.function(x_test)

        decay = x_test**2 * np.abs(phi_test)

        return np.all(decay < 0.1)

    @staticmethod
    def gaussian(sigma: float = 1.0) -> 'SchwartzFunction':
        """
        Gaussian test function: φ(x) = exp(-x²/(2σ²))

        This is the canonical Schwartz function.
        """
        return SchwartzFunction(lambda x: np.exp(-x**2 / (2 * sigma**2)))

    @staticmethod
    def bump_function(support: Tuple[float, float] = (-1, 1)) -> 'SchwartzFunction':
        """
        Smooth bump function with compact support.

        φ(x) = exp(-1/(1-x²)) for |x| < 1, 0 otherwise

        Infinitely differentiable everywhere.
        """
        a, b = support
        center = (a + b) / 2
        radius = (b - a) / 2

        def bump(x):
            normalized = (x - center) / radius
            result = np.zeros_like(x)
            mask = np.abs(normalized) < 1
            result[mask] = np.exp(-1 / (1 - normalized[mask]**2))
            return result

        return SchwartzFunction(bump)


# ============================================================================
# DISTRIBUTIONS
# ============================================================================

class Distribution:
    """
    Distribution (generalized function): T: S(ℝ) → ℂ

    Continuous linear functional on test functions.
    """

    def __init__(self, action: Callable[[SchwartzFunction], complex]):
        """
        Initialize distribution.

        Args:
            action: The functional T(φ) = ⟨T, φ⟩
        """
        self.action = action

    def __call__(self, phi: SchwartzFunction) -> complex:
        """
        Apply distribution to test function: ⟨T, φ⟩
        """
        return self.action(phi)

    def derivative(self, order: int = 1) -> 'Distribution':
        """
        Weak derivative: ⟨T', φ⟩ = -⟨T, φ'⟩

        Derivatives always exist for distributions!
        """
        def weak_derivative_action(phi: SchwartzFunction):
            # T'(φ) = -T(φ')
            sign = (-1) ** order

            # Create test function for derivative
            def phi_deriv_func(x):
                return phi.derivative(x, order)

            phi_deriv = SchwartzFunction(phi_deriv_func)
            return sign * self.action(phi_deriv)

        logger.info(f"Computed {order}-th weak derivative")
        return Distribution(weak_derivative_action)

    def __add__(self, other: 'Distribution') -> 'Distribution':
        """
        Sum of distributions: (T + S)(φ) = T(φ) + S(φ)
        """
        return Distribution(lambda phi: self.action(phi) + other.action(phi))

    def __mul__(self, scalar: complex) -> 'Distribution':
        """
        Scalar multiplication: (αT)(φ) = α T(φ)
        """
        return Distribution(lambda phi: scalar * self.action(phi))


# ============================================================================
# STANDARD DISTRIBUTIONS
# ============================================================================

class StandardDistributions:
    """
    Common distributions.
    """

    @staticmethod
    def dirac_delta(x0: float = 0) -> Distribution:
        """
        Dirac delta distribution: ⟨δ_x0, φ⟩ = φ(x0)

        Point mass at x0.
        """
        def delta_action(phi: SchwartzFunction):
            return phi(np.array([x0]))[0]

        logger.info(f"Dirac delta at x={x0}")
        return Distribution(delta_action)

    @staticmethod
    def heaviside(x0: float = 0) -> Distribution:
        """
        Heaviside step function: H(x) = 1 if x ≥ x0, 0 otherwise

        As distribution: ⟨H, φ⟩ = ∫_{x0}^∞ φ(x) dx
        """
        def heaviside_action(phi: SchwartzFunction):
            # Integrate from x0 to infinity
            x = np.linspace(x0, x0 + 20, 1000)  # Approximate infinity
            dx = x[1] - x[0]
            return np.sum(phi(x)) * dx

        logger.info(f"Heaviside step at x={x0}")
        return Distribution(heaviside_action)

    @staticmethod
    def principal_value(singularity: float = 0) -> Distribution:
        """
        Principal value distribution: PV(1/x)

        ⟨PV(1/x), φ⟩ = lim_{ε→0} ∫_{|x|>ε} φ(x)/x dx
        """
        def pv_action(phi: SchwartzFunction):
            epsilon = 1e-6

            # Integrate over (-∞, -ε) and (ε, ∞)
            x_neg = np.linspace(-20 + singularity, singularity - epsilon, 500)
            x_pos = np.linspace(singularity + epsilon, 20 + singularity, 500)

            dx_neg = x_neg[1] - x_neg[0]
            dx_pos = x_pos[1] - x_pos[0]

            integral_neg = np.sum(phi(x_neg) / (x_neg - singularity)) * dx_neg
            integral_pos = np.sum(phi(x_pos) / (x_pos - singularity)) * dx_pos

            return integral_neg + integral_pos

        logger.info("Principal value PV(1/x)")
        return Distribution(pv_action)

    @staticmethod
    def regular_distribution(f: Callable[[np.ndarray], np.ndarray]) -> Distribution:
        """
        Regular distribution from locally integrable function.

        ⟨T_f, φ⟩ = ∫ f(x) φ(x) dx
        """
        def regular_action(phi: SchwartzFunction):
            # Integrate over support
            x = np.linspace(-20, 20, 1000)
            dx = x[1] - x[0]
            return np.sum(f(x) * phi(x)) * dx

        logger.info("Regular distribution from function")
        return Distribution(regular_action)


# ============================================================================
# FOURIER TRANSFORM OF DISTRIBUTIONS
# ============================================================================

class DistributionFourier:
    """
    Fourier transform of tempered distributions.

    ⟨F[T], φ⟩ = ⟨T, F[φ]⟩
    """

    @staticmethod
    def fourier_transform(T: Distribution) -> Distribution:
        """
        Fourier transform of distribution T.

        F[T](φ) = T(F[φ])
        """
        def fourier_action(phi: SchwartzFunction):
            # Compute Fourier transform of φ
            def phi_hat_func(omega):
                # F[φ](ω) = ∫ φ(x) exp(-iωx) dx
                x = np.linspace(-20, 20, 1000)
                dx = x[1] - x[0]
                integrand = phi(x) * np.exp(-1j * omega * x)
                return np.sum(integrand) * dx

            phi_hat = SchwartzFunction(phi_hat_func)

            # Apply T to F[φ]
            return T.action(phi_hat)

        logger.info("Fourier transform of distribution")
        return Distribution(fourier_action)

    @staticmethod
    def fourier_dirac_delta() -> Distribution:
        """
        Fourier transform of Dirac delta.

        F[δ](ω) = 1 (constant distribution)
        """
        def constant_action(phi: SchwartzFunction):
            # ⟨1, φ⟩ = ∫ φ(x) dx
            x = np.linspace(-20, 20, 1000)
            dx = x[1] - x[0]
            return np.sum(phi(x)) * dx

        logger.info("F[δ] = 1 (constant)")
        return Distribution(constant_action)


# ============================================================================
# GREEN'S FUNCTIONS
# ============================================================================

class GreenFunction:
    """
    Green's function (fundamental solution) for differential operators.

    L G(x, y) = δ(x - y)
    """

    @staticmethod
    def laplacian_1d(x: np.ndarray, y: float = 0) -> np.ndarray:
        """
        Green's function for 1D Laplacian: -d²/dx²

        G(x, y) = -|x - y| / 2
        """
        return -np.abs(x - y) / 2

    @staticmethod
    def heat_kernel_1d(x: np.ndarray, t: float, y: float = 0) -> np.ndarray:
        """
        Heat kernel (fundamental solution for heat equation).

        ∂u/∂t = ∂²u/∂x²

        G(x, t; y) = (1/√(4πt)) exp(-(x-y)²/(4t))
        """
        if t <= 0:
            raise ValueError("Time must be positive")

        return (1 / np.sqrt(4 * np.pi * t)) * np.exp(-(x - y)**2 / (4 * t))

    @staticmethod
    def wave_kernel_1d(x: np.ndarray, t: float, c: float = 1.0) -> np.ndarray:
        """
        Wave kernel (d'Alembert formula for 1D wave equation).

        ∂²u/∂t² = c² ∂²u/∂x²

        G(x, t) = H(ct - |x|) / (2c)
        where H is Heaviside function
        """
        result = np.zeros_like(x)
        mask = np.abs(x) < c * t
        result[mask] = 1 / (2 * c)
        return result


# ============================================================================
# WEAK DERIVATIVES & SOBOLEV SPACES CONNECTION
# ============================================================================

class WeakDerivative:
    """
    Weak (distributional) derivatives.

    Connection to Sobolev spaces from functional_analysis.py
    """

    @staticmethod
    def compute_weak_derivative(
        f: Callable[[np.ndarray], np.ndarray],
        test_function: SchwartzFunction,
        order: int = 1
    ) -> complex:
        """
        Compute weak derivative using integration by parts.

        ⟨f', φ⟩ = -⟨f, φ'⟩ = -∫ f(x) φ'(x) dx
        """
        x = np.linspace(-10, 10, 1000)
        dx = x[1] - x[0]

        # Compute φ'(x)
        phi_prime = test_function.derivative(x, order)

        # Integrate
        sign = (-1) ** order
        weak_deriv = sign * np.sum(f(x) * phi_prime) * dx

        return weak_deriv

    @staticmethod
    def is_weakly_differentiable(
        f: Callable[[np.ndarray], np.ndarray],
        num_test_functions: int = 5
    ) -> bool:
        """
        Check if function is weakly differentiable.

        Tests consistency with multiple test functions.
        """
        # Generate test functions
        test_funcs = [
            SchwartzFunction.gaussian(sigma=s)
            for s in np.linspace(0.5, 2, num_test_functions)
        ]

        # Compute weak derivatives
        weak_derivs = []
        for phi in test_funcs:
            wd = WeakDerivative.compute_weak_derivative(f, phi, order=1)
            weak_derivs.append(wd)

        # Check if they're consistent (similar values)
        weak_derivs = np.array(weak_derivs)
        std = np.std(np.real(weak_derivs))

        return std < 1.0  # Threshold for consistency


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'SchwartzFunction',
    'Distribution',
    'StandardDistributions',
    'DistributionFourier',
    'GreenFunction',
    'WeakDerivative'
]
