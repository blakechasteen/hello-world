"""
Stochastic Calculus for HoloLoom Warp Drive
===========================================

Stochastic processes, Brownian motion, and stochastic differential equations.

Core Concepts:
- Brownian Motion: Fundamental continuous-time stochastic process
- Martingales: Fair game processes
- Ito Calculus: Stochastic integration and differentiation
- Stochastic Differential Equations (SDEs): Random dynamical systems
- Ito's Lemma: Chain rule for stochastic calculus

Mathematical Foundation:
Brownian Motion B(t) satisfies:
1. B(0) = 0
2. Independent increments
3. B(t) - B(s) ~ N(0, t-s) for t > s
4. Continuous paths

Ito Integral: ∫ f(t) dB(t) (stochastic integral)
Ito's Lemma: dF = (∂F/∂t + ½ ∂²F/∂x² σ²) dt + (∂F/∂x σ) dB

Applications to Warp Space:
- Stochastic gradient descent dynamics
- Uncertainty quantification in neural networks
- Random walk on knowledge graphs
- Diffusion processes on manifolds

Author: HoloLoom Team
Date: 2025-10-26
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# BROWNIAN MOTION
# ============================================================================

class BrownianMotion:
    """
    Standard Brownian motion (Wiener process).

    Properties:
    - B(0) = 0
    - B(t) - B(s) ~ N(0, t-s)
    - Independent increments
    - Continuous paths (a.s.)
    """

    @staticmethod
    def generate_path(
        T: float,
        n_steps: int,
        n_paths: int = 1,
        initial_value: float = 0.0
    ) -> np.ndarray:
        """
        Generate Brownian motion paths.

        Args:
            T: Final time
            n_steps: Number of time steps
            n_paths: Number of independent paths
            initial_value: Starting point

        Returns:
            Array of shape (n_paths, n_steps+1) containing paths
        """
        dt = T / n_steps

        # Generate independent increments
        # dB ~ N(0, dt)
        increments = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))

        # Cumulative sum to get path
        paths = np.cumsum(increments, axis=1)

        # Add initial value
        paths = np.hstack([initial_value * np.ones((n_paths, 1)), paths])

        logger.info(f"Generated {n_paths} Brownian paths with {n_steps} steps")

        return paths

    @staticmethod
    def geometric_brownian_motion(
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> np.ndarray:
        """
        Generate Geometric Brownian Motion (GBM).

        dS = μS dt + σS dB
        Solution: S(t) = S₀ exp((μ - σ²/2)t + σB(t))

        Used in finance (Black-Scholes) and population models.

        Args:
            S0: Initial value
            mu: Drift parameter
            sigma: Volatility parameter
            T: Final time
            n_steps: Number of steps
            n_paths: Number of paths

        Returns:
            GBM paths
        """
        # Generate standard Brownian motion
        B = BrownianMotion.generate_path(T, n_steps, n_paths, 0.0)

        # Time grid
        t = np.linspace(0, T, n_steps + 1)

        # GBM solution
        S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * B)

        return S

    @staticmethod
    def brownian_bridge(
        t: np.ndarray,
        a: float = 0.0,
        b: float = 0.0
    ) -> np.ndarray:
        """
        Generate Brownian bridge from (0,a) to (T,b).

        Brownian motion conditioned on B(0) = a and B(T) = b.
        """
        T = t[-1]
        n = len(t)

        # Generate standard Brownian motion
        B = BrownianMotion.generate_path(T, n-1, 1, 0.0)[0]

        # Bridge: B(t) - (t/T)B(T) + (1 - t/T)a + (t/T)b
        bridge = B - (t / T) * B[-1] + (1 - t / T) * a + (t / T) * b

        return bridge

    @staticmethod
    def first_passage_time(
        paths: np.ndarray,
        barrier: float,
        dt: float
    ) -> np.ndarray:
        """
        Compute first passage times to barrier.

        τ = inf{t : B(t) ≥ barrier}

        Args:
            paths: Brownian paths (n_paths × n_steps)
            barrier: Threshold level
            dt: Time step

        Returns:
            Array of first passage times for each path
        """
        n_paths = paths.shape[0]
        passage_times = np.full(n_paths, np.inf)

        for i in range(n_paths):
            crossing_indices = np.where(paths[i] >= barrier)[0]
            if len(crossing_indices) > 0:
                passage_times[i] = crossing_indices[0] * dt

        return passage_times


# ============================================================================
# MARTINGALES
# ============================================================================

class MartingaleAnalyzer:
    """
    Analyze martingale properties.

    Process M(t) is a martingale if:
    E[M(t) | ℱₛ] = M(s) for all t ≥ s
    (Expected future value = current value, given past)
    """

    @staticmethod
    def is_discrete_martingale(
        process: np.ndarray,
        tolerance: float = 0.1
    ) -> bool:
        """
        Check if discrete-time process is approximately martingale.

        Tests: E[X_{n+1} | X_n] ≈ X_n
        """
        # For each time step, check if expectation is preserved
        for i in range(len(process) - 1):
            # Compute conditional expectation (simple average)
            if abs(np.mean(process[i+1:]) - process[i]) > tolerance:
                return False

        return True

    @staticmethod
    def stopping_time(
        path: np.ndarray,
        condition: Callable[[float], bool]
    ) -> int:
        """
        Find stopping time: τ = inf{n : condition(Xₙ) = True}

        Args:
            path: Process path
            condition: Stopping condition

        Returns:
            Stopping time (index), or len(path) if never stops
        """
        for i, value in enumerate(path):
            if condition(value):
                return i

        return len(path)

    @staticmethod
    def optional_stopping_theorem(
        martingale: np.ndarray,
        stopping_time: int
    ) -> float:
        """
        Optional Stopping Theorem: E[M_τ] = E[M_0] for bounded τ.

        Returns value at stopping time.
        """
        if stopping_time >= len(martingale):
            stopping_time = len(martingale) - 1

        return martingale[stopping_time]


# ============================================================================
# ITO CALCULUS
# ============================================================================

class ItoIntegrator:
    """
    Ito stochastic integral: I(t) = ∫₀ᵗ f(s) dB(s)

    Properties:
    - E[I(t)] = 0
    - Var[I(t)] = ∫₀ᵗ f²(s) ds (Ito isometry)
    """

    @staticmethod
    def ito_integral(
        integrand: np.ndarray,
        brownian_increments: np.ndarray
    ) -> np.ndarray:
        """
        Compute Ito integral using Riemann sum approximation.

        ∫ f(t) dB(t) ≈ Σ f(tᵢ) ΔB(tᵢ)

        Args:
            integrand: f(t) values at time points
            brownian_increments: ΔB = B(t+dt) - B(t)

        Returns:
            Cumulative Ito integral
        """
        # Ito integral: left-point rule
        stochastic_increments = integrand[:-1] * brownian_increments

        # Cumulative sum
        integral = np.cumsum(np.concatenate([[0], stochastic_increments]))

        return integral

    @staticmethod
    def ito_isometry(
        integrand_squared: np.ndarray,
        dt: float
    ) -> float:
        """
        Ito isometry: E[|∫ f dB|²] = E[∫ f² dt]

        Compute variance of Ito integral.

        Args:
            integrand_squared: f²(t) values
            dt: Time step

        Returns:
            Variance: ∫ f²(t) dt
        """
        return np.sum(integrand_squared) * dt


class ItosLemma:
    """
    Ito's Lemma: Chain rule for stochastic calculus.

    If dX = μ dt + σ dB and F(t,x), then:
    dF = (∂F/∂t + μ ∂F/∂x + ½ σ² ∂²F/∂x²) dt + (σ ∂F/∂x) dB
    """

    @staticmethod
    def apply(
        F: Callable[[float, float], float],
        dF_dt: Callable[[float, float], float],
        dF_dx: Callable[[float, float], float],
        d2F_dx2: Callable[[float, float], float],
        X: np.ndarray,
        mu: float,
        sigma: float,
        dt: float,
        dB: np.ndarray
    ) -> np.ndarray:
        """
        Apply Ito's lemma to compute dF.

        Args:
            F: Function F(t, X)
            dF_dt: ∂F/∂t
            dF_dx: ∂F/∂x
            d2F_dx2: ∂²F/∂x²
            X: Process values
            mu: Drift of X
            sigma: Volatility of X
            dt: Time step
            dB: Brownian increments

        Returns:
            Increments dF
        """
        n = len(X) - 1
        t = np.arange(n+1) * dt

        dF = np.zeros(n)

        for i in range(n):
            # Drift term
            drift = (
                dF_dt(t[i], X[i]) +
                mu * dF_dx(t[i], X[i]) +
                0.5 * sigma**2 * d2F_dx2(t[i], X[i])
            ) * dt

            # Diffusion term
            diffusion = sigma * dF_dx(t[i], X[i]) * dB[i]

            dF[i] = drift + diffusion

        return dF


# ============================================================================
# STOCHASTIC DIFFERENTIAL EQUATIONS
# ============================================================================

@dataclass
class SDEResult:
    """Result of SDE simulation."""
    t: np.ndarray  # Time grid
    X: np.ndarray  # Solution paths
    method: str    # Numerical method used


class StochasticDifferentialEquation:
    """
    Solve SDEs: dX = μ(t, X) dt + σ(t, X) dB

    Numerical methods:
    - Euler-Maruyama: Simple explicit scheme
    - Milstein: Higher-order scheme using Ito's lemma
    """

    @staticmethod
    def euler_maruyama(
        mu: Callable[[float, float], float],
        sigma: Callable[[float, float], float],
        X0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> SDEResult:
        """
        Euler-Maruyama method for SDEs.

        X_{n+1} = X_n + μ(t_n, X_n) Δt + σ(t_n, X_n) ΔB_n

        Args:
            mu: Drift function μ(t, x)
            sigma: Diffusion function σ(t, x)
            X0: Initial value
            T: Final time
            n_steps: Number of time steps
            n_paths: Number of sample paths

        Returns:
            SDEResult with solution paths
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        # Initialize paths
        X = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = X0

        # Generate Brownian increments
        dB = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))

        # Euler-Maruyama iteration
        for i in range(n_steps):
            for j in range(n_paths):
                X[j, i+1] = (
                    X[j, i] +
                    mu(t[i], X[j, i]) * dt +
                    sigma(t[i], X[j, i]) * dB[j, i]
                )

        logger.info(f"Euler-Maruyama: {n_paths} paths, {n_steps} steps")

        return SDEResult(t=t, X=X, method="Euler-Maruyama")

    @staticmethod
    def milstein(
        mu: Callable[[float, float], float],
        sigma: Callable[[float, float], float],
        sigma_prime: Callable[[float, float], float],
        X0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> SDEResult:
        """
        Milstein method for SDEs (higher order).

        X_{n+1} = X_n + μΔt + σΔB + ½σσ'[(ΔB)² - Δt]

        Args:
            mu: Drift μ(t, x)
            sigma: Diffusion σ(t, x)
            sigma_prime: Derivative σ'(t, x) = ∂σ/∂x
            X0: Initial value
            T: Final time
            n_steps: Number of steps
            n_paths: Number of paths

        Returns:
            SDEResult with solution paths
        """
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        # Initialize
        X = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = X0

        # Brownian increments
        dB = np.random.normal(0, np.sqrt(dt), size=(n_paths, n_steps))

        # Milstein iteration
        for i in range(n_steps):
            for j in range(n_paths):
                mu_val = mu(t[i], X[j, i])
                sigma_val = sigma(t[i], X[j, i])
                sigma_prime_val = sigma_prime(t[i], X[j, i])

                X[j, i+1] = (
                    X[j, i] +
                    mu_val * dt +
                    sigma_val * dB[j, i] +
                    0.5 * sigma_val * sigma_prime_val * (dB[j, i]**2 - dt)
                )

        logger.info(f"Milstein: {n_paths} paths, {n_steps} steps")

        return SDEResult(t=t, X=X, method="Milstein")

    @staticmethod
    def ornstein_uhlenbeck(
        theta: float,
        mu: float,
        sigma: float,
        X0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1
    ) -> SDEResult:
        """
        Ornstein-Uhlenbeck process (mean-reverting).

        dX = θ(μ - X) dt + σ dB

        Has explicit solution:
        X(t) = μ + (X₀ - μ)e^(-θt) + σ∫₀ᵗ e^(-θ(t-s)) dB(s)

        Args:
            theta: Mean reversion speed
            mu: Long-term mean
            sigma: Volatility
            X0: Initial value
            T: Final time
            n_steps: Number of steps
            n_paths: Number of paths

        Returns:
            SDEResult
        """
        # Use exact solution
        dt = T / n_steps
        t = np.linspace(0, T, n_steps + 1)

        X = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = X0

        for i in range(n_steps):
            # Exact mean and variance of X(t+dt) given X(t)
            mean = mu + (X[:, i] - mu) * np.exp(-theta * dt)
            variance = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))

            X[:, i+1] = mean + np.sqrt(variance) * np.random.normal(size=n_paths)

        logger.info(f"Ornstein-Uhlenbeck: {n_paths} paths")

        return SDEResult(t=t, X=X, method="Exact (OU)")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'BrownianMotion',
    'MartingaleAnalyzer',
    'ItoIntegrator',
    'ItosLemma',
    'StochasticDifferentialEquation',
    'SDEResult'
]
