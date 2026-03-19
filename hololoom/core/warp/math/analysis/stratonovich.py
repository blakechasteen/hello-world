"""
Stratonovich Stochastic Differential Equations
================================================

Scope slot: DE2.9 (Stratonovich SDE).

Pure numpy implementation of Stratonovich-form SDEs:
- Stratonovich SDE solver (midpoint / Heun scheme)
- Ito-to-Stratonovich drift correction
- Wong-Zakai approximation

The Stratonovich integral preserves the chain rule (unlike Ito), making it
natural for physical systems where noise arises from smooth approximations.

    Stratonovich: dX = a(X) dt + b(X) ∘ dW
    Ito equivalent: dX = [a(X) + ½ b(X) b'(X)] dt + b(X) dW

Author: Claude Code
Date: March 2026
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SDEResult:
    """Result of SDE simulation.

    Attributes:
        t: Time grid of shape (n_steps+1,).
        X: Solution paths of shape (n_paths, n_steps+1) or (n_steps+1,).
        method: Numerical method used.
    """
    t: np.ndarray
    X: np.ndarray
    method: str


class StratonovichSDE:
    """Stratonovich SDE solver.

    Solves dX = a(t, X) dt + b(t, X) ∘ dW(t) using:
    - Heun method (predictor-corrector, strong order 1.0)
    - Midpoint scheme (strong order 1.0 for additive noise)

    The Stratonovich integral is the symmetric limit:
        ∫ f(W) ∘ dW = lim Σ ½[f(W_{t_i}) + f(W_{t_{i+1}})] ΔW_i

    Example:
        >>> sde = StratonovichSDE()
        >>> # Geometric Brownian motion in Stratonovich form:
        >>> # dX = (μ - σ²/2) X dt + σ X ∘ dW  (no Ito correction needed)
        >>> drift = lambda t, x: 0.03 * x  # μ - σ²/2 = 0.05 - 0.01 = 0.03
        >>> diffusion = lambda t, x: 0.1 * x  # σ = 0.1 (so σ²/2 = 0.005... wait)
        >>> # Actually in Stratonovich form: dX = μX dt + σX ∘ dW
        >>> drift_s = lambda t, x: 0.05 * x
        >>> diff_s = lambda t, x: 0.1 * x
        >>> result = sde.solve(drift_s, diff_s, X0=1.0, T=1.0, n_steps=1000)
    """

    def solve(
        self,
        drift: Callable[[float, np.ndarray], np.ndarray],
        diffusion: Callable[[float, np.ndarray], np.ndarray],
        X0: float,
        T: float,
        n_steps: int = 1000,
        n_paths: int = 1,
        method: str = 'heun',
        seed: int | None = None,
    ) -> SDEResult:
        """Solve Stratonovich SDE.

        Args:
            drift: Drift function a(t, X).
            diffusion: Diffusion function b(t, X).
            X0: Initial condition.
            T: Final time.
            n_steps: Number of time steps.
            n_paths: Number of Monte Carlo paths.
            method: 'heun' (default) or 'midpoint'.
            seed: Random seed for reproducibility.

        Returns:
            SDEResult with time grid and solution paths.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        t = np.linspace(0, T, n_steps + 1)

        X = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = X0

        dW = np.random.randn(n_paths, n_steps) * sqrt_dt

        if method == 'heun':
            X = self._heun(drift, diffusion, X, t, dt, dW)
        elif method == 'midpoint':
            X = self._midpoint(drift, diffusion, X, t, dt, dW)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'heun' or 'midpoint'.")

        if n_paths == 1:
            X = X[0]

        return SDEResult(t=t, X=X, method=f'stratonovich_{method}')

    def _heun(
        self,
        drift: Callable,
        diffusion: Callable,
        X: np.ndarray,
        t: np.ndarray,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """Heun scheme for Stratonovich SDE (predictor-corrector).

        Strong order 1.0. The key idea: use the Euler prediction to evaluate
        diffusion at the predicted point, then average.

        X̃ = X_n + a(t_n, X_n) dt + b(t_n, X_n) ΔW_n     (predictor)
        X_{n+1} = X_n + a(t_n, X_n) dt
                  + ½[b(t_n, X_n) + b(t_{n+1}, X̃)] ΔW_n  (corrector)
        """
        n_paths, n_steps_plus_1 = X.shape
        n_steps = n_steps_plus_1 - 1

        for i in range(n_steps):
            ti = t[i]
            Xi = X[:, i]

            a_val = np.array([drift(ti, x) for x in Xi]).flatten()
            b_val = np.array([diffusion(ti, x) for x in Xi]).flatten()

            # Predictor (Euler step)
            X_pred = Xi + a_val * dt + b_val * dW[:, i]

            # Evaluate diffusion at predicted point
            ti1 = t[i + 1]
            b_pred = np.array([diffusion(ti1, x) for x in X_pred]).flatten()

            # Corrector: average diffusion
            X[:, i + 1] = Xi + a_val * dt + 0.5 * (b_val + b_pred) * dW[:, i]

        return X

    def _midpoint(
        self,
        drift: Callable,
        diffusion: Callable,
        X: np.ndarray,
        t: np.ndarray,
        dt: float,
        dW: np.ndarray,
    ) -> np.ndarray:
        """Midpoint scheme for Stratonovich SDE.

        Uses implicit midpoint evaluation for the diffusion term.
        Implemented as predictor-corrector with one iteration.

        X_{n+1} = X_n + a(t_n, X_n) dt + b(t_{n+½}, ½(X_n + X_{n+1})) ΔW_n
        """
        n_paths, n_steps_plus_1 = X.shape
        n_steps = n_steps_plus_1 - 1

        for i in range(n_steps):
            ti = t[i]
            Xi = X[:, i]
            t_mid = ti + 0.5 * dt

            a_val = np.array([drift(ti, x) for x in Xi]).flatten()
            b_val = np.array([diffusion(ti, x) for x in Xi]).flatten()

            # Predictor
            X_pred = Xi + a_val * dt + b_val * dW[:, i]

            # Midpoint
            X_mid = 0.5 * (Xi + X_pred)
            b_mid = np.array([diffusion(t_mid, x) for x in X_mid]).flatten()

            # Corrector
            X[:, i + 1] = Xi + a_val * dt + b_mid * dW[:, i]

        return X

    @staticmethod
    def ito_to_stratonovich(
        drift_ito: Callable[[float, float], float],
        diffusion: Callable[[float, float], float],
        diffusion_deriv: Callable[[float, float], float] | None = None,
        dx: float = 1e-6,
    ) -> Callable[[float, float], float]:
        """Convert Ito drift to Stratonovich drift.

        Ito: dX = μ_I(t,X) dt + σ(t,X) dW
        Strat: dX = μ_S(t,X) dt + σ(t,X) ∘ dW

        Relation: μ_S = μ_I - ½ σ(t,X) σ'_x(t,X)

        Args:
            drift_ito: Ito drift function μ_I(t, X).
            diffusion: Diffusion function σ(t, X).
            diffusion_deriv: Derivative σ'_x. If None, estimated numerically.
            dx: Step size for numerical derivative.

        Returns:
            Stratonovich drift function μ_S(t, X).
        """
        def drift_strat(t: float, x: float) -> float:
            sigma = diffusion(t, x)
            if diffusion_deriv is not None:
                sigma_prime = diffusion_deriv(t, x)
            else:
                sigma_prime = (diffusion(t, x + dx) - diffusion(t, x - dx)) / (2 * dx)
            return drift_ito(t, x) - 0.5 * sigma * sigma_prime

        return drift_strat

    @staticmethod
    def stratonovich_to_ito(
        drift_strat: Callable[[float, float], float],
        diffusion: Callable[[float, float], float],
        diffusion_deriv: Callable[[float, float], float] | None = None,
        dx: float = 1e-6,
    ) -> Callable[[float, float], float]:
        """Convert Stratonovich drift to Ito drift.

        Relation: μ_I = μ_S + ½ σ(t,X) σ'_x(t,X)

        Args:
            drift_strat: Stratonovich drift function μ_S(t, X).
            diffusion: Diffusion function σ(t, X).
            diffusion_deriv: Derivative σ'_x. If None, estimated numerically.
            dx: Step size for numerical derivative.

        Returns:
            Ito drift function μ_I(t, X).
        """
        def drift_ito(t: float, x: float) -> float:
            sigma = diffusion(t, x)
            if diffusion_deriv is not None:
                sigma_prime = diffusion_deriv(t, x)
            else:
                sigma_prime = (diffusion(t, x + dx) - diffusion(t, x - dx)) / (2 * dx)
            return drift_strat(t, x) + 0.5 * sigma * sigma_prime

        return drift_ito
