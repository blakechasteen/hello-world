"""
Hamilton-Jacobi Theory
=======================

Scope slot: CA2.8 (Hamilton-Jacobi equation).

Pure numpy implementation:
- Hamilton-Jacobi PDE solver via Lax-Friedrichs / level set method
- Action functional (Hamilton's principal function)
- Classical mechanics connection: S(q, t) satisfies ∂S/∂t + H(q, ∂S/∂q) = 0

The Hamilton-Jacobi equation is the bridge between classical and quantum mechanics.
It transforms the equations of motion into a single PDE for the action S.

Author: Claude Code
Date: March 2026
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HJResult:
    """Result of Hamilton-Jacobi solve.

    Attributes:
        x: Spatial grid of shape (nx,).
        t: Time grid of shape (nt,).
        S: Action field S(x, t) of shape (nx, nt).
    """
    x: np.ndarray
    t: np.ndarray
    S: np.ndarray


class HamiltonJacobiSolver:
    """Solve the Hamilton-Jacobi equation via Lax-Friedrichs scheme.

    The Hamilton-Jacobi PDE:
        ∂S/∂t + H(x, ∂S/∂x, t) = 0

    where H is the Hamiltonian. This is a first-order nonlinear PDE
    that may develop shocks (non-smooth solutions), so we use a
    viscosity solution approach.

    The Lax-Friedrichs scheme adds numerical viscosity:
        S^{n+1}_j = ½(S^n_{j+1} + S^n_{j-1}) - dt · H(x_j, D⁰S^n_j, t_n)

    where D⁰ is the central difference operator.

    Example:
        >>> solver = HamiltonJacobiSolver()
        >>> # Free particle: H(x, p) = p²/2m
        >>> H = lambda x, p, t: 0.5 * p**2
        >>> # Initial condition: S(x, 0) = -|x|  (V-shaped)
        >>> S0 = lambda x: -np.abs(x)
        >>> result = solver.solve(H, S0, x_range=(-5, 5), t_range=(0, 2))
    """

    def solve(
        self,
        H: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        initial: Callable[[np.ndarray], np.ndarray],
        x_range: tuple[float, float],
        t_range: tuple[float, float],
        nx: int = 200,
        nt: int = 200,
        cfl: float = 0.5,
    ) -> HJResult:
        """Solve Hamilton-Jacobi PDE.

        Args:
            H: Hamiltonian H(x, p, t). Must handle array inputs.
            initial: Initial condition S(x, 0).
            x_range: (x_min, x_max) spatial domain.
            t_range: (t_start, t_end) time domain.
            nx: Number of spatial grid points.
            nt: Number of time steps.
            cfl: CFL number (< 1 for stability).

        Returns:
            HJResult with spatial grid, time grid, and action field.
        """
        x_min, x_max = x_range
        t_start, t_end = t_range

        x = np.linspace(x_min, x_max, nx)
        dx = x[1] - x[0]
        dt = (t_end - t_start) / nt
        t = np.linspace(t_start, t_end, nt + 1)

        # Store solution at all time steps
        S = np.zeros((nx, nt + 1))
        S[:, 0] = initial(x)

        for n in range(nt):
            S_curr = S[:, n]
            tn = t[n]

            # Central difference for p = ∂S/∂x
            p = np.zeros(nx)
            p[1:-1] = (S_curr[2:] - S_curr[:-2]) / (2 * dx)
            p[0] = (S_curr[1] - S_curr[0]) / dx
            p[-1] = (S_curr[-1] - S_curr[-2]) / dx

            # Lax-Friedrichs: average neighbors + evolution
            S_avg = np.zeros(nx)
            S_avg[1:-1] = 0.5 * (S_curr[2:] + S_curr[:-2])
            S_avg[0] = S_curr[0]
            S_avg[-1] = S_curr[-1]

            # Hamiltonian evaluation
            H_val = H(x, p, tn)

            # Update
            S[:, n + 1] = S_avg - dt * H_val

            # Boundary: extrapolate
            S[0, n + 1] = S[1, n + 1]
            S[-1, n + 1] = S[-2, n + 1]

        return HJResult(x=x, t=t, S=S)

    def solve_eikonal(
        self,
        speed: Callable[[np.ndarray], np.ndarray],
        source: np.ndarray,
        x_range: tuple[float, float],
        nx: int = 200,
    ) -> np.ndarray:
        """Solve eikonal equation |∇S| = 1/F(x) via fast sweeping.

        The eikonal equation is the stationary Hamilton-Jacobi equation:
            |∇S(x)| = 1/F(x)

        where F(x) is the speed function. S(x) gives the travel time
        from the source.

        Args:
            speed: Speed function F(x).
            source: Source points (where S = 0).
            x_range: (x_min, x_max).
            nx: Number of grid points.

        Returns:
            Travel time field S(x) of shape (nx,).
        """
        x_min, x_max = x_range
        x = np.linspace(x_min, x_max, nx)
        dx = x[1] - x[0]

        S = np.full(nx, np.inf)
        F = speed(x)

        # Set source points
        for src in np.atleast_1d(source):
            idx = int((src - x_min) / dx)
            idx = np.clip(idx, 0, nx - 1)
            S[idx] = 0.0

        # Fast sweeping: alternating forward/backward passes
        for sweep in range(4):
            if sweep % 2 == 0:
                indices = range(1, nx)
            else:
                indices = range(nx - 2, -1, -1)

            for i in indices:
                slowness = 1.0 / max(F[i], 1e-15)
                # One-sided update
                s_left = S[i - 1] if i > 0 else np.inf
                s_right = S[i + 1] if i < nx - 1 else np.inf
                s_min = min(s_left, s_right)
                s_new = s_min + slowness * dx
                S[i] = min(S[i], s_new)

        return S

    @staticmethod
    def action_functional(
        L: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        path: np.ndarray,
        dt: float,
    ) -> float:
        """Compute action S[q] = ∫₀ᵀ L(q, q̇, t) dt along a path.

        Hamilton's principal function. The classical trajectory
        extremizes this functional (Hamilton's principle / least action).

        Args:
            L: Lagrangian L(q, qdot, t). Handles array inputs.
            path: Discrete path q(t) of shape (n_steps+1,).
            dt: Time step.

        Returns:
            Action value (scalar).
        """
        n = len(path) - 1
        t = np.arange(n + 1) * dt

        # Velocity via central differences
        qdot = np.zeros(n + 1)
        qdot[1:-1] = (path[2:] - path[:-2]) / (2 * dt)
        qdot[0] = (path[1] - path[0]) / dt
        qdot[-1] = (path[-1] - path[-2]) / dt

        # Evaluate Lagrangian
        L_vals = L(path, qdot, t)

        # Trapezoidal integration
        action = np.trapz(L_vals, dx=dt)
        return float(action)

    @staticmethod
    def momentum_from_action(S: np.ndarray, dx: float) -> np.ndarray:
        """Compute canonical momentum p = ∂S/∂q from action field.

        Args:
            S: Action field S(x) of shape (nx,).
            dx: Grid spacing.

        Returns:
            Momentum field p(x) of shape (nx,).
        """
        p = np.zeros_like(S)
        p[1:-1] = (S[2:] - S[:-2]) / (2 * dx)
        p[0] = (S[1] - S[0]) / dx
        p[-1] = (S[-1] - S[-2]) / dx
        return p
