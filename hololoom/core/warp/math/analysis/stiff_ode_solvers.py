"""
Stiff ODE Solvers — BDF, Implicit Euler, Trapezoidal Rule
==========================================================

Implicit methods for stiff differential equations where explicit methods
require prohibitively small time steps.

Core Concepts:
- BDF (Backward Differentiation Formula): Multi-step implicit methods, orders 1-5
- Implicit Euler: First-order implicit, unconditionally stable
- Trapezoidal Rule: Second-order A-stable method (Crank-Nicolson)

Mathematical Foundation:
BDF-k: Σ αⱼ yₙ₋ⱼ = h β₀ f(tₙ, yₙ)
Implicit Euler: yₙ₊₁ = yₙ + h f(tₙ₊₁, yₙ₊₁)
Trapezoidal: yₙ₊₁ = yₙ + h/2 [f(tₙ, yₙ) + f(tₙ₊₁, yₙ₊₁)]

Applications to Warp Space:
- Stiff chemical kinetics in scoring term evolution
- Stability-critical embedding dynamics
- Warp tension fields with disparate time scales

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import fsolve as scipy_fsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ODEResult:
    """Result of an ODE solver.

    Attributes:
        t: Time points (n_steps,)
        y: Solution values (n_steps, n_dims)
        n_steps: Number of steps taken
        n_newton_iters: Total Newton iterations (implicit methods)
        success: Whether solver converged
        message: Status message
    """
    t: np.ndarray
    y: np.ndarray
    n_steps: int = 0
    n_newton_iters: int = 0
    success: bool = True
    message: str = ""


# ============================================================================
# NEWTON SOLVER (shared by implicit methods)
# ============================================================================

def _newton_solve(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 50,
    fd_eps: float = 1e-8,
) -> tuple[np.ndarray, int, bool]:
    """Newton iteration for implicit step.

    Uses finite-difference Jacobian when scipy is unavailable.

    Returns:
        (solution, n_iterations, converged)
    """
    x = x0.copy()
    n = len(x)

    for k in range(max_iter):
        F = residual_fn(x)

        if np.max(np.abs(F)) < tol:
            return x, k + 1, True

        # Finite-difference Jacobian
        J = np.zeros((n, n))
        for j in range(n):
            x_pert = x.copy()
            x_pert[j] += fd_eps
            J[:, j] = (residual_fn(x_pert) - F) / fd_eps

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return x, k + 1, False

        x += dx

    return x, max_iter, False


# ============================================================================
# BDF (Backward Differentiation Formulas)
# ============================================================================

# BDF coefficients: αⱼ for orders 1-5
# BDF-k: Σⱼ₌₀ᵏ αⱼ yₙ₋ⱼ = h β₀ f(tₙ, yₙ)
_BDF_ALPHA = {
    1: [1.0, -1.0],
    2: [3/2, -2.0, 1/2],
    3: [11/6, -3.0, 3/2, -1/3],
    4: [25/12, -4.0, 3.0, -4/3, 1/4],
    5: [137/60, -5.0, 5.0, -10/3, 5/4, -1/5],
}

_BDF_BETA = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
}


class BDF:
    """Backward Differentiation Formula solver for stiff ODEs.

    BDF methods are implicit multi-step methods. Orders 1-2 are A-stable,
    orders 3-5 are A(α)-stable. Higher orders converge faster but have
    narrower stability regions.

    Example:
        >>> def stiff_system(t, y):
        ...     return np.array([-1000*y[0] + y[1], y[0] - y[1]])
        >>> solver = BDF()
        >>> result = solver.solve(stiff_system, np.array([1.0, 0.0]),
        ...                       (0.0, 1.0), order=2, tol=1e-6)
    """

    def solve(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: tuple[float, float],
        order: int = 2,
        tol: float = 1e-6,
        dt: float | None = None,
        max_steps: int = 10000,
        newton_tol: float = 1e-10,
        newton_max_iter: int = 50,
    ) -> ODEResult:
        """Solve ODE using BDF method.

        Args:
            f: Right-hand side f(t, y)
            y0: Initial condition
            t_span: (t_start, t_end)
            order: BDF order (1-5)
            tol: Error tolerance for adaptive stepping
            dt: Fixed step size (if None, uses adaptive)
            max_steps: Maximum number of steps
            newton_tol: Newton iteration tolerance
            newton_max_iter: Max Newton iterations per step

        Returns:
            ODEResult with solution trajectory
        """
        if order < 1 or order > 5:
            raise ValueError(f"BDF order must be 1-5, got {order}")

        y0 = np.atleast_1d(np.asarray(y0, dtype=float))
        t0, tf = t_span

        if dt is None:
            dt = (tf - t0) / 100.0

        alpha = _BDF_ALPHA[order]
        beta = _BDF_BETA[order]

        # Bootstrap: use implicit Euler for first (order-1) steps
        t_list = [t0]
        y_list = [y0.copy()]
        total_newton = 0

        ie = ImplicitEuler()
        for i in range(order - 1):
            t_next = t0 + (i + 1) * dt
            if t_next > tf:
                break
            boot_result = ie.solve(f, y_list[-1], (t_list[-1], t_next), dt,
                                   newton_tol=newton_tol,
                                   newton_max_iter=newton_max_iter)
            t_list.append(t_next)
            y_list.append(boot_result.y[-1])
            total_newton += boot_result.n_newton_iters

        # Main BDF loop
        step = 0
        while t_list[-1] < tf - 1e-14 * abs(tf) and step < max_steps:
            t_n = t_list[-1]
            h = min(dt, tf - t_n)
            t_next = t_n + h

            # Build predictor from history
            k = min(order, len(y_list))
            alpha_k = _BDF_ALPHA[k]
            beta_k = _BDF_BETA[k]

            # Sum of α_j * y_{n-j+1} for j=1..k
            rhs_sum = np.zeros_like(y0)
            for j in range(1, k + 1):
                rhs_sum += alpha_k[j] * y_list[-j]

            def residual(y_new):
                return alpha_k[0] * y_new + rhs_sum - h * beta_k * f(t_next, y_new)

            # Initial guess: extrapolate
            y_pred = y_list[-1] + h * f(t_n, y_list[-1])

            y_new, n_iter, converged = _newton_solve(
                residual, y_pred, tol=newton_tol, max_iter=newton_max_iter
            )
            total_newton += n_iter

            if not converged:
                logger.warning(f"BDF Newton failed at t={t_next:.6f}")
                return ODEResult(
                    t=np.array(t_list), y=np.array(y_list),
                    n_steps=step, n_newton_iters=total_newton,
                    success=False, message=f"Newton diverged at t={t_next:.6f}"
                )

            t_list.append(t_next)
            y_list.append(y_new)
            step += 1

        return ODEResult(
            t=np.array(t_list), y=np.array(y_list),
            n_steps=step, n_newton_iters=total_newton,
            success=True, message=f"BDF-{order} completed in {step} steps"
        )


# ============================================================================
# IMPLICIT EULER
# ============================================================================

class ImplicitEuler:
    """Implicit (Backward) Euler method.

    First-order, unconditionally A-stable. The workhorse for extremely stiff
    systems where stability matters more than accuracy.

    yₙ₊₁ = yₙ + h f(tₙ₊₁, yₙ₊₁)

    Example:
        >>> result = ImplicitEuler().solve(
        ...     lambda t, y: -1000*y, np.array([1.0]), (0, 0.1), dt=0.01)
    """

    def solve(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: tuple[float, float],
        dt: float,
        newton_tol: float = 1e-10,
        newton_max_iter: int = 50,
    ) -> ODEResult:
        """Solve ODE using implicit Euler.

        Args:
            f: Right-hand side f(t, y)
            y0: Initial condition
            t_span: (t_start, t_end)
            dt: Step size
            newton_tol: Newton convergence tolerance
            newton_max_iter: Maximum Newton iterations per step

        Returns:
            ODEResult with solution trajectory
        """
        y0 = np.atleast_1d(np.asarray(y0, dtype=float))
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))

        t_arr = np.linspace(t0, tf, n_steps + 1)
        y_arr = np.zeros((n_steps + 1,) + y0.shape)
        y_arr[0] = y0

        total_newton = 0

        for i in range(n_steps):
            h = t_arr[i + 1] - t_arr[i]
            y_n = y_arr[i]
            t_next = t_arr[i + 1]

            def residual(y_new):
                return y_new - y_n - h * f(t_next, y_new)

            y_pred = y_n + h * f(t_arr[i], y_n)  # Explicit Euler predictor

            y_new, n_iter, converged = _newton_solve(
                residual, y_pred, tol=newton_tol, max_iter=newton_max_iter
            )
            total_newton += n_iter

            if not converged:
                return ODEResult(
                    t=t_arr[:i + 1], y=y_arr[:i + 1],
                    n_steps=i, n_newton_iters=total_newton,
                    success=False, message=f"Newton diverged at step {i}"
                )

            y_arr[i + 1] = y_new

        return ODEResult(
            t=t_arr, y=y_arr,
            n_steps=n_steps, n_newton_iters=total_newton,
            success=True, message=f"Implicit Euler completed in {n_steps} steps"
        )


# ============================================================================
# TRAPEZOIDAL RULE (Crank-Nicolson)
# ============================================================================

class TrapezoidalRule:
    """Trapezoidal rule (Crank-Nicolson) for ODEs.

    Second-order, A-stable. Balances accuracy and stability.

    yₙ₊₁ = yₙ + h/2 [f(tₙ, yₙ) + f(tₙ₊₁, yₙ₊₁)]

    Example:
        >>> result = TrapezoidalRule().solve(
        ...     lambda t, y: -y, np.array([1.0]), (0, 5), dt=0.1)
    """

    def solve(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        y0: np.ndarray,
        t_span: tuple[float, float],
        dt: float,
        newton_tol: float = 1e-10,
        newton_max_iter: int = 50,
    ) -> ODEResult:
        """Solve ODE using trapezoidal rule.

        Args:
            f: Right-hand side f(t, y)
            y0: Initial condition
            t_span: (t_start, t_end)
            dt: Step size
            newton_tol: Newton convergence tolerance
            newton_max_iter: Maximum Newton iterations per step

        Returns:
            ODEResult with solution trajectory
        """
        y0 = np.atleast_1d(np.asarray(y0, dtype=float))
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))

        t_arr = np.linspace(t0, tf, n_steps + 1)
        y_arr = np.zeros((n_steps + 1,) + y0.shape)
        y_arr[0] = y0

        total_newton = 0

        for i in range(n_steps):
            h = t_arr[i + 1] - t_arr[i]
            y_n = y_arr[i]
            t_n = t_arr[i]
            t_next = t_arr[i + 1]
            f_n = f(t_n, y_n)

            def residual(y_new, _fn=f_n):
                return y_new - y_n - 0.5 * h * (_fn + f(t_next, y_new))

            y_pred = y_n + h * f_n

            y_new, n_iter, converged = _newton_solve(
                residual, y_pred, tol=newton_tol, max_iter=newton_max_iter
            )
            total_newton += n_iter

            if not converged:
                return ODEResult(
                    t=t_arr[:i + 1], y=y_arr[:i + 1],
                    n_steps=i, n_newton_iters=total_newton,
                    success=False, message=f"Newton diverged at step {i}"
                )

            y_arr[i + 1] = y_new

        return ODEResult(
            t=t_arr, y=y_arr,
            n_steps=n_steps, n_newton_iters=total_newton,
            success=True, message=f"Trapezoidal rule completed in {n_steps} steps"
        )
