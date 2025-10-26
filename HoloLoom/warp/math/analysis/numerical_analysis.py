"""
Numerical Analysis for HoloLoom Warp Drive
==========================================

Practical numerical algorithms for computation.

Core Concepts:
- Root Finding: Solving f(x) = 0 numerically
- Numerical Linear Algebra: Matrix factorizations, iterative solvers
- ODE Solvers: Runge-Kutta methods, adaptive stepping
- Interpolation: Polynomial, spline interpolation
- Numerical Optimization: Gradient descent, Newton, quasi-Newton

Mathematical Foundation:
Root Finding: Newton's method: x_{n+1} = x_n - f(x_n)/f'(x_n)
ODE Solving: RK4: y_{n+1} = y_n + (k1 + 2k2 + 2k3 + k4)/6
Optimization: Gradient Descent: x_{n+1} = x_n - α∇f(x_n)

Applications to Warp Space:
- Neural network optimization (gradient descent)
- Embedding computation (eigenvalue problems)
- Time evolution of knowledge graphs (ODE solvers)
- Loss function minimization (optimization)

Author: HoloLoom Team
Date: 2025-10-26
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ROOT FINDING
# ============================================================================

class RootFinder:
    """
    Numerical root finding algorithms.

    Solve f(x) = 0 for scalar and vector functions.
    """

    @staticmethod
    def bisection(
        f: Callable[[float], float],
        a: float,
        b: float,
        tol: float = 1e-6,
        max_iter: int = 100
    ) -> Tuple[float, int]:
        """
        Bisection method for root finding.

        Requires f(a) and f(b) have opposite signs.

        Args:
            f: Function to find root of
            a, b: Interval endpoints
            tol: Tolerance
            max_iter: Maximum iterations

        Returns:
            (root, iterations)
        """
        fa = f(a)
        fb = f(b)

        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")

        for i in range(max_iter):
            c = (a + b) / 2
            fc = f(c)

            if abs(fc) < tol or abs(b - a) < tol:
                logger.info(f"Bisection converged in {i+1} iterations")
                return c, i + 1

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

        logger.warning("Bisection did not converge")
        return (a + b) / 2, max_iter

    @staticmethod
    def newton(
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        tol: float = 1e-6,
        max_iter: int = 100
    ) -> Tuple[float, int]:
        """
        Newton's method for root finding.

        x_{n+1} = x_n - f(x_n) / f'(x_n)

        Quadratic convergence when it works!
        """
        x = x0

        for i in range(max_iter):
            fx = f(x)
            dfx = df(x)

            if abs(fx) < tol:
                logger.info(f"Newton converged in {i+1} iterations")
                return x, i + 1

            if abs(dfx) < 1e-15:
                logger.warning("Derivative too small, Newton failed")
                return x, i + 1

            x = x - fx / dfx

        logger.warning("Newton did not converge")
        return x, max_iter

    @staticmethod
    def secant(
        f: Callable[[float], float],
        x0: float,
        x1: float,
        tol: float = 1e-6,
        max_iter: int = 100
    ) -> Tuple[float, int]:
        """
        Secant method (Newton without derivative).

        Uses finite difference approximation of derivative.
        """
        for i in range(max_iter):
            fx0 = f(x0)
            fx1 = f(x1)

            if abs(fx1) < tol:
                logger.info(f"Secant converged in {i+1} iterations")
                return x1, i + 1

            if abs(fx1 - fx0) < 1e-15:
                logger.warning("Division by zero in secant method")
                return x1, i + 1

            # Secant update
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            x0, x1 = x1, x_new

        logger.warning("Secant did not converge")
        return x1, max_iter

    @staticmethod
    def newton_multidim(
        f: Callable[[np.ndarray], np.ndarray],
        jacobian: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        tol: float = 1e-6,
        max_iter: int = 100
    ) -> Tuple[np.ndarray, int]:
        """
        Multidimensional Newton's method.

        x_{n+1} = x_n - J^{-1}(x_n) f(x_n)

        For solving systems f(x) = 0 where f: R^n -> R^n
        """
        x = x0.copy()

        for i in range(max_iter):
            fx = f(x)

            if np.linalg.norm(fx) < tol:
                logger.info(f"Multidim Newton converged in {i+1} iterations")
                return x, i + 1

            J = jacobian(x)

            # Solve J * delta = -f(x)
            try:
                delta = np.linalg.solve(J, -fx)
            except np.linalg.LinAlgError:
                logger.warning("Singular Jacobian")
                return x, i + 1

            x = x + delta

        logger.warning("Multidim Newton did not converge")
        return x, max_iter


# ============================================================================
# NUMERICAL LINEAR ALGEBRA
# ============================================================================

class NumericalLinearAlgebra:
    """
    Matrix factorizations and iterative solvers.
    """

    @staticmethod
    def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        LU decomposition: A = LU

        L is lower triangular, U is upper triangular.
        """
        n = A.shape[0]
        L = np.eye(n)
        U = A.copy()

        for k in range(n - 1):
            for i in range(k + 1, n):
                if abs(U[k, k]) < 1e-15:
                    continue

                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                U[i, k:] -= factor * U[k, k:]

        return L, U

    @staticmethod
    def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR decomposition using Gram-Schmidt: A = QR

        Q is orthogonal, R is upper triangular.
        """
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))

        for j in range(n):
            v = A[:, j].copy()

            for i in range(j):
                R[i, j] = Q[:, i] @ A[:, j]
                v -= R[i, j] * Q[:, i]

            R[j, j] = np.linalg.norm(v)

            if R[j, j] > 1e-15:
                Q[:, j] = v / R[j, j]
            else:
                Q[:, j] = v

        return Q, R

    @staticmethod
    def conjugate_gradient(
        A: np.ndarray,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        max_iter: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Conjugate Gradient method for solving Ax = b.

        Only works for symmetric positive definite A.

        Converges in at most n iterations (in exact arithmetic).
        """
        n = len(b)
        if max_iter is None:
            max_iter = n

        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()

        r = b - A @ x
        p = r.copy()
        rs_old = r @ r

        for i in range(max_iter):
            if np.sqrt(rs_old) < tol:
                logger.info(f"CG converged in {i} iterations")
                return x, i

            Ap = A @ p
            alpha = rs_old / (p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = r @ r

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        logger.warning(f"CG did not converge in {max_iter} iterations")
        return x, max_iter

    @staticmethod
    def power_iteration(
        A: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Tuple[float, np.ndarray]:
        """
        Power iteration for finding largest eigenvalue.

        Returns (eigenvalue, eigenvector).
        """
        n = A.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)

        eigenvalue = 0

        for i in range(max_iter):
            v_new = A @ v
            eigenvalue_new = np.linalg.norm(v_new)
            v_new = v_new / eigenvalue_new

            if abs(eigenvalue_new - eigenvalue) < tol:
                logger.info(f"Power iteration converged in {i+1} iterations")
                return eigenvalue_new, v_new

            eigenvalue = eigenvalue_new
            v = v_new

        return eigenvalue, v


# ============================================================================
# ODE SOLVERS
# ============================================================================

@dataclass
class ODESolution:
    """Result of ODE integration."""
    t: np.ndarray  # Time points
    y: np.ndarray  # Solution values
    method: str    # Solver method used


class ODESolver:
    """
    Numerical methods for ordinary differential equations.

    Solve dy/dt = f(t, y) with y(t0) = y0.
    """

    @staticmethod
    def euler(
        f: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        n_steps: int
    ) -> ODESolution:
        """
        Euler's method (first-order).

        y_{n+1} = y_n + h * f(t_n, y_n)

        Simple but only O(h) accurate.
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps

        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))
        y[0] = y0

        for i in range(n_steps):
            y[i + 1] = y[i] + h * f(t[i], y[i])

        logger.info(f"Euler method: {n_steps} steps")
        return ODESolution(t=t, y=y, method="Euler")

    @staticmethod
    def rk4(
        f: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        n_steps: int
    ) -> ODESolution:
        """
        Runge-Kutta 4th order (RK4).

        Classic method, O(h^4) accurate.

        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h*k1/2)
        k3 = f(t_n + h/2, y_n + h*k2/2)
        k4 = f(t_n + h, y_n + h*k3)
        y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps

        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))
        y[0] = y0

        for i in range(n_steps):
            k1 = f(t[i], y[i])
            k2 = f(t[i] + h/2, y[i] + h*k1/2)
            k3 = f(t[i] + h/2, y[i] + h*k2/2)
            k4 = f(t[i] + h, y[i] + h*k3)

            y[i + 1] = y[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

        logger.info(f"RK4 method: {n_steps} steps")
        return ODESolution(t=t, y=y, method="RK4")

    @staticmethod
    def rk45_adaptive(
        f: Callable[[float, np.ndarray], np.ndarray],
        t_span: Tuple[float, float],
        y0: np.ndarray,
        tol: float = 1e-6,
        h_init: float = 0.01,
        h_max: float = 0.1
    ) -> ODESolution:
        """
        Runge-Kutta-Fehlberg (RK45) with adaptive step size.

        Uses 5th and 4th order methods to estimate error.
        Adjusts step size to maintain tolerance.
        """
        t0, tf = t_span

        t_list = [t0]
        y_list = [y0]

        t = t0
        y = y0.copy()
        h = h_init

        while t < tf:
            if t + h > tf:
                h = tf - t

            # RK4 step
            k1 = f(t, y)
            k2 = f(t + h/2, y + h*k1/2)
            k3 = f(t + h/2, y + h*k2/2)
            k4 = f(t + h, y + h*k3)
            y_rk4 = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

            # RK5 step (using additional evaluation)
            k5 = f(t + h, y_rk4)
            y_rk5 = y + (h / 90) * (7*k1 + 32*k2 + 12*k3 + 32*k4 + 7*k5)

            # Error estimate
            error = np.linalg.norm(y_rk5 - y_rk4)

            if error < tol or h <= 1e-10:
                # Accept step
                t += h
                y = y_rk5
                t_list.append(t)
                y_list.append(y.copy())

                # Increase step size if error is very small
                if error < tol / 10 and h < h_max:
                    h = min(h * 1.5, h_max)
            else:
                # Reject step, decrease step size
                h = max(h * 0.5, 1e-10)

        logger.info(f"RK45 adaptive: {len(t_list)} steps")
        return ODESolution(
            t=np.array(t_list),
            y=np.array(y_list),
            method="RK45-adaptive"
        )


# ============================================================================
# INTERPOLATION
# ============================================================================

class Interpolation:
    """
    Polynomial and spline interpolation.
    """

    @staticmethod
    def lagrange(
        x_data: np.ndarray,
        y_data: np.ndarray,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Lagrange polynomial interpolation.

        P(x) = Σ y_i * L_i(x)
        where L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)
        """
        n = len(x_data)
        result = np.zeros_like(x)

        for i in range(n):
            # Compute Lagrange basis polynomial L_i
            L_i = np.ones_like(x)
            for j in range(n):
                if i != j:
                    L_i *= (x - x_data[j]) / (x_data[i] - x_data[j])

            result += y_data[i] * L_i

        return result

    @staticmethod
    def cubic_spline(
        x_data: np.ndarray,
        y_data: np.ndarray,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Natural cubic spline interpolation.

        Constructs piecewise cubic polynomials with continuous
        first and second derivatives.
        """
        from scipy.interpolate import CubicSpline

        # Use scipy for robust implementation
        cs = CubicSpline(x_data, y_data, bc_type='natural')
        return cs(x)

    @staticmethod
    def newton_divided_difference(
        x_data: np.ndarray,
        y_data: np.ndarray
    ) -> np.ndarray:
        """
        Compute Newton divided difference coefficients.

        Returns coefficients for Newton form of interpolating polynomial.
        """
        n = len(x_data)
        coeffs = np.zeros(n)
        coeffs[0] = y_data[0]

        # Divided difference table
        dd = y_data.copy()

        for i in range(1, n):
            for j in range(n - 1, i - 1, -1):
                dd[j] = (dd[j] - dd[j - 1]) / (x_data[j] - x_data[j - i])
            coeffs[i] = dd[i]

        return coeffs


# ============================================================================
# NUMERICAL OPTIMIZATION
# ============================================================================

class NumericalOptimization:
    """
    Optimization algorithms for minimizing functions.
    """

    @staticmethod
    def gradient_descent(
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Gradient descent optimization.

        x_{n+1} = x_n - α * ∇f(x_n)

        Returns (optimal_x, loss_history)
        """
        x = x0.copy()
        history = []

        for i in range(max_iter):
            fx = f(x)
            history.append(fx)

            grad = grad_f(x)

            if np.linalg.norm(grad) < tol:
                logger.info(f"Gradient descent converged in {i+1} iterations")
                return x, history

            x = x - learning_rate * grad

        logger.warning("Gradient descent did not converge")
        return x, history

    @staticmethod
    def bfgs(
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Tuple[np.ndarray, List[float]]:
        """
        BFGS quasi-Newton method.

        Builds approximation to Hessian inverse using gradient information.
        """
        from scipy.optimize import minimize

        result = minimize(f, x0, method='BFGS', jac=grad_f, tol=tol, options={'maxiter': max_iter})

        logger.info(f"BFGS: {result.nit} iterations, success={result.success}")
        return result.x, [result.fun]

    @staticmethod
    def adam(
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 1000
    ) -> Tuple[np.ndarray, int]:
        """
        Adam optimizer (adaptive moment estimation).

        Combines momentum and RMSprop.
        Popular for training neural networks.
        """
        x = x0.copy()
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment

        for t in range(1, max_iter + 1):
            grad = grad_f(x)

            # Update biased first moment
            m = beta1 * m + (1 - beta1) * grad

            # Update biased second moment
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Update parameters
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        logger.info(f"Adam: {max_iter} iterations")
        return x, max_iter


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'RootFinder',
    'NumericalLinearAlgebra',
    'ODESolver',
    'ODESolution',
    'Interpolation',
    'NumericalOptimization'
]
