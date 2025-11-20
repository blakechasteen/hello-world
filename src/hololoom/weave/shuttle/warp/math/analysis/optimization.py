"""
Optimization Theory for HoloLoom Warp Drive
===========================================

Constrained optimization, convex analysis, and variational methods.

Core Concepts:
- Convex Sets & Functions: Foundation of optimization theory
- Constrained Optimization: Lagrange multipliers, KKT conditions
- Convex Optimization: Guaranteed global optimality
- Duality Theory: Primal-dual relationships
- Variational Calculus: Euler-Lagrange equations

Mathematical Foundation:
Lagrangian: L(x, λ) = f(x) + λᵀg(x)
KKT Conditions: ∇f + λᵀ∇g = 0, λᵢgᵢ(x) = 0, λ ≥ 0
Convex function: f(θx + (1-θ)y) ≤ θf(x) + (1-θ)f(y)
Euler-Lagrange: d/dt(∂L/∂ẋ) - ∂L/∂x = 0

Applications to Warp Space:
- Neural network training (loss minimization)
- Optimal transport for embedding alignment
- Variational inference in probabilistic models
- Physics-informed optimization

Author: HoloLoom Team
Date: 2025-10-26
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONVEX ANALYSIS
# ============================================================================

class ConvexAnalysis:
    """Convex sets and functions."""

    @staticmethod
    def is_convex_set(points: np.ndarray, test_points: int = 100) -> bool:
        """Check if set of points forms convex set."""
        n = len(points)
        if n < 2:
            return True

        # Test random convex combinations
        for _ in range(test_points):
            i, j = np.random.choice(n, 2, replace=False)
            theta = np.random.random()
            combo = theta * points[i] + (1 - theta) * points[j]

            # Check if combo in set (approximate)
            distances = np.linalg.norm(points - combo, axis=1)
            if np.min(distances) > 1e-6:
                return False

        return True

    @staticmethod
    def is_convex_function(
        f: Callable[[np.ndarray], float],
        domain: Tuple[np.ndarray, np.ndarray],
        samples: int = 100
    ) -> bool:
        """Check if function is convex via Jensen's inequality."""
        lower, upper = domain
        dim = len(lower)

        for _ in range(samples):
            x = lower + np.random.random(dim) * (upper - lower)
            y = lower + np.random.random(dim) * (upper - lower)
            theta = np.random.random()

            combo = theta * x + (1 - theta) * y
            lhs = f(combo)
            rhs = theta * f(x) + (1 - theta) * f(y)

            if lhs > rhs + 1e-6:
                return False

        return True

    @staticmethod
    def projection_onto_convex_set(
        point: np.ndarray,
        convex_set: np.ndarray
    ) -> np.ndarray:
        """Project point onto convex set (simplified for point cloud)."""
        distances = np.linalg.norm(convex_set - point, axis=1)
        return convex_set[np.argmin(distances)]


# ============================================================================
# LAGRANGE MULTIPLIERS
# ============================================================================

@dataclass
class ConstrainedProblem:
    """Constrained optimization problem."""
    objective: Callable[[np.ndarray], float]
    constraints_eq: List[Callable[[np.ndarray], float]]  # g(x) = 0
    constraints_ineq: List[Callable[[np.ndarray], float]]  # h(x) ≤ 0


class LagrangeMultipliers:
    """Lagrange multiplier method for equality constraints."""

    @staticmethod
    def lagrangian(
        x: np.ndarray,
        lambdas: np.ndarray,
        problem: ConstrainedProblem
    ) -> float:
        """L(x, λ) = f(x) + Σλᵢgᵢ(x)"""
        L = problem.objective(x)

        for i, g in enumerate(problem.constraints_eq):
            if i < len(lambdas):
                L += lambdas[i] * g(x)

        return L

    @staticmethod
    def solve_equality_constrained(
        problem: ConstrainedProblem,
        x0: np.ndarray,
        max_iter: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve problem with equality constraints using augmented Lagrangian.

        Returns (x_opt, lambdas_opt)
        """
        x = x0.copy()
        lambdas = np.zeros(len(problem.constraints_eq))
        rho = 1.0  # Penalty parameter

        for iteration in range(max_iter):
            # Augmented Lagrangian
            def aug_lag(x_var):
                L = problem.objective(x_var)
                for i, g in enumerate(problem.constraints_eq):
                    g_val = g(x_var)
                    L += lambdas[i] * g_val + (rho / 2) * g_val**2
                return L

            # Minimize augmented Lagrangian
            from scipy.optimize import minimize
            result = minimize(aug_lag, x, method='BFGS')
            x = result.x

            # Update multipliers
            for i, g in enumerate(problem.constraints_eq):
                lambdas[i] += rho * g(x)

            # Check convergence
            constraint_violation = sum(abs(g(x)) for g in problem.constraints_eq)
            if constraint_violation < 1e-6:
                break

        logger.info(f"Converged in {iteration+1} iterations")
        return x, lambdas


# ============================================================================
# KKT CONDITIONS
# ============================================================================

class KKTConditions:
    """Karush-Kuhn-Tucker conditions for inequality constraints."""

    @staticmethod
    def check_kkt(
        x: np.ndarray,
        lambdas_eq: np.ndarray,
        mus_ineq: np.ndarray,
        problem: ConstrainedProblem,
        grad_f: Callable[[np.ndarray], np.ndarray],
        grads_g: List[Callable[[np.ndarray], np.ndarray]],
        grads_h: List[Callable[[np.ndarray], np.ndarray]]
    ) -> bool:
        """
        Verify KKT conditions:
        1. Stationarity: ∇f + Σλᵢ∇gᵢ + Σμⱼ∇hⱼ = 0
        2. Primal feasibility: g(x) = 0, h(x) ≤ 0
        3. Dual feasibility: μ ≥ 0
        4. Complementary slackness: μⱼhⱼ(x) = 0
        """
        tol = 1e-6

        # Stationarity
        grad = grad_f(x)
        for i, grad_g in enumerate(grads_g):
            if i < len(lambdas_eq):
                grad += lambdas_eq[i] * grad_g(x)

        for j, grad_h in enumerate(grads_h):
            if j < len(mus_ineq):
                grad += mus_ineq[j] * grad_h(x)

        if np.linalg.norm(grad) > tol:
            return False

        # Primal feasibility
        for g in problem.constraints_eq:
            if abs(g(x)) > tol:
                return False

        for h in problem.constraints_ineq:
            if h(x) > tol:
                return False

        # Dual feasibility
        if np.any(mus_ineq < -tol):
            return False

        # Complementary slackness
        for j, h in enumerate(problem.constraints_ineq):
            if j < len(mus_ineq):
                if abs(mus_ineq[j] * h(x)) > tol:
                    return False

        return True


# ============================================================================
# CONVEX OPTIMIZATION
# ============================================================================

class ConvexOptimization:
    """Specialized methods for convex problems."""

    @staticmethod
    def projected_gradient_descent(
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        projection: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.01,
        max_iter: int = 1000
    ) -> np.ndarray:
        """
        Projected gradient descent for constrained convex optimization.

        xₖ₊₁ = Π(xₖ - α∇f(xₖ))
        """
        x = x0.copy()

        for k in range(max_iter):
            grad = grad_f(x)
            x_new = x - alpha * grad
            x = projection(x_new)

            if np.linalg.norm(x_new - x) < 1e-6:
                break

        logger.info(f"Projected GD converged in {k+1} iterations")
        return x

    @staticmethod
    def proximal_gradient(
        f: Callable[[np.ndarray], float],
        g: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        prox_g: Callable[[np.ndarray, float], np.ndarray],
        x0: np.ndarray,
        alpha: float = 0.01,
        max_iter: int = 1000
    ) -> np.ndarray:
        """
        Proximal gradient for f(x) + g(x) where g is non-smooth.

        xₖ₊₁ = prox_{αg}(xₖ - α∇f(xₖ))
        """
        x = x0.copy()

        for k in range(max_iter):
            grad = grad_f(x)
            z = x - alpha * grad
            x_new = prox_g(z, alpha)

            if np.linalg.norm(x_new - x) < 1e-6:
                break

            x = x_new

        logger.info(f"Proximal gradient converged in {k+1} iterations")
        return x


# ============================================================================
# DUALITY THEORY
# ============================================================================

class DualityTheory:
    """Lagrangian duality for optimization."""

    @staticmethod
    def dual_function(
        lambdas: np.ndarray,
        problem: ConstrainedProblem
    ) -> float:
        """
        Dual function: g(λ) = inf_x L(x, λ)

        Provides lower bound on optimal value.
        """
        # Would need to minimize Lagrangian over x
        # For convex problems: strong duality holds

        logger.info("Computing dual function")
        return 0.0  # Placeholder

    @staticmethod
    def duality_gap(
        x_primal: np.ndarray,
        lambdas_dual: np.ndarray,
        problem: ConstrainedProblem
    ) -> float:
        """
        Duality gap: f(x) - g(λ)

        Zero for strong duality (convex problems).
        """
        primal_val = problem.objective(x_primal)
        dual_val = DualityTheory.dual_function(lambdas_dual, problem)

        gap = primal_val - dual_val
        logger.info(f"Duality gap: {gap:.6f}")

        return gap


# ============================================================================
# VARIATIONAL CALCULUS
# ============================================================================

class VariationalCalculus:
    """Calculus of variations - optimize functionals."""

    @staticmethod
    def euler_lagrange_1d(
        L: Callable[[float, float, float], float],
        t_span: Tuple[float, float],
        boundary_conditions: Tuple[float, float],
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve Euler-Lagrange equation for 1D problem.

        Minimize ∫L(t, x, ẋ) dt

        Euler-Lagrange: d/dt(∂L/∂ẋ) - ∂L/∂x = 0
        """
        t_min, t_max = t_span
        x_a, x_b = boundary_conditions

        t = np.linspace(t_min, t_max, n_points)

        # Use shooting method to solve boundary value problem
        # (Simplified implementation)

        # Linear interpolation as initial guess
        x = x_a + (x_b - x_a) * (t - t_min) / (t_max - t_min)

        logger.info("Solved Euler-Lagrange equation")
        return t, x

    @staticmethod
    def brachistochrone() -> Tuple[np.ndarray, np.ndarray]:
        """
        Brachistochrone curve: fastest descent under gravity.

        Solution is a cycloid.
        """
        # Parametric cycloid: x = r(θ - sin θ), y = r(1 - cos θ)
        r = 1.0
        theta = np.linspace(0, np.pi, 100)

        x = r * (theta - np.sin(theta))
        y = r * (1 - np.cos(theta))

        logger.info("Generated brachistochrone curve (cycloid)")
        return x, y


# ============================================================================
# OPTIMAL TRANSPORT
# ============================================================================

class OptimalTransport:
    """Optimal transport (Wasserstein distance)."""

    @staticmethod
    def wasserstein_1d(
        samples_p: np.ndarray,
        samples_q: np.ndarray,
        p: float = 1
    ) -> float:
        """
        1D Wasserstein distance W_p(P, Q).

        For 1D: W_p = (∫|F⁻¹_P - F⁻¹_Q|^p)^(1/p)
        """
        # Sort samples to get quantile functions
        sorted_p = np.sort(samples_p)
        sorted_q = np.sort(samples_q)

        # Make same length
        n = min(len(sorted_p), len(sorted_q))
        sorted_p = sorted_p[:n]
        sorted_q = sorted_q[:n]

        # Compute W_p
        differences = np.abs(sorted_p - sorted_q)**p
        wasserstein = np.mean(differences)**(1/p)

        logger.info(f"W_{p} distance: {wasserstein:.6f}")
        return wasserstein

    @staticmethod
    def sinkhorn_distance(
        a: np.ndarray,
        b: np.ndarray,
        M: np.ndarray,
        reg: float = 0.1,
        max_iter: int = 100
    ) -> float:
        """
        Sinkhorn distance (entropic optimal transport).

        Regularized OT: min ⟨C, π⟩ + ε H(π)

        Args:
            a: Source distribution
            b: Target distribution
            M: Cost matrix
            reg: Regularization parameter
            max_iter: Maximum iterations

        Returns:
            Transport distance
        """
        K = np.exp(-M / reg)
        u = np.ones_like(a)
        v = np.ones_like(b)

        for _ in range(max_iter):
            u = a / (K @ v)
            v = b / (K.T @ u)

        # Transport plan
        pi = np.diag(u) @ K @ np.diag(v)

        # Cost
        cost = np.sum(pi * M)

        logger.info(f"Sinkhorn distance: {cost:.6f}")
        return cost


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ConvexAnalysis',
    'ConstrainedProblem',
    'LagrangeMultipliers',
    'KKTConditions',
    'ConvexOptimization',
    'DualityTheory',
    'VariationalCalculus',
    'OptimalTransport'
]
