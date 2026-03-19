"""
Alternating Direction Method of Multipliers (ADMM) — AN12.9
============================================================

Distributed convex optimization via operator splitting. Decomposes problems
of the form minimize f(x) + g(z) subject to Ax = z into alternating
proximal updates with guaranteed convergence for convex objectives.

Core Algorithm:
    x-update: argmin_x f(x) + (rho/2)||Ax - z^k + u^k||^2
    z-update: argmin_z g(z) + (rho/2)||Ax^{k+1} - z + u^k||^2
    u-update: u^{k+1} = u^k + Ax^{k+1} - z^{k+1}

Classes:
    ADMM: Standard two-block splitting with augmented Lagrangian
    ConsensusADMM: Distributed consensus for multi-agent optimization

Applications:
    - Lasso (L1-regularized regression) via soft thresholding
    - Distributed model fitting across partitioned data
    - Constrained optimization with indicator functions
    - Basis pursuit, total variation denoising

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ADMMResult:
    """Result of ADMM optimization.

    Attributes:
        x: Primal variable (from f-subproblem)
        z: Consensus variable (from g-subproblem)
        dual: Scaled dual variable u
        converged: Whether primal/dual residuals fell below tolerance
        iterations: Number of ADMM iterations executed
        primal_residuals: History of ||Ax - z|| per iteration
        dual_residuals: History of ||rho * A^T(z^k - z^{k-1})|| per iteration
    """
    x: np.ndarray
    z: np.ndarray
    dual: np.ndarray
    converged: bool
    iterations: int
    primal_residuals: list[float] = field(default_factory=list)
    dual_residuals: list[float] = field(default_factory=list)


def soft_threshold(v: np.ndarray, kappa: float) -> np.ndarray:
    """Proximal operator for L1 norm: prox_{kappa * ||.||_1}(v)."""
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)


class ADMM:
    """
    Standard ADMM for problems: minimize f(x) + g(z) s.t. Ax - z = b.

    The augmented Lagrangian is:
        L_rho(x, z, u) = f(x) + g(z) + (rho/2)||Ax - z + u||^2

    where u is the scaled dual variable (u = y/rho).

    Convergence is guaranteed when f and g are closed, proper, convex
    and the unaugmented Lagrangian has a saddle point.
    """

    def solve(
        self,
        f_prox: Callable[[np.ndarray, float], np.ndarray],
        g_prox: Callable[[np.ndarray, float], np.ndarray],
        A: np.ndarray,
        b: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        rho: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        adaptive_rho: bool = False,
    ) -> ADMMResult:
        """
        Solve minimize f(x) + g(z) subject to Ax - z = b.

        Args:
            f_prox: Proximal operator for f. Takes (v, t) -> argmin_x f(x) + (1/2t)||x - v||^2
            g_prox: Proximal operator for g. Takes (v, t) -> argmin_z g(z) + (1/2t)||z - v||^2
            A: Linear constraint matrix, shape (m, n)
            b: Constraint offset, shape (m,). Defaults to zero.
            x0: Initial x, shape (n,). Defaults to zero.
            rho: Augmented Lagrangian penalty parameter
            max_iter: Maximum number of ADMM iterations
            tol: Convergence tolerance on primal and dual residuals
            adaptive_rho: If True, adapt rho based on residual balancing

        Returns:
            ADMMResult with optimal x, z, dual variables, convergence info
        """
        m, n = A.shape
        if b is None:
            b = np.zeros(m)
        if x0 is None:
            x0 = np.zeros(n)

        x = x0.copy()
        z = np.zeros(m)
        u = np.zeros(m)

        primal_residuals = []
        dual_residuals = []

        # Precompute for x-update when A^T A is invertible
        AtA = A.T @ A
        At = A.T

        converged = False
        for k in range(max_iter):
            z_old = z.copy()

            # x-update: minimize f(x) + (rho/2)||Ax - z + u||^2
            # Equivalent to prox_{f/rho}(x - (1/rho) A^T(Ax - z + u + b))
            # For general f, use the proximal operator
            v_x = x - (1.0 / rho) * At @ (A @ x - z + u + b)
            x = f_prox(v_x, 1.0 / rho)

            # z-update: minimize g(z) + (rho/2)||Ax - z + u||^2
            v_z = A @ x + u + b
            z = g_prox(v_z, 1.0 / rho)

            # u-update (dual ascent)
            r = A @ x - z + b
            u = u + r

            # Residuals
            primal_res = np.linalg.norm(r)
            dual_res = np.linalg.norm(rho * At @ (z - z_old))
            primal_residuals.append(primal_res)
            dual_residuals.append(dual_res)

            # Adaptive rho (residual balancing, Boyd et al. 2011)
            if adaptive_rho and k > 0:
                mu = 10.0
                tau = 2.0
                if primal_res > mu * dual_res:
                    rho *= tau
                    u /= tau
                elif dual_res > mu * primal_res:
                    rho /= tau
                    u *= tau

            if primal_res < tol and dual_res < tol:
                converged = True
                break

        return ADMMResult(
            x=x, z=z, dual=u,
            converged=converged,
            iterations=k + 1,
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
        )

    def solve_lasso(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lam: float = 1.0,
        rho: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> ADMMResult:
        """
        Solve Lasso: minimize (1/2)||Xw - y||^2 + lam * ||w||_1.

        Closed-form x-update via ridge regression, z-update via soft threshold.

        Args:
            X: Design matrix, shape (n_samples, n_features)
            y: Response vector, shape (n_samples,)
            lam: L1 regularization strength
            rho: ADMM penalty parameter
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            ADMMResult where x is the sparse coefficient vector
        """
        n, p = X.shape
        XtX = X.T @ X
        Xty = X.T @ y

        # Precompute factorization for x-update
        L = np.linalg.cholesky(XtX + rho * np.eye(p))

        x = np.zeros(p)
        z = np.zeros(p)
        u = np.zeros(p)

        primal_residuals = []
        dual_residuals = []
        converged = False

        for k in range(max_iter):
            z_old = z.copy()

            # x-update: (X^T X + rho I)^{-1} (X^T y + rho(z - u))
            rhs = Xty + rho * (z - u)
            x = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

            # z-update: soft thresholding
            z = soft_threshold(x + u, lam / rho)

            # u-update
            r = x - z
            u = u + r

            primal_res = np.linalg.norm(r)
            dual_res = np.linalg.norm(rho * (z - z_old))
            primal_residuals.append(primal_res)
            dual_residuals.append(dual_res)

            if primal_res < tol and dual_res < tol:
                converged = True
                break

        return ADMMResult(
            x=x, z=z, dual=u,
            converged=converged,
            iterations=k + 1,
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
        )

    def solve_basis_pursuit(
        self,
        A: np.ndarray,
        b: np.ndarray,
        rho: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> ADMMResult:
        """
        Solve basis pursuit: minimize ||x||_1 subject to Ax = b.

        Args:
            A: Measurement matrix, shape (m, n) with m < n
            b: Observation vector, shape (m,)
            rho: ADMM penalty parameter
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            ADMMResult where x is the sparsest solution
        """
        m, n = A.shape
        AAt = A @ A.T

        x = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)

        primal_residuals = []
        dual_residuals = []
        converged = False

        for k in range(max_iter):
            z_old = z.copy()

            # x-update: project onto Ax = b
            v = z - u
            x = v + A.T @ np.linalg.solve(AAt, b - A @ v)

            # z-update: soft thresholding
            z = soft_threshold(x + u, 1.0 / rho)

            # u-update
            r = x - z
            u = u + r

            primal_res = np.linalg.norm(r)
            dual_res = np.linalg.norm(rho * (z - z_old))
            primal_residuals.append(primal_res)
            dual_residuals.append(dual_res)

            if primal_res < tol and dual_res < tol:
                converged = True
                break

        return ADMMResult(
            x=x, z=z, dual=u,
            converged=converged,
            iterations=k + 1,
            primal_residuals=primal_residuals,
            dual_residuals=dual_residuals,
        )


class ConsensusADMM:
    """
    Consensus ADMM for distributed optimization.

    Solves: minimize sum_i f_i(x_i) subject to x_i = z for all i.

    Each agent i has a local objective f_i and maintains a local copy x_i.
    The consensus variable z enforces agreement. Updates are parallelizable
    across agents — only z-update and u-update require communication.
    """

    def solve(
        self,
        objectives: list[Callable[[np.ndarray], float]],
        proximal_ops: list[Callable[[np.ndarray, float], np.ndarray]],
        dim: int,
        rho: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """
        Solve distributed consensus optimization.

        Args:
            objectives: List of N local objective functions f_i(x)
            proximal_ops: List of N proximal operators for f_i
            dim: Dimension of the decision variable
            rho: ADMM penalty parameter
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Consensus solution z (average of converged local variables)
        """
        n_agents = len(objectives)
        x = [np.zeros(dim) for _ in range(n_agents)]
        u = [np.zeros(dim) for _ in range(n_agents)]
        z = np.zeros(dim)

        for k in range(max_iter):
            z_old = z.copy()

            # Local x-updates (parallelizable)
            for i in range(n_agents):
                v = z - u[i]
                x[i] = proximal_ops[i](v, 1.0 / rho)

            # Global z-update: average of (x_i + u_i)
            z = np.mean([x[i] + u[i] for i in range(n_agents)], axis=0)

            # Dual updates
            for i in range(n_agents):
                u[i] = u[i] + x[i] - z

            # Convergence check
            primal_res = np.sqrt(sum(np.linalg.norm(x[i] - z) ** 2
                                     for i in range(n_agents)))
            dual_res = np.linalg.norm(rho * n_agents * (z - z_old))

            if primal_res < tol and dual_res < tol:
                logger.info(f"ConsensusADMM converged in {k + 1} iterations")
                break

        return z

    def solve_distributed_regression(
        self,
        data_partitions: list[tuple[np.ndarray, np.ndarray]],
        lam: float = 0.0,
        rho: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """
        Distributed ridge regression across data partitions.

        Each agent i holds (X_i, y_i) and solves its local least squares.
        Consensus enforces a single shared parameter vector.

        Args:
            data_partitions: List of (X_i, y_i) data splits
            lam: Ridge regularization parameter
            rho: ADMM penalty parameter
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Consensus parameter vector
        """
        n_agents = len(data_partitions)
        dim = data_partitions[0][0].shape[1]

        # Precompute factorizations for each agent
        factors = []
        rhs_bases = []
        for X_i, y_i in data_partitions:
            XtX = X_i.T @ X_i + (lam + rho) * np.eye(dim)
            L = np.linalg.cholesky(XtX)
            factors.append(L)
            rhs_bases.append(X_i.T @ y_i)

        x = [np.zeros(dim) for _ in range(n_agents)]
        u = [np.zeros(dim) for _ in range(n_agents)]
        z = np.zeros(dim)

        for k in range(max_iter):
            z_old = z.copy()

            # Local x-updates
            for i in range(n_agents):
                rhs = rhs_bases[i] + rho * (z - u[i])
                x[i] = np.linalg.solve(
                    factors[i].T,
                    np.linalg.solve(factors[i], rhs)
                )

            # Global z-update
            z = np.mean([x[i] + u[i] for i in range(n_agents)], axis=0)

            # Dual updates
            for i in range(n_agents):
                u[i] = u[i] + x[i] - z

            primal_res = np.sqrt(sum(np.linalg.norm(x[i] - z) ** 2
                                     for i in range(n_agents)))
            dual_res = np.linalg.norm(rho * n_agents * (z - z_old))

            if primal_res < tol and dual_res < tol:
                break

        return z
