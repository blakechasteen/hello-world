"""
Krylov Methods for HoloLoom Warp Drive
========================================

GMRES, BiCGSTAB, and multigrid solvers for sparse linear systems Ax = b.

Krylov subspace methods build solutions from the space
K_k(A, b) = span{b, Ab, A²b, ..., A^{k-1}b}.

These are the methods of choice for large sparse systems where
direct factorization is too expensive.

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# GMRES
# ============================================================================

class GMRES:
    """
    Generalized Minimal Residual method for solving Ax = b.

    GMRES minimizes ||b - Ax||₂ over the Krylov subspace K_k(A, r₀).

    Uses Arnoldi iteration to build an orthonormal basis for K_k,
    then solves a small (k × k) least squares problem.

    Works for any nonsingular matrix (does not require symmetry or
    positive definiteness).

    Restarted GMRES(m): restart after m iterations to bound memory.
    """

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray,
              x0: np.ndarray = None,
              tol: float = 1e-6,
              max_iter: int = 100,
              restart: int = None,
              preconditioner: Callable = None) -> tuple[np.ndarray, bool]:
        """
        Solve Ax = b using GMRES.

        Args:
            A: Matrix (n × n), can be dense or applied as function.
            b: Right-hand side vector.
            x0: Initial guess (default: zero).
            tol: Convergence tolerance on relative residual.
            max_iter: Maximum iterations.
            restart: Restart parameter (None = no restart).
            preconditioner: M⁻¹ application (left preconditioning).

        Returns:
            (x, converged): Solution and convergence flag.
        """
        n = len(b)
        if x0 is None:
            x0 = np.zeros(n)

        M_inv = preconditioner if preconditioner else lambda v: v
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-15:
            return np.zeros(n), True

        if restart is None:
            restart = max_iter

        x = x0.copy()

        for outer in range(max_iter // restart + 1):
            r = M_inv(b - A @ x)
            r_norm = np.linalg.norm(r)

            if r_norm / b_norm < tol:
                logger.info(f"GMRES converged: iter={outer * restart}, res={r_norm / b_norm:.2e}")
                return x, True

            # Arnoldi process
            m = min(restart, max_iter - outer * restart)
            V = np.zeros((n, m + 1))
            H = np.zeros((m + 1, m))

            V[:, 0] = r / r_norm
            beta = r_norm

            # Givens rotations for least squares
            cs = np.zeros(m)
            sn = np.zeros(m)
            e1 = np.zeros(m + 1)
            e1[0] = beta

            converged = False

            for j in range(m):
                # Arnoldi step
                w = M_inv(A @ V[:, j])

                for i in range(j + 1):
                    H[i, j] = w @ V[:, i]
                    w -= H[i, j] * V[:, i]

                H[j + 1, j] = np.linalg.norm(w)

                if H[j + 1, j] > 1e-14:
                    V[:, j + 1] = w / H[j + 1, j]

                # Apply previous Givens rotations
                for i in range(j):
                    temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                    H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                    H[i, j] = temp

                # Compute new Givens rotation
                r_val = np.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)
                if r_val > 1e-15:
                    cs[j] = H[j, j] / r_val
                    sn[j] = H[j + 1, j] / r_val
                else:
                    cs[j] = 1.0
                    sn[j] = 0.0

                H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
                H[j + 1, j] = 0.0

                e1[j + 1] = -sn[j] * e1[j]
                e1[j] = cs[j] * e1[j]

                residual = abs(e1[j + 1]) / b_norm

                if residual < tol:
                    # Solve upper triangular system
                    y = np.linalg.solve(H[:j + 1, :j + 1], e1[:j + 1])
                    x += V[:, :j + 1] @ y
                    logger.info(
                        f"GMRES converged: iter={outer * restart + j + 1}, "
                        f"res={residual:.2e}"
                    )
                    return x, True

            # End of restart cycle: solve least squares
            y = np.linalg.solve(H[:m, :m], e1[:m])
            x += V[:, :m] @ y

        # Did not converge
        final_res = np.linalg.norm(b - A @ x) / b_norm
        logger.warning(f"GMRES did not converge: res={final_res:.2e}")
        return x, False


# ============================================================================
# BiCGSTAB
# ============================================================================

class BiCGSTAB:
    """
    Bi-Conjugate Gradient Stabilized method for Ax = b.

    BiCGSTAB avoids the irregular convergence of standard BiCG
    by combining BiCG with GMRES(1) smoothing.

    Works for non-symmetric systems. No Krylov basis storage needed
    (short recurrence — only a few vectors at a time).

    Requires two matrix-vector products per iteration.
    """

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray,
              x0: np.ndarray = None,
              tol: float = 1e-6,
              max_iter: int = 100,
              preconditioner: Callable = None) -> tuple[np.ndarray, bool]:
        """
        Solve Ax = b using BiCGSTAB.

        Args:
            A: Matrix (n × n).
            b: Right-hand side.
            x0: Initial guess.
            tol: Convergence tolerance.
            max_iter: Maximum iterations.
            preconditioner: M⁻¹ application.

        Returns:
            (x, converged).
        """
        n = len(b)
        if x0 is None:
            x0 = np.zeros(n)

        M_inv = preconditioner if preconditioner else lambda v: v

        x = x0.copy()
        r = b - A @ x
        r_hat = r.copy()  # Shadow residual (fixed)

        b_norm = np.linalg.norm(b)
        if b_norm < 1e-15:
            return np.zeros(n), True

        rho_prev = 1.0
        alpha = 1.0
        omega = 1.0

        v = np.zeros(n)
        p = np.zeros(n)

        for iteration in range(max_iter):
            rho = r_hat @ r

            if abs(rho) < 1e-30:
                logger.warning("BiCGSTAB: breakdown (rho ≈ 0)")
                break

            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)

            # Preconditioned direction
            p_hat = M_inv(p)
            v = A @ p_hat

            alpha = rho / (r_hat @ v)

            s = r - alpha * v
            s_norm = np.linalg.norm(s)

            if s_norm / b_norm < tol:
                x += alpha * p_hat
                logger.info(f"BiCGSTAB converged (early): iter={iteration + 1}")
                return x, True

            # GMRES(1) step
            s_hat = M_inv(s)
            t = A @ s_hat

            t_dot_t = t @ t
            if abs(t_dot_t) < 1e-30:
                omega = 0.0
            else:
                omega = (t @ s) / t_dot_t

            x += alpha * p_hat + omega * s_hat
            r = s - omega * t

            r_norm = np.linalg.norm(r)
            if r_norm / b_norm < tol:
                logger.info(
                    f"BiCGSTAB converged: iter={iteration + 1}, "
                    f"res={r_norm / b_norm:.2e}"
                )
                return x, True

            if abs(omega) < 1e-30:
                logger.warning("BiCGSTAB: breakdown (omega ≈ 0)")
                break

            rho_prev = rho

        final_res = np.linalg.norm(b - A @ x) / b_norm
        logger.warning(f"BiCGSTAB did not converge: res={final_res:.2e}")
        return x, False


# ============================================================================
# MULTIGRID
# ============================================================================

class Multigrid:
    """
    Geometric multigrid V-cycle for solving Ax = b.

    Multigrid exploits the observation that iterative smoothers (Gauss-Seidel)
    are good at eliminating high-frequency error but slow for low-frequency.
    By restricting to coarser grids, low-frequency error becomes high-frequency
    and can be smoothed efficiently.

    V-cycle:
    1. Pre-smooth (ν₁ Gauss-Seidel iterations)
    2. Restrict residual to coarse grid
    3. Solve (recursively or directly) on coarse grid
    4. Prolongate correction to fine grid
    5. Post-smooth (ν₂ Gauss-Seidel iterations)

    Convergence: O(n) for elliptic PDEs (optimal!).
    """

    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray,
              levels: int = 3,
              nu1: int = 2,
              nu2: int = 2,
              max_cycles: int = 50,
              tol: float = 1e-6) -> np.ndarray:
        """
        Solve Ax = b using multigrid V-cycles.

        For demonstration, uses algebraic multigrid-like coarsening
        via simple aggregation.

        Args:
            A: SPD matrix (n × n).
            b: Right-hand side.
            levels: Number of grid levels.
            nu1, nu2: Pre/post-smoothing iterations.
            max_cycles: Maximum V-cycles.
            tol: Convergence tolerance.

        Returns:
            Solution x.
        """
        n = A.shape[0]
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-15:
            return np.zeros(n)

        # Build hierarchy
        operators = Multigrid._build_hierarchy(A, levels)

        x = np.zeros(n)

        for cycle in range(max_cycles):
            x = Multigrid._v_cycle(operators, b, x, 0, nu1, nu2)

            r_norm = np.linalg.norm(b - A @ x)
            if r_norm / b_norm < tol:
                logger.info(
                    f"Multigrid converged: {cycle + 1} V-cycles, "
                    f"res={r_norm / b_norm:.2e}"
                )
                return x

        final_res = np.linalg.norm(b - A @ x) / b_norm
        logger.warning(f"Multigrid: {max_cycles} cycles, res={final_res:.2e}")
        return x

    @staticmethod
    def _build_hierarchy(A: np.ndarray, levels: int) -> list:
        """
        Build multigrid hierarchy via simple aggregation coarsening.

        Returns list of (A_level, R_level, P_level) tuples.
        R = restriction, P = prolongation.
        """
        hierarchy = [{'A': A, 'R': None, 'P': None}]
        current_A = A

        for lev in range(1, levels):
            n = current_A.shape[0]
            if n <= 4:
                break

            # Simple coarsening: aggregate pairs of unknowns
            nc = (n + 1) // 2  # Coarse grid size

            # Prolongation: piecewise constant
            P = np.zeros((n, nc))
            for i in range(n):
                P[i, i // 2] = 1.0

            # Normalize columns
            for j in range(nc):
                col_sum = np.sum(P[:, j])
                if col_sum > 0:
                    P[:, j] /= col_sum

            # Restriction: R = Pᵀ (Galerkin)
            R = P.T

            # Coarse grid operator: A_c = R A P (Galerkin)
            A_coarse = R @ current_A @ P

            hierarchy.append({'A': A_coarse, 'R': R, 'P': P})
            current_A = A_coarse

        return hierarchy

    @staticmethod
    def _v_cycle(hierarchy: list, b: np.ndarray, x: np.ndarray,
                 level: int, nu1: int, nu2: int) -> np.ndarray:
        """Recursive V-cycle."""
        A = hierarchy[level]['A']

        # Coarsest level: direct solve
        if level == len(hierarchy) - 1 or A.shape[0] <= 4:
            try:
                return np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return x

        # Pre-smoothing
        x = Multigrid._gauss_seidel(A, b, x, nu1)

        # Compute residual
        r = b - A @ x

        # Restrict residual
        R = hierarchy[level + 1]['R']
        P = hierarchy[level + 1]['P']
        r_coarse = R @ r

        # Solve on coarse grid
        e_coarse = np.zeros(len(r_coarse))
        e_coarse = Multigrid._v_cycle(hierarchy, r_coarse, e_coarse,
                                       level + 1, nu1, nu2)

        # Prolongate and correct
        x += P @ e_coarse

        # Post-smoothing
        x = Multigrid._gauss_seidel(A, b, x, nu2)

        return x

    @staticmethod
    def _gauss_seidel(A: np.ndarray, b: np.ndarray,
                      x: np.ndarray, iterations: int) -> np.ndarray:
        """
        Gauss-Seidel smoother.

        x_i^{new} = (b_i - Σ_{j≠i} a_{ij} x_j) / a_{ii}
        """
        n = A.shape[0]
        x = x.copy()

        for _ in range(iterations):
            for i in range(n):
                if abs(A[i, i]) < 1e-15:
                    continue
                sigma = A[i, :] @ x - A[i, i] * x[i]
                x[i] = (b[i] - sigma) / A[i, i]

        return x

    @staticmethod
    def _jacobi(A: np.ndarray, b: np.ndarray,
                x: np.ndarray, iterations: int,
                omega: float = 2.0 / 3.0) -> np.ndarray:
        """
        Weighted Jacobi smoother (alternative to Gauss-Seidel).

        x^{new} = (1-ω)x + ω D⁻¹(b - (L+U)x)
        """
        D_inv = 1.0 / np.diag(A)
        x = x.copy()

        for _ in range(iterations):
            x_new = D_inv * (b - A @ x + np.diag(A) * x)
            x = (1 - omega) * x + omega * x_new

        return x
