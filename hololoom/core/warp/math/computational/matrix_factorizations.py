"""
Matrix Factorizations for HoloLoom Warp Drive
===============================================

LU decomposition (with partial pivoting), Cholesky decomposition,
Lanczos iteration (symmetric → tridiagonal), and Arnoldi iteration
(general → upper Hessenberg).

All implementations are pure numpy, no scipy dependency.

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# LU DECOMPOSITION
# ============================================================================

class LUDecomposition:
    """
    PA = LU decomposition with partial pivoting.

    L: lower triangular with unit diagonal
    U: upper triangular
    P: permutation matrix (row pivoting for numerical stability)

    Solves Ax = b via:
    1. PAx = Pb
    2. Ly = Pb (forward substitution)
    3. Ux = y (back substitution)
    """

    @staticmethod
    def factorize(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute PA = LU with partial pivoting.

        Args:
            A: Square matrix (n × n).

        Returns:
            (L, U, P): Lower triangular, upper triangular, permutation matrix.
        """
        A = np.array(A, dtype=float)
        n = A.shape[0]
        assert A.shape == (n, n), "Matrix must be square"

        U = A.copy()
        L = np.eye(n)
        P = np.eye(n)

        for k in range(n - 1):
            # Partial pivoting: find row with largest |U[i,k]| for i >= k
            max_row = k + np.argmax(np.abs(U[k:, k]))

            if abs(U[max_row, k]) < 1e-15:
                continue  # Skip zero pivot (singular matrix)

            if max_row != k:
                # Swap rows in U
                U[[k, max_row]] = U[[max_row, k]]
                # Swap rows in P
                P[[k, max_row]] = P[[max_row, k]]
                # Swap previous L entries
                if k > 0:
                    L[[k, max_row], :k] = L[[max_row, k], :k]

            # Elimination
            for i in range(k + 1, n):
                if abs(U[k, k]) < 1e-15:
                    continue
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] -= L[i, k] * U[k, k:]

        logger.info(f"LU factorization: {n}×{n} matrix")
        return L, U, P

    @staticmethod
    def solve(L: np.ndarray, U: np.ndarray, P: np.ndarray,
              b: np.ndarray) -> np.ndarray:
        """
        Solve Ax = b given PA = LU.

        Forward substitution Ly = Pb, then back substitution Ux = y.
        """
        n = L.shape[0]
        Pb = P @ b

        # Forward substitution: Ly = Pb
        y = np.zeros(n)
        for i in range(n):
            y[i] = Pb[i] - L[i, :i] @ y[:i]

        # Back substitution: Ux = y
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) < 1e-15:
                x[i] = 0.0
            else:
                x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

        return x

    @staticmethod
    def determinant(L: np.ndarray, U: np.ndarray, P: np.ndarray) -> float:
        """
        det(A) = det(P⁻¹) · det(L) · det(U) = (-1)^s · Π U[i,i]

        where s = number of row swaps (sign of permutation).
        """
        n = U.shape[0]
        det_U = np.prod(np.diag(U))

        # Count swaps in P
        perm = np.argmax(P, axis=1)
        swaps = 0
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                j = i
                cycle_len = 0
                while not visited[j]:
                    visited[j] = True
                    j = perm[j]
                    cycle_len += 1
                swaps += cycle_len - 1

        sign = (-1) ** swaps
        return sign * det_U


# ============================================================================
# CHOLESKY DECOMPOSITION
# ============================================================================

class CholeskyDecomposition:
    """
    Cholesky decomposition A = LLᵀ for symmetric positive definite matrices.

    L is lower triangular with positive diagonal entries.
    Exists iff A is symmetric positive definite.

    Twice as efficient as LU (exploits symmetry).
    Used for: solving SPD systems, sampling multivariate normals,
    matrix square roots.
    """

    @staticmethod
    def factorize(A: np.ndarray) -> np.ndarray:
        """
        Compute A = LLᵀ.

        Args:
            A: Symmetric positive definite matrix (n × n).

        Returns:
            L: Lower triangular matrix with A = LLᵀ.

        Raises:
            ValueError: If A is not symmetric positive definite.
        """
        A = np.array(A, dtype=float)
        n = A.shape[0]

        # Check symmetry
        if np.max(np.abs(A - A.T)) > 1e-10:
            raise ValueError("Matrix is not symmetric")

        L = np.zeros((n, n))

        for j in range(n):
            # Diagonal element
            val = A[j, j] - np.sum(L[j, :j] ** 2)
            if val <= 0:
                raise ValueError(
                    f"Matrix is not positive definite: "
                    f"negative value {val} at position ({j},{j})"
                )
            L[j, j] = np.sqrt(val)

            # Off-diagonal elements
            for i in range(j + 1, n):
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

        logger.info(f"Cholesky factorization: {n}×{n} SPD matrix")
        return L

    @staticmethod
    def solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve Ax = b given A = LLᵀ.

        Forward solve Ly = b, then backward solve Lᵀx = y.
        """
        n = L.shape[0]

        # Forward: Ly = b
        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

        # Backward: Lᵀx = y
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - L[i + 1:, i] @ x[i + 1:]) / L[i, i]

        return x

    @staticmethod
    def is_spd(A: np.ndarray) -> bool:
        """Check if matrix is symmetric positive definite."""
        if np.max(np.abs(A - A.T)) > 1e-10:
            return False
        try:
            CholeskyDecomposition.factorize(A)
            return True
        except ValueError:
            return False


# ============================================================================
# LANCZOS ITERATION
# ============================================================================

class LanczosIteration:
    """
    Lanczos algorithm: reduce symmetric matrix A to tridiagonal form T.

    A V = V T  (approximately)

    where V is orthonormal and T is tridiagonal. The eigenvalues of T
    approximate the extreme eigenvalues of A.

    Used for: large sparse eigenvalue problems, matrix functions (exp, sqrt),
    conjugate gradient method.

    Numerically, Lanczos suffers from loss of orthogonality.
    We implement full reorthogonalization for stability.
    """

    @staticmethod
    def compute(A: np.ndarray, v0: np.ndarray = None,
                k: int = None,
                reorthogonalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Lanczos iteration for symmetric matrix A.

        Args:
            A: Symmetric matrix (n × n).
            v0: Starting vector (default: random unit vector).
            k: Number of Lanczos steps (default: n).
            reorthogonalize: Full reorthogonalization for stability.

        Returns:
            (T, V): Tridiagonal matrix (k × k), Lanczos vectors (n × k).
        """
        n = A.shape[0]
        if k is None:
            k = n
        k = min(k, n)

        if v0 is None:
            rng = np.random.default_rng(42)
            v0 = rng.standard_normal(n)

        v0 = v0 / np.linalg.norm(v0)

        V = np.zeros((n, k))
        alpha = np.zeros(k)  # Diagonal
        beta = np.zeros(k)   # Off-diagonal

        V[:, 0] = v0
        w = A @ v0
        alpha[0] = w @ v0
        w = w - alpha[0] * v0

        for j in range(1, k):
            beta[j] = np.linalg.norm(w)

            if beta[j] < 1e-14:
                # Invariant subspace found
                logger.info(f"Lanczos: invariant subspace at step {j}")
                T = np.diag(alpha[:j]) + np.diag(beta[1:j], 1) + np.diag(beta[1:j], -1)
                return T, V[:, :j]

            V[:, j] = w / beta[j]

            # Full reorthogonalization (Gram-Schmidt against all previous)
            if reorthogonalize:
                for i in range(j):
                    V[:, j] -= (V[:, j] @ V[:, i]) * V[:, i]
                V[:, j] /= np.linalg.norm(V[:, j])

            w = A @ V[:, j]
            alpha[j] = w @ V[:, j]
            w = w - alpha[j] * V[:, j] - beta[j] * V[:, j - 1]

        # Build tridiagonal matrix
        T = np.diag(alpha) + np.diag(beta[1:], 1) + np.diag(beta[1:], -1)

        logger.info(f"Lanczos iteration: {n}×{n} → {k}×{k} tridiagonal")
        return T, V

    @staticmethod
    def eigenvalues(T: np.ndarray) -> np.ndarray:
        """
        Eigenvalues of the tridiagonal matrix (Ritz values).

        These approximate the extreme eigenvalues of A.
        """
        return np.sort(np.linalg.eigvalsh(T))


# ============================================================================
# ARNOLDI ITERATION
# ============================================================================

class ArnoldiIteration:
    """
    Arnoldi iteration: reduce general matrix A to upper Hessenberg form H.

    A V = V H + h_{k+1,k} v_{k+1} e_kᵀ  (approximately)

    Generalizes Lanczos to non-symmetric matrices.
    Foundation for GMRES and other Krylov methods.
    """

    @staticmethod
    def compute(A: np.ndarray, v0: np.ndarray = None,
                k: int = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Arnoldi iteration for general matrix A.

        Args:
            A: Matrix (n × n).
            v0: Starting vector.
            k: Number of Arnoldi steps.

        Returns:
            (H, V): Upper Hessenberg (k+1 × k), Arnoldi vectors (n × k+1).
        """
        n = A.shape[0]
        if k is None:
            k = min(n, 30)
        k = min(k, n)

        if v0 is None:
            rng = np.random.default_rng(42)
            v0 = rng.standard_normal(n)

        v0 = v0 / np.linalg.norm(v0)

        V = np.zeros((n, k + 1))
        H = np.zeros((k + 1, k))

        V[:, 0] = v0

        for j in range(k):
            w = A @ V[:, j]

            # Modified Gram-Schmidt orthogonalization
            for i in range(j + 1):
                H[i, j] = w @ V[:, i]
                w = w - H[i, j] * V[:, i]

            H[j + 1, j] = np.linalg.norm(w)

            if H[j + 1, j] < 1e-14:
                logger.info(f"Arnoldi: invariant subspace at step {j + 1}")
                return H[:j + 1, :j], V[:, :j + 1]

            V[:, j + 1] = w / H[j + 1, j]

        logger.info(f"Arnoldi iteration: {n}×{n} → {k + 1}×{k} Hessenberg")
        return H, V

    @staticmethod
    def eigenvalues(H: np.ndarray) -> np.ndarray:
        """
        Eigenvalues of the Hessenberg matrix (Ritz values).

        These approximate eigenvalues of A.
        """
        k = H.shape[1]
        H_square = H[:k, :k]
        return np.sort(np.linalg.eigvals(H_square))
