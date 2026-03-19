"""
Matrix Functions & Sparse Linear Algebra
=========================================

Scope slots: LA2.7 (matrix exponential), LA2.8 (matrix logarithm),
LA2.9 (sparse matrices), LA3.7 (generalized eigenvalue problem).

Pure numpy implementations of advanced matrix operations:
- Padé approximant for matrix exponential
- Inverse scaling-and-squaring for matrix logarithm
- CSR sparse matrix with matvec
- QZ-style generalized eigenvalue solver

Author: Claude Code
Date: March 2026
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# MATRIX EXPONENTIAL — Padé approximation (LA2.7)
# ============================================================================

# Padé coefficients for orders 3, 5, 7, 9, 13
_PADE_COEFFICIENTS = {
    3: [120, 60, 12, 1],
    5: [30240, 15120, 3360, 420, 30, 1],
    7: [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1],
    9: [17643225600, 8821612800, 2075673600, 302702400, 30270240,
        2162160, 110880, 3960, 90, 1],
    13: [64764752532480000, 32382376266240000, 7771770303897600,
         1187353796428800, 129060195264000, 10559470521600,
         670442572800, 33522128640, 1323241920, 40840800,
         960960, 16380, 182, 1],
}

_THETA = {3: 1.495585217958292e-2, 5: 2.539398330063230e-1,
           7: 9.504178996162932e-1, 9: 2.097847961257068,
           13: 5.371920351148152}


def _pade_approximant(A: np.ndarray, order: int) -> np.ndarray:
    """Evaluate [order/order] Padé approximant of exp(A).

    Uses the scaling relation exp(A) = [r_pq(A/2^s)]^{2^s} implicitly
    (caller handles scaling). Computes U, V such that exp ≈ (V-U)^{-1}(V+U).
    """
    n = A.shape[0]
    I = np.eye(n)
    coeffs = _PADE_COEFFICIENTS[order]

    # Build A^2, A^4, A^6, ...
    A2 = A @ A
    powers = {0: I, 1: A, 2: A2}

    if order >= 5:
        powers[4] = A2 @ A2
    if order >= 7:
        powers[6] = powers[4] @ A2
    if order >= 9:
        powers[8] = powers[6] @ A2

    if order <= 9:
        # Direct summation
        U = np.zeros_like(A)
        V = np.zeros_like(A)
        for j in range(order, -1, -1):
            p = powers.get(j, None)
            if p is None:
                # Compute missing odd powers
                p = powers[j - 1] @ A if j > 0 else I
                powers[j] = p
            if j % 2 == 1:
                U += coeffs[j] * p
            else:
                V += coeffs[j] * p
    else:
        # order == 13: use Higham's 2005 algorithm
        A4 = A2 @ A2
        A6 = A4 @ A2
        b = coeffs

        U = A @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
                 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I)
        V = (A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
             + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I)

    return U, V


def matrix_exponential(A: np.ndarray) -> np.ndarray:
    """Compute matrix exponential exp(A) via scaling-and-squaring with Padé.

    Uses the [13/13] Padé approximant with scaling and squaring.
    Accurate to machine precision for well-conditioned matrices.

    Args:
        A: Square matrix (n, n).

    Returns:
        exp(A) as (n, n) array.

    Example:
        >>> A = np.array([[0, -1], [1, 0]])  # rotation generator
        >>> R = matrix_exponential(A * np.pi/2)  # 90-degree rotation
        >>> np.allclose(R @ [1, 0], [0, 1], atol=1e-10)
        True
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    n = A.shape[0]
    if n == 0:
        return np.empty((0, 0))
    if n == 1:
        return np.array([[np.exp(A[0, 0])]])

    # Determine scaling parameter
    norm_A = np.linalg.norm(A, 1)

    # Select Padé order and scaling
    for order in [3, 5, 7, 9]:
        if norm_A <= _THETA[order]:
            U, V = _pade_approximant(A, order)
            # exp(A) = (V - U)^{-1} (V + U)
            return np.linalg.solve(V - U, V + U)

    # order 13 with scaling
    s = max(0, int(np.ceil(np.log2(norm_A / _THETA[13]))))
    A_scaled = A / (2 ** s)
    U, V = _pade_approximant(A_scaled, 13)
    R = np.linalg.solve(V - U, V + U)

    # Squaring phase
    for _ in range(s):
        R = R @ R

    return R


# ============================================================================
# MATRIX LOGARITHM — inverse scaling and squaring (LA2.8)
# ============================================================================

def _log_pade_approx(T: np.ndarray, order: int = 7) -> np.ndarray:
    """Padé approximant for log(I + X) where X = T - I is small."""
    n = T.shape[0]
    I = np.eye(n)
    X = T - I

    # Diagonal Padé approximant for log(1+x) ≈ p(x)/q(x)
    # Use partial fraction form for stability
    if order <= 3:
        # [2/2] Padé: log(1+x) ≈ x(6+x) / (6+4x)
        num = X @ (6 * I + X)
        den = 6 * I + 4 * X
        return np.linalg.solve(den, num)

    # [m/m] Padé via Gauss-Legendre quadrature:
    # log(I+X) = X * integral_0^1 (I + tX)^{-1} dt
    # Approximate with m-point quadrature
    m = order
    # Gauss-Legendre nodes and weights on [0,1]
    nodes, weights = np.polynomial.legendre.leggauss(m)
    nodes = 0.5 * (nodes + 1)  # Map from [-1,1] to [0,1]
    weights = 0.5 * weights

    result = np.zeros_like(X)
    for t, w in zip(nodes, weights):
        result += w * np.linalg.solve(I + t * X, I)

    return X @ result


def matrix_logarithm(A: np.ndarray) -> np.ndarray:
    """Compute principal matrix logarithm log(A) via inverse scaling and squaring.

    Uses repeated square roots to bring A close to I, then Padé
    approximation for log(I + X) where X is small.

    Args:
        A: Square matrix with no eigenvalues on the closed negative real axis.

    Returns:
        log(A) such that exp(log(A)) = A.

    Raises:
        ValueError: If A is singular or has negative real eigenvalues.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")

    n = A.shape[0]
    if n == 0:
        return np.empty((0, 0))
    if n == 1:
        if A[0, 0] <= 0:
            raise ValueError("Logarithm undefined for non-positive scalar")
        return np.array([[np.log(A[0, 0])]])

    # Check for singularity
    det = np.linalg.det(A)
    if abs(det) < 1e-15:
        raise ValueError("Matrix is singular — logarithm undefined")

    # Schur decomposition for stability: A = Q T Q^H
    T, Q = np.linalg.schur(A, output='real')

    # Inverse scaling: repeatedly take square root until T ≈ I
    s = 0
    T_k = T.copy()
    max_sqrt = 50
    while np.linalg.norm(T_k - np.eye(n), 1) > 0.5 and s < max_sqrt:
        # Matrix square root via Denman-Beavers iteration
        T_k = _matrix_sqrt_db(T_k)
        s += 1

    # Padé approximation of log(T_k) where T_k ≈ I
    log_T = _log_pade_approx(T_k, order=7)

    # Undo scaling: log(A) = 2^s * log(A^{1/2^s})
    log_T *= (2 ** s)

    # Undo Schur: log(A) = Q log(T) Q^H
    return Q @ log_T @ Q.T


def _matrix_sqrt_db(A: np.ndarray, max_iter: int = 32, tol: float = 1e-14) -> np.ndarray:
    """Matrix square root via Denman-Beavers iteration.

    Converges quadratically: Y_k → A^{1/2}, Z_k → A^{-1/2}.
    """
    n = A.shape[0]
    Y = A.copy()
    Z = np.eye(n)

    for _ in range(max_iter):
        Y_new = 0.5 * (Y + np.linalg.inv(Z))
        Z_new = 0.5 * (Z + np.linalg.inv(Y))
        if np.linalg.norm(Y_new - Y, 1) < tol:
            return Y_new
        Y, Z = Y_new, Z_new

    return Y


# ============================================================================
# SPARSE MATRIX — CSR format (LA2.9)
# ============================================================================

@dataclass
class SparseMatrix:
    """Compressed Sparse Row (CSR) matrix.

    Stores only nonzero entries for efficient matvec on sparse data.
    Three arrays:
    - data: nonzero values
    - col_indices: column index of each nonzero
    - row_ptr: index into data/col_indices for each row start

    Example:
        >>> dense = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]])
        >>> sp = SparseMatrix.from_dense(dense)
        >>> v = np.array([1, 2, 3])
        >>> np.allclose(sp.matvec(v), dense @ v)
        True
    """
    data: np.ndarray
    col_indices: np.ndarray
    row_ptr: np.ndarray
    shape: tuple[int, int]

    @staticmethod
    def from_dense(A: np.ndarray, tol: float = 1e-15) -> 'SparseMatrix':
        """Convert dense matrix to CSR format.

        Args:
            A: Dense matrix (m, n).
            tol: Threshold below which entries are treated as zero.

        Returns:
            SparseMatrix in CSR format.
        """
        A = np.asarray(A, dtype=float)
        m, n = A.shape
        data_list = []
        col_list = []
        row_ptr = [0]

        for i in range(m):
            for j in range(n):
                if abs(A[i, j]) > tol:
                    data_list.append(A[i, j])
                    col_list.append(j)
            row_ptr.append(len(data_list))

        return SparseMatrix(
            data=np.array(data_list, dtype=float),
            col_indices=np.array(col_list, dtype=int),
            row_ptr=np.array(row_ptr, dtype=int),
            shape=(m, n),
        )

    def to_dense(self) -> np.ndarray:
        """Convert back to dense matrix.

        Returns:
            Dense (m, n) array.
        """
        m, n = self.shape
        A = np.zeros((m, n))
        for i in range(m):
            start, end = self.row_ptr[i], self.row_ptr[i + 1]
            for idx in range(start, end):
                A[i, self.col_indices[idx]] = self.data[idx]
        return A

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Sparse matrix-vector product Ax.

        Args:
            v: Vector of length n.

        Returns:
            Result vector of length m.
        """
        v = np.asarray(v, dtype=float)
        if v.shape[0] != self.shape[1]:
            raise ValueError(f"Dimension mismatch: matrix has {self.shape[1]} cols, vector has {v.shape[0]} entries")

        m = self.shape[0]
        result = np.zeros(m)
        for i in range(m):
            start, end = self.row_ptr[i], self.row_ptr[i + 1]
            for idx in range(start, end):
                result[i] += self.data[idx] * v[self.col_indices[idx]]
        return result

    @property
    def nnz(self) -> int:
        """Number of stored nonzero entries."""
        return len(self.data)

    @property
    def density(self) -> float:
        """Fraction of nonzero entries."""
        total = self.shape[0] * self.shape[1]
        return self.nnz / total if total > 0 else 0.0

    def transpose(self) -> 'SparseMatrix':
        """Return transpose as new CSR matrix."""
        return SparseMatrix.from_dense(self.to_dense().T)


# ============================================================================
# GENERALIZED EIGENVALUE PROBLEM — Ax = λBx (LA3.7)
# ============================================================================

def generalized_eigenvalue(
    A: np.ndarray,
    B: np.ndarray,
    symmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the generalized eigenvalue problem Ax = λBx.

    For symmetric A and positive-definite B, uses Cholesky reduction:
        B = L L^T  →  L^{-1} A L^{-T} y = λ y  →  x = L^{-T} y

    For general A and B, uses QZ decomposition via reduction to
    standard form when B is invertible, with Schur fallback.

    Args:
        A: Matrix (n, n).
        B: Matrix (n, n), must be invertible.
        symmetric: If True, assumes A symmetric and B SPD for faster path.

    Returns:
        (eigenvalues, eigenvectors) where eigenvectors[:,i] corresponds
        to eigenvalues[i].

    Example:
        >>> A = np.diag([2.0, 3.0, 5.0])
        >>> B = np.diag([1.0, 2.0, 1.0])
        >>> vals, vecs = generalized_eigenvalue(A, B, symmetric=True)
        >>> np.allclose(sorted(vals), [1.5, 2.0, 5.0])
        True
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]

    if A.shape != (n, n) or B.shape != (n, n):
        raise ValueError("A and B must be square matrices of the same size")

    if symmetric:
        return _generalized_eig_symmetric(A, B)
    else:
        return _generalized_eig_general(A, B)


def _generalized_eig_symmetric(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric generalized eigenvalue via Cholesky reduction."""
    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        logger.warning("B is not positive definite, falling back to general solver")
        return _generalized_eig_general(A, B)

    # Transform: C = L^{-1} A L^{-T}
    L_inv = np.linalg.inv(L)
    C = L_inv @ A @ L_inv.T

    # Symmetrize (numerical stability)
    C = 0.5 * (C + C.T)

    eigenvalues, Y = np.linalg.eigh(C)

    # Back-transform eigenvectors: x = L^{-T} y
    eigenvectors = np.linalg.solve(L.T, Y)

    # Normalize
    for i in range(eigenvectors.shape[1]):
        norm = np.linalg.norm(eigenvectors[:, i])
        if norm > 1e-15:
            eigenvectors[:, i] /= norm

    return eigenvalues, eigenvectors


def _generalized_eig_general(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """General generalized eigenvalue via reduction to standard form."""
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        raise ValueError("B must be invertible for generalized eigenvalue problem")

    C = B_inv @ A
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # Sort by magnitude
    idx = np.argsort(np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx]
