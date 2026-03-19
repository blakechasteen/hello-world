"""
Polynomial Algorithms for HoloLoom Warp Drive
===============================================

Groebner basis computation (Buchberger's algorithm), polynomial GCD,
Smith normal form, and Hermite normal form for integer matrices.

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# UNIVARIATE POLYNOMIAL (for GCD and related)
# ============================================================================

class UnivPoly:
    """
    Univariate polynomial over ℚ (float approximation).

    Stored as coefficient array: [a₀, a₁, ..., aₙ] = a₀ + a₁x + ... + aₙxⁿ.
    """

    def __init__(self, coeffs: np.ndarray):
        self.coeffs = np.array(coeffs, dtype=float)
        self._trim()

    def _trim(self):
        """Remove trailing near-zero coefficients."""
        while len(self.coeffs) > 1 and abs(self.coeffs[-1]) < 1e-14:
            self.coeffs = self.coeffs[:-1]

    @property
    def degree(self) -> int:
        if len(self.coeffs) == 1 and abs(self.coeffs[0]) < 1e-14:
            return -1  # Zero polynomial
        return len(self.coeffs) - 1

    def is_zero(self) -> bool:
        return self.degree < 0

    def leading_coefficient(self) -> float:
        return self.coeffs[-1] if not self.is_zero() else 0.0

    def monic(self) -> 'UnivPoly':
        """Return monic version (leading coefficient = 1)."""
        if self.is_zero():
            return UnivPoly(np.array([0.0]))
        return UnivPoly(self.coeffs / self.coeffs[-1])

    def __add__(self, other: 'UnivPoly') -> 'UnivPoly':
        n = max(len(self.coeffs), len(other.coeffs))
        a = np.zeros(n)
        a[:len(self.coeffs)] += self.coeffs
        a[:len(other.coeffs)] += other.coeffs
        return UnivPoly(a)

    def __sub__(self, other: 'UnivPoly') -> 'UnivPoly':
        n = max(len(self.coeffs), len(other.coeffs))
        a = np.zeros(n)
        a[:len(self.coeffs)] += self.coeffs
        a[:len(other.coeffs)] -= other.coeffs
        return UnivPoly(a)

    def __mul__(self, other: 'UnivPoly') -> 'UnivPoly':
        return UnivPoly(np.convolve(self.coeffs, other.coeffs))

    def scalar_mul(self, s: float) -> 'UnivPoly':
        return UnivPoly(self.coeffs * s)

    def divmod(self, other: 'UnivPoly') -> tuple['UnivPoly', 'UnivPoly']:
        """Polynomial division: self = quotient * other + remainder."""
        if other.is_zero():
            raise ValueError("Division by zero polynomial")

        if self.degree < other.degree:
            return UnivPoly(np.array([0.0])), UnivPoly(self.coeffs.copy())

        n = self.degree - other.degree + 1
        quot = np.zeros(n)
        rem = self.coeffs.copy()

        for i in range(n - 1, -1, -1):
            idx = i + other.degree
            if idx < len(rem):
                quot[i] = rem[idx] / other.coeffs[-1]
                for j in range(len(other.coeffs)):
                    if i + j < len(rem):
                        rem[i + j] -= quot[i] * other.coeffs[j]

        return UnivPoly(quot), UnivPoly(rem)

    def evaluate(self, x: float) -> float:
        """Horner's method."""
        result = 0.0
        for c in reversed(self.coeffs):
            result = result * x + c
        return result

    def __repr__(self):
        terms = []
        for i, c in enumerate(self.coeffs):
            if abs(c) < 1e-14:
                continue
            if i == 0:
                terms.append(f"{c:.4g}")
            elif i == 1:
                terms.append(f"{c:.4g}x")
            else:
                terms.append(f"{c:.4g}x^{i}")
        return " + ".join(terms) if terms else "0"


# ============================================================================
# GROEBNER COMPUTATION (univariate wrapper)
# ============================================================================

class GrobnerComputation:
    """
    Groebner basis for univariate polynomial ideals.

    For univariate polynomials, the ideal (f₁, ..., fₖ) = (gcd(f₁, ..., fₖ)),
    so the Groebner basis is just {gcd}.

    For multivariate, see commutative_algebra.GrobnerBasis.
    """

    @staticmethod
    def buchberger(polys: list[UnivPoly]) -> list[UnivPoly]:
        """
        Compute generating set for the ideal.

        For univariate: returns [gcd(polys)].
        """
        if not polys:
            return []

        result = polys[0]
        for p in polys[1:]:
            result = PolynomialGCD.euclidean(result, p)

        if not result.is_zero():
            result = result.monic()

        return [result]

    @staticmethod
    def reduced_basis(polys: list[UnivPoly]) -> list[UnivPoly]:
        """Reduced basis (monic gcd for univariate)."""
        basis = GrobnerComputation.buchberger(polys)
        return [p.monic() for p in basis if not p.is_zero()]


# ============================================================================
# POLYNOMIAL GCD
# ============================================================================

class PolynomialGCD:
    """
    GCD of univariate polynomials via the Euclidean algorithm.

    gcd(f, g) is the monic polynomial of highest degree dividing both f and g.
    """

    @staticmethod
    def euclidean(p: UnivPoly, q: UnivPoly) -> UnivPoly:
        """
        Euclidean algorithm: gcd(p, q).

        gcd(p, 0) = p
        gcd(p, q) = gcd(q, p mod q)
        """
        a, b = p, q

        max_iter = 1000
        iteration = 0

        while not b.is_zero() and iteration < max_iter:
            iteration += 1
            _, r = a.divmod(b)
            a, b = b, r

        if a.is_zero():
            return UnivPoly(np.array([1.0]))

        return a.monic()

    @staticmethod
    def extended_gcd(p: UnivPoly,
                     q: UnivPoly) -> tuple[UnivPoly, UnivPoly, UnivPoly]:
        """
        Extended Euclidean algorithm.

        Returns (g, s, t) where g = gcd(p, q) and s·p + t·q = g.
        """
        if q.is_zero():
            lc = p.leading_coefficient()
            if abs(lc) < 1e-14:
                return (UnivPoly(np.array([1.0])),
                        UnivPoly(np.array([1.0])),
                        UnivPoly(np.array([0.0])))
            return (p.monic(),
                    UnivPoly(np.array([1.0 / lc])),
                    UnivPoly(np.array([0.0])))

        old_r, r = p, q
        old_s, s = UnivPoly(np.array([1.0])), UnivPoly(np.array([0.0]))
        old_t, t = UnivPoly(np.array([0.0])), UnivPoly(np.array([1.0]))

        max_iter = 1000
        iteration = 0

        while not r.is_zero() and iteration < max_iter:
            iteration += 1
            quot, rem = old_r.divmod(r)
            old_r, r = r, rem
            old_s, s = s, old_s - quot * s
            old_t, t = t, old_t - quot * t

        # Normalize to monic
        lc = old_r.leading_coefficient()
        if abs(lc) > 1e-14:
            old_r = old_r.scalar_mul(1.0 / lc)
            old_s = old_s.scalar_mul(1.0 / lc)
            old_t = old_t.scalar_mul(1.0 / lc)

        return old_r, old_s, old_t


# ============================================================================
# SMITH NORMAL FORM
# ============================================================================

class SmithNormalForm:
    """
    Smith normal form of an integer matrix.

    For any m × n integer matrix A, there exist invertible integer matrices
    U (m × m) and V (n × n) such that S = U A V is diagonal with
    d₁ | d₂ | ... | dₖ (divisibility chain).

    S is the Smith normal form. The diagonal entries are the invariant factors.

    Applications: classification of finitely generated abelian groups,
    solving linear Diophantine equations.
    """

    @staticmethod
    def compute(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Smith normal form S = U A V.

        Args:
            matrix: Integer matrix (m × n).

        Returns:
            (S, U, V): Smith form, left transform, right transform.
                       S = U @ matrix @ V.
        """
        A = np.array(matrix, dtype=float).copy()
        m, n = A.shape

        U = np.eye(m)
        V = np.eye(n)

        min_dim = min(m, n)

        for k in range(min_dim):
            # Find pivot: smallest nonzero |A[i,j]| for i >= k, j >= k
            if not SmithNormalForm._find_and_move_pivot(A, U, V, k):
                continue  # All zeros in submatrix

            # Eliminate row k and column k
            changed = True
            max_iter = 100
            iteration = 0

            while changed and iteration < max_iter:
                changed = False
                iteration += 1

                # Eliminate column k entries below pivot
                for i in range(k + 1, m):
                    if abs(A[i, k]) > 0.5:
                        q = round(A[i, k] / A[k, k])
                        A[i, :] -= q * A[k, :]
                        U[i, :] -= q * U[k, :]
                        if abs(A[i, k]) > 0.5:
                            changed = True

                # Eliminate row k entries right of pivot
                for j in range(k + 1, n):
                    if abs(A[k, j]) > 0.5:
                        q = round(A[k, j] / A[k, k])
                        A[:, j] -= q * A[:, k]
                        V[:, j] -= q * V[:, k]
                        if abs(A[k, j]) > 0.5:
                            changed = True

            # Ensure divisibility: A[k,k] | A[i,j] for all i,j > k
            for i in range(k + 1, m):
                for j in range(k + 1, n):
                    if abs(A[i, j]) > 0.5 and abs(A[k, k]) > 0.5:
                        rem = round(A[i, j]) % round(A[k, k])
                        if abs(rem) > 0.5:
                            # Add row i to row k to fix divisibility
                            A[k, :] += A[i, :]
                            U[k, :] += U[i, :]

        # Round to integers
        S = np.round(A).astype(float)

        # Make diagonal entries positive
        for i in range(min_dim):
            if S[i, i] < -0.5:
                S[i, :] = -S[i, :]
                U[i, :] = -U[i, :]

        logger.info(f"Smith normal form: {m}×{n} matrix")
        return S, U, V

    @staticmethod
    def _find_and_move_pivot(A, U, V, k):
        """Find smallest nonzero entry in submatrix and move to (k,k)."""
        m, n = A.shape
        min_val = float('inf')
        min_i, min_j = -1, -1

        for i in range(k, m):
            for j in range(k, n):
                val = abs(round(A[i, j]))
                if val > 0 and val < min_val:
                    min_val = val
                    min_i, min_j = i, j

        if min_i == -1:
            return False

        # Swap rows
        if min_i != k:
            A[[k, min_i]] = A[[min_i, k]]
            U[[k, min_i]] = U[[min_i, k]]

        # Swap columns
        if min_j != k:
            A[:, [k, min_j]] = A[:, [min_j, k]]
            V[:, [k, min_j]] = V[:, [min_j, k]]

        return True

    @staticmethod
    def invariant_factors(matrix: np.ndarray) -> list[int]:
        """Extract nonzero invariant factors from Smith normal form."""
        S, _, _ = SmithNormalForm.compute(matrix)
        factors = []
        for i in range(min(S.shape)):
            d = abs(round(S[i, i]))
            if d > 0:
                factors.append(int(d))
        return factors


# ============================================================================
# HERMITE NORMAL FORM
# ============================================================================

class HermiteNormalForm:
    """
    Hermite normal form of an integer matrix.

    For an m × n integer matrix A with rank r, there exists a unimodular
    matrix U (m × m) such that H = UA is in Hermite normal form:

    - H is upper triangular (hij = 0 for i > j)
    - Pivot entries h_{i,πi} > 0
    - Entries above pivots satisfy 0 ≤ h_{j,πi} < h_{i,πi}

    Used for: integer linear programming, lattice problems,
    solving systems of linear Diophantine equations.
    """

    @staticmethod
    def compute(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Hermite normal form H = UA.

        Args:
            matrix: Integer matrix (m × n).

        Returns:
            (H, U): Hermite normal form and unimodular transform.
        """
        A = np.array(matrix, dtype=float).copy()
        m, n = A.shape
        U = np.eye(m)

        pivot_row = 0

        for j in range(n):
            if pivot_row >= m:
                break

            # Find nonzero entry in column j at or below pivot_row
            found = False
            for i in range(pivot_row, m):
                if abs(A[i, j]) > 0.5:
                    found = True
                    if i != pivot_row:
                        A[[pivot_row, i]] = A[[i, pivot_row]]
                        U[[pivot_row, i]] = U[[i, pivot_row]]
                    break

            if not found:
                continue

            # Make pivot positive
            if A[pivot_row, j] < -0.5:
                A[pivot_row, :] = -A[pivot_row, :]
                U[pivot_row, :] = -U[pivot_row, :]

            # Eliminate entries below pivot using GCD operations
            for i in range(pivot_row + 1, m):
                while abs(A[i, j]) > 0.5:
                    q = int(np.floor(A[i, j] / A[pivot_row, j]))
                    A[i, :] -= q * A[pivot_row, :]
                    U[i, :] -= q * U[pivot_row, :]

                    if abs(A[i, j]) > 0.5:
                        # Swap and try again (extended gcd step)
                        A[[pivot_row, i]] = A[[i, pivot_row]]
                        U[[pivot_row, i]] = U[[i, pivot_row]]
                        if A[pivot_row, j] < -0.5:
                            A[pivot_row, :] = -A[pivot_row, :]
                            U[pivot_row, :] = -U[pivot_row, :]

            # Reduce entries above pivot
            for i in range(pivot_row):
                if abs(A[i, j]) > 0.5 and abs(A[pivot_row, j]) > 0.5:
                    q = int(np.floor(A[i, j] / A[pivot_row, j]))
                    A[i, :] -= q * A[pivot_row, :]
                    U[i, :] -= q * U[pivot_row, :]

            pivot_row += 1

        H = np.round(A).astype(float)
        logger.info(f"Hermite normal form: {m}×{n} matrix")
        return H, U

    @staticmethod
    def lattice_basis(matrix: np.ndarray) -> np.ndarray:
        """
        Compute a canonical lattice basis from the HNF.

        Returns nonzero rows of the Hermite normal form.
        """
        H, _ = HermiteNormalForm.compute(matrix)
        nonzero_rows = []
        for i in range(H.shape[0]):
            if np.max(np.abs(H[i, :])) > 0.5:
                nonzero_rows.append(H[i, :])
        return np.array(nonzero_rows) if nonzero_rows else np.zeros((0, matrix.shape[1]))


# ============================================================================
# FINITE FIELD FACTORIZATION
# ============================================================================

class FiniteFieldPoly:
    """Polynomial over GF(p) (integers modulo p)."""

    def __init__(self, coeffs: np.ndarray, p: int):
        self.coeffs = np.array(coeffs, dtype=int) % p
        self.p = p
        self._trim()

    def _trim(self):
        while len(self.coeffs) > 1 and self.coeffs[-1] == 0:
            self.coeffs = self.coeffs[:-1]

    @property
    def degree(self) -> int:
        if len(self.coeffs) == 1 and self.coeffs[0] == 0:
            return -1
        return len(self.coeffs) - 1

    def is_zero(self) -> bool:
        return self.degree < 0

    def __add__(self, other: 'FiniteFieldPoly') -> 'FiniteFieldPoly':
        n = max(len(self.coeffs), len(other.coeffs))
        c = np.zeros(n, dtype=int)
        c[:len(self.coeffs)] += self.coeffs
        c[:len(other.coeffs)] += other.coeffs
        return FiniteFieldPoly(c % self.p, self.p)

    def __sub__(self, other: 'FiniteFieldPoly') -> 'FiniteFieldPoly':
        n = max(len(self.coeffs), len(other.coeffs))
        c = np.zeros(n, dtype=int)
        c[:len(self.coeffs)] += self.coeffs
        c[:len(other.coeffs)] -= other.coeffs
        return FiniteFieldPoly(c % self.p, self.p)

    def __mul__(self, other: 'FiniteFieldPoly') -> 'FiniteFieldPoly':
        c = np.convolve(self.coeffs, other.coeffs) % self.p
        return FiniteFieldPoly(c, self.p)

    def scalar_mul(self, s: int) -> 'FiniteFieldPoly':
        return FiniteFieldPoly((self.coeffs * s) % self.p, self.p)

    def monic(self) -> 'FiniteFieldPoly':
        if self.is_zero():
            return FiniteFieldPoly(np.array([0]), self.p)
        lc = int(self.coeffs[-1])
        inv = pow(lc, self.p - 2, self.p)  # Fermat's little theorem
        return self.scalar_mul(inv)

    def divmod(self, other: 'FiniteFieldPoly') -> tuple['FiniteFieldPoly', 'FiniteFieldPoly']:
        if other.is_zero():
            raise ValueError("Division by zero polynomial")
        if self.degree < other.degree:
            return FiniteFieldPoly(np.array([0]), self.p), FiniteFieldPoly(self.coeffs.copy(), self.p)
        n = self.degree - other.degree + 1
        quot = np.zeros(n, dtype=int)
        rem = self.coeffs.copy()
        lc_inv = pow(int(other.coeffs[-1]), self.p - 2, self.p)
        for i in range(n - 1, -1, -1):
            idx = i + other.degree
            if idx < len(rem):
                quot[i] = (rem[idx] * lc_inv) % self.p
                for j in range(len(other.coeffs)):
                    if i + j < len(rem):
                        rem[i + j] = (rem[i + j] - quot[i] * other.coeffs[j]) % self.p
        return FiniteFieldPoly(quot, self.p), FiniteFieldPoly(rem, self.p)

    def powmod(self, exp: int, mod: 'FiniteFieldPoly') -> 'FiniteFieldPoly':
        """Compute self^exp mod mod in GF(p)[x]."""
        result = FiniteFieldPoly(np.array([1]), self.p)
        base = FiniteFieldPoly(self.coeffs.copy(), self.p)
        while exp > 0:
            if exp % 2 == 1:
                result = result * base
                _, result = result.divmod(mod)
            base = base * base
            _, base = base.divmod(mod)
            exp //= 2
        return result

    def gcd(self, other: 'FiniteFieldPoly') -> 'FiniteFieldPoly':
        a, b = self, other
        for _ in range(1000):
            if b.is_zero():
                break
            _, r = a.divmod(b)
            a, b = b, r
        return a.monic() if not a.is_zero() else FiniteFieldPoly(np.array([1]), self.p)

    def __repr__(self):
        terms = []
        for i, c in enumerate(self.coeffs):
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{c}x")
            else:
                terms.append(f"{c}x^{i}")
        return " + ".join(terms) if terms else "0"


class FiniteFieldFactorization:
    """
    Factor polynomials over GF(p) using Cantor-Zassenhaus algorithm.

    Three stages:
    1. Square-free factorization: remove repeated factors
    2. Distinct-degree factorization: group factors by degree
    3. Equal-degree splitting: separate factors of the same degree
    """

    @staticmethod
    def factor(poly: FiniteFieldPoly) -> list['FiniteFieldPoly']:
        """
        Factor a polynomial over GF(p) into irreducible factors.

        Args:
            poly: Polynomial over GF(p).

        Returns:
            List of irreducible factors (monic).
        """
        if poly.degree <= 0:
            return [poly]

        p = poly.p
        f = poly.monic()

        # Step 1: Square-free factorization
        sq_free = FiniteFieldFactorization._square_free(f)

        # Step 2 & 3: Factor each square-free component
        factors = []
        for component in sq_free:
            if component.degree <= 0:
                continue
            ddf = FiniteFieldFactorization._distinct_degree(component)
            for d, g in ddf:
                if g.degree == d:
                    factors.append(g.monic())
                elif g.degree > d:
                    split = FiniteFieldFactorization._equal_degree_split(g, d)
                    factors.extend(split)

        return factors if factors else [f]

    @staticmethod
    def _square_free(f: FiniteFieldPoly) -> list[FiniteFieldPoly]:
        """Square-free decomposition."""
        p = f.p
        # Compute derivative
        deriv_coeffs = np.array(
            [(i * f.coeffs[i]) % p for i in range(1, len(f.coeffs))],
            dtype=int
        )
        if len(deriv_coeffs) == 0:
            deriv_coeffs = np.array([0])
        f_prime = FiniteFieldPoly(deriv_coeffs, p)

        if f_prime.is_zero():
            # f = g(x^p), recurse
            new_coeffs = f.coeffs[::p]
            g = FiniteFieldPoly(new_coeffs, p)
            sub_factors = FiniteFieldFactorization._square_free(g)
            return sub_factors  # Each factor has multiplicity p

        g = f.gcd(f_prime)
        if g.degree == 0:
            return [f]  # Already square-free

        _, w = f.divmod(g)
        return [w] + FiniteFieldFactorization._square_free(g)

    @staticmethod
    def _distinct_degree(f: FiniteFieldPoly) -> list[tuple[int, FiniteFieldPoly]]:
        """Distinct-degree factorization: group factors by degree."""
        p = f.p
        result = []
        h = f
        # x as polynomial
        x = FiniteFieldPoly(np.array([0, 1]), p)

        x_power = x  # Will hold x^{p^i}

        for d in range(1, h.degree + 1):
            if h.degree < 2 * d:
                if h.degree > 0:
                    result.append((h.degree, h))
                break

            # Compute x^{p^d} mod h
            x_power = x_power.powmod(p, h)

            # gcd(x^{p^d} - x, h)
            diff = x_power - x
            _, diff = diff.divmod(h)
            g = h.gcd(diff)

            if g.degree > 0:
                result.append((d, g))
                _, h = h.divmod(g)
                if h.is_zero():
                    break

        return result

    @staticmethod
    def _equal_degree_split(f: FiniteFieldPoly, d: int,
                            max_attempts: int = 50) -> list[FiniteFieldPoly]:
        """Cantor-Zassenhaus: split f into irreducible factors of degree d."""
        if f.degree == d:
            return [f.monic()]
        if f.degree <= 0:
            return []

        p = f.p

        for _ in range(max_attempts):
            # Random polynomial of degree < deg(f)
            r_coeffs = np.random.randint(0, p, f.degree)
            r = FiniteFieldPoly(r_coeffs, p)

            if r.is_zero():
                continue

            # Compute gcd(r^{(p^d - 1)/2} - 1, f) for odd p
            exp = (p ** d - 1) // 2
            g = r.powmod(exp, f)
            g = g - FiniteFieldPoly(np.array([1]), p)
            _, g = g.divmod(f)
            g = f.gcd(g)

            if 0 < g.degree < f.degree:
                _, other = f.divmod(g)
                left = FiniteFieldFactorization._equal_degree_split(g, d)
                right = FiniteFieldFactorization._equal_degree_split(other, d)
                return left + right

        logger.warning(f"Equal-degree split failed after {max_attempts} attempts")
        return [f.monic()]
