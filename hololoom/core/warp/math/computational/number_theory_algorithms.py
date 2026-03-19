"""
Number Theory Algorithms for HoloLoom Warp Drive
==================================================

Primality testing (Miller-Rabin), integer factorization (Pollard's rho),
discrete logarithm (baby-step giant-step), LLL lattice basis reduction,
and elliptic curve arithmetic.

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging
from math import gcd, isqrt

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# PRIMALITY TESTING
# ============================================================================

class PrimalityTesting:
    """
    Primality testing via Miller-Rabin (probabilistic) and
    deterministic checks for small numbers.

    Miller-Rabin: write n-1 = 2ˢ · d, then for random witness a:
    either aᵈ ≡ 1 (mod n) or a^{2ʳd} ≡ -1 (mod n) for some 0 ≤ r < s.

    If neither holds, n is composite (witness found).
    k rounds → probability of false positive ≤ 4⁻ᵏ.
    """

    @staticmethod
    def miller_rabin(n: int, k: int = 10) -> bool:
        """
        Miller-Rabin primality test.

        Args:
            n: Integer to test.
            k: Number of witnesses (rounds).

        Returns:
            True if probably prime, False if definitely composite.
        """
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False

        # Write n-1 = 2^s * d
        s, d = 0, n - 1
        while d % 2 == 0:
            d //= 2
            s += 1

        # Deterministic witnesses for n < 3.3×10²⁴
        if n < 2047:
            witnesses = [2]
        elif n < 1373653:
            witnesses = [2, 3]
        elif n < 3215031751:
            witnesses = [2, 3, 5, 7]
        elif n < 3317044064679887385961981:
            witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        else:
            # Random witnesses
            rng = np.random.default_rng(42)
            witnesses = [int(rng.integers(2, n - 1)) for _ in range(k)]

        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)

            if x == 1 or x == n - 1:
                continue

            composite = True
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    composite = False
                    break

            if composite:
                return False

        return True

    @staticmethod
    def is_prime(n: int) -> bool:
        """
        Deterministic primality test.

        Uses trial division for small n, Miller-Rabin with deterministic
        witnesses for larger n (correct up to 3.3×10²⁴).
        """
        if n < 2:
            return False
        if n < 4:
            return True

        # Trial division for small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False

        return PrimalityTesting.miller_rabin(n, k=20)

    @staticmethod
    def sieve_of_eratosthenes(limit: int) -> list[int]:
        """Generate all primes up to limit."""
        if limit < 2:
            return []

        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False

        for i in range(2, isqrt(limit) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False

        return [i for i in range(2, limit + 1) if is_prime[i]]


# ============================================================================
# INTEGER FACTORIZATION
# ============================================================================

class IntegerFactorization:
    """
    Integer factorization via Pollard's rho algorithm and trial division.

    Pollard's rho: expected O(n^{1/4}) for finding a factor.
    Uses Floyd's cycle detection on the sequence x_{i+1} = x_i² + c (mod n).
    """

    @staticmethod
    def trial_division(n: int) -> list[int]:
        """
        Factor n by trial division with small primes.

        Returns sorted list of prime factors (with multiplicity).
        """
        if n <= 1:
            return []

        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1 if d == 2 else 2

        if n > 1:
            factors.append(n)

        return factors

    @staticmethod
    def pollard_rho(n: int) -> list[int]:
        """
        Factor n using Pollard's rho algorithm.

        Returns sorted list of prime factors (with multiplicity).
        """
        if n <= 1:
            return []
        if PrimalityTesting.is_prime(n):
            return [n]
        if n % 2 == 0:
            factors = [2]
            remaining = n // 2
            while remaining % 2 == 0:
                factors.append(2)
                remaining //= 2
            if remaining > 1:
                factors.extend(IntegerFactorization.pollard_rho(remaining))
            return sorted(factors)

        factor = IntegerFactorization._rho_find_factor(n)
        if factor is None or factor == n:
            return [n]  # Could not factor (shouldn't happen for composites)

        factors = []
        factors.extend(IntegerFactorization.pollard_rho(factor))
        factors.extend(IntegerFactorization.pollard_rho(n // factor))
        return sorted(factors)

    @staticmethod
    def _rho_find_factor(n: int) -> int | None:
        """Find a non-trivial factor of n using Pollard's rho."""
        if n % 2 == 0:
            return 2

        for c in range(1, 100):
            x = 2
            y = 2
            d = 1

            f = lambda x: (x * x + c) % n

            max_iter = 100000
            iteration = 0

            while d == 1 and iteration < max_iter:
                iteration += 1
                x = f(x)
                y = f(f(y))
                d = gcd(abs(x - y), n)

            if 1 < d < n:
                return d

        return None  # Failed

    @staticmethod
    def factorization_dict(n: int) -> dict[int, int]:
        """Return factorization as {prime: exponent} dict."""
        factors = IntegerFactorization.pollard_rho(n)
        result = {}
        for p in factors:
            result[p] = result.get(p, 0) + 1
        return result


# ============================================================================
# DISCRETE LOGARITHM
# ============================================================================

class DiscreteLogarithm:
    """
    Discrete logarithm: find x such that g^x ≡ h (mod p).

    Baby-step giant-step algorithm: O(√p) time and space.

    Shanks' algorithm:
    1. m = ⌈√p⌉
    2. Baby steps: compute g^j for j = 0, 1, ..., m-1
    3. Giant steps: compute h · (g^{-m})^i for i = 0, 1, ...
    4. Match: g^j = h · g^{-mi} ⟹ x = im + j
    """

    @staticmethod
    def baby_giant(g: int, h: int, p: int) -> int:
        """
        Baby-step giant-step algorithm for discrete log.

        Finds x such that g^x ≡ h (mod p).

        Args:
            g: Base (generator).
            h: Target.
            p: Modulus (prime).

        Returns:
            x: Discrete logarithm, or -1 if not found.
        """
        m = isqrt(p) + 1

        # Baby step: build table {g^j mod p: j}
        table = {}
        power = 1
        for j in range(m):
            table[power] = j
            power = (power * g) % p

        # Giant step: g^{-m} mod p
        g_inv_m = pow(g, -m, p)  # Python 3.8+ supports negative exponents with mod

        # Search
        gamma = h
        for i in range(m):
            if gamma in table:
                x = (i * m + table[gamma]) % (p - 1)
                # Verify
                if pow(g, x, p) == h % p:
                    return x
            gamma = (gamma * g_inv_m) % p

        return -1  # Not found

    @staticmethod
    def pohlig_hellman(g: int, h: int, p: int) -> int:
        """
        Pohlig-Hellman algorithm for discrete log when p-1 has small factors.

        Reduces to discrete logs in subgroups of prime order.
        Only efficient when p-1 is smooth.
        """
        n = p - 1
        factors = IntegerFactorization.factorization_dict(n)

        # Solve discrete log modulo each prime power
        residues = []
        moduli = []

        for q, e in factors.items():
            q_e = q ** e
            # Reduce to subgroup of order q^e
            g_sub = pow(g, n // q_e, p)
            h_sub = pow(h, n // q_e, p)

            # Solve in subgroup (baby-step giant-step)
            x_q = DiscreteLogarithm.baby_giant(g_sub, h_sub, p)
            if x_q == -1:
                return -1

            residues.append(x_q % q_e)
            moduli.append(q_e)

        # Chinese Remainder Theorem
        return DiscreteLogarithm._crt(residues, moduli)

    @staticmethod
    def _crt(residues: list[int], moduli: list[int]) -> int:
        """Chinese Remainder Theorem: find x with x ≡ rᵢ (mod mᵢ)."""
        if not residues:
            return 0

        x = residues[0]
        m = moduli[0]

        for i in range(1, len(residues)):
            # Solve x ≡ x (mod m) and x ≡ residues[i] (mod moduli[i])
            g = gcd(m, moduli[i])
            if (residues[i] - x) % g != 0:
                return -1  # No solution

            lcm = m * moduli[i] // g
            # Extended gcd to find combination
            _, s, _ = DiscreteLogarithm._extended_gcd(m // g, moduli[i] // g)
            x = x + m * ((residues[i] - x) // g) * s
            x = x % lcm
            m = lcm

        return int(x)

    @staticmethod
    def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
        if b == 0:
            return a, 1, 0
        g, s, t = DiscreteLogarithm._extended_gcd(b, a % b)
        return g, t, s - (a // b) * t


# ============================================================================
# LLL LATTICE BASIS REDUCTION
# ============================================================================

class LLLReduction:
    """
    Lenstra-Lenstra-Lovász (LLL) lattice basis reduction.

    Given a basis B = {b₁, ..., bₙ} of a lattice L ⊂ ℝⁿ,
    LLL produces a "reduced" basis where vectors are short and nearly orthogonal.

    An LLL-reduced basis satisfies:
    1. |μᵢⱼ| ≤ 1/2 for all i > j (size reduction)
    2. δ|b*ᵢ|² ≤ |b*ᵢ₊₁ + μᵢ₊₁,ᵢ b*ᵢ|² (Lovász condition)

    where b* is the Gram-Schmidt orthogonalization and δ ∈ (1/4, 1].

    Applications: breaking knapsack cryptosystems, finding short vectors,
    polynomial factorization, integer programming.
    """

    @staticmethod
    def reduce(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
        """
        LLL reduction of a lattice basis.

        Args:
            basis: Basis vectors as rows of matrix (n × m).
            delta: Lovász parameter (0.25 < δ ≤ 1). Default 0.75.

        Returns:
            Reduced basis (rows).
        """
        B = np.array(basis, dtype=float).copy()
        n = B.shape[0]

        # Gram-Schmidt (without normalization)
        B_star = B.copy()
        mu = np.zeros((n, n))

        def gram_schmidt_update():
            for i in range(n):
                B_star[i] = B[i].copy()
                for j in range(i):
                    norm_sq = np.dot(B_star[j], B_star[j])
                    if norm_sq < 1e-14:
                        mu[i, j] = 0
                    else:
                        mu[i, j] = np.dot(B[i], B_star[j]) / norm_sq
                    B_star[i] -= mu[i, j] * B_star[j]

        gram_schmidt_update()

        k = 1
        max_iter = 10000
        iteration = 0

        while k < n and iteration < max_iter:
            iteration += 1

            # Size reduction
            for j in range(k - 1, -1, -1):
                if abs(mu[k, j]) > 0.5:
                    r = round(mu[k, j])
                    B[k] -= r * B[j]
                    gram_schmidt_update()

            # Lovász condition
            norm_k = np.dot(B_star[k], B_star[k])
            norm_k_minus_1 = np.dot(B_star[k - 1], B_star[k - 1])

            if norm_k >= (delta - mu[k, k - 1] ** 2) * norm_k_minus_1:
                k += 1
            else:
                # Swap b_k and b_{k-1}
                B[[k, k - 1]] = B[[k - 1, k]]
                gram_schmidt_update()
                k = max(k - 1, 1)

        logger.info(f"LLL reduction: {n} vectors in {iteration} iterations")
        return B

    @staticmethod
    def shortest_vector_estimate(basis: np.ndarray) -> np.ndarray:
        """
        Approximate shortest vector: first vector of LLL-reduced basis.

        LLL guarantees |b₁| ≤ 2^{(n-1)/2} λ₁(L) where λ₁ is the
        shortest vector length.
        """
        reduced = LLLReduction.reduce(basis)
        return reduced[0]


# ============================================================================
# ELLIPTIC CURVE
# ============================================================================

class EllipticCurve:
    """
    Elliptic curve E: y² = x³ + ax + b over 𝔽_p.

    Condition: 4a³ + 27b² ≢ 0 (mod p) (non-singular).

    Group law: points on E form an abelian group under geometric addition.
    Identity = point at infinity O.

    Applications: cryptography (ECDSA, ECDH), factorization (ECM),
    coding theory, number theory.
    """

    # Point at infinity represented as None
    INFINITY = None

    def __init__(self, a: int, b: int, p: int):
        """
        Args:
            a, b: Curve parameters y² = x³ + ax + b.
            p: Prime modulus.
        """
        self.a = a % p
        self.b = b % p
        self.p = p

        disc = (4 * a ** 3 + 27 * b ** 2) % p
        if disc == 0:
            raise ValueError(f"Singular curve: 4a³ + 27b² ≡ 0 (mod {p})")

        logger.info(f"EllipticCurve y²=x³+{a}x+{b} over F_{p}")

    def is_on_curve(self, P: tuple[int, int] | None) -> bool:
        """Check if point P lies on the curve."""
        if P is None:
            return True  # Point at infinity
        x, y = P
        lhs = (y * y) % self.p
        rhs = (x * x * x + self.a * x + self.b) % self.p
        return lhs == rhs

    def add(self, P: tuple[int, int] | None,
            Q: tuple[int, int] | None) -> tuple[int, int] | None:
        """
        Point addition P + Q on the elliptic curve.

        Rules:
        - P + O = P (identity)
        - P + (-P) = O (inverse)
        - Otherwise: secant/tangent line → third point → reflect
        """
        if P is None:
            return Q
        if Q is None:
            return P

        x1, y1 = P
        x2, y2 = Q

        if x1 == x2:
            if (y1 + y2) % self.p == 0:
                return None  # P + (-P) = O

            # Point doubling: tangent line
            # λ = (3x₁² + a) / (2y₁)
            num = (3 * x1 * x1 + self.a) % self.p
            den = (2 * y1) % self.p
        else:
            # Secant line
            # λ = (y₂ - y₁) / (x₂ - x₁)
            num = (y2 - y1) % self.p
            den = (x2 - x1) % self.p

        # Modular inverse of denominator
        den_inv = pow(den, self.p - 2, self.p)
        lam = (num * den_inv) % self.p

        # New point
        x3 = (lam * lam - x1 - x2) % self.p
        y3 = (lam * (x1 - x3) - y1) % self.p

        return (x3, y3)

    def negate(self, P: tuple[int, int] | None) -> tuple[int, int] | None:
        """Negation: -(x, y) = (x, -y)."""
        if P is None:
            return None
        x, y = P
        return (x, (-y) % self.p)

    def scalar_multiply(self, k: int,
                        P: tuple[int, int] | None) -> tuple[int, int] | None:
        """
        Scalar multiplication [k]P = P + P + ... + P (k times).

        Uses double-and-add (binary method): O(log k) additions.
        """
        if P is None or k == 0:
            return None

        if k < 0:
            k = -k
            P = self.negate(P)

        result = None  # Identity
        addend = P

        while k > 0:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1

        return result

    def order(self) -> int:
        """
        Compute the order of the curve (number of points including O).

        Brute force enumeration over 𝔽_p. For large p, use Schoof's algorithm.
        Hasse's theorem: |#E - (p+1)| ≤ 2√p.
        """
        count = 1  # Point at infinity

        for x in range(self.p):
            rhs = (x * x * x + self.a * x + self.b) % self.p

            # Check if rhs is a quadratic residue
            # Euler's criterion: rhs^{(p-1)/2} ≡ 1 (mod p)
            if rhs == 0:
                count += 1  # y = 0
            else:
                legendre = pow(rhs, (self.p - 1) // 2, self.p)
                if legendre == 1:
                    count += 2  # Two square roots

        return count

    def point_order(self, P: tuple[int, int] | None) -> int:
        """Order of point P: smallest k > 0 with [k]P = O."""
        if P is None:
            return 1

        Q = P
        for k in range(1, self.p + 2):
            if Q is None:
                return k
            Q = self.add(Q, P)

        return -1  # Should not happen for valid points

    def points(self) -> list[tuple[int, int] | None]:
        """Enumerate all points on the curve (including O)."""
        result = [None]  # Point at infinity

        for x in range(self.p):
            rhs = (x * x * x + self.a * x + self.b) % self.p

            # Find square roots of rhs mod p (Tonelli-Shanks simplified)
            if rhs == 0:
                result.append((x, 0))
            elif pow(rhs, (self.p - 1) // 2, self.p) == 1:
                # Square root exists
                y = self._sqrt_mod(rhs)
                if y is not None:
                    result.append((x, y))
                    if y != 0:
                        result.append((x, self.p - y))

        return result

    def _sqrt_mod(self, a: int) -> int | None:
        """Square root of a modulo p (Tonelli-Shanks)."""
        if a == 0:
            return 0
        if self.p == 2:
            return a

        # p ≡ 3 (mod 4): simple case
        if self.p % 4 == 3:
            r = pow(a, (self.p + 1) // 4, self.p)
            if (r * r) % self.p == a:
                return r
            return None

        # General Tonelli-Shanks
        s, q = 0, self.p - 1
        while q % 2 == 0:
            s += 1
            q //= 2

        # Find quadratic non-residue
        z = 2
        while pow(z, (self.p - 1) // 2, self.p) != self.p - 1:
            z += 1

        M = s
        c = pow(z, q, self.p)
        t = pow(a, q, self.p)
        R = pow(a, (q + 1) // 2, self.p)

        while True:
            if t == 1:
                return R

            # Find least i with t^{2^i} = 1
            i = 1
            temp = (t * t) % self.p
            while temp != 1:
                temp = (temp * temp) % self.p
                i += 1

            b = pow(c, 1 << (M - i - 1), self.p)
            M = i
            c = (b * b) % self.p
            t = (t * c) % self.p
            R = (R * b) % self.p
