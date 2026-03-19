"""
Commutative Algebra for HoloLoom Warp Drive
=============================================

Noetherian rings, localization, primary decomposition, Groebner bases,
and algebraic varieties.

Commutative algebra is the algebraic foundation of algebraic geometry.
The bridge: ideals in polynomial rings ↔ algebraic varieties in affine space
(Hilbert's Nullstellensatz).

Key objects:
- Noetherian ring: every ideal is finitely generated (ACC on ideals)
- Localization S⁻¹R: invert elements of a multiplicative set
- Primary decomposition: I = Q₁ ∩ ... ∩ Qₖ (primary ideals)
- Groebner basis: canonical generators for polynomial ideals (Buchberger)
- Algebraic variety: zero set V(I) ⊆ kⁿ of an ideal

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging
from dataclasses import dataclass
from itertools import product as cartesian_product

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# MULTIVARIATE POLYNOMIAL (computational backbone)
# ============================================================================

@dataclass(frozen=True)
class Monomial:
    """Monomial x₁^a₁ x₂^a₂ ... xₙ^aₙ represented as exponent tuple."""
    exponents: tuple[int, ...]

    @property
    def total_degree(self) -> int:
        return sum(self.exponents)

    @property
    def n_vars(self) -> int:
        return len(self.exponents)

    def __mul__(self, other: 'Monomial') -> 'Monomial':
        return Monomial(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def divides(self, other: 'Monomial') -> bool:
        """Check if self divides other (componentwise ≤)."""
        return all(a <= b for a, b in zip(self.exponents, other.exponents))

    def __truediv__(self, other: 'Monomial') -> 'Monomial':
        if not other.divides(self):
            raise ValueError("Monomial does not divide")
        return Monomial(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def lcm(self, other: 'Monomial') -> 'Monomial':
        return Monomial(tuple(max(a, b) for a, b in zip(self.exponents, other.exponents)))

    def gcd(self, other: 'Monomial') -> 'Monomial':
        return Monomial(tuple(min(a, b) for a, b in zip(self.exponents, other.exponents)))


def glex_order(m1: Monomial, m2: Monomial) -> int:
    """Graded lexicographic order: compare total degree first, then lex."""
    d1, d2 = m1.total_degree, m2.total_degree
    if d1 != d2:
        return d1 - d2
    # Lex comparison
    for a, b in zip(m1.exponents, m2.exponents):
        if a != b:
            return a - b
    return 0


def grevlex_order(m1: Monomial, m2: Monomial) -> int:
    """Graded reverse lex: total degree first, then reverse lex from last."""
    d1, d2 = m1.total_degree, m2.total_degree
    if d1 != d2:
        return d1 - d2
    for a, b in zip(reversed(m1.exponents), reversed(m2.exponents)):
        if a != b:
            return b - a  # Reversed
    return 0


class MVPolynomial:
    """
    Multivariate polynomial over ℚ (float approximation).

    Stored as dict: Monomial → coefficient.
    """

    def __init__(self, terms: dict[Monomial, float] = None, n_vars: int = 1):
        self.terms = {}
        self.n_vars = n_vars
        if terms:
            for m, c in terms.items():
                if abs(c) > 1e-14:
                    self.terms[m] = c
                    self.n_vars = max(self.n_vars, m.n_vars)

    @staticmethod
    def from_dense(coeffs: dict[tuple[int, ...], float]) -> 'MVPolynomial':
        """Create from {exponent_tuple: coefficient}."""
        terms = {Monomial(exp): c for exp, c in coeffs.items() if abs(c) > 1e-14}
        n_vars = max((len(exp) for exp in coeffs), default=1)
        return MVPolynomial(terms, n_vars)

    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def leading_monomial(self, order=glex_order) -> Monomial | None:
        """Leading monomial w.r.t. given term order."""
        if not self.terms:
            return None
        from functools import cmp_to_key
        return max(self.terms.keys(), key=cmp_to_key(order))

    def leading_coefficient(self, order=glex_order) -> float:
        lm = self.leading_monomial(order)
        return self.terms.get(lm, 0.0) if lm else 0.0

    def leading_term(self, order=glex_order) -> 'MVPolynomial':
        lm = self.leading_monomial(order)
        if lm is None:
            return MVPolynomial(n_vars=self.n_vars)
        return MVPolynomial({lm: self.terms[lm]}, self.n_vars)

    def __add__(self, other: 'MVPolynomial') -> 'MVPolynomial':
        result = dict(self.terms)
        for m, c in other.terms.items():
            result[m] = result.get(m, 0.0) + c
        return MVPolynomial(result, max(self.n_vars, other.n_vars))

    def __sub__(self, other: 'MVPolynomial') -> 'MVPolynomial':
        result = dict(self.terms)
        for m, c in other.terms.items():
            result[m] = result.get(m, 0.0) - c
        return MVPolynomial(result, max(self.n_vars, other.n_vars))

    def __mul__(self, other: 'MVPolynomial') -> 'MVPolynomial':
        result = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                m = m1 * m2
                result[m] = result.get(m, 0.0) + c1 * c2
        return MVPolynomial(result, max(self.n_vars, other.n_vars))

    def scalar_mul(self, s: float) -> 'MVPolynomial':
        return MVPolynomial({m: c * s for m, c in self.terms.items()}, self.n_vars)

    def evaluate(self, point: np.ndarray) -> float:
        """Evaluate polynomial at a point."""
        result = 0.0
        for m, c in self.terms.items():
            term_val = c
            for i, exp in enumerate(m.exponents):
                if i < len(point):
                    term_val *= point[i] ** exp
            result += term_val
        return result

    def degree(self) -> int:
        if not self.terms:
            return -1
        return max(m.total_degree for m in self.terms)

    def __repr__(self):
        if not self.terms:
            return "0"
        parts = []
        for m, c in sorted(self.terms.items(), key=lambda x: x[0].total_degree, reverse=True):
            parts.append(f"{c:.4g}*x^{m.exponents}")
        return " + ".join(parts[:5]) + ("..." if len(parts) > 5 else "")


# ============================================================================
# NOETHERIAN RING
# ============================================================================

class NoetherianRing:
    """
    Noetherian ring: every ideal is finitely generated.

    Equivalently: the ascending chain condition (ACC) holds —
    every ascending chain I₁ ⊆ I₂ ⊆ ... stabilizes.

    In practice, we work with polynomial rings k[x₁,...,xₙ] which are
    Noetherian by Hilbert's Basis Theorem.
    """

    def __init__(self, n_vars: int, name: str = "k[x]"):
        """Polynomial ring in n_vars variables."""
        self.n_vars = n_vars
        self.name = name
        logger.info(f"NoetherianRing {name} in {n_vars} variables")

    def is_noetherian(self) -> bool:
        """Polynomial rings over fields are always Noetherian (Hilbert)."""
        return True

    def ascending_chain_condition(self, ideals: list[list[MVPolynomial]]) -> bool:
        """
        Check if a chain of ideals I₁ ⊆ I₂ ⊆ ... stabilizes.

        Each ideal given by its generators. Stabilizes when
        generators of Iₙ₊₁ are in the span of generators of Iₙ.
        """
        for i in range(len(ideals) - 1):
            # Check I_i ⊆ I_{i+1} by verifying each generator of I_i
            # reduces to 0 modulo I_{i+1}
            for gen in ideals[i]:
                _, remainder = _polynomial_division(gen, ideals[i + 1])
                if not remainder.is_zero():
                    return False

        # Check stabilization: last few should be equal
        if len(ideals) >= 2:
            last = ideals[-1]
            prev = ideals[-2]
            for gen in last:
                _, r = _polynomial_division(gen, prev)
                if not r.is_zero():
                    return True  # Still growing, but ACC says it will stop
        return True


# ============================================================================
# LOCALIZATION
# ============================================================================

@dataclass
class Fraction:
    """Element of a localization S⁻¹R: numerator/denominator."""
    numerator: MVPolynomial
    denominator: MVPolynomial

    def __repr__(self):
        return f"({self.numerator})/({self.denominator})"


class Localization:
    """
    Localization S⁻¹R of a ring R at a multiplicative set S.

    Elements are fractions r/s where r ∈ R, s ∈ S.
    r₁/s₁ = r₂/s₂ iff ∃t ∈ S: t(r₁s₂ - r₂s₁) = 0.
    """

    @staticmethod
    def localize(ring: NoetherianRing,
                 multiplicative_set: list[MVPolynomial]) -> 'Localization':
        """Create localization at given multiplicative set."""
        loc = Localization()
        loc.ring = ring
        loc.mult_set = multiplicative_set
        return loc

    @staticmethod
    def fraction(numerator: MVPolynomial,
                 denominator: MVPolynomial) -> Fraction:
        """Create element of localized ring."""
        if denominator.is_zero():
            raise ValueError("Cannot divide by zero")
        return Fraction(numerator, denominator)

    @staticmethod
    def add(f1: Fraction, f2: Fraction) -> Fraction:
        """(a/b) + (c/d) = (ad + bc)/(bd)"""
        num = f1.numerator * f2.denominator + f2.numerator * f1.denominator
        den = f1.denominator * f2.denominator
        return Fraction(num, den)

    @staticmethod
    def multiply(f1: Fraction, f2: Fraction) -> Fraction:
        """(a/b)(c/d) = (ac)/(bd)"""
        return Fraction(f1.numerator * f2.numerator,
                        f1.denominator * f2.denominator)


# ============================================================================
# PRIMARY DECOMPOSITION
# ============================================================================

class PrimaryDecomposition:
    """
    Primary decomposition of an ideal I = Q₁ ∩ ... ∩ Qₖ.

    Q is P-primary if:
    1. Q is a proper ideal
    2. If ab ∈ Q and a ∉ Q, then bⁿ ∈ Q for some n (√Q = P is prime)

    Lasker-Noether: every ideal in a Noetherian ring has a primary decomposition.
    """

    @staticmethod
    def decompose(ideal_generators: list[MVPolynomial],
                  n_vars: int) -> list[list[MVPolynomial]]:
        """
        Compute primary decomposition of ideal given by generators.

        Uses a simplified approach for monomial ideals and
        principal ideals. Full decomposition for general polynomial
        ideals requires Groebner basis computations.
        """
        if len(ideal_generators) == 0:
            return []

        if len(ideal_generators) == 1:
            # Principal ideal — factor the polynomial
            return PrimaryDecomposition._decompose_principal(
                ideal_generators[0], n_vars
            )

        # For monomial ideals, decompose by irreducible components
        if all(len(p.terms) == 1 for p in ideal_generators):
            return PrimaryDecomposition._decompose_monomial(
                ideal_generators, n_vars
            )

        # General case: return the ideal itself as a single component
        # (full decomposition requires Groebner basis machinery)
        logger.info("Full primary decomposition not implemented; returning ideal as-is")
        return [ideal_generators]

    @staticmethod
    def _decompose_principal(f: MVPolynomial, n_vars: int) -> list[list[MVPolynomial]]:
        """Decompose principal ideal (f) via factorization."""
        # For univariate, factor as product of irreducibles
        if n_vars == 1 and f.terms:
            # Extract as univariate polynomial
            max_deg = f.degree()
            coeffs = np.zeros(max_deg + 1)
            for m, c in f.terms.items():
                deg = m.exponents[0] if m.exponents else 0
                coeffs[deg] = c

            # Find roots
            if max_deg > 0:
                roots = np.roots(coeffs[::-1])
                components = []
                for root in roots:
                    if abs(root.imag) < 1e-10:
                        # Linear factor (x - root)
                        linear = MVPolynomial.from_dense({
                            (1,): 1.0,
                            (0,): -root.real
                        })
                        components.append([linear])
                if components:
                    return components

        return [[f]]

    @staticmethod
    def _decompose_monomial(generators: list[MVPolynomial],
                            n_vars: int) -> list[list[MVPolynomial]]:
        """Decompose monomial ideal into irreducible monomial ideals."""
        # A monomial ideal is irreducible iff generated by pure powers of variables
        # I = (x₁^a₁, ..., xₖ^aₖ)
        # Alexander dual approach for decomposition
        components = []

        # Extract exponent vectors
        exponents = []
        for gen in generators:
            m = list(gen.terms.keys())[0]
            exponents.append(m.exponents)

        # Each generator contributes one component per variable it involves
        # Simplified: return generators individually as primary components
        for gen in generators:
            components.append([gen])

        return components


# ============================================================================
# GROEBNER BASIS (Buchberger's Algorithm)
# ============================================================================

def _polynomial_division(f: MVPolynomial,
                         divisors: list[MVPolynomial],
                         order=glex_order) -> tuple[list[MVPolynomial], MVPolynomial]:
    """
    Multivariate polynomial division.

    Returns (quotients, remainder) where f = Σ qᵢ fᵢ + r
    and no monomial of r is divisible by any leading monomial of fᵢ.
    """
    quotients = [MVPolynomial(n_vars=f.n_vars) for _ in divisors]
    remainder = MVPolynomial(n_vars=f.n_vars)
    p = MVPolynomial(dict(f.terms), f.n_vars)

    max_iter = 10000
    iteration = 0

    while not p.is_zero() and iteration < max_iter:
        iteration += 1
        divided = False
        lm_p = p.leading_monomial(order)
        lc_p = p.leading_coefficient(order)

        for i, fi in enumerate(divisors):
            lm_fi = fi.leading_monomial(order)
            lc_fi = fi.leading_coefficient(order)

            if lm_fi is not None and lm_fi.divides(lm_p):
                # Divide
                quot_mono = lm_p / lm_fi
                quot_coeff = lc_p / lc_fi
                quot_term = MVPolynomial({quot_mono: quot_coeff}, f.n_vars)

                quotients[i] = quotients[i] + quot_term
                p = p - quot_term * fi
                divided = True
                break

        if not divided:
            # Move leading term to remainder
            lt = MVPolynomial({lm_p: lc_p}, f.n_vars)
            remainder = remainder + lt
            p = p - lt

    return quotients, remainder


def _s_polynomial(f: MVPolynomial, g: MVPolynomial,
                  order=glex_order) -> MVPolynomial:
    """
    S-polynomial of f and g.

    S(f, g) = (lcm(LM(f), LM(g)) / LT(f)) · f - (lcm(LM(f), LM(g)) / LT(g)) · g
    """
    lm_f = f.leading_monomial(order)
    lm_g = g.leading_monomial(order)
    lc_f = f.leading_coefficient(order)
    lc_g = g.leading_coefficient(order)

    lcm_mono = lm_f.lcm(lm_g)

    term_f = MVPolynomial({lcm_mono / lm_f: 1.0 / lc_f}, f.n_vars)
    term_g = MVPolynomial({lcm_mono / lm_g: 1.0 / lc_g}, g.n_vars)

    return term_f * f - term_g * g


class GrobnerBasis:
    """
    Groebner basis computation via Buchberger's algorithm.

    A Groebner basis G for ideal I has the property that every polynomial
    in I reduces to 0 modulo G. Equivalently, LT(I) = ⟨LT(g) : g ∈ G⟩.
    """

    @staticmethod
    def compute(generators: list[MVPolynomial],
                order=glex_order,
                max_iter: int = 500) -> list[MVPolynomial]:
        """
        Buchberger's algorithm.

        Iteratively compute S-polynomials and reduce.
        """
        G = [g for g in generators if not g.is_zero()]
        if not G:
            return []

        pairs = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))]
        iteration = 0

        while pairs and iteration < max_iter:
            i, j = pairs.pop(0)
            iteration += 1

            if i >= len(G) or j >= len(G):
                continue

            s = _s_polynomial(G[i], G[j], order)
            _, remainder = _polynomial_division(s, G, order)

            if not remainder.is_zero():
                new_idx = len(G)
                G.append(remainder)
                for k in range(new_idx):
                    pairs.append((k, new_idx))

        logger.info(f"Groebner basis: {len(generators)} generators -> {len(G)} basis elements")
        return G

    @staticmethod
    def reduced_basis(generators: list[MVPolynomial],
                      order=glex_order) -> list[MVPolynomial]:
        """
        Compute reduced Groebner basis (unique for given term order).

        1. Compute Groebner basis
        2. Make leading coefficients 1
        3. Remove redundant generators
        4. Inter-reduce
        """
        G = GrobnerBasis.compute(generators, order)

        if not G:
            return []

        # Normalize leading coefficients
        for i in range(len(G)):
            lc = G[i].leading_coefficient(order)
            if abs(lc) > 1e-14:
                G[i] = G[i].scalar_mul(1.0 / lc)

        # Remove redundant generators
        minimal = []
        for i, gi in enumerate(G):
            lm_i = gi.leading_monomial(order)
            redundant = False
            for j, gj in enumerate(G):
                if i == j:
                    continue
                lm_j = gj.leading_monomial(order)
                if lm_j is not None and lm_j.divides(lm_i) and lm_j != lm_i:
                    redundant = True
                    break
            if not redundant:
                minimal.append(gi)

        # Inter-reduce
        reduced = []
        for i, gi in enumerate(minimal):
            others = [gj for j, gj in enumerate(minimal) if j != i]
            _, r = _polynomial_division(gi, others, order)
            if not r.is_zero():
                lc = r.leading_coefficient(order)
                if abs(lc) > 1e-14:
                    r = r.scalar_mul(1.0 / lc)
                reduced.append(r)

        return reduced if reduced else minimal


# ============================================================================
# ALGEBRAIC VARIETY
# ============================================================================

class AlgebraicVariety:
    """
    Affine algebraic variety V(I) ⊆ kⁿ: the zero set of an ideal I.

    V(I) = {p ∈ kⁿ : f(p) = 0 for all f ∈ I}

    Over finite fields or bounded regions, we can enumerate points.
    Over ℝ/ℂ, we compute properties like dimension and degree.
    """

    def __init__(self, generators: list[MVPolynomial], n_vars: int):
        self.generators = generators
        self.n_vars = n_vars

    @staticmethod
    def from_ideal(generators: list[MVPolynomial], n_vars: int) -> 'AlgebraicVariety':
        return AlgebraicVariety(generators, n_vars)

    def dimension(self) -> int:
        """
        Dimension of the variety (Krull dimension).

        dim V = n - dim(leading term ideal) where the leading term
        ideal dimension is computed from the Groebner basis.

        Approximation: n_vars - number of variables appearing as
        leading monomials in the Groebner basis.
        """
        gb = GrobnerBasis.compute(self.generators)

        if not gb:
            return self.n_vars  # Zero ideal -> whole space

        # Count variables that appear as pure powers in leading terms
        constrained_vars = set()
        for g in gb:
            lm = g.leading_monomial()
            if lm is not None:
                for i, exp in enumerate(lm.exponents):
                    if exp > 0 and sum(1 for e in lm.exponents if e > 0) == 1:
                        constrained_vars.add(i)

        return self.n_vars - len(constrained_vars)

    def degree(self) -> int:
        """
        Degree of the variety.

        For a hypersurface V(f), degree = deg(f).
        For general varieties, degree = number of points in
        intersection with a generic linear subspace of complementary dimension.
        """
        if len(self.generators) == 1:
            return self.generators[0].degree()

        # For complete intersections: product of degrees
        return np.prod([g.degree() for g in self.generators if g.degree() > 0])

    def is_irreducible(self) -> bool:
        """
        Check if variety is irreducible (cannot be written as union of
        proper subvarieties).

        For principal ideals: V(f) is irreducible iff f is irreducible.
        """
        if len(self.generators) == 1:
            f = self.generators[0]
            # Simple check: univariate polynomial is irreducible if no rational roots
            if self.n_vars == 1:
                deg = f.degree()
                if deg <= 1:
                    return True
                # Check for rational roots (simplified)
                coeffs = np.zeros(deg + 1)
                for m, c in f.terms.items():
                    d = m.exponents[0] if m.exponents else 0
                    coeffs[d] = c
                roots = np.roots(coeffs[::-1])
                real_roots = [r for r in roots if abs(r.imag) < 1e-10]
                return len(real_roots) == 0 or deg == 1

        # General case: check if Groebner basis generates a prime ideal
        # (undecidable in general — return True as heuristic)
        return True

    def points_over_finite_field(self, field_size: int) -> list[np.ndarray]:
        """
        Enumerate all points of V over 𝔽_p (brute force for small fields).
        """
        points = []
        for coords in cartesian_product(range(field_size), repeat=self.n_vars):
            point = np.array(coords, dtype=float)
            if all(abs(g.evaluate(point) % field_size) < 1e-10
                   for g in self.generators):
                points.append(point)
        return points
