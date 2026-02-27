"""
Abstract Algebra for HoloLoom Warp Drive
========================================

Groups, Rings, and Fields - The fundamental algebraic structures.

Core Concepts:
- Groups: Sets with associative binary operation, identity, inverses
- Rings: Sets with addition and multiplication (e.g., integers)
- Fields: Rings where every nonzero element has multiplicative inverse
- Morphisms: Structure-preserving maps between algebraic objects
- Ideals: Special subsets of rings (kernels of ring homomorphisms)
- Quotient Structures: Factor groups, quotient rings

Mathematical Foundation:
Group (G, ·):
  - Closure: a, b ∈ G ⟹ a·b ∈ G
  - Associativity: (a·b)·c = a·(b·c)
  - Identity: ∃e: e·a = a·e = a
  - Inverses: ∀a ∃a⁻¹: a·a⁻¹ = a⁻¹·a = e

Ring (R, +, ×):
  - (R, +) is an abelian group
  - (R, ×) is a monoid
  - Distributivity: a(b + c) = ab + ac

Field (F, +, ×):
  - (F, +, ×) is a ring
  - Every nonzero element has multiplicative inverse

Applications to Warp Space:
- Symmetry groups of knowledge graphs
- Polynomial rings for algebraic computations
- Finite fields for coding theory and cryptography
- Quotient structures for dimensional reduction

Author: HoloLoom Team
Date: 2025-10-26
"""

import numpy as np
from typing import Callable, List, Set, Optional, Tuple, Any, FrozenSet
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# GROUPS
# ============================================================================

class Group:
    """
    Abstract group (G, ·).

    A set with a binary operation satisfying group axioms.
    """

    def __init__(
        self,
        elements: List,
        operation: Callable[[Any, Any], Any],
        identity: Any,
        name: str = "G"
    ):
        """
        Initialize group.

        Args:
            elements: Set of group elements
            operation: Binary operation (a, b) -> a·b
            identity: Identity element e
            name: Group name
        """
        self.elements = elements
        self.operation = operation
        self.identity = identity
        self.name = name

        logger.info(f"Group {name} with {len(elements)} elements")

    def __mul__(self, other: 'Group') -> 'Group':
        """Direct product of groups: G × H"""
        # Elements are pairs (g, h)
        product_elements = [(g, h) for g in self.elements for h in other.elements]

        # Operation: (g1, h1) · (g2, h2) = (g1·g2, h1·h2)
        def product_op(pair1, pair2):
            g1, h1 = pair1
            g2, h2 = pair2
            return (self.operation(g1, g2), other.operation(h1, h2))

        product_identity = (self.identity, other.identity)

        return Group(
            elements=product_elements,
            operation=product_op,
            identity=product_identity,
            name=f"{self.name}×{other.name}"
        )

    def is_abelian(self) -> bool:
        """
        Check if group is abelian (commutative).

        Test ab = ba for sample of elements.
        """
        import random
        sample_size = min(10, len(self.elements))
        sample = random.sample(self.elements, sample_size)

        for a in sample:
            for b in sample:
                if self.operation(a, b) != self.operation(b, a):
                    return False

        return True

    def order(self) -> int:
        """Order of group |G|"""
        return len(self.elements)

    def element_order(self, element: Any) -> int:
        """
        Order of element: smallest n > 0 such that aⁿ = e.
        """
        current = element
        n = 1

        while current != self.identity and n <= self.order():
            current = self.operation(current, element)
            n += 1

        if n > self.order():
            return float('inf')  # Infinite order

        return n

    def subgroup(self, generators: List[Any]) -> 'Group':
        """
        Generate subgroup from generators.

        H = ⟨g₁, g₂, ...⟩ = smallest subgroup containing generators
        """
        subgroup_elements = set([self.identity])
        subgroup_elements.update(generators)

        # Closure: keep applying operation until no new elements
        changed = True
        iterations = 0
        max_iterations = 1000

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            new_elements = set(subgroup_elements)

            for a in subgroup_elements:
                for b in subgroup_elements:
                    result = self.operation(a, b)
                    if result not in subgroup_elements:
                        new_elements.add(result)
                        changed = True

            subgroup_elements = new_elements

        logger.info(f"Generated subgroup of order {len(subgroup_elements)}")

        return Group(
            elements=list(subgroup_elements),
            operation=self.operation,
            identity=self.identity,
            name=f"⟨{generators}⟩"
        )

    def center(self) -> 'Group':
        """
        Center of group: Z(G) = {z ∈ G : zg = gz for all g ∈ G}
        """
        center_elements = []

        for z in self.elements:
            commutes_with_all = True
            for g in self.elements:
                if self.operation(z, g) != self.operation(g, z):
                    commutes_with_all = False
                    break

            if commutes_with_all:
                center_elements.append(z)

        logger.info(f"Center has order {len(center_elements)}")

        return Group(
            elements=center_elements,
            operation=self.operation,
            identity=self.identity,
            name=f"Z({self.name})"
        )

    @staticmethod
    def cyclic(n: int) -> 'Group':
        """
        Cyclic group ℤ/nℤ of order n.

        Elements: {0, 1, 2, ..., n-1}
        Operation: addition mod n
        """
        elements = list(range(n))
        operation = lambda a, b: (a + b) % n

        return Group(elements, operation, identity=0, name=f"Z/{n}Z")

    @staticmethod
    def symmetric(n: int) -> 'Group':
        """
        Symmetric group Sₙ (permutations of n elements).

        Order: n!
        """
        from itertools import permutations

        elements = list(permutations(range(n)))

        # Composition of permutations
        def compose(sigma, tau):
            return tuple(sigma[tau[i]] for i in range(n))

        identity = tuple(range(n))

        return Group(elements, compose, identity, name=f"S_{n}")

    @staticmethod
    def dihedral(n: int) -> 'Group':
        """
        Dihedral group D_n (symmetries of regular n-gon).

        Order: 2n
        Elements: rotations r^k and reflections sr^k
        """
        # Represent elements as (rotation, reflection)
        # rotation: 0, 1, ..., n-1
        # reflection: 0 or 1
        elements = [(r, s) for r in range(n) for s in [0, 1]]

        def dihedral_op(elem1, elem2):
            r1, s1 = elem1
            r2, s2 = elem2

            if s1 == 0:
                # r^r1 * elem2
                r_new = (r1 + r2) % n if s2 == 0 else (r1 - r2) % n
                s_new = s2
            else:
                # s*r^r1 * elem2
                r_new = (r1 - r2) % n if s2 == 0 else (r1 + r2) % n
                s_new = (s1 + s2) % 2

            return (r_new, s_new)

        return Group(elements, dihedral_op, identity=(0, 0), name=f"D_{n}")


# ============================================================================
# GROUP HOMOMORPHISMS
# ============================================================================

@dataclass
class GroupHomomorphism:
    """
    Group homomorphism φ: G → H.

    Preserves group structure: φ(ab) = φ(a)φ(b)
    """
    source: Group
    target: Group
    mapping: Callable[[Any], Any]
    name: str = "φ"

    def __call__(self, element: Any) -> Any:
        """Apply homomorphism"""
        return self.mapping(element)

    def is_homomorphism(self) -> bool:
        """
        Verify φ(ab) = φ(a)φ(b) for sample elements.
        """
        import random
        sample = random.sample(self.source.elements, min(10, len(self.source.elements)))

        for a in sample:
            for b in sample:
                ab = self.source.operation(a, b)

                phi_ab = self.mapping(ab)
                phi_a_phi_b = self.target.operation(self.mapping(a), self.mapping(b))

                if phi_ab != phi_a_phi_b:
                    return False

        return True

    def kernel(self) -> Group:
        """
        Kernel: ker(φ) = {g ∈ G : φ(g) = e_H}

        Always a normal subgroup of G.
        """
        kernel_elements = [g for g in self.source.elements if self.mapping(g) == self.target.identity]

        logger.info(f"Kernel has order {len(kernel_elements)}")

        return Group(
            elements=kernel_elements,
            operation=self.source.operation,
            identity=self.source.identity,
            name=f"ker({self.name})"
        )

    def image(self) -> Group:
        """
        Image: im(φ) = {φ(g) : g ∈ G}

        Subgroup of H.
        """
        image_elements = list(set(self.mapping(g) for g in self.source.elements))

        logger.info(f"Image has order {len(image_elements)}")

        return Group(
            elements=image_elements,
            operation=self.target.operation,
            identity=self.target.identity,
            name=f"im({self.name})"
        )


# ============================================================================
# RINGS
# ============================================================================

class Ring:
    """
    Ring (R, +, ×).

    Set with two operations: addition (abelian group) and multiplication (monoid).
    """

    def __init__(
        self,
        elements: List,
        addition: Callable[[Any, Any], Any],
        multiplication: Callable[[Any, Any], Any],
        zero: Any,
        one: Any,
        name: str = "R"
    ):
        """
        Initialize ring.

        Args:
            elements: Ring elements
            addition: Additive operation
            multiplication: Multiplicative operation
            zero: Additive identity
            one: Multiplicative identity
            name: Ring name
        """
        self.elements = elements
        self.add = addition
        self.mul = multiplication
        self.zero = zero
        self.one = one
        self.name = name

        logger.info(f"Ring {name} with {len(elements)} elements")

    def is_commutative(self) -> bool:
        """
        Check if multiplication is commutative.
        """
        import random
        sample = random.sample(self.elements, min(10, len(self.elements)))

        for a in sample:
            for b in sample:
                if self.mul(a, b) != self.mul(b, a):
                    return False

        return True

    def is_integral_domain(self) -> bool:
        """
        Check if R is an integral domain.

        Commutative ring with no zero divisors (ab = 0 ⟹ a = 0 or b = 0).
        """
        if not self.is_commutative():
            return False

        # Check for zero divisors
        for a in self.elements:
            if a == self.zero:
                continue

            for b in self.elements:
                if b == self.zero:
                    continue

                if self.mul(a, b) == self.zero:
                    return False  # Found zero divisor

        return True

    def units(self) -> List[Any]:
        """
        Units: elements with multiplicative inverse.

        U(R) = {a ∈ R : ∃b, ab = ba = 1}
        """
        unit_elements = []

        for a in self.elements:
            for b in self.elements:
                if self.mul(a, b) == self.one and self.mul(b, a) == self.one:
                    unit_elements.append(a)
                    break

        logger.info(f"Ring has {len(unit_elements)} units")

        return unit_elements

    @staticmethod
    def integers_mod_n(n: int) -> 'Ring':
        """
        Ring ℤ/nℤ (integers modulo n).
        """
        elements = list(range(n))
        add = lambda a, b: (a + b) % n
        mul = lambda a, b: (a * b) % n

        return Ring(elements, add, mul, zero=0, one=1, name=f"Z/{n}Z")

    @staticmethod
    def matrix_ring(n: int, base_ring: 'Ring') -> 'Ring':
        """
        Matrix ring M_n(R) over base ring R.

        Elements: n×n matrices with entries from R
        """
        # For simplicity, we'll use NumPy arrays for matrices over ℤ/pℤ
        # Full implementation would need proper matrix class

        logger.info(f"Matrix ring M_{n}({base_ring.name})")

        # Placeholder - would need full matrix implementation
        return Ring(
            elements=[],  # Would enumerate matrices
            addition=lambda A, B: A + B,
            multiplication=lambda A, B: A @ B,
            zero=np.zeros((n, n)),
            one=np.eye(n),
            name=f"M_{n}({base_ring.name})"
        )


# ============================================================================
# IDEALS
# ============================================================================

class Ideal:
    """
    Ideal I of ring R.

    Subset closed under addition and multiplication by ring elements.
    """

    def __init__(self, ring: Ring, generators: List[Any]):
        """
        Initialize ideal from generators.

        I = ⟨g₁, g₂, ...⟩ = {r₁g₁ + r₂g₂ + ... : rᵢ ∈ R}
        """
        self.ring = ring
        self.generators = generators

        # Generate ideal elements
        self.elements = self._generate_ideal()

        logger.info(f"Ideal with {len(self.elements)} elements")

    def _generate_ideal(self) -> Set[Any]:
        """Generate all elements of ideal"""
        ideal_set = set([self.ring.zero])
        ideal_set.update(self.generators)

        # Closure under addition and R-multiplication
        changed = True
        iterations = 0
        max_iter = 100

        while changed and iterations < max_iter:
            changed = False
            iterations += 1
            new_elements = set(ideal_set)

            # Addition closure
            for a in ideal_set:
                for b in ideal_set:
                    result = self.ring.add(a, b)
                    if result not in ideal_set:
                        new_elements.add(result)
                        changed = True

            # R-multiplication closure
            for r in self.ring.elements:
                for a in ideal_set:
                    result = self.ring.mul(r, a)
                    if result not in ideal_set:
                        new_elements.add(result)
                        changed = True

            ideal_set = new_elements

        return ideal_set

    def __contains__(self, element: Any) -> bool:
        """Check if element is in ideal"""
        return element in self.elements

    def is_prime(self) -> bool:
        """
        Check if ideal is prime.

        I is prime if ab ∈ I ⟹ a ∈ I or b ∈ I
        """
        for a in self.ring.elements:
            if a in self.elements:
                continue

            for b in self.ring.elements:
                if b in self.elements:
                    continue

                ab = self.ring.mul(a, b)
                if ab in self.elements:
                    return False  # ab in I but a, b not in I

        return True

    def is_maximal(self) -> bool:
        """
        Check if ideal is maximal.

        I is maximal if I ≠ R and no proper ideal strictly between I and R.
        """
        # I must be proper
        if len(self.elements) == len(self.ring.elements):
            return False

        # Check no intermediate ideals (computationally expensive!)
        # For finite rings, we'd check all possible ideals
        # Simplified check for educational purposes

        return True  # Placeholder


# ============================================================================
# FIELDS
# ============================================================================

class Field(Ring):
    """
    Field (F, +, ×).

    Ring where every nonzero element has multiplicative inverse.
    """

    def __init__(
        self,
        elements: List,
        addition: Callable[[Any, Any], Any],
        multiplication: Callable[[Any, Any], Any],
        zero: Any,
        one: Any,
        name: str = "F"
    ):
        super().__init__(elements, addition, multiplication, zero, one, name)

        # Verify field property
        if not self._verify_field():
            logger.warning(f"{name} may not be a valid field")

    def _verify_field(self) -> bool:
        """Verify every nonzero element has inverse"""
        for a in self.elements:
            if a == self.zero:
                continue

            has_inverse = False
            for b in self.elements:
                if self.mul(a, b) == self.one and self.mul(b, a) == self.one:
                    has_inverse = True
                    break

            if not has_inverse:
                return False

        return True

    def characteristic(self) -> int:
        """
        Characteristic of field.

        char(F) = smallest n > 0 such that n·1 = 0, or 0 if no such n exists.
        """
        current = self.one
        n = 1

        while n <= len(self.elements):
            if current == self.zero:
                return n
            current = self.add(current, self.one)
            n += 1

        return 0  # Characteristic 0

    @staticmethod
    def finite_field(p: int, n: int = 1) -> 'Field':
        """
        Finite field 𝔽_{p^n} (Galois field).

        For n=1: 𝔽_p = ℤ/pℤ (prime field)
        For n>1: Extension field (would need polynomial representation)
        """
        if n > 1:
            raise NotImplementedError("Extension fields require polynomial implementation")

        # Prime field 𝔽_p
        elements = list(range(p))
        add = lambda a, b: (a + b) % p
        mul = lambda a, b: (a * b) % p

        return Field(elements, add, mul, zero=0, one=1, name=f"F_{p}")

    @staticmethod
    def rationals_sample() -> 'Field':
        """
        Sample of rational numbers ℚ (for demonstration).

        Full ℚ is infinite, so we use a finite subset.
        """
        from fractions import Fraction

        # Small rationals
        elements = []
        for num in range(-5, 6):
            for den in range(1, 6):
                elements.append(Fraction(num, den))

        elements = list(set(elements))  # Remove duplicates

        add = lambda a, b: a + b
        mul = lambda a, b: a * b

        return Field(elements, add, mul, zero=Fraction(0), one=Fraction(1), name="Q_sample")


# ============================================================================
# POLYNOMIAL RINGS
# ============================================================================

@dataclass
class Polynomial:
    """
    Polynomial over a ring.

    f(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ
    """
    coefficients: List[Any]  # [a₀, a₁, a₂, ..., aₙ]
    ring: Ring

    def __repr__(self) -> str:
        terms = []
        for i, coef in enumerate(self.coefficients):
            if coef != self.ring.zero:
                if i == 0:
                    terms.append(str(coef))
                elif i == 1:
                    terms.append(f"{coef}x")
                else:
                    terms.append(f"{coef}x^{i}")
        return " + ".join(terms) if terms else "0"

    def degree(self) -> int:
        """Degree of polynomial"""
        for i in range(len(self.coefficients) - 1, -1, -1):
            if self.coefficients[i] != self.ring.zero:
                return i
        return -1  # Zero polynomial

    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        """Add polynomials"""
        max_len = max(len(self.coefficients), len(other.coefficients))

        result_coeffs = []
        for i in range(max_len):
            a = self.coefficients[i] if i < len(self.coefficients) else self.ring.zero
            b = other.coefficients[i] if i < len(other.coefficients) else self.ring.zero
            result_coeffs.append(self.ring.add(a, b))

        return Polynomial(result_coeffs, self.ring)

    def __mul__(self, other: 'Polynomial') -> 'Polynomial':
        """Multiply polynomials"""
        result_degree = self.degree() + other.degree()
        if result_degree < 0:
            return Polynomial([self.ring.zero], self.ring)

        result_coeffs = [self.ring.zero] * (result_degree + 1)

        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                product = self.ring.mul(a, b)
                result_coeffs[i + j] = self.ring.add(result_coeffs[i + j], product)

        return Polynomial(result_coeffs, self.ring)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Group',
    'GroupHomomorphism',
    'Ring',
    'Ideal',
    'Field',
    'Polynomial'
]
