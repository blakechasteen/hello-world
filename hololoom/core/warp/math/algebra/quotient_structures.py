"""
Quotient Structures for HoloLoom Warp Drive
============================================

Quotient groups, quotient rings, free groups, and group presentations.

Quotient Group G/N:
  - N ◁ G (normal subgroup)
  - Cosets gN partition G
  - (g₁N)(g₂N) = (g₁g₂)N is well-defined iff N is normal

Quotient Ring R/I:
  - I is a two-sided ideal of R
  - Cosets r + I partition R
  - (r₁ + I)(r₂ + I) = r₁r₂ + I

Free Group F(S):
  - Words over S ∪ S⁻¹ with cancellation
  - Universal property: every map S → G extends uniquely to F(S) → G

Group Presentation ⟨S | R⟩:
  - F(S) / ⟪R⟫ (free group modulo normal closure of relations)

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# COSETS
# ============================================================================

class Coset:
    """
    Coset gH of a subgroup H in group G.

    Left coset: gH = {g·h : h ∈ H}
    """

    def __init__(self, representative: Any, elements: frozenset):
        self.representative = representative
        self.elements = elements

    def __eq__(self, other: 'Coset') -> bool:
        if not isinstance(other, Coset):
            return NotImplemented
        return self.elements == other.elements

    def __hash__(self):
        return hash(self.elements)

    def __repr__(self):
        return f"Coset({self.representative}, |{len(self.elements)}|)"

    def __contains__(self, element):
        return element in self.elements


# ============================================================================
# QUOTIENT GROUP
# ============================================================================

class QuotientGroup:
    """
    Quotient group G/N where N is a normal subgroup of G.

    Elements are cosets gN. Multiplication: (g₁N)(g₂N) = (g₁g₂)N.
    Well-defined because N is normal: gNg⁻¹ = N for all g.
    """

    def __init__(self, group, normal_subgroup):
        """
        Args:
            group: Parent group G (has .elements, .operation, .identity)
            normal_subgroup: Normal subgroup N (has .elements, .operation)
        """
        self.group = group
        self.normal_subgroup = normal_subgroup
        self._cosets = None
        self._coset_map = None  # element -> coset index

        if not self._verify_normal():
            raise ValueError("Subgroup is not normal in the given group")

        self._build_cosets()
        logger.info(
            f"QuotientGroup {group.name}/{normal_subgroup.name} "
            f"with {len(self._cosets)} cosets"
        )

    def _verify_normal(self) -> bool:
        """Check gNg⁻¹ = N for all g in G (sample-based for large groups)."""
        import random
        G = self.group
        N_set = set(map(self._freeze, self.normal_subgroup.elements))

        sample = G.elements
        if len(sample) > 20:
            sample = random.sample(G.elements, 20)

        for g in sample:
            g_inv = self._inverse(g)
            for n in self.normal_subgroup.elements:
                conjugate = G.operation(G.operation(g, n), g_inv)
                if self._freeze(conjugate) not in N_set:
                    return False
        return True

    def _freeze(self, element) -> Any:
        """Make element hashable."""
        if isinstance(element, (list, np.ndarray)):
            return tuple(element) if not isinstance(element, np.ndarray) else tuple(element.flat)
        return element

    def _inverse(self, element) -> Any:
        """Find inverse of element in parent group."""
        G = self.group
        for g in G.elements:
            if G.operation(element, g) == G.identity:
                return g
        raise ValueError(f"No inverse found for {element}")

    def _build_cosets(self):
        """Partition G into cosets of N."""
        G = self.group
        N = self.normal_subgroup
        assigned = set()
        self._cosets = []
        self._coset_map = {}

        for g in G.elements:
            g_key = self._freeze(g)
            if g_key in assigned:
                continue

            # Build coset gN
            coset_elements = set()
            for n in N.elements:
                product = G.operation(g, n)
                p_key = self._freeze(product)
                coset_elements.add(p_key)
                assigned.add(p_key)

            coset = Coset(g, frozenset(coset_elements))
            idx = len(self._cosets)
            self._cosets.append(coset)

            for elem in coset_elements:
                self._coset_map[elem] = idx

    def elements(self) -> list[Coset]:
        """Return all cosets (elements of G/N)."""
        return list(self._cosets)

    def multiply(self, coset1: Coset, coset2: Coset) -> Coset:
        """
        Coset multiplication: (g₁N)(g₂N) = (g₁g₂)N.
        """
        G = self.group
        product = G.operation(coset1.representative, coset2.representative)
        p_key = self._freeze(product)
        idx = self._coset_map[p_key]
        return self._cosets[idx]

    def order(self) -> int:
        """Order of quotient group |G/N| = |G|/|N| (Lagrange's theorem)."""
        return len(self._cosets)

    def identity_coset(self) -> Coset:
        """The coset containing the identity (= N itself)."""
        e_key = self._freeze(self.group.identity)
        idx = self._coset_map[e_key]
        return self._cosets[idx]

    def is_abelian(self) -> bool:
        """Check if quotient group is abelian."""
        cosets = self._cosets
        for c1 in cosets:
            for c2 in cosets:
                if self.multiply(c1, c2) != self.multiply(c2, c1):
                    return False
        return True

    def verify_homomorphism_theorem(self) -> bool:
        """
        First Isomorphism Theorem check:
        The natural projection π: G → G/N is a surjective homomorphism
        with kernel N.
        """
        G = self.group
        N_set = set(map(self._freeze, self.normal_subgroup.elements))
        e_coset = self.identity_coset()

        # Check kernel = N
        for g in G.elements:
            g_key = self._freeze(g)
            coset_idx = self._coset_map[g_key]
            in_kernel = self._cosets[coset_idx] == e_coset
            in_N = g_key in N_set
            if in_kernel != in_N:
                return False
        return True


# ============================================================================
# QUOTIENT RING
# ============================================================================

class QuotientRing:
    """
    Quotient ring R/I where I is a two-sided ideal of R.

    Elements are cosets r + I. Operations:
    (r₁ + I) + (r₂ + I) = (r₁ + r₂) + I
    (r₁ + I)(r₂ + I) = r₁r₂ + I
    """

    def __init__(self, ring, ideal_elements: list):
        """
        Args:
            ring: Parent ring R (has .elements, .add, .mul, .zero, .one)
            ideal_elements: Elements of the ideal I ⊆ R
        """
        self.ring = ring
        self.ideal = ideal_elements
        self._cosets = []
        self._coset_map = {}

        self._build_cosets()
        logger.info(f"QuotientRing with {len(self._cosets)} cosets")

    def _freeze(self, element) -> Any:
        if isinstance(element, (list, np.ndarray)):
            return tuple(element) if not isinstance(element, np.ndarray) else tuple(element.flat)
        return element

    def _build_cosets(self):
        """Partition R into cosets r + I."""
        R = self.ring
        assigned = set()
        self._cosets = []
        self._coset_map = {}

        for r in R.elements:
            r_key = self._freeze(r)
            if r_key in assigned:
                continue

            coset_elements = set()
            for i in self.ideal:
                s = R.add(r, i)
                s_key = self._freeze(s)
                coset_elements.add(s_key)
                assigned.add(s_key)

            coset = Coset(r, frozenset(coset_elements))
            idx = len(self._cosets)
            self._cosets.append(coset)

            for elem in coset_elements:
                self._coset_map[elem] = idx

    def elements(self) -> list[Coset]:
        return list(self._cosets)

    def add(self, a: Coset, b: Coset) -> Coset:
        """(r₁ + I) + (r₂ + I) = (r₁ + r₂) + I"""
        R = self.ring
        s = R.add(a.representative, b.representative)
        s_key = self._freeze(s)
        return self._cosets[self._coset_map[s_key]]

    def multiply(self, a: Coset, b: Coset) -> Coset:
        """(r₁ + I)(r₂ + I) = r₁r₂ + I"""
        R = self.ring
        p = R.mul(a.representative, b.representative)
        p_key = self._freeze(p)
        return self._cosets[self._coset_map[p_key]]

    def zero_coset(self) -> Coset:
        z_key = self._freeze(self.ring.zero)
        return self._cosets[self._coset_map[z_key]]

    def one_coset(self) -> Coset:
        o_key = self._freeze(self.ring.one)
        return self._cosets[self._coset_map[o_key]]

    def is_field(self) -> bool:
        """R/I is a field iff I is maximal."""
        zero = self.zero_coset()
        one = self.one_coset()
        if zero == one:
            return False

        for c in self._cosets:
            if c == zero:
                continue
            has_inverse = False
            for d in self._cosets:
                if self.multiply(c, d) == one:
                    has_inverse = True
                    break
            if not has_inverse:
                return False
        return True


# ============================================================================
# FREE GROUP
# ============================================================================

@dataclass
class Letter:
    """Generator or its inverse in a free group word."""
    symbol: str
    inverted: bool = False

    def inverse(self) -> 'Letter':
        return Letter(self.symbol, not self.inverted)

    def __eq__(self, other):
        if not isinstance(other, Letter):
            return NotImplemented
        return self.symbol == other.symbol and self.inverted == other.inverted

    def __hash__(self):
        return hash((self.symbol, self.inverted))

    def __repr__(self):
        return f"{self.symbol}⁻¹" if self.inverted else self.symbol


class FreeGroup:
    """
    Free group F(S) on a set of generators S.

    Elements are reduced words over S ∪ S⁻¹.
    Two letters cancel if they are inverses of each other.

    Universal property: for any group G and map f: S → G,
    there exists a unique homomorphism F(S) → G extending f.
    """

    def __init__(self, generators: list[str]):
        self.generators = generators
        self._gen_letters = [Letter(g) for g in generators]
        logger.info(f"FreeGroup on {len(generators)} generators: {generators}")

    def word(self, letters: list[Letter]) -> list[Letter]:
        """Create a word and reduce it."""
        return self.reduce(list(letters))

    def identity(self) -> list[Letter]:
        """Empty word = identity element."""
        return []

    def reduce(self, word: list[Letter]) -> list[Letter]:
        """
        Free reduction: cancel adjacent inverse pairs.

        aa⁻¹ → ε, a⁻¹a → ε

        Iterates until no more cancellations possible.
        """
        result = list(word)
        changed = True

        while changed:
            changed = False
            new_result = []
            i = 0
            while i < len(result):
                if (i + 1 < len(result)
                        and result[i].symbol == result[i + 1].symbol
                        and result[i].inverted != result[i + 1].inverted):
                    # Cancel pair
                    i += 2
                    changed = True
                else:
                    new_result.append(result[i])
                    i += 1
            result = new_result

        return result

    def multiply(self, w1: list[Letter], w2: list[Letter]) -> list[Letter]:
        """Concatenate and reduce."""
        return self.reduce(w1 + w2)

    def invert(self, word: list[Letter]) -> list[Letter]:
        """Inverse of a word: reverse and invert each letter."""
        return self.reduce([l.inverse() for l in reversed(word)])

    def is_trivial(self, word: list[Letter]) -> bool:
        """Check if word reduces to identity."""
        return len(self.reduce(word)) == 0

    def commutator(self, w1: list[Letter], w2: list[Letter]) -> list[Letter]:
        """[w1, w2] = w1 w2 w1⁻¹ w2⁻¹"""
        return self.multiply(
            self.multiply(w1, w2),
            self.multiply(self.invert(w1), self.invert(w2))
        )

    def word_length(self, word: list[Letter]) -> int:
        """Length of reduced word."""
        return len(self.reduce(word))

    def conjugate(self, word: list[Letter], by: list[Letter]) -> list[Letter]:
        """Conjugate: by · word · by⁻¹"""
        return self.multiply(self.multiply(by, word), self.invert(by))


# ============================================================================
# GROUP PRESENTATION
# ============================================================================

class GroupPresentation:
    """
    Finitely presented group ⟨S | R⟩ = F(S) / ⟪R⟫.

    S: generators
    R: relations (words in F(S) that equal identity)

    Examples:
        ⟨a | aⁿ⟩ = ℤ/nℤ
        ⟨r, s | rⁿ, s², srsr⟩ = D_n (dihedral)
    """

    def __init__(self, generators: list[str], relations: list[list[Letter]]):
        self.generators = generators
        self.relations = relations
        self.free_group = FreeGroup(generators)
        logger.info(
            f"GroupPresentation ⟨{', '.join(generators)} | "
            f"{len(relations)} relations⟩"
        )

    def is_consistent(self) -> bool:
        """
        Basic consistency: each relation should reduce to a non-trivial
        word in the free group (it becomes trivial in the quotient).
        Relations should not be redundant (one implied by others).

        This is undecidable in general — we do basic checks:
        1. Relations use only declared generators
        2. No relation is the empty word (trivially satisfied)
        """
        valid_symbols = set(self.generators)

        for rel in self.relations:
            if len(rel) == 0:
                logger.warning("Empty relation (trivially satisfied)")
                return False

            for letter in rel:
                if letter.symbol not in valid_symbols:
                    logger.warning(f"Unknown generator: {letter.symbol}")
                    return False

        return True

    def quotient(self) -> 'GroupPresentation':
        """Return self — the presentation IS the quotient F(S)/⟪R⟫."""
        return self

    def abelianization(self) -> dict[str, int]:
        """
        Compute the abelianization G/[G,G] as a ℤ-module.

        In the abelianization, order of letters doesn't matter —
        only the exponent sum per generator. Returns exponent constraints
        from relations.
        """
        constraints = []
        for rel in self.relations:
            exponents = dict.fromkeys(self.generators, 0)
            for letter in rel:
                exponents[letter.symbol] += -1 if letter.inverted else 1
            constraints.append(exponents)
        return constraints

    def relator_matrix(self) -> np.ndarray:
        """
        Build the relation matrix for abelianization.

        Rows = relations, Columns = generators.
        Entry (i, j) = exponent sum of generator j in relation i.
        """
        n_rel = len(self.relations)
        n_gen = len(self.generators)
        M = np.zeros((n_rel, n_gen), dtype=int)

        gen_idx = {g: j for j, g in enumerate(self.generators)}

        for i, rel in enumerate(self.relations):
            for letter in rel:
                j = gen_idx[letter.symbol]
                M[i, j] += -1 if letter.inverted else 1

        return M

    @staticmethod
    def cyclic(n: int) -> 'GroupPresentation':
        """⟨a | aⁿ⟩ ≅ ℤ/nℤ"""
        a = Letter('a')
        return GroupPresentation(['a'], [[a] * n])

    @staticmethod
    def dihedral(n: int) -> 'GroupPresentation':
        """⟨r, s | rⁿ, s², srsr⟩ ≅ D_n"""
        r = Letter('r')
        s = Letter('s')
        return GroupPresentation(
            ['r', 's'],
            [
                [r] * n,       # rⁿ = e
                [s, s],        # s² = e
                [s, r, s, r],  # srsr = e (equivalent to srs = r⁻¹)
            ]
        )

    @staticmethod
    def free_abelian(rank: int) -> 'GroupPresentation':
        """⟨x₁,...,xₙ | [xᵢ, xⱼ] for all i < j⟩ ≅ ℤⁿ"""
        gens = [f'x{i}' for i in range(rank)]
        relations = []
        for i in range(rank):
            for j in range(i + 1, rank):
                xi = Letter(gens[i])
                xj = Letter(gens[j])
                # [xi, xj] = xi xj xi⁻¹ xj⁻¹
                relations.append([xi, xj, xi.inverse(), xj.inverse()])
        return GroupPresentation(gens, relations)
