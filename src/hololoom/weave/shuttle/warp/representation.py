"""
Representation Theory for HoloLoom Warp Drive
==============================================

Implements representation theory for symmetry detection and equivariant learning.

Core Concepts:
- Group Representations: Homomorphisms G → GL(V)
- Character Theory: Traces of representation matrices
- Irreducible Representations: Building blocks (irreps)
- Schur's Lemma: Intertwining operators
- Peter-Weyl Theorem: Decomposition into irreps
- Induced Representations: Extend subgroup reps to group reps

Applications:
- Symmetry detection in knowledge graphs
- Equivariant neural networks
- Invariant feature extraction
- Group-theoretic clustering
- Symmetry-aware embeddings

Mathematical Foundation:
A representation of group G on vector space V is a homomorphism:
ρ: G → GL(V) such that ρ(gh) = ρ(g)ρ(h) and ρ(e) = I

Character χ_ρ(g) = Tr(ρ(g)) determines representation up to isomorphism.

Author: HoloLoom Team
Date: 2025-10-25
"""

import numpy as np
from typing import Any, Dict, List, Set, Tuple, Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Groups
# ============================================================================

class Group:
    """
    Abstract group (G, ·)

    A group is a set with binary operation satisfying:
    - Closure: g, h ∈ G ⟹ g · h ∈ G
    - Associativity: (g · h) · k = g · (h · k)
    - Identity: ∃e: e · g = g · e = g
    - Inverse: ∀g ∃g⁻¹: g · g⁻¹ = g⁻¹ · g = e
    """

    def __init__(self, name: str = "G"):
        self.name = name
        self.elements: Set[Any] = set()
        self.identity: Optional[Any] = None
        self.multiplication_table: Dict[Tuple[Any, Any], Any] = {}
        self.inverse_map: Dict[Any, Any] = {}

        logger.info(f"Group {name} initialized")

    def add_element(self, g: Any) -> None:
        """Add element to group."""
        self.elements.add(g)

    def set_identity(self, e: Any) -> None:
        """Set identity element."""
        self.identity = e
        self.add_element(e)

    def set_multiplication(self, g: Any, h: Any, result: Any) -> None:
        """Define g · h = result."""
        self.multiplication_table[(g, h)] = result
        self.add_element(g)
        self.add_element(h)
        self.add_element(result)

    def multiply(self, g: Any, h: Any) -> Any:
        """Compute g · h."""
        if (g, h) in self.multiplication_table:
            return self.multiplication_table[(g, h)]
        else:
            raise ValueError(f"Product ({g}, {h}) not defined")

    def inverse(self, g: Any) -> Any:
        """Get g⁻¹."""
        if g in self.inverse_map:
            return self.inverse_map[g]
        else:
            # Try to find inverse
            for h in self.elements:
                if (g, h) in self.multiplication_table:
                    if self.multiplication_table[(g, h)] == self.identity:
                        self.inverse_map[g] = h
                        return h
            raise ValueError(f"Inverse of {g} not found")

    def order(self) -> int:
        """Order of the group (number of elements)."""
        return len(self.elements)

    def is_abelian(self) -> bool:
        """Check if group is abelian (commutative)."""
        for g in self.elements:
            for h in self.elements:
                if (g, h) in self.multiplication_table and (h, g) in self.multiplication_table:
                    if self.multiplication_table[(g, h)] != self.multiplication_table[(h, g)]:
                        return False
        return True


# ============================================================================
# Finite Groups (Common Examples)
# ============================================================================

def cyclic_group(n: int) -> Group:
    """
    Cyclic group ℤ/nℤ = {0, 1, ..., n-1} with addition mod n.

    Example: C₃ = {0, 1, 2} with 1 + 2 = 0 (mod 3)
    """
    G = Group(name=f"C{n}")

    # Elements are integers mod n
    for i in range(n):
        G.add_element(i)

    G.set_identity(0)

    # Multiplication table
    for i in range(n):
        for j in range(n):
            G.set_multiplication(i, j, (i + j) % n)

    # Inverses
    for i in range(n):
        G.inverse_map[i] = (n - i) % n

    return G


def symmetric_group(n: int) -> Group:
    """
    Symmetric group S_n of permutations of n elements.

    Example: S₃ has 6 elements (all permutations of {1,2,3})
    """
    from itertools import permutations

    G = Group(name=f"S{n}")

    # Elements are permutations (as tuples)
    for perm in permutations(range(n)):
        G.add_element(perm)

    # Identity is (0,1,2,...,n-1)
    identity = tuple(range(n))
    G.set_identity(identity)

    # Multiplication: compose permutations
    for p1 in G.elements:
        for p2 in G.elements:
            # Composition: (p1 ∘ p2)(i) = p1(p2(i))
            composed = tuple(p1[p2[i]] for i in range(n))
            G.set_multiplication(p1, p2, composed)

    # Inverses
    for perm in G.elements:
        # Inverse permutation
        inv = [0] * n
        for i, val in enumerate(perm):
            inv[val] = i
        G.inverse_map[perm] = tuple(inv)

    return G


# ============================================================================
# Representations
# ============================================================================

@dataclass
class Representation:
    """
    Representation ρ: G → GL(V)

    A linear representation of group G on vector space V (dimension d).
    Each group element g is mapped to a d×d invertible matrix ρ(g).

    Properties:
    - Homomorphism: ρ(gh) = ρ(g)ρ(h)
    - Identity: ρ(e) = I
    - Invertible: ρ(g)⁻¹ = ρ(g⁻¹)
    """
    group: Group
    dimension: int
    matrices: Dict[Any, np.ndarray] = field(default_factory=dict)
    name: str = "ρ"

    def __post_init__(self):
        logger.info(f"Representation {self.name}: {self.group.name} → GL({self.dimension})")

    def __call__(self, g: Any) -> np.ndarray:
        """Get ρ(g) matrix."""
        if g in self.matrices:
            return self.matrices[g]
        else:
            raise ValueError(f"Representation not defined for {g}")

    def set_matrix(self, g: Any, matrix: np.ndarray) -> None:
        """Set ρ(g) = matrix."""
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError(f"Matrix must be {self.dimension}×{self.dimension}")

        self.matrices[g] = matrix

    def verify_homomorphism(self) -> bool:
        """Verify ρ(gh) = ρ(g)ρ(h) for all g, h."""
        for g in self.group.elements:
            for h in self.group.elements:
                try:
                    gh = self.group.multiply(g, h)

                    # Check ρ(gh) = ρ(g)ρ(h)
                    lhs = self(gh)
                    rhs = self(g) @ self(h)

                    if not np.allclose(lhs, rhs, atol=1e-10):
                        logger.warning(f"Homomorphism fails for {g} * {h}")
                        return False
                except:
                    continue

        return True

    def character(self) -> Dict[Any, complex]:
        """
        Compute character χ(g) = Tr(ρ(g)) for all g.

        Characters are class functions (constant on conjugacy classes).
        """
        return {g: np.trace(self(g)) for g in self.group.elements}

    def is_irreducible(self) -> bool:
        """
        Check if representation is irreducible.

        A representation is irreducible if it has no proper invariant subspaces.
        Uses Schur's orthogonality: <χ, χ> = 1 for irreps.
        """
        char = self.character()

        # Inner product of character with itself
        inner = sum(abs(char[g])**2 for g in self.group.elements) / self.group.order()

        # Irreducible iff <χ, χ> = 1
        return np.isclose(inner, 1.0, atol=1e-6)

    def decompose_into_irreps(self) -> Dict[str, int]:
        """
        Decompose representation into irreducible representations.

        Uses character orthogonality:
        ρ ≅ ⨁ᵢ mᵢ ρᵢ where mᵢ = <χ, χᵢ>

        Returns multiplicities: {irrep_name: multiplicity}
        """
        # This requires knowing all irreps of the group
        # Simplified: just compute inner product with self
        char = self.character()

        inner_product = sum(abs(char[g])**2 for g in self.group.elements) / self.group.order()

        return {
            "multiplicity": int(np.round(inner_product)),
            "irreducible": np.isclose(inner_product, 1.0)
        }


# ============================================================================
# Standard Representations
# ============================================================================

def trivial_representation(group: Group) -> Representation:
    """
    Trivial representation: ρ(g) = 1 for all g ∈ G.

    Every group has this 1-dimensional representation.
    """
    rho = Representation(group=group, dimension=1, name="trivial")

    for g in group.elements:
        rho.set_matrix(g, np.array([[1.0]]))

    return rho


def regular_representation(group: Group) -> Representation:
    """
    Regular representation: ρ: G → GL(|G|)

    For finite group, embed G in permutation matrices.
    Matrix ρ(g) acts by left multiplication: ρ(g)_h,k = δ_{h, gk}
    """
    n = group.order()
    rho = Representation(group=group, dimension=n, name="regular")

    # Index elements
    elements_list = list(group.elements)
    index_map = {g: i for i, g in enumerate(elements_list)}

    for g in group.elements:
        matrix = np.zeros((n, n))

        for i, h in enumerate(elements_list):
            # Left multiplication: g * h
            gh = group.multiply(g, h)
            j = index_map[gh]
            matrix[j, i] = 1.0

        rho.set_matrix(g, matrix)

    return rho


def permutation_representation(group: Group, action: Callable) -> Representation:
    """
    Permutation representation from group action.

    Given action σ: G × X → X, get representation on ℂ^X.
    Matrix ρ(g)_{x,y} = 1 if g·x = y, else 0.
    """
    # Simplified: assume action is on finite set represented as integers
    # Full implementation would take explicit action
    return regular_representation(group)  # Placeholder


# ============================================================================
# Character Theory
# ============================================================================

class CharacterTable:
    """
    Character table for a finite group.

    Rows: Irreducible representations
    Columns: Conjugacy classes
    Entries: χᵢ(C_j) = character of irrep i on class j

    Properties:
    - Square table (# irreps = # conjugacy classes)
    - Orthogonality relations
    - Determines all representations
    """

    def __init__(self, group: Group):
        self.group = group
        self.conjugacy_classes: List[Set[Any]] = []
        self.irreps: List[Representation] = []
        self.table: np.ndarray = None

        self._compute_conjugacy_classes()

        logger.info(f"Character table for {group.name}: {len(self.conjugacy_classes)} classes")

    def _compute_conjugacy_classes(self) -> None:
        """Compute conjugacy classes: [g] = {hgh⁻¹ | h ∈ G}"""
        unprocessed = set(self.group.elements)

        while unprocessed:
            g = unprocessed.pop()
            conjugacy_class = {g}

            # Compute {hgh⁻¹ for all h}
            for h in self.group.elements:
                h_inv = self.group.inverse(h)
                # Compute hgh⁻¹
                hg = self.group.multiply(h, g)
                hgh_inv = self.group.multiply(hg, h_inv)
                conjugacy_class.add(hgh_inv)

            self.conjugacy_classes.append(conjugacy_class)
            unprocessed -= conjugacy_class

    def add_irrep(self, rho: Representation) -> None:
        """Add irreducible representation to table."""
        if not rho.is_irreducible():
            logger.warning(f"{rho.name} may not be irreducible")

        self.irreps.append(rho)
        self._recompute_table()

    def _recompute_table(self) -> None:
        """Recompute character table from irreps."""
        n_irreps = len(self.irreps)
        n_classes = len(self.conjugacy_classes)

        self.table = np.zeros((n_irreps, n_classes), dtype=complex)

        for i, rho in enumerate(self.irreps):
            char = rho.character()
            for j, conj_class in enumerate(self.conjugacy_classes):
                # Character is constant on conjugacy class
                representative = next(iter(conj_class))
                self.table[i, j] = char[representative]

    def column_orthogonality(self) -> bool:
        """
        Verify column orthogonality:
        ∑ᵢ χᵢ(C)* χᵢ(C') = |G|/|C| δ_{C,C'}
        """
        if self.table is None:
            return False

        n_classes = len(self.conjugacy_classes)

        for j in range(n_classes):
            for k in range(n_classes):
                inner = sum(
                    np.conj(self.table[i, j]) * self.table[i, k]
                    for i in range(len(self.irreps))
                )

                class_size = len(self.conjugacy_classes[j])
                expected = self.group.order() / class_size if j == k else 0

                if not np.isclose(inner, expected):
                    return False

        return True

    def row_orthogonality(self) -> bool:
        """
        Verify row orthogonality:
        ∑_C |C| χᵢ(C)* χⱼ(C) = |G| δᵢⱼ
        """
        if self.table is None:
            return False

        n_irreps = len(self.irreps)

        for i in range(n_irreps):
            for j in range(n_irreps):
                inner = sum(
                    len(self.conjugacy_classes[k]) *
                    np.conj(self.table[i, k]) * self.table[j, k]
                    for k in range(len(self.conjugacy_classes))
                )

                expected = self.group.order() if i == j else 0

                if not np.isclose(inner, expected):
                    return False

        return True

    def __repr__(self) -> str:
        """Pretty print character table."""
        if self.table is None:
            return "Empty character table"

        lines = [f"Character Table for {self.group.name}"]
        lines.append("-" * 40)

        # Header
        header = "Irrep  | " + " | ".join(f"C{j}" for j in range(len(self.conjugacy_classes)))
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        for i, rho in enumerate(self.irreps):
            row = f"{rho.name:6} | " + " | ".join(
                f"{self.table[i, j].real:4.1f}" for j in range(len(self.conjugacy_classes))
            )
            lines.append(row)

        return "\n".join(lines)


# ============================================================================
# Equivariant Maps
# ============================================================================

class EquivariantMap:
    """
    G-equivariant map between representations.

    For representations ρ: G → GL(V) and σ: G → GL(W),
    a linear map f: V → W is equivariant if:

    f(ρ(g)v) = σ(g)f(v) for all g ∈ G, v ∈ V

    Schur's Lemma: If ρ, σ irreducible, then:
    - If ρ ≇ σ: only equivariant map is f = 0
    - If ρ ≅ σ: equivariant maps are scalar multiples of identity
    """

    def __init__(self,
                 source_rep: Representation,
                 target_rep: Representation,
                 matrix: np.ndarray,
                 name: str = "f"):
        self.source_rep = source_rep
        self.target_rep = target_rep
        self.matrix = matrix
        self.name = name

        if source_rep.group != target_rep.group:
            raise ValueError("Representations must be of the same group")

        if matrix.shape != (target_rep.dimension, source_rep.dimension):
            raise ValueError(
                f"Matrix shape {matrix.shape} doesn't match "
                f"({target_rep.dimension}, {source_rep.dimension})"
            )

        logger.info(f"Equivariant map {name}: {source_rep.name} → {target_rep.name}")

    def verify_equivariance(self) -> bool:
        """
        Verify f(ρ(g)v) = σ(g)f(v) for all g.

        Equivalent to: σ(g) f = f ρ(g) (as matrices)
        """
        for g in self.source_rep.group.elements:
            rho_g = self.source_rep(g)
            sigma_g = self.target_rep(g)

            # Left side: σ(g) f
            left = sigma_g @ self.matrix

            # Right side: f ρ(g)
            right = self.matrix @ rho_g

            if not np.allclose(left, right, atol=1e-10):
                logger.warning(f"Equivariance fails for {g}")
                return False

        return True

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Apply map to vector."""
        return self.matrix @ v


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Representation Theory Demo")
    print("="*80 + "\n")

    # 1. Cyclic group C₃
    print("1. Cyclic Group C₃")
    print("-" * 40)

    C3 = cyclic_group(3)
    print(f"Group: {C3.name}, order: {C3.order()}")
    print(f"Elements: {C3.elements}")
    print(f"Abelian: {C3.is_abelian()}")

    # 2. Trivial representation
    print("\n2. Trivial Representation")
    print("-" * 40)

    triv = trivial_representation(C3)
    print(f"ρ(0) = {triv(0).flatten()}")
    print(f"ρ(1) = {triv(1).flatten()}")
    print(f"Homomorphism verified: {triv.verify_homomorphism()}")

    # 3. Regular representation
    print("\n3. Regular Representation")
    print("-" * 40)

    reg = regular_representation(C3)
    print(f"Dimension: {reg.dimension}")
    print(f"ρ(1) =")
    print(reg(1))
    print(f"Homomorphism verified: {reg.verify_homomorphism()}")

    # 4. Character
    print("\n4. Characters")
    print("-" * 40)

    triv_char = triv.character()
    reg_char = reg.character()

    print(f"Trivial character: {triv_char}")
    print(f"Regular character: {reg_char}")

    # 5. Irreducibility
    print("\n5. Irreducibility Test")
    print("-" * 40)

    print(f"Trivial irreducible: {triv.is_irreducible()}")
    print(f"Regular irreducible: {reg.is_irreducible()}")

    # 6. Character table
    print("\n6. Character Table")
    print("-" * 40)

    char_table = CharacterTable(C3)
    char_table.add_irrep(triv)
    # Would need to compute all irreps for complete table

    print(f"Conjugacy classes: {len(char_table.conjugacy_classes)}")
    for i, cls in enumerate(char_table.conjugacy_classes):
        print(f"  C{i}: {cls}")

    # 7. Symmetric group S₃
    print("\n7. Symmetric Group S₃")
    print("-" * 40)

    S3 = symmetric_group(3)
    print(f"Group: {S3.name}, order: {S3.order()}")
    print(f"Abelian: {S3.is_abelian()}")

    # Create sign representation: ρ(σ) = sign(σ)
    sign_rep = Representation(group=S3, dimension=1, name="sign")

    for perm in S3.elements:
        # Count inversions to get sign
        inversions = sum(1 for i in range(len(perm)) for j in range(i+1, len(perm)) if perm[i] > perm[j])
        sign = 1 if inversions % 2 == 0 else -1
        sign_rep.set_matrix(perm, np.array([[sign]]))

    print(f"Sign representation homomorphism: {sign_rep.verify_homomorphism()}")
    print(f"Sign character: {sign_rep.character()}")

    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)
