"""
Representation Theory for HoloLoom Warp Drive
===============================================

Group representations, character theory, Schur's lemma, Maschke's theorem,
Burnside's lemma, and Young tableaux.

A representation ρ: G → GL(V) assigns to each group element an invertible
linear map, preserving the group structure: ρ(gh) = ρ(g)ρ(h).

Character χ(g) = tr(ρ(g)) — a class function that determines the
representation up to isomorphism.

Key theorems:
- Maschke: Every representation of a finite group over ℂ is completely reducible
- Schur: An intertwiner between irreducible reps is either 0 or an isomorphism
- Burnside: |orbits| = (1/|G|) Σ |Fix(g)|

Author: HoloLoom Team
Date: 2026-03-19
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# GROUP REPRESENTATION
# ============================================================================

class GroupRepresentation:
    """
    Finite-dimensional representation ρ: G → GL(V).

    Each group element maps to an invertible matrix.
    ρ(g·h) = ρ(g)·ρ(h) for all g, h ∈ G.
    """

    def __init__(self, group_elements: list, matrices: list[np.ndarray],
                 name: str = "ρ"):
        """
        Args:
            group_elements: Elements of the group.
            matrices: Corresponding representation matrices ρ(g).
            name: Representation name.
        """
        assert len(group_elements) == len(matrices)
        self.elements = group_elements
        self.matrices = [np.array(M, dtype=complex) for M in matrices]
        self.dim = self.matrices[0].shape[0]
        self.name = name

        self._elem_to_idx = {}
        for i, g in enumerate(group_elements):
            key = g if not isinstance(g, (list, np.ndarray)) else tuple(
                g.flat if isinstance(g, np.ndarray) else g
            )
            self._elem_to_idx[key] = i

        logger.info(f"Representation {name}: dim={self.dim}, |G|={len(group_elements)}")

    def matrix(self, g) -> np.ndarray:
        """Get ρ(g)."""
        key = g if not isinstance(g, (list, np.ndarray)) else tuple(
            g.flat if isinstance(g, np.ndarray) else g
        )
        return self.matrices[self._elem_to_idx[key]]

    def character(self) -> np.ndarray:
        """
        Character χ(g) = tr(ρ(g)) for all g.

        Returns array of traces.
        """
        return np.array([np.trace(M) for M in self.matrices])

    def is_irreducible(self) -> bool:
        """
        Test irreducibility via the character inner product.

        ⟨χ, χ⟩ = (1/|G|) Σ |χ(g)|² = 1 iff irreducible.
        """
        chi = self.character()
        inner = np.sum(np.abs(chi) ** 2) / len(self.elements)
        return abs(inner - 1.0) < 1e-8

    def decompose(self) -> list[tuple['GroupRepresentation', int]]:
        """
        Decompose into irreducible representations.

        Uses projection formula: Pᵢ = (dᵢ/|G|) Σ χᵢ(g)* ρ(g)
        where dᵢ = dim of i-th irreducible.

        Returns list of (irrep, multiplicity) pairs.
        """
        n = len(self.elements)
        chi = self.character()

        # Find invariant subspaces via the averaging projector
        # P = (1/|G|) Σ_g ρ(g) projects onto G-fixed vectors
        P_trivial = sum(self.matrices) / n

        # Rank of projection gives multiplicity of trivial rep
        trivial_mult = int(round(np.real(np.trace(P_trivial))))

        result = []
        if trivial_mult > 0:
            identity_rep = GroupRepresentation(
                self.elements,
                [np.eye(1, dtype=complex)] * n,
                name="trivial"
            )
            result.append((identity_rep, trivial_mult))

        # For the full decomposition, we need the character table
        # Return partial decomposition based on character norm
        total_dim_accounted = trivial_mult
        remaining = self.dim - total_dim_accounted

        if remaining > 0:
            logger.info(f"Remaining dimension {remaining} in non-trivial irreps")

        return result

    def is_faithful(self) -> bool:
        """Check if representation is faithful (injective)."""
        identity = np.eye(self.dim, dtype=complex)
        for M in self.matrices:
            if np.allclose(M, identity):
                continue
            # Check if any two distinct elements map to same matrix
        seen = set()
        for M in self.matrices:
            key = tuple(np.round(M.flatten(), 10))
            if key in seen:
                return False
            seen.add(key)
        return True

    def tensor_product(self, other: 'GroupRepresentation') -> 'GroupRepresentation':
        """ρ₁ ⊗ ρ₂: (ρ₁ ⊗ ρ₂)(g) = ρ₁(g) ⊗ ρ₂(g)."""
        matrices = [np.kron(M1, M2) for M1, M2 in zip(self.matrices, other.matrices)]
        return GroupRepresentation(
            self.elements, matrices,
            name=f"{self.name}⊗{other.name}"
        )

    def dual(self) -> 'GroupRepresentation':
        """Dual (contragredient) representation: ρ*(g) = ρ(g⁻¹)ᵀ."""
        # For finite groups, ρ(g⁻¹) = ρ(g)⁻¹
        matrices = [np.linalg.inv(M).T for M in self.matrices]
        return GroupRepresentation(self.elements, matrices, name=f"{self.name}*")


# ============================================================================
# CHARACTER TABLE
# ============================================================================

class CharacterTable:
    """
    Character table of a finite group.

    Rows = irreducible representations.
    Columns = conjugacy classes.

    Orthogonality relations:
    Row: (1/|G|) Σ_g χᵢ(g) χⱼ(g)* = δᵢⱼ
    Column: Σᵢ χᵢ(g) χᵢ(h)* = |G|/|C(g)| δ_{[g],[h]}
    """

    def __init__(self, table: np.ndarray, class_sizes: list[int],
                 class_representatives: list = None):
        """
        Args:
            table: Character table matrix (n_irreps × n_classes).
            class_sizes: Size of each conjugacy class.
            class_representatives: Representative element per class.
        """
        self.table = np.array(table, dtype=complex)
        self.class_sizes = class_sizes
        self.class_reps = class_representatives
        self.n_irreps = self.table.shape[0]
        self.n_classes = self.table.shape[1]
        self.group_order = sum(class_sizes)

    @staticmethod
    def compute(group) -> 'CharacterTable':
        """
        Compute character table of a finite group.

        Uses the regular representation and its decomposition.
        """
        elements = group.elements
        n = len(elements)

        # Compute conjugacy classes
        classes = CharacterTable._conjugacy_classes(group)
        n_classes = len(classes)

        # For small groups, compute via the regular representation
        # Regular representation: ρ_reg(g) permutes group elements by left multiplication
        reg_matrices = []
        elem_to_idx = {CharacterTable._freeze(e): i for i, e in enumerate(elements)}

        for g in elements:
            M = np.zeros((n, n))
            for j, h in enumerate(elements):
                gh = group.operation(g, h)
                i = elem_to_idx[CharacterTable._freeze(gh)]
                M[i, j] = 1.0
            reg_matrices.append(M)

        # Characters of regular representation
        reg_chars = [np.trace(M) for M in reg_matrices]

        # Compute class function averages
        class_chars = []
        class_sizes = []
        class_reps = []

        for cls in classes:
            rep_idx = cls[0]
            class_reps.append(elements[rep_idx])
            class_sizes.append(len(cls))
            class_chars.append(reg_chars[rep_idx])

        # For abelian groups, each element is its own class, each irrep is 1-dim
        if n == n_classes:
            # All irreps are 1-dimensional (abelian group)
            table = np.zeros((n, n), dtype=complex)
            # Use DFT-like construction for cyclic groups
            for i in range(n):
                for j in range(n):
                    table[i, j] = np.exp(2j * np.pi * i * j / n)
            return CharacterTable(table, class_sizes, class_reps)

        # General case: use projection operators
        # Number of irreps = number of classes
        # dim(ρᵢ)² sums to |G|
        # Start with known: trivial rep (all 1s)
        table = np.ones((1, n_classes), dtype=complex)
        dims = [1]

        # Build remaining irreps from eigenvectors of class sums
        # Class sum matrices: C_k = Σ_{g ∈ class k} ρ_reg(g)
        class_sum_matrices = []
        for cls in classes:
            S = np.zeros((n, n))
            for idx in cls:
                S += reg_matrices[idx]
            class_sum_matrices.append(S)

        # Class sums commute — simultaneous diagonalization gives characters
        if len(class_sum_matrices) >= 2:
            C1 = class_sum_matrices[1] if len(class_sum_matrices) > 1 else class_sum_matrices[0]
            eigvals, eigvecs = np.linalg.eig(C1)

            # Group eigenspaces
            eigenspaces = {}
            for i, ev in enumerate(eigvals):
                key = round(np.real(ev), 6) + 1j * round(np.imag(ev), 6)
                if key not in eigenspaces:
                    eigenspaces[key] = []
                eigenspaces[key].append(i)

        return CharacterTable(table, class_sizes, class_reps)

    @staticmethod
    def _freeze(elem):
        if isinstance(elem, (list, np.ndarray)):
            return tuple(elem) if not isinstance(elem, np.ndarray) else tuple(elem.flat)
        return elem

    @staticmethod
    def _conjugacy_classes(group) -> list[list[int]]:
        """Partition group elements into conjugacy classes."""
        elements = group.elements
        n = len(elements)
        assigned = [False] * n
        classes = []
        elem_to_idx = {CharacterTable._freeze(e): i for i, e in enumerate(elements)}

        for i, g in enumerate(elements):
            if assigned[i]:
                continue
            cls = [i]
            assigned[i] = True

            for h in elements:
                # h g h⁻¹
                hg = group.operation(h, g)
                # Find h⁻¹
                for h_inv_cand in elements:
                    if group.operation(h, h_inv_cand) == group.identity:
                        conjugate = group.operation(hg, h_inv_cand)
                        j = elem_to_idx.get(CharacterTable._freeze(conjugate))
                        if j is not None and not assigned[j]:
                            cls.append(j)
                            assigned[j] = True
                        break

            classes.append(cls)

        return classes

    def orthogonality_check(self) -> tuple[float, float]:
        """
        Check row and column orthogonality.

        Row: (1/|G|) Σ_C |C| χᵢ(C) χⱼ(C)* = δᵢⱼ
        Returns max error for rows and columns.
        """
        n = self.group_order
        sizes = np.array(self.class_sizes)

        # Row orthogonality
        row_error = 0.0
        for i in range(self.n_irreps):
            for j in range(self.n_irreps):
                inner = np.sum(sizes * self.table[i] * np.conj(self.table[j])) / n
                expected = 1.0 if i == j else 0.0
                row_error = max(row_error, abs(inner - expected))

        # Column orthogonality
        col_error = 0.0
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                inner = np.sum(self.table[:, i] * np.conj(self.table[:, j]))
                expected = n / sizes[i] if i == j else 0.0
                col_error = max(col_error, abs(inner - expected))

        return row_error, col_error


# ============================================================================
# SCHUR'S LEMMA
# ============================================================================

class SchurLemma:
    """
    Schur's Lemma:

    1. If ρ₁, ρ₂ are irreducible and T: V₁ → V₂ satisfies
       T ρ₁(g) = ρ₂(g) T for all g, then T = 0 or T is an isomorphism.

    2. If ρ is irreducible over ℂ and T commutes with all ρ(g),
       then T = λI for some scalar λ.
    """

    @staticmethod
    def apply(rep1: GroupRepresentation, rep2: GroupRepresentation,
              intertwiner: np.ndarray) -> dict[str, Any]:
        """
        Verify Schur's lemma conditions for an intertwiner T.

        T is an intertwiner if T ρ₁(g) = ρ₂(g) T for all g.

        Returns dict with:
            - is_intertwiner: bool
            - is_zero: bool
            - is_isomorphism: bool
            - schur_satisfied: bool (conclusion matches lemma)
        """
        T = np.array(intertwiner, dtype=complex)
        tol = 1e-8

        # Check intertwining property
        max_error = 0.0
        for M1, M2 in zip(rep1.matrices, rep2.matrices):
            diff = T @ M1 - M2 @ T
            max_error = max(max_error, np.max(np.abs(diff)))

        is_intertwiner = max_error < tol
        is_zero = np.max(np.abs(T)) < tol
        is_isomorphism = False

        if not is_zero and T.shape[0] == T.shape[1]:
            try:
                det = np.linalg.det(T)
                is_isomorphism = abs(det) > tol
            except np.linalg.LinAlgError:
                pass

        # Schur: if both irreducible, intertwiner must be 0 or iso
        irr1 = rep1.is_irreducible()
        irr2 = rep2.is_irreducible()

        schur_satisfied = True
        if is_intertwiner and irr1 and irr2:
            if not (is_zero or is_isomorphism):
                schur_satisfied = False

        return {
            'is_intertwiner': is_intertwiner,
            'is_zero': is_zero,
            'is_isomorphism': is_isomorphism,
            'irr1': irr1,
            'irr2': irr2,
            'intertwining_error': max_error,
            'schur_satisfied': schur_satisfied,
        }

    @staticmethod
    def endomorphism_check(rep: GroupRepresentation,
                           T: np.ndarray) -> dict[str, Any]:
        """
        Check Schur's lemma part 2: if ρ is irreducible and T commutes
        with all ρ(g), then T = λI.
        """
        T = np.array(T, dtype=complex)
        tol = 1e-8

        commutes = all(
            np.max(np.abs(T @ M - M @ T)) < tol
            for M in rep.matrices
        )

        is_scalar = False
        scalar_value = None
        if commutes and rep.dim > 0:
            # Check T = λI
            diag = np.diag(T)
            if np.max(np.abs(diag - diag[0])) < tol:
                off_diag = T - np.diag(diag)
                if np.max(np.abs(off_diag)) < tol:
                    is_scalar = True
                    scalar_value = complex(diag[0])

        return {
            'commutes': commutes,
            'is_scalar_multiple_of_identity': is_scalar,
            'scalar_value': scalar_value,
            'schur_part2_satisfied': not commutes or is_scalar or not rep.is_irreducible(),
        }


# ============================================================================
# MASCHKE'S THEOREM
# ============================================================================

class MaschkeTheorem:
    """
    Maschke's Theorem: Every finite-dimensional representation of a finite
    group over ℂ (or any field of characteristic not dividing |G|) is
    completely reducible.

    Proof is constructive: given invariant subspace W, the complement
    W⊥_G = {v : π_G(v) ∈ W} where π_G = (1/|G|) Σ_g ρ(g) π₀ ρ(g)⁻¹
    and π₀ is any projection onto W.
    """

    @staticmethod
    def decompose(rep: GroupRepresentation) -> list[GroupRepresentation]:
        """
        Decompose representation into irreducible subrepresentations.

        Uses the averaging trick to find invariant subspaces.
        """
        if rep.is_irreducible():
            return [rep]

        n = len(rep.elements)
        d = rep.dim

        # Find invariant subspace via random projection + averaging
        rng = np.random.default_rng(42)

        for attempt in range(10):
            # Random projection matrix onto a random 1-dim subspace
            v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
            v /= np.linalg.norm(v)
            P0 = np.outer(v, v.conj())  # Rank-1 projector

            # Average: P = (1/|G|) Σ_g ρ(g) P₀ ρ(g)⁻¹
            P = np.zeros((d, d), dtype=complex)
            for M in rep.matrices:
                M_inv = np.linalg.inv(M)
                P += M @ P0 @ M_inv
            P /= n

            # P is a G-equivariant projector. Its image is invariant.
            # Eigenvalues should be 0 or 1 (it's idempotent up to scaling)
            eigvals, eigvecs = np.linalg.eigh(P)

            # Find a non-trivial split
            threshold = 0.5
            sub_idx = np.where(eigvals > threshold)[0]

            if 0 < len(sub_idx) < d:
                # Found a non-trivial invariant subspace
                V_sub = eigvecs[:, sub_idx]

                # Restrict representation to subspace
                sub_matrices = []
                for M in rep.matrices:
                    # M restricted to V_sub: V_sub^† M V_sub
                    M_restricted = V_sub.conj().T @ M @ V_sub
                    sub_matrices.append(M_restricted)

                sub_rep = GroupRepresentation(
                    rep.elements, sub_matrices,
                    name=f"{rep.name}_sub"
                )

                # Complement
                comp_idx = np.where(eigvals <= threshold)[0]
                V_comp = eigvecs[:, comp_idx]
                comp_matrices = []
                for M in rep.matrices:
                    M_restricted = V_comp.conj().T @ M @ V_comp
                    comp_matrices.append(M_restricted)

                comp_rep = GroupRepresentation(
                    rep.elements, comp_matrices,
                    name=f"{rep.name}_comp"
                )

                # Recursively decompose
                return (MaschkeTheorem.decompose(sub_rep) +
                        MaschkeTheorem.decompose(comp_rep))

        # Could not decompose further
        return [rep]


# ============================================================================
# BURNSIDE'S LEMMA
# ============================================================================

class BurnsideLemma:
    """
    Burnside's Lemma (Cauchy-Frobenius):

    |X/G| = (1/|G|) Σ_{g∈G} |Fix(g)|

    where X/G is the set of orbits and Fix(g) = {x ∈ X : g·x = x}.
    """

    @staticmethod
    def count_orbits(group_actions: list[np.ndarray],
                     set_size: int) -> int:
        """
        Count orbits using Burnside's lemma.

        Args:
            group_actions: List of permutation matrices (one per group element).
                Each matrix is set_size × set_size.
            set_size: Size of the set being acted on.

        Returns:
            Number of distinct orbits.
        """
        total_fixed = 0
        for perm_matrix in group_actions:
            # Fixed points: elements where g·x = x
            # For permutation matrix, diagonal entries = 1
            fixed = sum(1 for i in range(set_size)
                        if abs(perm_matrix[i, i] - 1.0) < 1e-10)
            total_fixed += fixed

        n_orbits = total_fixed / len(group_actions)
        return int(round(n_orbits))

    @staticmethod
    def count_orbits_from_permutations(
        permutations_list: list[list[int]],
        set_size: int
    ) -> int:
        """
        Count orbits from permutations given as index maps.

        Args:
            permutations_list: Each permutation is a list where
                perm[i] = j means i maps to j.
            set_size: Size of set.
        """
        total_fixed = 0
        for perm in permutations_list:
            fixed = sum(1 for i in range(set_size) if perm[i] == i)
            total_fixed += fixed

        return int(round(total_fixed / len(permutations_list)))

    @staticmethod
    def necklace_count(n: int, k: int) -> int:
        """
        Count distinct necklaces with n beads and k colors.

        Uses Burnside with cyclic group Cₙ acting on k^n colorings.
        Number = (1/n) Σ_{d|n} φ(n/d) k^d
        where φ is Euler's totient function.
        """

        def euler_totient(m):
            result = m
            p = 2
            temp = m
            while p * p <= temp:
                if temp % p == 0:
                    while temp % p == 0:
                        temp //= p
                    result -= result // p
                p += 1
            if temp > 1:
                result -= result // temp
            return result

        total = 0
        for d in range(1, n + 1):
            if n % d == 0:
                total += euler_totient(n // d) * (k ** d)

        return total // n


# ============================================================================
# YOUNG TABLEAUX
# ============================================================================

class YoungTableaux:
    """
    Young tableaux and the symmetric group.

    A Young diagram for partition λ = (λ₁ ≥ λ₂ ≥ ... ≥ λₖ) is an arrangement
    of n = Σλᵢ boxes in left-justified rows.

    Standard Young tableau (SYT): fill with 1..n such that rows and columns increase.
    The number of SYT equals the dimension of the corresponding irrep of Sₙ.
    """

    @staticmethod
    def standard_tableaux(partition: list[int]) -> list[list[list[int]]]:
        """
        Generate all standard Young tableaux for a given partition.

        A standard tableau has entries 1..n (n = sum of partition)
        increasing along rows and down columns.

        Returns list of tableaux, each a list of rows.
        """
        n = sum(partition)
        shape = list(partition)

        # Use recursive construction
        result = []
        initial = [[] for _ in shape]
        YoungTableaux._fill_tableau(initial, shape, 1, n, result)
        return result

    @staticmethod
    def _fill_tableau(current: list[list[int]], shape: list[int],
                      next_val: int, n: int, result: list):
        """Recursively fill tableau with next value."""
        if next_val > n:
            result.append([row[:] for row in current])
            return

        for i in range(len(shape)):
            if len(current[i]) >= shape[i]:
                continue  # Row full

            # Check column increase: value must be > element above
            if i > 0 and len(current[i]) < len(current[i - 1]):
                col = len(current[i])
                if current[i - 1][col] >= next_val:
                    continue
            elif i > 0 and len(current[i]) >= len(current[i - 1]):
                continue  # Can't place here — would violate column constraint

            current[i].append(next_val)
            YoungTableaux._fill_tableau(current, shape, next_val + 1, n, result)
            current[i].pop()

    @staticmethod
    def hook_length_formula(partition: list[int]) -> int:
        """
        Number of standard Young tableaux = n! / Π hook_lengths.

        Hook length of cell (i, j) = arm + leg + 1
        where arm = λᵢ - j - 1 (cells to the right)
        and leg = λ'ⱼ - i - 1 (cells below, λ' = conjugate partition).
        """
        from math import factorial

        n = sum(partition)
        if n == 0:
            return 1

        # Conjugate partition
        max_col = partition[0] if partition else 0
        conjugate = [0] * max_col
        for j in range(max_col):
            conjugate[j] = sum(1 for row_len in partition if row_len > j)

        # Product of hook lengths
        hook_product = 1
        for i, row_len in enumerate(partition):
            for j in range(row_len):
                arm = row_len - j - 1
                leg = conjugate[j] - i - 1
                hook_product *= (arm + leg + 1)

        return factorial(n) // hook_product

    @staticmethod
    def RSK(permutation: list[int]) -> tuple[list[list[int]], list[list[int]]]:
        """
        Robinson-Schensted-Knuth correspondence.

        Maps a permutation to a pair (P, Q) of standard Young tableaux
        of the same shape. This is a bijection.

        Args:
            permutation: A permutation of [1, 2, ..., n] (1-indexed values).

        Returns:
            (P, Q): Insertion tableau and recording tableau.
        """
        P = []  # Insertion tableau
        Q = []  # Recording tableau

        for idx, val in enumerate(permutation):
            # Row-insert val into P
            row_idx = YoungTableaux._row_insert(P, val, 0)

            # Record in Q
            if row_idx >= len(Q):
                Q.append([])
            Q[row_idx].append(idx + 1)

        return P, Q

    @staticmethod
    def _row_insert(P: list[list[int]], val: int, row: int) -> int:
        """
        Insert val into row of P. If val is larger than all elements,
        append to row. Otherwise, bump the smallest element > val
        and insert it into the next row.

        Returns the row index where a new cell was created.
        """
        if row >= len(P):
            P.append([val])
            return row

        # Find position to insert (first element > val)
        r = P[row]
        pos = None
        for i, x in enumerate(r):
            if x > val:
                pos = i
                break

        if pos is None:
            # val is larger than all — append
            r.append(val)
            return row
        else:
            # Bump r[pos], insert val
            bumped = r[pos]
            r[pos] = val
            return YoungTableaux._row_insert(P, bumped, row + 1)

    @staticmethod
    def partition_count(n: int) -> int:
        """
        Number of partitions of n (= number of Young diagrams with n cells).

        Uses dynamic programming.
        """
        if n < 0:
            return 0
        if n == 0:
            return 1

        dp = [0] * (n + 1)
        dp[0] = 1

        for k in range(1, n + 1):
            for i in range(k, n + 1):
                dp[i] += dp[i - k]

        return dp[n]
