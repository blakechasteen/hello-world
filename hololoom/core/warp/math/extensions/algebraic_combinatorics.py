"""
Algebraic Combinatorics — Young Tableaux, RSK, Schur, Hall-Littlewood, Discrete Morse
======================================================================================

Intersection of algebra and combinatorics: symmetric functions, representation
theory via tableaux, and topological combinatorics.

Core Concepts:
- Young tableaux: Combinatorial objects indexing representations of S_n and GL_n
- RSK correspondence: Bijection between permutations and pairs of standard tableaux
- Schur functions: Basis of symmetric functions, characters of GL_n representations
- Hall-Littlewood: One-parameter deformation interpolating Schur and monomial
- Discrete Morse: Simplicial complex reduction via acyclic matchings

Mathematical Foundation:
s_λ(x) = det(h_{λ_i - i + j})_{1≤i,j≤l}     (Jacobi-Trudi formula)
P_λ(x; t) = Σ_{μ≤λ} K_{λμ}(t) m_μ(x)        (Hall-Littlewood expansion)
RSK: σ ↦ (P(σ), Q(σ)) with sh(P) = sh(Q) = λ (Robinson-Schensted-Knuth)

Applications to Warp Space:
- Tableaux for structured embedding decompositions
- RSK for permutation analysis in weaving cycles
- Schur functions for multi-scale feature fusion
- Discrete Morse for topological feature persistence

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# YOUNG TABLEAUX
# ============================================================================

class YoungTableauxOps:
    """
    Operations on Young tableaux.

    A Young diagram of partition λ = (λ_1 ≥ λ_2 ≥ ... ≥ λ_k > 0) is an
    arrangement of |λ| boxes in k left-justified rows. A standard Young
    tableau (SYT) fills boxes with 1,...,n increasing along rows and columns.
    A semistandard Young tableau (SSYT) allows repeated entries, increasing
    along columns and non-decreasing along rows.
    """

    @staticmethod
    def standard(partition: list[int]) -> list[list[list[int]]]:
        """
        Generate all standard Young tableaux of shape λ.

        Uses recursive insertion: place n in all valid positions and recurse.

        Args:
            partition: Partition as a list [λ_1, λ_2, ...] in decreasing order.

        Returns:
            List of SYTs, each represented as list of rows (lists of ints).
        """
        partition = [p for p in partition if p > 0]
        if not partition:
            return [[]]

        n = sum(partition)
        if n == 0:
            return [[]]

        # Build all SYTs by placing 1, 2, ..., n in order
        results = []
        YoungTableauxOps._generate_syt(
            partition, [[] for _ in partition], 1, n, results
        )
        return results

    @staticmethod
    def _generate_syt(partition: list[int], current: list[list[int]],
                      next_val: int, n: int, results: list):
        """Recursively generate SYTs by placing next_val in valid corners."""
        if next_val > n:
            results.append([row[:] for row in current])
            return

        k = len(partition)
        for i in range(k):
            row_len = len(current[i])
            if row_len >= partition[i]:
                continue  # Row full
            # Check increasing along columns: value must exceed entry above
            if i > 0 and len(current[i]) >= len(current[i - 1]):
                continue  # Column constraint: can't add here before row above
            # Check we're filling left-to-right within each row
            if row_len > 0 and current[i][-1] >= next_val:
                continue  # Should always hold since we insert in order

            current[i].append(next_val)
            YoungTableauxOps._generate_syt(partition, current, next_val + 1, n, results)
            current[i].pop()

    @staticmethod
    def semistandard(partition: list[int], max_val: int) -> list[list[list[int]]]:
        """
        Generate all semistandard Young tableaux of shape λ with entries in {1,...,max_val}.

        SSYT conditions: rows non-decreasing, columns strictly increasing.

        Args:
            partition: Partition shape.
            max_val: Maximum entry value.

        Returns:
            List of SSYTs.
        """
        partition = [p for p in partition if p > 0]
        if not partition:
            return [[]]

        k = len(partition)
        results = []
        empty = [[] for _ in range(k)]
        YoungTableauxOps._generate_ssyt(partition, empty, max_val, 0, 0, results)
        return results

    @staticmethod
    def _generate_ssyt(partition: list[int], current: list[list[int]],
                       max_val: int, row: int, col: int,
                       results: list):
        """Fill SSYT cell by cell, left-to-right then top-to-bottom."""
        k = len(partition)
        if row >= k:
            results.append([r[:] for r in current])
            return

        if col >= partition[row]:
            YoungTableauxOps._generate_ssyt(
                partition, current, max_val, row + 1, 0, results
            )
            return

        # Determine valid range for this cell
        min_val = 1
        if col > 0:
            min_val = current[row][col - 1]  # Non-decreasing in row
        if row > 0 and col < len(current[row - 1]):
            min_val = max(min_val, current[row - 1][col] + 1)  # Strictly increasing in column

        for v in range(min_val, max_val + 1):
            current[row].append(v)
            YoungTableauxOps._generate_ssyt(
                partition, current, max_val, row, col + 1, results
            )
            current[row].pop()

    @staticmethod
    def hook_length_formula(partition: list[int]) -> int:
        """
        Count SYTs of shape λ using the hook length formula.

        f^λ = n! / Π_{(i,j) ∈ λ} h(i,j)

        where h(i,j) = λ_i - j + λ'_j - i + 1 is the hook length.
        """
        partition = [p for p in partition if p > 0]
        if not partition:
            return 1

        n = sum(partition)
        # Compute conjugate partition
        max_col = partition[0] if partition else 0
        conjugate = [0] * max_col
        for p in partition:
            for j in range(p):
                conjugate[j] += 1

        product = 1
        for i, row_len in enumerate(partition):
            for j in range(row_len):
                hook = (row_len - j) + (conjugate[j] - i) - 1
                product *= hook

        return int(np.math.factorial(n)) // product


# ============================================================================
# RSK CORRESPONDENCE
# ============================================================================

class RSKCorrespondence:
    """
    Robinson-Schensted-Knuth correspondence.

    Bijection between permutations of {1,...,n} and pairs (P, Q) of standard
    Young tableaux of the same shape with |shape| = n.

    P = insertion tableau (row insertion algorithm)
    Q = recording tableau (records insertion order)
    """

    @staticmethod
    def forward(permutation: list[int]) -> tuple[list[list[int]], list[list[int]]]:
        """
        RSK insertion: permutation → (P, Q).

        Args:
            permutation: A permutation as a list of distinct positive integers.

        Returns:
            (P, Q): Insertion tableau and recording tableau.
        """
        P: list[list[int]] = []
        Q: list[list[int]] = []

        for idx, val in enumerate(permutation):
            row_idx = RSKCorrespondence._row_insert(P, val, 0)
            # Add to Q at the position where a new cell was created
            if row_idx >= len(Q):
                Q.append([idx + 1])
            else:
                Q[row_idx].append(idx + 1)

        return P, Q

    @staticmethod
    def _row_insert(P: list[list[int]], val: int, row: int) -> int:
        """
        Insert val into P starting at given row. Returns row where new cell added.

        If val is larger than all entries in the row, append it.
        Otherwise, bump the smallest entry > val and insert that into the next row.
        """
        if row >= len(P):
            P.append([val])
            return row

        # Find position to insert (leftmost entry > val)
        insert_pos = -1
        for j, entry in enumerate(P[row]):
            if entry > val:
                insert_pos = j
                break

        if insert_pos == -1:
            # val goes at the end of this row
            P[row].append(val)
            return row
        else:
            # Bump the entry at insert_pos
            bumped = P[row][insert_pos]
            P[row][insert_pos] = val
            return RSKCorrespondence._row_insert(P, bumped, row + 1)

    @staticmethod
    def inverse(P: list[list[int]], Q: list[list[int]]) -> list[int]:
        """
        Inverse RSK: (P, Q) → permutation.

        Reverse the insertion process using Q to determine deletion order.

        Args:
            P: Insertion tableau.
            Q: Recording tableau.

        Returns:
            The permutation that produces (P, Q) under forward RSK.
        """
        P = [row[:] for row in P]  # Deep copy
        n = sum(len(row) for row in P)
        result = [0] * n

        # Process Q entries in reverse order
        entries_with_pos = []
        for i, row in enumerate(Q):
            for j, val in enumerate(row):
                entries_with_pos.append((val, i, j))
        entries_with_pos.sort(reverse=True)

        for q_val, q_row, q_col in entries_with_pos:
            # Remove the last cell from row q_row
            # (we need to find where the cell at (q_row, q_col) is)
            val = RSKCorrespondence._reverse_insert(P, q_row)
            result[q_val - 1] = val

        return result

    @staticmethod
    def _reverse_insert(P: list[list[int]], row: int) -> int:
        """Reverse insert from given row, bumping upward to row 0."""
        if row == 0:
            return P[0].pop()

        # Pop rightmost entry from current row
        val = P[row].pop()
        if not P[row]:
            P.pop()

        # Find the entry in row-1 that was bumped by val (largest entry < val)
        prev_row = P[row - 1]
        bump_pos = -1
        for j in range(len(prev_row) - 1, -1, -1):
            if prev_row[j] < val:
                bump_pos = j
                break

        if bump_pos >= 0:
            bumped = prev_row[bump_pos]
            prev_row[bump_pos] = val
            if row - 1 == 0:
                return bumped
            return RSKCorrespondence._reverse_insert_up(P, bumped, row - 2)

        return val

    @staticmethod
    def _reverse_insert_up(P: list[list[int]], val: int, row: int) -> int:
        """Continue reverse insertion upward."""
        if row < 0:
            return val

        prev_row = P[row]
        bump_pos = -1
        for j in range(len(prev_row) - 1, -1, -1):
            if prev_row[j] < val:
                bump_pos = j
                break

        if bump_pos >= 0:
            bumped = prev_row[bump_pos]
            prev_row[bump_pos] = val
            return RSKCorrespondence._reverse_insert_up(P, bumped, row - 1)

        return val

    @staticmethod
    def shape(tableau: list[list[int]]) -> list[int]:
        """Return the shape (partition) of a tableau."""
        return [len(row) for row in tableau]


# ============================================================================
# SCHUR FUNCTIONS
# ============================================================================

class SchurFunction:
    """
    Schur symmetric functions s_λ(x_1, ..., x_n).

    Computed via the ratio of alternants (Weyl character formula):
        s_λ(x) = a_{λ+δ}(x) / a_δ(x)
    where a_μ(x) = det(x_i^{μ_j}) and δ = (n-1, n-2, ..., 1, 0).
    """

    @staticmethod
    def compute(partition: list[int], variables: np.ndarray) -> float:
        """
        Evaluate the Schur function s_λ(x_1, ..., x_n).

        Uses the ratio of alternants formula.

        Args:
            partition: Partition λ (weakly decreasing).
            variables: Array of n variable values.

        Returns:
            s_λ(x_1, ..., x_n).
        """
        n = len(variables)
        partition = list(partition) + [0] * max(0, n - len(partition))
        partition = partition[:n]

        # δ = (n-1, n-2, ..., 0)
        delta = list(range(n - 1, -1, -1))

        # λ + δ
        lam_delta = [partition[i] + delta[i] for i in range(n)]

        # Numerator: det(x_i^{(λ+δ)_j})
        num_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                num_matrix[i, j] = variables[i] ** lam_delta[j]

        # Denominator: det(x_i^{δ_j}) = Vandermonde
        den_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                den_matrix[i, j] = variables[i] ** delta[j]

        num = np.linalg.det(num_matrix)
        den = np.linalg.det(den_matrix)

        if abs(den) < 1e-14:
            # Variables coincide; use L'Hopital / limit form
            logger.warning("Vandermonde determinant near zero; using tableau formula")
            return SchurFunction._tableau_formula(partition, variables)

        return num / den

    @staticmethod
    def _tableau_formula(partition: list[int], variables: np.ndarray) -> float:
        """
        Compute s_λ via SSYT summation (fallback for coincident variables).

        s_λ(x) = Σ_{T ∈ SSYT(λ)} Π x_{T(i,j)}
        """
        n = len(variables)
        partition = [p for p in partition if p > 0]
        if not partition:
            return 1.0

        # For small partitions, enumerate SSYTs directly
        total_cells = sum(partition)
        if total_cells > 12:
            logger.warning(f"Tableau formula for |λ|={total_cells} may be slow")

        ssyts = YoungTableauxOps.semistandard(partition, n)
        result = 0.0
        for T in ssyts:
            product = 1.0
            for row in T:
                for entry in row:
                    product *= variables[entry - 1]
            result += product
        return result


# ============================================================================
# HALL-LITTLEWOOD POLYNOMIALS
# ============================================================================

class HallLittlewood:
    """
    Hall-Littlewood symmetric polynomials P_λ(x; t).

    One-parameter family interpolating between Schur functions (t=0)
    and monomial symmetric functions (t=1).

    P_λ(x; t) = (1/v_λ(t)) Σ_{w ∈ S_n} w(x^λ Π_{i<j} (x_i - t x_j)/(x_i - x_j))

    where v_λ(t) = Π_i Π_{j=1}^{m_i} (1-t^j)/(1-t).
    """

    @staticmethod
    def compute(partition: list[int], t: float, variables: np.ndarray) -> float:
        """
        Evaluate the Hall-Littlewood polynomial P_λ(x; t).

        For t=0, reduces to Schur function s_λ(x).

        Args:
            partition: Partition λ.
            t: Deformation parameter.
            variables: Array of variable values.

        Returns:
            P_λ(x_1, ..., x_n; t).
        """
        if abs(t) < 1e-14:
            return SchurFunction.compute(partition, variables)

        n = len(variables)
        partition = list(partition) + [0] * max(0, n - len(partition))
        partition = partition[:n]

        # Compute v_λ(t)
        v_lambda = HallLittlewood._v_factor(partition, t)

        # Sum over permutations of S_n
        from itertools import permutations
        total = 0.0

        for perm in permutations(range(n)):
            # Monomial x^{λ[perm]}
            mono = 1.0
            for i in range(n):
                mono *= variables[perm[i]] ** partition[i]

            # Product Π_{i<j} (x_{σ(i)} - t*x_{σ(j)}) / (x_{σ(i)} - x_{σ(j)})
            ratio_prod = 1.0
            for i in range(n):
                for j in range(i + 1, n):
                    xi = variables[perm[i]]
                    xj = variables[perm[j]]
                    denom = xi - xj
                    if abs(denom) < 1e-14:
                        continue  # Skip degenerate pairs
                    numer = xi - t * xj
                    ratio_prod *= numer / denom

            total += mono * ratio_prod

        if abs(v_lambda) < 1e-14:
            return 0.0

        return total / v_lambda

    @staticmethod
    def _v_factor(partition: list[int], t: float) -> float:
        """
        Compute v_λ(t) = Π_i Π_{j=1}^{m_i(λ)} (1 - t^j) / (1 - t).
        where m_i(λ) is the multiplicity of i in λ.
        """
        if abs(1 - t) < 1e-14:
            # At t=1: v_λ(1) = Π m_i!
            counts: dict[int, int] = {}
            for p in partition:
                counts[p] = counts.get(p, 0) + 1
            result = 1.0
            for m in counts.values():
                result *= np.math.factorial(m)
            return result

        counts: dict[int, int] = {}
        for p in partition:
            if p > 0:
                counts[p] = counts.get(p, 0) + 1
        # Also count zeros
        zero_count = sum(1 for p in partition if p == 0)
        if zero_count > 0:
            counts[0] = zero_count

        result = 1.0
        for part_val, mult in counts.items():
            for j in range(1, mult + 1):
                result *= (1 - t ** j) / (1 - t)

        return result


# ============================================================================
# DISCRETE MORSE INEQUALITIES
# ============================================================================

class DiscreteMorseInequalities:
    """
    Discrete Morse theory on simplicial complexes.

    A discrete Morse function f assigns values to simplices such that
    for each p-simplex σ:
    - at most one (p+1)-face τ > σ has f(τ) ≤ f(σ)
    - at most one (p-1)-face ν < σ has f(ν) ≥ f(σ)

    The weak Morse inequalities state:
        m_k ≥ β_k  for all k
    where m_k = number of critical k-simplices and β_k = k-th Betti number.

    The strong Morse inequalities:
        Σ_{j=0}^{k} (-1)^{k-j} m_j ≥ Σ_{j=0}^{k} (-1)^{k-j} β_j
    """

    @staticmethod
    def check(complex_faces: dict[int, list], morse_fn: dict[int, float]) -> bool:
        """
        Verify that the weak Morse inequalities hold for a given discrete
        Morse function on a simplicial complex.

        Args:
            complex_faces: Dict mapping dimension k -> list of k-simplices.
                           Each simplex is a tuple of vertex indices.
            morse_fn: Dict mapping simplex index -> Morse function value.
                      Index is (dim, position_in_list).

        Returns:
            True if weak Morse inequalities m_k ≥ β_k hold for all k.
        """
        # Count critical simplices per dimension
        critical_counts = {}
        for dim, faces in complex_faces.items():
            count = 0
            for idx, face in enumerate(faces):
                if DiscreteMorseInequalities._is_critical(
                    complex_faces, morse_fn, dim, idx
                ):
                    count += 1
            critical_counts[dim] = count

        # Compute Betti numbers via boundary matrices
        betti = DiscreteMorseInequalities._compute_betti(complex_faces)

        # Check weak inequalities
        max_dim = max(list(complex_faces.keys()) + [0])
        for k in range(max_dim + 1):
            m_k = critical_counts.get(k, 0)
            b_k = betti.get(k, 0)
            if m_k < b_k:
                logger.warning(
                    f"Weak Morse inequality violated: m_{k}={m_k} < β_{k}={b_k}"
                )
                return False

        return True

    @staticmethod
    def _is_critical(complex_faces: dict[int, list],
                     morse_fn: dict, dim: int, idx: int) -> bool:
        """Check if simplex (dim, idx) is critical (no paired face)."""
        key = (dim, idx)
        if key not in morse_fn:
            return True  # Unassigned = critical by default

        f_val = morse_fn[key]

        # Check for a coface with lower or equal value
        if dim + 1 in complex_faces:
            for cidx, coface in enumerate(complex_faces[dim + 1]):
                ckey = (dim + 1, cidx)
                if ckey in morse_fn and morse_fn[ckey] <= f_val:
                    return False  # Paired with a coface

        # Check for a face with higher or equal value
        if dim - 1 in complex_faces:
            for fidx, face in enumerate(complex_faces[dim - 1]):
                fkey = (dim - 1, fidx)
                if fkey in morse_fn and morse_fn[fkey] >= f_val:
                    return False  # Paired with a face

        return True

    @staticmethod
    def _compute_betti(complex_faces: dict[int, list]) -> dict[int, int]:
        """
        Compute Betti numbers from boundary matrices.

        β_k = dim(ker(∂_k)) - dim(im(∂_{k+1}))
        """
        if not complex_faces:
            return {}

        max_dim = max(complex_faces.keys())
        betti = {}

        for k in range(max_dim + 1):
            faces_k = complex_faces.get(k, [])
            n_k = len(faces_k)

            if n_k == 0:
                betti[k] = 0
                continue

            # Build boundary matrix ∂_k
            faces_km1 = complex_faces.get(k - 1, [])
            n_km1 = len(faces_km1)

            if k == 0 or n_km1 == 0:
                ker_dim = n_k
            else:
                boundary_k = DiscreteMorseInequalities._boundary_matrix(
                    faces_k, faces_km1, k
                )
                ker_dim = n_k - int(np.linalg.matrix_rank(boundary_k))

            # Image of ∂_{k+1}
            faces_kp1 = complex_faces.get(k + 1, [])
            if not faces_kp1:
                im_dim = 0
            else:
                boundary_kp1 = DiscreteMorseInequalities._boundary_matrix(
                    faces_kp1, faces_k, k + 1
                )
                im_dim = int(np.linalg.matrix_rank(boundary_kp1))

            betti[k] = ker_dim - im_dim

        return betti

    @staticmethod
    def _boundary_matrix(faces_k: list, faces_km1: list, k: int) -> np.ndarray:
        """Build the boundary matrix ∂_k mapping k-chains to (k-1)-chains."""
        n_k = len(faces_k)
        n_km1 = len(faces_km1)
        boundary = np.zeros((n_km1, n_k))

        # Index (k-1)-faces for fast lookup
        face_index = {}
        for idx, f in enumerate(faces_km1):
            face_index[tuple(sorted(f))] = idx

        for j, sigma in enumerate(faces_k):
            sigma_sorted = sorted(sigma)
            for i in range(len(sigma_sorted)):
                # Remove the i-th vertex
                face = tuple(sigma_sorted[:i] + sigma_sorted[i + 1:])
                if face in face_index:
                    boundary[face_index[face], j] = (-1) ** i

        return boundary

    @staticmethod
    def euler_characteristic(complex_faces: dict[int, list]) -> int:
        """
        Compute Euler characteristic χ = Σ (-1)^k |faces_k|.
        Also equals Σ (-1)^k β_k by Morse theory.
        """
        chi = 0
        for k, faces in complex_faces.items():
            chi += ((-1) ** k) * len(faces)
        return chi
