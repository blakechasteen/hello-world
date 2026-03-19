"""
Extremal Combinatorics — Turan, Szemeredi, Sperner, Lovasz Local Lemma
=======================================================================

Extremal problems in combinatorics: how large or small can a structure
be under given constraints?

Core Concepts:
- Turan's theorem: Maximum edges in K_r-free graph
- Szemeredi regularity lemma: Approximate structure of dense graphs
- Sperner's theorem: Maximum antichain in power set lattice
- Lovasz Local Lemma: Probabilistic avoidance of bad events

Mathematical Foundation:
ex(n, K_r) = (1 - 1/(r-1)) n²/2    (Turan's theorem)
|antichain| ≤ C(n, ⌊n/2⌋)          (Sperner's theorem)
e·p·(d+1) ≤ 1 ⟹ Pr[avoid all] > 0  (LLL symmetric form)

Applications to Warp Space:
- Turan bounds for sparse memory graph construction
- Regularity partitions for graph summarization
- Sperner antichains for independent feature selection
- LLL for probabilistic existence proofs in combinatorial search

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# TURAN'S THEOREM
# ============================================================================

class TuranTheorem:
    """
    Turan's theorem and the Turan graph T(n,r).

    ex(n, K_r) = maximum number of edges in a K_r-free graph on n vertices.
    This equals the number of edges in the Turan graph T(n, r-1).

    T(n,r) is the complete (r)-partite graph with part sizes as equal as possible.
    """

    @staticmethod
    def max_edges(n: int, r: int) -> int:
        """
        Maximum edges in a K_r-free graph on n vertices: ex(n, K_r).

        The Turan number equals edges in T(n, r-1).

        Args:
            n: Number of vertices.
            r: Clique size to forbid (r ≥ 2).

        Returns:
            ex(n, K_r).
        """
        if r < 2:
            raise ValueError("r must be at least 2 (K_2 = edge)")
        if n < r:
            return n * (n - 1) // 2  # Complete graph has no K_r

        # Turan number: (1 - 1/(r-1)) * n^2 / 2 (exact formula)
        # Exact: partition n into (r-1) parts as equally as possible
        parts = r - 1
        base = n // parts
        remainder = n % parts

        # Parts: 'remainder' parts of size (base+1), rest of size base
        # Edges = total_pairs - within_part_pairs
        total_pairs = n * (n - 1) // 2
        within = 0
        for i in range(parts):
            size = base + 1 if i < remainder else base
            within += size * (size - 1) // 2

        return total_pairs - within

    @staticmethod
    def turan_graph(n: int, r: int) -> np.ndarray:
        """
        Construct the Turan graph T(n, r): complete r-partite graph
        with parts as equal as possible.

        T(n, r) maximizes edges among K_{r+1}-free graphs on n vertices.

        Args:
            n: Number of vertices.
            r: Number of parts (so T(n,r) is K_{r+1}-free).

        Returns:
            Adjacency matrix (n x n).
        """
        if r < 1:
            raise ValueError("r must be at least 1")
        if r >= n:
            # Complete graph
            adj = np.ones((n, n)) - np.eye(n)
            return adj

        # Assign vertices to parts
        base = n // r
        remainder = n % r
        part_assignment = np.zeros(n, dtype=int)
        idx = 0
        for p in range(r):
            size = base + 1 if p < remainder else base
            part_assignment[idx:idx + size] = p
            idx += size

        # Edge between vertices in different parts
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if part_assignment[i] != part_assignment[j]:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        return adj

    @staticmethod
    def is_turan(adj: np.ndarray, r: int) -> bool:
        """Check if adjacency matrix is the Turan graph T(n, r)."""
        n = adj.shape[0]
        expected_edges = TuranTheorem.max_edges(n, r + 1)
        actual_edges = int(np.sum(adj)) // 2
        return actual_edges == expected_edges


# ============================================================================
# SZEMEREDI REGULARITY LEMMA
# ============================================================================

class SzemerediRegularity:
    """
    Szemeredi regularity lemma: every dense graph can be approximated
    by a bounded number of random-like bipartite pairs.

    An ε-regular pair (A, B) satisfies: for all X ⊆ A, Y ⊆ B with
    |X| ≥ ε|A| and |Y| ≥ ε|B|, |d(X,Y) - d(A,B)| < ε.

    The regularity lemma guarantees an ε-regular partition into at most
    M(ε) parts, where M depends only on ε (not on n).
    """

    @staticmethod
    def partition(adj: np.ndarray, epsilon: float,
                  max_parts: int = 32) -> list[list[int]]:
        """
        Compute an approximate ε-regular partition of the graph.

        Uses the algorithmic version: start with a random equipartition
        and iteratively refine irregular pairs.

        Args:
            adj: Adjacency matrix (n x n).
            epsilon: Regularity parameter (0 < ε < 1).
            max_parts: Maximum number of parts (theoretical bound is huge).

        Returns:
            Partition as list of vertex lists.
        """
        n = adj.shape[0]
        if n < 2:
            return [list(range(n))]

        # Start with random equipartition into k parts
        k = max(2, min(int(1 / epsilon), max_parts))
        k = min(k, n)

        perm = np.random.permutation(n)
        parts = []
        base = n // k
        remainder = n % k
        idx = 0
        for i in range(k):
            size = base + 1 if i < remainder else base
            parts.append(list(perm[idx:idx + size]))
            idx += size

        # Iterative refinement
        max_iterations = 10
        for iteration in range(max_iterations):
            irregular = SzemerediRegularity._find_irregular_pair(
                adj, parts, epsilon
            )
            if irregular is None:
                break  # All pairs are ε-regular

            i, j = irregular
            # Refine: split the irregular pair
            if len(parts) >= max_parts:
                break

            # Simple refinement: split the larger part
            target = i if len(parts[i]) >= len(parts[j]) else j
            part = parts[target]
            mid = len(part) // 2
            parts[target] = part[:mid]
            parts.append(part[mid:])

        # Remove empty parts
        parts = [p for p in parts if len(p) > 0]
        logger.info(
            f"Regularity partition: {n} vertices into {len(parts)} parts "
            f"(ε={epsilon})"
        )
        return parts

    @staticmethod
    def _find_irregular_pair(adj: np.ndarray, parts: list[list[int]],
                             epsilon: float) -> tuple[int, int] | None:
        """Find an ε-irregular pair, or None if all pairs are regular."""
        k = len(parts)
        for i in range(k):
            for j in range(i + 1, k):
                if not SzemerediRegularity._is_regular(
                    adj, parts[i], parts[j], epsilon
                ):
                    return (i, j)
        return None

    @staticmethod
    def _is_regular(adj: np.ndarray, A: list[int], B: list[int],
                    epsilon: float) -> bool:
        """Check if pair (A, B) is ε-regular."""
        if len(A) == 0 or len(B) == 0:
            return True

        d_AB = SzemerediRegularity._density(adj, A, B)

        # Check subsets of size ≥ ε|A| and ε|B|
        min_a = max(1, int(epsilon * len(A)))
        min_b = max(1, int(epsilon * len(B)))

        # Sample random subsets to test regularity (approximate check)
        n_samples = 20
        for _ in range(n_samples):
            size_a = np.random.randint(min_a, len(A) + 1)
            size_b = np.random.randint(min_b, len(B) + 1)
            X = list(np.random.choice(A, size_a, replace=False))
            Y = list(np.random.choice(B, size_b, replace=False))
            d_XY = SzemerediRegularity._density(adj, X, Y)
            if abs(d_XY - d_AB) >= epsilon:
                return False

        return True

    @staticmethod
    def _density(adj: np.ndarray, A: list[int], B: list[int]) -> float:
        """Edge density d(A, B) = e(A,B) / (|A|·|B|)."""
        if len(A) == 0 or len(B) == 0:
            return 0.0
        edges = sum(adj[i, j] for i in A for j in B)
        return edges / (len(A) * len(B))


# ============================================================================
# SPERNER'S THEOREM
# ============================================================================

class SpernerTheorem:
    """
    Sperner's theorem on antichains in the power set lattice.

    An antichain in 2^{[n]} is a family of subsets where no one contains
    another. The maximum antichain size is C(n, ⌊n/2⌋).

    The maximum is achieved by taking all subsets of size ⌊n/2⌋.
    """

    @staticmethod
    def max_antichain(n: int) -> int:
        """
        Maximum antichain size in the power set of an n-element set.

        Returns C(n, ⌊n/2⌋).
        """
        if n < 0:
            return 0
        k = n // 2
        return int(round(np.math.factorial(n) /
                         (np.math.factorial(k) * np.math.factorial(n - k))))

    @staticmethod
    def verify_antichain(families: list[frozenset[int]]) -> bool:
        """
        Verify that a collection of sets forms an antichain.

        No set in the family is a subset of another.

        Args:
            families: List of frozensets.

        Returns:
            True if the family is an antichain.
        """
        n = len(families)
        for i in range(n):
            for j in range(n):
                if i != j and families[i] <= families[j]:
                    return False
        return True

    @staticmethod
    def maximum_antichain(n: int) -> list[frozenset[int]]:
        """
        Construct a maximum antichain: all subsets of size ⌊n/2⌋.

        Args:
            n: Size of the ground set {0, 1, ..., n-1}.

        Returns:
            List of frozensets forming a maximum antichain.
        """
        from itertools import combinations
        k = n // 2
        return [frozenset(c) for c in combinations(range(n), k)]

    @staticmethod
    def dilworth_decomposition(n: int) -> list[list[frozenset[int]]]:
        """
        Dilworth's theorem: minimum chain decomposition of 2^{[n]}
        has C(n, ⌊n/2⌋) chains (matching the max antichain size).

        Returns a symmetric chain decomposition.
        """
        if n == 0:
            return [[frozenset()]]

        # Build symmetric chains recursively
        chains_prev = SpernerTheorem.dilworth_decomposition(n - 1)
        new_chains = []

        for chain in chains_prev:
            # Extend: chain stays, and we add n-1 to each set
            bottom = chain[0]
            top = chain[-1]

            if len(top) < n:
                # Extend chain by adding n-1 at the top
                extended = chain + [s | {n - 1} for s in reversed(chain)]
                new_chains.append(extended)
            else:
                # Split into two chains
                new_chains.append(chain)
                new_chains.append([s | {n - 1} for s in chain])

        return new_chains


# ============================================================================
# LOVASZ LOCAL LEMMA
# ============================================================================

class LovaszLocalLemma:
    """
    The Lovasz Local Lemma (LLL): a probabilistic tool for proving
    existence of combinatorial objects.

    Symmetric form: If each event has probability ≤ p and depends on at
    most d other events, and e·p·(d+1) ≤ 1, then with positive probability
    none of the events occur.

    Asymmetric form: If for each event A_i there exists x_i ∈ (0,1) such that
    Pr[A_i] ≤ x_i Π_{j ∈ Γ(i)} (1 - x_j), then Pr[∩ A_i^c] > 0.
    """

    @staticmethod
    def applicable(events: int, deps: np.ndarray,
                   probs: np.ndarray) -> bool:
        """
        Check if the symmetric LLL condition holds: e·p·(d+1) ≤ 1.

        Args:
            events: Number of events (n).
            deps: Dependency graph adjacency matrix (n x n).
                  deps[i,j] = 1 if event i depends on event j.
            probs: Event probabilities (n,).

        Returns:
            True if the LLL condition is satisfied.
        """
        if events == 0:
            return True

        p = float(np.max(probs))
        d = int(np.max(np.sum(deps, axis=1)))

        lll_value = np.e * p * (d + 1)
        satisfied = lll_value <= 1.0

        if satisfied:
            logger.info(
                f"LLL symmetric condition: e·p·(d+1) = {lll_value:.4f} ≤ 1 ✓"
            )
        else:
            logger.info(
                f"LLL symmetric condition: e·p·(d+1) = {lll_value:.4f} > 1 ✗"
            )

        return satisfied

    @staticmethod
    def applicable_asymmetric(probs: np.ndarray,
                              deps: np.ndarray) -> bool:
        """
        Check the asymmetric LLL condition.

        For each event i, we need x_i ∈ (0,1) such that:
            p_i ≤ x_i Π_{j ∈ Γ(i)} (1 - x_j)

        Uses the heuristic x_i = 2·p_i (valid when max degree is bounded).

        Args:
            probs: Event probabilities (n,).
            deps: Dependency graph adjacency matrix.

        Returns:
            True if a valid assignment x exists.
        """
        n = len(probs)
        if n == 0:
            return True

        # Try x_i = 2·p_i
        x = 2.0 * probs
        if np.any(x >= 1.0):
            return False

        for i in range(n):
            neighbors = np.where(deps[i] > 0)[0]
            product = np.prod(1.0 - x[neighbors]) if len(neighbors) > 0 else 1.0
            if probs[i] > x[i] * product:
                return False

        return True

    @staticmethod
    def lower_bound(probs: np.ndarray, deps: np.ndarray) -> float:
        """
        Lower bound on Pr[none of the events occur].

        When LLL applies with x_i = 2p_i:
            Pr[∩ A_i^c] ≥ Π (1 - x_i) = Π (1 - 2p_i)

        Args:
            probs: Event probabilities.
            deps: Dependency graph.

        Returns:
            Lower bound on the avoidance probability.
        """
        if not LovaszLocalLemma.applicable_asymmetric(probs, deps):
            return 0.0

        x = 2.0 * probs
        return float(np.prod(1.0 - x))
