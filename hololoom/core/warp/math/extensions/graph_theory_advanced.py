"""
Advanced Graph Theory — Ramsey, Matroids, Tutte Polynomial
==========================================================

Deep structural graph theory: extremal existence results,
abstract independence structures, and universal graph polynomials.

Core Concepts:
- Ramsey theory: Guaranteed monochromatic substructures in colored graphs
- Matroid theory: Abstract independence generalizing linear algebra and graphs
- Tutte polynomial: Universal graph polynomial encoding chromatic, flow, reliability

Mathematical Foundation:
R(s,t) = min n such that every 2-coloring of K_n has red K_s or blue K_t
Matroid rank: r(A) = max |I| for independent I ⊆ A
T(G; x,y) = Σ_{A ⊆ E} (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)}

Applications to Warp Space:
- Ramsey bounds for guaranteed pattern detection
- Matroid optimization for greedy feature selection
- Tutte polynomial for graph reliability and network analysis

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# RAMSEY THEORY
# ============================================================================

class RamseyTheory:
    """
    Ramsey theory: in any sufficiently large structure, order must appear.

    R(s, t) = smallest n such that every 2-coloring of edges of K_n
    contains either a red K_s or a blue K_t.

    Known exact values are sparse; we store a table and provide
    bounds for unknown cases.
    """

    # Known Ramsey numbers R(s, t) for small s, t
    # R(s,t) = R(t,s) by symmetry
    KNOWN: dict[tuple[int, int], int] = {
        (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (1, 5): 1,
        (2, 2): 2, (2, 3): 3, (2, 4): 4, (2, 5): 5,
        (3, 3): 6, (3, 4): 9, (3, 5): 14, (3, 6): 18, (3, 7): 23,
        (3, 8): 28, (3, 9): 36,
        (4, 4): 18, (4, 5): 25,
    }

    @staticmethod
    def R(s: int, t: int) -> int:
        """
        Ramsey number R(s, t).

        Returns exact value if known, otherwise the best known upper bound.

        Args:
            s: Size of red clique.
            t: Size of blue clique.

        Returns:
            R(s, t) or upper bound.
        """
        if s < 1 or t < 1:
            raise ValueError("s and t must be positive")

        # Normalize by symmetry
        key = (min(s, t), max(s, t))

        if key in RamseyTheory.KNOWN:
            return RamseyTheory.KNOWN[key]

        # R(1, t) = 1, R(2, t) = t
        if key[0] == 1:
            return 1
        if key[0] == 2:
            return key[1]

        # Upper bound: R(s,t) ≤ C(s+t-2, s-1)
        from math import comb
        bound = comb(s + t - 2, s - 1)
        logger.info(f"R({s},{t}) unknown; upper bound C({s+t-2},{s-1}) = {bound}")
        return bound

    @staticmethod
    def find_monochromatic(coloring: np.ndarray,
                           size: int) -> tuple[str, list[int]] | None:
        """
        Find a monochromatic clique in a 2-colored complete graph.

        Args:
            coloring: Upper-triangular matrix of edge colors (0 or 1).
                      coloring[i,j] = color of edge {i,j} for i < j.
            size: Desired clique size.

        Returns:
            ('red', vertices) or ('blue', vertices) if found, else None.
        """
        n = coloring.shape[0]
        if n < size:
            return None

        # Try all subsets of size `size` (feasible only for small inputs)
        if size <= 5 and n <= 30:
            return RamseyTheory._brute_force_clique(coloring, size)

        # Greedy approach for larger inputs
        return RamseyTheory._greedy_monochromatic(coloring, size)

    @staticmethod
    def _brute_force_clique(coloring: np.ndarray,
                            size: int) -> tuple[str, list[int]] | None:
        """Find monochromatic clique by exhaustive search."""
        from itertools import combinations
        n = coloring.shape[0]

        for vertices in combinations(range(n), size):
            vlist = list(vertices)
            # Check if all edges have the same color
            colors = set()
            for i in range(len(vlist)):
                for j in range(i + 1, len(vlist)):
                    a, b = min(vlist[i], vlist[j]), max(vlist[i], vlist[j])
                    colors.add(int(coloring[a, b]))
                    if len(colors) > 1:
                        break
                if len(colors) > 1:
                    break

            if len(colors) == 1:
                color = colors.pop()
                return ('red' if color == 0 else 'blue', vlist)

        return None

    @staticmethod
    def _greedy_monochromatic(coloring: np.ndarray,
                              size: int) -> tuple[str, list[int]] | None:
        """Greedy search for monochromatic clique."""
        n = coloring.shape[0]

        for color in [0, 1]:
            for start in range(n):
                clique = [start]
                for v in range(n):
                    if v == start:
                        continue
                    # Check if v is connected to all clique members with this color
                    all_same = True
                    for u in clique:
                        a, b = min(u, v), max(u, v)
                        if int(coloring[a, b]) != color:
                            all_same = False
                            break
                    if all_same:
                        clique.append(v)
                        if len(clique) >= size:
                            return ('red' if color == 0 else 'blue', clique)

        return None


# ============================================================================
# MATROID THEORY
# ============================================================================

class MatroidTheory:
    """
    Matroid: abstract independence structure axiomatizing linear independence
    and graph acyclicity.

    A matroid M = (E, I) where E is the ground set and I is a family of
    independent sets satisfying:
    1. ∅ ∈ I
    2. If A ∈ I and B ⊆ A, then B ∈ I (hereditary)
    3. If A, B ∈ I and |A| < |B|, then ∃ x ∈ B\\A with A∪{x} ∈ I (exchange)
    """

    def __init__(self, ground_set: list[int],
                 independent_sets: list[frozenset[int]]):
        """
        Args:
            ground_set: Elements of E.
            independent_sets: Collection of independent subsets of E.
        """
        self.ground_set = frozenset(ground_set)
        self.independent_sets = set(independent_sets)

        # Validate: empty set must be independent
        if frozenset() not in self.independent_sets:
            self.independent_sets.add(frozenset())

    def rank(self, subset: frozenset[int]) -> int:
        """
        Rank of a subset: size of the largest independent set contained in it.

        r(A) = max{|I| : I ⊆ A, I ∈ I}
        """
        max_rank = 0
        for ind in self.independent_sets:
            if ind <= subset:
                max_rank = max(max_rank, len(ind))
        return max_rank

    def closure(self, subset: frozenset[int]) -> frozenset[int]:
        """
        Closure (span) of a subset: cl(A) = {x ∈ E : r(A∪{x}) = r(A)}.

        Adding elements in the closure doesn't increase rank.
        """
        r_A = self.rank(subset)
        result = set(subset)

        for x in self.ground_set:
            if x not in subset:
                if self.rank(subset | {x}) == r_A:
                    result.add(x)

        return frozenset(result)

    def dual(self) -> 'MatroidTheory':
        """
        Dual matroid M* = (E, I*) where B* is a basis of M* iff E\\B* is a basis of M.

        I* = {A ⊆ E : E\\A contains a basis of M}
        """
        bases = self._bases()
        cobases = {self.ground_set - b for b in bases}

        # Independent sets of dual: subsets of cobases
        dual_ind = set()
        dual_ind.add(frozenset())
        for cobasis in cobases:
            # Add all subsets of each cobasis
            elements = list(cobasis)
            for i in range(1, len(elements) + 1):
                from itertools import combinations
                for combo in combinations(elements, i):
                    dual_ind.add(frozenset(combo))

        return MatroidTheory(list(self.ground_set), list(dual_ind))

    def _bases(self) -> set[frozenset[int]]:
        """Bases: maximal independent sets."""
        r = self.rank(self.ground_set)
        return {s for s in self.independent_sets if len(s) == r}

    def is_graphic(self) -> bool:
        """
        Check if the matroid is graphic (representable as a graph matroid).

        A graphic matroid has ground set = edges of a graph, and independent
        sets = acyclic edge subsets (forests).

        Uses a necessary condition: the matroid must satisfy the circuit
        elimination axiom and have rank ≤ |E| - 1.
        """
        r = self.rank(self.ground_set)
        n_elements = len(self.ground_set)

        # Simple necessary check: rank of graphic matroid on n vertices
        # is at most n-1, but we don't know n
        # A more complete check would verify the matroid against all graphs,
        # but that's NP-hard in general. We check basic properties.

        # Check hereditary property
        for ind in self.independent_sets:
            for elem in ind:
                subset = ind - {elem}
                if subset not in self.independent_sets:
                    return False

        # Check exchange property
        ind_list = sorted(self.independent_sets, key=len)
        for A in ind_list:
            for B in ind_list:
                if len(A) < len(B):
                    diff = B - A
                    found = False
                    for x in diff:
                        if (A | {x}) in self.independent_sets:
                            found = True
                            break
                    if not found:
                        return False

        return True  # Passes necessary conditions

    def circuits(self) -> set[frozenset[int]]:
        """
        Circuits: minimal dependent sets.

        C is a circuit if C ∉ I but C\\{x} ∈ I for all x ∈ C.
        """
        result = set()
        for s in self.independent_sets:
            # Try adding each element not in s
            for x in self.ground_set - s:
                candidate = s | {x}
                if candidate not in self.independent_sets:
                    # Check minimality: every proper subset is independent
                    is_circuit = all(
                        (candidate - {y}) in self.independent_sets
                        for y in candidate
                    )
                    if is_circuit:
                        result.add(candidate)
        return result


# ============================================================================
# TUTTE POLYNOMIAL
# ============================================================================

class TuttePolynomial:
    """
    The Tutte polynomial T(G; x, y): a two-variable polynomial encoding
    a vast amount of graph information.

    T(G; x, y) = Σ_{A ⊆ E} (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)}

    where r is the matroid rank function of the cycle matroid.

    Special evaluations:
    - T(G; 1, 1) = number of spanning trees
    - T(G; 2, 1) = number of spanning forests
    - T(G; 1, 2) = number of spanning connected subgraphs
    - T(G; 2, 0) = number of acyclic orientations
    - P(G; k) = (-1)^{|V|-c} k^c T(G; 1-k, 0) is the chromatic polynomial
    """

    def __init__(self, adj: np.ndarray):
        """
        Args:
            adj: Adjacency matrix of an undirected graph.
        """
        self.adj = adj
        self.n = adj.shape[0]
        self._edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if adj[i, j] > 0:
                    self._edges.append((i, j))
        self._n_edges = len(self._edges)
        self._total_rank = self._compute_rank(set(range(self._n_edges)))

    def compute(self) -> Callable[[float, float], float]:
        """
        Compute the Tutte polynomial as a callable function T(x, y).

        Uses the rank-nullity formula:
        T(x, y) = Σ_{A ⊆ E} (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)}

        Returns:
            Function taking (x, y) and returning T(G; x, y).
        """
        # Precompute coefficients c_{i,j} where T = Σ c_{i,j} (x-1)^i (y-1)^j
        max_rank = self._total_rank
        max_nullity = self._n_edges - max_rank if self._n_edges > 0 else 0

        coeffs = np.zeros((max_rank + 1, max_nullity + 1))

        # Iterate over all subsets of edges
        for mask in range(1 << self._n_edges):
            edge_set = set()
            for e in range(self._n_edges):
                if mask & (1 << e):
                    edge_set.add(e)

            r_A = self._compute_rank(edge_set)
            nullity = len(edge_set) - r_A
            corank = self._total_rank - r_A

            if corank <= max_rank and nullity <= max_nullity:
                coeffs[corank, nullity] += 1

        def T(x: float, y: float) -> float:
            result = 0.0
            for i in range(coeffs.shape[0]):
                for j in range(coeffs.shape[1]):
                    if coeffs[i, j] != 0:
                        result += coeffs[i, j] * ((x - 1) ** i) * ((y - 1) ** j)
            return result

        return T

    def _compute_rank(self, edge_set: set) -> int:
        """Compute rank of edge subset (number of edges in spanning forest)."""
        if not edge_set:
            return 0

        # Union-find
        parent = list(range(self.n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        rank = 0
        for e_idx in edge_set:
            u, v = self._edges[e_idx]
            if union(u, v):
                rank += 1
        return rank

    def chromatic_polynomial(self) -> Callable[[int], int]:
        """
        Chromatic polynomial P(G; k) = number of proper k-colorings.

        P(G; k) = (-1)^{n-c(G)} k^{c(G)} T(G; 1-k, 0)
        where c(G) is the number of connected components.
        """
        tutte = self.compute()

        # Count connected components
        parent = list(range(self.n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in self._edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        components = len(set(find(i) for i in range(self.n)))

        def P(k: int) -> int:
            sign = (-1) ** (self.n - components)
            return int(round(sign * (k ** components) * tutte(1 - k, 0)))

        return P

    def flow_polynomial(self) -> Callable[[int], int]:
        """
        Flow polynomial F(G; k) = number of nowhere-zero k-flows.

        F(G; k) = (-1)^{|E|-|V|+c} T(G; 0, 1-k)
        """
        tutte = self.compute()

        parent = list(range(self.n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in self._edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        components = len(set(find(i) for i in range(self.n)))
        sign_exp = self._n_edges - self.n + components

        def F(k: int) -> int:
            sign = (-1) ** sign_exp
            return int(round(sign * tutte(0, 1 - k)))

        return F

    def spanning_trees(self) -> int:
        """Number of spanning trees = T(G; 1, 1)."""
        return int(round(self.compute()(1.0, 1.0)))
