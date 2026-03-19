"""
Graph Coloring — Chromatic Numbers, Conflict Scheduling
=======================================================

Vertex coloring algorithms for conflict resolution and resource scheduling.

Core Concepts:
- Graph Coloring: Assign colors to vertices so no adjacent vertices share a color
- Chromatic Number: Minimum number of colors needed (NP-hard in general)
- DSatur: Degree of Saturation heuristic (best practical vertex coloring)
- Conflict Graphs: Transform scheduling/resource problems into coloring problems

Mathematical Foundation:
Chromatic number chi(G) satisfies:
    omega(G) <= chi(G) <= Delta(G) + 1   (clique number <= chi <= max_degree + 1)
    chi(G) <= 1 + max_i min(d(v_i), i)   (greedy bound with ordering)
Chromatic polynomial P(G, k) = number of proper k-colorings
    P(K_n, k) = k(k-1)(k-2)...(k-n+1)   (complete graph)
    P(G, k) = P(G-e, k) - P(G/e, k)     (deletion-contraction)

Applications to Warp Space:
- Resource conflict resolution in multi-agent scheduling
- Channel assignment in federation gossip protocols
- Register allocation analogy for memory slot assignment
- Independent set extraction for parallel batch processing

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

try:
    from scipy.sparse import csr_matrix, issparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ColoringResult:
    """Result of a graph coloring algorithm.

    Attributes:
        colors: Mapping from vertex index to color index (0-based).
        n_colors: Number of distinct colors used.
        is_optimal: True if the coloring is provably optimal (uses chi(G) colors).
    """
    colors: dict[int, int]
    n_colors: int
    is_optimal: bool

    @property
    def color_classes(self) -> dict[int, list[int]]:
        """Group vertices by color assignment."""
        classes: dict[int, list[int]] = {}
        for vertex, color in self.colors.items():
            if color not in classes:
                classes[color] = []
            classes[color].append(vertex)
        return classes

    @property
    def largest_color_class(self) -> int:
        """Size of the largest independent set (color class)."""
        classes = self.color_classes
        if not classes:
            return 0
        return max(len(v) for v in classes.values())

    def utilization(self) -> float:
        """Fraction of vertices that share a color with at least one other vertex.

        High utilization = efficient packing of vertices into color classes.
        """
        classes = self.color_classes
        shared = sum(len(v) for v in classes.values() if len(v) > 1)
        total = len(self.colors)
        return shared / total if total > 0 else 0.0


# ============================================================================
# GRAPH COLORING ALGORITHMS
# ============================================================================

class GraphColoring:
    """
    Vertex coloring algorithms for undirected graphs.

    All methods take an adjacency matrix (dense numpy array or compatible)
    where adj[i][j] != 0 indicates an edge between vertices i and j.
    The graph is treated as undirected (adj is symmetrized internally).
    """

    @staticmethod
    def greedy(
        adj: np.ndarray,
        ordering: list[int] | None = None,
    ) -> ColoringResult:
        """
        Sequential greedy graph coloring.

        Process vertices in the given order. Assign each vertex the
        smallest color not used by its already-colored neighbors.

        Guarantees at most Delta(G) + 1 colors, but quality depends
        heavily on vertex ordering.

        Args:
            adj: Adjacency matrix, shape (n, n).
            ordering: Vertex processing order. If None, uses natural order [0..n-1].

        Returns:
            ColoringResult with the greedy coloring.
        """
        adj = GraphColoring._normalize_adj(adj)
        n = adj.shape[0]

        if ordering is None:
            ordering = list(range(n))

        colors: dict[int, int] = {}

        for v in ordering:
            # Find colors used by already-colored neighbors
            neighbor_colors: set[int] = set()
            for u in range(n):
                if adj[v, u] != 0 and u in colors:
                    neighbor_colors.add(colors[u])

            # Assign smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            colors[v] = color

        n_colors = len(set(colors.values())) if colors else 0

        return ColoringResult(
            colors=colors,
            n_colors=n_colors,
            is_optimal=False,
        )

    @staticmethod
    def welsh_powell(adj: np.ndarray) -> ColoringResult:
        """
        Welsh-Powell coloring: greedy with largest-degree-first ordering.

        Vertices are sorted by degree (descending). High-degree vertices
        get colored first, which tends to produce fewer colors than
        natural ordering.

        Args:
            adj: Adjacency matrix, shape (n, n).

        Returns:
            ColoringResult with Welsh-Powell coloring.
        """
        adj = GraphColoring._normalize_adj(adj)
        n = adj.shape[0]

        # Compute degrees
        degrees = np.sum(adj != 0, axis=1)

        # Sort by degree descending (stable sort preserves order for ties)
        ordering = list(np.argsort(-degrees))

        return GraphColoring.greedy(adj, ordering=ordering)

    @staticmethod
    def dsatur(adj: np.ndarray) -> ColoringResult:
        """
        DSatur (Degree of Saturation) coloring algorithm.

        At each step, color the uncolored vertex with the highest
        saturation degree (number of distinct colors among its
        neighbors). Break ties by choosing the vertex with the
        highest degree in the uncolored subgraph.

        Generally produces better results than greedy or Welsh-Powell
        for most graph classes. Optimal for bipartite graphs.

        Args:
            adj: Adjacency matrix, shape (n, n).

        Returns:
            ColoringResult with DSatur coloring.
        """
        adj = GraphColoring._normalize_adj(adj)
        n = adj.shape[0]

        if n == 0:
            return ColoringResult(colors={}, n_colors=0, is_optimal=True)

        colors: dict[int, int] = {}
        # Track saturation degree for each uncolored vertex
        saturation: dict[int, set[int]] = {v: set() for v in range(n)}
        degrees = np.sum(adj != 0, axis=1).astype(int)
        uncolored = set(range(n))

        # Get neighbor lists once
        neighbors: dict[int, list[int]] = {}
        for v in range(n):
            neighbors[v] = [u for u in range(n) if adj[v, u] != 0]

        for _ in range(n):
            # Select vertex with max saturation degree, break ties by degree
            best_v = -1
            best_sat = -1
            best_deg = -1

            for v in uncolored:
                sat = len(saturation[v])
                deg = degrees[v]
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_v = v
                    best_sat = sat
                    best_deg = deg

            if best_v == -1:
                break

            # Find colors used by colored neighbors
            neighbor_colors: set[int] = set()
            for u in neighbors[best_v]:
                if u in colors:
                    neighbor_colors.add(colors[u])

            # Assign smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            colors[best_v] = color
            uncolored.discard(best_v)

            # Update saturation degrees of uncolored neighbors
            for u in neighbors[best_v]:
                if u in uncolored:
                    saturation[u].add(color)

        n_colors = len(set(colors.values())) if colors else 0

        return ColoringResult(
            colors=colors,
            n_colors=n_colors,
            is_optimal=False,
        )

    @staticmethod
    def chromatic_number_exact(adj: np.ndarray) -> int:
        """
        Compute exact chromatic number via backtracking.

        Warning: exponential time complexity. Only practical for small
        graphs (n <= 20). Uses incremental k-coloring attempts starting
        from the clique number lower bound.

        Args:
            adj: Adjacency matrix, shape (n, n).

        Returns:
            Exact chromatic number chi(G).

        Raises:
            ValueError: If n > 25 (safety limit).
        """
        adj = GraphColoring._normalize_adj(adj)
        n = adj.shape[0]

        if n == 0:
            return 0
        if n == 1:
            return 1
        if n > 25:
            raise ValueError(
                f"Exact chromatic number is exponential; n={n} > 25 safety limit. "
                "Use dsatur() for an approximation."
            )

        # Get neighbor sets for fast lookup
        neighbor_sets: list[set[int]] = []
        for v in range(n):
            neighbor_sets.append({u for u in range(n) if adj[v, u] != 0})

        # Lower bound: size of largest clique found by greedy
        lower = GraphColoring._greedy_clique_size(adj, neighbor_sets)

        # Upper bound: DSatur result
        upper = GraphColoring.dsatur(adj).n_colors

        if lower == upper:
            return lower

        # Try k-coloring for k = lower, lower+1, ...
        for k in range(lower, upper + 1):
            if GraphColoring._is_k_colorable(n, k, neighbor_sets):
                return k

        return upper  # Should not reach here

    @staticmethod
    def is_valid_coloring(adj: np.ndarray, colors: dict[int, int]) -> bool:
        """
        Verify that a coloring is proper (no adjacent vertices share a color).

        Args:
            adj: Adjacency matrix.
            colors: Vertex-to-color mapping.

        Returns:
            True if the coloring is valid.
        """
        adj = GraphColoring._normalize_adj(adj)
        n = adj.shape[0]

        for v in range(n):
            if v not in colors:
                return False  # Uncolored vertex
            for u in range(v + 1, n):
                if adj[v, u] != 0 and u in colors:
                    if colors[v] == colors[u]:
                        return False

        return True

    @staticmethod
    def _normalize_adj(adj: np.ndarray) -> np.ndarray:
        """Ensure adjacency matrix is symmetric with zero diagonal."""
        adj = np.asarray(adj, dtype=np.float64)
        n = adj.shape[0]
        if adj.shape != (n, n):
            raise ValueError(f"Adjacency matrix must be square, got {adj.shape}")
        # Symmetrize
        adj = np.maximum(adj, adj.T)
        # Zero diagonal (no self-loops)
        np.fill_diagonal(adj, 0)
        return adj

    @staticmethod
    def _greedy_clique_size(
        adj: np.ndarray,
        neighbor_sets: list[set[int]],
    ) -> int:
        """Find a clique greedily (lower bound on chromatic number)."""
        n = adj.shape[0]
        # Start from highest-degree vertex
        degrees = [len(ns) for ns in neighbor_sets]
        v = int(np.argmax(degrees))

        clique = {v}
        candidates = list(neighbor_sets[v])

        for u in sorted(candidates, key=lambda x: -degrees[x]):
            if all(u in neighbor_sets[c] for c in clique):
                clique.add(u)

        return len(clique)

    @staticmethod
    def _is_k_colorable(
        n: int,
        k: int,
        neighbor_sets: list[set[int]],
    ) -> bool:
        """Check if graph is k-colorable via backtracking."""
        colors = [-1] * n

        def backtrack(v: int) -> bool:
            if v == n:
                return True

            used = set()
            for u in neighbor_sets[v]:
                if colors[u] >= 0:
                    used.add(colors[u])

            for c in range(k):
                if c not in used:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1

            return False

        return backtrack(0)


# ============================================================================
# CONFLICT GRAPH CONSTRUCTION
# ============================================================================

class ConflictGraph:
    """
    Build conflict (incompatibility) graphs from domain-specific inputs.

    Transforms scheduling and resource allocation problems into
    graph coloring problems. Two vertices are adjacent if they
    cannot share the same color (time slot, channel, etc.).
    """

    @staticmethod
    def from_resources(
        agents: list[int],
        shared_resources: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Build a conflict graph from resource sharing relationships.

        Two agents conflict if they share a resource and cannot
        operate simultaneously. Each pair in shared_resources
        becomes an edge in the conflict graph.

        Args:
            agents: List of agent identifiers (used as vertex indices).
            shared_resources: List of (agent_i, agent_j) pairs that
                            share a resource and cannot coexist.

        Returns:
            Adjacency matrix (n x n) where n = max(agents) + 1.
        """
        if not agents:
            return np.zeros((0, 0))

        n = max(agents) + 1
        adj = np.zeros((n, n), dtype=np.float64)

        for i, j in shared_resources:
            if 0 <= i < n and 0 <= j < n and i != j:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        return adj

    @staticmethod
    def from_schedule(
        tasks: list[int],
        conflicts: list[tuple[int, int]],
    ) -> np.ndarray:
        """
        Build a conflict graph from task conflict pairs.

        Two tasks conflict if they cannot be scheduled in the same
        time slot (e.g., they share a machine, a person, or a room).

        Args:
            tasks: List of task identifiers (vertex indices).
            conflicts: List of (task_i, task_j) pairs that conflict.

        Returns:
            Adjacency matrix.
        """
        if not tasks:
            return np.zeros((0, 0))

        n = max(tasks) + 1
        adj = np.zeros((n, n), dtype=np.float64)

        for i, j in conflicts:
            if 0 <= i < n and 0 <= j < n and i != j:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        return adj

    @staticmethod
    def from_intervals(
        intervals: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Build a conflict (interval) graph from time intervals.

        Two intervals conflict if they overlap. The chromatic number
        of an interval graph equals its clique number (interval graphs
        are perfect), so greedy with sorted endpoints is optimal.

        Args:
            intervals: List of (start, end) time intervals.

        Returns:
            Adjacency matrix where adj[i][j]=1 iff intervals i,j overlap.
        """
        n = len(intervals)
        adj = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            si, ei = intervals[i]
            for j in range(i + 1, n):
                sj, ej = intervals[j]
                # Overlap if not (si >= ej or sj >= ei)
                if si < ej and sj < ei:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        return adj


# ============================================================================
# CHROMATIC POLYNOMIAL & INDEPENDENT SETS
# ============================================================================

def chromatic_polynomial(adj: np.ndarray, k: int) -> int:
    """
    Compute chromatic polynomial P(G, k) via deletion-contraction.

    P(G, k) = number of proper k-colorings of G.

    Uses the recurrence:
        P(G, k) = P(G - e, k) - P(G / e, k)
    where G - e deletes edge e, and G / e contracts edge e.

    Warning: exponential time. Only practical for small graphs (n <= 15).

    Args:
        adj: Adjacency matrix, shape (n, n).
        k: Number of colors.

    Returns:
        Number of proper k-colorings.

    Raises:
        ValueError: If n > 18.
    """
    adj = np.asarray(adj, dtype=np.float64)
    n = adj.shape[0]
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)

    if n > 18:
        raise ValueError(
            f"Chromatic polynomial is exponential; n={n} > 18 safety limit."
        )

    return _chromatic_poly_recursive(adj, k)


def _chromatic_poly_recursive(adj: np.ndarray, k: int) -> int:
    """Recursive deletion-contraction for chromatic polynomial."""
    n = adj.shape[0]

    # Base cases
    if n == 0:
        return 1
    if k <= 0:
        return 0

    # Empty graph (no edges): P = k^n
    edge = _find_edge(adj)
    if edge is None:
        return k ** n

    u, v = edge

    # Deletion: remove edge (u, v)
    adj_del = adj.copy()
    adj_del[u, v] = 0
    adj_del[v, u] = 0

    # Contraction: merge v into u, remove v
    adj_con = _contract_edge(adj, u, v)

    return _chromatic_poly_recursive(adj_del, k) - _chromatic_poly_recursive(adj_con, k)


def _find_edge(adj: np.ndarray) -> tuple[int, int] | None:
    """Find any edge in the graph, or None if no edges exist."""
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] != 0:
                return (i, j)
    return None


def _contract_edge(adj: np.ndarray, u: int, v: int) -> np.ndarray:
    """Contract edge (u, v): merge v into u, remove v."""
    n = adj.shape[0]

    # Merge v's neighbors into u
    adj_new = adj.copy()
    for w in range(n):
        if w != u and w != v:
            if adj_new[v, w] != 0:
                adj_new[u, w] = 1.0
                adj_new[w, u] = 1.0

    # Remove vertex v by deleting row/col v
    adj_new = np.delete(adj_new, v, axis=0)
    adj_new = np.delete(adj_new, v, axis=1)

    # Remove self-loop on u (the merged vertex)
    # Adjust index: if v < u, then u shifted down by 1
    u_new = u if v > u else u - 1
    if u_new < adj_new.shape[0]:
        adj_new[u_new, u_new] = 0

    return adj_new


def independent_sets(adj: np.ndarray) -> list[list[int]]:
    """
    Find all maximal independent sets of a graph.

    An independent set is a set of vertices with no edges between them.
    A maximal independent set cannot be extended by adding any vertex.

    Uses Bron-Kerbosch algorithm with pivoting.

    Warning: exponential in the worst case. Practical for n <= 25.

    Args:
        adj: Adjacency matrix.

    Returns:
        List of maximal independent sets, each a sorted list of vertex indices.
    """
    adj = np.asarray(adj, dtype=np.float64)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    n = adj.shape[0]

    if n > 30:
        raise ValueError(f"Independent set enumeration is exponential; n={n} > 30 safety limit.")

    # Build complement graph neighbor sets
    # Independent set in G = clique in complement(G)
    complement_neighbors: list[set[int]] = []
    for v in range(n):
        # Non-neighbors in original graph = neighbors in complement
        complement_neighbors.append(
            {u for u in range(n) if u != v and adj[v, u] == 0}
        )

    # Bron-Kerbosch on complement graph to find cliques = independent sets in G
    results: list[list[int]] = []

    def bron_kerbosch(R: set[int], P: set[int], X: set[int]) -> None:
        if not P and not X:
            if R:
                results.append(sorted(R))
            return

        # Choose pivot to minimize branching
        union_px = P | X
        if not union_px:
            return
        pivot = max(union_px, key=lambda v: len(complement_neighbors[v] & P))

        # Vertices to explore: P \ N(pivot) in complement
        candidates = P - complement_neighbors[pivot]

        for v in list(candidates):
            new_R = R | {v}
            new_P = P & complement_neighbors[v]
            new_X = X & complement_neighbors[v]
            bron_kerbosch(new_R, new_P, new_X)
            P = P - {v}
            X = X | {v}

    bron_kerbosch(set(), set(range(n)), set())

    return results


def maximum_independent_set(adj: np.ndarray) -> list[int]:
    """
    Find the largest independent set (greedy approximation).

    Uses a greedy approach: repeatedly select the vertex with the
    smallest degree and remove it and its neighbors.

    Not optimal, but runs in O(n^2).

    Args:
        adj: Adjacency matrix.

    Returns:
        List of vertex indices forming an independent set.
    """
    adj = np.asarray(adj, dtype=np.float64)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    n = adj.shape[0]

    available = set(range(n))
    independent: list[int] = []

    while available:
        # Pick vertex with minimum degree among available
        min_deg = n + 1
        best_v = -1
        for v in available:
            deg = sum(1 for u in available if adj[v, u] != 0)
            if deg < min_deg:
                min_deg = deg
                best_v = v

        if best_v == -1:
            break

        independent.append(best_v)
        # Remove best_v and its neighbors
        to_remove = {best_v}
        for u in available:
            if adj[best_v, u] != 0:
                to_remove.add(u)
        available -= to_remove

    return sorted(independent)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def color_graph(adj: np.ndarray, method: str = "dsatur") -> ColoringResult:
    """
    Color a graph using the specified method.

    Args:
        adj: Adjacency matrix.
        method: 'greedy', 'welsh_powell', or 'dsatur'.

    Returns:
        ColoringResult.
    """
    methods = {
        "greedy": GraphColoring.greedy,
        "welsh_powell": GraphColoring.welsh_powell,
        "dsatur": GraphColoring.dsatur,
    }

    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods.keys())}")

    return methods[method](adj)


def schedule_tasks(
    tasks: list[int],
    conflicts: list[tuple[int, int]],
) -> dict[int, int]:
    """
    Schedule tasks into time slots, minimizing the number of slots.

    Tasks that conflict cannot share a time slot. Returns a mapping
    from task to time slot index.

    Args:
        tasks: List of task identifiers.
        conflicts: List of conflicting task pairs.

    Returns:
        Dict mapping task -> time_slot.
    """
    adj = ConflictGraph.from_schedule(tasks, conflicts)
    result = GraphColoring.dsatur(adj)
    return {t: result.colors.get(t, 0) for t in tasks}
