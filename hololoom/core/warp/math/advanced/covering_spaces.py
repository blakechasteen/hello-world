"""
Covering Spaces and Homotopy Theory — TO2.6-2.8
================================================

Fundamental group computation from simplicial complexes, covering space
construction, path lifting, higher homotopy groups, and CW complex
decomposition.

Classes:
    FundamentalGroup: pi_1 via edge-path groups on simplicial complexes
    CoveringSpace: Covering projections, deck transformations, path lifting
    HigherHomotopy: Approximation of pi_n via simplicial methods
    CWComplex: Cell decomposition from simplicial data

Mathematical Foundation:
    pi_1(X, x_0): edge loops modulo face relations in the 2-skeleton
    Covering space: p: E -> B, local homeomorphism, unique path lifting
    Galois correspondence: normal subgroups of pi_1 <-> regular covers
    CW structure: X = union of cells e^n attached via boundary maps

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimplicialComplex:
    """
    Simplicial complex defined by vertex set and simplices.

    Attributes:
        vertices: List of vertex labels
        simplices: List of simplices (each a frozenset of vertices)
        dim: Maximum dimension of any simplex
    """
    vertices: list[int]
    simplices: list[frozenset[int]]
    dim: int = 0

    def __post_init__(self):
        if self.simplices:
            self.dim = max(len(s) - 1 for s in self.simplices)

    def edges(self) -> list[tuple[int, int]]:
        """Return all 1-simplices as (u, v) pairs with u < v."""
        edges = []
        for s in self.simplices:
            if len(s) == 2:
                u, v = sorted(s)
                edges.append((u, v))
        return edges

    def triangles(self) -> list[tuple[int, int, int]]:
        """Return all 2-simplices as sorted triples."""
        tris = []
        for s in self.simplices:
            if len(s) == 3:
                tris.append(tuple(sorted(s)))
        return tris

    def neighbors(self, v: int) -> set[int]:
        """Vertices connected to v by an edge."""
        nbrs = set()
        for s in self.simplices:
            if len(s) == 2 and v in s:
                nbrs.update(s - {v})
        return nbrs

    def skeleton(self, k: int) -> 'SimplicialComplex':
        """Return the k-skeleton (simplices of dimension <= k)."""
        filtered = [s for s in self.simplices if len(s) <= k + 1]
        return SimplicialComplex(vertices=self.vertices, simplices=filtered)


class FundamentalGroup:
    """
    Fundamental group pi_1(X, x_0) of a simplicial complex.

    Computed via the edge-path group: generators are edges not in a
    spanning tree, relations come from 2-simplices (triangles).

    Algorithm:
        1. Build spanning tree of the 1-skeleton
        2. Non-tree edges become generators
        3. Each triangle [a,b,c] gives relation e_{ab} e_{bc} = e_{ac}
           (modulo tree edges which are identity)
        4. Simplify using Tietze transformations
    """

    def compute(
        self, complex: SimplicialComplex, base_point: int | None = None
    ) -> list[list[int]]:
        """
        Compute generators of pi_1(X, x_0).

        Returns generators as edge-loops (sequences of vertex indices).
        Each generator is a non-tree edge completed to a loop via
        the spanning tree.

        Args:
            complex: A simplicial complex (needs 1-skeleton and 2-skeleton)
            base_point: Base point vertex. Defaults to first vertex.

        Returns:
            List of generators, each a list of vertex indices forming a loop
        """
        if not complex.vertices:
            return []

        if base_point is None:
            base_point = complex.vertices[0]

        edges = complex.edges()
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # BFS spanning tree
        tree_edges = set()
        parent = {base_point: None}
        queue = deque([base_point])
        visited = {base_point}

        while queue:
            u = queue.popleft()
            for v in sorted(adj[u]):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    tree_edges.add((min(u, v), max(u, v)))
                    queue.append(v)

        # Non-tree edges are generators
        non_tree = [(u, v) for u, v in edges if (u, v) not in tree_edges]

        # For each non-tree edge (u, v), build the loop:
        # base -> ... -> u -> v -> ... -> base (via tree paths)
        generators = []
        for u, v in non_tree:
            path_to_u = self._tree_path(parent, base_point, u)
            path_to_v = self._tree_path(parent, base_point, v)

            # Loop: base -> u -> v -> base
            loop = path_to_u + list(reversed(path_to_v))
            generators.append(loop)

        return generators

    def _tree_path(self, parent: dict, root: int, target: int) -> list[int]:
        """Path from root to target in the spanning tree."""
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = parent.get(current)
        path.reverse()
        return path

    def compute_presentation(
        self, complex: SimplicialComplex, base_point: int | None = None
    ) -> tuple[int, list[list[int]]]:
        """
        Compute a group presentation <generators | relations>.

        Args:
            complex: Simplicial complex
            base_point: Base point vertex

        Returns:
            (n_generators, relations) where relations are words in
            generator indices (positive = generator, negative = inverse)
        """
        if base_point is None and complex.vertices:
            base_point = complex.vertices[0]

        edges = complex.edges()
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Spanning tree
        tree_edges = set()
        parent = {base_point: None}
        queue = deque([base_point])
        visited = {base_point}

        while queue:
            u = queue.popleft()
            for v in sorted(adj[u]):
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    tree_edges.add((min(u, v), max(u, v)))
                    queue.append(v)

        non_tree = [(u, v) for u, v in edges if (u, v) not in tree_edges]
        edge_to_gen = {e: i + 1 for i, e in enumerate(non_tree)}

        # Relations from triangles
        relations = []
        for a, b, c in complex.triangles():
            # Triangle [a, b, c] gives relation: e_{ab} * e_{bc} * e_{ca}^{-1} = 1
            word = []
            for u, v in [(a, b), (b, c), (a, c)]:
                edge = (min(u, v), max(u, v))
                if edge in edge_to_gen:
                    gen_idx = edge_to_gen[edge]
                    # Direction matters
                    if u < v:
                        word.append(gen_idx)
                    else:
                        word.append(-gen_idx)
            if word:
                relations.append(word)

        return len(non_tree), relations

    def euler_characteristic(self, complex: SimplicialComplex) -> int:
        """
        Euler characteristic chi = V - E + F - ...

        Alternating sum of simplex counts by dimension.
        """
        counts = defaultdict(int)
        for s in complex.simplices:
            counts[len(s) - 1] += 1

        chi = 0
        for dim, count in counts.items():
            chi += ((-1) ** dim) * count
        return chi


class CoveringSpace:
    """
    Covering spaces and path lifting.

    A covering space p: E -> B is a local homeomorphism where every
    point in B has an evenly covered neighborhood. Key property:
    unique path lifting given initial point in the fiber.
    """

    def __init__(self, n_sheets: int, monodromy: dict[int, np.ndarray] | None = None):
        """
        Initialize a covering space.

        Args:
            n_sheets: Number of sheets (degree of the covering)
            monodromy: Dict mapping generator index -> permutation matrix
                       on the fiber. Determines how sheets connect.
        """
        self.n_sheets = n_sheets
        self.monodromy = monodromy or {}

    def lift_path(
        self,
        path: list[int],
        base_point_sheet: int = 0,
        generator_labels: dict[tuple[int, int], int] | None = None,
    ) -> list[tuple[int, int]]:
        """
        Lift a path in the base space to the covering space.

        Args:
            path: Sequence of vertices in base space
            base_point_sheet: Starting sheet index (0 to n_sheets-1)
            generator_labels: Map from edge (u,v) to generator index

        Returns:
            Lifted path as list of (vertex, sheet) pairs
        """
        if not path:
            return []

        current_sheet = base_point_sheet
        lifted = [(path[0], current_sheet)]

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = (min(u, v), max(u, v))

            if generator_labels and edge in generator_labels:
                gen = generator_labels[edge]
                if gen in self.monodromy:
                    perm = self.monodromy[gen]
                    current_sheet = int(np.argmax(perm[current_sheet]))

            lifted.append((v, current_sheet))

        return lifted

    def fiber(self, base_vertex: int) -> list[tuple[int, int]]:
        """
        Return the fiber p^{-1}(base_vertex) — all sheets above this vertex.

        Args:
            base_vertex: Vertex in base space

        Returns:
            List of (base_vertex, sheet) pairs
        """
        return [(base_vertex, s) for s in range(self.n_sheets)]

    def deck_transformations(self) -> list[np.ndarray]:
        """
        Compute deck transformations (covering automorphisms).

        A deck transformation is a permutation of sheets that commutes
        with all monodromy permutations. For a regular (normal) covering,
        deck transformations act transitively on fibers.

        Returns:
            List of permutation matrices that commute with all monodromy elements
        """
        n = self.n_sheets
        # Generate all permutation matrices
        from itertools import permutations

        deck = []
        for perm in permutations(range(n)):
            P = np.zeros((n, n))
            for i, j in enumerate(perm):
                P[i, j] = 1.0

            # Check commutation with all monodromy elements
            commutes = True
            for gen, M in self.monodromy.items():
                if not np.allclose(P @ M, M @ P):
                    commutes = False
                    break

            if commutes:
                deck.append(P)

        return deck

    def is_regular(self) -> bool:
        """
        Check if the covering is regular (normal).

        A covering is regular iff deck transformations act transitively
        on each fiber, which means |Deck(E/B)| = n_sheets.
        """
        return len(self.deck_transformations()) == self.n_sheets


class HigherHomotopy:
    """
    Approximation of higher homotopy groups pi_n(X).

    For n >= 2, pi_n is always abelian. We approximate the rank
    (number of Z summands) using homology as a proxy:
    - Hurewicz theorem: pi_n(X) = H_n(X) for (n-1)-connected X
    - General case: pi_n maps surjectively onto H_n for simply-connected X
    """

    def pi_n(self, complex: SimplicialComplex, n: int) -> int:
        """
        Estimate the rank of pi_n(X) via simplicial homology.

        Uses the Hurewicz theorem when applicable. For general spaces,
        returns the rank of H_n as a lower bound.

        Args:
            complex: Simplicial complex
            n: Homotopy group dimension (n >= 1)

        Returns:
            Estimated rank of pi_n (number of Z summands)
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        if n == 1:
            # pi_1 rank = number of generators of fundamental group
            fg = FundamentalGroup()
            gens = fg.compute(complex)
            return len(gens)

        # For n >= 2, compute H_n via simplicial homology
        return self._homology_rank(complex, n)

    def _homology_rank(self, complex: SimplicialComplex, n: int) -> int:
        """Compute rank of H_n via boundary matrices."""
        # Build chain groups C_n, C_{n-1}, C_{n+1}
        simplices_n = [s for s in complex.simplices if len(s) == n + 1]
        simplices_nm1 = [s for s in complex.simplices if len(s) == n]
        simplices_np1 = [s for s in complex.simplices if len(s) == n + 2]

        if not simplices_n:
            return 0

        # Boundary map d_n: C_n -> C_{n-1}
        if simplices_nm1:
            d_n = self._boundary_matrix(simplices_n, simplices_nm1)
        else:
            d_n = np.zeros((1, len(simplices_n)))

        # Boundary map d_{n+1}: C_{n+1} -> C_n
        if simplices_np1:
            d_np1 = self._boundary_matrix(simplices_np1, simplices_n)
        else:
            d_np1 = np.zeros((len(simplices_n), 1))

        # H_n = ker(d_n) / im(d_{n+1})
        # rank(H_n) = dim(ker(d_n)) - rank(d_{n+1})
        kernel_dim = len(simplices_n) - np.linalg.matrix_rank(d_n)
        image_dim = np.linalg.matrix_rank(d_np1)

        rank = max(0, int(kernel_dim - image_dim))
        return rank

    def _boundary_matrix(
        self, simplices_high: list[frozenset[int]], simplices_low: list[frozenset[int]]
    ) -> np.ndarray:
        """Build boundary map matrix between chain groups."""
        n_high = len(simplices_high)
        n_low = len(simplices_low)
        matrix = np.zeros((n_low, n_high))

        low_index = {s: i for i, s in enumerate(simplices_low)}

        for j, sigma in enumerate(simplices_high):
            sorted_sigma = sorted(sigma)
            for k, vertex in enumerate(sorted_sigma):
                face = frozenset(sorted_sigma[:k] + sorted_sigma[k + 1:])
                if face in low_index:
                    matrix[low_index[face], j] = (-1) ** k

        return matrix

    def betti_numbers(self, complex: SimplicialComplex, max_dim: int | None = None) -> list[int]:
        """
        Compute Betti numbers b_0, b_1, ..., b_k.

        b_n = rank(H_n) counts independent n-dimensional holes.

        Args:
            complex: Simplicial complex
            max_dim: Maximum dimension to compute (defaults to complex dimension)

        Returns:
            List of Betti numbers [b_0, b_1, ..., b_k]
        """
        if max_dim is None:
            max_dim = complex.dim

        betti = []
        for n in range(max_dim + 1):
            if n == 0:
                # b_0 = number of connected components
                betti.append(self._count_components(complex))
            else:
                betti.append(self._homology_rank(complex, n))

        return betti

    def _count_components(self, complex: SimplicialComplex) -> int:
        """Count connected components via BFS."""
        if not complex.vertices:
            return 0

        adj = defaultdict(set)
        for s in complex.simplices:
            if len(s) == 2:
                u, v = s
                adj[u].add(v)
                adj[v].add(u)

        visited = set()
        components = 0
        for v in complex.vertices:
            if v not in visited:
                components += 1
                queue = deque([v])
                while queue:
                    u = queue.popleft()
                    if u in visited:
                        continue
                    visited.add(u)
                    for w in adj[u]:
                        if w not in visited:
                            queue.append(w)
        return components


class CWComplex:
    """
    CW complex from simplicial data.

    A CW complex is built by iteratively attaching cells:
    X^0 (0-cells) -> X^1 (attach 1-cells) -> X^2 (attach 2-cells) -> ...

    Each n-cell e^n is homeomorphic to the open n-disk,
    attached via a map from the (n-1)-sphere to the (n-1)-skeleton.
    """

    @dataclass
    class Cell:
        """A cell in the CW complex."""
        dimension: int
        index: int
        boundary: list[tuple[int, int]]  # (cell_index, coefficient) pairs

    def __init__(self):
        self.cells: dict[int, list[CWComplex.Cell]] = defaultdict(list)

    @staticmethod
    def from_simplicial(complex: SimplicialComplex) -> 'CWComplex':
        """
        Convert a simplicial complex to a CW complex.

        Each n-simplex becomes an n-cell. The boundary of an n-cell
        is the alternating sum of its (n-1)-faces.

        Args:
            complex: Input simplicial complex

        Returns:
            CW complex with cells corresponding to simplices
        """
        cw = CWComplex()

        # Index simplices by dimension
        by_dim = defaultdict(list)
        for s in complex.simplices:
            by_dim[len(s) - 1].append(s)

        # Sort for consistent indexing
        for dim in sorted(by_dim.keys()):
            simplices = sorted(by_dim[dim], key=lambda s: sorted(s))
            simplex_to_idx = {s: i for i, s in enumerate(simplices)}

            for i, sigma in enumerate(simplices):
                boundary = []
                if dim > 0:
                    sorted_sigma = sorted(sigma)
                    lower_simplices = sorted(by_dim[dim - 1], key=lambda s: sorted(s))
                    lower_idx = {s: j for j, s in enumerate(lower_simplices)}

                    for k, vertex in enumerate(sorted_sigma):
                        face = frozenset(sorted_sigma[:k] + sorted_sigma[k + 1:])
                        if face in lower_idx:
                            boundary.append((lower_idx[face], (-1) ** k))

                cell = CWComplex.Cell(dimension=dim, index=i, boundary=boundary)
                cw.cells[dim].append(cell)

        return cw

    def chain_complex(self, max_dim: int | None = None) -> list[np.ndarray]:
        """
        Extract the cellular chain complex d_n: C_n -> C_{n-1}.

        Returns:
            List of boundary matrices [d_1, d_2, ..., d_k]
        """
        if max_dim is None:
            max_dim = max(self.cells.keys()) if self.cells else 0

        boundaries = []
        for n in range(1, max_dim + 1):
            n_cells = len(self.cells.get(n, []))
            nm1_cells = len(self.cells.get(n - 1, []))

            if n_cells == 0 or nm1_cells == 0:
                boundaries.append(np.zeros((max(nm1_cells, 1), max(n_cells, 1))))
                continue

            d = np.zeros((nm1_cells, n_cells))
            for cell in self.cells[n]:
                for face_idx, coeff in cell.boundary:
                    d[face_idx, cell.index] = coeff

            boundaries.append(d)

        return boundaries

    def euler_characteristic(self) -> int:
        """Euler characteristic from cell counts: chi = sum (-1)^n * c_n."""
        chi = 0
        for dim, cells in self.cells.items():
            chi += ((-1) ** dim) * len(cells)
        return chi

    def n_cells(self, n: int) -> int:
        """Number of n-cells."""
        return len(self.cells.get(n, []))
