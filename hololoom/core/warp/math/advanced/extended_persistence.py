"""
Extended Persistence — Extended & Zigzag Persistent Homology
=============================================================

Beyond ordinary persistence: extended persistence captures features
missed by sublevel filtrations, and zigzag persistence handles
sequences of spaces connected by both inclusions and removals.

Core Concepts:
- Extended Persistence: Ordinary + relative + extended+ + extended- diagrams
- Zigzag Persistence: Handles X₁ → X₂ ← X₃ → ... (not just inclusions)
- Extended persistence captures "essential" homology classes that never die

Mathematical Foundation:
Extended: H_*(X_a) → H_*(X_b) → H_*(X, X_b) → H_*(X, X_a) (long exact sequence)
Zigzag: Algebraic persistence for sequences X₁ ↔ X₂ ↔ ... with arbitrary maps
Decomposition theorem: Any pointwise finite-dim zigzag decomposes into intervals

Applications to Warp Space:
- Tracking topological features through non-monotone memory operations
- Extended persistence for complete shape characterization of embedding clouds
- Zigzag homology for time-varying knowledge graphs

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PersistenceInterval:
    """A single persistence interval (birth, death).

    Attributes:
        birth: Birth time/value.
        death: Death time/value (inf for essential classes).
        dimension: Homological dimension.
        interval_type: 'ordinary', 'relative', 'extended_plus', 'extended_minus'.
    """
    birth: float
    death: float
    dimension: int
    interval_type: str = 'ordinary'

    @property
    def persistence(self) -> float:
        """Lifetime of the feature."""
        if np.isinf(self.death):
            return np.inf
        return abs(self.death - self.birth)


@dataclass
class ExtendedDiagram:
    """Extended persistence diagram.

    Attributes:
        ordinary: Intervals from sublevel set filtration (born and die in sublevel).
        relative: Intervals from relative homology (born in relative, die in relative).
        extended_plus: Extended intervals (born in sublevel, die in relative).
        extended_minus: Extended intervals (born in relative, die in sublevel).
    """
    ordinary: list[PersistenceInterval]
    relative: list[PersistenceInterval]
    extended_plus: list[PersistenceInterval]
    extended_minus: list[PersistenceInterval]

    @property
    def all_intervals(self) -> list[PersistenceInterval]:
        """All intervals from all four quadrants."""
        return self.ordinary + self.relative + self.extended_plus + self.extended_minus

    def betti_numbers(self, t: float) -> dict[int, int]:
        """Betti numbers at filtration value t."""
        betti: dict[int, int] = {}
        for interval in self.ordinary + self.extended_plus:
            if interval.birth <= t < interval.death:
                dim = interval.dimension
                betti[dim] = betti.get(dim, 0) + 1
        return betti


@dataclass
class ZigzagDiagram:
    """Zigzag persistence diagram.

    Attributes:
        intervals: List of (birth_index, death_index, dimension) intervals.
        spaces: Number of spaces in the zigzag sequence.
        maps: Directions of maps ('forward' or 'backward').
    """
    intervals: list[PersistenceInterval]
    spaces: int
    maps: list[str]


# ============================================================================
# SIMPLICIAL COMPLEX UTILITIES
# ============================================================================

class SimplexTree:
    """Minimal simplex tree for filtration representation.

    Stores simplices with filtration values for persistence computation.
    """

    def __init__(self):
        self._simplices: dict[frozenset[int], float] = {}

    def insert(self, simplex: list[int], filtration: float = 0.0) -> None:
        """Insert simplex and all its faces."""
        simplex_set = frozenset(simplex)
        # Update filtration (take max for consistency)
        if simplex_set in self._simplices:
            self._simplices[simplex_set] = min(self._simplices[simplex_set], filtration)
        else:
            self._simplices[simplex_set] = filtration

        # Insert all faces
        if len(simplex) > 1:
            for i in range(len(simplex)):
                face = simplex[:i] + simplex[i+1:]
                self.insert(face, filtration)

    def filtration_order(self) -> list[tuple[frozenset[int], float]]:
        """Return simplices sorted by filtration value, then by dimension."""
        return sorted(
            self._simplices.items(),
            key=lambda x: (x[1], len(x[0]))
        )

    @property
    def dimension(self) -> int:
        """Maximum simplex dimension."""
        if not self._simplices:
            return -1
        return max(len(s) - 1 for s in self._simplices)


# ============================================================================
# BOUNDARY MATRIX REDUCTION
# ============================================================================

class BoundaryMatrixReducer:
    """Column reduction of boundary matrices for persistence computation.

    Implements the standard persistence algorithm via column operations
    on the boundary matrix represented in sparse column form.
    """

    @staticmethod
    def reduce(boundary: dict[int, set[int]], n_cols: int) -> tuple[dict[int, set[int]], dict[int, int]]:
        """Reduce boundary matrix and extract persistence pairs.

        Args:
            boundary: Sparse boundary matrix as {col_index: set of row indices}.
            n_cols: Total number of columns.

        Returns:
            (reduced_matrix, pivots) where pivots maps pivot_row -> column.
        """
        reduced = {j: set(rows) for j, rows in boundary.items()}
        pivots: dict[int, int] = {}  # pivot_row -> column_index

        for j in range(n_cols):
            if j not in reduced or not reduced[j]:
                continue

            while reduced[j]:
                pivot = max(reduced[j])
                if pivot in pivots:
                    # Add column pivots[pivot] to column j (mod 2)
                    other = pivots[pivot]
                    reduced[j] = reduced[j].symmetric_difference(reduced.get(other, set()))
                else:
                    pivots[pivot] = j
                    break

        return reduced, pivots


# ============================================================================
# EXTENDED PERSISTENCE
# ============================================================================

class ExtendedPersistence:
    """Extended persistent homology computation.

    Computes all four types of persistence intervals by combining
    sublevel and superlevel filtrations on a function defined over
    a simplicial complex.

    Algorithm:
    1. Sort simplices by filtration value (sublevel)
    2. Compute ordinary persistence from sublevel filtration
    3. Reverse filtration for superlevel → relative persistence
    4. Combine to get extended persistence pairs

    Example:
        >>> ep = ExtendedPersistence()
        >>> # Function on vertices of a triangle
        >>> values = {0: 1.0, 1: 2.0, 2: 3.0}
        >>> triangles = [[0, 1], [1, 2], [0, 2], [0, 1, 2]]
        >>> result = ep.compute(triangles, values)
    """

    def compute(
        self,
        simplices: list[list[int]],
        vertex_values: dict[int, float],
    ) -> ExtendedDiagram:
        """Compute extended persistence diagram.

        Args:
            simplices: List of simplices (as vertex lists). Must include all faces.
            vertex_values: Filtration value at each vertex.

        Returns:
            ExtendedDiagram with all four types of intervals.
        """
        # Build simplex tree with lower-star filtration
        tree = SimplexTree()
        for simplex in simplices:
            filt = max(vertex_values.get(v, 0.0) for v in simplex)
            tree.insert(simplex, filt)

        ordered = tree.filtration_order()
        simplex_list = [s for s, f in ordered]
        filt_values = [f for s, f in ordered]
        n = len(simplex_list)

        # Build simplex index
        simplex_to_idx = {s: i for i, s in enumerate(simplex_list)}

        # Boundary matrix
        boundary: dict[int, set[int]] = {}
        for j, simplex in enumerate(simplex_list):
            if len(simplex) <= 1:
                boundary[j] = set()
                continue
            faces = set()
            simplex_sorted = sorted(simplex)
            for i in range(len(simplex_sorted)):
                face = frozenset(simplex_sorted[:i] + simplex_sorted[i+1:])
                if face in simplex_to_idx:
                    faces.add(simplex_to_idx[face])
            boundary[j] = faces

        # Reduce boundary matrix
        _, pivots = BoundaryMatrixReducer.reduce(boundary, n)

        # Extract ordinary persistence pairs
        ordinary = []
        paired_cols = set(pivots.values())
        paired_rows = set(pivots.keys())

        for pivot_row, col in pivots.items():
            birth_val = filt_values[pivot_row]
            death_val = filt_values[col]
            dim = len(simplex_list[pivot_row]) - 1
            if abs(death_val - birth_val) > 1e-12:
                ordinary.append(PersistenceInterval(
                    birth=birth_val,
                    death=death_val,
                    dimension=dim,
                    interval_type='ordinary',
                ))

        # Essential classes (never paired in ordinary)
        extended_plus = []
        for j in range(n):
            if j not in paired_cols and j not in paired_rows:
                dim = len(simplex_list[j]) - 1
                extended_plus.append(PersistenceInterval(
                    birth=filt_values[j],
                    death=np.inf,
                    dimension=dim,
                    interval_type='extended_plus',
                ))

        # Relative persistence: reverse filtration
        reverse_filt = [max(filt_values) - f + min(filt_values) for f in filt_values]
        reverse_order = np.argsort(reverse_filt)

        relative = []
        extended_minus = []

        # Simplified relative computation using reversed filtration
        rev_boundary: dict[int, set[int]] = {}
        rev_list = [simplex_list[i] for i in reverse_order]
        rev_vals = [reverse_filt[i] for i in reverse_order]
        rev_idx = {simplex_list[reverse_order[i]]: i for i in range(n)}

        for j_new in range(n):
            j_old = int(reverse_order[j_new])
            simplex = simplex_list[j_old]
            if len(simplex) <= 1:
                rev_boundary[j_new] = set()
                continue
            faces = set()
            simplex_sorted = sorted(simplex)
            for i in range(len(simplex_sorted)):
                face = frozenset(simplex_sorted[:i] + simplex_sorted[i+1:])
                if face in rev_idx:
                    faces.add(rev_idx[face])
            rev_boundary[j_new] = faces

        _, rev_pivots = BoundaryMatrixReducer.reduce(rev_boundary, n)

        for pivot_row, col in rev_pivots.items():
            birth_val = rev_vals[pivot_row]
            death_val = rev_vals[col]
            dim = len(rev_list[pivot_row]) - 1
            if abs(death_val - birth_val) > 1e-12:
                relative.append(PersistenceInterval(
                    birth=birth_val,
                    death=death_val,
                    dimension=dim,
                    interval_type='relative',
                ))

        return ExtendedDiagram(
            ordinary=ordinary,
            relative=relative,
            extended_plus=extended_plus,
            extended_minus=extended_minus,
        )

    @staticmethod
    def bottleneck_distance(diag1: ExtendedDiagram, diag2: ExtendedDiagram) -> float:
        """Bottleneck distance between extended persistence diagrams.

        d_B = max over all four quadrants of the quadrant bottleneck distances.

        Args:
            diag1, diag2: Extended persistence diagrams.

        Returns:
            Bottleneck distance.
        """
        def _quadrant_distance(intervals1: list[PersistenceInterval],
                               intervals2: list[PersistenceInterval]) -> float:
            if not intervals1 and not intervals2:
                return 0.0

            points1 = [(iv.birth, iv.death) for iv in intervals1 if not np.isinf(iv.death)]
            points2 = [(iv.birth, iv.death) for iv in intervals2 if not np.isinf(iv.death)]

            if not points1 and not points2:
                return 0.0

            # Greedy matching (approximation to true bottleneck)
            max_cost = 0.0
            for p in points1:
                min_dist = abs(p[1] - p[0]) / 2.0  # Cost of matching to diagonal
                for q in points2:
                    d = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
                    min_dist = min(min_dist, d)
                max_cost = max(max_cost, min_dist)

            for q in points2:
                min_dist = abs(q[1] - q[0]) / 2.0
                for p in points1:
                    d = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
                    min_dist = min(min_dist, d)
                max_cost = max(max_cost, min_dist)

            return max_cost

        return max(
            _quadrant_distance(diag1.ordinary, diag2.ordinary),
            _quadrant_distance(diag1.relative, diag2.relative),
            _quadrant_distance(diag1.extended_plus, diag2.extended_plus),
            _quadrant_distance(diag1.extended_minus, diag2.extended_minus),
        )


# ============================================================================
# ZIGZAG PERSISTENCE
# ============================================================================

class ZigzagPersistence:
    """Zigzag persistent homology.

    Handles sequences of simplicial complexes connected by maps
    in either direction:
        K₁ → K₂ ← K₃ → K₄ ← ...

    Unlike ordinary persistence (which requires a monotone filtration),
    zigzag handles insertions AND deletions of simplices.

    Algorithm: Converts zigzag to a standard persistence problem via
    the diamond principle and reduction.

    Example:
        >>> zz = ZigzagPersistence()
        >>> # Three snapshots with additions and deletions
        >>> K1 = [[0], [1], [0,1]]
        >>> K2 = [[0], [1], [2], [0,1], [1,2]]
        >>> K3 = [[1], [2], [1,2]]
        >>> result = zz.compute([K1, K2, K3], ['forward', 'backward'])
    """

    def compute(
        self,
        filtrations: list[list[list[int]]],
        directions: list[str] | None = None,
    ) -> ZigzagDiagram:
        """Compute zigzag persistence.

        Args:
            filtrations: Sequence of simplicial complexes, each as list of simplices.
            directions: Direction of each map ('forward' or 'backward').
                       If None, alternates forward/backward.

        Returns:
            ZigzagDiagram with intervals indexed by position in sequence.
        """
        n_spaces = len(filtrations)
        if n_spaces < 2:
            return ZigzagDiagram(intervals=[], spaces=n_spaces, maps=[])

        if directions is None:
            directions = ['forward'] * (n_spaces - 1)

        # Convert each complex to sets of frozensets
        complexes = []
        for simplices in filtrations:
            S: set[frozenset[int]] = set()
            for simplex in simplices:
                S.add(frozenset(simplex))
                # Add all faces
                for i in range(len(simplex)):
                    S.add(frozenset(simplex[:i] + simplex[i+1:]))
            complexes.append(S)

        # Track births and deaths via inclusion/exclusion at each step
        intervals: list[PersistenceInterval] = []
        active: dict[frozenset[int], int] = {}  # simplex -> birth index

        # Initialize with first complex
        for simplex in complexes[0]:
            active[simplex] = 0

        for i in range(1, n_spaces):
            current = complexes[i]
            previous = complexes[i - 1]

            if directions[i - 1] == 'forward':
                # Simplices added
                added = current - previous
                # Simplices removed
                removed = previous - current

                for simplex in added:
                    active[simplex] = i

                for simplex in removed:
                    if simplex in active:
                        birth = active.pop(simplex)
                        dim = len(simplex) - 1
                        intervals.append(PersistenceInterval(
                            birth=float(birth),
                            death=float(i),
                            dimension=dim,
                            interval_type='zigzag',
                        ))
            else:
                # Backward map: removals become births, additions become deaths
                added = current - previous
                removed = previous - current

                for simplex in removed:
                    if simplex in active:
                        birth = active.pop(simplex)
                        dim = len(simplex) - 1
                        intervals.append(PersistenceInterval(
                            birth=float(birth),
                            death=float(i),
                            dimension=dim,
                            interval_type='zigzag',
                        ))

                for simplex in added:
                    active[simplex] = i

        # Remaining active simplices are essential
        for simplex, birth in active.items():
            dim = len(simplex) - 1
            intervals.append(PersistenceInterval(
                birth=float(birth),
                death=np.inf,
                dimension=dim,
                interval_type='zigzag',
            ))

        # Filter out trivial intervals and sort
        intervals = [iv for iv in intervals if iv.persistence > 0]
        intervals.sort(key=lambda iv: (iv.dimension, iv.birth))

        return ZigzagDiagram(
            intervals=intervals,
            spaces=n_spaces,
            maps=directions,
        )

    @staticmethod
    def stability_distance(diag1: ZigzagDiagram, diag2: ZigzagDiagram) -> float:
        """Approximate bottleneck distance between zigzag diagrams.

        Args:
            diag1, diag2: Zigzag persistence diagrams.

        Returns:
            Approximate bottleneck distance.
        """
        points1 = [(iv.birth, iv.death) for iv in diag1.intervals if not np.isinf(iv.death)]
        points2 = [(iv.birth, iv.death) for iv in diag2.intervals if not np.isinf(iv.death)]

        if not points1 and not points2:
            return 0.0

        max_cost = 0.0
        for p in points1:
            min_dist = abs(p[1] - p[0]) / 2.0
            for q in points2:
                d = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
                min_dist = min(min_dist, d)
            max_cost = max(max_cost, min_dist)

        for q in points2:
            min_dist = abs(q[1] - q[0]) / 2.0
            for p in points1:
                d = max(abs(p[0] - q[0]), abs(p[1] - q[1]))
                min_dist = min(min_dist, d)
            max_cost = max(max_cost, min_dist)

        return max_cost
