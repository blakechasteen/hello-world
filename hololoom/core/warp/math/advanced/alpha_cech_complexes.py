"""
Alpha & Cech Complexes — Geometric Simplicial Complexes and TDA Pipeline
=========================================================================

Construction of simplicial complexes from point cloud data via
geometric proximity, forming the input for persistent homology.

Core Concepts:
- Alpha Complex: Nerve of the Voronoi decomposition, efficient in low dimensions
- Cech Complex: Nerve of balls of radius r around each point
- TDA Pipeline: embed → complex → persistence → vectorize

Mathematical Foundation:
Cech: σ = {p₁,...,pₖ} ∈ C_r iff ∩ B(pᵢ, r) ≠ ∅ (common intersection)
Alpha: σ ∈ A_α iff σ ∈ Delaunay(P) and circumradius(σ) ≤ α
Nerve theorem: If all intersections are contractible, nerve ≃ union

Applications to Warp Space:
- Topological feature extraction from embedding point clouds
- Persistent homology of memory graphs at multiple scales
- Full TDA pipeline for automated shape analysis

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SimplexTree:
    """Weighted simplex tree for filtration-equipped simplicial complexes.

    Each simplex has a filtration value. Simplices are stored as frozensets
    of vertex indices with associated filtration values.
    """
    _simplices: dict[frozenset[int], float] = field(default_factory=dict)

    def insert(self, vertices: list[int], filtration: float = 0.0) -> None:
        """Insert a simplex with given filtration value."""
        key = frozenset(vertices)
        if key not in self._simplices or filtration < self._simplices[key]:
            self._simplices[key] = filtration

    def filtration(self, vertices: list[int]) -> float:
        """Get filtration value of a simplex."""
        return self._simplices.get(frozenset(vertices), np.inf)

    @property
    def simplices(self) -> list[tuple[list[int], float]]:
        """All simplices sorted by filtration value."""
        return [
            (sorted(list(s)), f)
            for s, f in sorted(self._simplices.items(), key=lambda x: (x[1], len(x[0])))
        ]

    @property
    def dimension(self) -> int:
        """Maximum dimension of any simplex."""
        if not self._simplices:
            return -1
        return max(len(s) - 1 for s in self._simplices)

    @property
    def num_simplices(self) -> int:
        return len(self._simplices)

    def skeleton(self, dim: int) -> 'SimplexTree':
        """Return sub-complex of dimension ≤ dim."""
        tree = SimplexTree()
        for s, f in self._simplices.items():
            if len(s) - 1 <= dim:
                tree._simplices[s] = f
        return tree


@dataclass
class PersistencePair:
    """A birth-death pair from persistent homology."""
    birth: float
    death: float
    dimension: int


@dataclass
class TDAResult:
    """Result of full TDA pipeline.

    Attributes:
        persistence_pairs: List of (birth, death, dimension) tuples.
        betti_numbers: Betti numbers at the median filtration value.
        persistence_landscape: Persistence landscape values.
        total_persistence: Sum of all lifetimes.
        complex_size: Number of simplices in the complex.
    """
    persistence_pairs: list[PersistencePair]
    betti_numbers: dict[int, int]
    persistence_landscape: np.ndarray
    total_persistence: float
    complex_size: int


# ============================================================================
# DELAUNAY TRIANGULATION (INCREMENTAL)
# ============================================================================

class DelaunayIncremental:
    """Simple incremental Delaunay triangulation for 2D/3D.

    Uses the Bowyer-Watson algorithm. For production use with higher
    dimensions, defer to scipy.spatial.Delaunay.
    """

    @staticmethod
    def triangulate_2d(points: np.ndarray) -> list[list[int]]:
        """2D Delaunay triangulation via Bowyer-Watson.

        Args:
            points: (n, 2) point array.

        Returns:
            List of triangles as vertex index triples.
        """
        n = len(points)
        if n < 3:
            if n == 2:
                return [[0, 1]]
            return []

        # Super-triangle enclosing all points
        margin = 10.0
        x_min, y_min = points.min(axis=0) - margin
        x_max, y_max = points.max(axis=0) + margin
        dx, dy = x_max - x_min, y_max - y_min

        # Three vertices of super-triangle
        p_super = np.array([
            [x_min - dx, y_min - dy],
            [x_min + 3 * dx, y_min - dy],
            [x_min + dx / 2, y_min + 3 * dy],
        ])
        all_points = np.vstack([points, p_super])
        super_idx = [n, n + 1, n + 2]

        triangles: list[list[int]] = [super_idx[:]]

        for i in range(n):
            p = all_points[i]
            bad_triangles = []

            for tri in triangles:
                if DelaunayIncremental._in_circumcircle(all_points, tri, p):
                    bad_triangles.append(tri)

            # Find boundary polygon of the hole
            boundary = []
            for tri in bad_triangles:
                for j in range(3):
                    edge = [tri[j], tri[(j + 1) % 3]]
                    shared = False
                    for other in bad_triangles:
                        if other is tri:
                            continue
                        if edge[0] in other and edge[1] in other:
                            shared = True
                            break
                    if not shared:
                        boundary.append(edge)

            # Remove bad triangles
            for tri in bad_triangles:
                triangles.remove(tri)

            # Re-triangulate with new point
            for edge in boundary:
                triangles.append([i, edge[0], edge[1]])

        # Remove triangles containing super-triangle vertices
        result = []
        for tri in triangles:
            if not any(v in super_idx for v in tri):
                result.append(sorted(tri))

        return result

    @staticmethod
    def _in_circumcircle(points: np.ndarray, tri: list[int], p: np.ndarray) -> bool:
        """Check if point p is inside the circumcircle of triangle tri."""
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        ax, ay = a[0] - p[0], a[1] - p[1]
        bx, by = b[0] - p[0], b[1] - p[1]
        cx, cy = c[0] - p[0], c[1] - p[1]
        det = (ax * ax + ay * ay) * (bx * cy - cx * by) \
            - (bx * bx + by * by) * (ax * cy - cx * ay) \
            + (cx * cx + cy * cy) * (ax * by - bx * ay)
        return det > 0


# ============================================================================
# ALPHA COMPLEX
# ============================================================================

class AlphaComplex:
    """Alpha complex construction from point cloud.

    The alpha complex is the nerve of the Voronoi decomposition
    restricted to balls of radius α. It is a subcomplex of the
    Delaunay triangulation where each simplex has circumradius ≤ α.

    More efficient than Cech for low-dimensional data since the
    Delaunay triangulation has O(n^⌈d/2⌉) simplices.

    Example:
        >>> ac = AlphaComplex()
        >>> points = np.random.randn(100, 2)
        >>> tree = ac.from_points(points)
    """

    def from_points(self, points: np.ndarray, max_alpha: float | None = None) -> SimplexTree:
        """Build alpha complex from point cloud.

        Args:
            points: (n, d) point array (currently supports d=2).
            max_alpha: Maximum alpha value. If None, includes all simplices.

        Returns:
            SimplexTree with alpha filtration.
        """
        points = np.asarray(points, dtype=float)
        n, d = points.shape

        tree = SimplexTree()

        # Add vertices at filtration 0
        for i in range(n):
            tree.insert([i], 0.0)

        if d == 2:
            # Use 2D Delaunay
            triangles = DelaunayIncremental.triangulate_2d(points)

            # Add edges and triangles with circumradius filtration
            edges_added: set[frozenset[int]] = set()

            for tri in triangles:
                # Circumradius of triangle
                cr = self._circumradius(points[tri[0]], points[tri[1]], points[tri[2]])

                if max_alpha is not None and cr > max_alpha:
                    # Still add edges if their filtration is valid
                    for j in range(3):
                        e = frozenset([tri[j], tri[(j + 1) % 3]])
                        if e not in edges_added:
                            edge_filt = np.linalg.norm(
                                points[tri[j]] - points[tri[(j + 1) % 3]]
                            ) / 2.0
                            if max_alpha is None or edge_filt <= max_alpha:
                                tree.insert(sorted(list(e)), edge_filt)
                                edges_added.add(e)
                    continue

                tree.insert(sorted(tri), cr)

                # Add edges with half-length filtration
                for j in range(3):
                    e = frozenset([tri[j], tri[(j + 1) % 3]])
                    if e not in edges_added:
                        edge_filt = np.linalg.norm(
                            points[tri[j]] - points[tri[(j + 1) % 3]]
                        ) / 2.0
                        tree.insert(sorted(list(e)), edge_filt)
                        edges_added.add(e)
        else:
            # For higher dimensions, use Rips as approximation
            for i in range(n):
                for j in range(i + 1, n):
                    d_ij = np.linalg.norm(points[i] - points[j]) / 2.0
                    if max_alpha is None or d_ij <= max_alpha:
                        tree.insert([i, j], d_ij)

        return tree

    @staticmethod
    def _circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Circumradius of a triangle with vertices a, b, c."""
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)
        s = (ab + bc + ca) / 2.0
        area = np.sqrt(max(s * (s - ab) * (s - bc) * (s - ca), 0.0))
        if area < 1e-12:
            return max(ab, bc, ca) / 2.0
        return (ab * bc * ca) / (4.0 * area)


# ============================================================================
# CECH COMPLEX
# ============================================================================

class CechComplex:
    """Cech complex construction from point cloud.

    The Cech complex at radius r includes a simplex {p₁,...,pₖ} iff
    the balls B(pᵢ, r) have a common intersection point.

    For pairs (edges), this is equivalent to d(pᵢ, pⱼ) ≤ 2r.
    For triples, it requires the miniball radius ≤ r.

    Example:
        >>> cc = CechComplex()
        >>> points = np.random.randn(50, 3)
        >>> tree = cc.from_points(points, radius=1.0)
    """

    def from_points(
        self, points: np.ndarray, radius: float, max_dim: int = 2
    ) -> SimplexTree:
        """Build Cech complex from point cloud.

        Args:
            points: (n, d) point array.
            radius: Maximum ball radius.
            max_dim: Maximum simplex dimension to compute.

        Returns:
            SimplexTree with Cech filtration.
        """
        points = np.asarray(points, dtype=float)
        n = points.shape[0]

        tree = SimplexTree()

        # Vertices
        for i in range(n):
            tree.insert([i], 0.0)

        # Edges: include if d(i,j) ≤ 2r
        edge_pairs: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = np.linalg.norm(points[i] - points[j])
                if d_ij <= 2.0 * radius:
                    tree.insert([i, j], d_ij / 2.0)
                    edge_pairs.append((i, j))

        if max_dim >= 2:
            # Triangles: include if miniball radius ≤ r
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in set(edge_pairs):
                        continue
                    for k in range(j + 1, n):
                        if (i, k) not in set(edge_pairs) or (j, k) not in set(edge_pairs):
                            continue
                        mb_r = self._miniball_radius(points[[i, j, k]])
                        if mb_r <= radius:
                            tree.insert([i, j, k], mb_r)

        if max_dim >= 3:
            # Tetrahedra: check miniball
            triangles = set()
            for s, f in tree.simplices:
                if len(s) == 3:
                    triangles.add(frozenset(s))

            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        if frozenset([i, j, k]) not in triangles:
                            continue
                        for l in range(k + 1, n):
                            if all(
                                frozenset([a, b, c]) in triangles
                                for a, b, c in [(i, j, l), (i, k, l), (j, k, l)]
                            ):
                                mb_r = self._miniball_radius(points[[i, j, k, l]])
                                if mb_r <= radius:
                                    tree.insert([i, j, k, l], mb_r)

        return tree

    @staticmethod
    def _miniball_radius(pts: np.ndarray) -> float:
        """Compute radius of smallest enclosing ball (miniball).

        Uses a simple iterative approach. For exact computation in
        high dimensions, Welzl's algorithm would be preferred.

        Args:
            pts: (k, d) point array.

        Returns:
            Miniball radius.
        """
        # Center estimate: centroid, then iterative refinement
        center = np.mean(pts, axis=0)

        for _ in range(20):
            dists = np.linalg.norm(pts - center, axis=1)
            farthest = np.argmax(dists)
            max_dist = dists[farthest]
            # Move center toward farthest point
            center = center + 0.5 * (pts[farthest] - center) * (1.0 - max_dist / (max_dist + 1e-12))

        return float(np.max(np.linalg.norm(pts - center, axis=1)))


# ============================================================================
# TDA PIPELINE
# ============================================================================

class TDAPipeline:
    """Full Topological Data Analysis pipeline.

    Combines complex construction, persistence computation, and
    feature extraction into a single callable.

    Pipeline:
    1. Build simplicial complex (Alpha, Cech, or Rips)
    2. Compute persistent homology via boundary matrix reduction
    3. Extract topological features (Betti, persistence landscape, etc.)

    Example:
        >>> tda = TDAPipeline()
        >>> data = np.random.randn(200, 3)
        >>> result = tda.fit(data)
        >>> print(f"Betti: {result.betti_numbers}")
    """

    def fit(
        self,
        data: np.ndarray,
        method: str = 'alpha',
        max_dim: int = 2,
        max_radius: float | None = None,
    ) -> TDAResult:
        """Run full TDA pipeline on point cloud data.

        Args:
            data: Point cloud (n, d).
            method: Complex construction method ('alpha', 'cech', 'rips').
            max_dim: Maximum homology dimension.
            max_radius: Maximum filtration radius. Auto-determined if None.

        Returns:
            TDAResult with topological features.
        """
        data = np.asarray(data, dtype=float)

        if max_radius is None:
            # Use 2x median pairwise distance
            n_sample = min(len(data), 100)
            sample = data[np.random.choice(len(data), n_sample, replace=False)]
            pairwise = []
            for i in range(n_sample):
                for j in range(i + 1, n_sample):
                    pairwise.append(np.linalg.norm(sample[i] - sample[j]))
            max_radius = 2.0 * np.median(pairwise) if pairwise else 1.0

        # Build complex
        if method == 'alpha':
            tree = AlphaComplex().from_points(data, max_alpha=max_radius)
        elif method == 'cech':
            tree = CechComplex().from_points(data, radius=max_radius, max_dim=max_dim)
        else:
            tree = self._rips_complex(data, max_radius, max_dim)

        # Compute persistence
        pairs = self._compute_persistence(tree)

        # Extract features
        betti = self._betti_at_median(pairs, max_radius)
        landscape = self._persistence_landscape(pairs, n_points=100)
        total_pers = sum(
            p.death - p.birth for p in pairs if not np.isinf(p.death)
        )

        return TDAResult(
            persistence_pairs=pairs,
            betti_numbers=betti,
            persistence_landscape=landscape,
            total_persistence=float(total_pers),
            complex_size=tree.num_simplices,
        )

    def _rips_complex(self, points: np.ndarray, max_radius: float, max_dim: int) -> SimplexTree:
        """Build Vietoris-Rips complex."""
        n = len(points)
        tree = SimplexTree()

        for i in range(n):
            tree.insert([i], 0.0)

        # Edges
        edges = {}
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = np.linalg.norm(points[i] - points[j])
                if d_ij <= 2.0 * max_radius:
                    tree.insert([i, j], d_ij / 2.0)
                    edges[frozenset([i, j])] = d_ij / 2.0

        # Higher simplices via clique expansion
        if max_dim >= 2:
            for i in range(n):
                nbrs_i = {j for (i2, j2) in [(min(a, b), max(a, b)) for a, b in edges
                          if not isinstance(a, frozenset)]
                          if i2 == i or j2 == i}
                # Simplified: just check triangles
                neighbors = set()
                for e in edges:
                    if i in e:
                        neighbors.update(e - {i})
                for j in neighbors:
                    for k in neighbors:
                        if j < k and frozenset([j, k]) in edges:
                            filt = max(edges.get(frozenset([i, j]), np.inf),
                                      edges.get(frozenset([i, k]), np.inf),
                                      edges.get(frozenset([j, k]), np.inf))
                            if filt <= max_radius:
                                tree.insert([i, j, k], filt)

        return tree

    @staticmethod
    def _compute_persistence(tree: SimplexTree) -> list[PersistencePair]:
        """Compute persistence from simplex tree via column reduction."""
        simplices = tree.simplices
        n = len(simplices)
        simplex_to_idx = {frozenset(s): i for i, (s, f) in enumerate(simplices)}

        # Build boundary matrix
        boundary: dict[int, set[int]] = {}
        for j, (simplex, filt) in enumerate(simplices):
            if len(simplex) <= 1:
                boundary[j] = set()
                continue
            faces = set()
            for i in range(len(simplex)):
                face = frozenset(simplex[:i] + simplex[i+1:])
                if face in simplex_to_idx:
                    faces.add(simplex_to_idx[face])
            boundary[j] = faces

        # Reduce
        reduced = {j: set(rows) for j, rows in boundary.items()}
        pivots: dict[int, int] = {}

        for j in range(n):
            if j not in reduced or not reduced[j]:
                continue
            while reduced[j]:
                pivot = max(reduced[j])
                if pivot in pivots:
                    other = pivots[pivot]
                    reduced[j] = reduced[j].symmetric_difference(reduced.get(other, set()))
                else:
                    pivots[pivot] = j
                    break

        # Extract pairs
        pairs = []
        paired = set(pivots.keys()) | set(pivots.values())

        for row, col in pivots.items():
            birth = simplices[row][1]
            death = simplices[col][1]
            dim = len(simplices[row][0]) - 1
            if abs(death - birth) > 1e-12:
                pairs.append(PersistencePair(birth=birth, death=death, dimension=dim))

        return pairs

    @staticmethod
    def _betti_at_median(pairs: list[PersistencePair], max_val: float) -> dict[int, int]:
        """Compute Betti numbers at median filtration value."""
        t = max_val / 2.0
        betti: dict[int, int] = {}
        for p in pairs:
            if p.birth <= t < p.death:
                betti[p.dimension] = betti.get(p.dimension, 0) + 1
        return betti

    @staticmethod
    def _persistence_landscape(pairs: list[PersistencePair], n_points: int = 100) -> np.ndarray:
        """Compute persistence landscape (first landscape function).

        The persistence landscape Λ(t) = max over all intervals of
        min(t - birth, death - t).

        Args:
            pairs: Persistence pairs.
            n_points: Resolution of landscape discretization.

        Returns:
            Landscape values (n_points,).
        """
        if not pairs:
            return np.zeros(n_points)

        finite_pairs = [(p.birth, p.death) for p in pairs if not np.isinf(p.death)]
        if not finite_pairs:
            return np.zeros(n_points)

        births = [p[0] for p in finite_pairs]
        deaths = [p[1] for p in finite_pairs]
        t_min = min(births)
        t_max = max(deaths)

        t_grid = np.linspace(t_min, t_max, n_points)
        landscape = np.zeros(n_points)

        for b, d in finite_pairs:
            tent = np.minimum(t_grid - b, d - t_grid)
            tent = np.maximum(tent, 0.0)
            landscape = np.maximum(landscape, tent)

        return landscape
