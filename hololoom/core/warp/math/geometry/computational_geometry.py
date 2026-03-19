"""
Computational Geometry
=======================

Scope slots: GE7.1 (Voronoi diagrams), GE7.2 (Delaunay triangulation),
GE7.3 (convex hull), GE7.4 (KD-tree), GE7.5 (extensions).

Pure numpy implementations:
- Voronoi diagram via half-plane intersection (simplified Fortune's)
- Delaunay triangulation via Bowyer-Watson algorithm
- Convex hull via Graham scan
- KD-tree for k-nearest neighbor queries

Author: Claude Code
Date: March 2026
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONVEX HULL — Graham scan (GE7.3)
# ============================================================================

class ConvexHull:
    """Convex hull computation via Graham scan.

    Computes the convex hull of a set of 2D points in O(n log n).
    The hull is returned as an ordered list of vertex indices.

    Example:
        >>> ch = ConvexHull()
        >>> points = np.array([[0,0], [1,0], [0.5,0.5], [1,1], [0,1]])
        >>> indices = ch.compute(points)
        >>> # Hull: [0, 1, 3, 4] (the four corners)
    """

    def compute(self, points: np.ndarray) -> np.ndarray:
        """Compute 2D convex hull via Graham scan.

        Args:
            points: Array of shape (n, 2).

        Returns:
            Array of hull vertex indices in counter-clockwise order.
        """
        points = np.asarray(points, dtype=float)
        n = len(points)
        if n < 3:
            return np.arange(n)

        # Find lowest point (break ties by x)
        start = 0
        for i in range(1, n):
            if points[i, 1] < points[start, 1] or (
                points[i, 1] == points[start, 1] and points[i, 0] < points[start, 0]
            ):
                start = i

        # Sort by polar angle from start
        def polar_key(idx):
            if idx == start:
                return (-np.inf, 0.0)
            dx = points[idx, 0] - points[start, 0]
            dy = points[idx, 1] - points[start, 1]
            return (np.arctan2(dy, dx), dx * dx + dy * dy)

        order = sorted(range(n), key=polar_key)

        # Graham scan
        stack = []
        for idx in order:
            while len(stack) >= 2:
                o = points[stack[-2]]
                a = points[stack[-1]]
                b = points[idx]
                cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
                if cross <= 0:
                    stack.pop()
                else:
                    break
            stack.append(idx)

        return np.array(stack)

    @staticmethod
    def area(points: np.ndarray, hull_indices: np.ndarray) -> float:
        """Compute area of convex hull via shoelace formula.

        Args:
            points: All points.
            hull_indices: Indices of hull vertices (ordered).

        Returns:
            Area of the convex hull.
        """
        hull_pts = points[hull_indices]
        n = len(hull_pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hull_pts[i, 0] * hull_pts[j, 1]
            area -= hull_pts[j, 0] * hull_pts[i, 1]
        return abs(area) / 2.0

    @staticmethod
    def is_inside(point: np.ndarray, points: np.ndarray, hull_indices: np.ndarray) -> bool:
        """Check if a point is inside the convex hull.

        Uses the winding number / cross product method.
        """
        hull_pts = points[hull_indices]
        n = len(hull_pts)
        for i in range(n):
            j = (i + 1) % n
            edge = hull_pts[j] - hull_pts[i]
            to_pt = point - hull_pts[i]
            cross = edge[0] * to_pt[1] - edge[1] * to_pt[0]
            if cross < 0:
                return False
        return True


# ============================================================================
# DELAUNAY TRIANGULATION — Bowyer-Watson (GE7.2)
# ============================================================================

@dataclass
class Triangle:
    """A triangle defined by three vertex indices."""
    v0: int
    v1: int
    v2: int

    @property
    def vertices(self) -> tuple[int, int, int]:
        return (self.v0, self.v1, self.v2)


class DelaunayTriangulation:
    """Delaunay triangulation via Bowyer-Watson algorithm.

    The Delaunay triangulation maximizes the minimum angle and has the
    property that no point lies inside the circumcircle of any triangle.

    Algorithm:
    1. Create super-triangle containing all points
    2. Insert points one at a time
    3. For each point, find triangles whose circumcircle contains it
    4. Remove those triangles, re-triangulate the hole
    5. Remove super-triangle vertices at the end

    Example:
        >>> dt = DelaunayTriangulation()
        >>> points = np.random.rand(20, 2)
        >>> triangles = dt.compute(points)
    """

    def compute(self, points: np.ndarray) -> list[Triangle]:
        """Compute Delaunay triangulation.

        Args:
            points: Array of shape (n, 2).

        Returns:
            List of Triangle objects.
        """
        points = np.asarray(points, dtype=float)
        n = len(points)
        if n < 3:
            return []

        # Create super-triangle that encloses all points
        margin = 10.0
        x_min, y_min = points.min(axis=0) - margin
        x_max, y_max = points.max(axis=0) + margin
        dx = x_max - x_min
        dy = y_max - y_min

        # Super-triangle vertices
        p0 = np.array([x_min - dx, y_min - dy])
        p1 = np.array([x_max + 2 * dx, y_min - dy])
        p2 = np.array([x_min + dx / 2, y_max + 2 * dy])

        all_points = np.vstack([points, p0, p1, p2])
        super_v = (n, n + 1, n + 2)

        triangles = [Triangle(super_v[0], super_v[1], super_v[2])]

        for i in range(n):
            # Find triangles whose circumcircle contains point i
            bad_triangles = []
            for tri in triangles:
                if self._in_circumcircle(all_points, tri, i):
                    bad_triangles.append(tri)

            # Find boundary of the polygonal hole
            boundary = []
            for tri in bad_triangles:
                edges = [
                    (tri.v0, tri.v1), (tri.v1, tri.v2), (tri.v2, tri.v0)
                ]
                for edge in edges:
                    shared = False
                    for other in bad_triangles:
                        if other is tri:
                            continue
                        other_edges = [
                            (other.v0, other.v1), (other.v1, other.v2),
                            (other.v2, other.v0)
                        ]
                        for oe in other_edges:
                            if set(edge) == set(oe):
                                shared = True
                                break
                        if shared:
                            break
                    if not shared:
                        boundary.append(edge)

            # Remove bad triangles
            for tri in bad_triangles:
                triangles.remove(tri)

            # Create new triangles from boundary edges to new point
            for e0, e1 in boundary:
                triangles.append(Triangle(e0, e1, i))

        # Remove triangles connected to super-triangle
        result = []
        for tri in triangles:
            verts = tri.vertices
            if any(v in super_v for v in verts):
                continue
            result.append(tri)

        return result

    @staticmethod
    def _in_circumcircle(points: np.ndarray, tri: Triangle, p_idx: int) -> bool:
        """Check if point p is inside the circumcircle of triangle."""
        ax, ay = points[tri.v0]
        bx, by = points[tri.v1]
        cx, cy = points[tri.v2]
        dx, dy = points[p_idx]

        # Use the determinant test
        a11 = ax - dx
        a12 = ay - dy
        a13 = a11 * a11 + a12 * a12
        a21 = bx - dx
        a22 = by - dy
        a23 = a21 * a21 + a22 * a22
        a31 = cx - dx
        a32 = cy - dy
        a33 = a31 * a31 + a32 * a32

        det = (a11 * (a22 * a33 - a23 * a32)
               - a12 * (a21 * a33 - a23 * a31)
               + a13 * (a21 * a32 - a22 * a31))

        # Positive det means inside circumcircle (for CCW triangle)
        # Check orientation
        orient = ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        if orient < 0:
            det = -det

        return det > 0

    @staticmethod
    def circumcenter(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        """Compute circumcenter of a triangle."""
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-15:
            return 0.5 * (p1 + p2)  # Degenerate

        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / D
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / D

        return np.array([ux, uy])


# ============================================================================
# VORONOI DIAGRAM (GE7.1)
# ============================================================================

@dataclass
class VoronoiResult:
    """Result of Voronoi diagram computation.

    Attributes:
        vertices: Voronoi vertices of shape (m, 2).
        regions: List of regions, each a list of vertex indices.
        ridge_points: Pairs of input point indices sharing a Voronoi edge.
        ridge_vertices: Pairs of Voronoi vertex indices forming each edge.
    """
    vertices: np.ndarray
    regions: list[list[int]]
    ridge_points: list[tuple[int, int]]
    ridge_vertices: list[tuple[int, int]]


class VoronoiDiagram:
    """Voronoi diagram computation via Delaunay dual.

    The Voronoi diagram is the dual of the Delaunay triangulation:
    - Voronoi vertices = circumcenters of Delaunay triangles
    - Voronoi edges connect circumcenters of adjacent Delaunay triangles
    - Voronoi cell of point p = intersection of half-planes

    Example:
        >>> vd = VoronoiDiagram()
        >>> points = np.array([[0,0], [1,0], [0.5, 0.87]])
        >>> result = vd.compute(points)
    """

    def compute(self, points: np.ndarray) -> VoronoiResult:
        """Compute Voronoi diagram as dual of Delaunay triangulation.

        Args:
            points: Array of shape (n, 2).

        Returns:
            VoronoiResult.
        """
        points = np.asarray(points, dtype=float)
        n = len(points)

        if n < 2:
            return VoronoiResult(
                vertices=np.empty((0, 2)),
                regions=[list(range(n))],
                ridge_points=[], ridge_vertices=[],
            )

        # Compute Delaunay triangulation
        dt = DelaunayTriangulation()
        triangles = dt.compute(points)

        if not triangles:
            return VoronoiResult(
                vertices=np.empty((0, 2)),
                regions=[[] for _ in range(n)],
                ridge_points=[], ridge_vertices=[],
            )

        # Voronoi vertices = circumcenters
        vertices = []
        for tri in triangles:
            cc = DelaunayTriangulation.circumcenter(
                points[tri.v0], points[tri.v1], points[tri.v2]
            )
            vertices.append(cc)
        vertices = np.array(vertices)

        # Build adjacency: for each edge, which triangles share it
        edge_to_tris: dict = {}
        for t_idx, tri in enumerate(triangles):
            edges = [
                tuple(sorted([tri.v0, tri.v1])),
                tuple(sorted([tri.v1, tri.v2])),
                tuple(sorted([tri.v0, tri.v2])),
            ]
            for edge in edges:
                if edge not in edge_to_tris:
                    edge_to_tris[edge] = []
                edge_to_tris[edge].append(t_idx)

        # Voronoi ridges: edges shared by exactly 2 triangles
        ridge_points = []
        ridge_vertices = []
        for edge, tri_indices in edge_to_tris.items():
            if len(tri_indices) == 2:
                ridge_points.append(edge)
                ridge_vertices.append(tuple(tri_indices))

        # Voronoi regions: for each input point, collect surrounding circumcenters
        regions = [[] for _ in range(n)]
        for t_idx, tri in enumerate(triangles):
            for v in tri.vertices:
                if v < n:
                    regions[v].append(t_idx)

        # Sort region vertices by angle around their input point
        for i in range(n):
            if regions[i]:
                centers = vertices[regions[i]]
                angles = np.arctan2(
                    centers[:, 1] - points[i, 1],
                    centers[:, 0] - points[i, 0],
                )
                order = np.argsort(angles)
                regions[i] = [regions[i][j] for j in order]

        return VoronoiResult(
            vertices=vertices,
            regions=regions,
            ridge_points=ridge_points,
            ridge_vertices=ridge_vertices,
        )


# ============================================================================
# KD-TREE (GE7.4)
# ============================================================================

@dataclass
class KDNode:
    """Node in a KD-tree."""
    point_idx: int
    split_dim: int
    split_val: float
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None


class KDTree:
    """KD-tree for k-nearest neighbor queries.

    Partitions d-dimensional space using axis-aligned hyperplanes,
    cycling through dimensions. Supports efficient nearest-neighbor
    and k-nearest-neighbor queries in O(log n) average case.

    Example:
        >>> kd = KDTree()
        >>> points = np.random.rand(100, 3)
        >>> kd.build(points)
        >>> dists, indices = kd.query(np.array([0.5, 0.5, 0.5]), k=5)
    """

    def __init__(self):
        self._root: KDNode | None = None
        self._points: np.ndarray | None = None
        self._ndim: int = 0

    def build(self, points: np.ndarray) -> None:
        """Build KD-tree from points.

        Args:
            points: Array of shape (n, d).
        """
        self._points = np.asarray(points, dtype=float)
        self._ndim = points.shape[1]
        indices = list(range(len(points)))
        self._root = self._build_recursive(indices, depth=0)

    def _build_recursive(self, indices: list[int], depth: int) -> KDNode | None:
        if not indices:
            return None

        dim = depth % self._ndim

        # Sort by the splitting dimension
        indices.sort(key=lambda i: self._points[i, dim])
        median = len(indices) // 2

        node = KDNode(
            point_idx=indices[median],
            split_dim=dim,
            split_val=self._points[indices[median], dim],
        )
        node.left = self._build_recursive(indices[:median], depth + 1)
        node.right = self._build_recursive(indices[median + 1:], depth + 1)

        return node

    def query(self, point: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors.

        Args:
            point: Query point of shape (d,).
            k: Number of neighbors.

        Returns:
            (distances, indices) — sorted by distance.
        """
        if self._root is None or self._points is None:
            return np.array([]), np.array([], dtype=int)

        point = np.asarray(point, dtype=float)

        # Use a bounded heap: list of (dist, idx), keep k smallest
        best = []
        self._query_recursive(self._root, point, k, best)

        best.sort(key=lambda x: x[0])
        if not best:
            return np.array([]), np.array([], dtype=int)

        distances = np.array([b[0] for b in best])
        indices = np.array([b[1] for b in best])
        return distances, indices

    def _query_recursive(
        self,
        node: KDNode | None,
        point: np.ndarray,
        k: int,
        best: list[tuple[float, int]],
    ) -> None:
        if node is None:
            return

        # Distance to this node's point
        dist = np.linalg.norm(point - self._points[node.point_idx])

        # Update best list
        if len(best) < k:
            best.append((dist, node.point_idx))
            best.sort(key=lambda x: x[0])
        elif dist < best[-1][0]:
            best[-1] = (dist, node.point_idx)
            best.sort(key=lambda x: x[0])

        # Determine which side to search first
        dim = node.split_dim
        diff = point[dim] - node.split_val

        if diff <= 0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left

        # Search near side
        self._query_recursive(near, point, k, best)

        # Search far side if the splitting plane is closer than current worst
        if len(best) < k or abs(diff) < best[-1][0]:
            self._query_recursive(far, point, k, best)

    def query_ball(self, point: np.ndarray, radius: float) -> list[int]:
        """Find all points within a given radius.

        Args:
            point: Query point.
            radius: Search radius.

        Returns:
            List of point indices within radius.
        """
        if self._root is None or self._points is None:
            return []

        point = np.asarray(point, dtype=float)
        result = []
        self._ball_recursive(self._root, point, radius, result)
        return result

    def _ball_recursive(
        self,
        node: KDNode | None,
        point: np.ndarray,
        radius: float,
        result: list[int],
    ) -> None:
        if node is None:
            return

        dist = np.linalg.norm(point - self._points[node.point_idx])
        if dist <= radius:
            result.append(node.point_idx)

        dim = node.split_dim
        diff = point[dim] - node.split_val

        if diff <= 0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left

        self._ball_recursive(near, point, radius, result)

        if abs(diff) <= radius:
            self._ball_recursive(far, point, radius, result)
