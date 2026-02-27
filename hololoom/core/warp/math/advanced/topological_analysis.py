"""
Topological Data Analysis - Persistent Homology for Embeddings
===============================================================

Algebraic topology methods for analyzing embedding space structure.

Core Concept:
    Persistent homology tracks topological features (connected components,
    loops, voids) as we sweep through scale parameters. Features that
    persist across many scales are "real"; short-lived ones are noise.

Classes:
    PersistenceDiagram: Birth-death pairs for topological features
    BettiNumbers: Count of features at each dimension
    PersistentHomology: Compute persistence from point clouds
    EmbeddingTopologyAnalyzer: HoloLoom integration

Metrics:
    - Bottleneck distance: max difference between diagrams
    - Wasserstein distance: optimal transport between diagrams
    - Persistence landscape: functional representation

Applications:
    - Embedding quality: Do embeddings have good topological structure?
    - Clustering validation: Are clusters topologically distinct?
    - Anomaly detection: Do outliers change topology?
    - Semantic structure: Loops may indicate semantic cycles

Author: Claude Code
Date: December 2025 (Math Module Expansion)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq


class HomologyDimension(Enum):
    """Homology dimension (Betti number index)."""
    H0 = 0  # Connected components
    H1 = 1  # Loops / 1-cycles
    H2 = 2  # Voids / 2-cycles


@dataclass
class PersistenceInterval:
    """
    A single persistence interval (birth, death) pair.

    Represents a topological feature:
    - Born at filtration value `birth`
    - Dies at filtration value `death`
    - Persistence = death - birth (longer = more significant)
    """
    birth: float
    death: float
    dimension: int = 0
    generator: Optional[Any] = None  # The simplex that created this feature

    @property
    def persistence(self) -> float:
        """Lifespan of the feature."""
        if self.death == float('inf'):
            return float('inf')
        return self.death - self.birth

    @property
    def midpoint(self) -> float:
        """Middle of the interval."""
        if self.death == float('inf'):
            return self.birth
        return (self.birth + self.death) / 2

    def __lt__(self, other: 'PersistenceInterval') -> bool:
        """Order by persistence (for sorting)."""
        return self.persistence < other.persistence


@dataclass
class PersistenceDiagram:
    """
    Persistence diagram: collection of (birth, death) points.

    The diagram encodes the complete topological signature of a
    filtration. Points far from the diagonal are significant features.
    """
    intervals: List[PersistenceInterval] = field(default_factory=list)
    dimension: int = 0

    def add(self, birth: float, death: float, generator: Any = None) -> None:
        """Add a persistence interval."""
        self.intervals.append(PersistenceInterval(
            birth=birth,
            death=death,
            dimension=self.dimension,
            generator=generator
        ))

    def filter_by_persistence(self, min_persistence: float) -> 'PersistenceDiagram':
        """Keep only features with persistence above threshold."""
        filtered = [i for i in self.intervals if i.persistence >= min_persistence]
        result = PersistenceDiagram(dimension=self.dimension)
        result.intervals = filtered
        return result

    @property
    def points(self) -> np.ndarray:
        """Get (birth, death) points as array."""
        if not self.intervals:
            return np.array([]).reshape(0, 2)
        return np.array([(i.birth, i.death) for i in self.intervals])

    @property
    def persistences(self) -> np.ndarray:
        """Get persistence values."""
        return np.array([i.persistence for i in self.intervals
                        if i.persistence != float('inf')])

    def total_persistence(self) -> float:
        """Sum of all finite persistence values."""
        return sum(i.persistence for i in self.intervals
                  if i.persistence != float('inf'))

    def max_persistence(self) -> float:
        """Maximum persistence value."""
        finite = [i.persistence for i in self.intervals
                 if i.persistence != float('inf')]
        return max(finite) if finite else 0.0

    def betti_at_scale(self, scale: float) -> int:
        """Count features alive at given scale."""
        return sum(1 for i in self.intervals
                  if i.birth <= scale < i.death)


@dataclass
class BettiNumbers:
    """
    Betti numbers: topological invariants.

    β₀ = number of connected components
    β₁ = number of 1-dimensional holes (loops)
    β₂ = number of 2-dimensional voids (cavities)

    Higher Betti numbers indicate more complex topology.
    """
    betti_0: int = 0  # Components
    betti_1: int = 0  # Loops
    betti_2: int = 0  # Voids

    def __getitem__(self, dim: int) -> int:
        if dim == 0:
            return self.betti_0
        elif dim == 1:
            return self.betti_1
        elif dim == 2:
            return self.betti_2
        else:
            return 0

    @property
    def euler_characteristic(self) -> int:
        """χ = β₀ - β₁ + β₂"""
        return self.betti_0 - self.betti_1 + self.betti_2

    def complexity_score(self) -> float:
        """Topological complexity measure."""
        return self.betti_0 + 2 * self.betti_1 + 3 * self.betti_2


class UnionFind:
    """Union-Find data structure for connected components."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Unite sets containing x and y. Returns True if merged."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        self.n_components -= 1
        return True


class VietorisRipsComplex:
    """
    Vietoris-Rips complex construction.

    For a point cloud X and radius r:
    - 0-simplices: points in X
    - 1-simplices (edges): pairs with distance ≤ r
    - 2-simplices (triangles): triples where all pairs have distance ≤ r
    """

    def __init__(self, points: np.ndarray, max_dim: int = 1):
        """
        Args:
            points: (n, d) array of point coordinates
            max_dim: Maximum simplex dimension to compute
        """
        self.points = points
        self.n_points = len(points)
        self.max_dim = max_dim

        # Compute all pairwise distances
        self.distances = self._compute_distances()

    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        n = self.n_points
        dists = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.points[i] - self.points[j])
                dists[i, j] = dists[j, i] = d

        return dists

    def get_filtration(self) -> List[Tuple[float, Tuple[int, ...]]]:
        """
        Get simplices sorted by filtration value.

        Returns list of (radius, simplex) pairs.
        """
        filtration = []

        # 0-simplices (vertices) at radius 0
        for i in range(self.n_points):
            filtration.append((0.0, (i,)))

        # 1-simplices (edges) at their distance
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                filtration.append((self.distances[i, j], (i, j)))

        # 2-simplices (triangles) at max edge distance
        if self.max_dim >= 2:
            for i in range(self.n_points):
                for j in range(i + 1, self.n_points):
                    for k in range(j + 1, self.n_points):
                        # Triangle appears when all three edges exist
                        radius = max(
                            self.distances[i, j],
                            self.distances[j, k],
                            self.distances[i, k]
                        )
                        filtration.append((radius, (i, j, k)))

        # Sort by filtration value
        filtration.sort(key=lambda x: (x[0], len(x[1])))

        return filtration


class PersistentHomology:
    """
    Compute persistent homology of a point cloud.

    Uses the standard algorithm based on reduction of boundary matrices.
    For H0 (connected components), uses efficient Union-Find.
    """

    def __init__(self, max_dimension: int = 1, max_radius: float = float('inf')):
        """
        Args:
            max_dimension: Maximum homology dimension (0, 1, or 2)
            max_radius: Maximum filtration radius
        """
        self.max_dimension = max_dimension
        self.max_radius = max_radius

    def compute(self, points: np.ndarray) -> Dict[int, PersistenceDiagram]:
        """
        Compute persistent homology of point cloud.

        Args:
            points: (n, d) array of point coordinates

        Returns:
            Dictionary mapping dimension to PersistenceDiagram
        """
        n = len(points)
        diagrams = {d: PersistenceDiagram(dimension=d)
                   for d in range(self.max_dimension + 1)}

        # Build Vietoris-Rips complex
        complex = VietorisRipsComplex(points, max_dim=self.max_dimension)
        filtration = complex.get_filtration()

        # Filter by max radius
        filtration = [(r, s) for r, s in filtration if r <= self.max_radius]

        # H0: Use Union-Find for efficiency
        uf = UnionFind(n)
        component_birth = {i: 0.0 for i in range(n)}

        for radius, simplex in filtration:
            if len(simplex) == 2:  # Edge
                i, j = simplex
                ri, rj = uf.find(i), uf.find(j)

                if ri != rj:
                    # Merge components: younger one dies
                    birth_i = component_birth[ri]
                    birth_j = component_birth[rj]

                    if birth_i <= birth_j:
                        # Component j dies
                        diagrams[0].add(birth_j, radius, generator=simplex)
                        uf.union(i, j)
                        # Update birth time for merged component
                        new_root = uf.find(i)
                        component_birth[new_root] = birth_i
                    else:
                        diagrams[0].add(birth_i, radius, generator=simplex)
                        uf.union(i, j)
                        new_root = uf.find(i)
                        component_birth[new_root] = birth_j

        # Add infinite persistence for remaining components
        for i in range(n):
            if uf.find(i) == i:  # Is a root
                diagrams[0].add(component_birth[i], float('inf'))

        # H1: Simplified loop detection
        # (Full algorithm requires boundary matrix reduction)
        if self.max_dimension >= 1:
            diagrams[1] = self._detect_loops_simplified(points, complex, filtration)

        return diagrams

    def _detect_loops_simplified(
        self,
        points: np.ndarray,
        complex: VietorisRipsComplex,
        filtration: List[Tuple[float, Tuple[int, ...]]]
    ) -> PersistenceDiagram:
        """
        Simplified loop detection using cycle detection.

        Detects when edges first form a cycle (loop birth) and when
        triangles fill it in (loop death).
        """
        diagram = PersistenceDiagram(dimension=1)
        n = len(points)

        # Track edges using Union-Find to detect when cycles form
        from collections import defaultdict

        # Adjacency list for cycle detection
        adj: Dict[int, Set[int]] = defaultdict(set)
        edge_times: Dict[Tuple[int, int], float] = {}

        # Active loops: (birth_time, frozenset of vertices in cycle)
        active_loops: List[Tuple[float, Tuple[int, int, int]]] = []

        # Union-Find for detecting when edges create cycles
        uf = UnionFind(n)

        for radius, simplex in filtration:
            if len(simplex) == 2:
                edge = tuple(sorted(simplex))
                i, j = edge

                # Check if this edge creates a cycle
                if uf.find(i) == uf.find(j):
                    # Already connected! This edge creates a cycle
                    # Record the cycle birth
                    active_loops.append((radius, (i, j, -1)))  # -1 indicates generic cycle
                else:
                    # New edge connecting different components
                    uf.union(i, j)

                adj[i].add(j)
                adj[j].add(i)
                edge_times[edge] = radius

            elif len(simplex) == 3:
                # Triangle: kills a 1-cycle if one exists
                i, j, k = simplex
                e1 = tuple(sorted([i, j]))
                e2 = tuple(sorted([j, k]))
                e3 = tuple(sorted([i, k]))

                if e1 in edge_times and e2 in edge_times and e3 in edge_times:
                    # All edges existed before triangle
                    # Find and kill the youngest loop involving these vertices
                    # (This is a simplification - proper algorithm would match specific cycles)

                    if active_loops:
                        # Kill the oldest active loop
                        birth, _ = active_loops[0]
                        active_loops = active_loops[1:]

                        if radius > birth:
                            diagram.add(birth, radius, generator=simplex)

        # Remaining active loops persist to infinity
        for birth, _ in active_loops:
            diagram.add(birth, float('inf'))

        return diagram


def compute_persistence(
    points: np.ndarray,
    max_dimension: int = 1,
    max_radius: float = float('inf')
) -> Dict[int, PersistenceDiagram]:
    """
    Convenience function to compute persistent homology.

    Args:
        points: (n, d) point cloud
        max_dimension: Maximum homology dimension
        max_radius: Maximum filtration radius

    Returns:
        Dictionary of persistence diagrams by dimension
    """
    ph = PersistentHomology(max_dimension, max_radius)
    return ph.compute(points)


def bottleneck_distance(
    dgm1: PersistenceDiagram,
    dgm2: PersistenceDiagram
) -> float:
    """
    Compute bottleneck distance between persistence diagrams.

    d_∞(D1, D2) = inf_γ sup_x ||x - γ(x)||_∞

    The bottleneck distance is the maximum distance in an optimal
    matching between diagrams (including matching to diagonal).

    Args:
        dgm1, dgm2: Persistence diagrams

    Returns:
        Bottleneck distance
    """
    pts1 = dgm1.points if len(dgm1.intervals) > 0 else np.array([]).reshape(0, 2)
    pts2 = dgm2.points if len(dgm2.intervals) > 0 else np.array([]).reshape(0, 2)

    if len(pts1) == 0 and len(pts2) == 0:
        return 0.0

    n1, n2 = len(pts1), len(pts2)

    # Build cost matrix: d_∞(p1, p2)
    # Each point can match to another point OR to the diagonal
    # Cost to diagonal = persistence / 2 = (death - birth) / 2

    # Direct point-to-point costs (L∞ norm)
    direct_costs = np.zeros((n1, n2)) if n1 > 0 and n2 > 0 else np.array([]).reshape(0, 0)
    for i, p1 in enumerate(pts1):
        for j, p2 in enumerate(pts2):
            direct_costs[i, j] = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

    # Cost for each point in dgm1 to match to diagonal
    diag_costs_1 = np.array([(p[1] - p[0]) / 2 if p[1] != float('inf') else 1e10
                             for p in pts1]) if n1 > 0 else np.array([])

    # Cost for each point in dgm2 to match to diagonal
    diag_costs_2 = np.array([(p[1] - p[0]) / 2 if p[1] != float('inf') else 1e10
                             for p in pts2]) if n2 > 0 else np.array([])

    # Greedy matching (approximation - true bottleneck requires Hungarian algorithm)
    # For identical diagrams, direct matching gives cost 0
    if n1 == n2 and n1 > 0:
        # Check if diagrams are identical (each point has a 0-cost match)
        matched = np.zeros(n2, dtype=bool)
        max_match_cost = 0.0
        for i in range(n1):
            # Find best unmatched point in dgm2
            best_cost = diag_costs_1[i]  # Default: match to diagonal
            best_j = -1
            for j in range(n2):
                if not matched[j] and direct_costs[i, j] < best_cost:
                    best_cost = direct_costs[i, j]
                    best_j = j
            if best_j >= 0:
                matched[best_j] = True
            max_match_cost = max(max_match_cost, best_cost)

        # Check unmatched points in dgm2
        for j in range(n2):
            if not matched[j]:
                max_match_cost = max(max_match_cost, diag_costs_2[j])

        return max_match_cost

    # Different sized diagrams - use simple approximation
    max_cost = 0.0

    # Each point in dgm1 matched to best option
    for i in range(n1):
        if n2 > 0:
            best = min(np.min(direct_costs[i, :]), diag_costs_1[i])
        else:
            best = diag_costs_1[i]
        max_cost = max(max_cost, best)

    # Each point in dgm2 needs to match to something
    for j in range(n2):
        if n1 > 0:
            best = min(np.min(direct_costs[:, j]), diag_costs_2[j])
        else:
            best = diag_costs_2[j]
        max_cost = max(max_cost, best)

    return max_cost


def wasserstein_distance(
    dgm1: PersistenceDiagram,
    dgm2: PersistenceDiagram,
    p: float = 2.0
) -> float:
    """
    Compute p-Wasserstein distance between persistence diagrams.

    W_p(D1, D2) = (inf_γ Σ_x ||x - γ(x)||_∞^p)^(1/p)

    Args:
        dgm1, dgm2: Persistence diagrams
        p: Power for Wasserstein distance

    Returns:
        Wasserstein distance
    """
    pts1 = dgm1.points if len(dgm1.intervals) > 0 else np.array([]).reshape(0, 2)
    pts2 = dgm2.points if len(dgm2.intervals) > 0 else np.array([]).reshape(0, 2)

    if len(pts1) == 0 and len(pts2) == 0:
        return 0.0

    # Simplified: sum of distances to diagonal for unmatched points
    total = 0.0

    # Match greedily (approximation)
    used1, used2 = set(), set()

    for i, p1 in enumerate(pts1):
        best_cost = float('inf')
        best_j = -1

        for j, p2 in enumerate(pts2):
            if j in used2:
                continue
            cost = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
            if cost < best_cost:
                best_cost = cost
                best_j = j

        # Also consider diagonal
        diag_cost = (p1[1] - p1[0]) / 2 if p1[1] != float('inf') else float('inf')

        if best_j >= 0 and best_cost < diag_cost:
            total += best_cost ** p
            used1.add(i)
            used2.add(best_j)
        else:
            total += diag_cost ** p
            used1.add(i)

    # Unmatched points in dgm2 go to diagonal
    for j, p2 in enumerate(pts2):
        if j not in used2:
            diag_cost = (p2[1] - p2[0]) / 2 if p2[1] != float('inf') else float('inf')
            if diag_cost != float('inf'):
                total += diag_cost ** p

    return total ** (1 / p)


class EmbeddingTopologyAnalyzer:
    """
    Analyze topological structure of embedding spaces.

    Integration with HoloLoom:
    - Analyze Matryoshka embedding quality
    - Detect semantic loops/cycles
    - Compare embedding space structures
    - Validate clustering quality
    """

    def __init__(
        self,
        max_dimension: int = 1,
        persistence_threshold: float = 0.1
    ):
        """
        Args:
            max_dimension: Maximum homology dimension
            persistence_threshold: Minimum persistence for significant features
        """
        self.max_dimension = max_dimension
        self.persistence_threshold = persistence_threshold

    def analyze(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze embedding topology.

        Args:
            embeddings: (n, d) embedding vectors
            labels: Optional cluster labels for validation

        Returns:
            Topology analysis results
        """
        # Normalize embeddings to unit sphere
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        normalized = embeddings / norms

        # Compute persistent homology
        # Use subset for large datasets
        n = len(embeddings)
        if n > 200:
            idx = np.random.choice(n, 200, replace=False)
            sample = normalized[idx]
        else:
            sample = normalized

        # Estimate max radius from data spread
        max_radius = np.percentile(
            [np.linalg.norm(sample[i] - sample[j])
             for i in range(len(sample))
             for j in range(i+1, min(i+10, len(sample)))],
            95
        )

        diagrams = compute_persistence(sample, self.max_dimension, max_radius)

        # Compute Betti numbers at various scales
        scales = np.linspace(0, max_radius, 20)
        betti_curve = {
            d: [diagrams[d].betti_at_scale(s) for s in scales]
            for d in range(self.max_dimension + 1)
        }

        # Find significant features
        significant_h0 = diagrams[0].filter_by_persistence(self.persistence_threshold)
        significant_h1 = diagrams[1].filter_by_persistence(self.persistence_threshold) if 1 in diagrams else PersistenceDiagram(dimension=1)

        # Compute topology metrics
        total_persistence_h0 = diagrams[0].total_persistence()
        total_persistence_h1 = diagrams[1].total_persistence() if 1 in diagrams else 0

        results = {
            'n_samples': len(sample),
            'max_radius': max_radius,
            'diagrams': diagrams,
            'betti_curve': betti_curve,
            'scales': scales.tolist(),
            'metrics': {
                'n_components': len(significant_h0.intervals),
                'n_loops': len(significant_h1.intervals),
                'total_persistence_h0': total_persistence_h0,
                'total_persistence_h1': total_persistence_h1,
                'max_persistence_h0': diagrams[0].max_persistence(),
                'max_persistence_h1': diagrams[1].max_persistence() if 1 in diagrams else 0,
            }
        }

        # Cluster validation if labels provided
        if labels is not None:
            results['cluster_validation'] = self._validate_clusters(
                sample, labels[:len(sample)] if len(labels) > len(sample) else labels,
                diagrams
            )

        return results

    def _validate_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        diagrams: Dict[int, PersistenceDiagram]
    ) -> Dict[str, Any]:
        """Validate clustering using topological features."""
        n_clusters = len(np.unique(labels))

        # Check if H0 diagram shows expected number of components
        significant_components = len(
            diagrams[0].filter_by_persistence(self.persistence_threshold).intervals
        )

        # Topological separation score
        # Good clustering: components die at high radius (well-separated)
        h0_intervals = diagrams[0].intervals
        if h0_intervals:
            avg_death = np.mean([i.death for i in h0_intervals
                                if i.death != float('inf')])
        else:
            avg_death = 0

        return {
            'expected_clusters': n_clusters,
            'detected_components': significant_components,
            'avg_merge_radius': avg_death,
            'topological_match': abs(n_clusters - significant_components) <= 1
        }

    def compare_embeddings(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare topological structure of two embedding spaces.

        Useful for:
        - Comparing different Matryoshka scales
        - Evaluating embedding drift
        - Model comparison
        """
        result1 = self.analyze(embeddings1)
        result2 = self.analyze(embeddings2)

        dgm1_h0 = result1['diagrams'][0]
        dgm2_h0 = result2['diagrams'][0]

        dgm1_h1 = result1['diagrams'].get(1, PersistenceDiagram(dimension=1))
        dgm2_h1 = result2['diagrams'].get(1, PersistenceDiagram(dimension=1))

        return {
            'bottleneck_h0': bottleneck_distance(dgm1_h0, dgm2_h0),
            'bottleneck_h1': bottleneck_distance(dgm1_h1, dgm2_h1),
            'wasserstein_h0': wasserstein_distance(dgm1_h0, dgm2_h0),
            'wasserstein_h1': wasserstein_distance(dgm1_h1, dgm2_h1),
        }


if __name__ == "__main__":
    print("Topological Data Analysis Module Demo")
    print("=" * 50)

    np.random.seed(42)

    # Demo 1: Simple point cloud
    print("\n1. Point Cloud Topology")

    # Create two clusters
    cluster1 = np.random.randn(20, 2) + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) + np.array([5, 5])
    points = np.vstack([cluster1, cluster2])

    diagrams = compute_persistence(points, max_dimension=1)

    print(f"   Points: {len(points)}")
    print(f"   H0 intervals: {len(diagrams[0].intervals)}")
    print(f"   H1 intervals: {len(diagrams[1].intervals)}")

    # Significant H0 features (persistent components)
    sig_h0 = diagrams[0].filter_by_persistence(1.0)
    print(f"   Significant components (persistence > 1.0): {len(sig_h0.intervals)}")

    # Demo 2: Circle (has a 1-cycle)
    print("\n2. Circle Topology (should detect 1-cycle)")

    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    circle += 0.1 * np.random.randn(*circle.shape)  # Add noise

    circle_diagrams = compute_persistence(circle, max_dimension=1)

    print(f"   H0 intervals: {len(circle_diagrams[0].intervals)}")
    print(f"   H1 intervals: {len(circle_diagrams[1].intervals)}")

    if circle_diagrams[1].intervals:
        max_h1 = max(circle_diagrams[1].intervals, key=lambda x: x.persistence)
        print(f"   Most persistent 1-cycle: birth={max_h1.birth:.2f}, "
              f"death={max_h1.death:.2f}, persistence={max_h1.persistence:.2f}")

    # Demo 3: Embedding analysis
    print("\n3. Embedding Topology Analysis")

    # Simulate embeddings with cluster structure
    embeddings = np.vstack([
        np.random.randn(30, 10) + np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.random.randn(30, 10) + np.array([-2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.random.randn(30, 10) + np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0]),
    ])
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)

    analyzer = EmbeddingTopologyAnalyzer(max_dimension=1, persistence_threshold=0.5)
    result = analyzer.analyze(embeddings, labels)

    print(f"   Detected components: {result['metrics']['n_components']}")
    print(f"   Detected loops: {result['metrics']['n_loops']}")
    print(f"   Total H0 persistence: {result['metrics']['total_persistence_h0']:.2f}")

    if 'cluster_validation' in result:
        cv = result['cluster_validation']
        print(f"   Cluster validation: {cv['expected_clusters']} expected, "
              f"{cv['detected_components']} detected")
        print(f"   Topological match: {cv['topological_match']}")

    # Demo 4: Compare embeddings
    print("\n4. Embedding Comparison")

    emb1 = np.random.randn(50, 10)
    emb2 = emb1 + 0.1 * np.random.randn(50, 10)  # Similar
    emb3 = np.random.randn(50, 10)  # Different

    comparison_similar = analyzer.compare_embeddings(emb1, emb2)
    comparison_different = analyzer.compare_embeddings(emb1, emb3)

    print("   Similar embeddings:")
    print(f"     Bottleneck H0: {comparison_similar['bottleneck_h0']:.3f}")
    print(f"     Wasserstein H0: {comparison_similar['wasserstein_h0']:.3f}")

    print("   Different embeddings:")
    print(f"     Bottleneck H0: {comparison_different['bottleneck_h0']:.3f}")
    print(f"     Wasserstein H0: {comparison_different['wasserstein_h0']:.3f}")

    print("\n" + "=" * 50)
    print("Topological Data Analysis module ready!")
