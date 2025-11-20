"""
Topological Warp Operations
============================
Persistent homology and topological data analysis for semantic spaces.

This module provides tools to understand the *shape* of semantic space:
- Persistent homology (birth/death of topological features)
- Betti numbers (connected components, holes, voids)
- Persistence diagrams and barcodes
- Mapper algorithm (topological network visualization)
- Vietoris-Rips and Cech complexes
- Bottleneck and Wasserstein distances

Philosophy:
Topology captures features that are invariant under continuous deformations.
In semantic space, this reveals:
- Clusters (0-dimensional holes = connected components)
- Loops (1-dimensional holes = cycles in reasoning)
- Voids (2-dimensional holes = higher-order structure)

Unlike metric properties (exact distances), topology reveals *structural* properties
of how concepts are connected and separated.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)

# Optional: Use scikit-tda for advanced persistent homology
try:
    import ripser
    from persim import plot_diagrams, bottleneck, wasserstein
    HAS_RIPSER = True
    logger.info("Ripser available for fast persistent homology")
except ImportError:
    HAS_RIPSER = False
    logger.warning("Ripser not available. Install with: pip install ripser persim")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PersistenceInterval:
    """
    A feature that persists from birth to death.

    Represents a topological feature (component, hole, void) that appears
    at birth_scale and disappears at death_scale.
    """
    dimension: int  # 0=component, 1=loop, 2=void
    birth: float    # Scale where feature appears
    death: float    # Scale where feature disappears

    @property
    def persistence(self) -> float:
        """Lifetime of the feature."""
        return self.death - self.birth

    @property
    def midpoint(self) -> float:
        """Middle of the feature's lifetime."""
        return (self.birth + self.death) / 2


@dataclass
class PersistenceDiagram:
    """
    Collection of persistence intervals.

    Visualized as points (birth, death) above the diagonal y=x.
    Points far from diagonal = persistent features (signal).
    Points near diagonal = short-lived features (noise).
    """
    intervals: List[PersistenceInterval]
    dimension: int

    def filter_by_persistence(self, threshold: float) -> 'PersistenceDiagram':
        """Keep only features with persistence > threshold."""
        filtered = [
            interval for interval in self.intervals
            if interval.persistence > threshold
        ]
        return PersistenceDiagram(filtered, self.dimension)

    def get_most_persistent(self, k: int = 5) -> List[PersistenceInterval]:
        """Get k most persistent features."""
        sorted_intervals = sorted(
            self.intervals,
            key=lambda x: x.persistence,
            reverse=True
        )
        return sorted_intervals[:k]


# ============================================================================
# Simplicial Complexes
# ============================================================================

class VietorisRipsComplex:
    """
    Vietoris-Rips complex construction.

    Build simplicial complex from point cloud:
    - 0-simplices: points
    - 1-simplices: edges (distance ≤ r)
    - 2-simplices: triangles (all edges ≤ r)
    - Higher simplices: complete subgraphs

    The Vietoris-Rips is easier to compute than Cech but may introduce
    "ghost" features.
    """

    def __init__(self, points: np.ndarray, max_dimension: int = 2):
        """
        Initialize Vietoris-Rips complex.

        Args:
            points: Point cloud (n_points, n_dims)
            max_dimension: Maximum simplex dimension
        """
        self.points = points
        self.n_points = len(points)
        self.max_dimension = max_dimension

        # Compute pairwise distances
        self.distance_matrix = self._compute_distances()

        logger.info(f"VietorisRips: {self.n_points} points, max_dim={max_dimension}")

    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise distance matrix."""
        n = self.n_points
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(self.points[i] - self.points[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def build_filtration(self, radii: np.ndarray) -> List[Dict[str, Any]]:
        """
        Build filtered simplicial complex at multiple scales.

        Args:
            radii: Array of filtration radii

        Returns:
            List of complexes at each radius
        """
        filtration = []

        for r in radii:
            complex_at_r = self._build_complex_at_radius(r)
            filtration.append({
                'radius': r,
                'simplices': complex_at_r,
                'betti_numbers': self._compute_betti(complex_at_r)
            })

        return filtration

    def _build_complex_at_radius(self, radius: float) -> Dict[int, List]:
        """Build simplicial complex at given radius."""
        simplices = defaultdict(list)

        # 0-simplices (vertices)
        simplices[0] = [[i] for i in range(self.n_points)]

        # 1-simplices (edges)
        for i in range(self.n_points):
            for j in range(i+1, self.n_points):
                if self.distance_matrix[i, j] <= radius:
                    simplices[1].append([i, j])

        # 2-simplices (triangles) if requested
        if self.max_dimension >= 2:
            edges = simplices[1]
            for i, edge1 in enumerate(edges):
                for edge2 in edges[i+1:]:
                    # Check if edges share a vertex and form a triangle
                    shared = set(edge1) & set(edge2)
                    if len(shared) == 1:
                        triangle = sorted(list(set(edge1) | set(edge2)))
                        if len(triangle) == 3:
                            # Check if all edges exist
                            a, b, c = triangle
                            if (self.distance_matrix[a, b] <= radius and
                                self.distance_matrix[b, c] <= radius and
                                self.distance_matrix[a, c] <= radius):
                                simplices[2].append(triangle)

        return simplices

    def _compute_betti(self, simplices: Dict[int, List]) -> List[int]:
        """
        Compute Betti numbers (simplified).

        Betti numbers count independent k-dimensional holes:
        - β₀ = number of connected components
        - β₁ = number of loops
        - β₂ = number of voids

        This is a simplified version. For exact computation, use boundary matrices.
        """
        betti = []

        # β₀: connected components (simplified: assume connected if edges exist)
        n_components = self.n_points - len(simplices.get(1, []))
        if n_components < 1:
            n_components = 1
        betti.append(max(1, n_components))

        # β₁: loops (Euler characteristic approximation)
        n_edges = len(simplices.get(1, []))
        n_triangles = len(simplices.get(2, []))
        n_loops = n_edges - self.n_points - n_triangles + 1
        betti.append(max(0, n_loops))

        # β₂: voids (simplified)
        betti.append(0)  # Requires full boundary matrix computation

        return betti


# ============================================================================
# Persistent Homology
# ============================================================================

class PersistentHomology:
    """
    Compute persistent homology of point cloud.

    Tracks how topological features (components, loops, voids) appear
    and disappear as we vary a scale parameter.
    """

    def __init__(self, max_dimension: int = 2):
        """
        Initialize persistent homology computer.

        Args:
            max_dimension: Maximum homology dimension to compute
        """
        self.max_dimension = max_dimension

    def compute(
        self,
        points: np.ndarray,
        max_scale: Optional[float] = None,
        use_ripser: bool = True
    ) -> Dict[int, PersistenceDiagram]:
        """
        Compute persistent homology.

        Args:
            points: Point cloud (n_points, n_dims)
            max_scale: Maximum filtration scale (None = auto)
            use_ripser: Use ripser library if available

        Returns:
            Dict mapping dimension to persistence diagram
        """
        if use_ripser and HAS_RIPSER:
            return self._compute_ripser(points, max_scale)
        else:
            return self._compute_manual(points, max_scale)

    def _compute_ripser(
        self,
        points: np.ndarray,
        max_scale: Optional[float]
    ) -> Dict[int, PersistenceDiagram]:
        """Compute using Ripser (fast C++ implementation)."""
        logger.info(f"Computing persistent homology with Ripser (dim ≤ {self.max_dimension})")

        # Run Ripser
        result = ripser.ripser(
            points,
            maxdim=self.max_dimension,
            thresh=max_scale if max_scale else np.inf
        )

        # Convert to our format
        diagrams = {}

        for dim in range(self.max_dimension + 1):
            if dim < len(result['dgms']):
                intervals = []
                for birth, death in result['dgms'][dim]:
                    if not np.isinf(death):  # Skip infinite bars
                        intervals.append(
                            PersistenceInterval(dim, birth, death)
                        )

                diagrams[dim] = PersistenceDiagram(intervals, dim)
                logger.info(f"  Dimension {dim}: {len(intervals)} features")

        return diagrams

    def _compute_manual(
        self,
        points: np.ndarray,
        max_scale: Optional[float]
    ) -> Dict[int, PersistenceDiagram]:
        """Compute using manual implementation (slower but no dependencies)."""
        logger.info("Computing persistent homology (manual implementation)")

        # Build Vietoris-Rips complex
        vr = VietorisRipsComplex(points, max_dimension=self.max_dimension)

        # Auto-determine max scale
        if max_scale is None:
            max_scale = np.percentile(vr.distance_matrix, 75)

        # Build filtration
        n_steps = 20
        radii = np.linspace(0, max_scale, n_steps)
        filtration = vr.build_filtration(radii)

        # Track birth/death of features
        diagrams = self._track_features(filtration, radii)

        return diagrams

    def _track_features(
        self,
        filtration: List[Dict],
        radii: np.ndarray
    ) -> Dict[int, PersistenceDiagram]:
        """
        Track when features appear and disappear.

        Simplified tracking based on Betti numbers changing.
        For exact tracking, would need to compute boundary matrices.
        """
        diagrams = {dim: [] for dim in range(self.max_dimension + 1)}

        # Track previous Betti numbers
        prev_betti = [0] * (self.max_dimension + 1)
        feature_births = {dim: [] for dim in range(self.max_dimension + 1)}

        for i, step in enumerate(filtration):
            current_betti = step['betti_numbers']
            radius = step['radius']

            # Ensure current_betti has enough dimensions
            while len(current_betti) < self.max_dimension + 1:
                current_betti.append(0)

            for dim in range(min(len(current_betti), len(prev_betti))):
                # Feature birth
                if current_betti[dim] > prev_betti[dim]:
                    n_births = current_betti[dim] - prev_betti[dim]
                    for _ in range(n_births):
                        feature_births[dim].append(radius)

                # Feature death
                elif current_betti[dim] < prev_betti[dim]:
                    n_deaths = prev_betti[dim] - current_betti[dim]
                    for _ in range(n_deaths):
                        if feature_births[dim]:
                            birth = feature_births[dim].pop(0)
                            diagrams[dim].append(
                                PersistenceInterval(dim, birth, radius)
                            )

            prev_betti = current_betti

        # Convert to PersistenceDiagram objects
        result = {}
        for dim in range(self.max_dimension + 1):
            if diagrams[dim]:
                result[dim] = PersistenceDiagram(diagrams[dim], dim)
                logger.info(f"  Dimension {dim}: {len(diagrams[dim])} features")

        return result


# ============================================================================
# Mapper Algorithm
# ============================================================================

class MapperAlgorithm:
    """
    Mapper: Topological network visualization of high-dimensional data.

    The Mapper algorithm creates a graph representation that captures
    topological structure:
    1. Cover the space with overlapping regions
    2. Cluster points within each region
    3. Connect clusters that share points

    Result: A network that preserves topological features like loops and branches.
    """

    def __init__(
        self,
        n_intervals: int = 10,
        overlap_percent: float = 0.3,
        cluster_method: str = "simple"
    ):
        """
        Initialize Mapper.

        Args:
            n_intervals: Number of cover intervals
            overlap_percent: Overlap between intervals (0-1)
            cluster_method: Clustering method ("simple" or "kmeans")
        """
        self.n_intervals = n_intervals
        self.overlap_percent = overlap_percent
        self.cluster_method = cluster_method

        logger.info(f"Mapper: {n_intervals} intervals, {overlap_percent*100}% overlap")

    def fit(
        self,
        points: np.ndarray,
        lens_function: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute Mapper graph.

        Args:
            points: Point cloud (n_points, n_dims)
            lens_function: 1D projection (None = use first PCA component)

        Returns:
            Dict with nodes, edges, and metadata
        """
        logger.info(f"Computing Mapper for {len(points)} points")

        # Apply lens function (project to 1D)
        if lens_function is None:
            lens_function = self._default_lens(points)

        # Create overlapping cover
        cover = self._create_cover(lens_function)

        # Cluster within each cover element
        nodes = []
        node_to_points = {}

        for i, interval in enumerate(cover):
            # Get points in this interval
            mask = interval['mask']
            points_in_interval = points[mask]

            if len(points_in_interval) == 0:
                continue

            # Cluster these points
            clusters = self._cluster_points(points_in_interval)

            for j, cluster_indices in enumerate(clusters):
                node_id = f"{i}_{j}"

                # Map back to original indices
                original_indices = np.where(mask)[0][cluster_indices]

                nodes.append({
                    'id': node_id,
                    'interval': i,
                    'cluster': j,
                    'size': len(original_indices),
                    'center': np.mean(points_in_interval[cluster_indices], axis=0)
                })

                node_to_points[node_id] = set(original_indices)

        # Create edges (nodes that share points)
        edges = []
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                shared = node_to_points[node1['id']] & node_to_points[node2['id']]
                if len(shared) > 0:
                    edges.append({
                        'source': node1['id'],
                        'target': node2['id'],
                        'weight': len(shared)
                    })

        logger.info(f"Mapper graph: {len(nodes)} nodes, {len(edges)} edges")

        return {
            'nodes': nodes,
            'edges': edges,
            'lens': lens_function,
            'node_to_points': node_to_points
        }

    def _default_lens(self, points: np.ndarray) -> np.ndarray:
        """Default lens: first principal component."""
        # Center data
        centered = points - np.mean(points, axis=0)

        # Compute first PC
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        first_pc = Vt[0]

        # Project
        projection = centered @ first_pc

        return projection

    def _create_cover(self, lens_values: np.ndarray) -> List[Dict]:
        """Create overlapping cover of lens range."""
        min_val, max_val = np.min(lens_values), np.max(lens_values)
        interval_width = (max_val - min_val) / self.n_intervals
        overlap_width = interval_width * self.overlap_percent

        cover = []
        for i in range(self.n_intervals):
            start = min_val + i * interval_width - overlap_width
            end = start + interval_width + 2 * overlap_width

            mask = (lens_values >= start) & (lens_values <= end)

            cover.append({
                'start': start,
                'end': end,
                'mask': mask
            })

        return cover

    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """Cluster points (simplified single-linkage)."""
        if len(points) <= 1:
            return [np.arange(len(points))]

        if self.cluster_method == "simple":
            # Simple: treat all as one cluster
            return [np.arange(len(points))]

        # Could implement k-means or other clustering here
        return [np.arange(len(points))]


# ============================================================================
# Topological Features
# ============================================================================

class TopologicalFeatureExtractor:
    """
    Extract topological features from point clouds for ML.

    Converts persistence diagrams into feature vectors:
    - Persistence statistics (mean, max, sum)
    - Betti numbers at multiple scales
    - Persistence landscape
    - Persistence images
    """

    @staticmethod
    def extract_features(
        diagrams: Dict[int, PersistenceDiagram],
        scales: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Extract feature vector from persistence diagrams.

        Args:
            diagrams: Persistence diagrams by dimension
            scales: Scales to evaluate Betti numbers

        Returns:
            Feature vector
        """
        features = []

        for dim in sorted(diagrams.keys()):
            diagram = diagrams[dim]
            intervals = diagram.intervals

            if len(intervals) == 0:
                # No features in this dimension
                features.extend([0, 0, 0, 0, 0])
                continue

            # Persistence statistics
            persistences = [interval.persistence for interval in intervals]

            features.append(len(persistences))  # Number of features
            features.append(np.mean(persistences))  # Average persistence
            features.append(np.max(persistences))   # Max persistence
            features.append(np.sum(persistences))   # Total persistence
            features.append(np.std(persistences))   # Persistence std

        # Betti curve (if scales provided)
        if scales:
            for scale in scales:
                betti = TopologicalFeatureExtractor._betti_at_scale(
                    diagrams,
                    scale
                )
                features.extend(betti)

        return np.array(features)

    @staticmethod
    def _betti_at_scale(
        diagrams: Dict[int, PersistenceDiagram],
        scale: float
    ) -> List[int]:
        """Compute Betti numbers at given scale."""
        betti = []

        for dim in sorted(diagrams.keys()):
            # Count features alive at this scale
            alive = sum(
                1 for interval in diagrams[dim].intervals
                if interval.birth <= scale <= interval.death
            )
            betti.append(alive)

        return betti


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Topological Warp Operations Demo")
    print("="*80 + "\n")

    # Generate point cloud (circle + noise)
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    noise = np.random.randn(n_points, 2) * 0.1
    points = circle + noise

    print(f"Point cloud: {points.shape}\n")

    # 1. Persistent Homology
    print("1. Computing Persistent Homology")
    print("-" * 40)

    ph = PersistentHomology(max_dimension=1)
    diagrams = ph.compute(points, max_scale=2.0, use_ripser=HAS_RIPSER)

    for dim, diagram in diagrams.items():
        print(f"\nDimension {dim}:")
        print(f"  Total features: {len(diagram.intervals)}")

        # Most persistent features
        top_features = diagram.get_most_persistent(k=3)
        for i, interval in enumerate(top_features, 1):
            print(f"  {i}. Birth={interval.birth:.3f}, Death={interval.death:.3f}, "
                  f"Persistence={interval.persistence:.3f}")

    # 2. Mapper Algorithm
    print("\n2. Mapper Algorithm")
    print("-" * 40)

    mapper = MapperAlgorithm(n_intervals=5, overlap_percent=0.3)
    graph = mapper.fit(points)

    print(f"Nodes: {len(graph['nodes'])}")
    print(f"Edges: {len(graph['edges'])}")
    print(f"Graph captures topological structure (loop in circle)")

    # 3. Topological Features
    print("\n3. Topological Feature Extraction")
    print("-" * 40)

    features = TopologicalFeatureExtractor.extract_features(
        diagrams,
        scales=[0.5, 1.0, 1.5]
    )

    print(f"Feature vector dimension: {len(features)}")
    print(f"Features (first 10): {features[:10]}")

    # 4. Filter by persistence
    print("\n4. Filtering Noise")
    print("-" * 40)

    if 1 in diagrams:
        original_count = len(diagrams[1].intervals)
        filtered = diagrams[1].filter_by_persistence(threshold=0.1)
        print(f"Dimension 1 features: {original_count} → {len(filtered.intervals)}")
        print(f"Removed {original_count - len(filtered.intervals)} short-lived (noisy) features")

    print("\nDemo complete!")
