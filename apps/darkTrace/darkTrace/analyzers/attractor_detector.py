"""
Attractor Detector
==================
Find stable semantic attractors in trajectory space.

Attractors are regions in semantic space that trajectories tend to
converge toward. They represent stable semantic concepts or themes
that an LLM gravitates to.

Types of attractors:
- Point attractors: Single stable position
- Limit cycles: Periodic orbits
- Strange attractors: Chaotic but bounded regions
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import logging

from darkTrace.observers.trajectory_recorder import Trajectory
from darkTrace.observers.semantic_observer import StateSnapshot


logger = logging.getLogger(__name__)


class AttractorType(Enum):
    """Types of attractors."""
    POINT = "point"  # Fixed point
    LIMIT_CYCLE = "limit_cycle"  # Periodic orbit
    STRANGE = "strange"  # Chaotic attractor


@dataclass
class Attractor:
    """A detected semantic attractor."""

    # Attractor metadata
    attractor_type: AttractorType
    confidence: float  # 0-1

    # Location
    center: List[float]  # Centroid position in 36D space
    radius: float  # Basin of attraction radius

    # Strength
    strength: float  # How strongly it attracts (0-1)
    visit_count: int  # How many times trajectories visited

    # Characteristics
    dominant_dimensions: List[str]
    dimension_scores: Dict[str, float]

    # For limit cycles
    period: Optional[int] = None  # Period in tokens

    # For strange attractors
    lyapunov_exponent: Optional[float] = None  # Chaos indicator

    # Description
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attractor_type": self.attractor_type.value,
            "confidence": self.confidence,
            "center": self.center,
            "radius": self.radius,
            "strength": self.strength,
            "visit_count": self.visit_count,
            "dominant_dimensions": self.dominant_dimensions,
            "dimension_scores": self.dimension_scores,
            "period": self.period,
            "lyapunov_exponent": self.lyapunov_exponent,
            "description": self.description,
        }


class AttractorDetector:
    """
    Detect semantic attractors in trajectory collections.

    Attractors are found by:
    1. Clustering positions to find dense regions
    2. Checking for convergence toward regions
    3. Measuring basin of attraction strength

    Usage:
        detector = AttractorDetector(n_attractors=5)

        # Detect attractors across multiple trajectories
        attractors = detector.detect(trajectories)

        # Find strongest attractor
        strongest = max(attractors, key=lambda a: a.strength)
    """

    def __init__(
        self,
        n_attractors: int = 5,
        min_visits: int = 3,
        convergence_threshold: float = 0.5,
    ):
        """
        Initialize attractor detector.

        Args:
            n_attractors: Maximum number of attractors to detect
            min_visits: Minimum visits for attractor detection
            convergence_threshold: Threshold for convergence detection
        """
        self.n_attractors = n_attractors
        self.min_visits = min_visits
        self.convergence_threshold = convergence_threshold

        logger.info(f"AttractorDetector initialized (max {n_attractors} attractors)")

    def detect(self, trajectories: List[Trajectory]) -> List[Attractor]:
        """
        Detect attractors across trajectories.

        Args:
            trajectories: List of trajectories to analyze

        Returns:
            List of detected Attractor objects
        """
        if not trajectories:
            return []

        logger.info(f"Detecting attractors in {len(trajectories)} trajectories")

        # Collect all positions and metadata
        all_positions = []
        all_snapshots = []

        for traj in trajectories:
            for snapshot in traj.snapshots:
                all_positions.append(snapshot.position)
                all_snapshots.append(snapshot)

        positions = np.array(all_positions)

        # Find clusters (potential attractors)
        cluster_centers, cluster_assignments = self._cluster_positions(
            positions, self.n_attractors
        )

        # Analyze each cluster
        attractors = []

        for cluster_id in range(len(cluster_centers)):
            # Get snapshots in this cluster
            mask = cluster_assignments == cluster_id
            cluster_positions = positions[mask]
            cluster_snapshots = [s for i, s in enumerate(all_snapshots) if mask[i]]

            if len(cluster_snapshots) < self.min_visits:
                continue

            # Compute attractor properties
            center = cluster_centers[cluster_id]
            radius = float(np.std(np.linalg.norm(cluster_positions - center, axis=1)))

            # Measure convergence strength
            strength = self._measure_attractor_strength(
                trajectories, center, radius
            )

            # Get dominant dimensions
            dominant_dims, dim_scores = self._get_cluster_dimensions(cluster_snapshots)

            # Classify attractor type
            attractor_type, period, lyapunov = self._classify_attractor_type(
                cluster_positions
            )

            # Create attractor
            attractor = Attractor(
                attractor_type=attractor_type,
                confidence=min(1.0, len(cluster_snapshots) / 50.0),
                center=center.tolist(),
                radius=radius,
                strength=strength,
                visit_count=len(cluster_snapshots),
                dominant_dimensions=dominant_dims,
                dimension_scores=dim_scores,
                period=period,
                lyapunov_exponent=lyapunov,
                description=f"{attractor_type.value} attractor at {dominant_dims[0] if dominant_dims else 'unknown'}"
            )

            attractors.append(attractor)

        # Sort by strength
        attractors.sort(key=lambda a: a.strength, reverse=True)

        logger.info(f"Detected {len(attractors)} attractors")
        return attractors

    def _cluster_positions(
        self,
        positions: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster positions using k-means.

        Args:
            positions: Position array (n_samples, n_dims)
            n_clusters: Number of clusters

        Returns:
            Tuple of (cluster_centers, cluster_assignments)
        """
        if len(positions) < n_clusters:
            # Not enough data, return all positions as centers
            assignments = np.arange(len(positions))
            return positions, assignments

        # Simple k-means implementation
        rng = np.random.RandomState(42)
        indices = rng.choice(len(positions), n_clusters, replace=False)
        centroids = positions[indices].copy()

        # Iterate
        for iteration in range(50):
            # Assign to nearest centroid
            distances = np.linalg.norm(
                positions[:, np.newaxis] - centroids,
                axis=2
            )
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                positions[assignments == i].mean(axis=0)
                if np.any(assignments == i)
                else centroids[i]
                for i in range(n_clusters)
            ])

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return centroids, assignments

    def _measure_attractor_strength(
        self,
        trajectories: List[Trajectory],
        center: np.ndarray,
        radius: float
    ) -> float:
        """
        Measure how strongly trajectories converge to this attractor.

        Args:
            trajectories: List of trajectories
            center: Attractor center position
            radius: Attractor radius

        Returns:
            Strength score (0-1)
        """
        convergence_count = 0
        total_sequences = 0

        for traj in trajectories:
            positions = np.array([s.position for s in traj.snapshots])

            # Check for convergent subsequences
            for i in range(len(positions) - 3):
                window = positions[i:i+3]
                distances = np.linalg.norm(window - center, axis=1)

                # Converging if distances are decreasing toward radius
                if distances[-1] < radius and distances[0] > distances[-1]:
                    convergence_count += 1

                total_sequences += 1

        if total_sequences == 0:
            return 0.0

        return min(1.0, convergence_count / total_sequences)

    def _get_cluster_dimensions(
        self,
        snapshots: List[StateSnapshot]
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Get dominant dimensions in cluster.

        Args:
            snapshots: Snapshots in cluster

        Returns:
            Tuple of (dominant_dimensions, dimension_scores)
        """
        dim_scores = {}

        for snapshot in snapshots:
            for dim, score in snapshot.dimension_scores.items():
                dim_scores[dim] = dim_scores.get(dim, 0.0) + score

        # Normalize
        total = sum(dim_scores.values())
        if total > 0:
            dim_scores = {dim: score / total for dim, score in dim_scores.items()}

        # Sort
        sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1], reverse=True)
        dominant_dims = [dim for dim, _ in sorted_dims[:5]]

        return dominant_dims, dim_scores

    def _classify_attractor_type(
        self,
        positions: np.ndarray
    ) -> Tuple[AttractorType, Optional[int], Optional[float]]:
        """
        Classify attractor type.

        Args:
            positions: Positions in attractor

        Returns:
            Tuple of (type, period, lyapunov_exponent)
        """
        if len(positions) < 10:
            return AttractorType.POINT, None, None

        # Check for periodicity (limit cycle)
        period = self._detect_period(positions)

        if period is not None:
            return AttractorType.LIMIT_CYCLE, period, None

        # Check for chaos (strange attractor)
        lyapunov = self._estimate_lyapunov(positions)

        if lyapunov is not None and lyapunov > 0:
            return AttractorType.STRANGE, None, lyapunov

        # Default to point attractor
        return AttractorType.POINT, None, None

    def _detect_period(
        self,
        positions: np.ndarray,
        max_period: int = 20
    ) -> Optional[int]:
        """
        Detect periodic pattern in positions.

        Args:
            positions: Position sequence
            max_period: Maximum period to check

        Returns:
            Detected period or None
        """
        for period in range(2, min(max_period, len(positions) // 2)):
            # Compare positions with lag=period
            similarities = []

            for i in range(len(positions) - period):
                dist = np.linalg.norm(positions[i] - positions[i + period])
                similarities.append(dist)

            # If consistently similar, we have periodicity
            avg_similarity = np.mean(similarities)

            if avg_similarity < 0.5:  # Threshold for similarity
                return period

        return None

    def _estimate_lyapunov(
        self,
        positions: np.ndarray
    ) -> Optional[float]:
        """
        Estimate largest Lyapunov exponent (chaos indicator).

        Args:
            positions: Position sequence

        Returns:
            Lyapunov exponent or None
        """
        if len(positions) < 50:
            return None

        # Simplified Lyapunov estimation
        # Track divergence of nearby trajectories
        divergences = []

        for i in range(len(positions) - 10):
            # Find nearest neighbor
            current = positions[i]
            distances = np.linalg.norm(positions[i+1:i+50] - current, axis=1)

            if len(distances) == 0:
                continue

            nearest_idx = np.argmin(distances) + i + 1

            if nearest_idx + 5 >= len(positions):
                continue

            # Track divergence over 5 steps
            initial_sep = distances[nearest_idx - i - 1]
            final_sep = np.linalg.norm(positions[i+5] - positions[nearest_idx+5])

            if initial_sep > 1e-6:
                divergence = np.log(final_sep / initial_sep) / 5
                divergences.append(divergence)

        if not divergences:
            return None

        # Average Lyapunov exponent
        return float(np.mean(divergences))
