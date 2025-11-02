"""
Fingerprint Generator
=====================
Create unique semantic fingerprints of LLMs from their trajectories.

A semantic fingerprint captures:
- Average semantic position (centroid in 36D space)
- Dimension preferences (which dimensions are most active)
- Trajectory characteristics (velocity, curvature patterns)
- Attractor locations (stable semantic regions)
- Signature patterns (recurring motifs)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import logging

from darkTrace.observers.trajectory_recorder import Trajectory


logger = logging.getLogger(__name__)


@dataclass
class SemanticFingerprint:
    """
    A unique semantic fingerprint of an LLM or model.

    Captures the characteristic semantic behavior patterns that
    distinguish one model from another.
    """

    # Metadata
    model_name: str
    num_trajectories: int
    total_tokens: int

    # Fingerprint vector (128D compressed representation)
    fingerprint_vector: List[float]

    # Dimension preferences (normalized scores 0-1)
    dimension_preferences: Dict[str, float]

    # Trajectory statistics
    avg_velocity: float
    std_velocity: float
    avg_curvature: float
    std_curvature: float

    # Signature patterns (recurring dimension combinations)
    signature_patterns: List[str] = field(default_factory=list)

    # Attractor locations (stable semantic positions)
    attractor_locations: List[List[float]] = field(default_factory=list)

    # Ethical profile (if available)
    ethical_profile: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "num_trajectories": self.num_trajectories,
            "total_tokens": self.total_tokens,
            "fingerprint_vector": self.fingerprint_vector,
            "dimension_preferences": self.dimension_preferences,
            "avg_velocity": self.avg_velocity,
            "std_velocity": self.std_velocity,
            "avg_curvature": self.avg_curvature,
            "std_curvature": self.std_curvature,
            "signature_patterns": self.signature_patterns,
            "attractor_locations": self.attractor_locations,
            "ethical_profile": self.ethical_profile,
        }


class FingerprintGenerator:
    """
    Generate semantic fingerprints from trajectory collections.

    Usage:
        generator = FingerprintGenerator(dimensions=128)

        # Generate fingerprint from trajectories
        fingerprint = generator.generate(trajectories, model_name="gpt-4")

        # Compare fingerprints
        similarity = generator.compare(fingerprint1, fingerprint2)
    """

    def __init__(
        self,
        dimensions: int = 128,  # Fingerprint vector dimensions
        min_trajectories: int = 5,  # Minimum trajectories needed
    ):
        """
        Initialize fingerprint generator.

        Args:
            dimensions: Dimensionality of fingerprint vector
            min_trajectories: Minimum trajectories required for robust fingerprint
        """
        self.dimensions = dimensions
        self.min_trajectories = min_trajectories

        logger.info(f"FingerprintGenerator initialized ({dimensions}D)")

    def generate(
        self,
        trajectories: List[Trajectory],
        model_name: str,
    ) -> SemanticFingerprint:
        """
        Generate semantic fingerprint from trajectories.

        Args:
            trajectories: List of trajectories from the model
            model_name: Name/identifier of the model

        Returns:
            SemanticFingerprint object

        Raises:
            ValueError: If insufficient trajectories provided
        """
        if len(trajectories) < self.min_trajectories:
            logger.warning(
                f"Only {len(trajectories)} trajectories provided, "
                f"recommended minimum is {self.min_trajectories}"
            )

        logger.info(f"Generating fingerprint for '{model_name}' from {len(trajectories)} trajectories")

        # Extract all positions
        all_positions = []
        all_velocities = []
        all_curvatures = []
        dimension_scores = {}

        total_tokens = 0

        for traj in trajectories:
            total_tokens += traj.total_tokens

            for snapshot in traj.snapshots:
                all_positions.append(snapshot.position)

                if snapshot.velocity_magnitude > 0:
                    all_velocities.append(snapshot.velocity_magnitude)

                if snapshot.curvature is not None:
                    all_curvatures.append(snapshot.curvature)

                # Accumulate dimension scores
                for dim, score in snapshot.dimension_scores.items():
                    dimension_scores[dim] = dimension_scores.get(dim, 0.0) + score

        # Convert to numpy
        positions = np.array(all_positions)

        # Generate fingerprint vector using PCA compression
        fingerprint_vector = self._compute_fingerprint_vector(positions)

        # Normalize dimension preferences
        total_score = sum(dimension_scores.values())
        dimension_preferences = {
            dim: score / total_score
            for dim, score in dimension_scores.items()
        }

        # Sort and keep top dimensions
        sorted_dims = sorted(
            dimension_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        dimension_preferences = dict(sorted_dims[:20])  # Top 20

        # Compute trajectory statistics
        avg_velocity = float(np.mean(all_velocities)) if all_velocities else 0.0
        std_velocity = float(np.std(all_velocities)) if all_velocities else 0.0
        avg_curvature = float(np.mean(all_curvatures)) if all_curvatures else 0.0
        std_curvature = float(np.std(all_curvatures)) if all_curvatures else 0.0

        # Find signature patterns
        signature_patterns = self._find_signature_patterns(trajectories)

        # Find attractors (clusters in position space)
        attractor_locations = self._find_attractors(positions)

        # Extract ethical profile if available
        ethical_profile = self._extract_ethical_profile(trajectories)

        return SemanticFingerprint(
            model_name=model_name,
            num_trajectories=len(trajectories),
            total_tokens=total_tokens,
            fingerprint_vector=fingerprint_vector,
            dimension_preferences=dimension_preferences,
            avg_velocity=avg_velocity,
            std_velocity=std_velocity,
            avg_curvature=avg_curvature,
            std_curvature=std_curvature,
            signature_patterns=signature_patterns,
            attractor_locations=attractor_locations,
            ethical_profile=ethical_profile,
        )

    def _compute_fingerprint_vector(
        self,
        positions: np.ndarray
    ) -> List[float]:
        """
        Compress position data to fingerprint vector using PCA.

        Args:
            positions: Array of shape (n_samples, n_dims)

        Returns:
            Fingerprint vector of length self.dimensions
        """
        if len(positions) == 0:
            return [0.0] * self.dimensions

        # Center the data
        centered = positions - np.mean(positions, axis=0)

        # Compute covariance matrix
        cov = np.cov(centered.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top components up to self.dimensions
        n_components = min(self.dimensions, len(eigenvalues))
        fingerprint = eigenvalues[:n_components].tolist()

        # Pad if needed
        if len(fingerprint) < self.dimensions:
            fingerprint += [0.0] * (self.dimensions - len(fingerprint))

        return fingerprint

    def _find_signature_patterns(
        self,
        trajectories: List[Trajectory]
    ) -> List[str]:
        """
        Find recurring dimension patterns.

        Args:
            trajectories: List of trajectories

        Returns:
            List of signature pattern strings
        """
        patterns = []

        # Track dimension co-occurrences
        cooccurrences = {}

        for traj in trajectories:
            for snapshot in traj.snapshots:
                # Get top 3 dimensions
                sorted_dims = sorted(
                    snapshot.dimension_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                dims = tuple(sorted([d for d, _ in sorted_dims]))

                if len(dims) >= 2:
                    cooccurrences[dims] = cooccurrences.get(dims, 0) + 1

        # Find most common patterns
        sorted_patterns = sorted(
            cooccurrences.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        patterns = ["+".join(dims) for dims, _ in sorted_patterns]

        return patterns

    def _find_attractors(
        self,
        positions: np.ndarray,
        n_attractors: int = 5,
    ) -> List[List[float]]:
        """
        Find attractor locations using simple k-means clustering.

        Args:
            positions: Position array
            n_attractors: Number of attractors to find

        Returns:
            List of attractor locations
        """
        if len(positions) < n_attractors:
            return positions.tolist()

        # Simple k-means (for real impl, use sklearn)
        # Random initialization
        rng = np.random.RandomState(42)
        indices = rng.choice(len(positions), n_attractors, replace=False)
        centroids = positions[indices]

        # Few iterations
        for _ in range(10):
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
                for i in range(n_attractors)
            ])

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return centroids.tolist()

    def _extract_ethical_profile(
        self,
        trajectories: List[Trajectory]
    ) -> Optional[Dict[str, float]]:
        """
        Extract ethical profile if available.

        Args:
            trajectories: List of trajectories

        Returns:
            Ethical profile dict or None
        """
        ethical_scores = {}
        count = 0

        for traj in trajectories:
            for snapshot in traj.snapshots:
                if snapshot.ethical_valence is not None:
                    count += 1

                    if snapshot.ethical_dimensions:
                        for dim, score in snapshot.ethical_dimensions.items():
                            ethical_scores[dim] = ethical_scores.get(dim, 0.0) + score

        if count == 0:
            return None

        # Normalize
        return {
            dim: score / count
            for dim, score in ethical_scores.items()
        }

    def compare(
        self,
        fingerprint1: SemanticFingerprint,
        fingerprint2: SemanticFingerprint,
    ) -> Dict[str, float]:
        """
        Compare two semantic fingerprints.

        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint

        Returns:
            Dictionary with similarity metrics
        """
        # Vector similarity (cosine)
        v1 = np.array(fingerprint1.fingerprint_vector)
        v2 = np.array(fingerprint2.fingerprint_vector)

        cosine_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

        # Dimension preference overlap
        dims1 = set(fingerprint1.dimension_preferences.keys())
        dims2 = set(fingerprint2.dimension_preferences.keys())
        dim_overlap = len(dims1 & dims2) / max(len(dims1 | dims2), 1)

        # Velocity similarity
        velocity_diff = abs(fingerprint1.avg_velocity - fingerprint2.avg_velocity)
        velocity_sim = 1.0 / (1.0 + velocity_diff)

        # Curvature similarity
        curvature_diff = abs(fingerprint1.avg_curvature - fingerprint2.avg_curvature)
        curvature_sim = 1.0 / (1.0 + curvature_diff)

        # Pattern overlap
        patterns1 = set(fingerprint1.signature_patterns)
        patterns2 = set(fingerprint2.signature_patterns)
        pattern_overlap = len(patterns1 & patterns2) / max(len(patterns1 | patterns2), 1)

        # Overall similarity (weighted average)
        overall = (
            0.4 * cosine_sim +
            0.2 * dim_overlap +
            0.15 * velocity_sim +
            0.15 * curvature_sim +
            0.1 * pattern_overlap
        )

        return {
            "overall_similarity": float(overall),
            "vector_similarity": float(cosine_sim),
            "dimension_overlap": float(dim_overlap),
            "velocity_similarity": float(velocity_sim),
            "curvature_similarity": float(curvature_sim),
            "pattern_overlap": float(pattern_overlap),
        }
