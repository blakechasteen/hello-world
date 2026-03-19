"""
Advanced Flocking Metrics — Order Parameters, Clustering, Correlation
=====================================================================

Quantitative analysis of emergent collective behavior in flocking
simulations. Extends the base Boids module with measurement tools.

Core Concepts:
- Order parameter: Global alignment measure (Vicsek model)
- Clustering coefficient: Local spatial density analysis
- Velocity correlation: Spatial correlation of velocities

Mathematical Foundation:
Φ = (1/N)|Σ v_i / |v_i||                    (alignment order parameter)
C(r) = <v_i · v_j> / <|v_i||v_j|>           (velocity correlation at distance r)
clustering = fraction of neighbors within radius forming connected subgroups

Applications to Warp Space:
- Measuring coherence of embedding drift in latent space
- Detecting phase transitions in collective retrieval behavior
- Quantifying consensus convergence in distributed scoring

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# FLOCKING METRICS
# ============================================================================

class FlockingMetrics:
    """
    Metrics for analyzing emergent collective behavior in flocking systems.

    These metrics quantify the degree of order, clustering, and correlation
    in a population of agents with positions and velocities.
    """

    @staticmethod
    def order_parameter(velocities: np.ndarray) -> float:
        """
        Vicsek order parameter: measures global velocity alignment.

        Φ = (1/N) |Σ_i v_i / |v_i||

        Φ = 0: completely disordered (random directions)
        Φ = 1: perfectly aligned (all moving in same direction)

        Args:
            velocities: Agent velocities (n_agents, d).

        Returns:
            Φ ∈ [0, 1].
        """
        n = velocities.shape[0]
        if n == 0:
            return 0.0

        # Normalize each velocity to unit vector
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        unit_vels = velocities / norms

        # Average direction
        avg_direction = np.mean(unit_vels, axis=0)
        phi = float(np.linalg.norm(avg_direction))

        return np.clip(phi, 0.0, 1.0)

    @staticmethod
    def clustering_coefficient(positions: np.ndarray,
                               radius: float) -> float:
        """
        Spatial clustering coefficient.

        For each agent, compute the fraction of its neighbors (within radius)
        that are also neighbors of each other. Average across all agents.

        This measures local spatial density and group formation.
        C = 0: no clustering (agents spread uniformly)
        C = 1: perfect clustering (all neighbors of each agent are mutual neighbors)

        Args:
            positions: Agent positions (n_agents, d).
            radius: Neighborhood radius.

        Returns:
            Average clustering coefficient ∈ [0, 1].
        """
        n = positions.shape[0]
        if n < 3:
            return 0.0

        # Compute distance matrix
        dists = np.zeros((n, n))
        for i in range(n):
            dists[i] = np.linalg.norm(positions - positions[i], axis=1)

        total_cc = 0.0
        counted = 0

        for i in range(n):
            # Find neighbors within radius (excluding self)
            neighbors = np.where((dists[i] < radius) & (dists[i] > 1e-10))[0]
            k = len(neighbors)

            if k < 2:
                continue

            # Count edges among neighbors
            edges = 0
            possible = k * (k - 1) / 2
            for a in range(len(neighbors)):
                for b in range(a + 1, len(neighbors)):
                    if dists[neighbors[a], neighbors[b]] < radius:
                        edges += 1

            total_cc += edges / possible
            counted += 1

        if counted == 0:
            return 0.0

        return float(total_cc / counted)

    @staticmethod
    def velocity_correlation(velocities: np.ndarray,
                             positions: np.ndarray | None = None,
                             n_bins: int = 20,
                             max_dist: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Spatial velocity correlation function C(r).

        C(r) = <(v_i · v_j) / (|v_i| |v_j|)>  for pairs at distance r.

        This reveals the correlation length: how far alignment propagates.

        Args:
            velocities: Agent velocities (n_agents, d).
            positions: Agent positions (n_agents, d). If None, uses indices.
            n_bins: Number of distance bins.
            max_dist: Maximum distance to consider.

        Returns:
            (distances, correlations): bin centers and average correlations.
        """
        n = velocities.shape[0]

        if positions is None:
            # Use index-based distances
            positions = np.arange(n).reshape(-1, 1).astype(float)

        # Compute pairwise distances and velocity correlations
        dists = []
        corrs = []

        norms = np.linalg.norm(velocities, axis=1)

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if norms[i] > 1e-10 and norms[j] > 1e-10:
                    c = float(np.dot(velocities[i], velocities[j]) /
                              (norms[i] * norms[j]))
                    dists.append(d)
                    corrs.append(c)

        if not dists:
            return np.zeros(n_bins), np.zeros(n_bins)

        dists = np.array(dists)
        corrs = np.array(corrs)

        if max_dist is None:
            max_dist = np.max(dists) * 1.01

        # Bin
        bin_edges = np.linspace(0, max_dist, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_corrs = np.zeros(n_bins)

        for b in range(n_bins):
            mask = (dists >= bin_edges[b]) & (dists < bin_edges[b + 1])
            if np.any(mask):
                bin_corrs[b] = np.mean(corrs[mask])

        return bin_centers, bin_corrs

    @staticmethod
    def polarization(velocities: np.ndarray) -> float:
        """
        Polarization: fraction of agents moving in the majority direction.

        Measures how unidirectional the flock is (vs. rotating/milling).
        """
        n = velocities.shape[0]
        if n == 0:
            return 0.0

        # Average velocity (not unit velocity)
        avg_vel = np.mean(velocities, axis=0)
        avg_speed = np.linalg.norm(avg_vel)

        # Average speed of individuals
        speeds = np.linalg.norm(velocities, axis=1)
        avg_individual_speed = np.mean(speeds)

        if avg_individual_speed < 1e-10:
            return 0.0

        return float(avg_speed / avg_individual_speed)

    @staticmethod
    def angular_momentum(positions: np.ndarray,
                         velocities: np.ndarray) -> float:
        """
        Normalized angular momentum: detects milling/rotating behavior.

        L = (1/N) |Σ (r_i × v_i) / (|r_i| |v_i|)|

        where r_i is position relative to centroid.
        L ≈ 0: translating flock, L ≈ 1: rotating/milling flock.
        """
        n = positions.shape[0]
        if n == 0:
            return 0.0

        centroid = np.mean(positions, axis=0)
        r = positions - centroid

        total = 0.0
        count = 0

        for i in range(n):
            r_norm = np.linalg.norm(r[i])
            v_norm = np.linalg.norm(velocities[i])
            if r_norm > 1e-10 and v_norm > 1e-10:
                # Cross product magnitude (works in 2D and 3D)
                if positions.shape[1] == 2:
                    cross = abs(r[i, 0] * velocities[i, 1] -
                                r[i, 1] * velocities[i, 0])
                else:
                    cross = np.linalg.norm(np.cross(r[i], velocities[i]))
                total += cross / (r_norm * v_norm)
                count += 1

        if count == 0:
            return 0.0

        return float(total / count)
