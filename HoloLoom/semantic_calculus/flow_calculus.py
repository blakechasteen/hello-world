"""
Semantic Flow Calculus: Mathematical framework for analyzing semantic trajectories

This module implements the core mathematical machinery for:
- Computing semantic derivatives (velocity, acceleration, jerk)
- Hamiltonian dynamics in embedding space
- Multi-scale harmonic analysis
- Potential field reconstruction
- Attractor detection and symbolic structure extraction

Based on the insight that word embeddings + temporal derivatives reveal
the underlying "flow field" of meaning in language.

PERFORMANCE OPTIMIZATIONS:
- Batch embedding with LRU cache
- Vectorized derivative computations
- Lazy evaluation for expensive operations
- JIT compilation where available

Author: BearL Labs
License: MIT
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import performance utilities
from .performance import (
    EmbeddingCache,
    LazyArray,
    compute_finite_difference_vectorized,
    compute_curvature_vectorized,
)


@dataclass
class SemanticState:
    """
    Complete state of semantic flow at one point in time

    Attributes:
        position: embedding vector (q in Hamiltonian mechanics)
        velocity: first derivative dq/dt
        acceleration: second derivative d²q/dt²
        jerk: third derivative d³q/dt³ (optional)
        potential: V(q) - semantic potential energy
        kinetic: T = ½||v||² - kinetic energy
        word: the actual word at this position (if available)
    """
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: Optional[np.ndarray] = None
    potential: Optional[float] = None
    kinetic: Optional[float] = None
    word: Optional[str] = None

    @property
    def total_energy(self) -> Optional[float]:
        """Hamiltonian H = T + V"""
        if self.potential is not None and self.kinetic is not None:
            return self.kinetic + self.potential
        return None

    @property
    def speed(self) -> float:
        """Magnitude of velocity"""
        return np.linalg.norm(self.velocity)

    @property
    def acceleration_magnitude(self) -> float:
        """Magnitude of acceleration (force)"""
        return np.linalg.norm(self.acceleration)


@dataclass
class SemanticTrajectory:
    """
    Complete trajectory through semantic space

    Contains sequence of states plus derived quantities
    """
    states: List[SemanticState]
    words: List[str]
    dt: float = 1.0  # time step between words

    @property
    def positions(self) -> np.ndarray:
        """All positions as array"""
        return np.array([s.position for s in self.states])

    @property
    def velocities(self) -> np.ndarray:
        """All velocities as array"""
        return np.array([s.velocity for s in self.states])

    @property
    def accelerations(self) -> np.ndarray:
        """All accelerations as array"""
        return np.array([s.acceleration for s in self.states])

    def curvature(self, idx: int) -> float:
        """
        Compute curvature at point idx
        κ = ||v × a|| / ||v||³ (in 3D)
        Approximation in high-D: κ ≈ ||a_perp|| / ||v||²
        """
        v = self.states[idx].velocity
        a = self.states[idx].acceleration

        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            return 0.0

        # Project acceleration perpendicular to velocity
        v_hat = v / v_norm
        a_perp = a - np.dot(a, v_hat) * v_hat

        return np.linalg.norm(a_perp) / (v_norm ** 2)

    def total_distance(self) -> float:
        """Total arc length traveled"""
        distances = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        return np.sum(distances)


class SemanticFlowCalculus:
    """
    Core calculus engine for semantic flows

    Computes derivatives, energies, and flow properties from
    sequences of embeddings.

    PERFORMANCE FEATURES:
    - Automatic embedding caching (10K word LRU cache)
    - Batch embedding for efficiency
    - Vectorized derivative computations
    - Lazy evaluation of expensive properties
    """

    def __init__(self, embedding_fn: Callable, dt: float = 1.0,
                 enable_cache: bool = True, cache_size: int = 10000):
        """
        Args:
            embedding_fn: Function that maps word(s) -> embedding vector(s)
                         Should accept both single words and lists
            dt: Time step between words (default 1.0)
            enable_cache: Enable embedding cache (default True)
            cache_size: Maximum cached embeddings (default 10000)
        """
        # Setup embedding with optional caching
        if enable_cache:
            self._cache = EmbeddingCache(embedding_fn, max_size=cache_size)
            self.embed = self._cache.get
            self.embed_batch = self._cache.get_batch
        else:
            self._cache = None
            self.embed = lambda w: embedding_fn([w])[0]
            self.embed_batch = lambda words: embedding_fn(words)

        self.dt = dt

        # Hamiltonian parameters
        self.mass = 1.0  # semantic "mass"
        self.damping = 0.1  # friction coefficient
        self.stiffness = 0.01  # harmonic restoring force

    def compute_trajectory(self, words: List[str]) -> SemanticTrajectory:
        """
        Convert word sequence to complete semantic trajectory

        OPTIMIZATIONS:
        - Uses batch embedding (single call instead of N calls)
        - Vectorized kinetic energy computation
        - All derivatives computed in one pass

        Args:
            words: List of words

        Returns:
            SemanticTrajectory with all derivatives computed
        """
        # OPTIMIZATION: Batch embed all words at once
        positions = self.embed_batch(words)

        # OPTIMIZATION: Compute all derivatives using vectorized operations
        velocities = self._compute_velocity(positions)
        accelerations = self._compute_acceleration(velocities)
        jerks = self._compute_jerk(accelerations)

        # OPTIMIZATION: Vectorized kinetic energy computation
        kinetic_energies = 0.5 * self.mass * np.sum(velocities**2, axis=1)

        # Build states (still sequential but with pre-computed arrays)
        states = []
        for i in range(len(words)):
            state = SemanticState(
                position=positions[i],
                velocity=velocities[i],
                acceleration=accelerations[i],
                jerk=jerks[i] if i < len(jerks) else None,
                kinetic=kinetic_energies[i],
                word=words[i]
            )
            states.append(state)

        return SemanticTrajectory(states=states, words=words, dt=self.dt)

    def get_cache_stats(self) -> Optional[Dict]:
        """Get embedding cache statistics"""
        if self._cache is not None:
            return self._cache.get_stats()
        return None

    def clear_cache(self):
        """Clear embedding cache"""
        if self._cache is not None:
            self._cache.clear()

    def _compute_velocity(self, positions: np.ndarray) -> np.ndarray:
        """First derivative: dq/dt"""
        if len(positions) < 2:
            return np.zeros_like(positions)

        # Use central differences where possible, forward/backward at edges
        velocity = np.gradient(positions, self.dt, axis=0)
        return velocity

    def _compute_acceleration(self, velocities: np.ndarray) -> np.ndarray:
        """Second derivative: d²q/dt²"""
        if len(velocities) < 2:
            return np.zeros_like(velocities)

        acceleration = np.gradient(velocities, self.dt, axis=0)
        return acceleration

    def _compute_jerk(self, accelerations: np.ndarray) -> np.ndarray:
        """Third derivative: d³q/dt³"""
        if len(accelerations) < 2:
            return np.zeros_like(accelerations)

        jerk = np.gradient(accelerations, self.dt, axis=0)
        return jerk

    def infer_potential_field(self, trajectories: List[SemanticTrajectory]) -> Callable:
        """
        Reconstruct semantic potential V(q) from observed trajectories

        Using F = ma and F = -∇V, we can reconstruct V from observed
        accelerations along trajectories.

        Args:
            trajectories: List of observed semantic trajectories

        Returns:
            Function V(q) that estimates potential at any point
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        # Collect training data: (position, force) pairs
        positions = []
        forces = []

        for traj in trajectories:
            for state in traj.states:
                positions.append(state.position)
                # F = ma (force from acceleration)
                forces.append(self.mass * state.acceleration)

        positions = np.array(positions)
        forces = np.array(forces)

        # Integrate forces to get potentials
        # V ≈ -∫ F·dq (simplified: use cumulative path integral)
        potentials = []
        for i in range(len(positions)):
            # Approximate integral by summing force contributions
            # (simplified - assumes origin has V=0)
            V = -np.dot(forces[i], positions[i])
            potentials.append(V)

        potentials = np.array(potentials).reshape(-1, 1)

        # Fit Gaussian Process to learn V(q)
        kernel = ConstantKernel(1.0) * RBF(length_scale=10.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
        gp.fit(positions, potentials)

        # Return function that predicts V at any point
        def potential_function(q: np.ndarray) -> float:
            """Estimate potential energy at position q"""
            if q.ndim == 1:
                q = q.reshape(1, -1)
            return gp.predict(q)[0, 0]

        return potential_function

    def find_attractors(self, trajectories: List[SemanticTrajectory],
                       velocity_threshold: float = 0.1) -> List[Dict]:
        """
        Find attractor basins (stable semantic concepts)

        Attractors are regions where velocity → 0 (fixed points)

        Args:
            trajectories: Observed trajectories
            velocity_threshold: Max velocity magnitude to consider fixed point

        Returns:
            List of attractors with centroids and nearby words
        """
        # Collect all low-velocity points
        fixed_points = []
        fixed_point_words = []

        for traj in trajectories:
            for state in traj.states:
                if state.speed < velocity_threshold:
                    fixed_points.append(state.position)
                    fixed_point_words.append(state.word)

        if len(fixed_points) < 2:
            return []

        fixed_points = np.array(fixed_points)

        # Cluster fixed points to find attractors
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(fixed_points)

        attractors = []
        for label in set(clustering.labels_):
            if label == -1:  # noise
                continue

            # Get points in this cluster
            mask = clustering.labels_ == label
            cluster_points = fixed_points[mask]
            cluster_words = [w for w, m in zip(fixed_point_words, mask) if m]

            # Compute centroid
            centroid = np.mean(cluster_points, axis=0)

            attractors.append({
                'centroid': centroid,
                'words': list(set(cluster_words)),  # unique words
                'basin_size': len(cluster_points),
                'label': label
            })

        return attractors


class SemanticFlowVisualizer:
    """
    Visualization tools for semantic flows
    """

    def __init__(self, dim_reduction: str = 'pca'):
        """
        Args:
            dim_reduction: 'pca' or 'tsne' for dimensionality reduction
        """
        self.dim_reduction = dim_reduction
        self.reducer = None

    def visualize_trajectory_3d(self, trajectory: SemanticTrajectory,
                                show_velocity: bool = True,
                                show_acceleration: bool = False,
                                attractors: Optional[List[Dict]] = None):
        """
        Visualize semantic trajectory in 3D

        Args:
            trajectory: The trajectory to plot
            show_velocity: Draw velocity vectors
            show_acceleration: Draw acceleration vectors
            attractors: Optional list of attractors to mark
        """
        # Reduce to 3D
        positions_3d = self._reduce_dimensions(trajectory.positions, n_components=3)

        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
                'b-', linewidth=2, alpha=0.7, label='Trajectory')

        # Mark start and end
        ax.scatter(*positions_3d[0], color='green', s=200, marker='o',
                   label='Start', zorder=5)
        ax.scatter(*positions_3d[-1], color='red', s=200, marker='X',
                   label='End', zorder=5)

        # Add velocity vectors
        if show_velocity:
            vel_3d = self._reduce_dimensions(
                trajectory.positions + trajectory.velocities,
                n_components=3
            ) - positions_3d

            # Sample every few points to avoid clutter
            step = max(1, len(positions_3d) // 20)
            ax.quiver(positions_3d[::step, 0],
                     positions_3d[::step, 1],
                     positions_3d[::step, 2],
                     vel_3d[::step, 0],
                     vel_3d[::step, 1],
                     vel_3d[::step, 2],
                     length=0.3, normalize=True, color='orange',
                     alpha=0.6, label='Velocity')

        # Add acceleration vectors
        if show_acceleration:
            acc_3d = self._reduce_dimensions(
                trajectory.positions + trajectory.accelerations,
                n_components=3
            ) - positions_3d

            step = max(1, len(positions_3d) // 20)
            ax.quiver(positions_3d[::step, 0],
                     positions_3d[::step, 1],
                     positions_3d[::step, 2],
                     acc_3d[::step, 0],
                     acc_3d[::step, 1],
                     acc_3d[::step, 2],
                     length=0.2, normalize=True, color='red',
                     alpha=0.5, label='Acceleration')

        # Mark attractors
        if attractors:
            for i, attr in enumerate(attractors):
                centroid_3d = self._reduce_dimensions(
                    attr['centroid'].reshape(1, -1),
                    n_components=3
                )[0]
                ax.scatter(*centroid_3d, color='purple', s=300,
                          marker='*', zorder=10)
                ax.text(*centroid_3d, f"  {attr['words'][0]}",
                       fontsize=10, color='purple')

        # Labels and styling
        ax.set_xlabel('PC1' if self.dim_reduction == 'pca' else 'Dim1')
        ax.set_ylabel('PC2' if self.dim_reduction == 'pca' else 'Dim2')
        ax.set_zlabel('PC3' if self.dim_reduction == 'pca' else 'Dim3')
        ax.set_title('Semantic Flow Trajectory', fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        return fig, ax

    def plot_flow_metrics(self, trajectory: SemanticTrajectory):
        """
        Plot velocity, acceleration, curvature over time
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Velocity magnitude
        speeds = [s.speed for s in trajectory.states]
        axes[0].plot(speeds, 'b-', linewidth=2)
        axes[0].set_ylabel('Speed ||v||', fontsize=12)
        axes[0].set_title('Semantic Flow Metrics', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Acceleration magnitude
        acc_mags = [s.acceleration_magnitude for s in trajectory.states]
        axes[1].plot(acc_mags, 'r-', linewidth=2)
        axes[1].set_ylabel('Acceleration ||a||', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        # Curvature
        curvatures = [trajectory.curvature(i) for i in range(len(trajectory.states))]
        axes[2].plot(curvatures, 'g-', linewidth=2)
        axes[2].set_ylabel('Curvature κ', fontsize=12)
        axes[2].set_xlabel('Word Index', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        # Add word labels on x-axis
        for ax in axes:
            if len(trajectory.words) <= 20:
                ax.set_xticks(range(len(trajectory.words)))
                ax.set_xticklabels(trajectory.words, rotation=45, ha='right')

        plt.tight_layout()
        return fig, axes

    def _reduce_dimensions(self, data: np.ndarray, n_components: int = 3) -> np.ndarray:
        """Reduce high-dimensional data for visualization"""
        if data.shape[1] <= n_components:
            return data

        if self.reducer is None:
            if self.dim_reduction == 'pca':
                self.reducer = PCA(n_components=n_components)
                return self.reducer.fit_transform(data)
            elif self.dim_reduction == 'tsne':
                from sklearn.manifold import TSNE
                self.reducer = TSNE(n_components=n_components, random_state=42)
                return self.reducer.fit_transform(data)
        else:
            # Use existing reducer
            return self.reducer.transform(data)


# Convenience function for quick analysis
def analyze_text_flow(words: List[str], embedding_fn: Callable) -> Tuple[SemanticTrajectory, Dict]:
    """
    Quick analysis of semantic flow for a text sequence

    Args:
        words: List of words to analyze
        embedding_fn: Function mapping word -> embedding

    Returns:
        trajectory: Complete semantic trajectory
        analysis: Dictionary of metrics and insights
    """
    calculus = SemanticFlowCalculus(embedding_fn)
    trajectory = calculus.compute_trajectory(words)

    analysis = {
        'total_distance': trajectory.total_distance(),
        'avg_speed': np.mean([s.speed for s in trajectory.states]),
        'max_speed': np.max([s.speed for s in trajectory.states]),
        'avg_acceleration': np.mean([s.acceleration_magnitude for s in trajectory.states]),
        'max_curvature': np.max([trajectory.curvature(i) for i in range(len(trajectory.states))]),
        'total_energy_change': (trajectory.states[-1].total_energy - trajectory.states[0].total_energy)
                               if trajectory.states[0].total_energy is not None else None
    }

    return trajectory, analysis