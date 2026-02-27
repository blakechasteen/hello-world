"""
Semantic Flow Calculus - Core trajectory and flow analysis.

This module provides:
- SemanticState: Position in semantic space
- SemanticTrajectory: Path through semantic space
- SemanticFlowCalculus: Differential calculus on semantic space
- SemanticFlowVisualizer: Visualization utilities

Author: Claude Code
Date: 2026-01-22
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
import numpy as np
import warnings


@dataclass
class SemanticState:
    """
    State of a point in semantic space.

    Attributes:
        position: Embedding vector (384D or projected)
        velocity: Rate of change (optional)
        time: Timestamp
    """
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    time: float = 0.0

    def __post_init__(self):
        self.position = np.asarray(self.position)
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity)


@dataclass
class SemanticTrajectory:
    """
    Trajectory through semantic space - sequence of states over time.

    Attributes:
        positions: Array of positions (n_steps, dim)
        velocities: Array of velocities (n_steps, dim)
        accelerations: Array of accelerations (n_steps, dim)
        curvature: Curvature at each point (n_steps,)
        times: Timestamps
    """
    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    accelerations: Optional[np.ndarray] = None
    curvature: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None

    def __post_init__(self):
        self.positions = np.asarray(self.positions)
        n_steps = len(self.positions)

        if self.velocities is not None:
            self.velocities = np.asarray(self.velocities)
        if self.accelerations is not None:
            self.accelerations = np.asarray(self.accelerations)
        if self.curvature is not None:
            self.curvature = np.asarray(self.curvature)
        if self.times is None:
            self.times = np.arange(n_steps, dtype=float)

    @property
    def n_steps(self) -> int:
        """Number of steps in trajectory."""
        return len(self.positions)

    @property
    def dim(self) -> int:
        """Dimensionality of semantic space."""
        return self.positions.shape[1] if len(self.positions.shape) > 1 else len(self.positions)


class SemanticFlowCalculus:
    """
    Differential calculus on semantic space.

    Computes velocity, acceleration, curvature of semantic trajectories.

    Args:
        embed_fn: Function to embed text -> vector
        dt: Time step for derivative computation
        mass: Semantic mass parameter (for dynamics)
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        dt: float = 1.0,
        mass: float = 1.0,
    ):
        self.embed_fn = embed_fn
        self.dt = dt
        self.mass = mass

        if embed_fn is None:
            warnings.warn(
                "SemanticFlowCalculus created without embed_fn. "
                "Call set_embedder() before computing trajectories."
            )

    def set_embedder(self, embed_fn: Callable[[str], np.ndarray]):
        """Set the embedding function."""
        self.embed_fn = embed_fn

    def embed(self, text: str) -> np.ndarray:
        """Embed text to vector."""
        if self.embed_fn is None:
            raise RuntimeError("No embedder set. Call set_embedder() first.")
        return np.asarray(self.embed_fn(text))

    def compute_trajectory(
        self,
        words: List[str],
        include_derivatives: bool = True
    ) -> SemanticTrajectory:
        """
        Compute semantic trajectory from word sequence.

        Args:
            words: Sequence of words/tokens
            include_derivatives: Compute velocity, acceleration, curvature

        Returns:
            SemanticTrajectory with positions and optionally derivatives
        """
        if len(words) == 0:
            raise ValueError("Need at least one word")

        # Embed all words
        positions = np.array([self.embed(w) for w in words])

        velocities = None
        accelerations = None
        curvature = None

        if include_derivatives and len(words) > 1:
            # Compute velocity (finite differences)
            velocities = np.zeros_like(positions)
            velocities[1:] = (positions[1:] - positions[:-1]) / self.dt
            velocities[0] = velocities[1] if len(words) > 1 else 0

            # Compute acceleration
            if len(words) > 2:
                accelerations = np.zeros_like(positions)
                accelerations[1:] = (velocities[1:] - velocities[:-1]) / self.dt
                accelerations[0] = accelerations[1] if len(words) > 2 else 0

                # Compute curvature: |v x a| / |v|^3
                curvature = np.zeros(len(words))
                for i in range(len(words)):
                    v_norm = np.linalg.norm(velocities[i])
                    if v_norm > 1e-10:
                        # Cross product in high dimensions uses a simplified formula
                        cross_norm = np.linalg.norm(
                            velocities[i] * np.linalg.norm(accelerations[i]) -
                            accelerations[i] * (np.dot(velocities[i], accelerations[i]) / v_norm)
                        )
                        curvature[i] = cross_norm / (v_norm ** 3)

        return SemanticTrajectory(
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            curvature=curvature,
            times=np.arange(len(words)) * self.dt,
        )

    def compute_state(self, text: str, prev_state: Optional[SemanticState] = None) -> SemanticState:
        """
        Compute semantic state for a single piece of text.

        Args:
            text: Text to analyze
            prev_state: Previous state (for velocity computation)

        Returns:
            SemanticState with position and velocity
        """
        position = self.embed(text)

        velocity = None
        time = 0.0
        if prev_state is not None:
            velocity = (position - prev_state.position) / self.dt
            time = prev_state.time + self.dt

        return SemanticState(position=position, velocity=velocity, time=time)


class SemanticFlowVisualizer:
    """
    Visualization utilities for semantic trajectories.

    Note: Actual visualization requires matplotlib (optional dependency).
    """

    def __init__(self):
        self._plt = None
        try:
            import matplotlib.pyplot as plt
            self._plt = plt
        except ImportError:
            warnings.warn("matplotlib not available for visualization")

    def plot_trajectory_2d(
        self,
        trajectory: SemanticTrajectory,
        dims: tuple = (0, 1),
        title: str = "Semantic Trajectory",
        ax=None,
    ):
        """
        Plot 2D projection of trajectory.

        Args:
            trajectory: Trajectory to plot
            dims: Which dimensions to plot (default: first two)
            title: Plot title
            ax: Matplotlib axes (optional)
        """
        if self._plt is None:
            warnings.warn("matplotlib not available")
            return None

        if ax is None:
            fig, ax = self._plt.subplots(figsize=(10, 8))

        x = trajectory.positions[:, dims[0]]
        y = trajectory.positions[:, dims[1]]

        # Plot trajectory
        ax.plot(x, y, 'b-', alpha=0.7, linewidth=2)
        ax.scatter(x, y, c=trajectory.times, cmap='viridis', s=50)
        ax.scatter([x[0]], [y[0]], c='green', s=100, marker='o', label='Start')
        ax.scatter([x[-1]], [y[-1]], c='red', s=100, marker='s', label='End')

        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_title(title)
        ax.legend()

        return ax

    def plot_curvature(
        self,
        trajectory: SemanticTrajectory,
        title: str = "Semantic Curvature",
        ax=None,
    ):
        """Plot curvature over time."""
        if self._plt is None:
            warnings.warn("matplotlib not available")
            return None

        if trajectory.curvature is None:
            warnings.warn("No curvature data in trajectory")
            return None

        if ax is None:
            fig, ax = self._plt.subplots(figsize=(10, 4))

        ax.plot(trajectory.times, trajectory.curvature, 'r-', linewidth=2)
        ax.fill_between(trajectory.times, 0, trajectory.curvature, alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Curvature')
        ax.set_title(title)

        return ax


def analyze_text_flow(
    texts: List[str],
    embed_fn: Callable[[str], np.ndarray],
    dt: float = 1.0,
) -> Dict[str, Any]:
    """
    Convenience function for quick text flow analysis.

    Args:
        texts: List of texts to analyze
        embed_fn: Embedding function
        dt: Time step

    Returns:
        Dictionary with trajectory and metrics
    """
    calculus = SemanticFlowCalculus(embed_fn, dt=dt)
    trajectory = calculus.compute_trajectory(texts)

    # Compute summary metrics
    metrics = {
        'n_steps': trajectory.n_steps,
        'dim': trajectory.dim,
        'total_distance': 0.0,
        'avg_speed': 0.0,
        'max_curvature': 0.0,
    }

    if trajectory.velocities is not None:
        speeds = np.linalg.norm(trajectory.velocities, axis=1)
        metrics['avg_speed'] = float(np.mean(speeds))
        metrics['total_distance'] = float(np.sum(speeds) * dt)

    if trajectory.curvature is not None:
        metrics['max_curvature'] = float(np.max(trajectory.curvature))

    return {
        'trajectory': trajectory,
        'metrics': metrics,
    }
