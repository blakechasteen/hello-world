"""
Poincare Sections — Phase Space Visualization for Dynamical Systems
===================================================================

Reduce continuous dynamical systems to discrete maps by recording
intersections with a hyperplane (the Poincare section).

Core Concepts:
- Poincare Section: Detect trajectory crossings of a hyperplane
- Poincare Map: Iterate the return map on the section
- Winding Number: Topological invariant counting rotations

Mathematical Foundation:
Section Σ = {x : n·(x - p) = 0} with crossing condition n·f(x) > 0
Return map P: Σ → Σ where P(x) is the next crossing point
Winding number: w = (1/2π) ∮ dθ

Applications to Warp Space:
- Detecting periodic orbits in scoring term evolution
- Phase portraits of Thompson Sampling convergence
- Topological classification of embedding trajectories

Author: Claude Code
Date: March 2026 (Math Module Expansion)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SectionResult:
    """Result of Poincare section computation.

    Attributes:
        crossings: Points where trajectory crosses the section (n_crossings, n_dims)
        crossing_times: Times of crossings (n_crossings,)
        crossing_indices: Indices into original trajectory (n_crossings,)
        n_crossings: Number of detected crossings
    """
    crossings: np.ndarray
    crossing_times: np.ndarray
    crossing_indices: np.ndarray
    n_crossings: int = 0


@dataclass
class ReturnMapResult:
    """Result of Poincare return map iteration.

    Attributes:
        crossings: Section crossing points (n_crossings, n_dims)
        return_times: Time between successive crossings (n_crossings - 1,)
        trajectory: Full trajectory between crossings
        trajectory_times: Time points for full trajectory
    """
    crossings: np.ndarray
    return_times: np.ndarray
    trajectory: np.ndarray
    trajectory_times: np.ndarray


# ============================================================================
# POINCARE SECTION
# ============================================================================

class PoincareSection:
    """Compute Poincare section crossings from a trajectory.

    Given a trajectory and a hyperplane defined by (normal, point),
    detect all crossings where the trajectory passes through the plane
    in a specified direction.

    Example:
        >>> # Circular trajectory in 2D
        >>> t = np.linspace(0, 10*np.pi, 10000)
        >>> traj = np.column_stack([np.cos(t), np.sin(t)])
        >>> ps = PoincareSection()
        >>> result = ps.compute(traj, np.array([0, 1]), np.array([0, 0]),
        ...                     times=t)
        >>> result.n_crossings  # ~5 crossings through y=0 going upward
    """

    def compute(
        self,
        trajectory: np.ndarray,
        plane_normal: np.ndarray,
        plane_point: np.ndarray,
        times: np.ndarray | None = None,
        direction: str = "positive",
    ) -> SectionResult:
        """Find trajectory crossings of a hyperplane.

        Args:
            trajectory: State trajectory (n_steps, n_dims)
            plane_normal: Normal vector to the section plane (n_dims,)
            plane_point: A point on the section plane (n_dims,)
            times: Time values for each trajectory point
            direction: "positive" (n·v > 0), "negative", or "both"

        Returns:
            SectionResult with crossing points and times
        """
        trajectory = np.asarray(trajectory, dtype=float)
        plane_normal = np.asarray(plane_normal, dtype=float)
        plane_point = np.asarray(plane_point, dtype=float)

        # Normalize the normal vector
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        n_steps = len(trajectory)
        if times is None:
            times = np.arange(n_steps, dtype=float)

        # Signed distances from the plane
        displacements = trajectory - plane_point
        signed_distances = displacements @ plane_normal

        crossings = []
        crossing_times = []
        crossing_indices = []

        for i in range(n_steps - 1):
            d0 = signed_distances[i]
            d1 = signed_distances[i + 1]

            # Check for sign change
            if d0 * d1 < 0:
                # Direction check
                if direction == "positive" and d0 >= 0:
                    continue
                if direction == "negative" and d0 <= 0:
                    continue

                # Linear interpolation for crossing point
                alpha = d0 / (d0 - d1)
                crossing = trajectory[i] + alpha * (trajectory[i + 1] - trajectory[i])
                t_cross = times[i] + alpha * (times[i + 1] - times[i])

                crossings.append(crossing)
                crossing_times.append(t_cross)
                crossing_indices.append(i)

        if len(crossings) == 0:
            return SectionResult(
                crossings=np.empty((0, trajectory.shape[1])),
                crossing_times=np.empty(0),
                crossing_indices=np.empty(0, dtype=int),
                n_crossings=0,
            )

        return SectionResult(
            crossings=np.array(crossings),
            crossing_times=np.array(crossing_times),
            crossing_indices=np.array(crossing_indices, dtype=int),
            n_crossings=len(crossings),
        )


# ============================================================================
# POINCARE MAP (Return Map)
# ============================================================================

class PoincareMap:
    """Iterate the Poincare return map.

    Given a dynamical system f and a section plane, integrate from a
    starting point until n_crossings are accumulated.

    Example:
        >>> def pendulum(t, y):
        ...     return np.array([y[1], -np.sin(y[0])])
        >>> pm = PoincareMap(dt=0.001)
        >>> result = pm.iterate(pendulum, np.array([0.5, 1.0]),
        ...                     (np.array([1, 0]), np.array([0, 0])),
        ...                     n_crossings=20)
    """

    def __init__(self, dt: float = 0.001, max_steps: int = 1000000):
        self.dt = dt
        self.max_steps = max_steps

    def iterate(
        self,
        f: Callable[[float, np.ndarray], np.ndarray],
        x0: np.ndarray,
        plane: tuple[np.ndarray, np.ndarray],
        n_crossings: int = 100,
        direction: str = "positive",
    ) -> ReturnMapResult:
        """Iterate the Poincare return map.

        Args:
            f: Dynamics function f(t, y) -> dy/dt
            x0: Initial state (n_dims,)
            plane: (normal, point) defining the section
            n_crossings: Number of crossings to collect
            direction: Crossing direction filter

        Returns:
            ReturnMapResult with crossing points and return times
        """
        x0 = np.atleast_1d(np.asarray(x0, dtype=float))
        plane_normal = np.asarray(plane[0], dtype=float)
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        plane_point = np.asarray(plane[1], dtype=float)

        trajectory_list = [x0.copy()]
        time_list = [0.0]

        crossings = []
        crossing_times_list = []

        y = x0.copy()
        t = 0.0
        d_prev = np.dot(y - plane_point, plane_normal)

        for step in range(self.max_steps):
            # RK4 step
            k1 = f(t, y)
            k2 = f(t + 0.5 * self.dt, y + 0.5 * self.dt * k1)
            k3 = f(t + 0.5 * self.dt, y + 0.5 * self.dt * k2)
            k4 = f(t + self.dt, y + self.dt * k3)
            y_new = y + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t_new = t + self.dt

            d_new = np.dot(y_new - plane_point, plane_normal)

            # Check crossing
            if d_prev * d_new < 0:
                cross_ok = True
                if direction == "positive" and d_prev >= 0:
                    cross_ok = False
                if direction == "negative" and d_prev <= 0:
                    cross_ok = False

                if cross_ok:
                    alpha = d_prev / (d_prev - d_new)
                    crossing = y + alpha * (y_new - y)
                    t_cross = t + alpha * self.dt
                    crossings.append(crossing)
                    crossing_times_list.append(t_cross)

                    if len(crossings) >= n_crossings:
                        trajectory_list.append(y_new)
                        time_list.append(t_new)
                        break

            trajectory_list.append(y_new)
            time_list.append(t_new)
            y = y_new
            t = t_new
            d_prev = d_new

        # Compute return times
        ct = np.array(crossing_times_list) if crossing_times_list else np.empty(0)
        return_times = np.diff(ct) if len(ct) > 1 else np.empty(0)

        return ReturnMapResult(
            crossings=np.array(crossings) if crossings else np.empty((0, len(x0))),
            return_times=return_times,
            trajectory=np.array(trajectory_list),
            trajectory_times=np.array(time_list),
        )


# ============================================================================
# WINDING NUMBER
# ============================================================================

def winding_number(trajectory_2d: np.ndarray, center: np.ndarray | None = None) -> int:
    """Compute the winding number of a 2D trajectory around a point.

    The winding number counts how many times the trajectory wraps
    around the center point (counterclockwise = positive).

    Args:
        trajectory_2d: 2D trajectory (n_steps, 2)
        center: Center point (default: origin)

    Returns:
        Integer winding number
    """
    trajectory_2d = np.asarray(trajectory_2d, dtype=float)
    if trajectory_2d.shape[1] != 2:
        raise ValueError("Trajectory must be 2D (n_steps, 2)")

    if center is not None:
        trajectory_2d = trajectory_2d - np.asarray(center, dtype=float)

    # Compute angles
    angles = np.arctan2(trajectory_2d[:, 1], trajectory_2d[:, 0])

    # Compute angle differences, handling branch cuts
    dtheta = np.diff(angles)
    dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi  # wrap to [-pi, pi]

    total_angle = np.sum(dtheta)
    return int(np.round(total_angle / (2 * np.pi)))
