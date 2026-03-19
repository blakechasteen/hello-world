"""
Velocity Field - Information Flow Vectors

Tracks the velocity of information flow through the knowledge graph.
Velocity is driven by pressure gradients (high → low).

Physics Model:
    - Velocity v(node) represents flow rate and direction
    - Driven by pressure gradients: ∇p
    - Damped by viscosity: ν∇²v
    - Advection: (v·∇)v (velocity influences itself)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FlowVector:
    """A single flow vector."""
    source: str
    target: str
    magnitude: float  # Flow strength
    direction: np.ndarray = None  # Flow direction (optional)

    def __post_init__(self):
        if self.direction is None:
            # Default direction (unit vector)
            self.direction = np.array([1.0, 0.0])


class VelocityField:
    """
    Information flow vectors across the knowledge graph.

    Velocity is driven by pressure gradients:
        - High pressure → low pressure = positive velocity
        - Flow accelerates where gradients are strong

    Usage:
        field = VelocityField()
        field.set_velocity("ThompsonSampling", "Bayesian", magnitude=0.5)

        # Update based on pressure gradient
        field.update_from_pressure(pressure_field, dt=0.01)
    """

    def __init__(self, viscosity: float = 0.01):
        """
        Initialize velocity field.

        Args:
            viscosity: Damping coefficient (0.0 = no damping, 1.0 = high damping)
        """
        self.velocities: dict[tuple[str, str], FlowVector] = {}
        self.viscosity = viscosity

    def set_velocity(
        self,
        source: str,
        target: str,
        magnitude: float,
        direction: np.ndarray = None
    ):
        """
        Set velocity between two nodes.

        Args:
            source: Source node
            target: Target node
            magnitude: Flow strength
            direction: Optional flow direction vector
        """
        key = (source, target)
        self.velocities[key] = FlowVector(
            source=source,
            target=target,
            magnitude=magnitude,
            direction=direction
        )

    def get_velocity(self, source: str, target: str) -> float:
        """Get velocity magnitude between two nodes."""
        key = (source, target)
        if key in self.velocities:
            return self.velocities[key].magnitude
        return 0.0

    def update_from_pressure_gradient(
        self,
        source: str,
        target: str,
        gradient: float,
        dt: float = 0.01
    ):
        """
        Update velocity based on pressure gradient.

        Navier-Stokes: dv/dt = -∇p + ν∇²v

        Simplified (1D):
            v_new = v_old - gradient * dt + viscosity_term

        Args:
            source: Source node
            target: Target node
            gradient: Pressure gradient (target_p - source_p)
            dt: Timestep
        """
        key = (source, target)
        current_v = self.get_velocity(source, target)

        # Pressure gradient term: -∇p
        # Negative because flow goes from high to low pressure
        pressure_term = -gradient

        # Viscosity term: ν∇²v (damping)
        # Simplified: just damp current velocity
        viscosity_term = -self.viscosity * current_v

        # Update: v_new = v_old + (pressure_term + viscosity_term) * dt
        new_v = current_v + (pressure_term + viscosity_term) * dt

        self.set_velocity(source, target, new_v)

    def advect(
        self,
        source: str,
        neighbors: list[str],
        dt: float = 0.01
    ):
        """
        Advection: velocity influences itself (v·∇)v.

        Simplified: velocity at a node is influenced by velocities from neighbors.

        Args:
            source: Source node
            neighbors: Neighbor nodes
            dt: Timestep
        """
        for target in neighbors:
            key = (source, target)
            if key not in self.velocities:
                continue

            # Get current velocity
            current_v = self.velocities[key].magnitude

            # Advection term: velocity transports itself
            # Simplified: average of neighbor velocities
            neighbor_velocities = [
                self.get_velocity(target, n)
                for n in neighbors
                if n != source
            ]

            if neighbor_velocities:
                advection_term = np.mean(neighbor_velocities) * current_v
                new_v = current_v + advection_term * dt
                self.set_velocity(source, target, new_v)

    def get_outflow(self, node: str) -> float:
        """
        Get total outflow from a node.

        Returns:
            Sum of outgoing velocities
        """
        outflow = 0.0
        for (src, tgt), flow in self.velocities.items():
            if src == node:
                outflow += flow.magnitude
        return outflow

    def get_inflow(self, node: str) -> float:
        """
        Get total inflow to a node.

        Returns:
            Sum of incoming velocities
        """
        inflow = 0.0
        for (src, tgt), flow in self.velocities.items():
            if tgt == node:
                inflow += flow.magnitude
        return inflow

    def get_net_flow(self, node: str) -> float:
        """
        Get net flow at a node (inflow - outflow).

        Positive = more flowing in (accumulation)
        Negative = more flowing out (depletion)
        """
        return self.get_inflow(node) - self.get_outflow(node)

    def get_flow_vectors(self, node: str) -> list[FlowVector]:
        """Get all flow vectors originating from a node."""
        return [
            flow for (src, tgt), flow in self.velocities.items()
            if src == node
        ]

    def get_statistics(self) -> dict[str, float]:
        """Get velocity field statistics."""
        if not self.velocities:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "total_flow": 0.0
            }

        magnitudes = [flow.magnitude for flow in self.velocities.values()]
        return {
            "mean": np.mean(magnitudes),
            "std": np.std(magnitudes),
            "min": np.min(magnitudes),
            "max": np.max(magnitudes),
            "total_flow": np.sum(magnitudes)
        }

    def decay(self, decay_rate: float = 0.95):
        """
        Apply exponential decay to all velocities.

        Simulates natural damping over time.

        Args:
            decay_rate: Decay factor (0.95 = 5% decay per step)
        """
        for key, flow in self.velocities.items():
            flow.magnitude *= decay_rate

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"VelocityField(flows={len(self.velocities)}, "
            f"mean={stats['mean']:.3f}, "
            f"total={stats['total_flow']:.2f})"
        )
