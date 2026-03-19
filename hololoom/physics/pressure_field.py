"""
Pressure Field - Importance Density Map

Maps context importance to pressure values across the knowledge graph.
High-pressure regions (important context) naturally flow to low-pressure regions (sparse context).

Physics Model:
    - Pressure P(node) represents importance density
    - High P = important/relevant context
    - Low P = sparse/missing context
    - Pressure gradients drive information flow
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PressureSource:
    """A source of pressure (context importance)."""
    node: str
    importance: float  # 0.0 - 1.0
    tokens: int = 0
    text: str | None = None


class PressureField:
    """
    Importance density map for context.

    High pressure = important/relevant context
    Low pressure = sparse/missing context

    Usage:
        field = PressureField()
        field.set_pressure("ThompsonSampling", 0.95)  # High importance
        field.set_pressure("Bayesian", 0.30)           # Low importance

        # Compute gradient (flow direction)
        gradient = field.compute_gradient("ThompsonSampling", neighbors)
    """

    def __init__(self, default_pressure: float = 0.0):
        """
        Initialize pressure field.

        Args:
            default_pressure: Default pressure for new nodes (0.0 = sparse)
        """
        self.pressures: dict[str, float] = {}
        self.sources: dict[str, PressureSource] = {}
        self.default_pressure = default_pressure

    def set_pressure(
        self,
        node: str,
        pressure: float,
        source: PressureSource | None = None
    ):
        """
        Set pressure at a node.

        Args:
            node: Node identifier
            pressure: Pressure value (0.0 - 1.0)
            source: Optional pressure source metadata
        """
        self.pressures[node] = np.clip(pressure, 0.0, 1.0)
        if source:
            self.sources[node] = source

    def get_pressure(self, node: str) -> float:
        """Get pressure at a node (defaults to default_pressure if not set)."""
        return self.pressures.get(node, self.default_pressure)

    def inject_pressure(
        self,
        node: str,
        importance: float,
        tokens: int = 0,
        text: str | None = None
    ):
        """
        Inject pressure at a node (like injecting water into a system).

        Args:
            node: Target node
            importance: Importance value (0.0 - 1.0)
            tokens: Token count for this context
            text: Optional context text
        """
        source = PressureSource(
            node=node,
            importance=importance,
            tokens=tokens,
            text=text
        )
        self.set_pressure(node, importance, source)

    def compute_gradient(
        self,
        node: str,
        neighbors: list[str]
    ) -> dict[str, float]:
        """
        Compute pressure gradient at a node.

        Gradient points from high pressure to low pressure.
        Positive gradient = neighbor has higher pressure (flow toward it)
        Negative gradient = neighbor has lower pressure (flow away from it)

        Args:
            node: Source node
            neighbors: Neighbor nodes

        Returns:
            Dict mapping neighbor -> gradient value
        """
        node_pressure = self.get_pressure(node)
        gradients = {}

        for neighbor in neighbors:
            neighbor_pressure = self.get_pressure(neighbor)
            # Gradient = (neighbor_pressure - node_pressure)
            # Positive = flow toward neighbor
            # Negative = flow away from neighbor
            gradients[neighbor] = neighbor_pressure - node_pressure

        return gradients

    def detect_sparse_regions(
        self,
        threshold: float = 0.3,
        nodes: set[str] | None = None
    ) -> list[str]:
        """
        Find low-pressure (sparse context) regions.

        Args:
            threshold: Pressure below this is considered sparse
            nodes: Optional set of nodes to check (defaults to all)

        Returns:
            List of sparse nodes
        """
        check_nodes = nodes if nodes else set(self.pressures.keys())
        sparse = [
            node for node in check_nodes
            if self.get_pressure(node) < threshold
        ]
        return sparse

    def get_high_pressure_regions(
        self,
        threshold: float = 0.7,
        nodes: set[str] | None = None
    ) -> list[str]:
        """
        Find high-pressure (important context) regions.

        Args:
            threshold: Pressure above this is considered high
            nodes: Optional set of nodes to check

        Returns:
            List of high-pressure nodes
        """
        check_nodes = nodes if nodes else set(self.pressures.keys())
        high = [
            node for node in check_nodes
            if self.get_pressure(node) > threshold
        ]
        return high

    def diffuse(
        self,
        node: str,
        neighbors: list[str],
        diffusion_rate: float = 0.1
    ):
        """
        Diffuse pressure to neighbors (like heat spreading).

        Args:
            node: Source node
            neighbors: Neighbor nodes
            diffusion_rate: How fast pressure spreads (0.0 - 1.0)
        """
        node_pressure = self.get_pressure(node)

        for neighbor in neighbors:
            neighbor_pressure = self.get_pressure(neighbor)

            # Diffusion: average pressure weighted by rate
            delta = diffusion_rate * (node_pressure - neighbor_pressure)

            # Update neighbor pressure
            new_pressure = neighbor_pressure + delta
            self.set_pressure(neighbor, new_pressure)

    def get_statistics(self) -> dict[str, float]:
        """Get pressure field statistics."""
        if not self.pressures:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sparse_count": 0,
                "high_count": 0
            }

        values = list(self.pressures.values())
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "sparse_count": len(self.detect_sparse_regions()),
            "high_count": len(self.get_high_pressure_regions())
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"PressureField(nodes={len(self.pressures)}, "
            f"mean={stats['mean']:.2f}, "
            f"sparse={stats['sparse_count']}, "
            f"high={stats['high_count']})"
        )
