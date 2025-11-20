"""
Activation Field - Dynamic activation spreading through awareness.

Key insight: Activation is a PROCESS (field that spreads/decays),
not a PROPERTY (static value stored per node).
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import networkx as nx
from collections import defaultdict


class ActivationField:
    """
    Dynamic field managing activation spreading through semantic space.

    Like a wave propagating through the graph:
    1. Initial activation at query point
    2. Spreads through connections (resonance edges)
    3. Decays over time
    4. Can be cleared/reset
    """

    def __init__(self):
        # Current activation levels (node_id → activation)
        self.levels: Dict[str, float] = {}

        # Previous levels (for decay calculation)
        self.prev_levels: Dict[str, float] = {}

        # Spatial index (position → node_ids for fast lookup)
        self.spatial_index: Dict[str, np.ndarray] = {}

    def activate_region(
        self,
        center: np.ndarray,
        radius: float,
        node_ids: List[str]
    ) -> Set[str]:
        """
        Activate nodes within semantic radius of center point.

        Args:
            center: Query position in 244D semantic space
            radius: Activation radius (0-1)
            node_ids: Candidate node IDs to check

        Returns:
            Set of activated node IDs
        """
        activated = set()

        for node_id in node_ids:
            if node_id not in self.spatial_index:
                continue

            # Get node position
            node_pos = self.spatial_index[node_id]

            # Compute semantic distance
            distance = float(np.linalg.norm(center - node_pos))

            # Activate if within radius (with decay)
            if distance < radius:
                # Activation decays linearly with distance
                activation = 1.0 - (distance / radius)
                self.levels[node_id] = activation
                activated.add(node_id)

        return activated

    def spread_via_graph(
        self,
        graph: nx.MultiDiGraph,
        iterations: int = 2,
        decay_factor: float = 0.5
    ):
        """
        Spread activation through graph connections.

        Activation flows through edges, decaying at each hop.
        Models semantic resonance spreading through connected concepts.

        Args:
            graph: NetworkX graph with nodes and edges
            iterations: Number of spreading iterations (hops)
            decay_factor: How much activation decays per hop (0-1)
        """
        for iteration in range(iterations):
            # Snapshot current levels
            current_levels = self.levels.copy()

            # Spread from each active node
            for node_id, activation in current_levels.items():
                if activation < 0.1:  # Skip weak activations
                    continue

                # Get neighbors
                try:
                    neighbors = list(graph.successors(node_id))
                except:
                    continue

                # Spread to each neighbor
                for neighbor_id in neighbors:
                    # Get edge data for resonance strength
                    edge_data = graph.get_edge_data(node_id, neighbor_id, default={})

                    # Extract resonance strength (defaults to 0.5)
                    if isinstance(edge_data, dict):
                        # Single edge
                        strength = edge_data.get('strength', 0.5)
                    else:
                        # Multiple edges (take max strength)
                        strength = max(
                            (e.get('strength', 0.5) for e in edge_data.values()),
                            default=0.5
                        )

                    # Compute spread activation
                    spread_activation = activation * strength * decay_factor

                    # Update neighbor (accumulate, don't overwrite)
                    if spread_activation > 0.05:  # Threshold
                        current_activation = self.levels.get(neighbor_id, 0.0)
                        self.levels[neighbor_id] = max(
                            current_activation,
                            spread_activation
                        )

    def above_threshold(self, threshold: float = 0.3) -> List[str]:
        """
        Get node IDs with activation above threshold.

        Returns: List sorted by activation (highest first)
        """
        filtered = [
            (node_id, activation)
            for node_id, activation in self.levels.items()
            if activation >= threshold
        ]

        # Sort by activation level (descending)
        filtered.sort(key=lambda x: x[1], reverse=True)

        return [node_id for node_id, _ in filtered]

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top K most activated nodes.

        Returns: List of (node_id, activation) tuples
        """
        sorted_items = sorted(
            self.levels.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_items[:k]

    def decay(self, rate: float = 0.5):
        """
        Decay all activations by rate.

        Models forgetting - activation naturally decays over time.
        """
        self.prev_levels = self.levels.copy()

        for node_id in self.levels:
            self.levels[node_id] *= (1.0 - rate)

        # Remove very weak activations
        self.levels = {
            node_id: level
            for node_id, level in self.levels.items()
            if level > 0.01
        }

    def clear(self):
        """Reset all activations."""
        self.prev_levels = self.levels.copy()
        self.levels.clear()

    def update_spatial_index(self, node_id: str, position: np.ndarray):
        """
        Update spatial index with node position.

        Args:
            node_id: Node identifier
            position: 244D semantic position
        """
        self.spatial_index[node_id] = position

    def remove_from_index(self, node_id: str):
        """Remove node from spatial index."""
        self.spatial_index.pop(node_id, None)
        self.levels.pop(node_id, None)

    # =========================================================================
    # Metrics
    # =========================================================================

    def n_active(self, threshold: float = 0.1) -> int:
        """Count nodes above activation threshold."""
        return sum(1 for level in self.levels.values() if level >= threshold)

    def density(self) -> float:
        """
        Activation density.

        Returns: n_active / n_indexed
        """
        if not self.spatial_index:
            return 0.0

        n_active = self.n_active()
        n_total = len(self.spatial_index)

        return n_active / n_total

    def total_activation(self) -> float:
        """Sum of all activation levels."""
        return sum(self.levels.values())

    def get_stats(self) -> Dict:
        """Get activation field statistics."""
        active_levels = [v for v in self.levels.values() if v > 0.1]

        return {
            'n_active': len(active_levels),
            'n_indexed': len(self.spatial_index),
            'density': self.density(),
            'total_activation': self.total_activation(),
            'mean_activation': np.mean(active_levels) if active_levels else 0.0,
            'max_activation': max(active_levels) if active_levels else 0.0
        }