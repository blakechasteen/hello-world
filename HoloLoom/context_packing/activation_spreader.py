"""
Beta Wave Activation Spreading
===============================

Physics-based activation propagation across knowledge graphs.

Beta waves (12-30 Hz in neuroscience) represent focused, alert attention.
We use beta wave dynamics to model how activation spreads from query-relevant
nodes to related concepts in the knowledge graph.

Key Properties:
- Exponential decay per hop (modeling energy dissipation)
- Frequency-dependent propagation (higher freq = faster spread)
- Multi-source activation (multiple query concepts activate simultaneously)
- Directional spreading (follows edge semantics)

Author: Claude Code
Date: 2025-11-22
"""

import math
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
import logging

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .protocol import BetaWave, ActivationMap, ActivationState
from .config import BetaWaveConfig

logger = logging.getLogger(__name__)


class ActivationSpreader:
    """
    Beta wave activation spreading engine.

    Spreads activation from source nodes across knowledge graph using
    physics-based wave propagation dynamics.
    """

    def __init__(self, config: Optional[BetaWaveConfig] = None):
        """
        Initialize activation spreader.

        Args:
            config: Beta wave configuration (uses defaults if None)
        """
        self.config = config or BetaWaveConfig()
        self.config.validate()

        # Tracking
        self.activation_history: List[Dict[str, float]] = []
        self._stats = {
            'total_spreads': 0,
            'avg_hops': 0.0,
            'avg_activated_nodes': 0.0
        }

    def spread_activation(
        self,
        source_nodes: List[str],
        graph: Any,
        initial_activation: float = 1.0,
        max_hops: Optional[int] = None,
        decay_rate: Optional[float] = None
    ) -> ActivationMap:
        """
        Spread activation from source nodes across graph.

        Uses breadth-first propagation with exponential decay.

        Args:
            source_nodes: Initial activation sources
            graph: Knowledge graph (KG or NetworkX MultiDiGraph)
            initial_activation: Starting activation level
            max_hops: Override config max_hops
            decay_rate: Override config decay_rate

        Returns:
            Dict mapping node_id -> final_activation
        """
        max_hops = max_hops or self.config.max_hops
        decay_rate = decay_rate or self.config.decay_rate

        # Initialize activation map
        activation: Dict[str, float] = {}
        visited: Set[str] = set()

        # BFS queue: (node_id, current_activation, hop_count)
        queue: deque = deque()

        # Add source nodes
        for node in source_nodes:
            activation[node] = initial_activation
            queue.append((node, initial_activation, 0))
            visited.add(node)

        # Track stats
        total_hops = 0
        hop_counts = []

        # Spread activation
        while queue:
            current_node, current_activation, hop_count = queue.popleft()

            # Stop if max hops reached
            if hop_count >= max_hops:
                continue

            # Get neighbors
            neighbors = self._get_neighbors(current_node, graph)

            for neighbor in neighbors:
                # Calculate decayed activation
                new_activation = current_activation * decay_rate

                # Skip if below minimum threshold
                if new_activation < self.config.min_activation:
                    continue

                # Update activation (accumulate if multiple paths)
                if neighbor in activation:
                    activation[neighbor] = max(activation[neighbor], new_activation)
                else:
                    activation[neighbor] = new_activation

                # Add to queue if not visited or activation increased
                if neighbor not in visited or activation[neighbor] > current_activation * decay_rate:
                    queue.append((neighbor, new_activation, hop_count + 1))
                    visited.add(neighbor)
                    total_hops += 1
                    hop_counts.append(hop_count + 1)

        # Update stats
        self._stats['total_spreads'] += 1
        if hop_counts:
            self._stats['avg_hops'] = sum(hop_counts) / len(hop_counts)
        self._stats['avg_activated_nodes'] = len(activation)

        # Store history
        self.activation_history.append(activation.copy())

        logger.debug(
            f"Activation spread: {len(source_nodes)} sources -> "
            f"{len(activation)} activated nodes "
            f"(avg {self._stats['avg_hops']:.1f} hops)"
        )

        return activation

    def get_activation_wave(
        self,
        wave: BetaWave,
        graph: Any
    ) -> ActivationMap:
        """
        Generate activation pattern from beta wave specification.

        Convenience wrapper around spread_activation using BetaWave params.

        Args:
            wave: Beta wave parameters
            graph: Knowledge graph

        Returns:
            Dict mapping node_id -> activation
        """
        return self.spread_activation(
            source_nodes=wave.source_nodes,
            graph=graph,
            initial_activation=wave.amplitude,
            max_hops=wave.max_hops,
            decay_rate=wave.decay_rate
        )

    def multi_frequency_spread(
        self,
        source_nodes: List[str],
        graph: Any,
        frequencies: List[float] = [15.0, 20.0, 25.0]
    ) -> Dict[float, ActivationMap]:
        """
        Spread activation at multiple frequencies simultaneously.

        Different frequencies create different activation patterns:
        - Low frequency (15 Hz): Slower, deeper spread
        - Medium frequency (20 Hz): Balanced spread (default)
        - High frequency (25 Hz): Faster, shallower spread

        Args:
            source_nodes: Initial sources
            graph: Knowledge graph
            frequencies: List of beta frequencies to use

        Returns:
            Dict mapping frequency -> activation_map
        """
        results = {}

        for freq in frequencies:
            # Frequency affects decay rate (higher freq = less decay)
            # Beta range: 12-30 Hz -> decay: 0.6-0.8
            normalized_freq = (freq - 12.0) / (30.0 - 12.0)  # 0.0-1.0
            freq_decay = 0.6 + (0.2 * normalized_freq)  # 0.6-0.8

            wave = BetaWave(
                frequency=freq,
                amplitude=self.config.amplitude,
                decay_rate=freq_decay,
                max_hops=self.config.max_hops,
                source_nodes=source_nodes
            )

            results[freq] = self.get_activation_wave(wave, graph)

        return results

    def directional_spread(
        self,
        source_nodes: List[str],
        graph: Any,
        edge_weights: Optional[Dict[Tuple[str, str], float]] = None
    ) -> ActivationMap:
        """
        Spread activation following edge semantics.

        Different edge types have different conductivity:
        - IS_A: High conductivity (0.9)
        - USES: Medium conductivity (0.7)
        - MENTIONS: Low conductivity (0.5)
        - etc.

        Args:
            source_nodes: Initial sources
            graph: Knowledge graph
            edge_weights: Optional custom edge weights

        Returns:
            Dict mapping node_id -> activation
        """
        # Default edge type conductivity
        default_conductivity = {
            'IS_A': 0.9,
            'USES': 0.7,
            'LEADS_TO': 0.8,
            'PART_OF': 0.75,
            'MENTIONS': 0.5,
            'IN_TIME': 0.6,
            'OCCURRED_AT': 0.65
        }

        activation: Dict[str, float] = {}
        visited: Set[str] = set()
        queue: deque = deque()

        # Initialize sources
        for node in source_nodes:
            activation[node] = self.config.amplitude
            queue.append((node, self.config.amplitude, 0))
            visited.add(node)

        # Spread with edge-aware decay
        while queue:
            current_node, current_activation, hop_count = queue.popleft()

            if hop_count >= self.config.max_hops:
                continue

            # Get neighbors with edge types
            neighbors_with_edges = self._get_neighbors_with_edges(current_node, graph)

            for neighbor, edge_type in neighbors_with_edges:
                # Get edge conductivity
                conductivity = default_conductivity.get(edge_type, 0.6)

                # Combine base decay with edge conductivity
                effective_decay = self.config.decay_rate * conductivity

                # Calculate new activation
                new_activation = current_activation * effective_decay

                if new_activation < self.config.min_activation:
                    continue

                # Update activation
                if neighbor in activation:
                    activation[neighbor] = max(activation[neighbor], new_activation)
                else:
                    activation[neighbor] = new_activation

                if neighbor not in visited:
                    queue.append((neighbor, new_activation, hop_count + 1))
                    visited.add(neighbor)

        return activation

    def get_activation_states(
        self,
        activation_map: ActivationMap
    ) -> Dict[str, ActivationState]:
        """
        Convert activation map to full ActivationState objects.

        Args:
            activation_map: Dict of node_id -> activation

        Returns:
            Dict of node_id -> ActivationState
        """
        import time
        now = time.time()

        states = {}
        for node_id, activation in activation_map.items():
            states[node_id] = ActivationState(
                node_id=node_id,
                activation=activation,
                last_accessed=now,
                access_count=1,  # Incremented if node accessed before
                heat=0.0,  # Will be filled by importance scorer
                importance_scores={}
            )

        return states

    def get_stats(self) -> Dict[str, Any]:
        """Get spreading statistics."""
        return self._stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            'total_spreads': 0,
            'avg_hops': 0.0,
            'avg_activated_nodes': 0.0
        }
        self.activation_history.clear()

    # Helper methods

    def _get_neighbors(self, node: str, graph: Any) -> List[str]:
        """Get neighbors of a node from graph."""
        if NETWORKX_AVAILABLE and isinstance(graph, nx.MultiDiGraph):
            return list(graph.successors(node))

        # Assume KG from HoloLoom.memory.graph
        if hasattr(graph, 'get_outgoing_edges'):
            edges = graph.get_outgoing_edges(node)
            return [edge.tail for edge in edges]

        # Fallback: assume dict-like adjacency
        if hasattr(graph, '__getitem__'):
            try:
                return list(graph[node].keys())
            except (KeyError, AttributeError):
                return []

        logger.warning(f"Unknown graph type: {type(graph)}")
        return []

    def _get_neighbors_with_edges(
        self,
        node: str,
        graph: Any
    ) -> List[Tuple[str, str]]:
        """Get neighbors with edge types."""
        if NETWORKX_AVAILABLE and isinstance(graph, nx.MultiDiGraph):
            neighbors = []
            for _, neighbor, data in graph.out_edges(node, data=True):
                edge_type = data.get('type', 'UNKNOWN')
                neighbors.append((neighbor, edge_type))
            return neighbors

        # KG from HoloLoom
        if hasattr(graph, 'get_outgoing_edges'):
            edges = graph.get_outgoing_edges(node)
            return [(edge.tail, edge.type) for edge in edges]

        # Fallback: no edge type info
        return [(n, 'UNKNOWN') for n in self._get_neighbors(node, graph)]


# Convenience functions

def spread_from_query(
    query_nodes: List[str],
    graph: Any,
    config: Optional[BetaWaveConfig] = None
) -> ActivationMap:
    """
    Quick activation spread from query-relevant nodes.

    Args:
        query_nodes: Nodes relevant to query
        graph: Knowledge graph
        config: Optional config override

    Returns:
        Activation map
    """
    spreader = ActivationSpreader(config)
    return spreader.spread_activation(query_nodes, graph)


def multi_scale_spread(
    query_nodes: List[str],
    graph: Any,
    scales: List[int] = [1, 2, 3]
) -> Dict[int, ActivationMap]:
    """
    Spread activation at multiple scales (hop distances).

    Args:
        query_nodes: Initial sources
        graph: Knowledge graph
        scales: List of max_hops values

    Returns:
        Dict mapping scale -> activation_map
    """
    spreader = ActivationSpreader()
    results = {}

    for scale in scales:
        results[scale] = spreader.spread_activation(
            query_nodes,
            graph,
            max_hops=scale
        )

    return results
