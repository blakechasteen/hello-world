"""
Dark Trace Multilayer: Feature Propagation Analysis

Analyzes how features propagate across layers and discovers
computational circuits within the model.

Key Capabilities:
- Feature path tracking through layers
- Circuit discovery (connected feature subgraphs)
- Propagation strength analysis
- Bottleneck detection

Research Basis:
- ACDC (Conmy et al. 2023): Automatic Circuit Discovery
- Path Patching (Goldowsky-Dill et al. 2023): Localization via path interventions
- Scaling Monosemanticity (Anthropic 2024): Multi-layer SAE analysis

Created: 2025-12-28
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# Data Structures
# =============================================================================

class PropagationType(Enum):
    """Types of feature propagation."""
    FORWARD = "forward"           # Layer N → Layer N+1
    BACKWARD = "backward"         # Layer N+1 → Layer N (for gradient analysis)
    BIDIRECTIONAL = "bidirectional"  # Both directions


class CircuitType(Enum):
    """Types of discovered circuits."""
    LINEAR = "linear"             # Sequential path through layers
    BRANCHING = "branching"       # One feature affects multiple downstream
    CONVERGING = "converging"     # Multiple features affect one downstream
    COMPLEX = "complex"           # Multiple branches and convergences


@dataclass
class PropagationEdge:
    """A single edge in the propagation graph."""
    source_feature: str
    target_feature: str
    source_layer: int
    target_layer: int
    strength: float          # Correlation or causal strength
    confidence: float        # Confidence in this edge
    edge_type: str = "correlation"  # "correlation", "causal", "patched"

    def __hash__(self):
        return hash((self.source_feature, self.target_feature))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_feature": self.source_feature,
            "target_feature": self.target_feature,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "strength": self.strength,
            "confidence": self.confidence,
            "edge_type": self.edge_type
        }


@dataclass
class PropagationPath:
    """
    A path showing how a feature propagates through layers.

    Represents the "journey" of information from an input feature
    through intermediate layers to output features.
    """
    start_feature: str
    start_layer: int
    edges: list[PropagationEdge] = field(default_factory=list)
    end_features: list[str] = field(default_factory=list)
    end_layer: int = 0
    total_strength: float = 0.0
    path_type: PropagationType = PropagationType.FORWARD

    def __post_init__(self):
        if self.edges:
            self.total_strength = self._compute_total_strength()
            self.end_layer = max(e.target_layer for e in self.edges)
            self.end_features = list({
                e.target_feature for e in self.edges
                if e.target_layer == self.end_layer
            })

    def _compute_total_strength(self) -> float:
        """Compute cumulative path strength (product of edge strengths)."""
        if not self.edges:
            return 0.0
        strength = 1.0
        for edge in self.edges:
            strength *= abs(edge.strength)
        return strength

    @property
    def length(self) -> int:
        """Path length (number of edges)."""
        return len(self.edges)

    @property
    def n_layers_spanned(self) -> int:
        """Number of layers this path spans."""
        if not self.edges:
            return 0
        return self.end_layer - self.start_layer

    def get_features_at_layer(self, layer: int) -> list[str]:
        """Get all features in this path at a specific layer."""
        features = set()
        if layer == self.start_layer:
            features.add(self.start_feature)

        for edge in self.edges:
            if edge.source_layer == layer:
                features.add(edge.source_feature)
            if edge.target_layer == layer:
                features.add(edge.target_feature)

        return list(features)

    def get_weakest_edge(self) -> PropagationEdge | None:
        """Get the weakest edge in the path (potential bottleneck)."""
        if not self.edges:
            return None
        return min(self.edges, key=lambda e: abs(e.strength))

    def get_strongest_edge(self) -> PropagationEdge | None:
        """Get the strongest edge in the path."""
        if not self.edges:
            return None
        return max(self.edges, key=lambda e: abs(e.strength))

    def summary(self) -> dict[str, Any]:
        """Get path summary."""
        return {
            "start_feature": self.start_feature,
            "start_layer": self.start_layer,
            "end_features": self.end_features,
            "end_layer": self.end_layer,
            "length": self.length,
            "n_layers_spanned": self.n_layers_spanned,
            "total_strength": self.total_strength,
            "path_type": self.path_type.value,
            "weakest_strength": self.get_weakest_edge().strength if self.edges else 0.0
        }


@dataclass
class CircuitCandidate:
    """
    A discovered circuit - a subgraph of features that work together.

    Circuits are connected sets of features across layers that
    collectively compute some function.
    """
    circuit_id: str
    features: list[str]             # All features in circuit
    edges: list[PropagationEdge]    # All edges in circuit
    input_features: list[str]       # Features at earliest layer
    output_features: list[str]      # Features at latest layer
    layers_spanned: list[int]       # All layers involved
    circuit_type: CircuitType = CircuitType.LINEAR
    total_strength: float = 0.0
    description: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.edges and self.total_strength == 0.0:
            self.total_strength = sum(abs(e.strength) for e in self.edges) / len(self.edges)

        # Determine circuit type
        self.circuit_type = self._classify_circuit()

    def _classify_circuit(self) -> CircuitType:
        """Classify circuit type based on structure."""
        if not self.edges:
            return CircuitType.LINEAR

        # Count fan-out and fan-in
        source_counts = {}
        target_counts = {}

        for edge in self.edges:
            source_counts[edge.source_feature] = source_counts.get(edge.source_feature, 0) + 1
            target_counts[edge.target_feature] = target_counts.get(edge.target_feature, 0) + 1

        has_branching = any(count > 1 for count in source_counts.values())
        has_converging = any(count > 1 for count in target_counts.values())

        if has_branching and has_converging:
            return CircuitType.COMPLEX
        elif has_branching:
            return CircuitType.BRANCHING
        elif has_converging:
            return CircuitType.CONVERGING
        else:
            return CircuitType.LINEAR

    @property
    def n_features(self) -> int:
        """Number of features in circuit."""
        return len(self.features)

    @property
    def n_edges(self) -> int:
        """Number of edges in circuit."""
        return len(self.edges)

    @property
    def depth(self) -> int:
        """Circuit depth (number of layers spanned)."""
        return len(self.layers_spanned)

    def get_features_at_layer(self, layer: int) -> list[str]:
        """Get features at a specific layer."""
        layer_features = set()

        for edge in self.edges:
            if edge.source_layer == layer:
                layer_features.add(edge.source_feature)
            if edge.target_layer == layer:
                layer_features.add(edge.target_feature)

        return list(layer_features)

    def contains_feature(self, feature_id: str) -> bool:
        """Check if circuit contains a specific feature."""
        return feature_id in self.features

    def merge(self, other: "CircuitCandidate") -> "CircuitCandidate":
        """Merge two overlapping circuits."""
        merged_features = list(set(self.features + other.features))
        merged_edges = list({
            (e.source_feature, e.target_feature): e
            for e in self.edges + other.edges
        }.values())
        merged_layers = sorted(set(self.layers_spanned + other.layers_spanned))

        # Determine inputs and outputs
        input_layer = min(merged_layers)
        output_layer = max(merged_layers)

        input_features = [
            f for f in merged_features
            if any(e.source_feature == f and e.source_layer == input_layer for e in merged_edges)
        ]
        output_features = [
            f for f in merged_features
            if any(e.target_feature == f and e.target_layer == output_layer for e in merged_edges)
        ]

        return CircuitCandidate(
            circuit_id=f"{self.circuit_id}+{other.circuit_id}",
            features=merged_features,
            edges=merged_edges,
            input_features=input_features or [merged_features[0]] if merged_features else [],
            output_features=output_features or [merged_features[-1]] if merged_features else [],
            layers_spanned=merged_layers,
            description=f"Merged: {self.description} + {other.description}"
        )

    def summary(self) -> dict[str, Any]:
        """Get circuit summary."""
        return {
            "circuit_id": self.circuit_id,
            "circuit_type": self.circuit_type.value,
            "n_features": self.n_features,
            "n_edges": self.n_edges,
            "depth": self.depth,
            "layers_spanned": self.layers_spanned,
            "total_strength": self.total_strength,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "description": self.description
        }


# =============================================================================
# Propagation Analyzer
# =============================================================================

@dataclass
class PropagationConfig:
    """Configuration for propagation analysis."""
    min_edge_strength: float = 0.3      # Minimum correlation to consider
    max_path_length: int = 10           # Maximum path length to explore
    max_circuits: int = 100             # Maximum circuits to discover
    min_circuit_strength: float = 0.2   # Minimum average edge strength for circuit
    merge_overlapping: bool = True      # Merge overlapping circuits


class PropagationAnalyzer:
    """
    Analyzes feature propagation across layers.

    Traces how features in early layers influence features in later layers,
    and discovers circuits - connected subgraphs that compute functions.

    Usage:
        from hololoom.dark_trace.multilayer import CorrelationTracker

        # Get correlations
        tracker = CorrelationTracker(n_layers=12)
        # ... update tracker with activations ...

        # Create analyzer
        analyzer = PropagationAnalyzer(tracker)

        # Trace a feature's propagation
        path = analyzer.trace_feature("layer_0.sae_42")
        print(f"Feature propagates through {path.n_layers_spanned} layers")

        # Discover circuits
        circuits = analyzer.discover_circuits()
        for circuit in circuits[:5]:
            print(f"Circuit: {circuit.circuit_type.value}, {circuit.n_features} features")
    """

    def __init__(
        self,
        correlation_tracker: Any,  # CorrelationTracker
        config: PropagationConfig | None = None
    ):
        """
        Initialize propagation analyzer.

        Args:
            correlation_tracker: CorrelationTracker with computed correlations
            config: Analysis configuration
        """
        self.tracker = correlation_tracker
        self.config = config or PropagationConfig()

        # Cache for discovered paths and circuits
        self._path_cache: dict[str, PropagationPath] = {}
        self._circuits: list[CircuitCandidate] = []
        self._circuit_discovery_time: float = 0.0

    def trace_feature(
        self,
        feature_id: str,
        direction: PropagationType = PropagationType.FORWARD,
        max_depth: int | None = None
    ) -> PropagationPath:
        """
        Trace how a feature propagates through layers.

        Args:
            feature_id: Feature ID (e.g., "layer_0.sae_42")
            direction: Direction of propagation
            max_depth: Maximum layers to trace (defaults to config)

        Returns:
            PropagationPath showing feature's journey
        """
        cache_key = f"{feature_id}_{direction.value}_{max_depth}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        max_depth = max_depth or self.config.max_path_length

        # Parse starting layer from feature ID
        start_layer = self._extract_layer(feature_id)

        edges = []

        if direction == PropagationType.FORWARD:
            edges = self._trace_forward(feature_id, start_layer, max_depth)
        elif direction == PropagationType.BACKWARD:
            edges = self._trace_backward(feature_id, start_layer, max_depth)
        else:  # BIDIRECTIONAL
            forward_edges = self._trace_forward(feature_id, start_layer, max_depth)
            backward_edges = self._trace_backward(feature_id, start_layer, max_depth)
            edges = backward_edges + forward_edges

        path = PropagationPath(
            start_feature=feature_id,
            start_layer=start_layer,
            edges=edges,
            path_type=direction
        )

        self._path_cache[cache_key] = path
        return path

    def _extract_layer(self, feature_id: str) -> int:
        """Extract layer number from feature ID."""
        parts = feature_id.split(".")
        for part in parts:
            if part.startswith("layer_"):
                try:
                    return int(part.replace("layer_", ""))
                except ValueError:
                    pass
        return 0

    def _trace_forward(
        self,
        start_feature: str,
        start_layer: int,
        max_depth: int
    ) -> list[PropagationEdge]:
        """Trace feature propagation forward through layers."""
        edges = []
        current_features = {start_feature}
        current_layer = start_layer

        for _ in range(max_depth):
            target_layer = current_layer + 1

            if target_layer >= self.tracker.n_layers:
                break

            # Get correlations for this layer pair
            layer_corr = self.tracker.get_layer_correlation(current_layer, target_layer)

            # Find strongest connections from current features
            next_features = set()

            for corr in layer_corr.correlations:
                if corr.source_feature not in current_features:
                    continue

                if abs(corr.correlation) < self.config.min_edge_strength:
                    continue

                edges.append(PropagationEdge(
                    source_feature=corr.source_feature,
                    target_feature=corr.target_feature,
                    source_layer=current_layer,
                    target_layer=target_layer,
                    strength=corr.correlation,
                    confidence=corr.confidence,
                    edge_type="correlation"
                ))

                next_features.add(corr.target_feature)

            if not next_features:
                break

            current_features = next_features
            current_layer = target_layer

        return edges

    def _trace_backward(
        self,
        start_feature: str,
        start_layer: int,
        max_depth: int
    ) -> list[PropagationEdge]:
        """Trace feature propagation backward through layers."""
        edges = []
        current_features = {start_feature}
        current_layer = start_layer

        for _ in range(max_depth):
            source_layer = current_layer - 1

            if source_layer < 0:
                break

            # Get correlations for this layer pair
            layer_corr = self.tracker.get_layer_correlation(source_layer, current_layer)

            # Find strongest connections to current features
            prev_features = set()

            for corr in layer_corr.correlations:
                if corr.target_feature not in current_features:
                    continue

                if abs(corr.correlation) < self.config.min_edge_strength:
                    continue

                edges.append(PropagationEdge(
                    source_feature=corr.source_feature,
                    target_feature=corr.target_feature,
                    source_layer=source_layer,
                    target_layer=current_layer,
                    strength=corr.correlation,
                    confidence=corr.confidence,
                    edge_type="correlation"
                ))

                prev_features.add(corr.source_feature)

            if not prev_features:
                break

            current_features = prev_features
            current_layer = source_layer

        # Reverse to maintain source→target order
        return list(reversed(edges))

    def discover_circuits(
        self,
        min_features: int = 3,
        min_layers: int = 2
    ) -> list[CircuitCandidate]:
        """
        Discover computational circuits in the model.

        A circuit is a connected subgraph of features across layers
        that collectively compute some function.

        Args:
            min_features: Minimum features per circuit
            min_layers: Minimum layers spanned

        Returns:
            List of discovered circuits, sorted by strength
        """
        start_time = time.time()

        # Build propagation graph
        edges = self._collect_all_edges()

        if not edges:
            return []

        # Find connected components
        circuits = self._find_connected_components(edges, min_features, min_layers)

        # Filter by strength
        circuits = [
            c for c in circuits
            if c.total_strength >= self.config.min_circuit_strength
        ]

        # Optionally merge overlapping circuits
        if self.config.merge_overlapping:
            circuits = self._merge_overlapping_circuits(circuits)

        # Sort by strength
        circuits.sort(key=lambda c: c.total_strength, reverse=True)

        # Limit count
        circuits = circuits[:self.config.max_circuits]

        self._circuits = circuits
        self._circuit_discovery_time = time.time() - start_time

        return circuits

    def _collect_all_edges(self) -> list[PropagationEdge]:
        """Collect all significant edges from correlation tracker."""
        edges = []

        for layer in range(self.tracker.n_layers - 1):
            layer_corr = self.tracker.get_layer_correlation(layer, layer + 1)

            for corr in layer_corr.correlations:
                if abs(corr.correlation) < self.config.min_edge_strength:
                    continue

                edges.append(PropagationEdge(
                    source_feature=corr.source_feature,
                    target_feature=corr.target_feature,
                    source_layer=layer,
                    target_layer=layer + 1,
                    strength=corr.correlation,
                    confidence=corr.confidence
                ))

        return edges

    def _find_connected_components(
        self,
        edges: list[PropagationEdge],
        min_features: int,
        min_layers: int
    ) -> list[CircuitCandidate]:
        """Find connected components in the propagation graph."""
        # Build adjacency representation
        adj: dict[str, set[str]] = {}
        edge_map: dict[tuple[str, str], PropagationEdge] = {}

        for edge in edges:
            if edge.source_feature not in adj:
                adj[edge.source_feature] = set()
            if edge.target_feature not in adj:
                adj[edge.target_feature] = set()

            adj[edge.source_feature].add(edge.target_feature)
            adj[edge.target_feature].add(edge.source_feature)  # Undirected for component finding

            edge_map[(edge.source_feature, edge.target_feature)] = edge

        # Find components using BFS
        visited = set()
        components = []

        for start_node in adj:
            if start_node in visited:
                continue

            # BFS to find component
            component_nodes = set()
            queue = [start_node]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                component_nodes.add(node)

                for neighbor in adj.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            components.append(component_nodes)

        # Convert components to CircuitCandidates
        circuits = []
        circuit_id = 0

        for comp_nodes in components:
            # Get edges within component
            comp_edges = []
            for (src, tgt), edge in edge_map.items():
                if src in comp_nodes and tgt in comp_nodes:
                    comp_edges.append(edge)

            if not comp_edges:
                continue

            # Get layers spanned
            layers = set()
            for edge in comp_edges:
                layers.add(edge.source_layer)
                layers.add(edge.target_layer)

            sorted_layers = sorted(layers)

            # Check minimum requirements
            if len(comp_nodes) < min_features:
                continue
            if len(sorted_layers) < min_layers:
                continue

            # Find input and output features
            input_layer = min(sorted_layers)
            output_layer = max(sorted_layers)

            input_features = [
                e.source_feature for e in comp_edges
                if e.source_layer == input_layer
            ]
            output_features = [
                e.target_feature for e in comp_edges
                if e.target_layer == output_layer
            ]

            circuit = CircuitCandidate(
                circuit_id=f"circuit_{circuit_id}",
                features=list(comp_nodes),
                edges=comp_edges,
                input_features=list(set(input_features)),
                output_features=list(set(output_features)),
                layers_spanned=sorted_layers
            )

            circuits.append(circuit)
            circuit_id += 1

        return circuits

    def _merge_overlapping_circuits(
        self,
        circuits: list[CircuitCandidate]
    ) -> list[CircuitCandidate]:
        """Merge circuits that share features."""
        if not circuits:
            return circuits

        # Find overlapping pairs
        merged = []
        used = set()

        for i, c1 in enumerate(circuits):
            if i in used:
                continue

            current = c1

            for j, c2 in enumerate(circuits[i + 1:], start=i + 1):
                if j in used:
                    continue

                # Check for overlap
                shared = set(current.features) & set(c2.features)
                if len(shared) >= 2:  # Significant overlap
                    current = current.merge(c2)
                    used.add(j)

            merged.append(current)
            used.add(i)

        return merged

    def find_feature_circuits(self, feature_id: str) -> list[CircuitCandidate]:
        """Find all circuits containing a specific feature."""
        if not self._circuits:
            self.discover_circuits()

        return [c for c in self._circuits if c.contains_feature(feature_id)]

    def get_circuit_overlap(
        self,
        circuit1: CircuitCandidate,
        circuit2: CircuitCandidate
    ) -> float:
        """Compute overlap between two circuits (Jaccard similarity)."""
        features1 = set(circuit1.features)
        features2 = set(circuit2.features)

        intersection = features1 & features2
        union = features1 | features2

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def analyze_bottlenecks(self) -> list[dict[str, Any]]:
        """
        Find bottleneck features - features many circuits pass through.

        Returns:
            List of bottleneck features with their statistics
        """
        if not self._circuits:
            self.discover_circuits()

        # Count feature appearances in circuits
        feature_counts: dict[str, int] = {}
        feature_strengths: dict[str, list[float]] = {}

        for circuit in self._circuits:
            for feature in circuit.features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
                if feature not in feature_strengths:
                    feature_strengths[feature] = []
                feature_strengths[feature].append(circuit.total_strength)

        # Find bottlenecks (features in multiple circuits)
        bottlenecks = []

        for feature, count in feature_counts.items():
            if count >= 2:
                strengths = feature_strengths[feature]
                bottlenecks.append({
                    "feature": feature,
                    "circuit_count": count,
                    "mean_circuit_strength": sum(strengths) / len(strengths),
                    "max_circuit_strength": max(strengths),
                    "layer": self._extract_layer(feature)
                })

        # Sort by circuit count
        bottlenecks.sort(key=lambda b: b["circuit_count"], reverse=True)

        return bottlenecks

    def statistics(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "n_cached_paths": len(self._path_cache),
            "n_circuits": len(self._circuits),
            "circuit_discovery_time": self._circuit_discovery_time,
            "circuit_types": {
                ctype.value: sum(1 for c in self._circuits if c.circuit_type == ctype)
                for ctype in CircuitType
            },
            "config": {
                "min_edge_strength": self.config.min_edge_strength,
                "max_path_length": self.config.max_path_length,
                "max_circuits": self.config.max_circuits
            }
        }

    def clear_cache(self) -> None:
        """Clear cached paths and circuits."""
        self._path_cache.clear()
        self._circuits.clear()
        self._circuit_discovery_time = 0.0


# =============================================================================
# Factory Functions
# =============================================================================

def create_propagation_analyzer(
    correlation_tracker: Any,
    preset: str = "default"
) -> PropagationAnalyzer:
    """
    Create a propagation analyzer with preset configuration.

    Args:
        correlation_tracker: CorrelationTracker instance
        preset: Configuration preset ("default", "thorough", "fast")

    Returns:
        Configured PropagationAnalyzer
    """
    configs = {
        "default": PropagationConfig(),
        "thorough": PropagationConfig(
            min_edge_strength=0.2,
            max_path_length=20,
            max_circuits=500,
            min_circuit_strength=0.1
        ),
        "fast": PropagationConfig(
            min_edge_strength=0.5,
            max_path_length=5,
            max_circuits=50,
            min_circuit_strength=0.3
        )
    }

    config = configs.get(preset, PropagationConfig())
    return PropagationAnalyzer(correlation_tracker, config)
