"""
Dark Trace Multilayer: Feature Hierarchy Analysis

Analyzes how features evolve and relate across layers,
building abstraction ladders from low-level to high-level concepts.

Research Basis:
- Feature splitting/merging across layers
- Abstraction ladder: concrete -> abstract
- Composition: simple features combine into complex ones
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from hololoom.dark_trace.multilayer.correlation_tracker import (
        CorrelationPair,
        CorrelationTracker,
    )
    from hololoom.dark_trace.multilayer.propagation_analyzer import (
        PropagationAnalyzer,
        PropagationPath,
    )


class AbstractionLevel(Enum):
    """Level of abstraction for features."""
    RAW = "raw"  # Raw input features (tokens, pixels)
    LOW = "low"  # Low-level patterns (n-grams, edges)
    MID = "mid"  # Mid-level concepts (phrases, textures)
    HIGH = "high"  # High-level concepts (topics, objects)
    ABSTRACT = "abstract"  # Abstract concepts (sentiment, style)


class FeatureRelationType(Enum):
    """Type of relationship between features across layers."""
    PERSISTS = "persists"  # Same feature continues
    SPLITS = "splits"  # One feature becomes multiple
    MERGES = "merges"  # Multiple features become one
    TRANSFORMS = "transforms"  # Feature changes meaning
    EMERGES = "emerges"  # New feature appears
    DISAPPEARS = "disappears"  # Feature stops activating


@dataclass
class FeatureEvolution:
    """
    Tracks how a feature evolves across layers.

    Captures where a feature emerges, how it changes,
    and where it has most impact.
    """

    feature_id: str  # Feature identifier (e.g., "sae.42")
    source_layer: int  # Which layer's SAE this feature belongs to

    # Evolution tracking
    emergence_layer: int = 0  # First layer where feature is active
    peak_layer: int = 0  # Layer with maximum activation
    disappearance_layer: int | None = None  # Where feature stops

    # Layer-by-layer activations
    layer_activations: dict[int, float] = field(default_factory=dict)

    # Relationships to features in other layers
    predecessors: list[tuple[str, float]] = field(default_factory=list)  # Features that lead to this
    successors: list[tuple[str, float]] = field(default_factory=list)  # Features this leads to

    # Abstraction progression
    abstraction_trajectory: list[AbstractionLevel] = field(default_factory=list)

    # Semantic drift (how much meaning changes)
    semantic_drift: float = 0.0  # 0 = stable, 1 = completely different

    def get_lifespan(self) -> int:
        """Get number of layers where feature is active."""
        if self.disappearance_layer is not None:
            return self.disappearance_layer - self.emergence_layer
        return len(self.layer_activations)

    def get_peak_activation(self) -> float:
        """Get maximum activation across layers."""
        if not self.layer_activations:
            return 0.0
        return max(self.layer_activations.values())

    def is_persistent(self, threshold: float = 0.5) -> bool:
        """Check if feature persists across many layers."""
        n_active = sum(1 for a in self.layer_activations.values() if a > threshold)
        return n_active >= len(self.layer_activations) * 0.7


@dataclass
class HierarchyNode:
    """Node in the feature hierarchy graph."""

    feature_id: str
    layer: int
    abstraction: AbstractionLevel

    # Node properties
    activation_mean: float = 0.0
    activation_std: float = 0.0
    importance: float = 0.0  # Based on downstream impact

    # Semantic label (if available)
    label: str | None = None
    label_confidence: float = 0.0

    # Grouping
    cluster_id: int | None = None


@dataclass
class HierarchyEdge:
    """Edge connecting features across layers."""

    source: str  # Source feature ID
    target: str  # Target feature ID
    source_layer: int
    target_layer: int

    # Edge properties
    correlation: float = 0.0  # Activation correlation
    causal_strength: float = 0.0  # From patching experiments
    relation_type: FeatureRelationType = FeatureRelationType.PERSISTS

    # Weight for graph algorithms
    weight: float = 0.0


@dataclass
class AbstractionLadder:
    """
    Complete abstraction ladder from raw to abstract.

    Shows how concepts build from simple to complex
    across model layers.
    """

    # Nodes at each abstraction level
    levels: dict[AbstractionLevel, list[HierarchyNode]] = field(default_factory=dict)

    # Edges connecting levels
    edges: list[HierarchyEdge] = field(default_factory=list)

    # Summary statistics
    n_features: int = 0
    n_layers: int = 0
    avg_abstraction_increase: float = 0.0

    # Key paths (most important abstraction chains)
    key_paths: list[list[str]] = field(default_factory=list)

    def get_level(self, level: AbstractionLevel) -> list[HierarchyNode]:
        """Get all nodes at an abstraction level."""
        return self.levels.get(level, [])

    def get_edges_between(
        self,
        source_level: AbstractionLevel,
        target_level: AbstractionLevel,
    ) -> list[HierarchyEdge]:
        """Get edges connecting two abstraction levels."""
        source_features = {n.feature_id for n in self.get_level(source_level)}
        target_features = {n.feature_id for n in self.get_level(target_level)}

        return [
            e for e in self.edges
            if e.source in source_features and e.target in target_features
        ]


class FeatureHierarchyAnalyzer:
    """
    Analyzes feature hierarchy across model layers.

    Builds abstraction ladders, tracks feature evolution,
    and identifies composition patterns.
    """

    def __init__(
        self,
        layer_sae_manager: Any,  # LayerSAEManager
        correlation_threshold: float = 0.5,
        abstraction_method: str = "layer_position",
    ):
        """
        Initialize hierarchy analyzer.

        Args:
            layer_sae_manager: Manager with trained layer SAEs
            correlation_threshold: Minimum correlation to link features
            abstraction_method: How to assign abstraction levels
                - "layer_position": Based on layer index
                - "activation_pattern": Based on what activates feature
                - "semantic": Based on feature labels
        """
        self.manager = layer_sae_manager
        self.correlation_threshold = correlation_threshold
        self.abstraction_method = abstraction_method

        # Cached analysis results
        self._feature_correlations: dict[tuple[int, int], torch.Tensor] = {}
        self._hierarchy_nodes: dict[str, HierarchyNode] = {}
        self._hierarchy_edges: list[HierarchyEdge] = []
        self._feature_evolutions: dict[str, FeatureEvolution] = {}

    def analyze_cross_layer_correlations(
        self,
        test_inputs: list[Any],
        layer_pairs: list[tuple[int, int]] | None = None,
    ) -> dict[tuple[int, int], torch.Tensor]:
        """
        Compute feature correlations between layer pairs.

        Args:
            test_inputs: Inputs to run through model
            layer_pairs: Which layer pairs to analyze (None = adjacent)

        Returns:
            Correlation matrices for each layer pair
        """
        n_layers = self.manager.n_layers

        if layer_pairs is None:
            # Default: analyze adjacent layer pairs
            layer_pairs = [(i, i + 1) for i in range(n_layers - 1)]

        correlations = {}

        for layer_a, layer_b in layer_pairs:
            # Get features from both layers
            features = self.manager.encode_all_layers(
                test_inputs,
                layers=[layer_a, layer_b],
            )

            feat_a = features.get_layer(layer_a)
            feat_b = features.get_layer(layer_b)

            if feat_a is None or feat_b is None:
                continue

            # Flatten to [n_samples, n_features]
            if feat_a.dim() == 3:
                feat_a = feat_a.reshape(-1, feat_a.shape[-1])
                feat_b = feat_b.reshape(-1, feat_b.shape[-1])

            # Compute correlation matrix
            # Normalize features
            feat_a_norm = feat_a - feat_a.mean(dim=0, keepdim=True)
            feat_b_norm = feat_b - feat_b.mean(dim=0, keepdim=True)

            feat_a_std = feat_a_norm.std(dim=0, keepdim=True) + 1e-8
            feat_b_std = feat_b_norm.std(dim=0, keepdim=True) + 1e-8

            feat_a_norm = feat_a_norm / feat_a_std
            feat_b_norm = feat_b_norm / feat_b_std

            # Correlation: [n_features_a, n_features_b]
            corr = torch.mm(feat_a_norm.T, feat_b_norm) / feat_a_norm.shape[0]

            correlations[(layer_a, layer_b)] = corr

        self._feature_correlations = correlations
        return correlations

    def track_feature_evolution(
        self,
        feature_id: str,
        test_inputs: list[Any],
    ) -> FeatureEvolution:
        """
        Track how a specific feature evolves across layers.

        Args:
            feature_id: Feature to track (e.g., "layer5.sae.42")
            test_inputs: Inputs to analyze

        Returns:
            FeatureEvolution with cross-layer tracking
        """
        # Parse feature ID to get source layer
        parts = feature_id.split(".")
        if len(parts) >= 2 and parts[0].startswith("layer"):
            source_layer = int(parts[0].replace("layer", ""))
            feature_idx = int(parts[-1])
        else:
            # Assume format "sae.42" with implicit layer
            source_layer = 0
            feature_idx = int(parts[-1]) if parts[-1].isdigit() else 0

        # Get this feature's activation pattern
        features = self.manager.encode_all_layers(test_inputs)
        source_features = features.get_layer(source_layer)

        if source_features is None:
            return FeatureEvolution(
                feature_id=feature_id,
                source_layer=source_layer,
            )

        # Get reference activation pattern
        if source_features.dim() == 3:
            source_features = source_features.reshape(-1, source_features.shape[-1])

        if feature_idx >= source_features.shape[-1]:
            return FeatureEvolution(
                feature_id=feature_id,
                source_layer=source_layer,
            )

        reference_pattern = source_features[:, feature_idx]

        # Track across all layers
        layer_activations = {}
        predecessors = []
        successors = []

        for layer_idx in range(self.manager.n_layers):
            layer_features = features.get_layer(layer_idx)
            if layer_features is None:
                continue

            if layer_features.dim() == 3:
                layer_features = layer_features.reshape(-1, layer_features.shape[-1])

            # Find most correlated feature in this layer
            correlations = []
            for feat_idx in range(layer_features.shape[-1]):
                feat_pattern = layer_features[:, feat_idx]
                corr = torch.corrcoef(
                    torch.stack([reference_pattern, feat_pattern])
                )[0, 1]
                correlations.append((feat_idx, corr.item() if not torch.isnan(corr) else 0))

            if correlations:
                best_idx, best_corr = max(correlations, key=lambda x: abs(x[1]))
                layer_activations[layer_idx] = best_corr

                # Track relationships
                if layer_idx < source_layer and abs(best_corr) > self.correlation_threshold:
                    predecessors.append((f"layer{layer_idx}.sae.{best_idx}", best_corr))
                elif layer_idx > source_layer and abs(best_corr) > self.correlation_threshold:
                    successors.append((f"layer{layer_idx}.sae.{best_idx}", best_corr))

        # Determine emergence and peak
        active_layers = [l for l, a in layer_activations.items() if abs(a) > self.correlation_threshold]
        emergence_layer = min(active_layers) if active_layers else source_layer
        peak_layer = max(layer_activations.items(), key=lambda x: abs(x[1]))[0] if layer_activations else source_layer

        # Compute semantic drift (how much feature changes from source)
        if layer_activations:
            source_corr = layer_activations.get(source_layer, 1.0)
            final_corr = layer_activations.get(max(layer_activations.keys()), 0.0)
            semantic_drift = 1.0 - abs(final_corr)
        else:
            semantic_drift = 0.0

        evolution = FeatureEvolution(
            feature_id=feature_id,
            source_layer=source_layer,
            emergence_layer=emergence_layer,
            peak_layer=peak_layer,
            layer_activations=layer_activations,
            predecessors=predecessors,
            successors=successors,
            semantic_drift=semantic_drift,
        )

        self._feature_evolutions[feature_id] = evolution
        return evolution

    def build_hierarchy_graph(
        self,
        test_inputs: list[Any],
        top_k_per_layer: int = 50,
    ) -> tuple[list[HierarchyNode], list[HierarchyEdge]]:
        """
        Build complete hierarchy graph across all layers.

        Args:
            test_inputs: Inputs to analyze
            top_k_per_layer: Number of top features per layer to include

        Returns:
            Tuple of (nodes, edges)
        """
        nodes = []
        edges = []

        # Get features from all layers
        features = self.manager.encode_all_layers(test_inputs)

        # Build nodes for each layer
        for layer_idx in range(self.manager.n_layers):
            layer_features = features.get_layer(layer_idx)
            if layer_features is None:
                continue

            # Flatten and get statistics
            if layer_features.dim() == 3:
                layer_features = layer_features.reshape(-1, layer_features.shape[-1])

            mean_activations = layer_features.mean(dim=0)
            std_activations = layer_features.std(dim=0)

            # Get top-k features by mean activation
            top_indices = torch.argsort(mean_activations, descending=True)[:top_k_per_layer]

            for rank, feat_idx in enumerate(top_indices.tolist()):
                feature_id = f"layer{layer_idx}.sae.{feat_idx}"

                # Assign abstraction level based on layer position
                abstraction = self._assign_abstraction_level(layer_idx)

                node = HierarchyNode(
                    feature_id=feature_id,
                    layer=layer_idx,
                    abstraction=abstraction,
                    activation_mean=mean_activations[feat_idx].item(),
                    activation_std=std_activations[feat_idx].item(),
                    importance=1.0 / (rank + 1),  # Importance by rank
                )
                nodes.append(node)
                self._hierarchy_nodes[feature_id] = node

        # Compute correlations between adjacent layers
        correlations = self.analyze_cross_layer_correlations(test_inputs)

        # Build edges based on correlations
        for (layer_a, layer_b), corr_matrix in correlations.items():
            # Get nodes for each layer
            nodes_a = [n for n in nodes if n.layer == layer_a]
            nodes_b = [n for n in nodes if n.layer == layer_b]

            for node_a in nodes_a:
                feat_idx_a = int(node_a.feature_id.split(".")[-1])
                if feat_idx_a >= corr_matrix.shape[0]:
                    continue

                for node_b in nodes_b:
                    feat_idx_b = int(node_b.feature_id.split(".")[-1])
                    if feat_idx_b >= corr_matrix.shape[1]:
                        continue

                    corr = corr_matrix[feat_idx_a, feat_idx_b].item()

                    if abs(corr) >= self.correlation_threshold:
                        # Determine relation type
                        relation = self._determine_relation_type(
                            corr, node_a, node_b, corr_matrix, feat_idx_a
                        )

                        edge = HierarchyEdge(
                            source=node_a.feature_id,
                            target=node_b.feature_id,
                            source_layer=layer_a,
                            target_layer=layer_b,
                            correlation=corr,
                            relation_type=relation,
                            weight=abs(corr),
                        )
                        edges.append(edge)

        self._hierarchy_nodes = {n.feature_id: n for n in nodes}
        self._hierarchy_edges = edges

        return nodes, edges

    def build_abstraction_ladder(
        self,
        test_inputs: list[Any],
    ) -> AbstractionLadder:
        """
        Build complete abstraction ladder.

        Args:
            test_inputs: Inputs to analyze

        Returns:
            AbstractionLadder showing concept hierarchy
        """
        # Build hierarchy graph first
        nodes, edges = self.build_hierarchy_graph(test_inputs)

        # Organize nodes by abstraction level
        levels: dict[AbstractionLevel, list[HierarchyNode]] = defaultdict(list)
        for node in nodes:
            levels[node.abstraction].append(node)

        # Find key paths (chains from low to high abstraction)
        key_paths = self._find_key_paths(nodes, edges)

        # Compute abstraction increase
        layer_abstractions = {}
        for node in nodes:
            if node.layer not in layer_abstractions:
                layer_abstractions[node.layer] = []
            layer_abstractions[node.layer].append(node.abstraction.value)

        ladder = AbstractionLadder(
            levels=dict(levels),
            edges=edges,
            n_features=len(nodes),
            n_layers=self.manager.n_layers,
            key_paths=key_paths,
        )

        return ladder

    def _assign_abstraction_level(self, layer_idx: int) -> AbstractionLevel:
        """Assign abstraction level based on layer position."""
        n_layers = self.manager.n_layers
        fraction = layer_idx / max(n_layers - 1, 1)

        if fraction < 0.1:
            return AbstractionLevel.RAW
        elif fraction < 0.3:
            return AbstractionLevel.LOW
        elif fraction < 0.6:
            return AbstractionLevel.MID
        elif fraction < 0.85:
            return AbstractionLevel.HIGH
        else:
            return AbstractionLevel.ABSTRACT

    def _determine_relation_type(
        self,
        correlation: float,
        node_a: HierarchyNode,
        node_b: HierarchyNode,
        corr_matrix: torch.Tensor,
        feat_idx_a: int,
    ) -> FeatureRelationType:
        """Determine the type of relationship between features."""
        # Check if feature splits (one source, multiple targets)
        targets_from_a = (corr_matrix[feat_idx_a, :].abs() > self.correlation_threshold).sum().item()

        if targets_from_a > 2:
            return FeatureRelationType.SPLITS

        # Check if feature merges (multiple sources, one target)
        feat_idx_b = int(node_b.feature_id.split(".")[-1])
        if feat_idx_b < corr_matrix.shape[1]:
            sources_to_b = (corr_matrix[:, feat_idx_b].abs() > self.correlation_threshold).sum().item()
            if sources_to_b > 2:
                return FeatureRelationType.MERGES

        # High correlation = persists
        if abs(correlation) > 0.8:
            return FeatureRelationType.PERSISTS

        # Medium correlation = transforms
        if abs(correlation) > 0.5:
            return FeatureRelationType.TRANSFORMS

        # Low correlation = weak link
        return FeatureRelationType.TRANSFORMS

    def _find_key_paths(
        self,
        nodes: list[HierarchyNode],
        edges: list[HierarchyEdge],
        n_paths: int = 5,
    ) -> list[list[str]]:
        """Find most important paths through the hierarchy."""
        # Build adjacency list
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.source].append((edge.target, edge.weight))

        # Find starting nodes (early layers)
        start_nodes = [n.feature_id for n in nodes if n.layer <= 1]

        # Find ending nodes (late layers)
        end_nodes = [n.feature_id for n in nodes if n.layer >= self.manager.n_layers - 2]

        # Simple path finding (DFS with pruning)
        paths = []

        def dfs(current: str, path: list[str], score: float):
            if current in end_nodes:
                paths.append((path.copy(), score))
                return
            if len(path) > self.manager.n_layers:
                return

            for next_node, weight in adjacency.get(current, []):
                if next_node not in path:  # Avoid cycles
                    path.append(next_node)
                    dfs(next_node, path, score + weight)
                    path.pop()

        for start in start_nodes[:10]:  # Limit search
            dfs(start, [start], 0.0)

        # Sort by score and return top paths
        paths.sort(key=lambda x: x[1], reverse=True)
        return [p for p, s in paths[:n_paths]]

    def get_feature_at_layer(
        self,
        feature_id: str,
        target_layer: int,
    ) -> str | None:
        """
        Find the corresponding feature at a different layer.

        Args:
            feature_id: Source feature
            target_layer: Layer to find corresponding feature in

        Returns:
            Feature ID at target layer, or None if not found
        """
        # Get source layer
        source_layer = int(feature_id.split(".")[0].replace("layer", ""))

        if source_layer == target_layer:
            return feature_id

        # Look through edges
        current = feature_id
        direction = 1 if target_layer > source_layer else -1

        for _ in range(abs(target_layer - source_layer)):
            # Find edge from current
            found = None
            for edge in self._hierarchy_edges:
                if direction > 0 and edge.source == current:
                    found = edge.target
                    break
                elif direction < 0 and edge.target == current:
                    found = edge.source
                    break

            if found is None:
                return None
            current = found

        return current

    def get_hierarchy_summary(self) -> dict[str, Any]:
        """Get summary of hierarchy analysis."""
        return {
            "n_nodes": len(self._hierarchy_nodes),
            "n_edges": len(self._hierarchy_edges),
            "n_evolutions_tracked": len(self._feature_evolutions),
            "abstraction_distribution": {
                level.value: len([n for n in self._hierarchy_nodes.values() if n.abstraction == level])
                for level in AbstractionLevel
            },
            "edge_type_distribution": {
                rel.value: len([e for e in self._hierarchy_edges if e.relation_type == rel])
                for rel in FeatureRelationType
            },
        }

    # =============================================================================
    # Integration with CorrelationTracker and PropagationAnalyzer
    # =============================================================================

    def integrate_correlation_tracker(
        self,
        correlation_tracker: "CorrelationTracker",
    ) -> dict[tuple[int, int], list["CorrelationPair"]]:
        """
        Integrate with CorrelationTracker for streaming correlation updates.

        Uses Welford's online algorithm for memory-efficient correlation
        computation across layer pairs.

        Args:
            correlation_tracker: Initialized CorrelationTracker instance

        Returns:
            Dictionary mapping layer pairs to their top correlation pairs
        """
        from hololoom.dark_trace.multilayer.correlation_tracker import (
            CorrelationPair,
        )

        results: dict[tuple[int, int], list[CorrelationPair]] = {}

        # Get tracked layer pairs from correlation tracker
        for layer_pair in correlation_tracker.get_tracked_layer_pairs():
            # Get top correlations for this layer pair
            top_correlations = correlation_tracker.get_top_correlations(
                layer_pair[0],
                layer_pair[1],
                k=50,
            )
            results[layer_pair] = top_correlations

            # Update hierarchy edges with streaming correlation data
            for corr_pair in top_correlations:
                source_id = f"layer{layer_pair[0]}.sae.{corr_pair.feature_a}"
                target_id = f"layer{layer_pair[1]}.sae.{corr_pair.feature_b}"

                # Check if edge already exists
                existing_edge = None
                for edge in self._hierarchy_edges:
                    if edge.source == source_id and edge.target == target_id:
                        existing_edge = edge
                        break

                if existing_edge:
                    # Update with streaming correlation
                    existing_edge.correlation = corr_pair.correlation
                    existing_edge.weight = abs(corr_pair.correlation)
                elif abs(corr_pair.correlation) >= self.correlation_threshold:
                    # Create new edge
                    new_edge = HierarchyEdge(
                        source=source_id,
                        target=target_id,
                        source_layer=layer_pair[0],
                        target_layer=layer_pair[1],
                        correlation=corr_pair.correlation,
                        weight=abs(corr_pair.correlation),
                    )
                    self._hierarchy_edges.append(new_edge)

        return results

    def trace_with_propagation_analyzer(
        self,
        propagation_analyzer: "PropagationAnalyzer",
        feature_id: str,
        direction: str = "forward",
        max_hops: int = 5,
    ) -> list["PropagationPath"]:
        """
        Trace feature propagation using PropagationAnalyzer.

        Finds how a feature's influence propagates through layers
        using the more sophisticated propagation analysis.

        Args:
            propagation_analyzer: Initialized PropagationAnalyzer instance
            feature_id: Feature to trace (e.g., "layer0.sae.42")
            direction: "forward" (early→late) or "backward" (late→early)
            max_hops: Maximum number of layer hops to trace

        Returns:
            List of PropagationPath objects showing feature paths
        """

        # Parse feature ID
        parts = feature_id.split(".")
        if len(parts) >= 3 and parts[0].startswith("layer"):
            layer = int(parts[0].replace("layer", ""))
            feature_idx = int(parts[-1])
        else:
            layer = 0
            feature_idx = int(parts[-1]) if parts[-1].isdigit() else 0

        # Use propagation analyzer to find paths
        if direction == "forward":
            paths = propagation_analyzer.trace_forward(
                layer=layer,
                feature_idx=feature_idx,
                max_hops=max_hops,
            )
        else:
            paths = propagation_analyzer.trace_backward(
                layer=layer,
                feature_idx=feature_idx,
                max_hops=max_hops,
            )

        # Update feature evolution with propagation data
        if feature_id in self._feature_evolutions:
            evolution = self._feature_evolutions[feature_id]
            for path in paths:
                if direction == "forward":
                    for i, (layer_idx, feat_idx) in enumerate(path.path[1:], 1):
                        successor_id = f"layer{layer_idx}.sae.{feat_idx}"
                        if successor_id not in [s[0] for s in evolution.successors]:
                            evolution.successors.append(
                                (successor_id, path.strength / (i + 1))
                            )
                else:
                    for i, (layer_idx, feat_idx) in enumerate(path.path[:-1]):
                        predecessor_id = f"layer{layer_idx}.sae.{feat_idx}"
                        if predecessor_id not in [p[0] for p in evolution.predecessors]:
                            evolution.predecessors.append(
                                (predecessor_id, path.strength / (i + 1))
                            )

        return paths

    def discover_circuits_with_propagation(
        self,
        propagation_analyzer: "PropagationAnalyzer",
        min_features: int = 3,
        min_strength: float = 0.3,
    ) -> list[Any]:
        """
        Discover circuits (connected feature subgraphs) using propagation analyzer.

        Finds groups of features that work together across layers,
        similar to ACDC circuit discovery.

        Args:
            propagation_analyzer: Initialized PropagationAnalyzer instance
            min_features: Minimum features in a circuit
            min_strength: Minimum connection strength

        Returns:
            List of CircuitCandidate objects
        """
        # Use propagation analyzer's circuit discovery
        circuits = propagation_analyzer.discover_circuits(
            min_features=min_features,
            min_strength=min_strength,
        )

        # Enrich circuits with hierarchy data
        for circuit in circuits:
            # Add abstraction level information
            circuit.metadata = circuit.metadata or {}
            circuit.metadata["abstraction_levels"] = []

            for layer, feat_idx in circuit.features:
                feature_id = f"layer{layer}.sae.{feat_idx}"
                if feature_id in self._hierarchy_nodes:
                    node = self._hierarchy_nodes[feature_id]
                    circuit.metadata["abstraction_levels"].append(
                        (feature_id, node.abstraction.value)
                    )

            # Determine if circuit spans abstraction levels
            if circuit.metadata["abstraction_levels"]:
                levels = [lvl for _, lvl in circuit.metadata["abstraction_levels"]]
                unique_levels = set(levels)
                circuit.metadata["spans_abstraction"] = len(unique_levels) > 1
                circuit.metadata["abstraction_span"] = (min(levels), max(levels))

        return circuits

    def get_streaming_correlation_update(
        self,
        activations_a: torch.Tensor,
        activations_b: torch.Tensor,
        layer_a: int,
        layer_b: int,
    ) -> dict[str, Any]:
        """
        Get a single streaming correlation update for incremental learning.

        Useful for online learning scenarios where we process one
        batch at a time without storing all data.

        Args:
            activations_a: Features from layer A [batch, n_features_a]
            activations_b: Features from layer B [batch, n_features_b]
            layer_a: Source layer index
            layer_b: Target layer index

        Returns:
            Dictionary with correlation update statistics
        """
        from hololoom.dark_trace.multilayer.correlation_tracker import (
            WelfordCorrelator,
        )

        # Create or get Welford correlator for this layer pair
        key = (layer_a, layer_b)
        if not hasattr(self, "_welford_correlators"):
            self._welford_correlators: dict[tuple[int, int], WelfordCorrelator] = {}

        if key not in self._welford_correlators:
            n_features_a = activations_a.shape[-1]
            n_features_b = activations_b.shape[-1]
            self._welford_correlators[key] = WelfordCorrelator(
                n_features_a, n_features_b
            )

        correlator = self._welford_correlators[key]

        # Update with new batch
        correlator.update(activations_a, activations_b)

        # Get current correlation matrix
        corr_matrix = correlator.get_correlation()

        # Find top correlations above threshold
        high_corr_mask = corr_matrix.abs() > self.correlation_threshold
        high_corr_indices = torch.nonzero(high_corr_mask, as_tuple=False)

        top_pairs = []
        for idx in high_corr_indices[:100]:  # Limit to top 100
            feat_a, feat_b = idx[0].item(), idx[1].item()
            corr = corr_matrix[feat_a, feat_b].item()
            top_pairs.append({
                "feature_a": feat_a,
                "feature_b": feat_b,
                "correlation": corr,
            })

        return {
            "layer_pair": (layer_a, layer_b),
            "n_updates": correlator.n,
            "n_high_correlations": len(top_pairs),
            "top_pairs": sorted(top_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:20],
            "mean_abs_correlation": corr_matrix.abs().mean().item(),
        }
