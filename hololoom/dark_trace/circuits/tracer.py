from __future__ import annotations
"""
Circuit Tracing for Mechanistic Interpretability

Traces computational circuits through model internals:
- Forward tracing (input → features → output)
- Backward tracing (output → features → input)
- Bidirectional analysis

Research Basis:
- ACDC (Conmy et al., 2023): Automated Circuit Discovery
- Causal Tracing (Meng et al., 2022): Locating factual knowledge
- Path Patching: Isolating causal pathways
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class TraceDirection(Enum):
    """Direction of circuit tracing."""
    FORWARD = "forward"  # Input → Output
    BACKWARD = "backward"  # Output → Input
    BIDIRECTIONAL = "bidirectional"  # Both directions


class ComponentType(Enum):
    """Type of model component."""
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    MLP = "mlp"
    LAYER_NORM = "layer_norm"
    RESIDUAL = "residual"
    UNEMBEDDING = "unembedding"
    SAE_FEATURE = "sae_feature"
    UNKNOWN = "unknown"


@dataclass
class TracerConfig:
    """Configuration for circuit tracing."""

    # Tracing settings
    direction: TraceDirection = TraceDirection.FORWARD
    include_residual: bool = True
    include_attention: bool = True
    include_mlp: bool = True

    # Thresholds
    min_activation: float = 0.01  # Minimum activation to include
    min_effect_size: float = 0.1  # Minimum effect size for edges
    max_nodes_per_layer: int = 100  # Limit nodes per layer

    # Attribution method
    use_integrated_gradients: bool = False
    n_ig_steps: int = 50  # Steps for integrated gradients

    # Patching settings
    enable_patching: bool = True
    patch_noise_std: float = 0.1

    # Performance
    batch_size: int = 32
    use_cache: bool = True

    @classmethod
    def default(cls) -> "TracerConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def fast(cls) -> "TracerConfig":
        """Fast tracing with reduced computation."""
        return cls(
            use_integrated_gradients=False,
            max_nodes_per_layer=50,
            batch_size=64
        )

    @classmethod
    def thorough(cls) -> "TracerConfig":
        """Thorough tracing with all methods."""
        return cls(
            direction=TraceDirection.BIDIRECTIONAL,
            use_integrated_gradients=True,
            n_ig_steps=100,
            min_effect_size=0.05,
            max_nodes_per_layer=200
        )


@dataclass
class TraceNode:
    """Node in a circuit trace."""

    node_id: str
    layer_idx: int
    component_type: ComponentType
    position: int | None = None  # Token position

    # Activation info
    activation: float = 0.0
    activation_std: float = 0.0

    # Attribution
    importance: float = 0.0  # Importance to output
    gradient_norm: float = 0.0

    # Metadata
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "layer_idx": self.layer_idx,
            "component_type": self.component_type.value,
            "position": self.position,
            "activation": self.activation,
            "activation_std": self.activation_std,
            "importance": self.importance,
            "gradient_norm": self.gradient_norm,
            "label": self.label,
            "metadata": self.metadata
        }


@dataclass
class TraceEdge:
    """Edge in a circuit trace (causal connection)."""

    source_id: str
    target_id: str

    # Edge strength
    weight: float = 0.0  # Correlation or causal effect
    effect_size: float = 0.0  # Direct causal effect

    # Edge type
    is_direct: bool = True  # Direct vs mediated
    is_causal: bool = False  # Causally verified via patching

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "effect_size": self.effect_size,
            "is_direct": self.is_direct,
            "is_causal": self.is_causal,
            "metadata": self.metadata
        }


@dataclass
class CircuitTrace:
    """Complete circuit trace through the model."""

    # Trace content
    nodes: list[TraceNode] = field(default_factory=list)
    edges: list[TraceEdge] = field(default_factory=list)

    # Trace metadata
    direction: TraceDirection = TraceDirection.FORWARD
    start_layer: int = 0
    end_layer: int = -1

    # Input/output info
    input_tokens: list[int] | None = None
    target_position: int | None = None
    target_token: int | None = None

    # Statistics
    total_activation: float = 0.0
    total_effect: float = 0.0
    n_layers_traced: int = 0

    def get_nodes_at_layer(self, layer_idx: int) -> list[TraceNode]:
        """Get all nodes at a specific layer."""
        return [n for n in self.nodes if n.layer_idx == layer_idx]

    def get_node_by_id(self, node_id: str) -> TraceNode | None:
        """Get node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_edges_from(self, node_id: str) -> list[TraceEdge]:
        """Get all edges from a node."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_edges_to(self, node_id: str) -> list[TraceEdge]:
        """Get all edges to a node."""
        return [e for e in self.edges if e.target_id == node_id]

    def get_critical_path(self) -> list[TraceNode]:
        """Get the path with highest total importance."""
        if not self.nodes:
            return []

        # Group by layer
        layers: dict[int, list[TraceNode]] = {}
        for node in self.nodes:
            if node.layer_idx not in layers:
                layers[node.layer_idx] = []
            layers[node.layer_idx].append(node)

        # Find most important node at each layer
        path = []
        for layer_idx in sorted(layers.keys()):
            layer_nodes = layers[layer_idx]
            if layer_nodes:
                best = max(layer_nodes, key=lambda n: n.importance)
                path.append(best)

        return path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "direction": self.direction.value,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "input_tokens": self.input_tokens,
            "target_position": self.target_position,
            "target_token": self.target_token,
            "total_activation": self.total_activation,
            "total_effect": self.total_effect,
            "n_layers_traced": self.n_layers_traced
        }


class CircuitTracer:
    """
    Traces circuits through model internals.

    Implements:
    - Forward tracing with activation recording
    - Backward tracing with gradient attribution
    - Causal verification via activation patching
    - SAE feature integration

    Usage:
        tracer = CircuitTracer(model, probe, config=TracerConfig.default())

        # Forward trace
        trace = tracer.trace_forward(input_ids, target_position=-1)

        # Backward trace from output
        trace = tracer.trace_backward(input_ids, target_token_id=42)

        # Bidirectional
        trace = tracer.trace_bidirectional(input_ids, target_position=-1)

        # Get critical circuit
        path = trace.get_critical_path()
    """

    def __init__(
        self,
        model: Any,
        probe: Any = None,  # MindProbe for hook access
        sae_manager: Any = None,  # LayerSAEManager for feature tracing
        config: TracerConfig | None = None
    ):
        """
        Initialize circuit tracer.

        Args:
            model: The model to trace
            probe: MindProbe for activation access
            sae_manager: Optional SAE manager for feature-level tracing
            config: Tracer configuration
        """
        self.model = model
        self.probe = probe
        self.sae_manager = sae_manager
        self.config = config or TracerConfig.default()

        # Cache
        self._activation_cache: dict[str, Any] = {}
        self._gradient_cache: dict[str, Any] = {}

    def trace_forward(
        self,
        input_ids: Any,
        target_position: int = -1,
        layers: list[int] | None = None
    ) -> CircuitTrace:
        """
        Trace circuit forward from input to output.

        Args:
            input_ids: Input token IDs
            target_position: Position to trace to (default: last token)
            layers: Specific layers to trace (default: all)

        Returns:
            CircuitTrace with nodes and edges
        """
        trace = CircuitTrace(
            direction=TraceDirection.FORWARD,
            target_position=target_position
        )

        if TORCH_AVAILABLE and isinstance(input_ids, torch.Tensor):
            trace.input_tokens = input_ids.tolist() if input_ids.dim() == 1 else input_ids[0].tolist()

        # Determine layers
        n_layers = self._get_n_layers()
        if layers is None:
            layers = list(range(n_layers))

        trace.start_layer = min(layers)
        trace.end_layer = max(layers)
        trace.n_layers_traced = len(layers)

        # Clear cache
        if not self.config.use_cache:
            self._activation_cache.clear()

        # Record activations at each layer
        activations = self._record_activations(input_ids, layers)

        # Build trace nodes
        for layer_idx in layers:
            layer_nodes = self._build_layer_nodes(
                layer_idx, activations, target_position
            )
            trace.nodes.extend(layer_nodes)

        # Build edges between consecutive layers
        for i in range(len(layers) - 1):
            source_layer = layers[i]
            target_layer = layers[i + 1]
            layer_edges = self._build_layer_edges(
                source_layer, target_layer, activations, trace.nodes
            )
            trace.edges.extend(layer_edges)

        # Compute importance via gradient if enabled
        if self.config.use_integrated_gradients:
            self._compute_integrated_gradients(trace, input_ids, target_position)
        else:
            self._compute_gradient_importance(trace, input_ids, target_position)

        # Compute statistics
        trace.total_activation = sum(n.activation for n in trace.nodes)
        trace.total_effect = sum(e.effect_size for e in trace.edges)

        return trace

    def trace_backward(
        self,
        input_ids: Any,
        target_token_id: int | None = None,
        target_position: int = -1,
        layers: list[int] | None = None
    ) -> CircuitTrace:
        """
        Trace circuit backward from output to input.

        Args:
            input_ids: Input token IDs
            target_token_id: Optional target token to trace from
            target_position: Position to trace from
            layers: Specific layers to trace

        Returns:
            CircuitTrace with backward attribution
        """
        trace = CircuitTrace(
            direction=TraceDirection.BACKWARD,
            target_position=target_position,
            target_token=target_token_id
        )

        if TORCH_AVAILABLE and isinstance(input_ids, torch.Tensor):
            trace.input_tokens = input_ids.tolist() if input_ids.dim() == 1 else input_ids[0].tolist()

        # Determine layers (reverse order)
        n_layers = self._get_n_layers()
        if layers is None:
            layers = list(range(n_layers))

        layers_reversed = sorted(layers, reverse=True)
        trace.start_layer = max(layers)
        trace.end_layer = min(layers)
        trace.n_layers_traced = len(layers)

        # Compute gradients
        gradients = self._compute_gradients(input_ids, target_position, target_token_id)

        # Build nodes with gradient attribution
        for layer_idx in layers_reversed:
            layer_nodes = self._build_layer_nodes_gradient(
                layer_idx, gradients, target_position
            )
            trace.nodes.extend(layer_nodes)

        # Build edges (backward direction)
        for i in range(len(layers_reversed) - 1):
            source_layer = layers_reversed[i]
            target_layer = layers_reversed[i + 1]
            layer_edges = self._build_layer_edges_gradient(
                source_layer, target_layer, gradients, trace.nodes
            )
            trace.edges.extend(layer_edges)

        return trace

    def trace_bidirectional(
        self,
        input_ids: Any,
        target_position: int = -1,
        layers: list[int] | None = None
    ) -> CircuitTrace:
        """
        Trace circuit in both directions and combine.

        Args:
            input_ids: Input token IDs
            target_position: Position to trace to/from
            layers: Specific layers to trace

        Returns:
            Combined CircuitTrace
        """
        # Forward trace
        forward = self.trace_forward(input_ids, target_position, layers)

        # Backward trace
        backward = self.trace_backward(input_ids, None, target_position, layers)

        # Combine traces
        combined = CircuitTrace(
            direction=TraceDirection.BIDIRECTIONAL,
            target_position=target_position
        )

        # Merge nodes (taking max importance)
        node_map: dict[str, TraceNode] = {}
        for node in forward.nodes + backward.nodes:
            if node.node_id in node_map:
                existing = node_map[node.node_id]
                existing.importance = max(existing.importance, node.importance)
                existing.gradient_norm = max(existing.gradient_norm, node.gradient_norm)
            else:
                node_map[node.node_id] = node

        combined.nodes = list(node_map.values())

        # Merge edges
        edge_map: dict[tuple[str, str], TraceEdge] = {}
        for edge in forward.edges + backward.edges:
            key = (edge.source_id, edge.target_id)
            if key in edge_map:
                existing = edge_map[key]
                existing.weight = max(existing.weight, edge.weight)
                existing.effect_size = max(existing.effect_size, edge.effect_size)
            else:
                edge_map[key] = edge

        combined.edges = list(edge_map.values())

        combined.start_layer = min(forward.start_layer, backward.end_layer)
        combined.end_layer = max(forward.end_layer, backward.start_layer)
        combined.n_layers_traced = forward.n_layers_traced
        combined.input_tokens = forward.input_tokens

        return combined

    def verify_causal(
        self,
        trace: CircuitTrace,
        input_ids: Any,
        target_position: int = -1,
        n_samples: int = 100
    ) -> CircuitTrace:
        """
        Verify causal connections in trace via patching.

        Args:
            trace: Existing trace to verify
            input_ids: Input token IDs
            target_position: Target position
            n_samples: Number of patching samples

        Returns:
            Updated trace with causal verification
        """
        if not self.config.enable_patching:
            return trace

        # Get baseline output
        baseline_output = self._get_output(input_ids, target_position)

        # Verify each edge
        for edge in trace.edges:
            effect = self._patch_and_measure(
                input_ids, edge.source_id, edge.target_id,
                baseline_output, target_position, n_samples
            )

            edge.effect_size = abs(effect)
            edge.is_causal = edge.effect_size > self.config.min_effect_size

        # Update node importance based on causal effects
        node_effects: dict[str, float] = {}
        for edge in trace.edges:
            if edge.is_causal:
                node_effects[edge.source_id] = node_effects.get(edge.source_id, 0) + edge.effect_size

        for node in trace.nodes:
            if node.node_id in node_effects:
                node.importance = node_effects[node.node_id]

        return trace

    def _get_n_layers(self) -> int:
        """Get number of model layers."""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'n_layer'):
            return self.model.config.n_layer
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        if hasattr(self.model, 'n_layers'):
            return self.model.n_layers
        return 12  # Default

    def _record_activations(
        self,
        input_ids: Any,
        layers: list[int]
    ) -> dict[int, Any]:
        """Record activations at specified layers."""
        activations = {}

        if self.probe is not None:
            # Use probe to extract activations
            for layer_idx in layers:
                if hasattr(self.probe, 'extract_layer'):
                    activations[layer_idx] = self.probe.extract_layer(input_ids, layer_idx)
                else:
                    # Fallback: Run forward and cache
                    pass
        elif self.sae_manager is not None:
            # Use SAE manager
            multi_features = self.sae_manager.encode_all_layers(input_ids, layers)
            for layer_idx in layers:
                if layer_idx in multi_features.layer_features:
                    activations[layer_idx] = multi_features.layer_features[layer_idx]
        else:
            # Placeholder - would need model-specific hooks
            for layer_idx in layers:
                activations[layer_idx] = None

        return activations

    def _build_layer_nodes(
        self,
        layer_idx: int,
        activations: dict[int, Any],
        target_position: int
    ) -> list[TraceNode]:
        """Build trace nodes for a layer."""
        nodes = []

        layer_acts = activations.get(layer_idx)
        if layer_acts is None:
            return nodes

        if TORCH_AVAILABLE and isinstance(layer_acts, torch.Tensor):
            acts_np = layer_acts.detach().cpu().numpy()
        else:
            acts_np = np.array(layer_acts) if layer_acts is not None else np.array([])

        if acts_np.size == 0:
            return nodes

        # Handle different shapes
        if acts_np.ndim == 3:
            # [batch, seq, hidden]
            if target_position < 0:
                target_position = acts_np.shape[1] + target_position
            pos_acts = acts_np[0, target_position, :]
        elif acts_np.ndim == 2:
            # [seq, hidden] or [batch, hidden]
            pos_acts = acts_np[-1, :] if acts_np.shape[0] > acts_np.shape[1] else acts_np[0, :]
        else:
            pos_acts = acts_np.flatten()

        # Find top activating features
        top_k = min(self.config.max_nodes_per_layer, len(pos_acts))
        top_indices = np.argsort(np.abs(pos_acts))[-top_k:][::-1]

        for idx in top_indices:
            act_value = float(pos_acts[idx])
            if abs(act_value) < self.config.min_activation:
                continue

            node = TraceNode(
                node_id=f"L{layer_idx}.{idx}",
                layer_idx=layer_idx,
                component_type=ComponentType.SAE_FEATURE if self.sae_manager else ComponentType.UNKNOWN,
                position=target_position,
                activation=abs(act_value),
                label=f"Feature {idx}"
            )
            nodes.append(node)

        return nodes

    def _build_layer_nodes_gradient(
        self,
        layer_idx: int,
        gradients: dict[int, Any],
        target_position: int
    ) -> list[TraceNode]:
        """Build trace nodes from gradients."""
        nodes = []

        layer_grads = gradients.get(layer_idx)
        if layer_grads is None:
            return nodes

        if TORCH_AVAILABLE and isinstance(layer_grads, torch.Tensor):
            grads_np = layer_grads.detach().cpu().numpy()
        else:
            grads_np = np.array(layer_grads) if layer_grads is not None else np.array([])

        if grads_np.size == 0:
            return nodes

        # Flatten and get top gradients
        flat_grads = np.abs(grads_np.flatten())
        top_k = min(self.config.max_nodes_per_layer, len(flat_grads))
        top_indices = np.argsort(flat_grads)[-top_k:][::-1]

        for idx in top_indices:
            grad_value = float(flat_grads[idx])
            if grad_value < self.config.min_activation:
                continue

            node = TraceNode(
                node_id=f"L{layer_idx}.{idx}",
                layer_idx=layer_idx,
                component_type=ComponentType.UNKNOWN,
                gradient_norm=grad_value,
                importance=grad_value
            )
            nodes.append(node)

        return nodes

    def _build_layer_edges(
        self,
        source_layer: int,
        target_layer: int,
        activations: dict[int, Any],
        nodes: list[TraceNode]
    ) -> list[TraceEdge]:
        """Build edges between layers."""
        edges = []

        source_nodes = [n for n in nodes if n.layer_idx == source_layer]
        target_nodes = [n for n in nodes if n.layer_idx == target_layer]

        if not source_nodes or not target_nodes:
            return edges

        source_acts = activations.get(source_layer)
        target_acts = activations.get(target_layer)

        if source_acts is None or target_acts is None:
            # Create edges based on activation similarity
            for src in source_nodes:
                for tgt in target_nodes:
                    # Simple heuristic: connect high-activation nodes
                    weight = (src.activation * tgt.activation) ** 0.5
                    if weight > self.config.min_effect_size:
                        edges.append(TraceEdge(
                            source_id=src.node_id,
                            target_id=tgt.node_id,
                            weight=weight
                        ))
            return edges

        # Compute correlations between source and target activations
        if TORCH_AVAILABLE:
            if isinstance(source_acts, torch.Tensor):
                source_np = source_acts.detach().cpu().numpy().flatten()
            else:
                source_np = np.array(source_acts).flatten()
            if isinstance(target_acts, torch.Tensor):
                target_np = target_acts.detach().cpu().numpy().flatten()
            else:
                target_np = np.array(target_acts).flatten()
        else:
            source_np = np.array(source_acts).flatten()
            target_np = np.array(target_acts).flatten()

        # Sample edges based on correlation
        for src in source_nodes[:20]:  # Limit for performance
            src_idx = int(src.node_id.split('.')[-1])
            if src_idx >= len(source_np):
                continue

            for tgt in target_nodes[:20]:
                tgt_idx = int(tgt.node_id.split('.')[-1])
                if tgt_idx >= len(target_np):
                    continue

                # Simple correlation-based weight
                weight = abs(float(source_np[src_idx] * target_np[tgt_idx]))
                weight = min(1.0, weight)

                if weight > self.config.min_effect_size:
                    edges.append(TraceEdge(
                        source_id=src.node_id,
                        target_id=tgt.node_id,
                        weight=weight
                    ))

        return edges

    def _build_layer_edges_gradient(
        self,
        source_layer: int,
        target_layer: int,
        gradients: dict[int, Any],
        nodes: list[TraceNode]
    ) -> list[TraceEdge]:
        """Build edges from gradients."""
        edges = []

        source_nodes = [n for n in nodes if n.layer_idx == source_layer]
        target_nodes = [n for n in nodes if n.layer_idx == target_layer]

        # Connect based on gradient magnitude
        for src in source_nodes[:20]:
            for tgt in target_nodes[:20]:
                weight = (src.gradient_norm * tgt.gradient_norm) ** 0.5
                if weight > self.config.min_effect_size:
                    edges.append(TraceEdge(
                        source_id=src.node_id,
                        target_id=tgt.node_id,
                        weight=weight
                    ))

        return edges

    def _compute_gradients(
        self,
        input_ids: Any,
        target_position: int,
        target_token_id: int | None
    ) -> dict[int, Any]:
        """Compute gradients for backward tracing."""
        gradients = {}

        if not TORCH_AVAILABLE:
            return gradients

        # Would need model-specific implementation
        # Placeholder returning empty gradients
        return gradients

    def _compute_gradient_importance(
        self,
        trace: CircuitTrace,
        input_ids: Any,
        target_position: int
    ) -> None:
        """Compute importance via simple gradients."""
        # Simplified: use activation magnitude as proxy for importance
        for node in trace.nodes:
            node.importance = node.activation

    def _compute_integrated_gradients(
        self,
        trace: CircuitTrace,
        input_ids: Any,
        target_position: int
    ) -> None:
        """Compute importance via integrated gradients."""
        # Would need model-specific implementation
        # Fallback to activation-based importance
        self._compute_gradient_importance(trace, input_ids, target_position)

    def _get_output(
        self,
        input_ids: Any,
        target_position: int
    ) -> Any:
        """Get model output at target position."""
        if not TORCH_AVAILABLE or self.model is None:
            return None

        # Placeholder - would need actual forward pass
        return None

    def _patch_and_measure(
        self,
        input_ids: Any,
        source_id: str,
        target_id: str,
        baseline_output: Any,
        target_position: int,
        n_samples: int
    ) -> float:
        """Patch activation and measure effect."""
        # Placeholder - would need actual patching implementation
        return 0.0


# Convenience function
def trace_circuit(
    model: Any,
    input_ids: Any,
    probe: Any = None,
    config: TracerConfig | None = None,
    direction: TraceDirection = TraceDirection.FORWARD
) -> CircuitTrace:
    """
    Convenience function to trace a circuit.

    Args:
        model: The model to trace
        input_ids: Input token IDs
        probe: Optional MindProbe
        config: Optional tracer config
        direction: Tracing direction

    Returns:
        CircuitTrace
    """
    tracer = CircuitTracer(model, probe, config=config)

    if direction == TraceDirection.FORWARD:
        return tracer.trace_forward(input_ids)
    elif direction == TraceDirection.BACKWARD:
        return tracer.trace_backward(input_ids)
    else:
        return tracer.trace_bidirectional(input_ids)
