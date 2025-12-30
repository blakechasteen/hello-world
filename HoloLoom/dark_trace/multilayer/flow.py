"""
Information Flow Analysis for Multi-Layer SAE

Analyzes how information flows across layers:
- Layer-to-layer feature mapping
- Bottleneck detection
- Skip connection / residual stream analysis
- Information theoretic metrics

Research Basis:
- Residual Stream Analysis (Anthropic): Information accumulation patterns
- Logit Lens (nostalgebraist): Layer-wise prediction analysis
- Information Bottleneck Theory (Tishby): Compression-prediction tradeoff
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, TYPE_CHECKING
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

if TYPE_CHECKING:
    from HoloLoom.dark_trace.multilayer.causal_metrics import (
        CausalStrengthEstimator,
        TransferEntropyEstimator,
        CausalStrengthResult,
        TransferEntropyResult,
    )


class FlowDirection(Enum):
    """Direction of information flow."""
    FORWARD = "forward"  # Layer N -> N+1
    BACKWARD = "backward"  # Layer N+1 -> N (for attribution)
    BIDIRECTIONAL = "bidirectional"  # Both directions


class BottleneckType(Enum):
    """Types of information bottlenecks."""
    COMPRESSION = "compression"  # Information is compressed
    EXPANSION = "expansion"  # Information expands
    SATURATION = "saturation"  # Layer is saturated
    DROPOUT = "dropout"  # Information is lost
    REDUNDANCY = "redundancy"  # High redundancy between features


@dataclass
class FlowConfig:
    """Configuration for information flow analysis."""

    # Analysis settings
    direction: FlowDirection = FlowDirection.FORWARD
    analyze_residual: bool = True
    analyze_attention: bool = True
    analyze_mlp: bool = True

    # Bottleneck detection
    detect_bottlenecks: bool = True
    bottleneck_threshold: float = 0.3  # Threshold for bottleneck detection
    compression_ratio_threshold: float = 0.5  # Below this = compression bottleneck
    saturation_threshold: float = 0.95  # Above this = saturation

    # Information metrics
    compute_mutual_info: bool = True
    compute_entropy: bool = True
    n_bins: int = 50  # Bins for entropy estimation

    # Skip connection analysis
    analyze_skip_connections: bool = True
    skip_connection_layers: Optional[List[Tuple[int, int]]] = None  # (from, to) pairs

    # Sampling
    n_samples: int = 1000
    batch_size: int = 32

    @classmethod
    def default(cls) -> "FlowConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def fast(cls) -> "FlowConfig":
        """Fast analysis with reduced computation."""
        return cls(
            compute_mutual_info=False,
            n_samples=500,
            batch_size=64
        )

    @classmethod
    def thorough(cls) -> "FlowConfig":
        """Thorough analysis with all metrics."""
        return cls(
            direction=FlowDirection.BIDIRECTIONAL,
            compute_mutual_info=True,
            compute_entropy=True,
            n_samples=2000,
            n_bins=100
        )


@dataclass
class LayerFlow:
    """Information flow at a single layer."""

    layer_idx: int

    # Feature statistics
    n_active_features: int = 0
    mean_activation: float = 0.0
    activation_entropy: float = 0.0
    sparsity: float = 0.0  # Fraction of zero activations

    # Information metrics
    total_information: float = 0.0  # Estimated bits
    redundancy: float = 0.0  # Mutual info between features

    # Component contributions (if available)
    attention_contribution: float = 0.0
    mlp_contribution: float = 0.0
    residual_contribution: float = 0.0

    # Capacity utilization
    capacity_used: float = 0.0  # Fraction of capacity used

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_idx": self.layer_idx,
            "n_active_features": self.n_active_features,
            "mean_activation": self.mean_activation,
            "activation_entropy": self.activation_entropy,
            "sparsity": self.sparsity,
            "total_information": self.total_information,
            "redundancy": self.redundancy,
            "attention_contribution": self.attention_contribution,
            "mlp_contribution": self.mlp_contribution,
            "residual_contribution": self.residual_contribution,
            "capacity_used": self.capacity_used
        }


@dataclass
class CrossLayerFlow:
    """Information flow between two layers."""

    source_layer: int
    target_layer: int

    # Flow metrics
    mutual_information: float = 0.0  # I(source; target)
    transfer_entropy: float = 0.0  # Directional information transfer
    flow_strength: float = 0.0  # Overall flow strength [0, 1]

    # Feature mapping
    n_preserved_features: int = 0  # Features that persist
    n_new_features: int = 0  # New features at target
    n_lost_features: int = 0  # Features lost from source

    # Transformation type
    is_compression: bool = False
    is_expansion: bool = False
    compression_ratio: float = 1.0  # < 1 = compression, > 1 = expansion

    # Top feature mappings (source_feature_id, target_feature_id, strength)
    top_mappings: List[Tuple[str, str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "mutual_information": self.mutual_information,
            "transfer_entropy": self.transfer_entropy,
            "flow_strength": self.flow_strength,
            "n_preserved_features": self.n_preserved_features,
            "n_new_features": self.n_new_features,
            "n_lost_features": self.n_lost_features,
            "is_compression": self.is_compression,
            "is_expansion": self.is_expansion,
            "compression_ratio": self.compression_ratio,
            "top_mappings": self.top_mappings[:10]  # Limit for serialization
        }


@dataclass
class FlowBottleneck:
    """Detected information bottleneck."""

    layer_idx: int
    bottleneck_type: BottleneckType
    severity: float  # 0 = mild, 1 = severe

    # Metrics
    information_before: float = 0.0
    information_after: float = 0.0
    information_loss: float = 0.0

    # Affected features
    affected_features: List[str] = field(default_factory=list)
    n_affected: int = 0

    # Description
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_idx": self.layer_idx,
            "bottleneck_type": self.bottleneck_type.value,
            "severity": self.severity,
            "information_before": self.information_before,
            "information_after": self.information_after,
            "information_loss": self.information_loss,
            "affected_features": self.affected_features[:20],
            "n_affected": self.n_affected,
            "description": self.description
        }


@dataclass
class ResidualContribution:
    """Contribution of residual stream at each layer."""

    layer_idx: int

    # Contribution breakdown
    residual_norm: float = 0.0  # ||residual||
    attention_delta_norm: float = 0.0  # ||attention output||
    mlp_delta_norm: float = 0.0  # ||MLP output||

    # Relative contributions
    residual_fraction: float = 0.0  # residual / total
    attention_fraction: float = 0.0
    mlp_fraction: float = 0.0

    # Cumulative from layer 0
    cumulative_residual: float = 0.0

    # Direct effect on output
    logit_attribution: float = 0.0  # How much this layer affects logits

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_idx": self.layer_idx,
            "residual_norm": self.residual_norm,
            "attention_delta_norm": self.attention_delta_norm,
            "mlp_delta_norm": self.mlp_delta_norm,
            "residual_fraction": self.residual_fraction,
            "attention_fraction": self.attention_fraction,
            "mlp_fraction": self.mlp_fraction,
            "cumulative_residual": self.cumulative_residual,
            "logit_attribution": self.logit_attribution
        }


@dataclass
class FlowAnalysisResult:
    """Complete flow analysis result."""

    # Per-layer flows
    layer_flows: Dict[int, LayerFlow] = field(default_factory=dict)

    # Cross-layer flows
    cross_layer_flows: Dict[Tuple[int, int], CrossLayerFlow] = field(default_factory=dict)

    # Bottlenecks
    bottlenecks: List[FlowBottleneck] = field(default_factory=list)

    # Residual contributions
    residual_contributions: Dict[int, ResidualContribution] = field(default_factory=dict)

    # Summary statistics
    total_information_preserved: float = 0.0
    overall_compression_ratio: float = 1.0
    bottleneck_layer: Optional[int] = None  # Most severe bottleneck

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_flows": {k: v.to_dict() for k, v in self.layer_flows.items()},
            "cross_layer_flows": {
                f"{k[0]}->{k[1]}": v.to_dict()
                for k, v in self.cross_layer_flows.items()
            },
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "residual_contributions": {
                k: v.to_dict() for k, v in self.residual_contributions.items()
            },
            "total_information_preserved": self.total_information_preserved,
            "overall_compression_ratio": self.overall_compression_ratio,
            "bottleneck_layer": self.bottleneck_layer
        }


class InformationFlowAnalyzer:
    """
    Analyzes information flow across model layers using SAE features.

    Implements:
    - Layer-to-layer feature mapping
    - Information theoretic metrics
    - Bottleneck detection
    - Residual stream analysis

    Usage:
        manager = LayerSAEManager(model, n_layers=12)
        flow_analyzer = InformationFlowAnalyzer(manager, config=FlowConfig.default())

        result = flow_analyzer.analyze(test_inputs)
        print(f"Bottleneck at layer {result.bottleneck_layer}")

        for bottleneck in result.bottlenecks:
            print(f"Layer {bottleneck.layer_idx}: {bottleneck.description}")
    """

    def __init__(
        self,
        layer_sae_manager: Any,  # LayerSAEManager
        config: Optional[FlowConfig] = None
    ):
        """
        Initialize flow analyzer.

        Args:
            layer_sae_manager: LayerSAEManager with trained SAEs
            config: Flow analysis configuration
        """
        self.manager = layer_sae_manager
        self.config = config or FlowConfig.default()

    def analyze(
        self,
        test_inputs: Any,
        layers: Optional[List[int]] = None
    ) -> FlowAnalysisResult:
        """
        Analyze information flow across layers.

        Args:
            test_inputs: Input data to analyze
            layers: Specific layers to analyze (default: all)

        Returns:
            Complete flow analysis result
        """
        if layers is None:
            layers = list(range(self.manager.n_layers))

        result = FlowAnalysisResult()

        # Get multi-layer features
        multi_features = self.manager.encode_all_layers(test_inputs, layers)

        # Analyze each layer
        for layer_idx in layers:
            layer_flow = self._analyze_layer(layer_idx, multi_features)
            result.layer_flows[layer_idx] = layer_flow

        # Analyze cross-layer flows
        for i in range(len(layers) - 1):
            source_layer = layers[i]
            target_layer = layers[i + 1]

            cross_flow = self._analyze_cross_layer(
                source_layer, target_layer, multi_features
            )
            result.cross_layer_flows[(source_layer, target_layer)] = cross_flow

        # Detect bottlenecks
        if self.config.detect_bottlenecks:
            result.bottlenecks = self._detect_bottlenecks(
                result.layer_flows, result.cross_layer_flows
            )

            # Find most severe bottleneck
            if result.bottlenecks:
                most_severe = max(result.bottlenecks, key=lambda b: b.severity)
                result.bottleneck_layer = most_severe.layer_idx

        # Analyze residual contributions
        if self.config.analyze_residual:
            result.residual_contributions = self._analyze_residual_stream(
                test_inputs, layers
            )

        # Compute summary statistics
        self._compute_summary(result, layers)

        return result

    def _analyze_layer(
        self,
        layer_idx: int,
        multi_features: Any  # MultiLayerFeatures
    ) -> LayerFlow:
        """Analyze information flow at a single layer."""
        layer_flow = LayerFlow(layer_idx=layer_idx)

        if layer_idx not in multi_features.layer_features:
            return layer_flow

        features = multi_features.layer_features[layer_idx]

        if TORCH_AVAILABLE and isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = np.array(features)

        # Basic statistics
        layer_flow.n_active_features = int(np.sum(features_np.max(axis=0) > 0))
        layer_flow.mean_activation = float(np.mean(features_np[features_np > 0])) if np.any(features_np > 0) else 0.0
        layer_flow.sparsity = float(np.mean(features_np == 0))

        # Entropy estimation
        if self.config.compute_entropy:
            layer_flow.activation_entropy = self._estimate_entropy(features_np)
            layer_flow.total_information = layer_flow.activation_entropy * layer_flow.n_active_features

        # Redundancy (average pairwise MI between active features)
        if self.config.compute_mutual_info and layer_flow.n_active_features > 1:
            layer_flow.redundancy = self._estimate_redundancy(features_np)

        # Capacity utilization
        total_features = features_np.shape[1] if len(features_np.shape) > 1 else 1
        layer_flow.capacity_used = layer_flow.n_active_features / total_features if total_features > 0 else 0

        return layer_flow

    def _analyze_cross_layer(
        self,
        source_layer: int,
        target_layer: int,
        multi_features: Any  # MultiLayerFeatures
    ) -> CrossLayerFlow:
        """Analyze information flow between two layers."""
        cross_flow = CrossLayerFlow(
            source_layer=source_layer,
            target_layer=target_layer
        )

        if source_layer not in multi_features.layer_features:
            return cross_flow
        if target_layer not in multi_features.layer_features:
            return cross_flow

        source_features = multi_features.layer_features[source_layer]
        target_features = multi_features.layer_features[target_layer]

        if TORCH_AVAILABLE:
            if isinstance(source_features, torch.Tensor):
                source_np = source_features.detach().cpu().numpy()
            else:
                source_np = np.array(source_features)
            if isinstance(target_features, torch.Tensor):
                target_np = target_features.detach().cpu().numpy()
            else:
                target_np = np.array(target_features)
        else:
            source_np = np.array(source_features)
            target_np = np.array(target_features)

        # Feature counts
        source_active = set(np.where(source_np.max(axis=0) > 0)[0])
        target_active = set(np.where(target_np.max(axis=0) > 0)[0])

        # This is a simplification - real analysis would track feature identity
        cross_flow.n_preserved_features = len(source_active & target_active)
        cross_flow.n_new_features = len(target_active - source_active)
        cross_flow.n_lost_features = len(source_active - target_active)

        # Compression ratio
        source_count = len(source_active) or 1
        target_count = len(target_active) or 1
        cross_flow.compression_ratio = target_count / source_count
        cross_flow.is_compression = cross_flow.compression_ratio < 1.0
        cross_flow.is_expansion = cross_flow.compression_ratio > 1.0

        # Mutual information between layers
        if self.config.compute_mutual_info:
            cross_flow.mutual_information = self._estimate_mutual_info(
                source_np, target_np
            )

        # Flow strength (normalized MI)
        max_entropy = max(
            self._estimate_entropy(source_np),
            self._estimate_entropy(target_np)
        )
        if max_entropy > 0:
            cross_flow.flow_strength = cross_flow.mutual_information / max_entropy
        else:
            cross_flow.flow_strength = 0.0

        # Compute top feature mappings via correlation
        cross_flow.top_mappings = self._compute_feature_mappings(
            source_np, target_np, source_layer, target_layer, top_k=20
        )

        return cross_flow

    def _detect_bottlenecks(
        self,
        layer_flows: Dict[int, LayerFlow],
        cross_flows: Dict[Tuple[int, int], CrossLayerFlow]
    ) -> List[FlowBottleneck]:
        """Detect information bottlenecks."""
        bottlenecks = []

        sorted_layers = sorted(layer_flows.keys())

        for i, layer_idx in enumerate(sorted_layers):
            layer_flow = layer_flows[layer_idx]

            # Check for saturation (very high capacity utilization)
            if layer_flow.capacity_used > self.config.saturation_threshold:
                bottleneck = FlowBottleneck(
                    layer_idx=layer_idx,
                    bottleneck_type=BottleneckType.SATURATION,
                    severity=layer_flow.capacity_used,
                    description=f"Layer {layer_idx} is saturated ({layer_flow.capacity_used:.1%} capacity)"
                )
                bottlenecks.append(bottleneck)

            # Check for high redundancy
            if layer_flow.redundancy > 0.5:
                bottleneck = FlowBottleneck(
                    layer_idx=layer_idx,
                    bottleneck_type=BottleneckType.REDUNDANCY,
                    severity=layer_flow.redundancy,
                    description=f"Layer {layer_idx} has high feature redundancy ({layer_flow.redundancy:.2f})"
                )
                bottlenecks.append(bottleneck)

        # Check cross-layer flows for compression bottlenecks
        for (source, target), cross_flow in cross_flows.items():
            if cross_flow.compression_ratio < self.config.compression_ratio_threshold:
                info_loss = 1.0 - cross_flow.compression_ratio
                bottleneck = FlowBottleneck(
                    layer_idx=source,
                    bottleneck_type=BottleneckType.COMPRESSION,
                    severity=info_loss,
                    information_before=layer_flows.get(source, LayerFlow(source)).total_information,
                    information_after=layer_flows.get(target, LayerFlow(target)).total_information,
                    information_loss=info_loss,
                    n_affected=cross_flow.n_lost_features,
                    description=f"Compression bottleneck at layer {source} ({cross_flow.compression_ratio:.2f}x)"
                )
                bottlenecks.append(bottleneck)

            # Check for information dropout (low flow strength)
            if cross_flow.flow_strength < self.config.bottleneck_threshold:
                bottleneck = FlowBottleneck(
                    layer_idx=source,
                    bottleneck_type=BottleneckType.DROPOUT,
                    severity=1.0 - cross_flow.flow_strength,
                    n_affected=cross_flow.n_lost_features,
                    description=f"Information dropout between layers {source}->{target} (flow: {cross_flow.flow_strength:.2f})"
                )
                bottlenecks.append(bottleneck)

        # Sort by severity
        bottlenecks.sort(key=lambda b: b.severity, reverse=True)

        return bottlenecks

    def _analyze_residual_stream(
        self,
        test_inputs: Any,
        layers: List[int]
    ) -> Dict[int, ResidualContribution]:
        """Analyze residual stream contributions."""
        contributions = {}

        # Extract activations with component breakdown
        # This requires model hooks - simplified version here
        activations = self.manager.extract_activations(test_inputs, layers)

        cumulative = 0.0

        for layer_idx in layers:
            contrib = ResidualContribution(layer_idx=layer_idx)

            if layer_idx in activations:
                layer_acts = activations[layer_idx]

                if TORCH_AVAILABLE and hasattr(layer_acts, 'residual'):
                    # If we have component breakdown
                    residual = layer_acts.residual
                    if isinstance(residual, torch.Tensor):
                        contrib.residual_norm = float(torch.norm(residual).item())
                else:
                    # Simplified: use raw activations
                    raw = layer_acts.raw_activations
                    if TORCH_AVAILABLE and isinstance(raw, torch.Tensor):
                        contrib.residual_norm = float(torch.norm(raw).item())
                    elif raw is not None:
                        contrib.residual_norm = float(np.linalg.norm(raw))

                cumulative += contrib.residual_norm
                contrib.cumulative_residual = cumulative

            contributions[layer_idx] = contrib

        # Normalize fractions
        if cumulative > 0:
            for layer_idx, contrib in contributions.items():
                contrib.residual_fraction = contrib.residual_norm / cumulative

        return contributions

    def _compute_summary(
        self,
        result: FlowAnalysisResult,
        layers: List[int]
    ) -> None:
        """Compute summary statistics."""
        if not layers:
            return

        first_layer = layers[0]
        last_layer = layers[-1]

        first_flow = result.layer_flows.get(first_layer)
        last_flow = result.layer_flows.get(last_layer)

        if first_flow and last_flow:
            if first_flow.total_information > 0:
                result.total_information_preserved = (
                    last_flow.total_information / first_flow.total_information
                )

        # Overall compression
        total_compression = 1.0
        for cross_flow in result.cross_layer_flows.values():
            total_compression *= cross_flow.compression_ratio
        result.overall_compression_ratio = total_compression

    def _estimate_entropy(self, features: np.ndarray) -> float:
        """Estimate entropy of feature activations."""
        if features.size == 0:
            return 0.0

        # Flatten and filter to positive values
        flat = features.flatten()
        flat = flat[flat > 0]

        if len(flat) == 0:
            return 0.0

        # Histogram-based entropy estimation
        hist, _ = np.histogram(flat, bins=self.config.n_bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins

        if len(hist) == 0:
            return 0.0

        # Normalize to probability
        hist = hist / hist.sum()

        # Compute entropy in bits
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return float(entropy)

    def _estimate_redundancy(self, features: np.ndarray) -> float:
        """Estimate redundancy (average pairwise correlation) between features."""
        if features.ndim != 2 or features.shape[1] < 2:
            return 0.0

        # Find active features
        active_mask = features.max(axis=0) > 0
        active_features = features[:, active_mask]

        if active_features.shape[1] < 2:
            return 0.0

        # Sample features if too many
        max_features = 100
        if active_features.shape[1] > max_features:
            indices = np.random.choice(
                active_features.shape[1], max_features, replace=False
            )
            active_features = active_features[:, indices]

        # Compute correlation matrix
        try:
            corr = np.corrcoef(active_features.T)
            # Average absolute off-diagonal correlation
            n = corr.shape[0]
            mask = ~np.eye(n, dtype=bool)
            redundancy = float(np.mean(np.abs(corr[mask])))
        except:
            redundancy = 0.0

        return redundancy

    def _estimate_mutual_info(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Estimate mutual information between source and target features."""
        if source.size == 0 or target.size == 0:
            return 0.0

        # Use average activation as summary statistic
        source_summary = source.mean(axis=1) if source.ndim > 1 else source
        target_summary = target.mean(axis=1) if target.ndim > 1 else target

        # Bin both
        n_bins = min(self.config.n_bins, len(source_summary) // 5)
        n_bins = max(n_bins, 2)

        try:
            # Joint histogram
            joint_hist, _, _ = np.histogram2d(
                source_summary, target_summary, bins=n_bins
            )
            joint_prob = joint_hist / joint_hist.sum()

            # Marginals
            source_prob = joint_prob.sum(axis=1)
            target_prob = joint_prob.sum(axis=0)

            # Mutual information
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if joint_prob[i, j] > 0:
                        mi += joint_prob[i, j] * np.log2(
                            joint_prob[i, j] / (source_prob[i] * target_prob[j] + 1e-10) + 1e-10
                        )
        except:
            mi = 0.0

        return float(max(0, mi))

    def _compute_feature_mappings(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_layer: int,
        target_layer: int,
        top_k: int = 20
    ) -> List[Tuple[str, str, float]]:
        """Compute top feature mappings between layers via correlation."""
        mappings = []

        if source.ndim != 2 or target.ndim != 2:
            return mappings

        # Find active features in both
        source_active = np.where(source.max(axis=0) > 0)[0]
        target_active = np.where(target.max(axis=0) > 0)[0]

        if len(source_active) == 0 or len(target_active) == 0:
            return mappings

        # Limit to prevent combinatorial explosion
        max_features = 50
        if len(source_active) > max_features:
            source_active = np.random.choice(source_active, max_features, replace=False)
        if len(target_active) > max_features:
            target_active = np.random.choice(target_active, max_features, replace=False)

        # Compute correlations
        all_correlations = []
        for s_idx in source_active:
            for t_idx in target_active:
                try:
                    corr = np.corrcoef(source[:, s_idx], target[:, t_idx])[0, 1]
                    if not np.isnan(corr):
                        all_correlations.append((
                            f"L{source_layer}.{s_idx}",
                            f"L{target_layer}.{t_idx}",
                            abs(corr)
                        ))
                except:
                    pass

        # Sort and return top-k
        all_correlations.sort(key=lambda x: x[2], reverse=True)
        return all_correlations[:top_k]

    def get_flow_summary(self, result: FlowAnalysisResult) -> str:
        """Generate human-readable flow summary."""
        lines = ["=== Information Flow Summary ===", ""]

        # Layer summary
        lines.append("Layer Information:")
        for layer_idx, flow in sorted(result.layer_flows.items()):
            lines.append(
                f"  Layer {layer_idx}: {flow.n_active_features} active features, "
                f"{flow.capacity_used:.1%} capacity, "
                f"entropy={flow.activation_entropy:.2f} bits"
            )

        lines.append("")

        # Cross-layer flows
        lines.append("Information Transfer:")
        for (src, tgt), flow in sorted(result.cross_layer_flows.items()):
            direction = "→" if flow.compression_ratio >= 1 else "↘"
            lines.append(
                f"  Layer {src} {direction} Layer {tgt}: "
                f"MI={flow.mutual_information:.2f} bits, "
                f"flow={flow.flow_strength:.2f}, "
                f"compression={flow.compression_ratio:.2f}x"
            )

        lines.append("")

        # Bottlenecks
        if result.bottlenecks:
            lines.append("Detected Bottlenecks:")
            for bottleneck in result.bottlenecks[:5]:  # Top 5
                lines.append(f"  ⚠ {bottleneck.description}")
        else:
            lines.append("No significant bottlenecks detected.")

        lines.append("")

        # Summary
        lines.append(f"Overall compression: {result.overall_compression_ratio:.2f}x")
        lines.append(f"Information preserved: {result.total_information_preserved:.1%}")

        if result.bottleneck_layer is not None:
            lines.append(f"Primary bottleneck: Layer {result.bottleneck_layer}")

        return "\n".join(lines)

    # =============================================================================
    # Integration with CausalMetrics (Pearl's do-calculus, Transfer Entropy)
    # =============================================================================

    def analyze_causal_strength(
        self,
        source_layer: int,
        target_layer: int,
        test_inputs: Any,
        intervention_features: Optional[List[int]] = None,
        n_interventions: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze causal strength between layers using Pearl's do-calculus.

        Uses interventional analysis to measure true causal effects rather
        than mere correlations. This distinguishes causal mechanisms from
        spurious correlations.

        Args:
            source_layer: Source layer index
            target_layer: Target layer index
            test_inputs: Input data for intervention analysis
            intervention_features: Specific features to intervene on (default: auto-select)
            n_interventions: Number of interventions to perform

        Returns:
            Dict containing causal strength results with:
            - causal_effect: Average causal effect magnitude
            - confidence_interval: (lower, upper) confidence bounds
            - intervention_results: Per-feature intervention outcomes
            - significant_features: Features with significant causal effects
        """
        from HoloLoom.dark_trace.multilayer.causal_metrics import (
            CausalStrengthEstimator,
            CausalStrengthResult,
        )

        # Get multi-layer features
        multi_features = self.manager.encode_all_layers(
            test_inputs, [source_layer, target_layer]
        )

        source_features = multi_features.layer_features.get(source_layer)
        target_features = multi_features.layer_features.get(target_layer)

        if source_features is None or target_features is None:
            return {
                "causal_effect": 0.0,
                "confidence_interval": (0.0, 0.0),
                "intervention_results": [],
                "significant_features": [],
                "error": "Could not extract features for specified layers",
            }

        # Convert to numpy for estimator
        if TORCH_AVAILABLE and isinstance(source_features, torch.Tensor):
            source_np = source_features.detach().cpu().numpy()
            target_np = target_features.detach().cpu().numpy()
        else:
            source_np = np.array(source_features)
            target_np = np.array(target_features)

        # Auto-select intervention features if not specified
        if intervention_features is None:
            # Select top active features by variance
            variances = np.var(source_np, axis=0)
            active_mask = source_np.max(axis=0) > 0
            variances[~active_mask] = -1
            intervention_features = list(
                np.argsort(variances)[-min(n_interventions, 20):]
            )

        # Create causal estimator
        estimator = CausalStrengthEstimator()

        # Estimate causal effects
        result: CausalStrengthResult = estimator.estimate(
            source_activations=source_np,
            target_activations=target_np,
            intervention_indices=intervention_features,
        )

        return {
            "causal_effect": result.causal_effect,
            "confidence_interval": result.confidence_interval,
            "intervention_results": result.intervention_results,
            "significant_features": [
                f"L{source_layer}.{idx}"
                for idx, effect in result.intervention_results
                if abs(effect) > 0.1
            ],
            "source_layer": source_layer,
            "target_layer": target_layer,
            "n_interventions": len(intervention_features),
        }

    def compute_transfer_entropy_detailed(
        self,
        source_layer: int,
        target_layer: int,
        test_inputs: Any,
        history_length: int = 3,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute detailed transfer entropy between layers.

        Transfer entropy measures directional information flow:
        T(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

        This quantifies how much knowing the source reduces uncertainty
        about the target's future state.

        Args:
            source_layer: Source layer index
            target_layer: Target layer index
            test_inputs: Input data (should have temporal structure)
            history_length: Number of past timesteps to consider
            n_bins: Number of bins for entropy estimation

        Returns:
            Dict containing transfer entropy results with:
            - transfer_entropy: Bits transferred from source to target
            - normalized_te: TE normalized by target entropy
            - significance: Statistical significance p-value
            - direction_ratio: T(source→target) / T(target→source)
            - is_significant: Whether TE is statistically significant
        """
        from HoloLoom.dark_trace.multilayer.causal_metrics import (
            TransferEntropyEstimator,
            TransferEntropyResult,
        )

        # Get multi-layer features
        multi_features = self.manager.encode_all_layers(
            test_inputs, [source_layer, target_layer]
        )

        source_features = multi_features.layer_features.get(source_layer)
        target_features = multi_features.layer_features.get(target_layer)

        if source_features is None or target_features is None:
            return {
                "transfer_entropy": 0.0,
                "normalized_te": 0.0,
                "significance": 1.0,
                "direction_ratio": 1.0,
                "is_significant": False,
                "error": "Could not extract features for specified layers",
            }

        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(source_features, torch.Tensor):
            source_np = source_features.detach().cpu().numpy()
            target_np = target_features.detach().cpu().numpy()
        else:
            source_np = np.array(source_features)
            target_np = np.array(target_features)

        # Use mean activation as summary signal
        source_signal = source_np.mean(axis=1) if source_np.ndim > 1 else source_np
        target_signal = target_np.mean(axis=1) if target_np.ndim > 1 else target_np

        # Create transfer entropy estimator
        estimator = TransferEntropyEstimator(
            history_length=history_length,
            n_bins=n_bins,
        )

        # Compute transfer entropy both directions
        result_forward: TransferEntropyResult = estimator.estimate(
            source_signal=source_signal,
            target_signal=target_signal,
        )

        result_backward: TransferEntropyResult = estimator.estimate(
            source_signal=target_signal,
            target_signal=source_signal,
        )

        # Compute direction ratio (asymmetry of information flow)
        forward_te = max(result_forward.transfer_entropy, 1e-10)
        backward_te = max(result_backward.transfer_entropy, 1e-10)
        direction_ratio = forward_te / backward_te

        return {
            "transfer_entropy": result_forward.transfer_entropy,
            "normalized_te": result_forward.normalized_te,
            "significance": result_forward.significance,
            "is_significant": result_forward.is_significant,
            "direction_ratio": direction_ratio,
            "reverse_te": result_backward.transfer_entropy,
            "source_layer": source_layer,
            "target_layer": target_layer,
            "dominant_direction": (
                f"L{source_layer}→L{target_layer}"
                if direction_ratio > 1.5
                else f"L{target_layer}→L{source_layer}"
                if direction_ratio < 0.67
                else "bidirectional"
            ),
        }

    def get_causal_flow_analysis(
        self,
        test_inputs: Any,
        layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive causal flow analysis across all layer pairs.

        Combines:
        - Pearl's do-calculus for causal strength estimation
        - Transfer entropy for directional information flow
        - Standard information flow metrics

        Args:
            test_inputs: Input data for analysis
            layers: Specific layers to analyze (default: all)

        Returns:
            Dict containing comprehensive causal flow analysis:
            - layer_pairs: Per-pair causal analysis results
            - causal_graph: Adjacency dict of significant causal connections
            - dominant_pathways: Strongest causal pathways through network
            - information_hubs: Layers that are causal hubs
        """
        if layers is None:
            layers = list(range(self.manager.n_layers))

        causal_results = {}
        te_results = {}

        # Analyze each adjacent layer pair
        for i in range(len(layers) - 1):
            source_layer = layers[i]
            target_layer = layers[i + 1]
            pair_key = (source_layer, target_layer)

            # Causal strength analysis
            causal_results[pair_key] = self.analyze_causal_strength(
                source_layer=source_layer,
                target_layer=target_layer,
                test_inputs=test_inputs,
            )

            # Transfer entropy analysis
            te_results[pair_key] = self.compute_transfer_entropy_detailed(
                source_layer=source_layer,
                target_layer=target_layer,
                test_inputs=test_inputs,
            )

        # Build causal graph (significant connections only)
        causal_graph: Dict[int, List[Tuple[int, float]]] = {l: [] for l in layers}
        for (src, tgt), result in causal_results.items():
            if result.get("causal_effect", 0) > 0.1:
                causal_graph[src].append((tgt, result["causal_effect"]))

        # Identify dominant pathways
        dominant_pathways = []
        for start_layer in layers[:len(layers) // 2]:  # Start from early layers
            pathway = [start_layer]
            current = start_layer

            while current in causal_graph and causal_graph[current]:
                # Follow strongest outgoing edge
                next_edges = causal_graph[current]
                if not next_edges:
                    break
                next_layer = max(next_edges, key=lambda x: x[1])[0]
                if next_layer in pathway:  # Avoid cycles
                    break
                pathway.append(next_layer)
                current = next_layer

            if len(pathway) > 2:
                pathway_strength = sum(
                    causal_results.get((pathway[i], pathway[i + 1]), {}).get(
                        "causal_effect", 0
                    )
                    for i in range(len(pathway) - 1)
                ) / (len(pathway) - 1)
                dominant_pathways.append({
                    "pathway": pathway,
                    "strength": pathway_strength,
                    "length": len(pathway),
                })

        dominant_pathways.sort(key=lambda p: p["strength"], reverse=True)

        # Identify information hubs (high in/out degree in causal graph)
        hub_scores = {}
        for layer in layers:
            # Outgoing causal strength
            out_strength = sum(e[1] for e in causal_graph.get(layer, []))

            # Incoming causal strength
            in_strength = sum(
                e[1]
                for src_layer, edges in causal_graph.items()
                for tgt, e_strength in [(e[0], e[1]) for e in edges]
                if tgt == layer
                for e in [(tgt, e_strength)]
            )

            hub_scores[layer] = {
                "out_strength": out_strength,
                "in_strength": in_strength,
                "total": out_strength + in_strength,
            }

        information_hubs = sorted(
            hub_scores.items(),
            key=lambda x: x[1]["total"],
            reverse=True,
        )[:5]

        return {
            "layer_pairs": {
                f"L{src}→L{tgt}": {
                    "causal": causal_results.get((src, tgt), {}),
                    "transfer_entropy": te_results.get((src, tgt), {}),
                }
                for src, tgt in causal_results.keys()
            },
            "causal_graph": {
                str(k): [(str(t), s) for t, s in v]
                for k, v in causal_graph.items()
            },
            "dominant_pathways": dominant_pathways[:5],
            "information_hubs": [
                {"layer": layer, **scores}
                for layer, scores in information_hubs
            ],
            "n_layers_analyzed": len(layers),
            "n_significant_connections": sum(
                1 for r in causal_results.values()
                if r.get("causal_effect", 0) > 0.1
            ),
        }


# Convenience function
def analyze_information_flow(
    layer_sae_manager: Any,
    test_inputs: Any,
    config: Optional[FlowConfig] = None
) -> FlowAnalysisResult:
    """
    Convenience function to analyze information flow.

    Args:
        layer_sae_manager: LayerSAEManager with trained SAEs
        test_inputs: Input data to analyze
        config: Optional flow configuration

    Returns:
        Complete flow analysis result
    """
    analyzer = InformationFlowAnalyzer(layer_sae_manager, config)
    return analyzer.analyze(test_inputs)
