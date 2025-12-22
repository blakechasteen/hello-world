"""
Dark Trace Causal: Activation Patching

Implements activation patching for causal tracing - transferring activations
between contexts to isolate causal mechanisms and trace information flow.

Research Basis:
- Causal Tracing (Meng et al., 2022): Locating factual knowledge in GPT
- Activation Patching: Isolating which components contribute to behavior
- ROME/MEMIT: Precise model editing via causal analysis
- EAP-IG: Integrated gradient-based causal attribution

Key Concepts:
- Source context: The context that produces desired behavior
- Target context: The context being patched (counterfactual)
- Patching: Replace target activations with source activations
- Causal trace: Map of which components matter for behavior

Usage:
    patcher = ActivationPatcher(model, probe)

    # Basic patching: transfer activations from source to target
    result = patcher.patch(
        source_input="The capital of France is Paris",
        target_input="The capital of France is",
        features=["sae.42", "sae.108"]
    )

    # Causal tracing: find which features matter
    trace = patcher.trace_causal_path(
        source_input="The Eiffel Tower is in Paris",
        target_input="The Eiffel Tower is in"
    )
    print(f"Critical features: {trace.critical_features}")

    # Information flow analysis
    flow = patcher.analyze_information_flow(
        inputs=["Paris is the capital", "France has Paris as capital"],
        features=["sae.42"]
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from enum import Enum
import numpy as np
import torch


class PatchingMethod(Enum):
    """Method for applying activation patches."""

    REPLACEMENT = "replacement"  # Replace target with source
    ADDITION = "addition"  # Add (source - target_mean) to target
    INTERPOLATION = "interpolation"  # Blend source and target
    RESIDUAL = "residual"  # Add residual from source


class TraceDirection(Enum):
    """Direction for causal tracing."""

    FORWARD = "forward"  # From input toward output
    BACKWARD = "backward"  # From output toward input
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class PatchConfig:
    """Configuration for activation patching."""

    # Patching settings
    method: PatchingMethod = PatchingMethod.REPLACEMENT
    interpolation_alpha: float = 0.5  # For INTERPOLATION method

    # Tracing settings
    trace_direction: TraceDirection = TraceDirection.BIDIRECTIONAL
    trace_granularity: str = "feature"  # "feature", "layer", "position"

    # Significance testing
    n_noise_samples: int = 10  # Noised run samples for baseline
    significance_threshold: float = 0.1  # Minimum effect to consider causal
    confidence_level: float = 0.95

    # Performance
    batch_size: int = 32
    cache_activations: bool = True

    @classmethod
    def default(cls) -> "PatchConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def quick(cls) -> "PatchConfig":
        """Fast configuration for exploration."""
        return cls(
            n_noise_samples=3,
            significance_threshold=0.2,
            batch_size=64,
        )

    @classmethod
    def rigorous(cls) -> "PatchConfig":
        """Rigorous configuration for publication."""
        return cls(
            n_noise_samples=50,
            significance_threshold=0.05,
            confidence_level=0.99,
            trace_direction=TraceDirection.BIDIRECTIONAL,
        )


@dataclass
class PatchPoint:
    """A single point in the patching process."""

    feature_id: str
    layer: int
    position: int
    source_activation: float
    target_activation: float
    patched_activation: float


@dataclass
class PatchResult:
    """Result of an activation patching operation."""

    # Core results
    source_output: Any  # Output with source activations
    target_output: Any  # Output with target activations
    patched_output: Any  # Output after patching

    # Effect measurement
    effect_on_target: float  # How much patching changed target output
    recovery_rate: float  # How much of source behavior was recovered

    # Patch details
    features_patched: List[str]
    patch_points: List[PatchPoint]
    method: PatchingMethod

    # Statistics
    is_significant: bool
    p_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]

    # Metadata
    source_input: str
    target_input: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Activation Patching Result",
            f"=" * 40,
            f"Features patched: {len(self.features_patched)}",
            f"Method: {self.method.value}",
            f"",
            f"Effect on target: {self.effect_on_target:.4f}",
            f"Recovery rate: {self.recovery_rate:.2%}",
            f"Significant: {'Yes' if self.is_significant else 'No'}",
        ]

        if self.p_value is not None:
            lines.append(f"p-value: {self.p_value:.4f}")

        if self.confidence_interval:
            lines.append(
                f"95% CI: [{self.confidence_interval[0]:.4f}, "
                f"{self.confidence_interval[1]:.4f}]"
            )

        return "\n".join(lines)


@dataclass
class CausalNode:
    """A node in the causal trace graph."""

    feature_id: str
    layer: int
    position: int
    causal_effect: float  # Effect of patching this node
    is_critical: bool  # Above significance threshold


@dataclass
class CausalEdge:
    """An edge in the causal trace graph."""

    source_node: str
    target_node: str
    information_flow: float  # Strength of causal connection
    direction: str  # "forward" or "backward"


@dataclass
class CausalTrace:
    """Result of causal tracing analysis."""

    # Graph structure
    nodes: List[CausalNode]
    edges: List[CausalEdge]

    # Summary statistics
    critical_features: List[str]  # Features with significant causal effect
    bottleneck_features: List[str]  # High incoming, low outgoing flow
    hub_features: List[str]  # High both incoming and outgoing

    # Layer-wise analysis
    layer_importance: Dict[int, float]  # Importance score per layer
    position_importance: Dict[int, float]  # Importance score per position

    # Overall metrics
    total_causal_effect: float
    localization_score: float  # How concentrated causal effects are
    pathway_count: int  # Number of distinct causal pathways

    # Metadata
    source_input: str
    target_input: str
    trace_direction: TraceDirection
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_critical_path(self) -> List[str]:
        """Get the most important causal pathway."""
        # Sort critical features by causal effect
        critical_with_effect = [
            (node.feature_id, node.causal_effect)
            for node in self.nodes
            if node.is_critical
        ]
        critical_with_effect.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in critical_with_effect]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Causal Trace Analysis",
            f"=" * 40,
            f"Total nodes: {len(self.nodes)}",
            f"Total edges: {len(self.edges)}",
            f"Critical features: {len(self.critical_features)}",
            f"",
            f"Total causal effect: {self.total_causal_effect:.4f}",
            f"Localization score: {self.localization_score:.4f}",
            f"Distinct pathways: {self.pathway_count}",
            f"",
            f"Top critical features:",
        ]

        for feature in self.critical_features[:5]:
            node = next((n for n in self.nodes if n.feature_id == feature), None)
            if node:
                lines.append(f"  {feature}: effect={node.causal_effect:.4f}")

        return "\n".join(lines)


@dataclass
class FlowNode:
    """Node in information flow analysis."""

    feature_id: str
    incoming_flow: float
    outgoing_flow: float
    net_flow: float  # outgoing - incoming
    is_source: bool  # Net producer of information
    is_sink: bool  # Net consumer of information


@dataclass
class InformationFlow:
    """Result of information flow analysis."""

    # Flow graph
    flow_nodes: List[FlowNode]
    flow_matrix: np.ndarray  # [n_features, n_features] flow strengths

    # Summary
    source_features: List[str]  # Net information producers
    sink_features: List[str]  # Net information consumers
    bottleneck_features: List[str]  # High throughput, constrained capacity

    # Metrics
    total_flow: float
    flow_efficiency: float  # Ratio of useful to total flow
    redundancy_score: float  # How much parallel flow exists

    # Per-input analysis (if multiple inputs)
    per_input_flows: Optional[Dict[str, Dict[str, float]]]

    # Metadata
    inputs_analyzed: List[str]
    features_analyzed: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_flow(self, source: str, target: str) -> float:
        """Get flow strength between two features."""
        try:
            src_idx = self.features_analyzed.index(source)
            tgt_idx = self.features_analyzed.index(target)
            return float(self.flow_matrix[src_idx, tgt_idx])
        except (ValueError, IndexError):
            return 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Information Flow Analysis",
            f"=" * 40,
            f"Features analyzed: {len(self.features_analyzed)}",
            f"Inputs analyzed: {len(self.inputs_analyzed)}",
            f"",
            f"Total flow: {self.total_flow:.4f}",
            f"Flow efficiency: {self.flow_efficiency:.2%}",
            f"Redundancy score: {self.redundancy_score:.4f}",
            f"",
            f"Source features (net producers): {len(self.source_features)}",
            f"Sink features (net consumers): {len(self.sink_features)}",
            f"Bottleneck features: {len(self.bottleneck_features)}",
        ]

        return "\n".join(lines)


class ActivationPatcher:
    """
    Activation patcher for causal tracing.

    Implements activation patching to isolate causal mechanisms
    by transferring activations between source and target contexts.

    Usage:
        patcher = ActivationPatcher(model, probe)

        # Basic patching
        result = patcher.patch(
            source_input="Paris is in France",
            target_input="Berlin is in",
            features=["sae.42"]
        )

        # Causal tracing
        trace = patcher.trace_causal_path(
            source_input="The Eiffel Tower is in Paris",
            target_input="The Eiffel Tower is in"
        )
    """

    def __init__(
        self,
        model: Any,
        probe: Any,  # MindProbe for activation access
        config: Optional[PatchConfig] = None,
    ):
        """
        Initialize activation patcher.

        Args:
            model: Model to patch activations in
            probe: MindProbe for accessing activations
            config: Patching configuration
        """
        self.model = model
        self.probe = probe
        self.config = config or PatchConfig.default()

        # Activation cache
        self._activation_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def patch(
        self,
        source_input: str,
        target_input: str,
        features: List[str],
        method: Optional[PatchingMethod] = None,
    ) -> PatchResult:
        """
        Patch activations from source to target context.

        Replaces target activations with source activations for specified
        features and measures the effect on output.

        Args:
            source_input: Input that produces desired behavior
            target_input: Input to patch (counterfactual)
            features: Feature IDs to patch (e.g., ["sae.42", "sae.108"])
            method: Patching method (defaults to config)

        Returns:
            PatchResult with patching effects
        """
        method = method or self.config.method

        # Get clean activations
        source_acts = self._get_activations(source_input)
        target_acts = self._get_activations(target_input)

        # Get clean outputs
        source_output = self._run_model(source_input)
        target_output = self._run_model(target_input)

        # Apply patches and get patched output
        patch_points = []
        patched_acts = {}

        for feature_id in features:
            if feature_id in source_acts and feature_id in target_acts:
                src_act = source_acts[feature_id]
                tgt_act = target_acts[feature_id]

                # Apply patching method
                patched_act = self._apply_patch(src_act, tgt_act, method)
                patched_acts[feature_id] = patched_act

                # Record patch point
                patch_points.append(PatchPoint(
                    feature_id=feature_id,
                    layer=self._get_feature_layer(feature_id),
                    position=0,  # Simplified - full version tracks position
                    source_activation=float(src_act.mean()),
                    target_activation=float(tgt_act.mean()),
                    patched_activation=float(patched_act.mean()),
                ))

        # Run model with patched activations
        patched_output = self._run_with_patches(target_input, patched_acts)

        # Measure effects
        effect_on_target = self._measure_output_change(
            target_output, patched_output
        )
        recovery_rate = self._measure_recovery(
            source_output, target_output, patched_output
        )

        # Statistical testing
        is_significant, p_value, ci = self._test_significance(
            source_input, target_input, features, effect_on_target
        )

        return PatchResult(
            source_output=source_output,
            target_output=target_output,
            patched_output=patched_output,
            effect_on_target=effect_on_target,
            recovery_rate=recovery_rate,
            features_patched=features,
            patch_points=patch_points,
            method=method,
            is_significant=is_significant,
            p_value=p_value,
            confidence_interval=ci,
            source_input=source_input,
            target_input=target_input,
        )

    def trace_causal_path(
        self,
        source_input: str,
        target_input: str,
        features: Optional[List[str]] = None,
        direction: Optional[TraceDirection] = None,
    ) -> CausalTrace:
        """
        Trace causal pathway from input to output.

        Systematically patches each feature to identify which ones
        are causally responsible for the behavioral difference.

        Args:
            source_input: Input with desired behavior
            target_input: Counterfactual input
            features: Features to trace (None = all available)
            direction: Tracing direction (defaults to config)

        Returns:
            CausalTrace with causal pathway analysis
        """
        direction = direction or self.config.trace_direction

        # Get all features if not specified
        if features is None:
            features = self._get_all_features()

        # Get baseline outputs
        source_output = self._run_model(source_input)
        target_output = self._run_model(target_input)
        baseline_diff = self._measure_output_change(source_output, target_output)

        # Trace each feature
        nodes = []
        for feature_id in features:
            # Patch this single feature
            result = self.patch(
                source_input=source_input,
                target_input=target_input,
                features=[feature_id],
            )

            # Causal effect = how much patching this feature changes output
            causal_effect = result.effect_on_target
            is_critical = causal_effect > self.config.significance_threshold

            nodes.append(CausalNode(
                feature_id=feature_id,
                layer=self._get_feature_layer(feature_id),
                position=0,
                causal_effect=causal_effect,
                is_critical=is_critical,
            ))

        # Build edge structure based on direction
        edges = self._build_causal_edges(nodes, direction)

        # Identify critical, bottleneck, and hub features
        critical_features = [n.feature_id for n in nodes if n.is_critical]

        # Bottleneck: high incoming, low outgoing flow
        bottleneck_features = self._find_bottlenecks(nodes, edges)

        # Hub: high both incoming and outgoing
        hub_features = self._find_hubs(nodes, edges)

        # Layer and position importance
        layer_importance = self._compute_layer_importance(nodes)
        position_importance = self._compute_position_importance(nodes)

        # Overall metrics
        total_effect = sum(n.causal_effect for n in nodes)
        localization = self._compute_localization(nodes)
        pathway_count = self._count_pathways(edges)

        return CausalTrace(
            nodes=nodes,
            edges=edges,
            critical_features=critical_features,
            bottleneck_features=bottleneck_features,
            hub_features=hub_features,
            layer_importance=layer_importance,
            position_importance=position_importance,
            total_causal_effect=total_effect,
            localization_score=localization,
            pathway_count=pathway_count,
            source_input=source_input,
            target_input=target_input,
            trace_direction=direction,
        )

    def analyze_information_flow(
        self,
        inputs: List[str],
        features: List[str],
    ) -> InformationFlow:
        """
        Analyze information flow between features.

        Measures how information propagates through the feature network
        by systematically patching and measuring downstream effects.

        Args:
            inputs: List of inputs to analyze
            features: Features to analyze flow between

        Returns:
            InformationFlow analysis result
        """
        n_features = len(features)
        flow_matrix = np.zeros((n_features, n_features))

        # For each pair of features, measure information flow
        for i, src_feature in enumerate(features):
            for j, tgt_feature in enumerate(features):
                if i == j:
                    continue

                # Measure how patching src affects tgt's contribution
                flow = self._measure_pairwise_flow(
                    inputs, src_feature, tgt_feature
                )
                flow_matrix[i, j] = flow

        # Compute per-feature statistics
        flow_nodes = []
        for i, feature_id in enumerate(features):
            incoming = float(np.sum(flow_matrix[:, i]))
            outgoing = float(np.sum(flow_matrix[i, :]))
            net = outgoing - incoming

            flow_nodes.append(FlowNode(
                feature_id=feature_id,
                incoming_flow=incoming,
                outgoing_flow=outgoing,
                net_flow=net,
                is_source=net > self.config.significance_threshold,
                is_sink=net < -self.config.significance_threshold,
            ))

        # Identify sources, sinks, bottlenecks
        source_features = [n.feature_id for n in flow_nodes if n.is_source]
        sink_features = [n.feature_id for n in flow_nodes if n.is_sink]

        # Bottleneck: high throughput but limited capacity
        bottleneck_features = [
            n.feature_id for n in flow_nodes
            if n.incoming_flow > np.mean([fn.incoming_flow for fn in flow_nodes])
            and n.outgoing_flow > np.mean([fn.outgoing_flow for fn in flow_nodes])
        ]

        # Overall metrics
        total_flow = float(np.sum(flow_matrix))
        flow_efficiency = self._compute_flow_efficiency(flow_matrix)
        redundancy = self._compute_redundancy(flow_matrix)

        # Per-input analysis
        per_input_flows = {}
        for inp in inputs:
            per_input_flows[inp] = self._analyze_single_input_flow(
                inp, features
            )

        return InformationFlow(
            flow_nodes=flow_nodes,
            flow_matrix=flow_matrix,
            source_features=source_features,
            sink_features=sink_features,
            bottleneck_features=bottleneck_features,
            total_flow=total_flow,
            flow_efficiency=flow_efficiency,
            redundancy_score=redundancy,
            per_input_flows=per_input_flows,
            inputs_analyzed=inputs,
            features_analyzed=features,
        )

    def patch_sweep(
        self,
        source_input: str,
        target_input: str,
        features: List[str],
        sweep_values: Optional[List[float]] = None,
    ) -> List[PatchResult]:
        """
        Sweep interpolation values for patching.

        Useful for understanding how gradually introducing source
        activations affects output.

        Args:
            source_input: Source context
            target_input: Target context
            features: Features to patch
            sweep_values: Interpolation values (default: 0.0 to 1.0)

        Returns:
            List of PatchResult for each sweep value
        """
        if sweep_values is None:
            sweep_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        results = []
        original_alpha = self.config.interpolation_alpha

        for alpha in sweep_values:
            self.config.interpolation_alpha = alpha
            result = self.patch(
                source_input=source_input,
                target_input=target_input,
                features=features,
                method=PatchingMethod.INTERPOLATION,
            )
            result.metadata["interpolation_alpha"] = alpha
            results.append(result)

        self.config.interpolation_alpha = original_alpha
        return results

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_activations(self, input_text: str) -> Dict[str, torch.Tensor]:
        """Get activations for input, using cache if available."""
        if self.config.cache_activations and input_text in self._activation_cache:
            return self._activation_cache[input_text]

        # Use probe to get activations
        activations = {}
        if hasattr(self.probe, 'get_activations'):
            activations = self.probe.get_activations(input_text)
        else:
            # Fallback: run model and extract
            with torch.no_grad():
                _ = self._run_model(input_text)
                if hasattr(self.probe, 'last_activations'):
                    activations = self.probe.last_activations

        if self.config.cache_activations:
            self._activation_cache[input_text] = activations

        return activations

    def _run_model(self, input_text: str) -> Any:
        """Run model on input and return output."""
        if hasattr(self.model, 'generate'):
            return self.model.generate(input_text)
        elif hasattr(self.model, 'forward'):
            return self.model.forward(input_text)
        elif callable(self.model):
            return self.model(input_text)
        else:
            raise ValueError("Model must have generate(), forward(), or be callable")

    def _run_with_patches(
        self,
        input_text: str,
        patches: Dict[str, torch.Tensor],
    ) -> Any:
        """Run model with activation patches applied."""
        if hasattr(self.probe, 'run_with_patches'):
            return self.probe.run_with_patches(self.model, input_text, patches)
        else:
            # Fallback: use hooks to inject patches
            return self._run_with_hooks(input_text, patches)

    def _run_with_hooks(
        self,
        input_text: str,
        patches: Dict[str, torch.Tensor],
    ) -> Any:
        """Run model with hook-based activation patching."""
        handles = []

        def create_patch_hook(feature_id: str, patch_value: torch.Tensor):
            def hook(module, input, output):
                # Apply patch to output
                return patch_value
            return hook

        try:
            # Register hooks for each patch
            for feature_id, patch_value in patches.items():
                module = self._get_feature_module(feature_id)
                if module is not None:
                    handle = module.register_forward_hook(
                        create_patch_hook(feature_id, patch_value)
                    )
                    handles.append(handle)

            # Run model
            output = self._run_model(input_text)

        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()

        return output

    def _apply_patch(
        self,
        source_act: torch.Tensor,
        target_act: torch.Tensor,
        method: PatchingMethod,
    ) -> torch.Tensor:
        """Apply patching method to activations."""
        if method == PatchingMethod.REPLACEMENT:
            return source_act.clone()

        elif method == PatchingMethod.ADDITION:
            # Add difference from mean
            diff = source_act - source_act.mean()
            return target_act + diff

        elif method == PatchingMethod.INTERPOLATION:
            alpha = self.config.interpolation_alpha
            return (1 - alpha) * target_act + alpha * source_act

        elif method == PatchingMethod.RESIDUAL:
            # Add residual stream contribution
            residual = source_act - target_act
            return target_act + residual

        else:
            raise ValueError(f"Unknown patching method: {method}")

    def _measure_output_change(self, output1: Any, output2: Any) -> float:
        """Measure change between two outputs."""
        if isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
            return float(torch.nn.functional.mse_loss(output1, output2))
        elif isinstance(output1, str) and isinstance(output2, str):
            # String similarity (simplified)
            return 1.0 - (len(set(output1) & set(output2)) /
                         max(len(set(output1) | set(output2)), 1))
        else:
            # Generic comparison
            return 0.0 if output1 == output2 else 1.0

    def _measure_recovery(
        self,
        source_output: Any,
        target_output: Any,
        patched_output: Any,
    ) -> float:
        """Measure how much of source behavior was recovered."""
        source_target_diff = self._measure_output_change(source_output, target_output)
        patched_target_diff = self._measure_output_change(patched_output, target_output)
        patched_source_diff = self._measure_output_change(patched_output, source_output)

        if source_target_diff == 0:
            return 1.0

        # Recovery = how much closer patched is to source than target was
        recovery = 1.0 - (patched_source_diff / source_target_diff)
        return max(0.0, min(1.0, recovery))

    def _test_significance(
        self,
        source_input: str,
        target_input: str,
        features: List[str],
        observed_effect: float,
    ) -> Tuple[bool, Optional[float], Optional[Tuple[float, float]]]:
        """Test statistical significance of patching effect."""
        # Generate null distribution via noised runs
        null_effects = []
        for _ in range(self.config.n_noise_samples):
            noised_effect = self._compute_noised_effect(
                source_input, target_input, features
            )
            null_effects.append(noised_effect)

        if not null_effects:
            return observed_effect > self.config.significance_threshold, None, None

        null_effects = np.array(null_effects)
        null_mean = np.mean(null_effects)
        null_std = np.std(null_effects) + 1e-10

        # Z-score
        z_score = (observed_effect - null_mean) / null_std

        # p-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Confidence interval
        ci_half = stats.norm.ppf(
            1 - (1 - self.config.confidence_level) / 2
        ) * null_std
        ci = (observed_effect - ci_half, observed_effect + ci_half)

        is_significant = p_value < (1 - self.config.confidence_level)

        return is_significant, float(p_value), ci

    def _compute_noised_effect(
        self,
        source_input: str,
        target_input: str,
        features: List[str],
    ) -> float:
        """Compute effect with noise added."""
        # Add noise to source activations before patching
        source_acts = self._get_activations(source_input)

        noised_acts = {}
        for feature_id in features:
            if feature_id in source_acts:
                act = source_acts[feature_id]
                noise = torch.randn_like(act) * act.std() * 0.1
                noised_acts[feature_id] = act + noise

        # Run with noised patches
        target_output = self._run_model(target_input)
        noised_output = self._run_with_patches(target_input, noised_acts)

        return self._measure_output_change(target_output, noised_output)

    def _get_feature_layer(self, feature_id: str) -> int:
        """Get layer index for a feature."""
        # Parse feature ID (e.g., "sae.42" -> layer based on probe config)
        if hasattr(self.probe, 'get_feature_layer'):
            return self.probe.get_feature_layer(feature_id)
        return 0  # Default to layer 0

    def _get_feature_module(self, feature_id: str) -> Optional[torch.nn.Module]:
        """Get the module corresponding to a feature."""
        if hasattr(self.probe, 'get_feature_module'):
            return self.probe.get_feature_module(feature_id)
        return None

    def _get_all_features(self) -> List[str]:
        """Get all available feature IDs."""
        if hasattr(self.probe, 'get_all_features'):
            return self.probe.get_all_features()
        return []

    def _build_causal_edges(
        self,
        nodes: List[CausalNode],
        direction: TraceDirection,
    ) -> List[CausalEdge]:
        """Build causal edge structure."""
        edges = []

        # Group nodes by layer
        by_layer = {}
        for node in nodes:
            if node.layer not in by_layer:
                by_layer[node.layer] = []
            by_layer[node.layer].append(node)

        layers = sorted(by_layer.keys())

        # Build edges based on direction
        if direction in [TraceDirection.FORWARD, TraceDirection.BIDIRECTIONAL]:
            for i in range(len(layers) - 1):
                src_layer = layers[i]
                tgt_layer = layers[i + 1]

                for src_node in by_layer[src_layer]:
                    for tgt_node in by_layer[tgt_layer]:
                        # Flow based on causal effects
                        flow = min(src_node.causal_effect, tgt_node.causal_effect)
                        if flow > self.config.significance_threshold:
                            edges.append(CausalEdge(
                                source_node=src_node.feature_id,
                                target_node=tgt_node.feature_id,
                                information_flow=flow,
                                direction="forward",
                            ))

        if direction in [TraceDirection.BACKWARD, TraceDirection.BIDIRECTIONAL]:
            for i in range(len(layers) - 1, 0, -1):
                src_layer = layers[i]
                tgt_layer = layers[i - 1]

                for src_node in by_layer[src_layer]:
                    for tgt_node in by_layer[tgt_layer]:
                        flow = min(src_node.causal_effect, tgt_node.causal_effect)
                        if flow > self.config.significance_threshold:
                            edges.append(CausalEdge(
                                source_node=src_node.feature_id,
                                target_node=tgt_node.feature_id,
                                information_flow=flow,
                                direction="backward",
                            ))

        return edges

    def _find_bottlenecks(
        self,
        nodes: List[CausalNode],
        edges: List[CausalEdge],
    ) -> List[str]:
        """Find bottleneck features (high incoming, low outgoing)."""
        incoming = {n.feature_id: 0.0 for n in nodes}
        outgoing = {n.feature_id: 0.0 for n in nodes}

        for edge in edges:
            incoming[edge.target_node] = incoming.get(edge.target_node, 0) + edge.information_flow
            outgoing[edge.source_node] = outgoing.get(edge.source_node, 0) + edge.information_flow

        mean_incoming = np.mean(list(incoming.values())) if incoming else 0
        mean_outgoing = np.mean(list(outgoing.values())) if outgoing else 0

        bottlenecks = [
            node_id for node_id in incoming
            if incoming[node_id] > mean_incoming and outgoing[node_id] < mean_outgoing
        ]

        return bottlenecks

    def _find_hubs(
        self,
        nodes: List[CausalNode],
        edges: List[CausalEdge],
    ) -> List[str]:
        """Find hub features (high both incoming and outgoing)."""
        incoming = {n.feature_id: 0.0 for n in nodes}
        outgoing = {n.feature_id: 0.0 for n in nodes}

        for edge in edges:
            incoming[edge.target_node] = incoming.get(edge.target_node, 0) + edge.information_flow
            outgoing[edge.source_node] = outgoing.get(edge.source_node, 0) + edge.information_flow

        mean_incoming = np.mean(list(incoming.values())) if incoming else 0
        mean_outgoing = np.mean(list(outgoing.values())) if outgoing else 0

        hubs = [
            node_id for node_id in incoming
            if incoming[node_id] > mean_incoming and outgoing[node_id] > mean_outgoing
        ]

        return hubs

    def _compute_layer_importance(
        self,
        nodes: List[CausalNode],
    ) -> Dict[int, float]:
        """Compute importance score per layer."""
        by_layer = {}
        for node in nodes:
            if node.layer not in by_layer:
                by_layer[node.layer] = []
            by_layer[node.layer].append(node.causal_effect)

        return {
            layer: float(np.mean(effects))
            for layer, effects in by_layer.items()
        }

    def _compute_position_importance(
        self,
        nodes: List[CausalNode],
    ) -> Dict[int, float]:
        """Compute importance score per position."""
        by_position = {}
        for node in nodes:
            if node.position not in by_position:
                by_position[node.position] = []
            by_position[node.position].append(node.causal_effect)

        return {
            pos: float(np.mean(effects))
            for pos, effects in by_position.items()
        }

    def _compute_localization(self, nodes: List[CausalNode]) -> float:
        """Compute how localized causal effects are (Gini coefficient)."""
        effects = sorted([n.causal_effect for n in nodes], reverse=True)
        if not effects or sum(effects) == 0:
            return 0.0

        n = len(effects)
        cumsum = np.cumsum(effects)
        return float((n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n)

    def _count_pathways(self, edges: List[CausalEdge]) -> int:
        """Count distinct causal pathways."""
        # Simplified: count connected components
        if not edges:
            return 0

        nodes = set()
        for edge in edges:
            nodes.add(edge.source_node)
            nodes.add(edge.target_node)

        # Union-find to count components
        parent = {n: n for n in nodes}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for edge in edges:
            union(edge.source_node, edge.target_node)

        return len(set(find(n) for n in nodes))

    def _measure_pairwise_flow(
        self,
        inputs: List[str],
        src_feature: str,
        tgt_feature: str,
    ) -> float:
        """Measure information flow from src to tgt feature."""
        flows = []

        for inp in inputs:
            # Get baseline tgt effect
            baseline_result = self.patch(
                source_input=inp,
                target_input=inp,  # Self-patching baseline
                features=[tgt_feature],
            )
            baseline_effect = baseline_result.effect_on_target

            # Patch src, then measure tgt effect
            # First patch src
            patched_result = self.patch(
                source_input=inp,
                target_input=inp,
                features=[src_feature, tgt_feature],
            )
            combined_effect = patched_result.effect_on_target

            # Flow = how much src patching changes tgt's contribution
            flow = abs(combined_effect - baseline_effect)
            flows.append(flow)

        return float(np.mean(flows)) if flows else 0.0

    def _compute_flow_efficiency(self, flow_matrix: np.ndarray) -> float:
        """Compute ratio of useful to total flow."""
        # Useful flow = along main pathways (diagonal adjacent)
        total = np.sum(flow_matrix)
        if total == 0:
            return 0.0

        # Consider flow along adjacent layers as "useful"
        n = flow_matrix.shape[0]
        useful = 0.0
        for i in range(n - 1):
            useful += flow_matrix[i, i + 1]

        return useful / total

    def _compute_redundancy(self, flow_matrix: np.ndarray) -> float:
        """Compute redundancy in information flow."""
        # High redundancy = multiple parallel paths
        # Measure as ratio of off-diagonal to diagonal-adjacent flow
        n = flow_matrix.shape[0]
        if n < 2:
            return 0.0

        diagonal_adjacent = sum(
            flow_matrix[i, i + 1] for i in range(n - 1)
        )
        total = np.sum(flow_matrix)

        if diagonal_adjacent == 0:
            return 1.0 if total > 0 else 0.0

        return (total - diagonal_adjacent) / total

    def _analyze_single_input_flow(
        self,
        input_text: str,
        features: List[str],
    ) -> Dict[str, float]:
        """Analyze flow for a single input."""
        flows = {}
        for feature in features:
            result = self.patch(
                source_input=input_text,
                target_input=input_text,
                features=[feature],
            )
            flows[feature] = result.effect_on_target

        return flows

    def clear_cache(self) -> None:
        """Clear activation cache."""
        self._activation_cache.clear()
