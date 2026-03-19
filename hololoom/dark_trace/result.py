"""
Dark Trace Result: Dataclasses for Interpretability Analysis Results

This module defines the result types returned by DarkTraceEngine analysis.
All results include complete provenance for transparency and debugging.

Key Classes:
- TraceResult: Main result from DarkTraceEngine.analyze()
- LensResult: Result from a single lens
- CircuitTrace: Mechanistic circuit tracing result
- SteeringResult: Result from steering application
- AblationResult: Result from ablation testing
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import torch

from hololoom.dark_trace.protocol import (
    FeatureActivation,
    LensType,
    SafetyFlag,
    SteeringVector,
)

# =============================================================================
# Result Enums
# =============================================================================

class AnalysisStatus(Enum):
    """Status of an analysis operation."""
    SUCCESS = auto()
    PARTIAL = auto()      # Some lenses succeeded
    FAILED = auto()
    TIMEOUT = auto()


class SteeringOutcome(Enum):
    """Outcome of a steering operation."""
    APPLIED = auto()       # Successfully applied
    VALIDATED = auto()     # Applied and validated safe
    REJECTED = auto()      # Rejected by safety checks
    PARTIAL = auto()       # Partially applied (some features blocked)


# =============================================================================
# Core Result Types
# =============================================================================

@dataclass
class LensResult:
    """
    Result from a single interpretability lens.

    Encapsulates the analysis from one lens (SAE, Semantic, Domain).
    """
    lens_type: LensType
    lens_name: str

    # Core results
    active_features: list[FeatureActivation]
    feature_projections: torch.Tensor | None = None  # [n_features] or [batch, n_features]

    # Explanation
    explanation: str | None = None

    # Metrics
    reconstruction_loss: float | None = None    # For SAE
    sparsity: float | None = None               # Fraction of active features
    coherence_score: float | None = None        # How coherent the activations are

    # Timing
    compute_time_ms: float = 0.0

    # Status
    status: AnalysisStatus = AnalysisStatus.SUCCESS
    error_message: str | None = None

    def top_features(self, k: int = 5) -> list[FeatureActivation]:
        """Get top-k features by activation."""
        return sorted(self.active_features, reverse=True)[:k]

    def get_feature_dict(self) -> dict[str, float]:
        """Get dictionary of {feature_id: activation}."""
        return {act.feature.id: act.activation for act in self.active_features}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "lens_type": self.lens_type.name,
            "lens_name": self.lens_name,
            "active_features": [
                {"id": act.feature.id, "activation": act.activation}
                for act in self.active_features
            ],
            "explanation": self.explanation,
            "reconstruction_loss": self.reconstruction_loss,
            "sparsity": self.sparsity,
            "compute_time_ms": self.compute_time_ms,
            "status": self.status.name,
        }


@dataclass
class TraceResult:
    """
    Main result from DarkTraceEngine.analyze().

    Aggregates results from all active lenses with complete provenance.
    """
    # Identification
    trace_id: str                        # Unique trace identifier
    timestamp: datetime = field(default_factory=datetime.now)

    # Input context
    input_shape: tuple[int, ...] = ()
    input_dtype: str = ""
    layer_name: str | None = None

    # Results per lens
    lens_results: dict[LensType, LensResult] = field(default_factory=dict)

    # Aggregated results
    all_features: list[FeatureActivation] = field(default_factory=list)
    dominant_features: list[FeatureActivation] = field(default_factory=list)

    # Safety analysis
    safety_flags: dict[str, SafetyFlag] = field(default_factory=dict)
    has_safety_concerns: bool = False
    safety_summary: str | None = None

    # Cross-lens analysis
    sae_semantic_correlations: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    # Status
    status: AnalysisStatus = AnalysisStatus.SUCCESS
    error_messages: list[str] = field(default_factory=list)

    # Timing
    total_time_ms: float = 0.0

    # Provenance
    config_used: dict[str, Any] = field(default_factory=dict)
    lenses_used: list[str] = field(default_factory=list)

    def get_lens_result(self, lens_type: LensType) -> LensResult | None:
        """Get result for a specific lens."""
        return self.lens_results.get(lens_type)

    def get_feature(self, feature_id: str) -> FeatureActivation | None:
        """Find a specific feature across all lenses."""
        for act in self.all_features:
            if act.feature.id == feature_id:
                return act
        return None

    def top_features(self, k: int = 10, lens_type: LensType | None = None) -> list[FeatureActivation]:
        """
        Get top-k features by activation.

        Args:
            k: Number of features to return
            lens_type: Filter by lens type (optional)
        """
        features = self.all_features
        if lens_type:
            features = [f for f in features if f.feature.source.name == lens_type.name]
        return sorted(features, reverse=True)[:k]

    def explain(self, verbosity: int = 1) -> str:
        """
        Generate human-readable explanation of the trace.

        Args:
            verbosity: 1=brief, 2=moderate, 3=detailed
        """
        lines = [
            f"Dark Trace Analysis (ID: {self.trace_id})",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Status: {self.status.name}",
            f"Total Time: {self.total_time_ms:.2f}ms",
            "",
        ]

        # Safety summary
        if self.has_safety_concerns:
            lines.append(f"SAFETY CONCERNS: {self.safety_summary}")
            lines.append("")

        # Top features
        lines.append("Top Features (across all lenses):")
        for act in self.dominant_features[:5]:
            label = act.feature.label or act.feature.id
            source = act.feature.source.value
            lines.append(f"  [{source}] {label}: {act.activation:.3f}")

        if verbosity >= 2:
            # Per-lens results
            lines.append("\nPer-Lens Analysis:")
            for lens_type, result in self.lens_results.items():
                lines.append(f"\n  {lens_type.name} ({result.lens_name}):")
                lines.append(f"    Status: {result.status.name}")
                lines.append(f"    Time: {result.compute_time_ms:.2f}ms")
                if result.reconstruction_loss is not None:
                    lines.append(f"    Reconstruction Loss: {result.reconstruction_loss:.4f}")
                if result.sparsity is not None:
                    lines.append(f"    Sparsity: {result.sparsity:.3f}")

                top = result.top_features(3)
                if top:
                    lines.append("    Top Features:")
                    for act in top:
                        lines.append(f"      • {act.feature.label or act.feature.id}: {act.activation:.3f}")

        if verbosity >= 3:
            # SAE-Semantic correlations
            if self.sae_semantic_correlations:
                lines.append("\nSAE ↔ Semantic Correlations:")
                for sae_feat, correlations in list(self.sae_semantic_correlations.items())[:5]:
                    lines.append(f"  {sae_feat}:")
                    for sem_feat, corr in correlations[:3]:
                        lines.append(f"    → {sem_feat}: {corr:.3f}")

            # Safety details
            if self.safety_flags:
                lines.append("\nSafety Flags:")
                for feat_id, flag in self.safety_flags.items():
                    if flag != SafetyFlag.SAFE:
                        lines.append(f"  {feat_id}: {flag.name}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging/storage."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.name,
            "input_shape": self.input_shape,
            "layer_name": self.layer_name,
            "lens_results": {
                k.name: v.to_dict() for k, v in self.lens_results.items()
            },
            "dominant_features": [
                {"id": act.feature.id, "activation": act.activation, "source": act.feature.source.value}
                for act in self.dominant_features
            ],
            "has_safety_concerns": self.has_safety_concerns,
            "safety_summary": self.safety_summary,
            "total_time_ms": self.total_time_ms,
            "lenses_used": self.lenses_used,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def create_failed(cls, trace_id: str, error: str) -> "TraceResult":
        """Create a failed trace result."""
        return cls(
            trace_id=trace_id,
            status=AnalysisStatus.FAILED,
            error_messages=[error],
        )


# =============================================================================
# Steering and Intervention Results
# =============================================================================

@dataclass
class SteeringResult:
    """Result from applying a steering vector."""
    # Input
    steering_vector: SteeringVector
    target_features: dict[str, float]

    # Outcome
    outcome: SteeringOutcome
    features_applied: list[str] = field(default_factory=list)
    features_blocked: list[str] = field(default_factory=list)

    # Validation
    pre_steering_activations: dict[str, float] | None = None
    post_steering_activations: dict[str, float] | None = None
    effectiveness_scores: dict[str, float] = field(default_factory=dict)  # How well did steering work?

    # Safety
    safety_validated: bool = False
    safety_warnings: list[str] = field(default_factory=list)

    def effectiveness(self) -> float:
        """Overall steering effectiveness (0-1)."""
        if not self.effectiveness_scores:
            return 0.0
        return sum(self.effectiveness_scores.values()) / len(self.effectiveness_scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.name,
            "features_applied": self.features_applied,
            "features_blocked": self.features_blocked,
            "effectiveness": self.effectiveness(),
            "safety_validated": self.safety_validated,
            "safety_warnings": self.safety_warnings,
        }


@dataclass
class AblationResult:
    """Result from ablating (removing) features."""
    # Input
    ablated_features: list[str]
    original_activations: torch.Tensor | None = None
    modified_activations: torch.Tensor | None = None

    # Impact measurement
    output_change: float | None = None          # Change in model output
    confidence_change: float | None = None      # Change in confidence
    behavior_change: str | None = None          # Qualitative description

    # Per-feature impact
    feature_impacts: dict[str, float] = field(default_factory=dict)

    # Statistical significance
    is_significant: bool = False
    p_value: float | None = None

    def most_impactful_feature(self) -> tuple[str, float] | None:
        """Get the feature with highest impact."""
        if not self.feature_impacts:
            return None
        return max(self.feature_impacts.items(), key=lambda x: abs(x[1]))


@dataclass
class InjectionResult:
    """Result from injecting feature values."""
    # Input
    injected_features: dict[str, float]
    original_activations: torch.Tensor | None = None
    modified_activations: torch.Tensor | None = None

    # Impact measurement
    output_change: float | None = None
    expected_behavior: str | None = None
    actual_behavior: str | None = None
    match_expected: bool = False

    # Dose-response (if multiple strengths tested)
    dose_response_curve: dict[float, float] | None = None


@dataclass
class PatchResult:
    """Result from patching activations between inputs."""
    # Input
    patched_features: list[str]
    source_description: str | None = None
    target_description: str | None = None

    # Results
    behavior_transferred: bool = False
    transfer_strength: float = 0.0  # 0-1, how much behavior transferred

    # Causal inference
    causal_evidence: float = 0.0  # Strength of causal claim (0-1)
    confounders_checked: list[str] = field(default_factory=list)


# =============================================================================
# Circuit Analysis Results
# =============================================================================

@dataclass
class CircuitNode:
    """A node in a mechanistic circuit."""
    node_id: str
    node_type: str              # "attention_head", "mlp", "residual", etc.
    layer: int
    position: int | None = None

    # Attribution
    attribution_score: float = 0.0
    direct_effect: float = 0.0
    indirect_effect: float = 0.0

    # Features
    associated_features: list[str] = field(default_factory=list)


@dataclass
class CircuitEdge:
    """An edge in a mechanistic circuit."""
    source: str
    target: str
    edge_type: str              # "attention", "residual", "mlp"
    weight: float = 0.0

    # Causality
    is_causal: bool = False
    causal_strength: float = 0.0


@dataclass
class CircuitTrace:
    """Result from circuit discovery/tracing."""
    # Input
    input_description: str
    output_description: str

    # Circuit structure
    nodes: list[CircuitNode] = field(default_factory=list)
    edges: list[CircuitEdge] = field(default_factory=list)

    # Critical path
    critical_path: list[str] = field(default_factory=list)  # Node IDs in order
    bottleneck_nodes: list[str] = field(default_factory=list)

    # Vulnerability analysis
    single_points_of_failure: list[str] = field(default_factory=list)
    adversarial_targets: list[str] = field(default_factory=list)

    # Visualization
    visualization_data: dict[str, Any] | None = None

    def get_node(self, node_id: str) -> CircuitNode | None:
        """Get a node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_edges_from(self, node_id: str) -> list[CircuitEdge]:
        """Get all edges from a node."""
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> list[CircuitEdge]:
        """Get all edges to a node."""
        return [e for e in self.edges if e.target == node_id]


# =============================================================================
# Evaluation Metrics
# =============================================================================

@dataclass
class LensEvaluationMetrics:
    """Evaluation metrics for a single lens."""
    lens_name: str
    lens_type: LensType

    # Reconstruction quality
    mean_reconstruction_loss: float = 0.0
    reconstruction_r2: float = 0.0          # R² score

    # Sparsity
    mean_sparsity: float = 0.0              # Average L0 norm / n_features
    dead_feature_ratio: float = 0.0         # Features that never activate

    # Interpretability
    monosemanticity_score: float = 0.0      # How monosemantic are features
    human_interpretability: float | None = None  # Human eval score

    # Causal validity
    ablation_effectiveness: float = 0.0      # How effective are ablations
    steering_effectiveness: float = 0.0      # How effective is steering

    # Sample size
    n_samples: int = 0


@dataclass
class DarkTraceEvaluationReport:
    """Complete evaluation report for Dark Trace system."""
    # Identification
    report_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    evaluation_dataset: str | None = None

    # Per-lens metrics
    lens_metrics: dict[LensType, LensEvaluationMetrics] = field(default_factory=dict)

    # Cross-lens metrics
    sae_semantic_correlation_strength: float = 0.0
    cross_lens_agreement: float = 0.0       # How much do lenses agree?

    # Safety metrics
    safety_feature_coverage: float = 0.0    # What % of safety features detected
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

    # Overall
    overall_quality_score: float = 0.0

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Dark Trace Evaluation Report",
            f"ID: {self.report_id}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Overall Quality: {self.overall_quality_score:.3f}",
            "",
            "Per-Lens Metrics:",
        ]

        for lens_type, metrics in self.lens_metrics.items():
            lines.append(f"\n  {lens_type.name}:")
            lines.append(f"    Reconstruction Loss: {metrics.mean_reconstruction_loss:.4f}")
            lines.append(f"    Sparsity: {metrics.mean_sparsity:.3f}")
            lines.append(f"    Dead Features: {metrics.dead_feature_ratio:.1%}")
            lines.append(f"    Monosemanticity: {metrics.monosemanticity_score:.3f}")

        lines.append("\nSafety Metrics:")
        lines.append(f"  Coverage: {self.safety_feature_coverage:.1%}")
        lines.append(f"  False Positive Rate: {self.false_positive_rate:.1%}")
        lines.append(f"  False Negative Rate: {self.false_negative_rate:.1%}")

        return "\n".join(lines)
