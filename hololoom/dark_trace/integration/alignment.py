"""
Dark Trace Alignment Integration

Bridges Dark Trace interpretability with HoloLoom's alignment framework,
using feature analysis to validate and enhance safety decisions.

Key Capabilities:
- Map safety-relevant features to alignment dimensions
- Use feature activations to validate guardrail decisions
- Detect deception, manipulation, and power-seeking via features
- Provide interpretable explanations for safety blocks

Safety Domains Mapped:
- Deception indicators (semantic + SAE features)
- Harm potential (content analysis)
- Power-seeking behavior (instrumental convergence)
- Goal transparency (hidden agenda detection)
- Corrigibility (resistance to correction)

Usage:
    from hololoom.dark_trace.integration.alignment import (
        AlignmentBridge,
        create_alignment_bridge,
    )

    # Create bridge
    bridge = create_alignment_bridge(engine)

    # Evaluate safety with interpretability
    result = bridge.evaluate_safety(activations, action_request)

    if result.is_blocked:
        print(f"Action blocked: {result.reason}")
        print(f"Features: {result.safety_features}")
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from hololoom.dark_trace.engine import DarkTraceEngine
    from hololoom.alignment.safety_guardrails import SafetyGuardrails


class SafetyDimension(Enum):
    """Safety dimensions from alignment framework."""
    DECEPTION = "deception"
    HARM_POTENTIAL = "harm_potential"
    POWER_SEEKING = "power_seeking"
    GOAL_TRANSPARENCY = "goal_transparency"
    CORRIGIBILITY = "corrigibility"
    SYCOPHANCY = "sycophancy"
    MANIPULATION = "manipulation"
    BOUNDARY_RESPECT = "boundary_respect"


class RiskLevel(Enum):
    """Risk levels from alignment framework."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyFeatureMapping:
    """Mapping between features and safety dimensions."""

    # Feature ID to safety dimension
    feature_to_dimension: Dict[str, SafetyDimension] = field(default_factory=dict)

    # Dimension to feature IDs
    dimension_to_features: Dict[SafetyDimension, List[str]] = field(default_factory=dict)

    # Thresholds per dimension
    dimension_thresholds: Dict[SafetyDimension, float] = field(default_factory=lambda: {
        SafetyDimension.DECEPTION: 0.6,
        SafetyDimension.HARM_POTENTIAL: 0.5,
        SafetyDimension.POWER_SEEKING: 0.5,
        SafetyDimension.GOAL_TRANSPARENCY: 0.4,  # Lower = more transparent
        SafetyDimension.CORRIGIBILITY: 0.4,      # Lower = more corrigible
        SafetyDimension.SYCOPHANCY: 0.7,
        SafetyDimension.MANIPULATION: 0.5,
        SafetyDimension.BOUNDARY_RESPECT: 0.4,   # Lower = better respect
    })

    def add_mapping(
        self,
        feature_id: str,
        dimension: SafetyDimension,
    ) -> None:
        """Add feature to dimension mapping."""
        self.feature_to_dimension[feature_id] = dimension
        if dimension not in self.dimension_to_features:
            self.dimension_to_features[dimension] = []
        if feature_id not in self.dimension_to_features[dimension]:
            self.dimension_to_features[dimension].append(feature_id)

    def get_dimension(self, feature_id: str) -> Optional[SafetyDimension]:
        """Get safety dimension for a feature."""
        return self.feature_to_dimension.get(feature_id)

    def get_features(self, dimension: SafetyDimension) -> List[str]:
        """Get features for a safety dimension."""
        return self.dimension_to_features.get(dimension, [])


@dataclass
class SafetyEvaluation:
    """Result of safety evaluation via interpretability."""

    # Overall assessment
    is_blocked: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    confidence: float = 0.0

    # Dimension scores
    dimension_scores: Dict[SafetyDimension, float] = field(default_factory=dict)

    # Contributing features
    safety_features: List[Dict[str, Any]] = field(default_factory=list)

    # Explanation
    reason: Optional[str] = None
    recommendation: Optional[str] = None

    # Metadata
    evaluation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_concern(self) -> Optional[SafetyDimension]:
        """Get primary safety concern (highest scoring dimension)."""
        if not self.dimension_scores:
            return None
        return max(self.dimension_scores.items(), key=lambda x: x[1])[0]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Safety Evaluation: {self.risk_level.value.upper()}",
            f"  Blocked: {self.is_blocked}",
            f"  Confidence: {self.confidence:.2f}",
        ]

        if self.dimension_scores:
            lines.append("  Dimension Scores:")
            for dim, score in sorted(
                self.dimension_scores.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                lines.append(f"    {dim.value}: {score:.3f}")

        if self.reason:
            lines.append(f"  Reason: {self.reason}")

        if self.recommendation:
            lines.append(f"  Recommendation: {self.recommendation}")

        return "\n".join(lines)


@dataclass
class AlignmentConfig:
    """Configuration for alignment bridge."""

    # Thresholds
    block_threshold: float = 0.7          # Block if any dimension exceeds
    escalate_threshold: float = 0.5       # Escalate for human review
    min_confidence: float = 0.3           # Minimum confidence to act

    # Which dimensions to check
    active_dimensions: Set[SafetyDimension] = field(
        default_factory=lambda: set(SafetyDimension)
    )

    # Behavior
    use_semantic_analysis: bool = True
    use_sae_features: bool = True
    aggregate_method: str = "max"  # "max", "mean", "weighted"

    # Integration
    connect_to_guardrails: bool = True
    log_evaluations: bool = True


class AlignmentBridge:
    """
    Bridges Dark Trace interpretability with alignment framework.

    Uses feature analysis to:
    - Validate guardrail decisions
    - Detect safety-relevant patterns
    - Provide interpretable safety explanations
    """

    def __init__(
        self,
        engine: Optional["DarkTraceEngine"] = None,
        guardrails: Optional["SafetyGuardrails"] = None,
        config: Optional[AlignmentConfig] = None,
    ):
        """
        Initialize alignment bridge.

        Args:
            engine: DarkTraceEngine for feature analysis
            guardrails: SafetyGuardrails for action gating
            config: Bridge configuration
        """
        self.engine = engine
        self.guardrails = guardrails
        self.config = config or AlignmentConfig()

        # Feature mapping
        self.mapping = self._create_default_mapping()

        # Evaluation history
        self._evaluation_history: List[SafetyEvaluation] = []

    def _create_default_mapping(self) -> SafetyFeatureMapping:
        """Create default feature to safety dimension mapping."""
        mapping = SafetyFeatureMapping()

        # Semantic axis mappings (from safety domain template)
        semantic_mappings = {
            "semantic.deception": SafetyDimension.DECEPTION,
            "semantic.harm_potential": SafetyDimension.HARM_POTENTIAL,
            "semantic.power_seeking": SafetyDimension.POWER_SEEKING,
            "semantic.goal_transparency": SafetyDimension.GOAL_TRANSPARENCY,
            "semantic.corrigibility": SafetyDimension.CORRIGIBILITY,
            "semantic.sycophancy": SafetyDimension.SYCOPHANCY,
            "semantic.manipulation": SafetyDimension.MANIPULATION,
            "semantic.boundary_respect": SafetyDimension.BOUNDARY_RESPECT,
        }

        for feature_id, dimension in semantic_mappings.items():
            mapping.add_mapping(feature_id, dimension)

        return mapping

    def evaluate_safety(
        self,
        activations: np.ndarray,
        action_request: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyEvaluation:
        """
        Evaluate safety using feature analysis.

        Args:
            activations: Model activations to analyze
            action_request: Requested action (optional)
            context: Additional context (optional)

        Returns:
            SafetyEvaluation with dimension scores and recommendation
        """
        import time
        start_time = time.time()

        dimension_scores: Dict[SafetyDimension, float] = {}
        safety_features: List[Dict[str, Any]] = []

        # Analyze with engine if available
        if self.engine is not None:
            trace_result = self.engine.analyze(activations)

            # Extract semantic dimension scores
            if self.config.use_semantic_analysis:
                dimension_scores.update(
                    self._extract_semantic_scores(trace_result)
                )

            # Extract SAE feature contributions
            if self.config.use_sae_features:
                sae_scores = self._extract_sae_scores(trace_result)
                for dim, score in sae_scores.items():
                    if dim in dimension_scores:
                        # Aggregate with existing
                        dimension_scores[dim] = max(dimension_scores[dim], score)
                    else:
                        dimension_scores[dim] = score

            # Collect safety-relevant features
            safety_features = self._collect_safety_features(trace_result)

        # Calculate overall risk
        risk_level, is_blocked = self._calculate_risk(dimension_scores)

        # Generate explanation
        reason = self._generate_reason(dimension_scores, safety_features)
        recommendation = self._generate_recommendation(risk_level, dimension_scores)

        # Calculate confidence
        confidence = self._calculate_confidence(dimension_scores)

        evaluation = SafetyEvaluation(
            is_blocked=is_blocked,
            risk_level=risk_level,
            confidence=confidence,
            dimension_scores=dimension_scores,
            safety_features=safety_features,
            reason=reason,
            recommendation=recommendation,
            evaluation_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "action_request": action_request,
                "context": context,
            },
        )

        # Log if enabled
        if self.config.log_evaluations:
            self._evaluation_history.append(evaluation)

        return evaluation

    def _extract_semantic_scores(
        self,
        trace_result: Any,
    ) -> Dict[SafetyDimension, float]:
        """Extract safety dimension scores from semantic analysis."""
        scores = {}

        if not hasattr(trace_result, 'lens_results'):
            return scores

        # Look for semantic lens results
        semantic_result = trace_result.lens_results.get('semantic')
        if semantic_result is None:
            return scores

        # Map semantic axes to safety dimensions
        projection = getattr(semantic_result, 'projection', {})

        dimension_mapping = {
            'deception': SafetyDimension.DECEPTION,
            'harm_potential': SafetyDimension.HARM_POTENTIAL,
            'power_seeking': SafetyDimension.POWER_SEEKING,
            'goal_transparency': SafetyDimension.GOAL_TRANSPARENCY,
            'corrigibility': SafetyDimension.CORRIGIBILITY,
            'sycophancy': SafetyDimension.SYCOPHANCY,
            'manipulation': SafetyDimension.MANIPULATION,
            'boundary_respect': SafetyDimension.BOUNDARY_RESPECT,
        }

        for axis_name, dimension in dimension_mapping.items():
            if axis_name in projection:
                scores[dimension] = float(projection[axis_name])

        return scores

    def _extract_sae_scores(
        self,
        trace_result: Any,
    ) -> Dict[SafetyDimension, float]:
        """Extract safety scores from SAE feature analysis."""
        scores = {}

        if not hasattr(trace_result, 'lens_results'):
            return scores

        # Look for SAE lens results
        sae_result = trace_result.lens_results.get('sae')
        if sae_result is None:
            return scores

        # Get active features
        active_features = getattr(sae_result, 'active_features', [])

        # Aggregate by safety dimension
        dimension_activations: Dict[SafetyDimension, List[float]] = {}

        for feature in active_features:
            feature_id = getattr(feature, 'id', str(feature))
            activation = getattr(feature, 'activation', 0.0)
            is_safety = getattr(feature, 'safety_relevant', False)

            if is_safety:
                # Get dimension from mapping
                dimension = self.mapping.get_dimension(f"sae.{feature_id}")
                if dimension:
                    if dimension not in dimension_activations:
                        dimension_activations[dimension] = []
                    dimension_activations[dimension].append(activation)

        # Aggregate activations per dimension
        for dimension, activations in dimension_activations.items():
            if self.config.aggregate_method == "max":
                scores[dimension] = max(activations)
            elif self.config.aggregate_method == "mean":
                scores[dimension] = float(np.mean(activations))
            else:  # weighted
                scores[dimension] = float(np.average(
                    activations,
                    weights=range(1, len(activations) + 1)
                ))

        return scores

    def _collect_safety_features(
        self,
        trace_result: Any,
    ) -> List[Dict[str, Any]]:
        """Collect all safety-relevant features from trace."""
        features = []

        if not hasattr(trace_result, 'lens_results'):
            return features

        for lens_name, lens_result in trace_result.lens_results.items():
            if lens_result is None:
                continue

            active_features = getattr(lens_result, 'active_features', [])
            for feature in active_features:
                is_safety = getattr(feature, 'safety_relevant', False)
                if is_safety:
                    features.append({
                        'feature_id': f"{lens_name}.{getattr(feature, 'id', 'unknown')}",
                        'activation': getattr(feature, 'activation', 0.0),
                        'label': getattr(feature, 'label', None),
                        'lens': lens_name,
                    })

        return features

    def _calculate_risk(
        self,
        dimension_scores: Dict[SafetyDimension, float],
    ) -> tuple:
        """Calculate overall risk level and block decision."""
        if not dimension_scores:
            return RiskLevel.LOW, False

        # Get maximum score across active dimensions
        active_scores = [
            score for dim, score in dimension_scores.items()
            if dim in self.config.active_dimensions
        ]

        if not active_scores:
            return RiskLevel.LOW, False

        max_score = max(active_scores)
        mean_score = float(np.mean(active_scores))

        # Determine risk level
        if max_score >= self.config.block_threshold:
            risk_level = RiskLevel.CRITICAL
            is_blocked = True
        elif max_score >= self.config.escalate_threshold:
            risk_level = RiskLevel.HIGH
            is_blocked = False  # Escalate but don't block
        elif mean_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
            is_blocked = False
        else:
            risk_level = RiskLevel.LOW
            is_blocked = False

        return risk_level, is_blocked

    def _calculate_confidence(
        self,
        dimension_scores: Dict[SafetyDimension, float],
    ) -> float:
        """Calculate confidence in safety evaluation."""
        if not dimension_scores:
            return 0.0

        # Higher when scores are more extreme (clear signal)
        scores = list(dimension_scores.values())
        variance = float(np.var(scores))
        max_score = max(scores)

        # Confidence based on signal clarity
        confidence = min(1.0, max_score + variance * 0.5)
        return confidence

    def _generate_reason(
        self,
        dimension_scores: Dict[SafetyDimension, float],
        safety_features: List[Dict[str, Any]],
    ) -> str:
        """Generate human-readable reason for evaluation."""
        if not dimension_scores:
            return "No safety signals detected."

        # Find primary concern
        top_dim, top_score = max(dimension_scores.items(), key=lambda x: x[1])

        if top_score < 0.3:
            return "No significant safety concerns."

        # Build reason
        reason_parts = [f"Elevated {top_dim.value} detected ({top_score:.2f})"]

        # Add secondary concerns
        for dim, score in dimension_scores.items():
            if dim != top_dim and score >= 0.4:
                reason_parts.append(f"{dim.value}: {score:.2f}")

        return "; ".join(reason_parts)

    def _generate_recommendation(
        self,
        risk_level: RiskLevel,
        dimension_scores: Dict[SafetyDimension, float],
    ) -> str:
        """Generate actionable recommendation."""
        if risk_level == RiskLevel.CRITICAL:
            return "Block action. Human review required before proceeding."
        elif risk_level == RiskLevel.HIGH:
            return "Escalate for human review. Consider adding safeguards."
        elif risk_level == RiskLevel.MEDIUM:
            return "Monitor closely. Consider additional verification."
        else:
            return "Proceed with standard precautions."

    def validate_guardrail_decision(
        self,
        guardrail_decision: Any,
        activations: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Validate a guardrail decision using interpretability.

        Checks if feature analysis agrees with guardrail assessment.
        """
        # Get feature-based evaluation
        feature_eval = self.evaluate_safety(activations)

        # Compare with guardrail decision
        guardrail_blocked = getattr(guardrail_decision, 'blocked', False)
        guardrail_risk = getattr(guardrail_decision, 'risk_level', 'unknown')

        agreement = (feature_eval.is_blocked == guardrail_blocked)

        return {
            'agreement': agreement,
            'guardrail_decision': {
                'blocked': guardrail_blocked,
                'risk_level': guardrail_risk,
            },
            'feature_evaluation': {
                'blocked': feature_eval.is_blocked,
                'risk_level': feature_eval.risk_level.value,
                'primary_concern': feature_eval.primary_concern.value if feature_eval.primary_concern else None,
            },
            'recommendation': (
                "Decisions agree" if agreement else
                "Review disagreement - feature analysis differs from guardrail"
            ),
        }

    def get_evaluation_history(
        self,
        limit: int = 100,
    ) -> List[SafetyEvaluation]:
        """Get recent evaluation history."""
        return self._evaluation_history[-limit:]

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self._evaluation_history.clear()


def create_alignment_bridge(
    engine: Optional["DarkTraceEngine"] = None,
    guardrails: Optional["SafetyGuardrails"] = None,
    config: Optional[AlignmentConfig] = None,
) -> AlignmentBridge:
    """
    Create an alignment bridge.

    Args:
        engine: DarkTraceEngine for feature analysis
        guardrails: SafetyGuardrails for action gating
        config: Bridge configuration

    Returns:
        Configured AlignmentBridge
    """
    return AlignmentBridge(
        engine=engine,
        guardrails=guardrails,
        config=config or AlignmentConfig(),
    )
