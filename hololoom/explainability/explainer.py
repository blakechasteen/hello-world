"""
Unified Explainer - One API for all explanation techniques

Combines all 7 explainability techniques into a single, coherent API:
1. Feature Attribution (SHAP/LIME)
2. Attention Visualization
3. Counterfactual Generation
4. Natural Language Explanations
5. Decision Tree Extraction
6. Provenance Tracking
7. Integrated Explanations

This is the main entry point for Layer 5: Explainability.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from .feature_attribution import FeatureAttributor, AttributionMethod, FeatureImportance
from .attention_explainer import AttentionExplainer, AttentionHeatmap
from .counterfactual_generator import CounterfactualGenerator, Counterfactual, CounterfactualMethod
from .natural_language import NaturalLanguageExplainer, ExplanationType
from .decision_tree_extractor import DecisionTreeExtractor, RuleSet
from .provenance_tracker import ProvenanceTracker, LineageGraph


@dataclass
class Explanation:
    """
    Unified explanation containing all explanation types.

    This is the main output of the explainability system.
    """
    # Core decision info
    decision: Any
    confidence: float = 1.0
    inputs: Optional[Dict[str, Any]] = None

    # 1. Feature Attribution
    feature_importances: List[FeatureImportance] = field(default_factory=list)

    # 2. Attention Visualization
    attention_heatmaps: List[AttentionHeatmap] = field(default_factory=list)

    # 3. Counterfactuals
    counterfactuals: List[Counterfactual] = field(default_factory=list)

    # 4. Natural Language
    natural_language_explanation: str = ""

    # 5. Decision Rules
    rules: Optional[RuleSet] = None

    # 6. Provenance
    lineage: Optional[LineageGraph] = None

    # Metadata
    explanation_method: str = "unified"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary of explanation"""
        lines = []
        lines.append("=" * 80)
        lines.append("UNIFIED EXPLANATION")
        lines.append("=" * 80)
        lines.append(f"Decision: {self.decision}")
        lines.append(f"Confidence: {self.confidence:.0%}")
        lines.append("")

        # Feature Attribution
        if self.feature_importances:
            lines.append("TOP 5 FEATURE CONTRIBUTIONS:")
            for feat in self.feature_importances[:5]:
                sign = "+" if feat.is_positive else "-"
                bar_len = int(abs(feat.importance) * 40)
                bar = "█" * bar_len
                lines.append(f"  {feat.feature_name:20s} {sign}{abs(feat.importance):.3f} [{bar}]")
            lines.append("")

        # Attention (if available)
        if self.attention_heatmaps:
            lines.append("ATTENTION PATTERNS:")
            for heatmap in self.attention_heatmaps[:2]:
                lines.append(f"  Layer {heatmap.layer_id}, Head {heatmap.head_id}: {heatmap.pattern.value}")
            lines.append("")

        # Counterfactuals
        if self.counterfactuals:
            lines.append("COUNTERFACTUALS (What if...):")
            for i, cf in enumerate(self.counterfactuals[:3], 1):
                lines.append(f"  {i}. Change {cf.num_changes} feature(s) → {cf.counterfactual_prediction}")
            lines.append("")

        # Natural Language
        if self.natural_language_explanation:
            lines.append("EXPLANATION:")
            lines.append(f"  {self.natural_language_explanation}")
            lines.append("")

        # Rules (if available)
        if self.rules and len(self.rules.rules) > 0:
            lines.append("DECISION RULES:")
            for rule in self.rules.rules[:3]:
                lines.append(f"  • {rule}")
            lines.append("")

        # Provenance
        if self.lineage:
            lines.append("COMPUTATIONAL LINEAGE:")
            lines.append(f"  Total Duration: {self.lineage.total_duration():.1f}ms")
            lines.append(f"  Steps: {len(self.lineage.traces)}")
            bottlenecks = self.lineage.bottleneck_stages(threshold=0.2)
            if bottlenecks:
                lines.append(f"  Bottlenecks: {len(bottlenecks)} stage(s)")

        lines.append("=" * 80)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'decision': self.decision,
            'confidence': self.confidence,
            'feature_importances': [
                {
                    'feature': f.feature_name,
                    'importance': f.importance,
                    'rank': f.rank
                }
                for f in self.feature_importances
            ],
            'counterfactuals': [
                {
                    'num_changes': cf.num_changes,
                    'prediction': cf.counterfactual_prediction,
                    'distance': cf.distance
                }
                for cf in self.counterfactuals
            ],
            'explanation': self.natural_language_explanation,
            'lineage': {
                'total_duration_ms': self.lineage.total_duration() if self.lineage else 0,
                'num_steps': len(self.lineage.traces) if self.lineage else 0
            } if self.lineage else None
        }


class UnifiedExplainer:
    """
    Unified explainer combining all 7 explainability techniques.

    This is the main entry point for Layer 5: Explainability.
    """

    def __init__(
        self,
        model: Optional[Callable] = None,
        enable_attribution: bool = True,
        enable_attention: bool = True,
        enable_counterfactuals: bool = True,
        enable_natural_language: bool = True,
        enable_rules: bool = False,  # Expensive
        enable_provenance: bool = True,
        twin_network: Optional[Any] = None  # From Layer 4
    ):
        """
        Args:
            model: Black-box model to explain
            enable_attribution: Enable feature attribution (SHAP/LIME)
            enable_attention: Enable attention visualization
            enable_counterfactuals: Enable counterfactual generation
            enable_natural_language: Enable natural language explanations
            enable_rules: Enable decision tree/rule extraction (expensive)
            enable_provenance: Enable provenance tracking
            twin_network: Twin network from Layer 4 (for counterfactuals)
        """
        self.model = model
        self.twin_network = twin_network

        # Initialize components
        self.attributor = FeatureAttributor(model=model) if enable_attribution else None
        self.attention_explainer = AttentionExplainer(model=model) if enable_attention else None
        self.counterfactual_generator = CounterfactualGenerator(
            model=model,
            twin_network=twin_network
        ) if enable_counterfactuals else None
        self.nl_explainer = NaturalLanguageExplainer() if enable_natural_language else None
        self.tree_extractor = DecisionTreeExtractor(model=model) if enable_rules else None
        self.provenance_tracker = ProvenanceTracker() if enable_provenance else None

        self.enable_rules = enable_rules

    def explain(
        self,
        features: Dict[str, Any],
        decision: Optional[Any] = None,
        target_prediction: Optional[Any] = None,
        input_tokens: Optional[List[str]] = None,
        training_data: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 1.0
    ) -> Explanation:
        """
        Generate unified explanation.

        Args:
            features: Input features
            decision: Model's decision (if not provided, will query model)
            target_prediction: Target for counterfactuals
            input_tokens: Input tokens for attention visualization
            training_data: Training data for rule extraction
            confidence: Model confidence in decision

        Returns:
            Unified explanation with all techniques
        """
        # 1. Get decision (if not provided)
        if decision is None and self.model is not None:
            decision = self._evaluate_model(features)

        # Start provenance tracking
        if self.provenance_tracker:
            self.provenance_tracker.record_input(features)
            self.provenance_tracker.start_timing("explanation")

        explanation = Explanation(
            decision=decision,
            confidence=confidence,
            inputs=features
        )

        # 2. Feature Attribution
        if self.attributor:
            if self.provenance_tracker:
                self.provenance_tracker.start_timing("attribution")

            explanation.feature_importances = self.attributor.attribute(
                features,
                decision
            )

            if self.provenance_tracker:
                duration = self.provenance_tracker.end_timing("attribution")
                self.provenance_tracker.record_computation(
                    stage="attribution",
                    inputs=features,
                    outputs={'num_features': len(explanation.feature_importances)},
                    duration_ms=duration
                )

        # 3. Attention Visualization
        if self.attention_explainer and input_tokens:
            if self.provenance_tracker:
                self.provenance_tracker.start_timing("attention")

            explanation.attention_heatmaps = self.attention_explainer.extract_attention(
                input_tokens
            )

            if self.provenance_tracker:
                duration = self.provenance_tracker.end_timing("attention")
                self.provenance_tracker.record_computation(
                    stage="attention",
                    inputs={'tokens': input_tokens},
                    outputs={'num_heatmaps': len(explanation.attention_heatmaps)},
                    duration_ms=duration
                )

        # 4. Counterfactual Generation
        if self.counterfactual_generator and target_prediction:
            if self.provenance_tracker:
                self.provenance_tracker.start_timing("counterfactuals")

            explanation.counterfactuals = self.counterfactual_generator.generate(
                features,
                target_prediction,
                current_prediction=decision,
                num_counterfactuals=3
            )

            if self.provenance_tracker:
                duration = self.provenance_tracker.end_timing("counterfactuals")
                self.provenance_tracker.record_computation(
                    stage="counterfactuals",
                    inputs=features,
                    outputs={'num_counterfactuals': len(explanation.counterfactuals)},
                    duration_ms=duration
                )

        # 5. Natural Language Explanation
        if self.nl_explainer:
            if self.provenance_tracker:
                self.provenance_tracker.start_timing("natural_language")

            nl_explanation = self.nl_explainer.explain(
                decision,
                explanation.feature_importances,
                counterfactuals=explanation.counterfactuals,
                confidence=confidence,
                explanation_type=ExplanationType.WHY
            )
            explanation.natural_language_explanation = nl_explanation.text

            if self.provenance_tracker:
                duration = self.provenance_tracker.end_timing("natural_language")
                self.provenance_tracker.record_computation(
                    stage="natural_language",
                    inputs={'decision': decision},
                    outputs={'explanation_length': len(nl_explanation.text)},
                    duration_ms=duration
                )

        # 6. Rule Extraction (expensive, only if enabled and training data provided)
        if self.enable_rules and self.tree_extractor and training_data:
            if self.provenance_tracker:
                self.provenance_tracker.start_timing("rule_extraction")

            tree = self.tree_extractor.extract(training_data)
            explanation.rules = self.tree_extractor.extract_rules()

            if self.provenance_tracker:
                duration = self.provenance_tracker.end_timing("rule_extraction")
                self.provenance_tracker.record_computation(
                    stage="rule_extraction",
                    inputs={'num_training_examples': len(training_data)},
                    outputs={'num_rules': len(explanation.rules.rules)},
                    duration_ms=duration
                )

        # 7. Finalize Provenance
        if self.provenance_tracker:
            total_duration = self.provenance_tracker.end_timing("explanation")
            self.provenance_tracker.record_output({'decision': decision})
            explanation.lineage = self.provenance_tracker.get_lineage()

            # Add provenance metadata
            explanation.metadata['total_duration_ms'] = total_duration
            explanation.metadata['bottlenecks'] = [
                {'stage': t.stage, 'duration_ms': t.duration_ms}
                for t in explanation.lineage.bottleneck_stages()
            ]

        return explanation

    def _evaluate_model(self, features: Dict[str, Any]) -> Any:
        """Evaluate model on features"""
        if self.model is None:
            return None

        result = self.model(features)

        # Convert to simple type
        try:
            import torch
            if isinstance(result, torch.Tensor):
                if result.numel() == 1:
                    return float(result.item())
                else:
                    return result.argmax().item()
        except ImportError:
            pass

        if isinstance(result, (list, tuple)):
            return result[0] if len(result) > 0 else None

        return result


def explain(
    model: Callable,
    features: Dict[str, Any],
    target_prediction: Optional[Any] = None,
    confidence: float = 1.0,
    enable_all: bool = True
) -> Explanation:
    """
    Convenience function to explain a model's decision.

    Args:
        model: Black-box model
        features: Input features
        target_prediction: Target for counterfactuals
        confidence: Model confidence
        enable_all: Enable all explainability techniques

    Returns:
        Unified explanation
    """
    explainer = UnifiedExplainer(
        model=model,
        enable_attribution=enable_all,
        enable_attention=enable_all,
        enable_counterfactuals=enable_all and target_prediction is not None,
        enable_natural_language=enable_all,
        enable_rules=False,  # Too expensive for quick explains
        enable_provenance=enable_all
    )

    return explainer.explain(
        features=features,
        target_prediction=target_prediction,
        confidence=confidence
    )
