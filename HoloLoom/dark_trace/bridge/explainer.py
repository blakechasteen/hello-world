"""
Dark Trace Bridge: Bridged Explanations

Generates human-readable explanations that combine SAE feature activations
with semantic interpretations.

Key Features:
- "Feature 42 = Warmth (0.85) + Formality (0.62)" style explanations
- Multi-level verbosity (brief, moderate, detailed)
- Comparison explanations (why response A differs from B)
- Steering recommendations based on explanations

Research Basis:
- SAE features alone are opaque (just indices)
- Semantic axes provide interpretability
- Bridging enables human-understandable explanations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import torch

from HoloLoom.dark_trace.protocol import TraceLens, FeatureActivation, LensType
from HoloLoom.dark_trace.bridge.mapper import (
    FeatureSemanticMapper,
    FeatureMapping,
    SemanticMeaning,
)


class ExplanationVerbosity(Enum):
    """Verbosity levels for explanations."""

    BRIEF = 1       # One-line summary
    MODERATE = 2    # Key features with semantic meanings
    DETAILED = 3    # Full breakdown with statistics


@dataclass
class FeatureExplanation:
    """
    Explanation for a single feature activation.

    Combines SAE feature activation with semantic interpretation.
    """

    # SAE feature info
    feature_id: str
    feature_index: int
    activation: float

    # Semantic interpretation
    semantic_signature: str  # e.g., "Warmth(+0.85) + Formality(-0.62)"
    top_meanings: List[SemanticMeaning] = field(default_factory=list)

    # Human label if available
    human_label: Optional[str] = None

    # Confidence in interpretation
    interpretation_confidence: float = 0.0

    def to_brief(self) -> str:
        """One-line summary."""
        if self.human_label:
            return f"{self.feature_id} ({self.human_label}): {self.activation:.3f}"
        elif self.semantic_signature and self.semantic_signature != "unmapped":
            return f"{self.feature_id} = {self.semantic_signature}: {self.activation:.3f}"
        else:
            return f"{self.feature_id}: {self.activation:.3f} (unmapped)"

    def to_moderate(self) -> str:
        """Moderate detail explanation."""
        lines = [self.to_brief()]

        if self.top_meanings:
            for meaning in self.top_meanings[:3]:
                direction = "↑" if meaning.correlation > 0 else "↓"
                lines.append(
                    f"  {direction} {meaning.dimension}: {meaning.correlation:+.3f}"
                )

        return "\n".join(lines)

    def to_detailed(self) -> str:
        """Full detailed explanation."""
        lines = [
            f"Feature: {self.feature_id}",
            f"  Activation: {self.activation:.4f}",
        ]

        if self.human_label:
            lines.append(f"  Label: {self.human_label}")

        lines.append(f"  Signature: {self.semantic_signature}")
        lines.append(f"  Confidence: {self.interpretation_confidence:.1%}")

        if self.top_meanings:
            lines.append("  Semantic Components:")
            for meaning in self.top_meanings[:5]:
                lines.append(f"    • {meaning.describe()}")

        return "\n".join(lines)


@dataclass
class BridgedExplanation:
    """
    Complete bridged explanation for model activations.

    Combines multiple feature explanations into a coherent narrative.
    """

    # Feature-level explanations
    feature_explanations: List[FeatureExplanation] = field(default_factory=list)

    # Aggregate semantic profile
    semantic_profile: Dict[str, float] = field(default_factory=dict)

    # Summary statistics
    n_active_features: int = 0
    n_well_mapped: int = 0
    n_unmapped: int = 0
    mean_activation: float = 0.0

    # Overall interpretation
    dominant_dimensions: List[Tuple[str, float]] = field(default_factory=list)
    interpretation_summary: str = ""

    # Metadata
    verbosity: ExplanationVerbosity = ExplanationVerbosity.MODERATE

    def to_string(self, verbosity: Optional[ExplanationVerbosity] = None) -> str:
        """
        Generate explanation string at specified verbosity.

        Args:
            verbosity: Override default verbosity

        Returns:
            Human-readable explanation
        """
        v = verbosity or self.verbosity

        if v == ExplanationVerbosity.BRIEF:
            return self._to_brief()
        elif v == ExplanationVerbosity.MODERATE:
            return self._to_moderate()
        else:
            return self._to_detailed()

    def _to_brief(self) -> str:
        """One-line summary."""
        if self.interpretation_summary:
            return self.interpretation_summary

        if not self.feature_explanations:
            return "No significant feature activations."

        # Top features summary
        top_features = sorted(
            self.feature_explanations,
            key=lambda f: abs(f.activation),
            reverse=True
        )[:3]

        parts = [f.to_brief() for f in top_features]
        return " | ".join(parts)

    def _to_moderate(self) -> str:
        """Moderate detail explanation."""
        lines = ["Bridged Explanation:"]

        # Stats
        lines.append(
            f"\nActive Features: {self.n_active_features} "
            f"({self.n_well_mapped} mapped, {self.n_unmapped} unmapped)"
        )

        # Dominant dimensions
        if self.dominant_dimensions:
            lines.append("\nDominant Semantic Dimensions:")
            for dim, strength in self.dominant_dimensions[:5]:
                sign = "+" if strength > 0 else ""
                lines.append(f"  • {dim}: {sign}{strength:.3f}")

        # Top features
        if self.feature_explanations:
            lines.append("\nTop Active Features:")
            for fe in sorted(
                self.feature_explanations,
                key=lambda f: abs(f.activation),
                reverse=True
            )[:5]:
                lines.append(f"  {fe.to_brief()}")

        return "\n".join(lines)

    def _to_detailed(self) -> str:
        """Full detailed explanation."""
        lines = [
            "=" * 60,
            "BRIDGED EXPLANATION",
            "=" * 60,
        ]

        # Summary
        lines.extend([
            "",
            f"Summary: {self.interpretation_summary}" if self.interpretation_summary else "",
            "",
            "Statistics:",
            f"  Active Features: {self.n_active_features}",
            f"  Well-Mapped: {self.n_well_mapped}",
            f"  Unmapped: {self.n_unmapped}",
            f"  Mean Activation: {self.mean_activation:.4f}",
        ])

        # Semantic profile
        if self.semantic_profile:
            lines.extend([
                "",
                "Aggregate Semantic Profile:",
            ])
            sorted_profile = sorted(
                self.semantic_profile.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for dim, strength in sorted_profile[:10]:
                sign = "+" if strength > 0 else ""
                bar = "█" * int(abs(strength) * 20)
                lines.append(f"  {dim:20s} {sign}{strength:+.3f} {bar}")

        # Feature details
        if self.feature_explanations:
            lines.extend([
                "",
                "Feature Details:",
                "-" * 40,
            ])
            for fe in sorted(
                self.feature_explanations,
                key=lambda f: abs(f.activation),
                reverse=True
            )[:10]:
                lines.append("")
                lines.append(fe.to_detailed())

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


class BridgedExplainer:
    """
    Generates bridged explanations combining SAE and semantic interpretations.

    Produces human-readable explanations that connect learned SAE features
    to interpretable semantic axes.

    Usage:
        explainer = BridgedExplainer(sae_lens, semantic_lens, mapper)

        # Explain activations
        explanation = explainer.explain(activations)
        print(explanation.to_string())

        # Compare two responses
        comparison = explainer.compare(activations1, activations2)
        print(comparison)

        # Get steering recommendations
        recommendations = explainer.recommend_steering(
            current=activations,
            goal={"Warmth": 0.8, "Formality": -0.5}
        )
    """

    def __init__(
        self,
        sae_lens: TraceLens,
        semantic_lens: TraceLens,
        mapper: FeatureSemanticMapper,
        default_verbosity: ExplanationVerbosity = ExplanationVerbosity.MODERATE,
    ):
        """
        Initialize BridgedExplainer.

        Args:
            sae_lens: SAELens for feature extraction
            semantic_lens: SemanticLens for semantic projection
            mapper: FeatureSemanticMapper for bridging
            default_verbosity: Default explanation verbosity
        """
        self.sae_lens = sae_lens
        self.semantic_lens = semantic_lens
        self.mapper = mapper
        self.default_verbosity = default_verbosity

    def explain(
        self,
        activations: torch.Tensor,
        threshold: float = 0.0,
        top_k: Optional[int] = 10,
        verbosity: Optional[ExplanationVerbosity] = None,
    ) -> BridgedExplanation:
        """
        Generate bridged explanation for activations.

        Args:
            activations: Input activations [batch, dim] or [dim]
            threshold: Minimum activation to include
            top_k: Maximum features to explain
            verbosity: Explanation verbosity

        Returns:
            BridgedExplanation with feature-level and aggregate explanations
        """
        verbosity = verbosity or self.default_verbosity

        # Get active SAE features
        sae_activations = self.sae_lens.get_active(
            activations, threshold=threshold, top_k=top_k
        )

        # Get semantic projections
        semantic_projections = {}
        try:
            semantic_acts = self.semantic_lens.get_active(
                activations, threshold=0.1, top_k=20
            )
            for sa in semantic_acts:
                semantic_projections[sa.feature.label] = sa.activation
        except Exception:
            pass  # Semantic lens may not have axes learned yet

        # Build feature explanations
        feature_explanations = []
        n_well_mapped = 0
        n_unmapped = 0

        for fa in sae_activations:
            # Get semantic mapping for this feature
            mapping = self.mapper.get_feature_mapping(fa.feature.id)

            if mapping is not None:
                semantic_signature = mapping.semantic_signature
                top_meanings = mapping.semantic_meanings[:3]
                human_label = mapping.human_label
                is_mapped = mapping.is_well_mapped

                # Compute interpretation confidence
                if top_meanings:
                    interp_conf = np.mean([m.confidence for m in top_meanings])
                else:
                    interp_conf = 0.0
            else:
                semantic_signature = "unmapped"
                top_meanings = []
                human_label = None
                is_mapped = False
                interp_conf = 0.0

            if is_mapped:
                n_well_mapped += 1
            else:
                n_unmapped += 1

            fe = FeatureExplanation(
                feature_id=fa.feature.id,
                feature_index=fa.feature.index,
                activation=fa.activation,
                semantic_signature=semantic_signature,
                top_meanings=top_meanings,
                human_label=human_label,
                interpretation_confidence=interp_conf,
            )
            feature_explanations.append(fe)

        # Build aggregate semantic profile
        semantic_profile = self.mapper.get_composite_mapping(
            feature_ids=[fe.feature_id for fe in feature_explanations],
            weights=[fe.activation for fe in feature_explanations]
        )

        # Get dominant dimensions
        dominant_dimensions = sorted(
            semantic_profile.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        # Generate interpretation summary
        summary = self._generate_summary(
            feature_explanations, dominant_dimensions
        )

        # Mean activation
        if feature_explanations:
            mean_act = np.mean([fe.activation for fe in feature_explanations])
        else:
            mean_act = 0.0

        return BridgedExplanation(
            feature_explanations=feature_explanations,
            semantic_profile=semantic_profile,
            n_active_features=len(feature_explanations),
            n_well_mapped=n_well_mapped,
            n_unmapped=n_unmapped,
            mean_activation=mean_act,
            dominant_dimensions=dominant_dimensions,
            interpretation_summary=summary,
            verbosity=verbosity,
        )

    def _generate_summary(
        self,
        features: List[FeatureExplanation],
        dominant: List[Tuple[str, float]],
    ) -> str:
        """Generate one-line interpretation summary."""
        if not features:
            return "No significant activations detected."

        if not dominant:
            return f"{len(features)} features active (semantically unmapped)"

        # Build summary from dominant dimensions
        parts = []
        for dim, strength in dominant[:3]:
            if strength > 0.3:
                parts.append(f"high {dim}")
            elif strength < -0.3:
                parts.append(f"low {dim}")

        if parts:
            return f"Response characterized by: {', '.join(parts)}"
        else:
            return f"{len(features)} features active with moderate semantic profile"

    def compare(
        self,
        activations1: torch.Tensor,
        activations2: torch.Tensor,
        labels: Tuple[str, str] = ("Response A", "Response B"),
    ) -> str:
        """
        Compare two sets of activations and explain differences.

        Args:
            activations1: First activation tensor
            activations2: Second activation tensor
            labels: Labels for the two responses

        Returns:
            Human-readable comparison explanation
        """
        exp1 = self.explain(activations1, verbosity=ExplanationVerbosity.BRIEF)
        exp2 = self.explain(activations2, verbosity=ExplanationVerbosity.BRIEF)

        lines = [
            "Comparison:",
            f"  {labels[0]}: {exp1.interpretation_summary}",
            f"  {labels[1]}: {exp2.interpretation_summary}",
            "",
            "Semantic Differences:",
        ]

        # Compute semantic differences
        all_dims = set(exp1.semantic_profile.keys()) | set(exp2.semantic_profile.keys())
        differences = []

        for dim in all_dims:
            val1 = exp1.semantic_profile.get(dim, 0.0)
            val2 = exp2.semantic_profile.get(dim, 0.0)
            diff = val2 - val1

            if abs(diff) > 0.1:  # Significant difference
                differences.append((dim, diff, val1, val2))

        # Sort by absolute difference
        differences.sort(key=lambda x: abs(x[1]), reverse=True)

        for dim, diff, val1, val2 in differences[:5]:
            direction = "↑" if diff > 0 else "↓"
            lines.append(
                f"  {dim}: {val1:.2f} → {val2:.2f} ({direction}{abs(diff):.2f})"
            )

        if not differences:
            lines.append("  No significant semantic differences detected.")

        return "\n".join(lines)

    def recommend_steering(
        self,
        current: torch.Tensor,
        goal: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Recommend steering adjustments to move toward semantic goal.

        Args:
            current: Current activation tensor
            goal: Target semantic profile {dimension: target_value}

        Returns:
            Dict with steering recommendations
        """
        # Get current semantic profile
        current_exp = self.explain(current, verbosity=ExplanationVerbosity.BRIEF)

        recommendations = {
            "adjustments": [],
            "features_to_boost": [],
            "features_to_suppress": [],
            "expected_change": {},
        }

        for dim, target in goal.items():
            current_val = current_exp.semantic_profile.get(dim, 0.0)
            diff = target - current_val

            if abs(diff) < 0.1:
                continue  # Already close

            recommendations["adjustments"].append({
                "dimension": dim,
                "current": current_val,
                "target": target,
                "change_needed": diff,
            })

            # Find features to adjust
            positive_corr = diff > 0
            features = self.mapper.get_features_for_semantic(
                dim, k=3, positive_only=positive_corr
            )

            for feature_id, corr in features:
                if diff > 0 and corr > 0:
                    recommendations["features_to_boost"].append(feature_id)
                elif diff < 0 and corr < 0:
                    recommendations["features_to_boost"].append(feature_id)
                elif diff > 0 and corr < 0:
                    recommendations["features_to_suppress"].append(feature_id)
                elif diff < 0 and corr > 0:
                    recommendations["features_to_suppress"].append(feature_id)

            recommendations["expected_change"][dim] = diff

        return recommendations

    def explain_feature(
        self,
        feature_id: str,
        verbose: bool = False,
    ) -> str:
        """
        Explain a single SAE feature using semantic mapping.

        Args:
            feature_id: Feature ID (e.g., "sae.42")
            verbose: If True, include more detail

        Returns:
            Human-readable explanation
        """
        mapping = self.mapper.get_feature_mapping(feature_id)
        if mapping is None:
            return f"Feature {feature_id}: No semantic mapping available."

        return mapping.describe(verbose=verbose)

    def get_unmapped_features_report(self) -> str:
        """
        Generate report on unmapped features (potential novel dimensions).

        Returns:
            Report on features lacking strong semantic correlations
        """
        unmapped = self.mapper.get_unmapped_features()

        if not unmapped:
            return "All active features have semantic mappings."

        lines = [
            "Unmapped Features Report:",
            f"  Total Unmapped: {len(unmapped)}",
            "",
            "These features may represent novel semantic dimensions",
            "not captured by the existing 244 axes.",
            "",
            "Feature IDs:",
        ]

        for fid in unmapped[:20]:
            lines.append(f"  • {fid}")

        if len(unmapped) > 20:
            lines.append(f"  ... and {len(unmapped) - 20} more")

        return "\n".join(lines)


def create_bridged_explainer(
    sae_lens: TraceLens,
    semantic_lens: TraceLens,
    correlator: Any,  # SAESemanticCorrelator
    default_verbosity: ExplanationVerbosity = ExplanationVerbosity.MODERATE,
) -> BridgedExplainer:
    """
    Factory function to create BridgedExplainer from components.

    Args:
        sae_lens: SAELens for feature extraction
        semantic_lens: SemanticLens for semantic projection
        correlator: SAESemanticCorrelator with learned correlations
        default_verbosity: Default explanation verbosity

    Returns:
        Configured BridgedExplainer
    """
    # Create mapper from correlator
    mapper = FeatureSemanticMapper(correlator)

    return BridgedExplainer(
        sae_lens=sae_lens,
        semantic_lens=semantic_lens,
        mapper=mapper,
        default_verbosity=default_verbosity,
    )
