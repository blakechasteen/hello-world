"""
Dark Trace Causal: Hypothesis Validation Framework

Orchestrates ablation, injection, and patching tests to validate
causal hypotheses about feature functions with rigorous evidence.

Research Basis:
- Scientific method applied to interpretability
- Multiple evidence streams for robust conclusions
- Statistical rigor with effect sizes and confidence intervals
- Ground truth comparison when available

Key Concepts:
- Causal Hypothesis: Testable claim about feature function
- Causal Necessity: Ablation shows feature is required
- Causal Sufficiency: Injection shows feature is sufficient
- Validation Suite: Collection of tests for systematic validation

Usage:
    validator = CausalValidator(ablator, injector, patcher)

    # Define hypothesis
    hypothesis = CausalHypothesis(
        feature="sae.42",
        claim="Controls response warmth",
        expected_effect={"warmth": 0.5},
        test_type="behavioral"
    )

    # Validate with multiple evidence streams
    result = validator.validate(hypothesis, test_suite)

    if result.supported:
        print(f"Hypothesis supported (confidence: {result.confidence:.2f})")
        print(f"Evidence: {result.evidence_summary}")
    else:
        print(f"Hypothesis rejected: {result.rejection_reason}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
from datetime import datetime
import numpy as np

from hololoom.dark_trace.causal.ablation import (
    FeatureAblator,
    AblationResult,
    AblationConfig,
)
from hololoom.dark_trace.causal.injection import (
    FeatureInjector,
    InjectionResult,
    InjectionConfig,
)
from hololoom.dark_trace.causal.patching import (
    ActivationPatcher,
    PatchResult,
    CausalTrace,
    PatchConfig,
)


class HypothesisType(Enum):
    """Type of causal hypothesis."""

    BEHAVIORAL = "behavioral"  # Feature affects specific behavior
    REPRESENTATIONAL = "representational"  # Feature represents concept
    MECHANISTIC = "mechanistic"  # Feature is part of circuit
    STEERING = "steering"  # Feature can steer output


class EvidenceType(Enum):
    """Type of causal evidence."""

    ABLATION = "ablation"  # Removal evidence (necessity)
    INJECTION = "injection"  # Activation evidence (sufficiency)
    PATCHING = "patching"  # Transfer evidence (causal mechanism)
    DOSE_RESPONSE = "dose_response"  # Strength relationship
    INTERACTION = "interaction"  # Feature combination effects
    GROUND_TRUTH = "ground_truth"  # Comparison to known labels


class ValidationLevel(Enum):
    """Level of validation rigor."""

    EXPLORATORY = "exploratory"  # Quick checks
    STANDARD = "standard"  # Normal validation
    RIGOROUS = "rigorous"  # Publication-ready
    COMPREHENSIVE = "comprehensive"  # All tests


@dataclass
class CausalHypothesis:
    """A testable causal hypothesis about a feature."""

    # Core hypothesis
    feature: str  # Feature ID (e.g., "sae.42")
    claim: str  # Human-readable claim
    expected_effect: Dict[str, float]  # Expected behavioral effects

    # Hypothesis type and constraints
    hypothesis_type: HypothesisType = HypothesisType.BEHAVIORAL
    direction: str = "positive"  # "positive", "negative", "bidirectional"
    min_effect_size: float = 0.1  # Minimum effect to consider supported

    # Test specifications
    test_inputs: Optional[List[str]] = None  # Inputs to test on
    control_features: Optional[List[str]] = None  # Features that shouldn't change

    # Metadata
    source: str = ""  # Where hypothesis came from
    confidence_prior: float = 0.5  # Prior confidence in hypothesis
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.test_inputs is None:
            self.test_inputs = []
        if self.control_features is None:
            self.control_features = []
        if not self.tags:
            self.tags = []


@dataclass
class EvidenceItem:
    """A single piece of causal evidence."""

    evidence_type: EvidenceType
    result: Any  # AblationResult, InjectionResult, PatchResult, etc.
    supports_hypothesis: bool
    effect_size: float
    p_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str

    def strength(self) -> float:
        """Compute evidence strength (0-1)."""
        # Combine effect size, significance, and confidence
        base = min(1.0, abs(self.effect_size) / 0.5)

        if self.p_value is not None:
            sig_weight = 1.0 - min(1.0, self.p_value / 0.05)
            base = 0.7 * base + 0.3 * sig_weight

        if self.supports_hypothesis:
            return base
        else:
            return -base


@dataclass
class ValidationResult:
    """Result of hypothesis validation."""

    # Core result
    hypothesis: CausalHypothesis
    supported: bool
    confidence: float  # 0-1 confidence in conclusion

    # Evidence
    evidence: List[EvidenceItem]
    evidence_summary: str

    # Statistics
    mean_effect_size: float
    combined_p_value: Optional[float]
    effect_consistency: float  # How consistent effects are across tests

    # Detailed results
    ablation_supports: bool
    injection_supports: bool
    patching_supports: bool

    # If rejected, why
    rejection_reason: Optional[str]

    # Metadata
    validation_level: ValidationLevel
    timestamp: str
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "SUPPORTED" if self.supported else "REJECTED"

        lines = [
            f"Hypothesis Validation: {status}",
            f"=" * 50,
            f"Feature: {self.hypothesis.feature}",
            f"Claim: {self.hypothesis.claim}",
            f"",
            f"Confidence: {self.confidence:.2%}",
            f"Mean effect size: {self.mean_effect_size:.4f}",
            f"Effect consistency: {self.effect_consistency:.2%}",
            f"",
            f"Evidence streams:",
            f"  Ablation: {'Supports' if self.ablation_supports else 'Does not support'}",
            f"  Injection: {'Supports' if self.injection_supports else 'Does not support'}",
            f"  Patching: {'Supports' if self.patching_supports else 'Does not support'}",
            f"",
            f"Evidence items: {len(self.evidence)}",
        ]

        if self.rejection_reason:
            lines.append(f"")
            lines.append(f"Rejection reason: {self.rejection_reason}")

        lines.append(f"")
        lines.append(f"Summary: {self.evidence_summary}")

        return "\n".join(lines)


@dataclass
class GroundTruthComparison:
    """Comparison against ground truth labels."""

    # Ground truth data
    feature_id: str
    ground_truth_label: str  # Human-assigned label
    ground_truth_examples: List[str]  # Examples that should activate

    # Comparison results
    activation_recall: float  # % of ground truth examples that activate
    activation_precision: float  # % of activations that match ground truth
    f1_score: float

    # Behavioral alignment
    behavioral_alignment: float  # How well behavior matches label
    semantic_similarity: float  # Embedding similarity to label

    # Metadata
    n_examples_tested: int
    n_activations_found: int
    mismatches: List[Dict[str, Any]]  # Examples that don't match


@dataclass
class ValidationSuite:
    """Collection of test inputs and ground truths for validation."""

    name: str
    description: str

    # Test data
    test_inputs: List[str]  # Inputs to test
    ground_truths: Dict[str, str]  # feature_id -> ground truth label

    # Expected behaviors (optional)
    expected_behaviors: Dict[str, Dict[str, float]]  # input -> {metric: value}

    # Configuration
    min_examples: int = 10
    require_ground_truth: bool = False

    @classmethod
    def from_examples(
        cls,
        name: str,
        positive_examples: List[str],
        negative_examples: List[str],
        feature_id: str,
        label: str,
    ) -> "ValidationSuite":
        """Create suite from positive/negative examples."""
        test_inputs = positive_examples + negative_examples
        ground_truths = {feature_id: label}

        expected_behaviors = {}
        for ex in positive_examples:
            expected_behaviors[ex] = {"activation": 1.0}
        for ex in negative_examples:
            expected_behaviors[ex] = {"activation": 0.0}

        return cls(
            name=name,
            description=f"Validation suite for {feature_id}: {label}",
            test_inputs=test_inputs,
            ground_truths=ground_truths,
            expected_behaviors=expected_behaviors,
        )


class CausalValidator:
    """
    Orchestrates causal validation of interpretability hypotheses.

    Combines ablation, injection, and patching tests to build
    comprehensive evidence for or against causal claims.

    Usage:
        validator = CausalValidator(ablator, injector, patcher)

        hypothesis = CausalHypothesis(
            feature="sae.42",
            claim="Detects questions",
            expected_effect={"question_prob": 0.3}
        )

        result = validator.validate(hypothesis, test_suite)
        print(result.summary())
    """

    def __init__(
        self,
        ablator: FeatureAblator,
        injector: FeatureInjector,
        patcher: ActivationPatcher,
        behavior_measurer: Optional[Callable] = None,
    ):
        """
        Initialize validator.

        Args:
            ablator: Feature ablator for necessity tests
            injector: Feature injector for sufficiency tests
            patcher: Activation patcher for mechanism tests
            behavior_measurer: Function to measure behavioral effects
        """
        self.ablator = ablator
        self.injector = injector
        self.patcher = patcher
        self.behavior_measurer = behavior_measurer or self._default_behavior_measurer

        # Validation history
        self._history: List[ValidationResult] = []

    def validate(
        self,
        hypothesis: CausalHypothesis,
        test_suite: ValidationSuite,
        level: ValidationLevel = ValidationLevel.STANDARD,
    ) -> ValidationResult:
        """
        Validate a causal hypothesis with multiple evidence streams.

        Args:
            hypothesis: Hypothesis to validate
            test_suite: Test data and ground truths
            level: Validation rigor level

        Returns:
            ValidationResult with comprehensive evidence
        """
        start_time = datetime.now()

        # Determine which tests to run based on level
        run_ablation = level != ValidationLevel.EXPLORATORY
        run_injection = level != ValidationLevel.EXPLORATORY
        run_patching = level in [ValidationLevel.RIGOROUS, ValidationLevel.COMPREHENSIVE]
        run_dose_response = level == ValidationLevel.COMPREHENSIVE
        run_interactions = level == ValidationLevel.COMPREHENSIVE

        # Collect evidence
        evidence = []

        # 1. Ablation tests (causal necessity)
        ablation_supports = False
        if run_ablation:
            ablation_evidence = self._test_ablation(
                hypothesis, test_suite.test_inputs
            )
            evidence.extend(ablation_evidence)
            ablation_supports = any(e.supports_hypothesis for e in ablation_evidence)

        # 2. Injection tests (causal sufficiency)
        injection_supports = False
        if run_injection:
            injection_evidence = self._test_injection(
                hypothesis, test_suite.test_inputs
            )
            evidence.extend(injection_evidence)
            injection_supports = any(e.supports_hypothesis for e in injection_evidence)

        # 3. Patching tests (causal mechanism)
        patching_supports = False
        if run_patching:
            patching_evidence = self._test_patching(
                hypothesis, test_suite.test_inputs
            )
            evidence.extend(patching_evidence)
            patching_supports = any(e.supports_hypothesis for e in patching_evidence)

        # 4. Dose-response analysis
        if run_dose_response:
            dr_evidence = self._test_dose_response(
                hypothesis, test_suite.test_inputs
            )
            evidence.extend(dr_evidence)

        # 5. Interaction analysis
        if run_interactions and hypothesis.control_features:
            interaction_evidence = self._test_interactions(
                hypothesis, test_suite.test_inputs
            )
            evidence.extend(interaction_evidence)

        # Aggregate evidence
        supported, confidence, rejection_reason = self._aggregate_evidence(
            evidence, hypothesis, level
        )

        # Compute statistics
        effect_sizes = [e.effect_size for e in evidence]
        mean_effect = float(np.mean(effect_sizes)) if effect_sizes else 0.0
        effect_consistency = 1.0 - float(np.std(effect_sizes)) if len(effect_sizes) > 1 else 1.0

        # Combine p-values (Fisher's method)
        combined_p = self._combine_p_values(evidence)

        # Generate summary
        evidence_summary = self._generate_evidence_summary(
            evidence, supported, hypothesis
        )

        duration = (datetime.now() - start_time).total_seconds()

        result = ValidationResult(
            hypothesis=hypothesis,
            supported=supported,
            confidence=confidence,
            evidence=evidence,
            evidence_summary=evidence_summary,
            mean_effect_size=mean_effect,
            combined_p_value=combined_p,
            effect_consistency=effect_consistency,
            ablation_supports=ablation_supports,
            injection_supports=injection_supports,
            patching_supports=patching_supports,
            rejection_reason=rejection_reason,
            validation_level=level,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
        )

        self._history.append(result)
        return result

    def validate_batch(
        self,
        hypotheses: List[CausalHypothesis],
        test_suite: ValidationSuite,
        level: ValidationLevel = ValidationLevel.STANDARD,
    ) -> List[ValidationResult]:
        """
        Validate multiple hypotheses.

        Args:
            hypotheses: List of hypotheses to validate
            test_suite: Shared test data
            level: Validation rigor level

        Returns:
            List of ValidationResult for each hypothesis
        """
        results = []
        for hypothesis in hypotheses:
            result = self.validate(hypothesis, test_suite, level)
            results.append(result)
        return results

    def compare_to_ground_truth(
        self,
        feature_id: str,
        ground_truth_label: str,
        positive_examples: List[str],
        negative_examples: List[str],
    ) -> GroundTruthComparison:
        """
        Compare feature behavior to ground truth labels.

        Args:
            feature_id: Feature to evaluate
            ground_truth_label: Human-assigned label
            positive_examples: Examples that should activate feature
            negative_examples: Examples that should not activate

        Returns:
            GroundTruthComparison with precision/recall/F1
        """
        # Test positive examples
        true_positives = 0
        false_negatives = 0
        mismatches = []

        for example in positive_examples:
            activates = self._check_activation(feature_id, example)
            if activates:
                true_positives += 1
            else:
                false_negatives += 1
                mismatches.append({
                    "example": example,
                    "expected": "activate",
                    "actual": "no activation",
                })

        # Test negative examples
        true_negatives = 0
        false_positives = 0

        for example in negative_examples:
            activates = self._check_activation(feature_id, example)
            if not activates:
                true_negatives += 1
            else:
                false_positives += 1
                mismatches.append({
                    "example": example,
                    "expected": "no activation",
                    "actual": "activate",
                })

        # Compute metrics
        total_positives = true_positives + false_negatives
        total_predicted = true_positives + false_positives

        recall = true_positives / total_positives if total_positives > 0 else 0.0
        precision = true_positives / total_predicted if total_predicted > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # Behavioral alignment (simplified)
        behavioral_alignment = (true_positives + true_negatives) / (
            len(positive_examples) + len(negative_examples)
        )

        # Semantic similarity (would use embeddings in full implementation)
        semantic_similarity = 0.5  # Placeholder

        return GroundTruthComparison(
            feature_id=feature_id,
            ground_truth_label=ground_truth_label,
            ground_truth_examples=positive_examples,
            activation_recall=recall,
            activation_precision=precision,
            f1_score=f1,
            behavioral_alignment=behavioral_alignment,
            semantic_similarity=semantic_similarity,
            n_examples_tested=len(positive_examples) + len(negative_examples),
            n_activations_found=true_positives + false_positives,
            mismatches=mismatches,
        )

    def build_causal_graph(
        self,
        features: List[str],
        test_inputs: List[str],
    ) -> Dict[str, Any]:
        """
        Build causal graph showing relationships between features.

        Args:
            features: Features to analyze
            test_inputs: Inputs to test on

        Returns:
            Causal graph structure
        """
        # Use patcher to trace causal relationships
        nodes = []
        edges = []

        for i, feature_a in enumerate(features):
            # Add node
            ablation_result = self.ablator.ablate_feature(
                feature_a, test_inputs
            )
            nodes.append({
                "id": feature_a,
                "causal_effect": ablation_result.effect.output_delta,
                "is_significant": ablation_result.effect.is_significant,
            })

            # Check relationships with other features
            for j, feature_b in enumerate(features):
                if i >= j:
                    continue

                # Test interaction
                interaction = self.injector.detect_interactions(
                    [feature_a, feature_b],
                    strength=0.5,
                    test_inputs=test_inputs,
                )

                if interaction:
                    inter = interaction[0]
                    edges.append({
                        "source": feature_a,
                        "target": feature_b,
                        "interaction": inter.interaction_score,
                        "is_synergistic": inter.is_synergistic,
                        "is_antagonistic": inter.is_antagonistic,
                    })

        return {
            "nodes": nodes,
            "edges": edges,
            "n_features": len(features),
            "n_edges": len(edges),
        }

    def get_validation_history(self) -> List[ValidationResult]:
        """Get history of all validations."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear validation history."""
        self._history.clear()

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _test_ablation(
        self,
        hypothesis: CausalHypothesis,
        test_inputs: List[str],
    ) -> List[EvidenceItem]:
        """Test hypothesis via ablation (causal necessity)."""
        evidence = []

        # Run ablation
        result = self.ablator.ablate_feature(
            hypothesis.feature,
            test_inputs,
        )

        # Check if effect matches expected direction
        effect_direction = result.effect.output_delta
        expected_direction = 1.0 if hypothesis.direction == "positive" else -1.0

        direction_matches = (effect_direction * expected_direction) > 0
        magnitude_sufficient = abs(effect_direction) >= hypothesis.min_effect_size

        supports = (
            direction_matches and
            magnitude_sufficient and
            result.effect.is_significant
        )

        interpretation = self._interpret_ablation(result, hypothesis)

        evidence.append(EvidenceItem(
            evidence_type=EvidenceType.ABLATION,
            result=result,
            supports_hypothesis=supports,
            effect_size=result.effect.effect_size,
            p_value=result.effect.p_value,
            confidence_interval=result.effect.confidence_interval,
            interpretation=interpretation,
        ))

        return evidence

    def _test_injection(
        self,
        hypothesis: CausalHypothesis,
        test_inputs: List[str],
    ) -> List[EvidenceItem]:
        """Test hypothesis via injection (causal sufficiency)."""
        evidence = []

        # Run injection
        result = self.injector.inject_feature(
            hypothesis.feature,
            strength=0.5,  # Medium strength
            test_inputs=test_inputs,
        )

        # Check if behavioral shift matches expected
        behavioral_shift = result.behavioral_shift

        # For now, simplified check - in full version would check against
        # hypothesis.expected_effect
        supports = (
            behavioral_shift is not None and
            result.baseline_output != result.injection_output
        )

        interpretation = self._interpret_injection(result, hypothesis)

        evidence.append(EvidenceItem(
            evidence_type=EvidenceType.INJECTION,
            result=result,
            supports_hypothesis=supports,
            effect_size=result.effect_magnitude,
            p_value=result.p_value,
            confidence_interval=result.confidence_interval,
            interpretation=interpretation,
        ))

        return evidence

    def _test_patching(
        self,
        hypothesis: CausalHypothesis,
        test_inputs: List[str],
    ) -> List[EvidenceItem]:
        """Test hypothesis via activation patching."""
        evidence = []

        if len(test_inputs) < 2:
            return evidence

        # Use first two inputs as source/target
        source_input = test_inputs[0]
        target_input = test_inputs[1]

        result = self.patcher.patch(
            source_input=source_input,
            target_input=target_input,
            features=[hypothesis.feature],
        )

        # Recovery rate indicates causal role
        supports = (
            result.recovery_rate > 0.3 and
            result.is_significant
        )

        interpretation = self._interpret_patching(result, hypothesis)

        evidence.append(EvidenceItem(
            evidence_type=EvidenceType.PATCHING,
            result=result,
            supports_hypothesis=supports,
            effect_size=result.recovery_rate,
            p_value=result.p_value,
            confidence_interval=result.confidence_interval,
            interpretation=interpretation,
        ))

        return evidence

    def _test_dose_response(
        self,
        hypothesis: CausalHypothesis,
        test_inputs: List[str],
    ) -> List[EvidenceItem]:
        """Test dose-response relationship."""
        evidence = []

        result = self.injector.compute_dose_response(
            hypothesis.feature,
            test_inputs=test_inputs,
        )

        # Monotonic dose-response supports causal relationship
        supports = result.slope is not None and result.slope > 0

        interpretation = (
            f"Dose-response {'shows' if supports else 'does not show'} "
            f"monotonic relationship. "
        )
        if result.is_saturating:
            interpretation += "Effect saturates at high doses. "
        if result.is_linear:
            interpretation += "Relationship is approximately linear."

        # Effect size from slope
        effect_size = result.slope if result.slope is not None else 0.0

        evidence.append(EvidenceItem(
            evidence_type=EvidenceType.DOSE_RESPONSE,
            result=result,
            supports_hypothesis=supports,
            effect_size=effect_size,
            p_value=None,
            confidence_interval=None,
            interpretation=interpretation,
        ))

        return evidence

    def _test_interactions(
        self,
        hypothesis: CausalHypothesis,
        test_inputs: List[str],
    ) -> List[EvidenceItem]:
        """Test interactions with control features."""
        evidence = []

        if not hypothesis.control_features:
            return evidence

        # Check if feature interacts with control features
        all_features = [hypothesis.feature] + hypothesis.control_features

        interactions = self.injector.detect_interactions(
            all_features,
            strength=0.5,
            test_inputs=test_inputs,
        )

        for interaction in interactions:
            # Feature should NOT interact with controls (independence)
            is_main_feature = (
                interaction.feature_a == hypothesis.feature or
                interaction.feature_b == hypothesis.feature
            )

            if is_main_feature:
                # No significant interaction supports specificity
                supports = not (interaction.is_synergistic or interaction.is_antagonistic)

                interpretation = (
                    f"Feature {hypothesis.feature} "
                    f"{'does not interact' if supports else 'interacts'} with "
                    f"control features (specificity {'supported' if supports else 'violated'})."
                )

                evidence.append(EvidenceItem(
                    evidence_type=EvidenceType.INTERACTION,
                    result=interaction,
                    supports_hypothesis=supports,
                    effect_size=interaction.interaction_score,
                    p_value=None,
                    confidence_interval=None,
                    interpretation=interpretation,
                ))

        return evidence

    def _aggregate_evidence(
        self,
        evidence: List[EvidenceItem],
        hypothesis: CausalHypothesis,
        level: ValidationLevel,
    ) -> Tuple[bool, float, Optional[str]]:
        """Aggregate evidence into final conclusion."""
        if not evidence:
            return False, 0.0, "No evidence collected"

        # Count supporting vs opposing
        supporting = sum(1 for e in evidence if e.supports_hypothesis)
        opposing = len(evidence) - supporting

        # Weight by evidence strength
        weighted_support = sum(
            e.strength() for e in evidence
            if e.supports_hypothesis
        )
        weighted_oppose = sum(
            abs(e.strength()) for e in evidence
            if not e.supports_hypothesis
        )

        total_weight = weighted_support + weighted_oppose
        if total_weight == 0:
            confidence = 0.5
        else:
            confidence = weighted_support / total_weight

        # Decision thresholds based on validation level
        thresholds = {
            ValidationLevel.EXPLORATORY: (0.5, 0.6),  # majority, 60% conf
            ValidationLevel.STANDARD: (0.6, 0.7),  # 60% support, 70% conf
            ValidationLevel.RIGOROUS: (0.7, 0.8),  # 70% support, 80% conf
            ValidationLevel.COMPREHENSIVE: (0.8, 0.9),  # 80% support, 90% conf
        }

        support_threshold, conf_threshold = thresholds.get(
            level, (0.6, 0.7)
        )

        support_ratio = supporting / len(evidence)
        supported = support_ratio >= support_threshold and confidence >= conf_threshold

        # Determine rejection reason if not supported
        rejection_reason = None
        if not supported:
            if support_ratio < support_threshold:
                rejection_reason = (
                    f"Insufficient evidence: {supporting}/{len(evidence)} "
                    f"({support_ratio:.0%}) support vs {support_threshold:.0%} required"
                )
            elif confidence < conf_threshold:
                rejection_reason = (
                    f"Low confidence: {confidence:.0%} vs {conf_threshold:.0%} required"
                )

        return supported, confidence, rejection_reason

    def _combine_p_values(
        self,
        evidence: List[EvidenceItem],
    ) -> Optional[float]:
        """Combine p-values using Fisher's method."""
        p_values = [e.p_value for e in evidence if e.p_value is not None]

        if not p_values:
            return None

        # Fisher's method: -2 * sum(log(p))
        from scipy import stats

        chi2_stat = -2 * sum(np.log(p) for p in p_values)
        df = 2 * len(p_values)
        combined_p = 1 - stats.chi2.cdf(chi2_stat, df)

        return float(combined_p)

    def _generate_evidence_summary(
        self,
        evidence: List[EvidenceItem],
        supported: bool,
        hypothesis: CausalHypothesis,
    ) -> str:
        """Generate human-readable evidence summary."""
        if not evidence:
            return "No evidence collected."

        # Count by type
        by_type = {}
        for e in evidence:
            t = e.evidence_type.value
            if t not in by_type:
                by_type[t] = {"support": 0, "oppose": 0}
            if e.supports_hypothesis:
                by_type[t]["support"] += 1
            else:
                by_type[t]["oppose"] += 1

        # Build summary
        parts = []
        for etype, counts in by_type.items():
            if counts["support"] > 0 and counts["oppose"] == 0:
                parts.append(f"{etype}: fully supports")
            elif counts["oppose"] > 0 and counts["support"] == 0:
                parts.append(f"{etype}: does not support")
            else:
                parts.append(f"{etype}: mixed ({counts['support']}/{counts['support']+counts['oppose']})")

        conclusion = "SUPPORTED" if supported else "NOT SUPPORTED"

        return f"Hypothesis {conclusion}. Evidence: {', '.join(parts)}."

    def _interpret_ablation(
        self,
        result: AblationResult,
        hypothesis: CausalHypothesis,
    ) -> str:
        """Generate interpretation of ablation result."""
        effect = result.effect

        if effect.is_significant:
            if effect.output_delta > 0:
                direction = "increases"
            else:
                direction = "decreases"

            return (
                f"Ablating {hypothesis.feature} significantly {direction} output "
                f"(effect={effect.output_delta:.3f}, p={effect.p_value:.4f}). "
                f"This {'supports' if hypothesis.direction == 'positive' else 'contradicts'} "
                f"the hypothesis of {hypothesis.direction} causal role."
            )
        else:
            return (
                f"Ablating {hypothesis.feature} has no significant effect "
                f"(effect={effect.output_delta:.3f}, p={effect.p_value:.4f}). "
                f"Feature may not be causally necessary for claimed behavior."
            )

    def _interpret_injection(
        self,
        result: InjectionResult,
        hypothesis: CausalHypothesis,
    ) -> str:
        """Generate interpretation of injection result."""
        if result.behavioral_shift:
            return (
                f"Injecting {hypothesis.feature} produces behavioral shift: "
                f"{result.behavioral_shift}. "
                f"Effect magnitude: {result.effect_magnitude:.3f}. "
                f"This suggests feature is causally sufficient for behavior."
            )
        else:
            return (
                f"Injecting {hypothesis.feature} produces no clear behavioral shift. "
                f"Feature may not be sufficient for claimed behavior."
            )

    def _interpret_patching(
        self,
        result: PatchResult,
        hypothesis: CausalHypothesis,
    ) -> str:
        """Generate interpretation of patching result."""
        if result.recovery_rate > 0.5:
            strength = "strongly"
        elif result.recovery_rate > 0.3:
            strength = "moderately"
        else:
            strength = "weakly"

        return (
            f"Patching {hypothesis.feature} {strength} recovers source behavior "
            f"(recovery={result.recovery_rate:.1%}). "
            f"This {'supports' if result.recovery_rate > 0.3 else 'weakly supports'} "
            f"causal role in behavior transfer."
        )

    def _check_activation(
        self,
        feature_id: str,
        input_text: str,
        threshold: float = 0.1,
    ) -> bool:
        """Check if feature activates on input."""
        # Use injector's probe to get activation
        if hasattr(self.injector, 'probe'):
            probe = self.injector.probe
            if hasattr(probe, 'get_activation'):
                activation = probe.get_activation(feature_id, input_text)
                return activation > threshold

        # Fallback: assume no activation
        return False

    def _default_behavior_measurer(
        self,
        output: Any,
        metrics: List[str],
    ) -> Dict[str, float]:
        """Default behavior measurement function."""
        # Simplified - returns dummy values
        return {metric: 0.5 for metric in metrics}


def create_validator(
    model: Any,
    probe: Any,
    ablation_config: Optional[AblationConfig] = None,
    injection_config: Optional[InjectionConfig] = None,
    patch_config: Optional[PatchConfig] = None,
) -> CausalValidator:
    """
    Factory function to create CausalValidator with all components.

    Args:
        model: Model to validate
        probe: MindProbe for activation access
        ablation_config: Ablation configuration
        injection_config: Injection configuration
        patch_config: Patching configuration

    Returns:
        Configured CausalValidator
    """
    ablator = FeatureAblator(model, probe, ablation_config)
    injector = FeatureInjector(model, probe, injection_config)
    patcher = ActivationPatcher(model, probe, patch_config)

    return CausalValidator(ablator, injector, patcher)
