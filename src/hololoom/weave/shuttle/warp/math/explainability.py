#!/usr/bin/env python3
"""
Explainability Module - "Why did you choose this operation?"
=============================================================

Provides human-readable explanations for operation selection decisions.

Key Features:
- Why explanations: "Why was operation X chosen?"
- Why-not explanations: "Why wasn't operation Y chosen?"
- Counterfactual explanations: "What would need to change for Y to be chosen?"
- Feature importance: Which context features influenced the decision?

Research-backed approaches:
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations) concepts
- Counterfactual reasoning

Expected benefit: Much better interpretability and user trust.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Explanation Types
# ============================================================================

@dataclass
class WhyExplanation:
    """
    Explanation for why an operation was chosen.

    Attributes:
        operation: Operation name
        reason: Primary reason (short)
        full_explanation: Detailed explanation
        supporting_evidence: Evidence supporting this choice
        confidence: Confidence in explanation (0-1)
    """
    operation: str
    reason: str
    full_explanation: str
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class WhyNotExplanation:
    """
    Explanation for why an operation was NOT chosen.

    Attributes:
        operation: Operation that wasn't chosen
        reason: Primary reason (short)
        full_explanation: Detailed explanation
        blockers: Factors that prevented selection
    """
    operation: str
    reason: str
    full_explanation: str
    blockers: List[str] = field(default_factory=list)


@dataclass
class CounterfactualExplanation:
    """
    Counterfactual: "What would need to change for this operation to be chosen?"

    Attributes:
        operation: Operation in question
        current_score: Current selection score
        required_score: Score needed to be selected
        changes_needed: List of required changes
        minimal_change: Most minimal change that would flip decision
    """
    operation: str
    current_score: float
    required_score: float
    changes_needed: List[str] = field(default_factory=list)
    minimal_change: Optional[str] = None


# ============================================================================
# Feature Importance
# ============================================================================

class FeatureImportanceAnalyzer:
    """
    Analyzes which context features influenced operation selection.

    Uses perturbation-based approach similar to LIME.
    """

    def analyze(
        self,
        base_context: np.ndarray,
        selected_operation: str,
        selector_function: callable,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Analyze feature importance for operation selection.

        Args:
            base_context: 470-dim context vector
            selected_operation: Operation that was selected
            selector_function: Function that selects operation given context
            top_k: Number of top features to return

        Returns:
            Dict mapping feature indices to importance scores
        """
        importances = {}

        # Perturb each feature and measure impact on selection
        for feature_idx in range(min(len(base_context), 50)):  # Limit to first 50 features
            # Create perturbed context (zero out this feature)
            perturbed_context = base_context.copy()
            perturbed_context[feature_idx] = 0.0

            try:
                # Re-run selection
                new_operation, _ = selector_function(perturbed_context)

                # If operation changed, this feature was important
                if new_operation != selected_operation:
                    importances[feature_idx] = 1.0
                else:
                    importances[feature_idx] = abs(base_context[feature_idx])
            except:
                pass

        # Sort by importance
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k])

        return sorted_importances


# ============================================================================
# Explanation Generator
# ============================================================================

class ExplanationGenerator:
    """
    Generates human-readable explanations for operation selection.

    Provides multiple explanation types: why, why-not, counterfactual.
    """

    def __init__(self):
        self.feature_analyzer = FeatureImportanceAnalyzer()

        # Feature name mapping (470-dim context)
        self.feature_names = self._create_feature_names()

        logger.info("ExplanationGenerator initialized")

    def explain_why(
        self,
        operation: str,
        context_features: Dict[str, Any],
        selection_metadata: Dict[str, Any]
    ) -> WhyExplanation:
        """
        Generate "why was this operation chosen?" explanation.

        Args:
            operation: Selected operation
            context_features: Context features that influenced selection
            selection_metadata: Metadata from selection process (scores, etc.)

        Returns:
            WhyExplanation
        """
        # Extract key factors
        scores = selection_metadata.get("scores", {})
        score = scores.get(operation, 0.0)

        uncertainties = selection_metadata.get("uncertainties", {})
        uncertainty = uncertainties.get(operation, 0.0)

        exploration_bonus = selection_metadata.get("exploration_bonus", {}).get(operation, 0.0)

        # Generate reason
        if exploration_bonus > score * 0.3:
            reason = "High exploration value"
            full_explanation = (
                f"{operation} was chosen primarily for exploration. "
                f"It has high uncertainty ({uncertainty:.2f}) which makes it worth trying "
                f"to learn more about its performance."
            )
        elif score > 100:
            reason = "High predicted reward"
            full_explanation = (
                f"{operation} was chosen because it has the highest predicted reward "
                f"({score:.2f}) based on past performance in similar contexts."
            )
        else:
            reason = "Best available option"
            full_explanation = (
                f"{operation} was chosen as it scored highest ({score:.2f}) among available operations."
            )

        # Add context-specific factors
        supporting_evidence = {
            "selection_score": score,
            "uncertainty": uncertainty,
            "exploration_bonus": exploration_bonus,
            "context_factors": self._extract_context_factors(context_features),
        }

        return WhyExplanation(
            operation=operation,
            reason=reason,
            full_explanation=full_explanation,
            supporting_evidence=supporting_evidence,
            confidence=0.9
        )

    def explain_why_not(
        self,
        operation: str,
        selected_operation: str,
        selection_metadata: Dict[str, Any]
    ) -> WhyNotExplanation:
        """
        Generate "why wasn't this operation chosen?" explanation.

        Args:
            operation: Operation that wasn't chosen
            selected_operation: Operation that was chosen
            selection_metadata: Metadata from selection process

        Returns:
            WhyNotExplanation
        """
        scores = selection_metadata.get("scores", {})
        op_score = scores.get(operation, 0.0)
        selected_score = scores.get(selected_operation, 0.0)

        blockers = []

        # Analyze why not chosen
        if operation not in scores:
            reason = "Not applicable"
            full_explanation = (
                f"{operation} was not chosen because it's not applicable to this query type."
            )
            blockers.append("Not in candidate set")
        elif op_score < selected_score * 0.5:
            reason = "Much lower predicted reward"
            full_explanation = (
                f"{operation} was not chosen because its predicted reward ({op_score:.2f}) "
                f"is significantly lower than {selected_operation} ({selected_score:.2f})."
            )
            blockers.append(f"Score too low: {op_score:.2f} vs {selected_score:.2f}")
        else:
            reason = "Slightly lower reward"
            full_explanation = (
                f"{operation} was not chosen because {selected_operation} scored slightly higher "
                f"({selected_score:.2f} vs {op_score:.2f})."
            )
            blockers.append(f"Marginally lower score: {op_score:.2f}")

        return WhyNotExplanation(
            operation=operation,
            reason=reason,
            full_explanation=full_explanation,
            blockers=blockers
        )

    def explain_counterfactual(
        self,
        operation: str,
        selected_operation: str,
        context_features: Dict[str, Any],
        selection_metadata: Dict[str, Any]
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual explanation.

        "What would need to change for this operation to be chosen?"

        Args:
            operation: Operation in question
            selected_operation: Operation that was actually chosen
            context_features: Current context
            selection_metadata: Selection metadata

        Returns:
            CounterfactualExplanation
        """
        scores = selection_metadata.get("scores", {})
        current_score = scores.get(operation, 0.0)
        required_score = scores.get(selected_operation, 0.0) + 0.01  # Need to beat selected

        changes_needed = []

        # Analyze what changes would help
        score_gap = required_score - current_score

        if score_gap < 10:
            changes_needed.append("Small improvement in performance (~5% better success rate)")
            minimal_change = "One successful execution would likely make this operation competitive"
        elif score_gap < 50:
            changes_needed.append("Moderate improvement in performance (~20% better success rate)")
            changes_needed.append("Or reduction in uncertainty through more observations")
            minimal_change = "3-5 successful executions would make this operation competitive"
        else:
            changes_needed.append("Significant improvement needed in predicted performance")
            changes_needed.append("Or substantial reduction in cost")
            changes_needed.append("Or different query context (e.g., different intent)")
            minimal_change = "10+ successful executions or change in query type needed"

        # Check context-specific changes
        if context_features.get("budget_remaining", 50) < 20:
            if current_score > required_score - 5:
                changes_needed.insert(0, "Higher budget (operation may be too expensive)")
                minimal_change = "Budget > 30 would make this operation viable"

        return CounterfactualExplanation(
            operation=operation,
            current_score=current_score,
            required_score=required_score,
            changes_needed=changes_needed,
            minimal_change=minimal_change
        )

    # ========================================================================
    # Internal Helpers
    # ========================================================================

    def _create_feature_names(self) -> Dict[int, str]:
        """Create human-readable names for context features."""
        names = {}

        # Query features (0-99)
        for i in range(64):
            names[i] = f"query_embedding_{i}"
        names[64] = "query_length"
        names[65] = "query_complexity"
        names[66] = "entity_count"
        names[67] = "has_numerical"

        # Intent features (100-149)
        names[100] = "intent_similarity"
        names[101] = "intent_optimization"
        names[102] = "intent_analysis"
        names[103] = "intent_verification"
        names[104] = "intent_confidence"

        # Cost features
        names[420] = "budget_remaining"
        names[421] = "budget_usage_rate"
        names[422] = "cost_constraint"

        return names

    def _extract_context_factors(self, context_features: Dict[str, Any]) -> List[str]:
        """Extract human-readable context factors."""
        factors = []

        # Intent
        if context_features.get("intent"):
            factors.append(f"Query intent: {context_features['intent']}")

        # Budget
        budget = context_features.get("budget_remaining", 50)
        if budget < 20:
            factors.append(f"Low budget remaining: {budget}")
        elif budget > 40:
            factors.append(f"High budget available: {budget}")

        # Query characteristics
        if context_features.get("query_complexity", 0) > 0.7:
            factors.append("Complex query")
        elif context_features.get("query_complexity", 0) < 0.3:
            factors.append("Simple query")

        return factors


# ============================================================================
# Complete Explainability Interface
# ============================================================================

class ExplainableSelector:
    """
    Wrapper that adds explainability to any operation selector.

    Usage:
        selector = ContextualOperationSelector(...)
        explainable = ExplainableSelector(selector)

        # Select operation
        operation, metadata = explainable.select_with_explanation(query, intent)

        # Get explanations
        why = explainable.explain_why(operation)
        why_not = explainable.explain_why_not("other_operation")
        counterfactual = explainable.explain_counterfactual("other_operation")
    """

    def __init__(self, base_selector: Any):
        """
        Args:
            base_selector: Base operation selector to wrap
        """
        self.base_selector = base_selector
        self.explainer = ExplanationGenerator()

        # Store last selection for explanation
        self.last_selection = None
        self.last_context_features = None
        self.last_metadata = None

        logger.info("ExplainableSelector initialized")

    def select_with_explanation(
        self,
        query_text: str,
        intent: str = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select operation and store info for explanation.

        Args:
            query_text: Query string
            intent: Intent classification
            **kwargs: Additional context

        Returns:
            (operation, metadata): Selected operation and metadata
        """
        # Select using base selector
        operation, metadata = self.base_selector.select(
            query_text=query_text,
            intent=intent,
            **kwargs
        )

        # Store for explanation
        self.last_selection = operation
        self.last_context_features = {
            "query": query_text,
            "intent": intent,
            **kwargs
        }
        self.last_metadata = metadata

        return operation, metadata

    def explain_why(self, operation: str = None) -> WhyExplanation:
        """Explain why operation was chosen."""
        if operation is None:
            operation = self.last_selection

        if not self.last_selection:
            raise ValueError("No selection to explain - call select_with_explanation first")

        return self.explainer.explain_why(
            operation=operation,
            context_features=self.last_context_features,
            selection_metadata=self.last_metadata
        )

    def explain_why_not(self, operation: str) -> WhyNotExplanation:
        """Explain why operation was NOT chosen."""
        if not self.last_selection:
            raise ValueError("No selection to explain - call select_with_explanation first")

        return self.explainer.explain_why_not(
            operation=operation,
            selected_operation=self.last_selection,
            selection_metadata=self.last_metadata
        )

    def explain_counterfactual(self, operation: str) -> CounterfactualExplanation:
        """Generate counterfactual explanation."""
        if not self.last_selection:
            raise ValueError("No selection to explain - call select_with_explanation first")

        return self.explainer.explain_counterfactual(
            operation=operation,
            selected_operation=self.last_selection,
            context_features=self.last_context_features,
            selection_metadata=self.last_metadata
        )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("EXPLAINABILITY DEMO")
    print("="*80)
    print()

    # Mock selector for demo
    class MockSelector:
        def select(self, query_text, intent=None, **kwargs):
            # Mock selection
            scores = {
                "inner_product": 150.0,
                "metric_distance": 120.0,
                "kl_divergence": 90.0,
                "gradient": 60.0,
            }

            uncertainties = {
                "inner_product": 0.2,
                "metric_distance": 0.3,
                "kl_divergence": 0.5,
                "gradient": 0.8,
            }

            selected = "inner_product"

            metadata = {
                "scores": scores,
                "uncertainties": uncertainties,
                "exploration_bonus": {op: 10.0 * unc for op, unc in uncertainties.items()},
            }

            return selected, metadata

    # Create explainable selector
    base = MockSelector()
    explainable = ExplainableSelector(base)

    # Select operation
    print("[1] Selecting operation...")
    operation, metadata = explainable.select_with_explanation(
        query_text="Find documents similar to quantum computing",
        intent="similarity",
        budget_remaining=45.0
    )

    print(f"  Selected: {operation}")
    print(f"  Score: {metadata['scores'][operation]:.2f}")
    print()

    # Explain WHY
    print("[2] WHY was inner_product chosen?")
    print("-" * 60)
    why = explainable.explain_why()
    print(f"  Reason: {why.reason}")
    print(f"  Explanation: {why.full_explanation}")
    print(f"  Evidence:")
    for key, val in why.supporting_evidence.items():
        print(f"    {key}: {val}")
    print()

    # Explain WHY NOT
    print("[3] WHY NOT gradient?")
    print("-" * 60)
    why_not = explainable.explain_why_not("gradient")
    print(f"  Reason: {why_not.reason}")
    print(f"  Explanation: {why_not.full_explanation}")
    print(f"  Blockers:")
    for blocker in why_not.blockers:
        print(f"    - {blocker}")
    print()

    # Counterfactual
    print("[4] What would need to change for gradient to be chosen?")
    print("-" * 60)
    counterfactual = explainable.explain_counterfactual("gradient")
    print(f"  Current score: {counterfactual.current_score:.2f}")
    print(f"  Required score: {counterfactual.required_score:.2f}")
    print(f"  Gap: {counterfactual.required_score - counterfactual.current_score:.2f}")
    print(f"\n  Changes needed:")
    for change in counterfactual.changes_needed:
        print(f"    - {change}")
    print(f"\n  Minimal change: {counterfactual.minimal_change}")
    print()

    print("Explainability module ready for integration!")
