"""
Natural Language Explanations - Human-readable explanations

Generates natural language explanations from technical decision data,
making AI decisions understandable to non-experts.

Research:
- Miller (2019): Explanation in Artificial Intelligence: Insights from the Social Sciences
- Ehsan & Riedl (2020): Human-Centered Explainable AI
- Lakkaraju et al. (2019): Faithful and Customizable Explanations
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class ExplanationType(Enum):
    """Types of natural language explanations"""
    WHY = "why"  # Why did you make this decision?
    HOW = "how"  # How did you arrive at this decision?
    WHAT_IF = "what_if"  # What if we changed X?
    WHY_NOT = "why_not"  # Why not choose Y instead?
    EVIDENCE = "evidence"  # What evidence supports this?
    CONFIDENCE = "confidence"  # How confident are you?
    COMPARISON = "comparison"  # How does this compare to alternatives?


@dataclass
class NaturalLanguageExplanation:
    """A natural language explanation"""
    explanation_type: ExplanationType
    text: str  # Natural language explanation
    supporting_data: Optional[Dict[str, Any]] = None  # Technical data
    confidence: float = 1.0

    def __repr__(self) -> str:
        return f"{self.explanation_type.value.upper()}: {self.text}"


class NaturalLanguageExplainer:
    """
    Generate natural language explanations from decision data.

    Converts technical attribution, attention, and provenance data
    into human-readable narratives.
    """

    def __init__(
        self,
        persona: str = "expert",  # or "novice", "technical"
        verbosity: str = "medium"  # or "low", "high"
    ):
        """
        Args:
            persona: Target audience (novice, expert, technical)
            verbosity: Detail level (low, medium, high)
        """
        self.persona = persona
        self.verbosity = verbosity

    def explain(
        self,
        decision: Any,
        feature_importances: Optional[List[Any]] = None,
        attention_weights: Optional[Dict[str, float]] = None,
        counterfactuals: Optional[List[Any]] = None,
        confidence: float = 1.0,
        explanation_type: ExplanationType = ExplanationType.WHY
    ) -> NaturalLanguageExplanation:
        """
        Generate natural language explanation.

        Args:
            decision: The decision/prediction made
            feature_importances: Feature attribution data
            attention_weights: Attention weights
            counterfactuals: Counterfactual explanations
            confidence: Model confidence in decision
            explanation_type: Type of explanation to generate

        Returns:
            Natural language explanation
        """
        if explanation_type == ExplanationType.WHY:
            text = self._explain_why(decision, feature_importances, confidence)
        elif explanation_type == ExplanationType.HOW:
            text = self._explain_how(decision, feature_importances, attention_weights)
        elif explanation_type == ExplanationType.WHAT_IF:
            text = self._explain_what_if(decision, counterfactuals)
        elif explanation_type == ExplanationType.WHY_NOT:
            text = self._explain_why_not(decision, counterfactuals)
        elif explanation_type == ExplanationType.EVIDENCE:
            text = self._explain_evidence(decision, feature_importances)
        elif explanation_type == ExplanationType.CONFIDENCE:
            text = self._explain_confidence(decision, confidence)
        else:
            text = f"I decided on '{decision}'."

        return NaturalLanguageExplanation(
            explanation_type=explanation_type,
            text=text,
            supporting_data={
                'decision': decision,
                'confidence': confidence
            },
            confidence=confidence
        )

    def _explain_why(
        self,
        decision: Any,
        feature_importances: Optional[List[Any]],
        confidence: float
    ) -> str:
        """Generate 'why' explanation"""
        parts = []

        # Opening
        if self.persona == "novice":
            parts.append(f"I chose '{decision}' because ")
        else:
            parts.append(f"Decision: '{decision}' - ")

        # Main reasons
        if feature_importances and len(feature_importances) > 0:
            # Get top features
            top_features = feature_importances[:3]

            if self.verbosity == "low":
                # Just the top reason
                top = top_features[0]
                feature_name = getattr(top, 'feature_name', 'the primary factor')
                importance = getattr(top, 'importance', 1.0)

                parts.append(f"the most important factor was {feature_name} (impact: {importance:.2f}).")

            elif self.verbosity == "medium":
                # Top 2-3 reasons
                reasons = []
                for feat in top_features:
                    feature_name = getattr(feat, 'feature_name', 'a factor')
                    importance = getattr(feat, 'importance', 1.0)

                    if importance > 0:
                        direction = "strongly influenced" if importance > 0.5 else "influenced"
                    else:
                        direction = "negatively influenced"

                    reasons.append(f"{feature_name} {direction} the decision")

                if len(reasons) > 1:
                    parts.append(", ".join(reasons[:-1]) + f", and {reasons[-1]}.")
                else:
                    parts.append(f"{reasons[0]}.")

            else:  # high verbosity
                # Detailed breakdown
                parts.append("based on the following factors:\n")
                for i, feat in enumerate(top_features, 1):
                    feature_name = getattr(feat, 'feature_name', f'Factor {i}')
                    importance = getattr(feat, 'importance', 1.0)
                    rank = getattr(feat, 'rank', i)

                    parts.append(f"  {i}. {feature_name} (ranked #{rank}, importance: {importance:.3f})")
                    if i < len(top_features):
                        parts.append("\n")
        else:
            parts.append("it seemed like the best choice.")

        # Confidence
        if self.verbosity in ["medium", "high"]:
            if confidence > 0.9:
                parts.append(f"\n\nI am very confident ({confidence:.0%}) in this decision.")
            elif confidence > 0.7:
                parts.append(f"\n\nI am moderately confident ({confidence:.0%}) in this decision.")
            else:
                parts.append(f"\n\nI have low confidence ({confidence:.0%}) in this decision.")

        return "".join(parts)

    def _explain_how(
        self,
        decision: Any,
        feature_importances: Optional[List[Any]],
        attention_weights: Optional[Dict[str, float]]
    ) -> str:
        """Generate 'how' explanation (process)"""
        parts = []

        if self.persona == "technical":
            parts.append(f"Decision process for '{decision}':\n\n")
            parts.append("1. Input Processing: Extracted features and computed embeddings\n")
            parts.append("2. Feature Analysis: ")

            if feature_importances:
                parts.append(f"Identified {len(feature_importances)} relevant features\n")
            else:
                parts.append("Analyzed input features\n")

            if attention_weights:
                parts.append("3. Attention Mechanism: Focused on key elements\n")
                top_attended = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)[:3]
                for element, weight in top_attended:
                    parts.append(f"   - {element}: {weight:.3f}\n")

            parts.append("4. Decision: Combined signals to produce final output")

        else:  # novice or expert
            parts.append(f"Here's how I arrived at '{decision}':\n\n")

            if feature_importances:
                parts.append("First, I looked at the input and identified the most important factors. ")
                top = feature_importances[0]
                feature_name = getattr(top, 'feature_name', 'the key factor')
                parts.append(f"The most significant was {feature_name}.\n\n")

            if attention_weights:
                parts.append("Then, I focused my attention on specific parts of the input. ")
                top_attended = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)[:2]
                elements = [elem for elem, _ in top_attended]
                if len(elements) > 1:
                    parts.append(f"I paid the most attention to {elements[0]} and {elements[1]}.\n\n")
                elif elements:
                    parts.append(f"I paid the most attention to {elements[0]}.\n\n")

            parts.append("Finally, I combined all this information to reach my decision.")

        return "".join(parts)

    def _explain_what_if(
        self,
        decision: Any,
        counterfactuals: Optional[List[Any]]
    ) -> str:
        """Generate 'what if' explanation"""
        if not counterfactuals or len(counterfactuals) == 0:
            return f"I cannot find alternative scenarios that would change the decision from '{decision}'."

        parts = []
        parts.append(f"If you wanted a different outcome instead of '{decision}', ")

        cf = counterfactuals[0]  # Use first counterfactual

        # Extract changes
        if hasattr(cf, 'changes'):
            changes = cf.changes
            cf_prediction = getattr(cf, 'counterfactual_prediction', 'something different')

            num_changes = len(changes)

            if num_changes == 1:
                parts.append("you would need to make 1 change:\n")
            else:
                parts.append(f"you would need to make {num_changes} changes:\n")

            for feature, (old_val, new_val) in changes.items():
                parts.append(f"  • Change {feature} from {old_val} to {new_val}\n")

            parts.append(f"\nThis would result in: '{cf_prediction}'")

        else:
            parts.append("you would need to modify some input features.")

        return "".join(parts)

    def _explain_why_not(
        self,
        decision: Any,
        counterfactuals: Optional[List[Any]]
    ) -> str:
        """Generate 'why not' explanation (why not choose alternative)"""
        if not counterfactuals or len(counterfactuals) == 0:
            return f"I chose '{decision}' because it was the best option available. No viable alternatives were found."

        parts = []
        cf = counterfactuals[0]
        cf_prediction = getattr(cf, 'counterfactual_prediction', 'the alternative')

        parts.append(f"You might wonder why I didn't choose '{cf_prediction}' instead of '{decision}'. ")

        if hasattr(cf, 'distance'):
            distance = cf.distance
            if distance > 0.5:
                parts.append("The main reason is that choosing that alternative would require significant changes to the input. ")
            else:
                parts.append("While that alternative was close, ")

        if hasattr(cf, 'feasibility'):
            feasibility = cf.feasibility
            if feasibility < 0.7:
                parts.append("Additionally, that alternative may not be feasible in this context. ")

        parts.append(f"Based on the current input, '{decision}' was the most appropriate choice.")

        return "".join(parts)

    def _explain_evidence(
        self,
        decision: Any,
        feature_importances: Optional[List[Any]]
    ) -> str:
        """Generate 'evidence' explanation"""
        parts = []
        parts.append(f"Evidence supporting '{decision}':\n\n")

        if feature_importances and len(feature_importances) > 0:
            for i, feat in enumerate(feature_importances[:5], 1):
                feature_name = getattr(feat, 'feature_name', f'Factor {i}')
                importance = getattr(feat, 'importance', 0)
                confidence = getattr(feat, 'confidence', 1.0)

                parts.append(f"{i}. {feature_name}\n")
                parts.append(f"   Impact: {importance:.3f}, Confidence: {confidence:.0%}\n")

                if importance > 0:
                    parts.append("   → Supports the decision\n")
                else:
                    parts.append("   → Works against the decision\n")

                parts.append("\n")
        else:
            parts.append("No explicit evidence data available.")

        return "".join(parts)

    def _explain_confidence(
        self,
        decision: Any,
        confidence: float
    ) -> str:
        """Generate 'confidence' explanation"""
        parts = []
        parts.append(f"Confidence in '{decision}': {confidence:.0%}\n\n")

        if confidence > 0.95:
            parts.append("I am extremely confident in this decision. ")
            parts.append("The evidence strongly supports this choice, and there is little uncertainty.")

        elif confidence > 0.8:
            parts.append("I am very confident in this decision. ")
            parts.append("The evidence supports this choice well, though there is some minor uncertainty.")

        elif confidence > 0.6:
            parts.append("I am moderately confident in this decision. ")
            parts.append("The evidence supports this choice, but there are alternative possibilities.")

        elif confidence > 0.4:
            parts.append("I have low confidence in this decision. ")
            parts.append("The evidence is mixed, and alternative choices may be equally valid.")

        else:
            parts.append("I have very low confidence in this decision. ")
            parts.append("The evidence is unclear, and this choice is highly uncertain. ")
            parts.append("You may want to provide more information or reconsider the inputs.")

        return "".join(parts)


def generate_explanation(
    decision: Any,
    explanation_type: ExplanationType = ExplanationType.WHY,
    feature_importances: Optional[List[Any]] = None,
    attention_weights: Optional[Dict[str, float]] = None,
    counterfactuals: Optional[List[Any]] = None,
    confidence: float = 1.0,
    persona: str = "expert"
) -> str:
    """
    Convenience function to generate natural language explanation.

    Args:
        decision: The decision made
        explanation_type: Type of explanation
        feature_importances: Feature attribution data
        attention_weights: Attention weights
        counterfactuals: Counterfactual explanations
        confidence: Decision confidence
        persona: Target audience (novice, expert, technical)

    Returns:
        Natural language explanation text
    """
    explainer = NaturalLanguageExplainer(persona=persona)
    explanation = explainer.explain(
        decision,
        feature_importances,
        attention_weights,
        counterfactuals,
        confidence,
        explanation_type
    )
    return explanation.text


def explain_decision(
    decision: Any,
    feature_importances: Optional[List[Any]] = None,
    confidence: float = 1.0
) -> str:
    """
    Quick function to explain a decision (why).

    Args:
        decision: The decision made
        feature_importances: Feature attribution data
        confidence: Decision confidence

    Returns:
        Natural language explanation
    """
    return generate_explanation(
        decision,
        ExplanationType.WHY,
        feature_importances=feature_importances,
        confidence=confidence
    )
