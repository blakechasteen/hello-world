"""
Intervention Reasoning for NeuroHood

Enables "what if?" questions about social scenarios using the do() operator.

Examples:
- "What if Alice apologized?" -> do(communication_quality = high)
- "What if Bob turned down the music?" -> do(conflict_intensity = low)
- "What if they had been friends from the start?" -> do(relationship_strength = high)

The intervention system:
1. Parses natural language into causal interventions
2. Applies do() operator to sever parent edges
3. Simulates outcome under intervention
4. Compares to factual outcome (no intervention)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from HoloLoom.causal.intervention import InterventionEngine, Intervention

from .relationship_scm import RelationshipSCM, RelationshipData

logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    """Result of applying an intervention."""
    intervention: Dict[str, float]  # Variables set by do()
    factual_outcome: Dict[str, float]  # Without intervention
    counterfactual_outcome: Dict[str, float]  # With intervention
    causal_effect: Dict[str, float]  # Difference (counterfactual - factual)
    interpretation: str  # Human-readable interpretation


class NeuroHoodInterventionEngine:
    """
    Intervention reasoning for neighborhood relationships.

    Uses causal interventions (do() operator) to answer "what if?" questions
    about social scenarios.

    The do() operator breaks the normal causal flow by *forcing* a variable
    to a specific value, cutting incoming edges.

    Example:
        Factual: Alice is stressed -> communication is negative -> relationship weakens
        Intervention: do(communication_quality = high)
        Counterfactual: Alice is stressed (still!) -> communication is positive (forced!) -> relationship strengthens

    This shows the *causal effect* of communication on the relationship,
    independent of Alice's stress level.
    """

    def __init__(self, scm: RelationshipSCM):
        """
        Initialize intervention engine.

        Args:
            scm: Trained RelationshipSCM
        """
        self.scm = scm
        self.intervention_engine = InterventionEngine(scm.nscm)

    def intervene(
        self,
        intervention: Dict[str, float],
        context: Optional[Dict[str, float]] = None
    ) -> InterventionResult:
        """
        Apply causal intervention and measure effect.

        Args:
            intervention: Variables to set via do() operator
                Example: {"communication_quality": 0.9} = "what if communication was excellent?"
            context: Optional context variables (set normally, not via do())

        Returns:
            InterventionResult with factual, counterfactual, and causal effect

        Example:
            # What if Bob had apologized?
            result = engine.intervene(
                intervention={"communication_quality": 0.9},
                context={"personality_compatibility": 0.7, "conflict_intensity": 0.6}
            )

            print(f"Factual relationship strength: {result.factual_outcome['relationship_strength']:.2f}")
            print(f"With apology: {result.counterfactual_outcome['relationship_strength']:.2f}")
            print(f"Causal effect: {result.causal_effect['relationship_strength']:+.2f}")
        """
        if not self.scm.trained:
            raise ValueError("SCM must be trained before interventions")

        # Set up context (variables not intervened on)
        if context is None:
            context = {}

        # Factual outcome (no intervention)
        factual_outcome = self.scm.predict(context)

        # Counterfactual outcome (with intervention)
        # Merge context and intervention
        intervened_inputs = {**context, **intervention}
        counterfactual_outcome = self.scm.predict(intervened_inputs)

        # Compute causal effect (difference)
        causal_effect = {
            var: counterfactual_outcome[var] - factual_outcome[var]
            for var in self.scm.variable_names
        }

        # Generate interpretation
        interpretation = self._interpret_effect(intervention, causal_effect)

        return InterventionResult(
            intervention=intervention,
            factual_outcome=factual_outcome,
            counterfactual_outcome=counterfactual_outcome,
            causal_effect=causal_effect,
            interpretation=interpretation
        )

    def _interpret_effect(
        self,
        intervention: Dict[str, float],
        causal_effect: Dict[str, float]
    ) -> str:
        """Generate human-readable interpretation of causal effect."""
        lines = ["Intervention Analysis:", ""]

        # Describe intervention
        intervention_desc = ", ".join([
            f"{var} set to {value:.2f}"
            for var, value in intervention.items()
        ])
        lines.append(f"  If we {intervention_desc}:")
        lines.append("")

        # Describe effects (only significant changes >0.1)
        significant_effects = [
            (var, effect)
            for var, effect in causal_effect.items()
            if abs(effect) > 0.1 and var not in intervention
        ]

        if not significant_effects:
            lines.append("  No significant causal effects detected.")
        else:
            lines.append("  Causal effects:")
            for var, effect in sorted(significant_effects, key=lambda x: abs(x[1]), reverse=True):
                direction = "increase" if effect > 0 else "decrease"
                magnitude = abs(effect)

                if magnitude > 0.3:
                    strength = "strongly"
                elif magnitude > 0.2:
                    strength = "moderately"
                else:
                    strength = "slightly"

                lines.append(f"    - {var} would {strength} {direction} ({effect:+.2f})")

        return "\n".join(lines)

    def compare_interventions(
        self,
        interventions: List[Dict[str, float]],
        context: Optional[Dict[str, float]] = None,
        outcome_variable: str = "relationship_strength"
    ) -> List[Tuple[Dict[str, float], float]]:
        """
        Compare multiple interventions to find the best one.

        Args:
            interventions: List of intervention dicts to compare
            context: Shared context across all interventions
            outcome_variable: Variable to optimize (default: relationship_strength)

        Returns:
            List of (intervention, outcome_value) tuples, sorted best to worst

        Example:
            # Which intervention is most effective?
            interventions = [
                {"communication_quality": 0.9},  # Apology
                {"conflict_intensity": 0.0},     # Avoid conflict
                {"communication_quality": 0.9, "conflict_intensity": 0.1},  # Both
            ]

            ranked = engine.compare_interventions(
                interventions,
                context={"personality_compatibility": 0.7}
            )

            print("Best intervention:")
            print(f"  {ranked[0][0]} -> {ranked[0][1]:.2f} relationship strength")
        """
        results = []

        for intervention in interventions:
            result = self.intervene(intervention, context)
            outcome_value = result.counterfactual_outcome[outcome_variable]
            results.append((intervention, outcome_value))

        # Sort by outcome value (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def find_minimal_intervention(
        self,
        target_variable: str,
        target_value: float,
        candidate_variables: List[str],
        context: Optional[Dict[str, float]] = None,
        tolerance: float = 0.1
    ) -> Optional[Dict[str, float]]:
        """
        Find the minimal intervention to achieve a target outcome.

        Searches for the smallest intervention (fewest variables, smallest changes)
        that achieves the desired outcome.

        Args:
            target_variable: Variable to achieve target value for
            target_value: Desired value for target variable
            candidate_variables: Variables we can intervene on
            context: Context variables
            tolerance: Acceptable error in achieving target

        Returns:
            Minimal intervention dict, or None if target unreachable

        Example:
            # What's the minimal intervention to save the relationship?
            minimal = engine.find_minimal_intervention(
                target_variable="relationship_strength",
                target_value=0.8,
                candidate_variables=["communication_quality", "conflict_intensity"],
                context={"personality_compatibility": 0.6}
            )

            if minimal:
                print(f"Minimal intervention: {minimal}")
            else:
                print("Target unreachable with these interventions")
        """
        if context is None:
            context = {}

        # Try single-variable interventions first (most minimal)
        for var in candidate_variables:
            # Try different values
            for value in np.linspace(0.0, 1.0, 11):
                intervention = {var: value}
                result = self.intervene(intervention, context)
                outcome = result.counterfactual_outcome[target_variable]

                if abs(outcome - target_value) <= tolerance:
                    logger.info(f"Found minimal intervention: {intervention} -> {outcome:.2f}")
                    return intervention

        # Try two-variable interventions
        for i, var1 in enumerate(candidate_variables):
            for var2 in candidate_variables[i+1:]:
                for value1 in np.linspace(0.0, 1.0, 6):
                    for value2 in np.linspace(0.0, 1.0, 6):
                        intervention = {var1: value1, var2: value2}
                        result = self.intervene(intervention, context)
                        outcome = result.counterfactual_outcome[target_variable]

                        if abs(outcome - target_value) <= tolerance:
                            logger.info(f"Found 2-var minimal intervention: {intervention} -> {outcome:.2f}")
                            return intervention

        # Target unreachable
        logger.warning(f"Could not find intervention to achieve {target_variable} = {target_value}")
        return None


def parse_natural_language_intervention(query: str) -> Optional[Dict[str, float]]:
    """
    Parse natural language "what if?" questions into interventions.

    This is a simple pattern-based parser. In production, you'd use an LLM
    or more sophisticated NLP.

    Args:
        query: Natural language question

    Returns:
        Intervention dict, or None if query couldn't be parsed

    Examples:
        "What if Alice apologized?" -> {"communication_quality": 0.9}
        "What if there was no conflict?" -> {"conflict_intensity": 0.0}
        "What if they were compatible?" -> {"personality_compatibility": 0.9}
    """
    query_lower = query.lower()

    # Apology / good communication
    if any(word in query_lower for word in ["apolog", "polite", "kind", "respectful"]):
        return {"communication_quality": 0.9}

    # Conflict reduction
    if any(word in query_lower for word in ["no conflict", "peaceful", "calm", "quiet"]):
        return {"conflict_intensity": 0.1}

    # Conflict intensification
    if any(word in query_lower for word in ["big fight", "argument", "yelling", "shouting"]):
        return {"conflict_intensity": 0.9}

    # Compatibility
    if any(word in query_lower for word in ["compatible", "similar", "matched", "suited"]):
        return {"personality_compatibility": 0.9}

    # Incompatibility
    if any(word in query_lower for word in ["incompatible", "different", "mismatched", "opposed"]):
        return {"personality_compatibility": 0.1}

    # Strong relationship
    if any(word in query_lower for word in ["best friends", "close bond", "strong relationship"]):
        return {"relationship_strength": 0.9}

    # Weak relationship
    if any(word in query_lower for word in ["strangers", "distant", "weak bond"]):
        return {"relationship_strength": 0.1}

    # No clear intervention found
    logger.warning(f"Could not parse intervention from query: {query}")
    return None
