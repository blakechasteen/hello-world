"""
Counterfactual Analysis for NeuroHood

Answers "what would have happened if...?" questions using twin network reasoning.

Difference from interventions:
- **Interventions**: "What if we DO X?" (prospective, planning)
- **Counterfactuals**: "What if we HAD DONE X?" (retrospective, blame/credit)

Counterfactuals use the "twin network" method:
1. Factual world: What actually happened (with all noise factors)
2. Counterfactual world: Same noise factors, but with intervention applied
3. Compare outcomes to measure "what would have been different"

Example:
    Alice yelled at Bob (factual).
    What if Alice had been polite? (counterfactual)

    Twin network ensures:
    - Same background factors (Alice's stress, Bob's personality)
    - Only difference: Alice's communication
    - Fair comparison of outcomes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from HoloLoom.causal.counterfactual import CounterfactualEngine, CounterfactualQuery

from .relationship_scm import RelationshipSCM, RelationshipData
from .interventions import NeuroHoodInterventionEngine

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualResult:
    """Result of counterfactual reasoning."""
    factual_data: RelationshipData  # What actually happened
    counterfactual_intervention: Dict[str, float]  # What we imagine instead
    factual_outcome: Dict[str, float]  # Actual outcome
    counterfactual_outcome: Dict[str, float]  # Imagined outcome
    difference: Dict[str, float]  # What would have been different
    interpretation: str  # Human-readable summary
    probability_of_causation: float  # PC = P(outcome | intervention) - P(outcome | no intervention)


class NeuroHoodCounterfactualEngine:
    """
    Counterfactual reasoning for neighborhood relationships.

    Uses twin network method to answer "what would have happened if...?" questions.

    Key insight: Counterfactuals preserve *noise* (unobserved factors) from the
    factual world, unlike interventions which average over all possible noise.

    This makes counterfactuals appropriate for:
    - Blame attribution: "Would the conflict have happened without Alice's rudeness?"
    - Credit assignment: "Did Bob's apology save the relationship?"
    - Learning from history: "What should we have done differently?"
    """

    def __init__(self, scm: RelationshipSCM):
        """
        Initialize counterfactual engine.

        Args:
            scm: Trained RelationshipSCM
        """
        self.scm = scm
        self.counterfactual_engine = CounterfactualEngine(scm.nscm)
        self.intervention_engine = NeuroHoodInterventionEngine(scm)

    def analyze_counterfactual(
        self,
        factual_data: RelationshipData,
        counterfactual_intervention: Dict[str, float]
    ) -> CounterfactualResult:
        """
        Analyze a counterfactual scenario.

        Args:
            factual_data: What actually happened (full data from simulation)
            counterfactual_intervention: What we imagine instead
                Example: {"communication_quality": 0.9} = "what if Alice had been polite?"

        Returns:
            CounterfactualResult with factual vs counterfactual comparison

        Example:
            # Alice yelled at Bob (factual: communication_quality = 0.2)
            # What if Alice had been polite? (counterfactual: communication_quality = 0.9)

            factual = extract_relationship_data(relationship, residents, personality)

            result = engine.analyze_counterfactual(
                factual_data=factual,
                counterfactual_intervention={"communication_quality": 0.9}
            )

            print(result.interpretation)
            print(f"Probability of causation: {result.probability_of_causation:.1%}")
        """
        # Factual outcome (what actually happened)
        factual_inputs = {
            "personality_compatibility": factual_data.personality_compatibility,
            "communication_quality": factual_data.communication_quality,
            "conflict_intensity": factual_data.conflict_intensity,
        }
        factual_outcome = self.scm.predict(factual_inputs)

        # Counterfactual outcome (what would have happened)
        counterfactual_inputs = {**factual_inputs, **counterfactual_intervention}
        counterfactual_outcome = self.scm.predict(counterfactual_inputs)

        # Compute difference
        difference = {
            var: counterfactual_outcome[var] - factual_outcome[var]
            for var in self.scm.variable_names
        }

        # Compute probability of causation (simplified)
        # PC = P(outcome in factual) - P(outcome in counterfactual)
        # For continuous variables, use relationship_strength as proxy
        factual_strength = factual_outcome["relationship_strength"]
        counterfactual_strength = counterfactual_outcome["relationship_strength"]
        probability_of_causation = abs(counterfactual_strength - factual_strength)

        # Generate interpretation
        interpretation = self._interpret_counterfactual(
            factual_data,
            counterfactual_intervention,
            difference
        )

        return CounterfactualResult(
            factual_data=factual_data,
            counterfactual_intervention=counterfactual_intervention,
            factual_outcome=factual_outcome,
            counterfactual_outcome=counterfactual_outcome,
            difference=difference,
            interpretation=interpretation,
            probability_of_causation=probability_of_causation
        )

    def _interpret_counterfactual(
        self,
        factual_data: RelationshipData,
        counterfactual_intervention: Dict[str, float],
        difference: Dict[str, float]
    ) -> str:
        """Generate human-readable interpretation of counterfactual."""
        lines = ["Counterfactual Analysis:", ""]

        # Describe what actually happened
        lines.append("  What actually happened:")
        lines.append(f"    - Event: {factual_data.event_context}")
        lines.append(f"    - Communication quality: {factual_data.communication_quality:.2f}")
        lines.append(f"    - Conflict intensity: {factual_data.conflict_intensity:.2f}")
        lines.append(f"    - Relationship strength: {factual_data.relationship_strength:.2f}")
        lines.append("")

        # Describe counterfactual scenario
        intervention_desc = ", ".join([
            f"{var} = {value:.2f}"
            for var, value in counterfactual_intervention.items()
        ])
        lines.append(f"  What if {intervention_desc}?")
        lines.append("")

        # Describe what would have been different
        significant_diffs = [
            (var, diff)
            for var, diff in difference.items()
            if abs(diff) > 0.1
        ]

        if not significant_diffs:
            lines.append("  The outcome would have been similar.")
        else:
            lines.append("  The outcome would have been:")
            for var, diff in sorted(significant_diffs, key=lambda x: abs(x[1]), reverse=True):
                direction = "higher" if diff > 0 else "lower"
                lines.append(f"    - {var}: {diff:+.2f} ({direction})")

        return "\n".join(lines)

    def compute_blame_attribution(
        self,
        factual_data: RelationshipData,
        suspect_resident: str,
        suspect_action_variable: str,
        alternative_value: float,
        negative_outcome_variable: str = "conflict_intensity",
        negative_outcome_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute blame attribution using Probability of Necessity and Sufficiency.

        Blame metrics:
        - **PN (Probability of Necessity)**: P(outcome wouldn't have happened without action)
        - **PS (Probability of Sufficiency)**: P(outcome would happen given action)
        - **PNS (Necessity and Sufficiency)**: PN × PS (strongest causal responsibility)

        Args:
            factual_data: What actually happened
            suspect_resident: Resident suspected of causing outcome
            suspect_action_variable: Variable to test (e.g., "communication_quality")
            alternative_value: Value if resident had acted differently
            negative_outcome_variable: Outcome to assess blame for
            negative_outcome_threshold: Threshold for "negative outcome"

        Returns:
            Dict with PN, PS, PNS scores

        Example:
            # How much is Alice to blame for the conflict?
            blame = engine.compute_blame_attribution(
                factual_data=factual,
                suspect_resident="Alice",
                suspect_action_variable="communication_quality",
                alternative_value=0.9,  # If Alice had been polite
                negative_outcome_variable="conflict_intensity"
            )

            print(f"Probability of Necessity: {blame['PN']:.1%}")
            print(f"Probability of Sufficiency: {blame['PS']:.1%}")
            print(f"Total causal responsibility: {blame['PNS']:.1%}")
        """
        # Factual outcome
        factual_inputs = {
            "personality_compatibility": factual_data.personality_compatibility,
            "communication_quality": factual_data.communication_quality,
            "conflict_intensity": factual_data.conflict_intensity,
        }
        factual_outcome = self.scm.predict(factual_inputs)
        factual_negative_outcome = factual_outcome[negative_outcome_variable] > negative_outcome_threshold

        # Counterfactual outcome (if suspect had acted differently)
        counterfactual_intervention = {suspect_action_variable: alternative_value}
        counterfactual_inputs = {**factual_inputs, **counterfactual_intervention}
        counterfactual_outcome = self.scm.predict(counterfactual_inputs)
        counterfactual_negative_outcome = counterfactual_outcome[negative_outcome_variable] > negative_outcome_threshold

        # Compute PN (Probability of Necessity)
        # PN = P(outcome wouldn't happen | action didn't happen, outcome happened)
        # Simplified: Did changing the action prevent the outcome?
        if factual_negative_outcome and not counterfactual_negative_outcome:
            PN = 1.0  # Action was necessary
        elif factual_negative_outcome and counterfactual_negative_outcome:
            PN = 0.0  # Action wasn't necessary (outcome would have happened anyway)
        else:
            PN = 0.0  # Outcome didn't happen, so necessity is irrelevant

        # Compute PS (Probability of Sufficiency)
        # PS = P(outcome would happen | action happened, outcome didn't happen)
        # Reverse counterfactual: if outcome didn't happen, would action cause it?
        # For simplicity, use the magnitude of change
        ps_change = factual_outcome[negative_outcome_variable] - counterfactual_outcome[negative_outcome_variable]
        PS = max(0.0, min(1.0, ps_change))

        # Compute PNS (Probability of Necessity and Sufficiency)
        PNS = PN * PS

        logger.info(f"Blame attribution for {suspect_resident}'s {suspect_action_variable}:")
        logger.info(f"  PN (necessity): {PN:.2f}")
        logger.info(f"  PS (sufficiency): {PS:.2f}")
        logger.info(f"  PNS (total responsibility): {PNS:.2f}")

        return {
            "PN": PN,
            "PS": PS,
            "PNS": PNS,
            "suspect": suspect_resident,
            "action_variable": suspect_action_variable,
            "factual_outcome": factual_outcome[negative_outcome_variable],
            "counterfactual_outcome": counterfactual_outcome[negative_outcome_variable],
        }

    def compare_blame_attribution(
        self,
        factual_data: RelationshipData,
        suspects: List[Tuple[str, str, float]],  # (resident, variable, alternative_value)
        negative_outcome_variable: str = "conflict_intensity"
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Compare blame attribution across multiple suspects.

        Args:
            factual_data: What actually happened
            suspects: List of (resident_name, variable, alternative_value) tuples
            negative_outcome_variable: Outcome to assess blame for

        Returns:
            List of (suspect_name, blame_scores) tuples, sorted by PNS (most to least responsible)

        Example:
            # Who's more to blame: Alice for being rude, or Bob for playing loud music?
            suspects = [
                ("Alice", "communication_quality", 0.9),  # If Alice had been polite
                ("Bob", "conflict_intensity", 0.1),      # If Bob had been quiet
            ]

            blame_ranking = engine.compare_blame_attribution(
                factual_data=factual,
                suspects=suspects
            )

            print("Responsibility ranking:")
            for name, scores in blame_ranking:
                print(f"  {name}: {scores['PNS']:.1%} causal responsibility")
        """
        results = []

        for suspect_name, variable, alternative_value in suspects:
            blame = self.compute_blame_attribution(
                factual_data=factual_data,
                suspect_resident=suspect_name,
                suspect_action_variable=variable,
                alternative_value=alternative_value,
                negative_outcome_variable=negative_outcome_variable
            )
            results.append((suspect_name, blame))

        # Sort by PNS (descending)
        results.sort(key=lambda x: x[1]["PNS"], reverse=True)

        return results


def parse_counterfactual_query(query: str) -> Optional[Dict[str, float]]:
    """
    Parse natural language counterfactual questions.

    Args:
        query: Natural language question

    Returns:
        Counterfactual intervention dict, or None if query couldn't be parsed

    Examples:
        "What if Alice had been polite?" -> {"communication_quality": 0.9}
        "What would have happened if there was no fight?" -> {"conflict_intensity": 0.1}
    """
    query_lower = query.lower()

    # Look for "had been" or "would have" patterns
    is_counterfactual = any(phrase in query_lower for phrase in [
        "had been", "would have", "if they", "if there", "without"
    ])

    if not is_counterfactual:
        return None

    # Reuse intervention parser (same mapping, different tense)
    from .interventions import parse_natural_language_intervention
    return parse_natural_language_intervention(query)
