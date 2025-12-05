"""
Blame Attribution System

High-level interface for assigning responsibility in neighborhood conflicts.

Uses counterfactual reasoning (PN/PS scores) to fairly attribute blame
across multiple parties.

Key principles:
- Fair attribution: Multiple parties can share blame
- Quantitative scoring: PN/PS metrics, not binary blame
- Actionable insights: Suggests what could have prevented the outcome
- Transparent reasoning: Shows why scores were assigned
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .relationship_scm import RelationshipSCM, RelationshipData
from .counterfactuals import NeuroHoodCounterfactualEngine

logger = logging.getLogger(__name__)


@dataclass
class BlameScore:
    """Blame attribution score for a single party."""
    resident_name: str
    action_description: str
    PN: float  # Probability of Necessity (0-1)
    PS: float  # Probability of Sufficiency (0-1)
    PNS: float  # Combined score (PN × PS)
    responsibility_percentage: float  # Relative blame (0-100%)
    interpretation: str


@dataclass
class BlameReport:
    """Complete blame attribution report for a conflict."""
    conflict_description: str
    parties: List[BlameScore]
    total_blame_explained: float  # Sum of all PNS scores
    primary_cause: str  # Party with highest PNS
    recommendations: List[str]  # Suggested interventions


class BlameAttributionSystem:
    """
    Fair blame attribution system for neighborhood conflicts.

    Analyzes conflicts to determine:
    - Who is responsible (and how much)
    - What actions caused the outcome
    - What interventions could have prevented it

    This enables fair resolution and learning from conflicts.
    """

    def __init__(self, scm: RelationshipSCM):
        """
        Initialize blame attribution system.

        Args:
            scm: Trained RelationshipSCM
        """
        self.scm = scm
        self.counterfactual_engine = NeuroHoodCounterfactualEngine(scm)

    def attribute_blame(
        self,
        factual_data: RelationshipData,
        conflict_description: str,
        parties: List[Tuple[str, str, str, float]],  # (name, action_desc, variable, alternative)
        negative_outcome_variable: str = "conflict_intensity"
    ) -> BlameReport:
        """
        Create a comprehensive blame attribution report.

        Args:
            factual_data: What actually happened
            conflict_description: Human description of the conflict
            parties: List of suspected parties as (name, action_desc, variable, alternative_value)
                Example: [
                    ("Alice", "yelling at Bob", "communication_quality", 0.9),
                    ("Bob", "playing loud music", "conflict_intensity", 0.1),
                ]
            negative_outcome_variable: Outcome to assess (default: conflict_intensity)

        Returns:
            BlameReport with fair blame attribution

        Example:
            parties = [
                ("Alice", "yelling at Bob", "communication_quality", 0.9),
                ("Bob", "playing loud music", "conflict_intensity", 0.1),
                ("Charlie", "not mediating", "communication_quality", 0.7),
            ]

            report = system.attribute_blame(
                factual_data=factual,
                conflict_description="Noise complaint escalated to shouting match",
                parties=parties
            )

            print(report.format())
        """
        blame_scores = []

        # Compute blame for each party
        suspects = [(name, var, alt) for name, _, var, alt in parties]
        blame_results = self.counterfactual_engine.compare_blame_attribution(
            factual_data=factual_data,
            suspects=suspects,
            negative_outcome_variable=negative_outcome_variable
        )

        # Create BlameScore objects
        total_pns = sum(scores["PNS"] for _, scores in blame_results)

        for i, (name, blame) in enumerate(blame_results):
            # Find original action description
            action_desc = next(
                (desc for n, desc, _, _ in parties if n == name),
                "unknown action"
            )

            # Compute responsibility percentage
            if total_pns > 0:
                responsibility_pct = (blame["PNS"] / total_pns) * 100
            else:
                responsibility_pct = 0.0

            # Generate interpretation
            interpretation = self._interpret_blame(blame)

            blame_scores.append(BlameScore(
                resident_name=name,
                action_description=action_desc,
                PN=blame["PN"],
                PS=blame["PS"],
                PNS=blame["PNS"],
                responsibility_percentage=responsibility_pct,
                interpretation=interpretation
            ))

        # Identify primary cause
        primary_cause = blame_scores[0].resident_name if blame_scores else "unknown"

        # Generate recommendations
        recommendations = self._generate_recommendations(blame_scores, factual_data)

        return BlameReport(
            conflict_description=conflict_description,
            parties=blame_scores,
            total_blame_explained=total_pns,
            primary_cause=primary_cause,
            recommendations=recommendations
        )

    def _interpret_blame(self, blame: Dict[str, float]) -> str:
        """Generate human-readable interpretation of blame scores."""
        PN = blame["PN"]
        PS = blame["PS"]
        PNS = blame["PNS"]

        if PNS > 0.7:
            return "Highly responsible - action was both necessary and sufficient"
        elif PNS > 0.4:
            return "Moderately responsible - action contributed significantly"
        elif PNS > 0.2:
            return "Partially responsible - action was a factor"
        elif PN > 0.5:
            return "Necessary but not sufficient - action enabled the outcome"
        elif PS > 0.5:
            return "Sufficient but not necessary - action could have caused it alone"
        else:
            return "Minimally responsible - action had little causal effect"

    def _generate_recommendations(
        self,
        blame_scores: List[BlameScore],
        factual_data: RelationshipData
    ) -> List[str]:
        """Generate actionable recommendations based on blame attribution."""
        recommendations = []

        # Prioritize by responsibility
        for score in sorted(blame_scores, key=lambda x: x.PNS, reverse=True):
            if score.PNS > 0.3:  # Significant responsibility
                recommendations.append(
                    f"{score.resident_name} should avoid {score.action_description} "
                    f"({score.responsibility_percentage:.0f}% causal responsibility)"
                )

        # Suggest mediation if multiple parties responsible
        if len([s for s in blame_scores if s.PNS > 0.2]) > 1:
            recommendations.append(
                "Consider mediation - multiple parties contributed to the conflict"
            )

        # Suggest communication improvement if that was a factor
        if factual_data.communication_quality < 0.5:
            recommendations.append(
                "Improve communication quality - current tone is negative"
            )

        # Suggest stress reduction if relevant
        if factual_data.stress_level > 0.6:
            recommendations.append(
                "Address underlying stress - high stress increases conflict risk"
            )

        return recommendations

    def format_report(self, report: BlameReport) -> str:
        """Format blame report as human-readable text."""
        lines = ["=" * 70]
        lines.append("BLAME ATTRIBUTION REPORT")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Conflict: {report.conflict_description}")
        lines.append(f"Primary Cause: {report.primary_cause}")
        lines.append(f"Total Blame Explained: {report.total_blame_explained:.1%}")
        lines.append("")

        lines.append("RESPONSIBILITY BREAKDOWN:")
        lines.append("-" * 70)
        for score in report.parties:
            lines.append(f"  {score.resident_name}:")
            lines.append(f"    Action: {score.action_description}")
            lines.append(f"    Responsibility: {score.responsibility_percentage:.0f}%")
            lines.append(f"    PN (Necessity): {score.PN:.1%}")
            lines.append(f"    PS (Sufficiency): {score.PS:.1%}")
            lines.append(f"    Interpretation: {score.interpretation}")
            lines.append("")

        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 70)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


def quick_blame_attribution(
    scm: RelationshipSCM,
    factual_data: RelationshipData,
    resident_a_name: str,
    resident_b_name: str,
    conflict_description: str = "Neighborhood conflict"
) -> BlameReport:
    """
    Quick blame attribution for a two-party conflict.

    Assumes:
    - Resident A: negative communication (yelling, rudeness)
    - Resident B: conflict trigger (noise, behavior)

    Args:
        scm: Trained SCM
        factual_data: What actually happened
        resident_a_name: First party
        resident_b_name: Second party
        conflict_description: Brief description

    Returns:
        BlameReport

    Example:
        report = quick_blame_attribution(
            scm, factual,
            resident_a_name="Alice",
            resident_b_name="Bob",
            conflict_description="Noise complaint escalation"
        )
        print(report.format())
    """
    system = BlameAttributionSystem(scm)

    parties = [
        (resident_a_name, "negative communication", "communication_quality", 0.9),
        (resident_b_name, "conflict trigger", "conflict_intensity", 0.1),
    ]

    return system.attribute_blame(
        factual_data=factual_data,
        conflict_description=conflict_description,
        parties=parties
    )
