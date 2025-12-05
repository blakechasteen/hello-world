"""
BossPig Scoring Algorithm

Implements quality score (0-100) and grade calculation (A-F).

Scoring Components:
- Clarity: 30% weight (1.0 - jargon_count / total_words)
- Specificity: 25% weight (1.0 - vague_phrases / total_sentences)
- Actionability: 20% weight (action_items / total_paragraphs)
- Professionalism: 15% weight (1.0 - (hallucinations + formatting_issues) / total_sections)
- Completeness: 10% weight (filled_sections / total_sections)

Created: 2025-11-22
Status: Production Ready (Week 3)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from .core import Finding, FindingCategory, BossPigFindings, QualityMetrics


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of quality score calculation"""
    clarity_score: float
    clarity_weight: float
    clarity_contribution: float

    specificity_score: float
    specificity_weight: float
    specificity_contribution: float

    actionability_score: float
    actionability_weight: float
    actionability_contribution: float

    professionalism_score: float
    professionalism_weight: float
    professionalism_contribution: float

    completeness_score: float
    completeness_weight: float
    completeness_contribution: float

    total_score: int
    grade: str
    grade_label: str

    def to_dict(self) -> Dict:
        """Convert breakdown to dictionary"""
        return {
            "clarity": {
                "score": self.clarity_score,
                "weight": self.clarity_weight,
                "contribution": self.clarity_contribution
            },
            "specificity": {
                "score": self.specificity_score,
                "weight": self.specificity_weight,
                "contribution": self.specificity_contribution
            },
            "actionability": {
                "score": self.actionability_score,
                "weight": self.actionability_weight,
                "contribution": self.actionability_contribution
            },
            "professionalism": {
                "score": self.professionalism_score,
                "weight": self.professionalism_weight,
                "contribution": self.professionalism_contribution
            },
            "completeness": {
                "score": self.completeness_score,
                "weight": self.completeness_weight,
                "contribution": self.completeness_contribution
            },
            "total_score": self.total_score,
            "grade": self.grade,
            "grade_label": self.grade_label
        }


class QualityScorer:
    """
    Advanced quality scoring with detailed breakdowns.

    Provides transparency into how each component contributes
    to the overall quality score.
    """

    # Component weights (must sum to 1.0)
    WEIGHTS = {
        "clarity": 0.30,        # 30% - Most important
        "specificity": 0.25,    # 25%
        "actionability": 0.20,  # 20%
        "professionalism": 0.15, # 15%
        "completeness": 0.10    # 10%
    }

    # Grade thresholds
    GRADE_THRESHOLDS = {
        "A": (90, 100, "Excellent"),
        "B": (80, 89, "Good"),
        "C": (70, 79, "Fair"),
        "D": (60, 69, "Poor"),
        "F": (0, 59, "Critical")
    }

    @classmethod
    def calculate_detailed_score(
        cls,
        findings: BossPigFindings
    ) -> ScoreBreakdown:
        """
        Calculate detailed score breakdown showing contribution of each component.

        Args:
            findings: BossPigFindings with metrics

        Returns:
            ScoreBreakdown with detailed calculations
        """
        metrics = findings.quality_metrics

        # Calculate contributions
        clarity_contrib = metrics.clarity_score * cls.WEIGHTS["clarity"]
        specificity_contrib = metrics.specificity_score * cls.WEIGHTS["specificity"]
        actionability_contrib = metrics.actionability_score * cls.WEIGHTS["actionability"]
        professionalism_contrib = metrics.professionalism_score * cls.WEIGHTS["professionalism"]
        completeness_contrib = metrics.completeness_score * cls.WEIGHTS["completeness"]

        # Total score
        total_score = int((
            clarity_contrib +
            specificity_contrib +
            actionability_contrib +
            professionalism_contrib +
            completeness_contrib
        ) * 100)

        # Grade
        grade, grade_label = cls.calculate_grade(total_score)

        return ScoreBreakdown(
            clarity_score=metrics.clarity_score,
            clarity_weight=cls.WEIGHTS["clarity"],
            clarity_contribution=clarity_contrib,

            specificity_score=metrics.specificity_score,
            specificity_weight=cls.WEIGHTS["specificity"],
            specificity_contribution=specificity_contrib,

            actionability_score=metrics.actionability_score,
            actionability_weight=cls.WEIGHTS["actionability"],
            actionability_contribution=actionability_contrib,

            professionalism_score=metrics.professionalism_score,
            professionalism_weight=cls.WEIGHTS["professionalism"],
            professionalism_contribution=professionalism_contrib,

            completeness_score=metrics.completeness_score,
            completeness_weight=cls.WEIGHTS["completeness"],
            completeness_contribution=completeness_contrib,

            total_score=total_score,
            grade=grade,
            grade_label=grade_label
        )

    @classmethod
    def calculate_grade(cls, score: int) -> Tuple[str, str]:
        """
        Convert numeric score to letter grade.

        Args:
            score: Quality score (0-100)

        Returns:
            Tuple of (grade, label)
        """
        for grade, (min_score, max_score, label) in cls.GRADE_THRESHOLDS.items():
            if min_score <= score <= max_score:
                return grade, label
        return "F", "Critical"

    @classmethod
    def get_improvement_recommendations(
        cls,
        breakdown: ScoreBreakdown
    ) -> List[Dict]:
        """
        Generate prioritized improvement recommendations based on score breakdown.

        Args:
            breakdown: ScoreBreakdown with calculated scores

        Returns:
            List of recommendations sorted by priority
        """
        recommendations = []

        # Check clarity (30% weight)
        if breakdown.clarity_score < 0.7:
            recommendations.append({
                "priority": "high",
                "component": "clarity",
                "score": breakdown.clarity_score,
                "message": f"Clarity score is low ({int(breakdown.clarity_score * 100)}/100). "
                           f"Remove corporate jargon and use plain language.",
                "impact": "30% of total score"
            })

        # Check specificity (25% weight)
        if breakdown.specificity_score < 0.7:
            recommendations.append({
                "priority": "high",
                "component": "specificity",
                "score": breakdown.specificity_score,
                "message": f"Specificity score is low ({int(breakdown.specificity_score * 100)}/100). "
                           f"Replace vague commitments with specific actions and deadlines.",
                "impact": "25% of total score"
            })

        # Check actionability (20% weight)
        if breakdown.actionability_score < 0.7:
            recommendations.append({
                "priority": "medium",
                "component": "actionability",
                "score": breakdown.actionability_score,
                "message": f"Actionability score is low ({int(breakdown.actionability_score * 100)}/100). "
                           f"Convert passive voice to active voice with clear ownership.",
                "impact": "20% of total score"
            })

        # Check professionalism (15% weight)
        if breakdown.professionalism_score < 0.7:
            recommendations.append({
                "priority": "critical",
                "component": "professionalism",
                "score": breakdown.professionalism_score,
                "message": f"Professionalism score is low ({int(breakdown.professionalism_score * 100)}/100). "
                           f"Remove AI hallucination markers and fix formatting issues.",
                "impact": "15% of total score"
            })

        # Check completeness (10% weight)
        if breakdown.completeness_score < 0.7:
            recommendations.append({
                "priority": "high",
                "component": "completeness",
                "score": breakdown.completeness_score,
                "message": f"Completeness score is low ({int(breakdown.completeness_score * 100)}/100). "
                           f"Fill in placeholders and complete all sections.",
                "impact": "10% of total score"
            })

        # Sort by priority: critical > high > medium > low
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))

        return recommendations

    @classmethod
    def generate_score_report(
        cls,
        findings: BossPigFindings,
        include_recommendations: bool = True
    ) -> str:
        """
        Generate human-readable score report.

        Args:
            findings: BossPigFindings with analysis results
            include_recommendations: Include improvement recommendations

        Returns:
            Formatted report string
        """
        breakdown = cls.calculate_detailed_score(findings)

        lines = []
        lines.append("=" * 60)
        lines.append("BossPig Quality Score Report")
        lines.append("=" * 60)
        lines.append("")

        # Overall score
        lines.append(f"Overall Score: {breakdown.total_score}/100")
        lines.append(f"Grade: {breakdown.grade} - {breakdown.grade_label}")
        lines.append("")

        # Component breakdown
        lines.append("Component Breakdown:")
        lines.append(f"  Clarity:        {int(breakdown.clarity_score * 100)}/100  "
                     f"(weight: {int(breakdown.clarity_weight * 100)}%, "
                     f"contribution: {int(breakdown.clarity_contribution * 100)}/100)")
        lines.append(f"  Specificity:    {int(breakdown.specificity_score * 100)}/100  "
                     f"(weight: {int(breakdown.specificity_weight * 100)}%, "
                     f"contribution: {int(breakdown.specificity_contribution * 100)}/100)")
        lines.append(f"  Actionability:  {int(breakdown.actionability_score * 100)}/100  "
                     f"(weight: {int(breakdown.actionability_weight * 100)}%, "
                     f"contribution: {int(breakdown.actionability_contribution * 100)}/100)")
        lines.append(f"  Professionalism:{int(breakdown.professionalism_score * 100)}/100  "
                     f"(weight: {int(breakdown.professionalism_weight * 100)}%, "
                     f"contribution: {int(breakdown.professionalism_contribution * 100)}/100)")
        lines.append(f"  Completeness:   {int(breakdown.completeness_score * 100)}/100  "
                     f"(weight: {int(breakdown.completeness_weight * 100)}%, "
                     f"contribution: {int(breakdown.completeness_contribution * 100)}/100)")
        lines.append("")

        # Findings summary
        lines.append("Findings Summary:")
        lines.append(f"  Critical: {findings.critical_count()}")
        lines.append(f"  Warnings: {findings.warning_count()}")
        lines.append(f"  Info:     {findings.info_count()}")
        lines.append(f"  Total:    {len(findings.findings)}")
        lines.append("")

        # Recommendations
        if include_recommendations:
            recommendations = cls.get_improvement_recommendations(breakdown)
            if recommendations:
                lines.append("Improvement Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    priority_icon = {
                        "critical": "[!]",
                        "high": "[H]",
                        "medium": "[M]",
                        "low": "[L]"
                    }.get(rec["priority"], "[ ]")

                    lines.append(f"{i}. {priority_icon} {rec['message']}")
                    lines.append(f"   Impact: {rec['impact']}")
                lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


if __name__ == "__main__":
    # Test scorer with mock data
    from .core import QualityMetrics, BossPigFindings

    # Create mock findings
    mock_metrics = QualityMetrics(
        clarity_score=0.65,
        specificity_score=0.55,
        actionability_score=0.70,
        professionalism_score=0.40,
        completeness_score=0.85
    )

    mock_findings = BossPigFindings(
        findings=[],
        quality_metrics=mock_metrics,
        document_stats={"word_count": 500, "sentence_count": 25}
    )

    # Calculate detailed score
    breakdown = QualityScorer.calculate_detailed_score(mock_findings)
    print(QualityScorer.generate_score_report(mock_findings))
