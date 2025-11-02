"""
Composable analytics and query utilities for Keep.

Provides elegant, functional analysis tools for apiary data,
enabling complex insights through simple composition.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

from apps.keep.models import Hive, Colony, Inspection, HarvestRecord, Alert
from apps.keep.apiary import Apiary
from apps.keep.types import HealthStatus, QueenStatus, InspectionType, AlertLevel
from apps.keep.transforms import (
    filter_healthy,
    filter_concerning,
    filter_recent,
    sort_by_health,
    compute_health_distribution,
    compute_trend,
    extract_time_series,
)


@dataclass
class HealthTrend:
    """
    Health trend analysis result.

    Attributes:
        colony_id: Colony analyzed
        trend: 'improving', 'declining', 'stable'
        slope: Numeric slope of trend
        recent_health: Latest health status
        prediction: Predicted health in 30 days
    """
    colony_id: str
    trend: str
    slope: float
    recent_health: HealthStatus
    prediction: Optional[HealthStatus] = None


@dataclass
class ProductivityMetrics:
    """
    Apiary productivity metrics.

    Attributes:
        total_harvest: Total harvest in lbs
        avg_per_hive: Average harvest per hive
        top_producers: List of (hive_id, quantity) for top hives
        seasonal_breakdown: Harvest by season
    """
    total_harvest: float
    avg_per_hive: float
    top_producers: List[tuple[str, float]]
    seasonal_breakdown: Dict[str, float]


@dataclass
class RiskAssessment:
    """
    Risk assessment for apiary health.

    Attributes:
        overall_risk: 'low', 'medium', 'high', 'critical'
        risk_factors: List of identified risk factors
        at_risk_colonies: Colony IDs at risk
        recommended_actions: Prioritized action list
    """
    overall_risk: str
    risk_factors: List[str]
    at_risk_colonies: List[str]
    recommended_actions: List[str]


class ApiaryAnalytics:
    """
    Composable analytics engine for apiary data.

    Provides rich analysis capabilities through functional composition.
    """

    def __init__(self, apiary: Apiary):
        """
        Initialize analytics engine.

        Args:
            apiary: Apiary to analyze
        """
        self.apiary = apiary

    # =========================================================================
    # Health Analytics
    # =========================================================================

    def analyze_health_trends(self, days: int = 90) -> List[HealthTrend]:
        """
        Analyze colony health trends over time.

        Args:
            days: Number of days to analyze

        Returns:
            List of health trend analyses
        """
        trends = []

        for colony_id, colony in self.apiary.colonies.items():
            # Get colony's inspection history
            colony_inspections = [
                insp for insp in self.apiary.inspections
                if insp.colony_id == colony_id
            ]

            if len(colony_inspections) < 2:
                continue

            # Extract health status over time (using proxy from findings)
            def health_score(inspection: Inspection) -> Optional[float]:
                """Convert inspection findings to health score."""
                findings = inspection.findings
                score = 0.0

                if findings.get("queen_seen"): score += 2
                if findings.get("eggs_seen"): score += 2
                if findings.get("larvae_seen"): score += 2
                if findings.get("capped_brood_seen"): score += 2
                if findings.get("mites_observed"): score -= 3
                if findings.get("beetles_observed"): score -= 2
                if findings.get("disease_signs"): score -= 4

                return score

            time_series = extract_time_series(colony_inspections, health_score)

            if len(time_series) < 2:
                continue

            # Compute trend
            trend_data = compute_trend(time_series)

            # Classify trend
            if trend_data["slope"] > 0.1:
                trend_str = "improving"
            elif trend_data["slope"] < -0.1:
                trend_str = "declining"
            else:
                trend_str = "stable"

            trends.append(HealthTrend(
                colony_id=colony_id,
                trend=trend_str,
                slope=trend_data["slope"],
                recent_health=colony.health_status,
            ))

        return trends

    def compute_health_score(self) -> Dict[str, Any]:
        """
        Compute overall apiary health score.

        Returns:
            Dict with score (0-100), grade (A-F), and breakdown
        """
        colonies = list(self.apiary.colonies.values())

        if not colonies:
            return {
                "score": 0,
                "grade": "N/A",
                "breakdown": {},
                "strengths": [],
                "weaknesses": [],
            }

        # Health status weights
        health_weights = {
            HealthStatus.EXCELLENT: 100,
            HealthStatus.GOOD: 80,
            HealthStatus.FAIR: 60,
            HealthStatus.POOR: 40,
            HealthStatus.CRITICAL: 20,
        }

        # Compute weighted average
        total_score = sum(health_weights[c.health_status] for c in colonies)
        avg_score = total_score / len(colonies)

        # Assign grade
        if avg_score >= 90: grade = "A"
        elif avg_score >= 80: grade = "B"
        elif avg_score >= 70: grade = "C"
        elif avg_score >= 60: grade = "D"
        else: grade = "F"

        # Compute breakdown
        distribution = compute_health_distribution(colonies)

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        excellent_count = distribution.get(HealthStatus.EXCELLENT, 0)
        if excellent_count > len(colonies) * 0.3:
            strengths.append(f"{excellent_count} colonies in excellent health")

        critical_count = distribution.get(HealthStatus.CRITICAL, 0)
        poor_count = distribution.get(HealthStatus.POOR, 0)
        if critical_count + poor_count > len(colonies) * 0.2:
            weaknesses.append(f"{critical_count + poor_count} colonies in poor/critical health")

        # Check queen issues
        queenless = len([c for c in colonies if c.queen_status == QueenStatus.ABSENT])
        if queenless > 0:
            weaknesses.append(f"{queenless} colonies potentially queenless")

        return {
            "score": round(avg_score, 1),
            "grade": grade,
            "distribution": {k.value: v for k, v in distribution.items()},
            "strengths": strengths,
            "weaknesses": weaknesses,
        }

    # =========================================================================
    # Productivity Analytics
    # =========================================================================

    def analyze_productivity(self, days: int = 365) -> ProductivityMetrics:
        """
        Analyze apiary productivity.

        Args:
            days: Number of days to analyze

        Returns:
            Productivity metrics
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_harvests = [
            h for h in self.apiary.harvests
            if h.timestamp >= cutoff and h.product_type == "honey"
        ]

        # Total harvest
        total = sum(h.quantity for h in recent_harvests)

        # Per-hive breakdown
        hive_totals = defaultdict(float)
        for harvest in recent_harvests:
            hive_totals[harvest.hive_id] += harvest.quantity

        # Average per hive
        avg = total / len(self.apiary.hives) if self.apiary.hives else 0.0

        # Top producers
        top_producers = sorted(
            hive_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Seasonal breakdown
        seasonal = {
            "spring": 0.0,  # Mar-May
            "summer": 0.0,  # Jun-Aug
            "fall": 0.0,    # Sep-Nov
            "winter": 0.0,  # Dec-Feb
        }

        for harvest in recent_harvests:
            month = harvest.timestamp.month
            if 3 <= month <= 5:
                seasonal["spring"] += harvest.quantity
            elif 6 <= month <= 8:
                seasonal["summer"] += harvest.quantity
            elif 9 <= month <= 11:
                seasonal["fall"] += harvest.quantity
            else:
                seasonal["winter"] += harvest.quantity

        return ProductivityMetrics(
            total_harvest=total,
            avg_per_hive=avg,
            top_producers=top_producers,
            seasonal_breakdown=seasonal,
        )

    # =========================================================================
    # Risk Assessment
    # =========================================================================

    def assess_risk(self) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.

        Returns:
            Risk assessment with factors and recommendations
        """
        risk_factors = []
        at_risk_colonies = []
        recommended_actions = []

        # Check colony health
        colonies = list(self.apiary.colonies.values())
        critical_colonies = [
            c for c in colonies
            if c.health_status in [HealthStatus.CRITICAL, HealthStatus.POOR]
        ]

        if critical_colonies:
            risk_factors.append(f"{len(critical_colonies)} colonies in critical/poor health")
            at_risk_colonies.extend([c.colony_id for c in critical_colonies])
            recommended_actions.append("Immediate health intervention for critical colonies")

        # Check queen issues
        queenless = [c for c in colonies if c.queen_status == QueenStatus.ABSENT]
        if queenless:
            risk_factors.append(f"{len(queenless)} potentially queenless colonies")
            at_risk_colonies.extend([c.colony_id for c in queenless])
            recommended_actions.append("Requeen or provide emergency queen cells")

        # Check for overdue inspections
        recent = datetime.now() - timedelta(days=14)
        for hive_id, hive in self.apiary.hives.items():
            hive_inspections = [
                i for i in self.apiary.inspections
                if i.hive_id == hive_id
            ]
            if not hive_inspections or hive_inspections[-1].timestamp < recent:
                risk_factors.append(f"Hive {hive.name} overdue for inspection")
                recommended_actions.append(f"Inspect {hive.name} within 2-3 days")

        # Check active critical alerts
        critical_alerts = [
            a for a in self.apiary.alerts
            if not a.resolved and a.level == AlertLevel.CRITICAL
        ]
        if critical_alerts:
            risk_factors.append(f"{len(critical_alerts)} unresolved critical alerts")
            recommended_actions.append("Address all critical alerts immediately")

        # Assess overall risk
        if len(risk_factors) >= 5 or critical_colonies:
            overall_risk = "critical"
        elif len(risk_factors) >= 3:
            overall_risk = "high"
        elif len(risk_factors) >= 1:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        return RiskAssessment(
            overall_risk=overall_risk,
            risk_factors=risk_factors,
            at_risk_colonies=list(set(at_risk_colonies)),
            recommended_actions=recommended_actions,
        )

    # =========================================================================
    # Comparative Analysis
    # =========================================================================

    def compare_colonies(self) -> List[Dict[str, Any]]:
        """
        Compare colonies across multiple dimensions.

        Returns:
            List of colony comparisons with rankings
        """
        colonies = list(self.apiary.colonies.values())
        comparisons = []

        for colony in colonies:
            hive = self.apiary.hives.get(colony.hive_id)

            # Get inspection count
            inspection_count = len([
                i for i in self.apiary.inspections
                if i.colony_id == colony.colony_id
            ])

            # Get harvest total
            harvest_total = sum(
                h.quantity for h in self.apiary.harvests
                if h.hive_id == colony.hive_id
            )

            comparisons.append({
                "colony_id": colony.colony_id,
                "hive_name": hive.name if hive else "Unknown",
                "health_status": colony.health_status.value,
                "queen_status": colony.queen_status.value,
                "population": colony.population_estimate,
                "inspections": inspection_count,
                "harvest_lbs": harvest_total,
                "age_days": (datetime.now() - colony.established_date).days,
            })

        return comparisons

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive apiary report.

        Returns:
            Complete analytics report
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "apiary": {
                "name": self.apiary.name,
                "location": self.apiary.location,
                "summary": self.apiary.get_apiary_summary(),
            },
            "health": {
                "score": self.compute_health_score(),
                "trends": [
                    {
                        "colony_id": t.colony_id,
                        "trend": t.trend,
                        "slope": t.slope,
                        "health": t.recent_health.value,
                    }
                    for t in self.analyze_health_trends()
                ],
            },
            "productivity": {
                "metrics": self.analyze_productivity().__dict__,
            },
            "risk": {
                "assessment": self.assess_risk().__dict__,
            },
            "comparisons": self.compare_colonies(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_health_check(apiary: Apiary) -> Dict[str, Any]:
    """
    Quick health check for an apiary.

    Args:
        apiary: Apiary to check

    Returns:
        Dict with critical metrics
    """
    analytics = ApiaryAnalytics(apiary)
    health = analytics.compute_health_score()
    risk = analytics.assess_risk()

    return {
        "health_grade": health["grade"],
        "health_score": health["score"],
        "risk_level": risk.overall_risk,
        "critical_actions": risk.recommended_actions[:3],
    }


def productivity_summary(apiary: Apiary, days: int = 365) -> str:
    """
    Get human-readable productivity summary.

    Args:
        apiary: Apiary to analyze
        days: Days to analyze

    Returns:
        Formatted summary string
    """
    analytics = ApiaryAnalytics(apiary)
    metrics = analytics.analyze_productivity(days)

    return f"""
Productivity Summary ({days} days):
- Total Harvest: {metrics.total_harvest:.1f} lbs
- Average per Hive: {metrics.avg_per_hive:.1f} lbs
- Top Producer: {metrics.top_producers[0][0] if metrics.top_producers else 'N/A'}
- Best Season: {max(metrics.seasonal_breakdown.items(), key=lambda x: x[1])[0] if metrics.seasonal_breakdown else 'N/A'}
    """.strip()
