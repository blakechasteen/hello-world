"""
BeeKeeper - AI-powered beekeeping assistant with HoloLoom integration.

The BeeKeeper provides intelligent decision support for apiary management,
using HoloLoom's memory and reasoning capabilities to help beekeepers make
informed decisions about hive inspections, treatments, and management.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from apps.keep.apiary import Apiary
from apps.keep.models import Hive, Colony, Inspection, Alert
from apps.keep.types import HealthStatus, QueenStatus, AlertLevel, InspectionType


@dataclass
class KeeperRecommendation:
    """
    Recommendation from the BeeKeeper assistant.

    Attributes:
        title: Brief recommendation summary
        priority: Urgency level
        reasoning: Explanation of why this is recommended
        actions: Specific actions to take
        timeline: When to act
        related_hives: Hives this recommendation applies to
    """
    title: str
    priority: AlertLevel
    reasoning: str
    actions: List[str]
    timeline: str
    related_hives: List[str]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BeeKeeper:
    """
    AI-powered beekeeping assistant.

    Provides intelligent recommendations and decision support based on
    apiary state, inspection history, and beekeeping best practices.

    Attributes:
        apiary: The apiary being managed
        hololoom_enabled: Whether to use HoloLoom for enhanced reasoning
    """

    def __init__(self, apiary: Apiary, hololoom_enabled: bool = False):
        """
        Initialize the BeeKeeper assistant.

        Args:
            apiary: Apiary to manage
            hololoom_enabled: Enable HoloLoom integration (requires setup)
        """
        self.apiary = apiary
        self.hololoom_enabled = hololoom_enabled
        self._hololoom_shuttle = None

    async def initialize_hololoom(self) -> None:
        """Initialize HoloLoom integration for enhanced reasoning."""
        if not self.hololoom_enabled:
            return

        try:
            from HoloLoom.weaving_shuttle import WeavingShuttle
            from HoloLoom.config import Config
            from HoloLoom.Documentation.types import Query

            # Create config and shuttle
            config = Config.fast()
            self._hololoom_shuttle = WeavingShuttle(
                cfg=config,
                shards=self._create_apiary_memory_shards()
            )

            print("HoloLoom integration initialized")

        except ImportError:
            print("Warning: HoloLoom not available, using rule-based reasoning only")
            self.hololoom_enabled = False

    def _create_apiary_memory_shards(self) -> List[Any]:
        """Create memory shards from apiary state for HoloLoom."""
        # TODO: Convert apiary state into MemoryShards
        # This would encode hive status, inspection history, etc.
        # into HoloLoom's memory format for reasoning
        return []

    async def get_recommendations(self, days_ahead: int = 14) -> List[KeeperRecommendation]:
        """
        Get recommendations for apiary management.

        Args:
            days_ahead: Planning horizon in days

        Returns:
            List of prioritized recommendations
        """
        recommendations = []

        # Check for overdue inspections
        recommendations.extend(self._check_inspection_schedule())

        # Check colony health issues
        recommendations.extend(self._check_colony_health())

        # Check seasonal tasks
        recommendations.extend(self._check_seasonal_tasks(days_ahead))

        # Check alerts
        recommendations.extend(self._check_alerts())

        # Sort by priority
        priority_order = {
            AlertLevel.CRITICAL: 0,
            AlertLevel.URGENT: 1,
            AlertLevel.WARNING: 2,
            AlertLevel.INFO: 3,
        }
        recommendations.sort(key=lambda r: priority_order[r.priority])

        return recommendations

    def _check_inspection_schedule(self) -> List[KeeperRecommendation]:
        """Check which hives are due for inspection."""
        recommendations = []
        now = datetime.now()

        for hive_id, hive in self.apiary.hives.items():
            # Get last inspection
            hive_inspections = [
                insp for insp in self.apiary.inspections
                if insp.hive_id == hive_id
            ]

            if not hive_inspections:
                # Never inspected
                recommendations.append(KeeperRecommendation(
                    title=f"Initial inspection needed: {hive.name}",
                    priority=AlertLevel.URGENT,
                    reasoning="Hive has never been inspected",
                    actions=["Conduct thorough initial inspection", "Document baseline colony status"],
                    timeline="ASAP",
                    related_hives=[hive_id],
                ))
                continue

            # Sort by timestamp
            hive_inspections.sort(key=lambda i: i.timestamp, reverse=True)
            last_inspection = hive_inspections[0]

            # Check if overdue (standard: inspect every 7-10 days during active season)
            days_since = (now - last_inspection.timestamp).days

            if days_since > 10:
                recommendations.append(KeeperRecommendation(
                    title=f"Routine inspection due: {hive.name}",
                    priority=AlertLevel.WARNING if days_since < 14 else AlertLevel.URGENT,
                    reasoning=f"Last inspected {days_since} days ago",
                    actions=["Check queen status", "Assess brood pattern", "Check for pests"],
                    timeline="Within 2-3 days",
                    related_hives=[hive_id],
                ))

        return recommendations

    def _check_colony_health(self) -> List[KeeperRecommendation]:
        """Check for colony health concerns."""
        recommendations = []

        for colony_id, colony in self.apiary.colonies.items():
            hive = self.apiary.hives.get(colony.hive_id)
            hive_name = hive.name if hive else colony.hive_id

            # Poor or critical health
            if colony.health_status in [HealthStatus.POOR, HealthStatus.CRITICAL]:
                recommendations.append(KeeperRecommendation(
                    title=f"Health concern: {hive_name}",
                    priority=AlertLevel.CRITICAL if colony.health_status == HealthStatus.CRITICAL else AlertLevel.URGENT,
                    reasoning=f"Colony health status: {colony.health_status.value}",
                    actions=[
                        "Conduct detailed health inspection",
                        "Check for disease signs",
                        "Assess food stores",
                        "Consider consulting experienced beekeeper",
                    ],
                    timeline="Immediate",
                    related_hives=[colony.hive_id],
                ))

            # Queen issues
            if colony.queen_status in [QueenStatus.ABSENT, QueenStatus.PRESENT_NOT_LAYING]:
                recommendations.append(KeeperRecommendation(
                    title=f"Queen issue: {hive_name}",
                    priority=AlertLevel.URGENT,
                    reasoning=f"Queen status: {colony.queen_status.value}",
                    actions=[
                        "Verify queen presence thoroughly",
                        "Check for eggs and young larvae",
                        "Consider requeening if confirmed queenless",
                        "Check for laying workers",
                    ],
                    timeline="Within 1-2 days",
                    related_hives=[colony.hive_id],
                ))

            # Old queen
            if colony.queen_age_months and colony.queen_age_months > 24:
                recommendations.append(KeeperRecommendation(
                    title=f"Aging queen: {hive_name}",
                    priority=AlertLevel.INFO,
                    reasoning=f"Queen is {colony.queen_age_months} months old",
                    actions=[
                        "Monitor brood pattern closely",
                        "Consider requeening next season",
                        "Allow natural supersedure if colony attempts it",
                    ],
                    timeline="Monitor over next few months",
                    related_hives=[colony.hive_id],
                ))

        return recommendations

    def _check_seasonal_tasks(self, days_ahead: int) -> List[KeeperRecommendation]:
        """Check for seasonal management tasks."""
        recommendations = []
        now = datetime.now()
        month = now.month

        # Spring (March-May in Northern Hemisphere)
        if month in [3, 4, 5]:
            recommendations.append(KeeperRecommendation(
                title="Spring buildup monitoring",
                priority=AlertLevel.INFO,
                reasoning="Active brood rearing season",
                actions=[
                    "Check food stores weekly",
                    "Add supers before nectar flow",
                    "Monitor for swarm preparation",
                    "Ensure adequate space for expansion",
                ],
                timeline="Ongoing through spring",
                related_hives=list(self.apiary.hives.keys()),
            ))

        # Summer (June-August)
        elif month in [6, 7, 8]:
            recommendations.append(KeeperRecommendation(
                title="Summer management",
                priority=AlertLevel.INFO,
                reasoning="Peak production season",
                actions=[
                    "Ensure adequate ventilation",
                    "Monitor for small hive beetles",
                    "Prepare for honey harvest",
                    "Watch for nectar dearth stress",
                ],
                timeline="Ongoing through summer",
                related_hives=list(self.apiary.hives.keys()),
            ))

        # Fall (September-November)
        elif month in [9, 10, 11]:
            recommendations.append(KeeperRecommendation(
                title="Winter preparation",
                priority=AlertLevel.WARNING,
                reasoning="Critical preparation period for winter",
                actions=[
                    "Assess food stores (40-60 lbs honey)",
                    "Combine weak colonies if needed",
                    "Reduce entrance for pest control",
                    "Complete mite treatments",
                    "Install mouse guards",
                ],
                timeline="Complete by end of October",
                related_hives=list(self.apiary.hives.keys()),
            ))

        # Winter (December-February)
        else:
            recommendations.append(KeeperRecommendation(
                title="Winter monitoring",
                priority=AlertLevel.INFO,
                reasoning="Minimal intervention period",
                actions=[
                    "Check hive weight periodically",
                    "Ensure good ventilation",
                    "Clear snow from entrances",
                    "Avoid opening hives in cold weather",
                ],
                timeline="Monthly checks",
                related_hives=list(self.apiary.hives.keys()),
            ))

        return recommendations

    def _check_alerts(self) -> List[KeeperRecommendation]:
        """Convert active alerts to recommendations."""
        recommendations = []

        active_alerts = self.apiary.get_active_alerts()

        for alert in active_alerts:
            hive = self.apiary.hives.get(alert.hive_id)
            hive_name = hive.name if hive else alert.hive_id

            recommendations.append(KeeperRecommendation(
                title=f"{alert.title} - {hive_name}",
                priority=alert.level,
                reasoning=alert.message,
                actions=["Address alert conditions", "Document actions taken"],
                timeline="See alert details",
                related_hives=[alert.hive_id],
                metadata={"alert_id": alert.alert_id},
            ))

        return recommendations

    async def ask_question(self, question: str) -> str:
        """
        Ask the BeeKeeper a question about apiary management.

        Args:
            question: Natural language question

        Returns:
            Answer based on apiary state and beekeeping knowledge
        """
        if self.hololoom_enabled and self._hololoom_shuttle:
            # Use HoloLoom for enhanced reasoning
            try:
                from HoloLoom.Documentation.types import Query
                query = Query(text=question)
                spacetime = await self._hololoom_shuttle.weave(query)
                return spacetime.response
            except Exception as e:
                print(f"HoloLoom query failed: {e}, falling back to rule-based")

        # Fallback: Rule-based question answering
        return self._answer_question_rule_based(question)

    def _answer_question_rule_based(self, question: str) -> str:
        """Rule-based question answering fallback."""
        question_lower = question.lower()

        # Apiary status questions
        if "how many" in question_lower and "hive" in question_lower:
            return f"You have {len(self.apiary.hives)} hives in your apiary."

        if "colony" in question_lower and "health" in question_lower:
            healthy = len([
                c for c in self.apiary.colonies.values()
                if c.health_status in [HealthStatus.EXCELLENT, HealthStatus.GOOD]
            ])
            total = len(self.apiary.colonies)
            return f"{healthy} out of {total} colonies are in good or excellent health."

        if "alert" in question_lower:
            active = len(self.apiary.get_active_alerts())
            return f"There are {active} active alerts requiring attention."

        if "harvest" in question_lower:
            total = self.apiary.get_total_harvest(days=365)
            return f"Total honey harvest in the last year: {total:.1f} lbs"

        # Generic response
        return (
            f"Based on your apiary status: {len(self.apiary.hives)} hives, "
            f"{len(self.apiary.colonies)} colonies, "
            f"{len(self.apiary.get_active_alerts())} active alerts. "
            "For more detailed analysis, enable HoloLoom integration."
        )

    def get_hive_report(self, hive_id: str) -> Dict[str, Any]:
        """
        Generate detailed report for a specific hive.

        Args:
            hive_id: Hive to report on

        Returns:
            Detailed hive report
        """
        hive = self.apiary.hives.get(hive_id)
        if not hive:
            return {"error": "Hive not found"}

        colony = self.apiary.get_colony(hive_id)
        recent_inspections = self.apiary.get_hive_history(hive_id, days=30)
        alerts = [a for a in self.apiary.get_active_alerts() if a.hive_id == hive_id]

        return {
            "hive": {
                "id": hive.hive_id,
                "name": hive.name,
                "type": hive.hive_type.value,
                "location": hive.location,
                "installation_date": hive.installation_date.isoformat(),
            },
            "colony": {
                "health_status": colony.health_status.value if colony else "no colony",
                "queen_status": colony.queen_status.value if colony else "N/A",
                "population_estimate": colony.population_estimate if colony else 0,
                "queen_age_months": colony.queen_age_months if colony else None,
            } if colony else None,
            "recent_activity": {
                "inspections_last_30_days": len(recent_inspections),
                "last_inspection": recent_inspections[0].timestamp.isoformat() if recent_inspections else None,
                "active_alerts": len(alerts),
            },
            "alerts": [
                {
                    "level": a.level.value,
                    "title": a.title,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in alerts
            ],
        }

    def __repr__(self) -> str:
        return f"BeeKeeper(apiary={self.apiary.name}, hololoom={self.hololoom_enabled})"