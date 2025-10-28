"""
Apiary management - core beekeeping operations and business logic.

The Apiary class manages collections of hives and colonies, performing
operations like adding hives, conducting inspections, generating alerts,
and tracking harvest records.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from apps.keep.models import (
    Hive,
    Colony,
    Inspection,
    HarvestRecord,
    Alert,
)
from apps.keep.types import (
    HealthStatus,
    QueenStatus,
    InspectionType,
    AlertLevel,
)


@dataclass
class Apiary:
    """
    Manages a collection of beehives and colonies.

    The apiary is the central management system for all beekeeping operations,
    maintaining state for hives, colonies, inspections, and harvest records.

    Attributes:
        name: Name of the apiary
        location: Geographic location
        hives: Collection of hives
        colonies: Collection of colonies
        inspections: Historical inspection records
        harvests: Historical harvest records
        alerts: Active and resolved alerts
    """
    name: str = "My Apiary"
    location: str = ""
    hives: Dict[str, Hive] = field(default_factory=dict)
    colonies: Dict[str, Colony] = field(default_factory=dict)
    inspections: List[Inspection] = field(default_factory=list)
    harvests: List[HarvestRecord] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_hive(self, hive: Hive) -> str:
        """
        Add a new hive to the apiary.

        Args:
            hive: Hive to add

        Returns:
            The hive_id of the added hive
        """
        self.hives[hive.hive_id] = hive
        return hive.hive_id

    def add_colony(self, colony: Colony) -> str:
        """
        Add a new colony to the apiary.

        Args:
            colony: Colony to add

        Returns:
            The colony_id of the added colony
        """
        if colony.hive_id not in self.hives:
            raise ValueError(f"Hive {colony.hive_id} does not exist in apiary")

        self.colonies[colony.colony_id] = colony
        return colony.colony_id

    def record_inspection(self, inspection: Inspection) -> str:
        """
        Record a hive inspection.

        Args:
            inspection: Inspection record to add

        Returns:
            The inspection_id
        """
        if inspection.hive_id not in self.hives:
            raise ValueError(f"Hive {inspection.hive_id} does not exist")

        if inspection.colony_id and inspection.colony_id not in self.colonies:
            raise ValueError(f"Colony {inspection.colony_id} does not exist")

        self.inspections.append(inspection)

        # Update colony status based on findings
        if inspection.colony_id:
            self._update_colony_from_inspection(inspection)

        # Generate alerts based on findings
        self._generate_alerts_from_inspection(inspection)

        return inspection.inspection_id

    def record_harvest(self, harvest: HarvestRecord) -> str:
        """
        Record a harvest.

        Args:
            harvest: Harvest record to add

        Returns:
            The harvest_id
        """
        if harvest.hive_id not in self.hives:
            raise ValueError(f"Hive {harvest.hive_id} does not exist")

        self.harvests.append(harvest)
        return harvest.harvest_id

    def get_hive_history(self, hive_id: str, days: int = 90) -> List[Inspection]:
        """
        Get inspection history for a specific hive.

        Args:
            hive_id: The hive to query
            days: Number of days of history to retrieve

        Returns:
            List of inspections for the hive
        """
        cutoff = datetime.now() - timedelta(days=days)
        return [
            insp for insp in self.inspections
            if insp.hive_id == hive_id and insp.timestamp >= cutoff
        ]

    def get_colony(self, hive_id: str) -> Optional[Colony]:
        """
        Get the colony currently living in a hive.

        Args:
            hive_id: The hive to query

        Returns:
            The colony or None if no colony in that hive
        """
        for colony in self.colonies.values():
            if colony.hive_id == hive_id:
                return colony
        return None

    def get_active_alerts(self, hive_id: Optional[str] = None) -> List[Alert]:
        """
        Get active (unresolved) alerts.

        Args:
            hive_id: Optional filter by hive

        Returns:
            List of active alerts
        """
        alerts = [a for a in self.alerts if not a.resolved]

        if hive_id:
            alerts = [a for a in alerts if a.hive_id == hive_id]

        return sorted(alerts, key=lambda a: (a.level.value, a.timestamp), reverse=True)

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.

        Args:
            alert_id: Alert to resolve

        Returns:
            True if alert was found and resolved
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                return True
        return False

    def get_total_harvest(self, product_type: str = "honey", days: Optional[int] = None) -> float:
        """
        Calculate total harvest amount.

        Args:
            product_type: Type of product to sum
            days: Optional number of days to look back

        Returns:
            Total quantity harvested
        """
        harvests = self.harvests

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            harvests = [h for h in harvests if h.timestamp >= cutoff]

        return sum(
            h.quantity for h in harvests
            if h.product_type == product_type
        )

    def get_apiary_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the apiary.

        Returns:
            Dictionary with summary statistics
        """
        active_colonies = len([c for c in self.colonies.values()])
        healthy_colonies = len([
            c for c in self.colonies.values()
            if c.health_status in [HealthStatus.EXCELLENT, HealthStatus.GOOD]
        ])

        active_alerts = len(self.get_active_alerts())
        critical_alerts = len([
            a for a in self.get_active_alerts()
            if a.level == AlertLevel.CRITICAL
        ])

        recent_harvest = self.get_total_harvest(days=365)

        return {
            "name": self.name,
            "location": self.location,
            "total_hives": len(self.hives),
            "active_colonies": active_colonies,
            "healthy_colonies": healthy_colonies,
            "active_alerts": active_alerts,
            "critical_alerts": critical_alerts,
            "inspections_recorded": len(self.inspections),
            "yearly_harvest_lbs": recent_harvest,
        }

    def _update_colony_from_inspection(self, inspection: Inspection) -> None:
        """Update colony status based on inspection findings."""
        colony = self.colonies.get(inspection.colony_id)
        if not colony:
            return

        findings = inspection.findings

        # Update queen status
        if findings.get("queen_seen"):
            if findings.get("eggs_seen"):
                colony.queen_status = QueenStatus.PRESENT_LAYING
            else:
                colony.queen_status = QueenStatus.PRESENT_NOT_LAYING
        elif findings.get("eggs_seen"):
            colony.queen_status = QueenStatus.PRESENT_LAYING
        elif findings.get("swarm_cells", 0) > 0 or findings.get("supersedure_cells", 0) > 0:
            colony.queen_status = QueenStatus.CELLS_PRESENT

        # Update population estimate
        if "population_estimate" in findings:
            colony.population_estimate = findings["population_estimate"]

        # Update health status based on various indicators
        health_score = 0
        if findings.get("eggs_seen"):
            health_score += 2
        if findings.get("larvae_seen"):
            health_score += 2
        if findings.get("capped_brood_seen"):
            health_score += 2
        if findings.get("mites_observed"):
            health_score -= 3
        if findings.get("beetles_observed"):
            health_score -= 2
        if findings.get("disease_signs"):
            health_score -= 4

        if health_score >= 5:
            colony.health_status = HealthStatus.EXCELLENT
        elif health_score >= 3:
            colony.health_status = HealthStatus.GOOD
        elif health_score >= 0:
            colony.health_status = HealthStatus.FAIR
        elif health_score >= -2:
            colony.health_status = HealthStatus.POOR
        else:
            colony.health_status = HealthStatus.CRITICAL

    def _generate_alerts_from_inspection(self, inspection: Inspection) -> None:
        """Generate alerts based on inspection findings."""
        findings = inspection.findings

        # Queen issues
        if not findings.get("queen_seen") and not findings.get("eggs_seen"):
            self.alerts.append(Alert(
                hive_id=inspection.hive_id,
                colony_id=inspection.colony_id,
                level=AlertLevel.URGENT,
                title="No queen or eggs detected",
                message="Colony may be queenless. Verify status and consider requeening.",
            ))

        # Pest issues
        if findings.get("mites_observed"):
            self.alerts.append(Alert(
                hive_id=inspection.hive_id,
                colony_id=inspection.colony_id,
                level=AlertLevel.WARNING,
                title="Varroa mites detected",
                message="Consider mite treatment. Monitor mite levels closely.",
            ))

        if findings.get("beetles_observed"):
            self.alerts.append(Alert(
                hive_id=inspection.hive_id,
                colony_id=inspection.colony_id,
                level=AlertLevel.WARNING,
                title="Small hive beetles detected",
                message="Monitor beetle population. Consider traps or treatment if heavy infestation.",
            ))

        # Disease signs
        if findings.get("disease_signs"):
            diseases = ", ".join(findings["disease_signs"])
            self.alerts.append(Alert(
                hive_id=inspection.hive_id,
                colony_id=inspection.colony_id,
                level=AlertLevel.CRITICAL,
                title=f"Disease signs detected: {diseases}",
                message="Consult with local bee inspector or experienced beekeeper immediately.",
            ))

        # Swarm preparation
        if findings.get("swarm_cells", 0) > 3:
            self.alerts.append(Alert(
                hive_id=inspection.hive_id,
                colony_id=inspection.colony_id,
                level=AlertLevel.URGENT,
                title="Multiple swarm cells present",
                message="Colony is preparing to swarm. Consider split or swarm prevention measures.",
            ))

    def __repr__(self) -> str:
        return f"Apiary(name={self.name}, hives={len(self.hives)}, colonies={len(self.colonies)})"