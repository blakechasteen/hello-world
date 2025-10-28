"""
Builder patterns for elegant object construction in Keep.

Provides fluent, chainable builders for complex domain objects,
enabling clear, readable construction with sensible defaults.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import replace

from apps.keep.models import Hive, Colony, Inspection, HarvestRecord, Alert
from apps.keep.types import (
    HiveType,
    HealthStatus,
    QueenStatus,
    InspectionType,
    InspectionData,
    AlertLevel,
)


class HiveBuilder:
    """
    Fluent builder for Hive objects.

    Example:
        hive = (HiveBuilder()
            .named("Hive 001")
            .langstroth()
            .at("East field")
            .installed_on(datetime(2024, 4, 1))
            .notes("Good morning sun exposure")
            .build())
    """

    def __init__(self):
        self._hive = Hive()

    def named(self, name: str) -> 'HiveBuilder':
        """Set hive name."""
        self._hive.name = name
        return self

    def at(self, location: str) -> 'HiveBuilder':
        """Set location."""
        self._hive.location = location
        return self

    def langstroth(self) -> 'HiveBuilder':
        """Set type to Langstroth."""
        self._hive.hive_type = HiveType.LANGSTROTH
        return self

    def top_bar(self) -> 'HiveBuilder':
        """Set type to Top Bar."""
        self._hive.hive_type = HiveType.TOP_BAR
        return self

    def warre(self) -> 'HiveBuilder':
        """Set type to Warre."""
        self._hive.hive_type = HiveType.WARRE
        return self

    def flow_hive(self) -> 'HiveBuilder':
        """Set type to Flow Hive."""
        self._hive.hive_type = HiveType.FLOW_HIVE
        return self

    def observation(self) -> 'HiveBuilder':
        """Set type to Observation."""
        self._hive.hive_type = HiveType.OBSERVATION
        return self

    def type(self, hive_type: HiveType) -> 'HiveBuilder':
        """Set hive type explicitly."""
        self._hive.hive_type = hive_type
        return self

    def installed_on(self, date: datetime) -> 'HiveBuilder':
        """Set installation date."""
        self._hive.installation_date = date
        return self

    def notes(self, notes: str) -> 'HiveBuilder':
        """Set notes."""
        self._hive.notes = notes
        return self

    def metadata(self, **kwargs) -> 'HiveBuilder':
        """Add metadata."""
        self._hive.metadata.update(kwargs)
        return self

    def build(self) -> Hive:
        """Build the hive."""
        return self._hive


class ColonyBuilder:
    """
    Fluent builder for Colony objects.

    Example:
        colony = (ColonyBuilder()
            .in_hive(hive.hive_id)
            .italian()
            .from_package()
            .healthy()
            .queen_laying()
            .population(50000)
            .established_on(datetime(2024, 4, 5))
            .build())
    """

    def __init__(self):
        self._colony = Colony()

    def in_hive(self, hive_id: str) -> 'ColonyBuilder':
        """Set hive ID."""
        self._colony.hive_id = hive_id
        return self

    def italian(self) -> 'ColonyBuilder':
        """Set breed to Italian."""
        self._colony.breed = "Italian"
        return self

    def carniolan(self) -> 'ColonyBuilder':
        """Set breed to Carniolan."""
        self._colony.breed = "Carniolan"
        return self

    def russian(self) -> 'ColonyBuilder':
        """Set breed to Russian."""
        self._colony.breed = "Russian"
        return self

    def buckfast(self) -> 'ColonyBuilder':
        """Set breed to Buckfast."""
        self._colony.breed = "Buckfast"
        return self

    def breed(self, breed: str) -> 'ColonyBuilder':
        """Set breed explicitly."""
        self._colony.breed = breed
        return self

    def from_package(self) -> 'ColonyBuilder':
        """Set origin to package."""
        self._colony.origin = "package"
        return self

    def from_nuc(self) -> 'ColonyBuilder':
        """Set origin to nuc."""
        self._colony.origin = "nuc"
        return self

    def from_swarm(self) -> 'ColonyBuilder':
        """Set origin to swarm."""
        self._colony.origin = "swarm"
        return self

    def from_split(self, parent_hive: Optional[str] = None) -> 'ColonyBuilder':
        """Set origin to split."""
        if parent_hive:
            self._colony.origin = f"split from {parent_hive}"
        else:
            self._colony.origin = "split"
        return self

    def origin(self, origin: str) -> 'ColonyBuilder':
        """Set origin explicitly."""
        self._colony.origin = origin
        return self

    def healthy(self) -> 'ColonyBuilder':
        """Set health to GOOD."""
        self._colony.health_status = HealthStatus.GOOD
        return self

    def excellent_health(self) -> 'ColonyBuilder':
        """Set health to EXCELLENT."""
        self._colony.health_status = HealthStatus.EXCELLENT
        return self

    def fair_health(self) -> 'ColonyBuilder':
        """Set health to FAIR."""
        self._colony.health_status = HealthStatus.FAIR
        return self

    def poor_health(self) -> 'ColonyBuilder':
        """Set health to POOR."""
        self._colony.health_status = HealthStatus.POOR
        return self

    def health(self, status: HealthStatus) -> 'ColonyBuilder':
        """Set health status explicitly."""
        self._colony.health_status = status
        return self

    def queen_laying(self) -> 'ColonyBuilder':
        """Set queen to present and laying."""
        self._colony.queen_status = QueenStatus.PRESENT_LAYING
        return self

    def queen_not_laying(self) -> 'ColonyBuilder':
        """Set queen to present but not laying."""
        self._colony.queen_status = QueenStatus.PRESENT_NOT_LAYING
        return self

    def queenless(self) -> 'ColonyBuilder':
        """Set as queenless."""
        self._colony.queen_status = QueenStatus.ABSENT
        return self

    def queen_status(self, status: QueenStatus) -> 'ColonyBuilder':
        """Set queen status explicitly."""
        self._colony.queen_status = status
        return self

    def population(self, estimate: int) -> 'ColonyBuilder':
        """Set population estimate."""
        self._colony.population_estimate = estimate
        return self

    def queen_age(self, months: int) -> 'ColonyBuilder':
        """Set queen age in months."""
        self._colony.queen_age_months = months
        return self

    def established_on(self, date: datetime) -> 'ColonyBuilder':
        """Set establishment date."""
        self._colony.established_date = date
        return self

    def notes(self, notes: str) -> 'ColonyBuilder':
        """Set notes."""
        self._colony.notes = notes
        return self

    def metadata(self, **kwargs) -> 'ColonyBuilder':
        """Add metadata."""
        self._colony.metadata.update(kwargs)
        return self

    def build(self) -> Colony:
        """Build the colony."""
        return self._colony


class InspectionBuilder:
    """
    Fluent builder for Inspection objects.

    Example:
        inspection = (InspectionBuilder()
            .for_hive(hive.hive_id)
            .colony(colony.colony_id)
            .routine()
            .on(datetime.now())
            .weather("Sunny, 72°F")
            .queen_seen()
            .eggs_present()
            .brood_frames(8)
            .honey_frames(6)
            .no_pests()
            .action("Added honey super")
            .inspector("John Doe")
            .build())
    """

    def __init__(self):
        self._inspection = Inspection()
        self._findings: InspectionData = {}

    def for_hive(self, hive_id: str) -> 'InspectionBuilder':
        """Set hive ID."""
        self._inspection.hive_id = hive_id
        return self

    def colony(self, colony_id: str) -> 'InspectionBuilder':
        """Set colony ID."""
        self._inspection.colony_id = colony_id
        return self

    def routine(self) -> 'InspectionBuilder':
        """Set type to routine."""
        self._inspection.inspection_type = InspectionType.ROUTINE
        return self

    def health_check(self) -> 'InspectionBuilder':
        """Set type to health check."""
        self._inspection.inspection_type = InspectionType.HEALTH_CHECK
        return self

    def swarm_check(self) -> 'InspectionBuilder':
        """Set type to swarm check."""
        self._inspection.inspection_type = InspectionType.SWARM_CHECK
        return self

    def harvest(self) -> 'InspectionBuilder':
        """Set type to harvest."""
        self._inspection.inspection_type = InspectionType.HARVEST
        return self

    def type(self, inspection_type: InspectionType) -> 'InspectionBuilder':
        """Set inspection type explicitly."""
        self._inspection.inspection_type = inspection_type
        return self

    def on(self, timestamp: datetime) -> 'InspectionBuilder':
        """Set timestamp."""
        self._inspection.timestamp = timestamp
        return self

    def weather(self, description: str) -> 'InspectionBuilder':
        """Set weather description."""
        self._inspection.weather = description
        return self

    def temperature(self, temp: float) -> 'InspectionBuilder':
        """Set temperature."""
        self._inspection.temperature = temp
        return self

    def inspector(self, name: str) -> 'InspectionBuilder':
        """Set inspector name."""
        self._inspection.inspector = name
        return self

    def duration(self, minutes: int) -> 'InspectionBuilder':
        """Set duration in minutes."""
        self._inspection.duration_minutes = minutes
        return self

    # Findings builders

    def queen_seen(self, seen: bool = True) -> 'InspectionBuilder':
        """Set queen seen."""
        self._findings["queen_seen"] = seen
        return self

    def eggs_present(self, present: bool = True) -> 'InspectionBuilder':
        """Set eggs present."""
        self._findings["eggs_seen"] = present
        return self

    def larvae_present(self, present: bool = True) -> 'InspectionBuilder':
        """Set larvae present."""
        self._findings["larvae_seen"] = present
        return self

    def capped_brood(self, present: bool = True) -> 'InspectionBuilder':
        """Set capped brood present."""
        self._findings["capped_brood_seen"] = present
        return self

    def brood_frames(self, count: int) -> 'InspectionBuilder':
        """Set frames with brood."""
        self._findings["frames_with_brood"] = count
        return self

    def honey_frames(self, count: int) -> 'InspectionBuilder':
        """Set frames with honey."""
        self._findings["frames_with_honey"] = count
        return self

    def pollen_frames(self, count: int) -> 'InspectionBuilder':
        """Set frames with pollen."""
        self._findings["frames_with_pollen"] = count
        return self

    def population(self, estimate: int) -> 'InspectionBuilder':
        """Set population estimate."""
        self._findings["population_estimate"] = estimate
        return self

    def swarm_cells(self, count: int) -> 'InspectionBuilder':
        """Set swarm cell count."""
        self._findings["swarm_cells"] = count
        return self

    def mites(self, observed: bool = True) -> 'InspectionBuilder':
        """Set mites observed."""
        self._findings["mites_observed"] = observed
        return self

    def beetles(self, observed: bool = True) -> 'InspectionBuilder':
        """Set beetles observed."""
        self._findings["beetles_observed"] = observed
        return self

    def no_pests(self) -> 'InspectionBuilder':
        """Set no pests observed."""
        self._findings["mites_observed"] = False
        self._findings["beetles_observed"] = False
        self._findings["moths_observed"] = False
        return self

    def finding(self, key: str, value: Any) -> 'InspectionBuilder':
        """Add custom finding."""
        self._findings[key] = value
        return self

    def findings(self, **kwargs) -> 'InspectionBuilder':
        """Add multiple findings."""
        self._findings.update(kwargs)
        return self

    # Actions and recommendations

    def action(self, action: str) -> 'InspectionBuilder':
        """Add action taken."""
        self._inspection.actions_taken.append(action)
        return self

    def recommend(self, recommendation: str) -> 'InspectionBuilder':
        """Add recommendation."""
        self._inspection.recommendations.append(recommendation)
        return self

    def notes(self, notes: str) -> 'InspectionBuilder':
        """Set inspection notes."""
        self._inspection.notes = notes
        return self

    def build(self) -> Inspection:
        """Build the inspection."""
        self._inspection.findings = self._findings
        return self._inspection


class AlertBuilder:
    """
    Fluent builder for Alert objects.

    Example:
        alert = (AlertBuilder()
            .for_hive(hive.hive_id)
            .critical()
            .titled("Queenless colony")
            .message("No queen or eggs detected for 2 weeks")
            .build())
    """

    def __init__(self):
        self._alert = Alert()

    def for_hive(self, hive_id: str) -> 'AlertBuilder':
        """Set hive ID."""
        self._alert.hive_id = hive_id
        return self

    def colony(self, colony_id: str) -> 'AlertBuilder':
        """Set colony ID."""
        self._alert.colony_id = colony_id
        return self

    def info(self) -> 'AlertBuilder':
        """Set level to INFO."""
        self._alert.level = AlertLevel.INFO
        return self

    def warning(self) -> 'AlertBuilder':
        """Set level to WARNING."""
        self._alert.level = AlertLevel.WARNING
        return self

    def urgent(self) -> 'AlertBuilder':
        """Set level to URGENT."""
        self._alert.level = AlertLevel.URGENT
        return self

    def critical(self) -> 'AlertBuilder':
        """Set level to CRITICAL."""
        self._alert.level = AlertLevel.CRITICAL
        return self

    def level(self, level: AlertLevel) -> 'AlertBuilder':
        """Set alert level explicitly."""
        self._alert.level = level
        return self

    def titled(self, title: str) -> 'AlertBuilder':
        """Set title."""
        self._alert.title = title
        return self

    def message(self, message: str) -> 'AlertBuilder':
        """Set message."""
        self._alert.message = message
        return self

    def at(self, timestamp: datetime) -> 'AlertBuilder':
        """Set timestamp."""
        self._alert.timestamp = timestamp
        return self

    def metadata(self, **kwargs) -> 'AlertBuilder':
        """Add metadata."""
        self._alert.metadata.update(kwargs)
        return self

    def build(self) -> Alert:
        """Build the alert."""
        return self._alert


# =============================================================================
# Convenience Functions
# =============================================================================

def hive(name: str) -> HiveBuilder:
    """Start building a hive."""
    return HiveBuilder().named(name)


def colony() -> ColonyBuilder:
    """Start building a colony."""
    return ColonyBuilder()


def inspection() -> InspectionBuilder:
    """Start building an inspection."""
    return InspectionBuilder()


def alert() -> AlertBuilder:
    """Start building an alert."""
    return AlertBuilder()