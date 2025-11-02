"""
Domain models for Keep beekeeping application.

Models represent the core entities in beekeeping management:
- Hive: Physical hive structure and location
- Colony: The bee population living in a hive
- Inspection: Record of hive checks and observations
- HarvestRecord: Honey and product harvest tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4

from apps.keep.types import (
    HealthStatus,
    QueenStatus,
    HiveType,
    InspectionType,
    InspectionData,
    AlertLevel,
)


@dataclass
class Hive:
    """
    Represents a physical beehive structure.

    Attributes:
        hive_id: Unique identifier
        name: Human-readable name
        hive_type: Type of hive structure
        location: Physical location (GPS coordinates, address, or description)
        installation_date: When the hive was established
        notes: Additional information
    """
    hive_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    hive_type: HiveType = HiveType.LANGSTROTH
    location: str = ""
    installation_date: datetime = field(default_factory=datetime.now)
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Hive(name={self.name}, type={self.hive_type.value}, location={self.location})"


@dataclass
class Colony:
    """
    Represents a bee colony living in a hive.

    Attributes:
        colony_id: Unique identifier
        hive_id: ID of the hive this colony occupies
        queen_status: Current status of the queen
        health_status: Overall colony health
        population_estimate: Estimated number of bees
        origin: Where the colony came from (nuc, package, swarm, split)
        established_date: When colony was introduced to the hive
        breed: Bee breed/genetics (Italian, Carniolan, etc.)
    """
    colony_id: str = field(default_factory=lambda: str(uuid4()))
    hive_id: str = ""
    queen_status: QueenStatus = QueenStatus.PRESENT_LAYING
    health_status: HealthStatus = HealthStatus.GOOD
    population_estimate: int = 0
    origin: str = "unknown"
    established_date: datetime = field(default_factory=datetime.now)
    breed: str = "unknown"
    queen_age_months: Optional[int] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Colony(id={self.colony_id[:8]}, health={self.health_status.value}, queen={self.queen_status.value})"


@dataclass
class Inspection:
    """
    Record of a hive inspection.

    Attributes:
        inspection_id: Unique identifier
        hive_id: Hive that was inspected
        colony_id: Colony that was inspected
        timestamp: When inspection occurred
        inspection_type: Type of inspection performed
        weather: Weather conditions during inspection
        temperature: Ambient temperature
        findings: Structured data collected
        actions_taken: List of actions performed
        recommendations: Suggested next steps
        next_inspection_due: When to inspect again
    """
    inspection_id: str = field(default_factory=lambda: str(uuid4()))
    hive_id: str = ""
    colony_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    inspection_type: InspectionType = InspectionType.ROUTINE
    weather: str = ""
    temperature: Optional[float] = None
    findings: InspectionData = field(default_factory=dict)
    actions_taken: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_inspection_due: Optional[datetime] = None
    inspector: str = ""
    duration_minutes: Optional[int] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Inspection(hive={self.hive_id[:8]}, type={self.inspection_type.value}, date={self.timestamp.date()})"


@dataclass
class HarvestRecord:
    """
    Record of honey or product harvest.

    Attributes:
        harvest_id: Unique identifier
        hive_id: Source hive
        timestamp: When harvest occurred
        product_type: What was harvested (honey, wax, propolis, etc.)
        quantity: Amount harvested
        unit: Unit of measurement
        quality_notes: Quality assessment
        processing_notes: How product was processed
    """
    harvest_id: str = field(default_factory=lambda: str(uuid4()))
    hive_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    product_type: str = "honey"
    quantity: float = 0.0
    unit: str = "lbs"
    quality_notes: str = ""
    processing_notes: str = ""
    moisture_content: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Harvest(hive={self.hive_id[:8]}, product={self.product_type}, qty={self.quantity}{self.unit})"


@dataclass
class Alert:
    """
    Alert or notification about colony/hive status.

    Attributes:
        alert_id: Unique identifier
        hive_id: Related hive
        colony_id: Related colony
        level: Severity level
        title: Brief description
        message: Detailed message
        timestamp: When alert was created
        resolved: Whether alert has been addressed
        resolved_at: When alert was resolved
    """
    alert_id: str = field(default_factory=lambda: str(uuid4()))
    hive_id: str = ""
    colony_id: str = ""
    level: AlertLevel = AlertLevel.INFO
    title: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "resolved" if self.resolved else "active"
        return f"Alert({self.level.value}, {self.title}, {status})"