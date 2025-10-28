"""
Type definitions for Keep beekeeping application.
"""

from enum import Enum
from typing import TypedDict, Optional, List
from datetime import datetime


class HealthStatus(str, Enum):
    """Health status of a colony."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class QueenStatus(str, Enum):
    """Status of the queen bee."""
    PRESENT_LAYING = "present_laying"
    PRESENT_NOT_LAYING = "present_not_laying"
    ABSENT = "absent"
    CELLS_PRESENT = "cells_present"
    VIRGIN = "virgin"


class HiveType(str, Enum):
    """Type of hive structure."""
    LANGSTROTH = "langstroth"
    TOP_BAR = "top_bar"
    WARRE = "warre"
    FLOW_HIVE = "flow_hive"
    OBSERVATION = "observation"


class InspectionType(str, Enum):
    """Type of hive inspection."""
    ROUTINE = "routine"
    HEALTH_CHECK = "health_check"
    SWARM_CHECK = "swarm_check"
    HARVEST = "harvest"
    FEEDING = "feeding"
    TREATMENT = "treatment"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    URGENT = "urgent"
    CRITICAL = "critical"


class InspectionData(TypedDict, total=False):
    """Data collected during an inspection."""
    temperature: float
    frames_with_brood: int
    frames_with_honey: int
    frames_with_pollen: int
    population_estimate: int
    queen_seen: bool
    eggs_seen: bool
    larvae_seen: bool
    capped_brood_seen: bool
    swarm_cells: int
    supersedure_cells: int
    mites_observed: bool
    beetles_observed: bool
    moths_observed: bool
    disease_signs: List[str]
    notes: str
