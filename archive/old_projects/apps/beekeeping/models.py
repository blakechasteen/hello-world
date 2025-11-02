# -*- coding: utf-8 -*-
"""
Beekeeping Models
=================
Domain-specific models for beekeeping/apiary management.

Assets:
- Hive: Individual beehive/colony
- Queen: Queen bee (can be tracked separately)
- Apiary: Location grouping multiple hives

Logs:
- Inspection: Hive inspection observations
- Treatment: Varroa, disease treatments
- Feeding: Sugar syrup, pollen patties
- Harvest: Honey, wax, propolis
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from apps.farm_core.models import Asset, Log, AssetType, LogType, AssetStatus, ObservationLog


class HiveEquipmentType(Enum):
    """Types of beehive equipment."""
    LANGSTROTH = "langstroth"
    TOP_BAR = "top_bar"
    WARRE = "warre"
    FLOW_HIVE = "flow_hive"
    OTHER = "other"


class QueenStatus(Enum):
    """Queen bee status."""
    PRESENT_LAYING = "present_laying"
    PRESENT_NOT_CONFIRMED = "present_not_confirmed"
    QUEENLESS = "queenless"
    QUEEN_CELLS = "queen_cells"
    SUPERSEDURE = "supersedure"
    UNKNOWN = "unknown"


class PopulationStrength(Enum):
    """Colony population strength."""
    STRONG = "strong"      # 8+ frames of bees
    MODERATE = "moderate"  # 5-7 frames
    WEAK = "weak"          # <5 frames
    DEAD = "dead"
    UNKNOWN = "unknown"


class Temperament(Enum):
    """Hive temperament."""
    CALM = "calm"
    GENTLE = "gentle"
    MODERATE = "moderate"
    DEFENSIVE = "defensive"
    AGGRESSIVE = "aggressive"
    UNKNOWN = "unknown"


@dataclass
class Hive(Asset):
    """
    Beehive asset.

    Represents a single colony/hive in an apiary.
    """
    # Equipment
    equipment_type: HiveEquipmentType = HiveEquipmentType.LANGSTROTH
    num_boxes: int = 2

    # Queen
    queen_status: QueenStatus = QueenStatus.UNKNOWN
    queen_marked: bool = False
    queen_color: Optional[str] = None  # Year marking color
    queen_source: Optional[str] = None  # Package, split, swarm, purchased

    # Population
    population: PopulationStrength = PopulationStrength.UNKNOWN
    frames_of_bees: Optional[int] = None

    # Behavior
    temperament: Temperament = Temperament.UNKNOWN

    # Tracking
    last_inspection: Optional[datetime] = None
    last_harvest: Optional[datetime] = None
    last_treatment: Optional[datetime] = None

    def __post_init__(self):
        # Ensure asset_type is set to ANIMAL for bees
        if self.asset_type != AssetType.ANIMAL:
            object.__setattr__(self, 'asset_type', AssetType.ANIMAL)

    @classmethod
    def create(cls, name: str, **kwargs):
        """Create hive with auto-generated ID."""
        asset_id = kwargs.pop('asset_id', f"hive_{name.lower().replace(' ', '_')}")
        return super().create(name=name, asset_type=AssetType.ANIMAL, asset_id=asset_id, **kwargs)


@dataclass
class Queen(Asset):
    """
    Queen bee asset (optional separate tracking).

    For beekeepers who track queen lineage carefully.
    """
    # Queen details
    birth_date: Optional[datetime] = None
    marked: bool = False
    marking_color: Optional[str] = None
    breeder: Optional[str] = None
    genetics: Optional[str] = None  # Breed/strain

    # Performance
    temperament: Temperament = Temperament.UNKNOWN
    productivity: Optional[str] = None
    disease_resistance: Optional[str] = None

    def __post_init__(self):
        if self.asset_type != AssetType.ANIMAL:
            object.__setattr__(self, 'asset_type', AssetType.ANIMAL)

    @classmethod
    def create(cls, name: str, **kwargs):
        """Create queen with auto-generated ID."""
        asset_id = kwargs.pop('asset_id', f"queen_{name.lower().replace(' ', '_')}")
        return super().create(name=name, asset_type=AssetType.ANIMAL, asset_id=asset_id, **kwargs)


@dataclass
class Apiary(Asset):
    """
    Apiary location (group of hives).

    Represents a physical location where hives are kept.
    """
    # Location details
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: Optional[str] = None

    # Environment
    forage_type: Optional[str] = None  # Urban, agricultural, forest, etc.
    sun_exposure: Optional[str] = None  # Full sun, partial shade, etc.

    def __post_init__(self):
        if self.asset_type != AssetType.GROUP:
            object.__setattr__(self, 'asset_type', AssetType.GROUP)

    @classmethod
    def create(cls, name: str, **kwargs):
        """Create apiary with auto-generated ID."""
        asset_id = kwargs.pop('asset_id', f"apiary_{name.lower().replace(' ', '_')}")
        return super().create(name=name, asset_type=AssetType.GROUP, asset_id=asset_id, **kwargs)


# === Specialized Logs ===

@dataclass
class InspectionLog(ObservationLog):
    """
    Hive inspection log.

    Records observations during hive inspection.
    """
    # Queen status
    queen_status: QueenStatus = QueenStatus.UNKNOWN
    queen_seen: bool = False
    eggs_seen: bool = False
    larvae_seen: bool = False

    # Population & brood
    population: PopulationStrength = PopulationStrength.UNKNOWN
    frames_of_bees: Optional[int] = None
    frames_of_brood: Optional[int] = None
    brood_pattern: Optional[str] = None  # Solid, spotty, poor

    # Health
    mites_observed: bool = False
    disease_signs: List[str] = field(default_factory=list)
    pests: List[str] = field(default_factory=list)

    # Resources
    honey_stores: Optional[str] = None  # Abundant, moderate, low
    pollen_stores: Optional[str] = None
    nectar_flow: bool = False

    # Behavior
    temperament: Temperament = Temperament.UNKNOWN

    # Actions taken
    boxes_added: int = 0
    boxes_removed: int = 0
    frames_added: int = 0
    frames_removed: int = 0

    # Weather
    weather: Optional[str] = None
    temperature: Optional[float] = None

    @classmethod
    def create(cls, name: str, hive_id: str, **kwargs):
        """Create inspection log."""
        log_id = kwargs.pop('log_id', f"inspection_{hive_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        asset_ids = kwargs.pop('asset_ids', [hive_id])
        return super().create(name=name, log_id=log_id, asset_ids=asset_ids, **kwargs)


@dataclass
class TreatmentLog(Log):
    """
    Treatment log for varroa, diseases, pests.
    """

    # Treatment details
    treatment_type: str = ""  # Varroa, nosema, foulbrood, etc.
    product: str = ""
    dosage: str = ""
    method: str = ""  # Drip, strip, spray, etc.

    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    withdrawal_date: Optional[datetime] = None  # Safe to harvest after

    @classmethod
    def create(cls, name: str, hive_id: str, **kwargs):
        """Create treatment log."""
        log_id = kwargs.pop('log_id', f"treatment_{hive_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        asset_ids = kwargs.pop('asset_ids', [hive_id])
        return super().create(name=name, log_id=log_id, asset_ids=asset_ids, **kwargs)


@dataclass
class FeedingLog(Log):
    """
    Feeding log for sugar syrup, pollen patties, fondant.
    """

    # Feed details
    feed_type: str = ""  # Sugar syrup, pollen patty, fondant, etc.
    quantity: float = 0.0
    unit: str = ""  # liters, pounds, patties, etc.
    ratio: Optional[str] = None  # 1:1, 2:1 for syrup

    # Reason
    reason: str = ""  # Build-up, winter stores, emergency, etc.

    @classmethod
    def create(cls, name: str, hive_id: str, **kwargs):
        """Create feeding log."""
        log_id = kwargs.pop('log_id', f"feeding_{hive_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        asset_ids = kwargs.pop('asset_ids', [hive_id])
        return super().create(name=name, log_id=log_id, asset_ids=asset_ids, **kwargs)


@dataclass
class HarvestLog(Log):
    """
    Harvest log for honey, wax, propolis.
    """

    # Harvest details
    product: str = "honey"  # Honey, wax, propolis, pollen
    quantity: float = 0.0
    unit: str = "pounds"
    quality: Optional[str] = None  # Grade, moisture content

    # Processing
    moisture_content: Optional[float] = None
    capped_percentage: Optional[float] = None

    @classmethod
    def create(cls, name: str, hive_id: str, **kwargs):
        """Create harvest log."""
        log_id = kwargs.pop('log_id', f"harvest_{hive_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        asset_ids = kwargs.pop('asset_ids', [hive_id])
        return super().create(name=name, log_id=log_id, asset_ids=asset_ids, **kwargs)


@dataclass
class SplitLog(Log):
    """
    Hive split/divide log.

    Records splitting one hive into multiple.
    """

    # Split details
    parent_hive_id: str = ""
    child_hive_ids: List[str] = field(default_factory=list)
    frames_moved: int = 0
    queen_cells: int = 0

    # Method
    split_method: str = ""  # Walk-away, with cells, purchased queen, etc.

    @classmethod
    def create(cls, name: str, parent_hive_id: str, **kwargs):
        """Create split log."""
        log_id = kwargs.pop('log_id', f"split_{parent_hive_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        asset_ids = kwargs.pop('asset_ids', [parent_hive_id])
        return super().create(name=name, log_id=log_id, asset_ids=asset_ids, **kwargs)