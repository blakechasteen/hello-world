# -*- coding: utf-8 -*-
"""
Farm Core Models
================
Base classes for farm management entities.

Inspired by farmOS:
- Assets: Physical/biological things on farm
- Logs: Events, observations, activities
- Relationships: Graph connections between entities
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

# Import HoloLoom types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from HoloLoom.Documentation.types import MemoryShard
except ImportError:
    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)


class AssetType(Enum):
    """farmOS asset types."""
    LAND = "land"              # Fields, beds, greenhouses
    PLANT = "plant"            # Plantings, trees, crops
    ANIMAL = "animal"          # Livestock, bees
    EQUIPMENT = "equipment"    # Tractors, tools, machinery
    STRUCTURE = "structure"    # Buildings, fences, infrastructure
    WATER = "water"            # Ponds, irrigation systems
    MATERIAL = "material"      # Soil, compost, feed, seed
    SENSOR = "sensor"          # IoT devices, monitors
    GROUP = "group"            # Herds, flocks, colonies
    OTHER = "other"


class LogType(Enum):
    """farmOS log types."""
    ACTIVITY = "activity"           # General farm activities
    OBSERVATION = "observation"     # Passive observations
    INPUT = "input"                 # Resources applied (feed, fertilizer)
    HARVEST = "harvest"             # Yields collected
    SEEDING = "seeding"             # Planting seeds
    TRANSPLANTING = "transplanting" # Moving plants
    MAINTENANCE = "maintenance"     # Equipment/structure upkeep
    MEDICAL = "medical"             # Animal health treatments
    BIRTH = "birth"                 # Animal births
    TEST = "test"                   # Soil, water, tissue tests
    SALE = "sale"                   # Sales transactions
    PURCHASE = "purchase"           # Purchase transactions
    OTHER = "other"


class AssetStatus(Enum):
    """Status of farm assets."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    SOLD = "sold"
    DIED = "died"
    COMBINED = "combined"
    SPLIT = "split"
    LOST = "lost"


@dataclass
class Asset:
    """
    Base class for all farm assets.

    Represents physical/biological things on the farm.
    Stored as nodes in Neo4j graph with properties.
    """
    asset_id: str
    asset_type: AssetType
    name: str
    status: AssetStatus = AssetStatus.ACTIVE
    location: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)
    archived_date: Optional[datetime] = None

    # Relationships (stored as graph edges)
    parent_ids: List[str] = field(default_factory=list)  # For splits, offspring
    group_ids: List[str] = field(default_factory=list)   # Group memberships

    # Additional data
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, asset_type: AssetType, **kwargs):
        """Factory method to create asset with auto-generated ID."""
        asset_id = kwargs.pop('asset_id', f"{asset_type.value}_{uuid.uuid4().hex[:8]}")
        return cls(asset_id=asset_id, name=name, asset_type=asset_type, **kwargs)

    def to_shard(self) -> MemoryShard:
        """Convert asset to memory shard for HoloLoom storage."""
        text_parts = [
            f"Asset: {self.name} ({self.asset_type.value})",
            f"ID: {self.asset_id}",
            f"Status: {self.status.value}"
        ]

        if self.location:
            text_parts.append(f"Location: {self.location}")

        if self.notes:
            text_parts.append(f"Notes: {self.notes}")

        text = "\n".join(text_parts)

        entities = [self.asset_id, self.name]
        if self.location:
            entities.append(self.location)

        motifs = [
            'asset',
            self.asset_type.value,
            f'status_{self.status.value}'
        ] + self.tags

        return MemoryShard(
            id=f"asset_{self.asset_id}",
            text=text,
            episode=f"asset_metadata_{self.asset_id}",
            entities=entities,
            motifs=motifs,
            metadata={
                'type': 'asset',
                'asset_type': self.asset_type.value,
                'asset_id': self.asset_id,
                'created_date': self.created_date.isoformat(),
                **self.metadata
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'asset_id': self.asset_id,
            'asset_type': self.asset_type.value,
            'name': self.name,
            'status': self.status.value,
            'location': self.location,
            'created_date': self.created_date.isoformat(),
            'archived_date': self.archived_date.isoformat() if self.archived_date else None,
            'parent_ids': self.parent_ids,
            'group_ids': self.group_ids,
            'tags': self.tags,
            'notes': self.notes,
            'metadata': self.metadata
        }


@dataclass
class Log:
    """
    Base class for all farm event logs.

    Records activities, observations, and transactions.
    Stored as MemoryShards with relationships to assets.
    """
    log_id: str
    log_type: LogType
    name: str  # Human-readable title
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""

    # Relationships
    asset_ids: List[str] = field(default_factory=list)  # Assets involved
    location: Optional[str] = None

    # User context
    user: Optional[str] = None

    # Additional data
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, log_type: LogType, **kwargs):
        """Factory method to create log with auto-generated ID."""
        log_id = kwargs.pop('log_id', f"{log_type.value}_{uuid.uuid4().hex[:8]}")
        timestamp = kwargs.pop('timestamp', datetime.now())
        return cls(log_id=log_id, name=name, log_type=log_type, timestamp=timestamp, **kwargs)

    def to_shard(self) -> MemoryShard:
        """Convert log to memory shard for HoloLoom storage."""
        text_parts = [
            f"{self.log_type.value.title()}: {self.name}",
            f"Date: {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
        ]

        if self.asset_ids:
            text_parts.append(f"Assets: {', '.join(self.asset_ids)}")

        if self.location:
            text_parts.append(f"Location: {self.location}")

        if self.user:
            text_parts.append(f"User: {self.user}")

        if self.description:
            text_parts.append(f"\n{self.description}")

        text = "\n".join(text_parts)

        entities = [self.log_id] + self.asset_ids
        if self.user:
            entities.append(self.user)
        if self.location:
            entities.append(self.location)

        motifs = [
            'log',
            self.log_type.value,
            f'log_{self.log_type.value}'
        ] + self.tags

        return MemoryShard(
            id=f"log_{self.log_id}",
            text=text,
            episode=f"log_{self.timestamp.strftime('%Y%m%d_%H%M%S')}",
            entities=entities,
            motifs=motifs,
            metadata={
                'type': 'log',
                'log_type': self.log_type.value,
                'log_id': self.log_id,
                'timestamp': self.timestamp.isoformat(),
                'asset_ids': self.asset_ids,
                **self.metadata
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'log_id': self.log_id,
            'log_type': self.log_type.value,
            'timestamp': self.timestamp.isoformat(),
            'name': self.name,
            'description': self.description,
            'asset_ids': self.asset_ids,
            'location': self.location,
            'user': self.user,
            'tags': self.tags,
            'metadata': self.metadata
        }


# Specialized log types with additional fields

@dataclass
class ObservationLog(Log):
    """Observation log with measurements."""
    # Measurements
    measurements: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.log_type != LogType.OBSERVATION:
            object.__setattr__(self, 'log_type', LogType.OBSERVATION)


@dataclass
class InputLog(Log):
    """Input log for resources applied."""
    # What was applied
    material: str = ""
    quantity: float = 0.0
    unit: str = ""
    method: str = ""  # Spray, broadcast, injection, etc.

    def __post_init__(self):
        if self.log_type != LogType.INPUT:
            object.__setattr__(self, 'log_type', LogType.INPUT)


@dataclass
class HarvestLog(Log):
    """Harvest log with yields."""
    # Yield data
    quantity: float = 0.0
    unit: str = ""
    quality: Optional[str] = None

    def __post_init__(self):
        if self.log_type != LogType.HARVEST:
            object.__setattr__(self, 'log_type', LogType.HARVEST)


@dataclass
class MedicalLog(Log):
    """Medical treatment log for animals."""
    # Treatment details
    treatment: str = ""
    dosage: str = ""
    withdrawal_date: Optional[datetime] = None
    veterinarian: Optional[str] = None

    def __post_init__(self):
        if self.log_type != LogType.MEDICAL:
            object.__setattr__(self, 'log_type', LogType.MEDICAL)


@dataclass
class TestLog(Log):
    """Test/analysis log (soil, water, tissue)."""
    # Test results
    test_type: str = ""  # Soil, water, tissue, etc.
    lab: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.log_type != LogType.TEST:
            object.__setattr__(self, 'log_type', LogType.TEST)