"""Domain models for Elle's world understanding."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

__all__ = [
    'SceneSnapshot',
    'ObjectInScene',
    'SpatialRelation',
]


@dataclass(frozen=True)
class ObjectInScene:
    """An object detected in the scene."""
    
    id: str
    name: str
    category: str  # e.g., "tool", "plant", "material", "structure"
    location: str  # Where in the scene: "left shelf", "ground", "hanging"
    
    # Optional details
    condition: Optional[str] = None  # e.g., "rusty", "overgrown", "broken"
    quantity: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialRelation:
    """Describes relationship between objects in space."""
    
    subject: str  # Object ID
    predicate: str  # "on", "near", "blocking", "beside"
    object: str  # Object ID


@dataclass(frozen=True)
class SceneSnapshot:
    """
    Complete snapshot of what Elle perceives in a moment.
    
    This is the "world state" that feeds into decision-making.
    Frozen = immutable = pure functions all the way down.
    """
    
    # Identity
    scene_id: str
    timestamp: datetime
    location: str  # "shed", "cucumber_bed_3", "front_fence"
    
    # What's there
    objects: List[ObjectInScene]
    relations: List[SpatialRelation]
    
    # Ambient conditions
    weather: Optional[str] = None  # "sunny", "overcast", "rain"
    time_of_day: Optional[str] = None  # "morning", "afternoon", "evening"
    
    # High-level assessment (could come from vision tool)
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)  # ["cluttered", "urgent", "peaceful"]
    
    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def object_count(self) -> int:
        """Total objects in scene."""
        return len(self.objects)
    
    def objects_by_category(self, category: str) -> List[ObjectInScene]:
        """Filter objects by category."""
        return [obj for obj in self.objects if obj.category == category]
    
    def has_tag(self, tag: str) -> bool:
        """Check if scene has a specific tag."""
        return tag in self.tags
