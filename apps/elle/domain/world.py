"""World areas and spatial organization."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AreaType(Enum):
    """Categories of spaces."""
    STRUCTURE = "structure"  # Shed, house, greenhouse
    GARDEN = "garden"  # Raised beds, rows
    BOUNDARY = "boundary"  # Fences, gates, paths
    UTILITY = "utility"  # Compost, water, storage
    NATURAL = "natural"  # Trees, wild areas


@dataclass(frozen=True)
class WorldArea:
    """
    A defined space in the physical world.
    
    Examples: "shed", "cucumber_bed_3", "front_fence", "compost_corner"
    """
    
    # Identity
    area_id: str
    name: str
    area_type: AreaType
    
    # Spatial
    location_description: str  # "Northeast corner of property"
    size_description: Optional[str] = None  # "4x8 bed", "8x10 shed"
    
    # Relationships
    parent_area_id: Optional[str] = None  # e.g., shed is in "north_property"
    adjacent_areas: List[str] = field(default_factory=list)  # Neighbor area IDs
    
    # State
    current_condition: Optional[str] = None  # "well-maintained", "needs-attention", "critical"
    tags: List[str] = field(default_factory=list)  # ["active", "seasonal", "project"]
    
    # Purpose
    intended_use: Optional[str] = None  # "tool storage", "cucumber cultivation"
    current_use: Optional[str] = None  # Might differ from intended
    
    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialContext:
    """
    Understanding of how areas relate in space.
    
    Used for reasoning about access, visibility, workflow.
    """
    
    # Core areas
    areas: List[WorldArea]
    
    # Relationships
    proximity_map: Dict[str, List[str]] = field(default_factory=dict)  # area_id -> nearby area_ids
    access_paths: Dict[str, List[str]] = field(default_factory=dict)  # area_id -> path through other areas
    
    # Context
    current_season: Optional[str] = None  # Affects which areas are active
    
    def get_area(self, area_id: str) -> Optional[WorldArea]:
        """Get area by ID."""
        for area in self.areas:
            if area.area_id == area_id:
                return area
        return None
    
    def nearby_areas(self, area_id: str) -> List[WorldArea]:
        """Get areas near a given area."""
        nearby_ids = self.proximity_map.get(area_id, [])
        return [area for area in self.areas if area.area_id in nearby_ids]
