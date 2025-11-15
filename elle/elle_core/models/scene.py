"""
Elle Core - Scene Models

Defines the physical environment representation.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class ObjectType(Enum):
    """Categories of physical objects Elle might reference."""
    TOOL = "tool"
    CONTAINER = "container"
    FURNITURE = "furniture"
    PLANT = "plant"
    STRUCTURE = "structure"
    MATERIAL = "material"
    EQUIPMENT = "equipment"
    ANIMAL = "animal"
    UNKNOWN = "unknown"


@dataclass
class SceneObject:
    """
    A physical object in the scene.

    Detected via vision, manually tagged, or inferred from context.
    """
    id: str  # Unique identifier (e.g., "shop_vac_1", "toolbox_red")
    name: str  # Human-readable name
    type: ObjectType = ObjectType.UNKNOWN
    location: Optional[Tuple[float, float, float]] = None  # 3D coordinates if known
    properties: Dict[str, Any] = field(default_factory=dict)  # e.g., {"color": "red", "size": "large"}
    visible: bool = True  # Is it currently visible to user?
    accessible: bool = True  # Can user easily reach it?

    def __str__(self) -> str:
        return f"{self.name} ({self.id})"


@dataclass
class SceneSnapshot:
    """
    A snapshot of the physical environment at a moment in time.

    This is what Elle "sees" - the world state she'll reason about.
    """
    description: str  # Natural language description of the scene
    location: str  # Where this scene is (e.g., "shed workshop", "vegetable garden")
    objects: List[SceneObject] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Weather, time, etc.

    def get_object(self, object_id: str) -> Optional[SceneObject]:
        """Find an object by ID."""
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def get_objects_by_type(self, obj_type: ObjectType) -> List[SceneObject]:
        """Get all objects of a specific type."""
        return [obj for obj in self.objects if obj.type == obj_type]

    def has_object(self, object_id: str) -> bool:
        """Check if object exists in scene."""
        return any(obj.id == object_id for obj in self.objects)

    def __str__(self) -> str:
        obj_count = len(self.objects)
        return f"{self.location}: {self.description} ({obj_count} objects)"
