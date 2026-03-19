"""
AR Context Module
==================
Data structures for augmented reality context integration.

This module provides the ARContext dataclass and related structures for
integrating VoiceAgent with Elle AR assistant.

Author: HoloLoom Team
Date: November 15, 2025
"""

import math
from dataclasses import dataclass, field
from enum import Enum

# ============================================================================
# Vector and Spatial Types
# ============================================================================

@dataclass
class Vector3:
    """3D vector for positions and directions."""
    x: float
    y: float
    z: float

    def distance_to(self, other: 'Vector3') -> float:
        """Calculate Euclidean distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def normalize(self) -> 'Vector3':
        """Return normalized vector (length 1)."""
        length = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / length, self.y / length, self.z / length)

    def dot(self, other: 'Vector3') -> float:
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


@dataclass
class Quaternion:
    """Quaternion for representing rotations."""
    x: float
    y: float
    z: float
    w: float

    def to_euler(self) -> tuple[float, float, float]:
        """Convert to Euler angles (pitch, yaw, roll) in radians."""
        # Pitch (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        pitch = math.atan2(sinr_cosp, cosr_cosp)

        # Yaw (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            yaw = math.copysign(math.pi / 2, sinp)
        else:
            yaw = math.asin(sinp)

        # Roll (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        roll = math.atan2(siny_cosp, cosy_cosp)

        return (pitch, yaw, roll)

    def __str__(self) -> str:
        pitch, yaw, roll = self.to_euler()
        return f"Quaternion(pitch={math.degrees(pitch):.1f}°, yaw={math.degrees(yaw):.1f}°, roll={math.degrees(roll):.1f}°)"


# ============================================================================
# AR Object Types
# ============================================================================

class ARObjectType(Enum):
    """Types of AR objects."""
    BEEHIVE = "beehive"
    FRAME = "frame"
    TOOL = "tool"
    MARKER = "marker"
    ANNOTATION = "annotation"
    INFO_PANEL = "info_panel"
    NAVIGATION_ARROW = "navigation_arrow"
    OTHER = "other"


@dataclass
class ARObject:
    """Represents an object in AR space."""
    id: str
    type: ARObjectType
    position: Vector3
    name: str = ""
    metadata: dict = field(default_factory=dict)
    visible: bool = True
    selectable: bool = True

    def distance_from(self, position: Vector3) -> float:
        """Calculate distance from a position."""
        return self.position.distance_to(position)

    def __str__(self) -> str:
        return f"ARObject({self.id}, {self.type.value}, {self.position})"


# ============================================================================
# AR Events
# ============================================================================

class AREventType(Enum):
    """Types of AR events."""
    OBJECT_SELECTED = "object_selected"
    OBJECT_DESELECTED = "object_deselected"
    NAVIGATION_STARTED = "navigation_started"
    NAVIGATION_COMPLETED = "navigation_completed"
    GESTURE_DETECTED = "gesture_detected"
    GAZE_CHANGED = "gaze_changed"
    SCENE_CHANGED = "scene_changed"


@dataclass
class AREvent:
    """Represents an AR event."""
    type: AREventType
    timestamp: float
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"AREvent({self.type.value}, t={self.timestamp:.2f})"


@dataclass
class SelectionEvent(AREvent):
    """Object selection event."""
    object_id: str = ""
    object_type: str = ""
    selection_method: str = "gaze"  # "gaze", "tap", "gesture", "voice"

    def __post_init__(self):
        self.type = AREventType.OBJECT_SELECTED
        self.data = {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "selection_method": self.selection_method
        }


@dataclass
class NavigationEvent(AREvent):
    """Navigation event."""
    destination: str = ""
    distance_remaining: float = 0.0
    estimated_time: float = 0.0

    def __post_init__(self):
        self.type = AREventType.NAVIGATION_STARTED
        self.data = {
            "destination": self.destination,
            "distance_remaining": self.distance_remaining,
            "estimated_time": self.estimated_time
        }


class GestureType(Enum):
    """Types of hand gestures."""
    POINT = "point"
    CIRCLE = "circle"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH = "pinch"
    SPREAD = "spread"
    TAP = "tap"
    UNKNOWN = "unknown"


@dataclass
class GestureEvent(AREvent):
    """Gesture detection event."""
    gesture_type: GestureType = GestureType.UNKNOWN
    target: str | None = None
    confidence: float = 0.0

    def __post_init__(self):
        self.type = AREventType.GESTURE_DETECTED
        self.data = {
            "gesture_type": self.gesture_type.value,
            "target": self.target,
            "confidence": self.confidence
        }


# ============================================================================
# AR Context
# ============================================================================

@dataclass
class ARContext:
    """
    Encapsulates current AR state for context-aware voice processing.

    This provides all the spatial and contextual information needed to
    resolve voice commands like "this hive" or "the one I'm looking at".

    Attributes:
        user_position: User's (x, y, z) position in AR space
        user_orientation: User's head/body orientation (quaternion)
        gaze_direction: Normalized vector of where user is looking
        gaze_target: ID of object user is looking at (if any)
        visible_objects: All objects currently in user's view
        selected_object: Currently selected object (if any)
        active_scene: Current AR scene/context identifier
        recent_actions: Last 5 actions performed
        conversation_context: Current conversation topic/state
        nearby_objects: Spatial references mapping ("this", "that", etc.)
    """

    # User state
    user_position: Vector3 = field(default_factory=lambda: Vector3(0, 1.7, 0))
    user_orientation: Quaternion = field(default_factory=lambda: Quaternion(0, 0, 0, 1))
    gaze_direction: Vector3 = field(default_factory=lambda: Vector3(0, 0, 1))
    gaze_target: str | None = None

    # Environment state
    visible_objects: list[ARObject] = field(default_factory=list)
    selected_object: ARObject | None = None
    active_scene: str = "default"

    # Interaction state
    recent_actions: list[str] = field(default_factory=list)
    conversation_context: str = ""

    # Spatial references (for "this", "that", etc.)
    nearby_objects: dict[str, ARObject] = field(default_factory=dict)

    def add_action(self, action: str) -> None:
        """Add an action to the recent actions list (max 5)."""
        self.recent_actions.append(action)
        if len(self.recent_actions) > 5:
            self.recent_actions.pop(0)

    def find_object_by_id(self, object_id: str) -> ARObject | None:
        """Find an object by ID in visible objects."""
        for obj in self.visible_objects:
            if obj.id == object_id:
                return obj
        return None

    def find_objects_by_type(self, object_type: ARObjectType) -> list[ARObject]:
        """Find all objects of a specific type."""
        return [obj for obj in self.visible_objects if obj.type == object_type]

    def find_closest_object(self, object_type: ARObjectType | None = None) -> ARObject | None:
        """
        Find the closest object to the user.

        Args:
            object_type: Optional type filter

        Returns:
            Closest object, or None if no objects visible
        """
        candidates = self.visible_objects
        if object_type:
            candidates = self.find_objects_by_type(object_type)

        if not candidates:
            return None

        return min(candidates, key=lambda obj: obj.distance_from(self.user_position))

    def find_object_in_direction(
        self,
        direction: str,
        object_type: ARObjectType | None = None
    ) -> ARObject | None:
        """
        Find object in a specific direction (left, right, front, back).

        Args:
            direction: "left", "right", "front", "back", "above", "below"
            object_type: Optional type filter

        Returns:
            Object in specified direction, or None
        """
        candidates = self.visible_objects
        if object_type:
            candidates = self.find_objects_by_type(object_type)

        if not candidates:
            return None

        # Get user's forward vector from orientation
        _, yaw, _ = self.user_orientation.to_euler()
        forward = Vector3(math.sin(yaw), 0, math.cos(yaw))
        right = Vector3(math.cos(yaw), 0, -math.sin(yaw))
        up = Vector3(0, 1, 0)

        # Direction vectors
        direction_vectors = {
            "front": forward,
            "back": Vector3(-forward.x, -forward.y, -forward.z),
            "left": Vector3(-right.x, -right.y, -right.z),
            "right": right,
            "above": up,
            "below": Vector3(0, -1, 0)
        }

        if direction.lower() not in direction_vectors:
            return None

        dir_vec = direction_vectors[direction.lower()]

        # Find object with maximum dot product in direction
        best_obj = None
        best_score = -float('inf')

        for obj in candidates:
            # Vector from user to object
            to_obj = Vector3(
                obj.position.x - self.user_position.x,
                obj.position.y - self.user_position.y,
                obj.position.z - self.user_position.z
            ).normalize()

            # Dot product gives alignment with direction
            score = to_obj.dot(dir_vec)

            if score > best_score and score > 0.5:  # At least 60° cone
                best_score = score
                best_obj = obj

        return best_obj

    def update_spatial_references(self) -> None:
        """
        Update the nearby_objects mapping for spatial references.

        Maps:
        - "this" → gaze target or closest object
        - "that" → second closest object
        - "nearest" → closest object
        """
        self.nearby_objects.clear()

        # "this" = gaze target if available, otherwise closest
        if self.gaze_target:
            gaze_obj = self.find_object_by_id(self.gaze_target)
            if gaze_obj:
                self.nearby_objects["this"] = gaze_obj

        if "this" not in self.nearby_objects:
            closest = self.find_closest_object()
            if closest:
                self.nearby_objects["this"] = closest
                self.nearby_objects["nearest"] = closest

        # "that" = second closest object
        if self.visible_objects:
            sorted_by_distance = sorted(
                self.visible_objects,
                key=lambda obj: obj.distance_from(self.user_position)
            )
            if len(sorted_by_distance) > 1:
                self.nearby_objects["that"] = sorted_by_distance[1]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "user_position": [self.user_position.x, self.user_position.y, self.user_position.z],
            "gaze_target": self.gaze_target,
            "visible_objects_count": len(self.visible_objects),
            "selected_object": self.selected_object.id if self.selected_object else None,
            "active_scene": self.active_scene,
            "recent_actions": self.recent_actions,
            "conversation_context": self.conversation_context
        }

    def __str__(self) -> str:
        return (
            f"ARContext(user_pos={self.user_position}, "
            f"gaze_target={self.gaze_target}, "
            f"visible_objects={len(self.visible_objects)}, "
            f"scene={self.active_scene})"
        )


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_context() -> ARContext:
    """
    Create a test AR context for development and testing.

    Returns:
        ARContext with sample beehive scenario
    """
    context = ARContext(
        user_position=Vector3(0, 1.7, 0),  # 1.7m height (average human eye level)
        user_orientation=Quaternion(0, 0, 0, 1),  # Facing forward
        gaze_direction=Vector3(0, 0, 1),  # Looking forward
        active_scene="beekeeping_inspection"
    )

    # Add sample beehives
    context.visible_objects = [
        ARObject(
            id="hive_001",
            type=ARObjectType.BEEHIVE,
            position=Vector3(2, 0.5, 5),
            name="Hive 001",
            metadata={"health": 0.87, "population": 45000, "last_inspection": "2025-11-10"}
        ),
        ARObject(
            id="hive_002",
            type=ARObjectType.BEEHIVE,
            position=Vector3(-2, 0.5, 5),
            name="Hive 002",
            metadata={"health": 0.92, "population": 52000, "last_inspection": "2025-11-12"}
        ),
        ARObject(
            id="hive_003",
            type=ARObjectType.BEEHIVE,
            position=Vector3(0, 0.5, 3),
            name="Hive 003",
            metadata={"health": 0.65, "population": 38000, "last_inspection": "2025-11-08"}
        ),
        ARObject(
            id="smoker_001",
            type=ARObjectType.TOOL,
            position=Vector3(-1, 0.8, 1),
            name="Smoker",
            metadata={"tool_type": "smoker"}
        )
    ]

    # User is looking at hive_003 (closest and in gaze direction)
    context.gaze_target = "hive_003"

    # Update spatial references
    context.update_spatial_references()

    return context


if __name__ == "__main__":
    # Example usage
    print("Creating test AR context...")
    context = create_test_context()

    print(f"\n{context}")
    print(f"\nUser position: {context.user_position}")
    print(f"Gaze target: {context.gaze_target}")
    print(f"Visible objects: {len(context.visible_objects)}")

    print("\nSpatial references:")
    for ref, obj in context.nearby_objects.items():
        print(f"  '{ref}' → {obj.id} at {obj.position}")

    print("\nFinding closest beehive:")
    closest = context.find_closest_object(ARObjectType.BEEHIVE)
    if closest:
        distance = closest.distance_from(context.user_position)
        print(f"  {closest.id} ({closest.name}) at {distance:.2f}m")

    print("\nFinding hive to the left:")
    left_hive = context.find_object_in_direction("left", ARObjectType.BEEHIVE)
    if left_hive:
        print(f"  {left_hive.id} ({left_hive.name})")
