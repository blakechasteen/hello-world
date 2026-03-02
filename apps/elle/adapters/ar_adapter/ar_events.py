"""
AR Event Models - Platform-agnostic AR event representations

Defines AR events (gaze, scan, selection, gesture, voice) and spatial context
that are translated to Elle requests.

Created: 2025-11-22
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime


# ============================================================================
# 3D Math Primitives
# ============================================================================

@dataclass(frozen=True)
class Vector3:
    """3D vector for positions and directions"""
    x: float
    y: float
    z: float

    def distance_to(self, other: "Vector3") -> float:
        """Calculate Euclidean distance to another vector"""
        import math
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def normalized(self) -> "Vector3":
        """Return normalized (unit length) vector"""
        import math
        length = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x/length, self.y/length, self.z/length)


@dataclass(frozen=True)
class Quaternion:
    """Quaternion for 3D rotations"""
    x: float
    y: float
    z: float
    w: float

    @staticmethod
    def identity() -> "Quaternion":
        """Return identity quaternion (no rotation)"""
        return Quaternion(0, 0, 0, 1)


# ============================================================================
# AR Object Representation
# ============================================================================

@dataclass
class ARObject:
    """Object detected in AR scene"""
    id: str
    label: str
    position: Vector3
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    confidence: float = 1.0
    object_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    bounding_box: Optional[Dict[str, Vector3]] = None  # min/max corners


# ============================================================================
# AR Context (Spatial Awareness)
# ============================================================================

@dataclass
class ARContext:
    """
    Complete AR spatial context at a moment in time.

    Represents user's position, orientation, gaze, and all visible objects.
    Integrated from hololoom/voice/ar_context.py patterns.
    """
    # User state
    user_position: Vector3
    user_rotation: Quaternion
    gaze_direction: Vector3
    gaze_target: Optional[str] = None  # Object ID if looking at something

    # Scene state
    visible_objects: List[ARObject] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    # Session info
    session_id: str = ""
    platform: str = "webxr"  # webxr, arcore, arkit

    def get_object_by_id(self, object_id: str) -> Optional[ARObject]:
        """Find object by ID"""
        for obj in self.visible_objects:
            if obj.id == object_id:
                return obj
        return None

    def get_closest_object(self, position: Optional[Vector3] = None) -> Optional[ARObject]:
        """Get closest object to user or given position"""
        if not self.visible_objects:
            return None

        ref_pos = position or self.user_position
        closest = None
        min_distance = float('inf')

        for obj in self.visible_objects:
            dist = ref_pos.distance_to(obj.position)
            if dist < min_distance:
                min_distance = dist
                closest = obj

        return closest

    def get_objects_in_gaze(self, cone_angle: float = 15.0) -> List[ARObject]:
        """
        Get objects within gaze cone.

        Args:
            cone_angle: Cone half-angle in degrees (default 15° = 30° total cone)
        """
        import math

        if not self.visible_objects:
            return []

        gaze = self.gaze_direction.normalized()
        in_gaze = []

        for obj in self.visible_objects:
            # Vector from user to object
            to_obj = Vector3(
                obj.position.x - self.user_position.x,
                obj.position.y - self.user_position.y,
                obj.position.z - self.user_position.z
            ).normalized()

            # Dot product = cos(angle)
            dot = gaze.x*to_obj.x + gaze.y*to_obj.y + gaze.z*to_obj.z
            angle_rad = math.acos(max(-1.0, min(1.0, dot)))
            angle_deg = math.degrees(angle_rad)

            if angle_deg <= cone_angle:
                in_gaze.append(obj)

        return in_gaze


# ============================================================================
# AR Event Types
# ============================================================================

class AREventType(str, Enum):
    """Types of AR events"""
    GAZE = "gaze"                # User looking at something
    SCAN = "scan"                # User scanning environment
    SELECTION = "selection"      # User selected an object
    GESTURE = "gesture"          # Hand/body gesture
    VOICE = "voice"              # Voice command
    PLACEMENT = "placement"      # User placed virtual object
    SESSION_START = "session_start"
    SESSION_END = "session_end"


# ============================================================================
# AR Event Base
# ============================================================================

@dataclass
class AREvent:
    """Base AR event"""
    event_type: AREventType
    context: ARContext
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Specific Event Types
# ============================================================================

@dataclass
class GazeEvent(AREvent):
    """
    User gazing at an object.

    Triggered when user's gaze dwells on an object for >500ms.
    """
    target_object_id: str = ""
    dwell_duration_ms: float = 0.0

    def __post_init__(self):
        self.event_type = AREventType.GAZE


@dataclass
class ScanEvent(AREvent):
    """
    User scanning environment.

    Triggered by slow head movement pattern (Elle's "slow scan" detection).
    """
    scan_type: str = "slow_scan"  # slow_scan, quick_scan, spatial_mapping
    objects_discovered: List[str] = field(default_factory=list)  # Object IDs

    def __post_init__(self):
        self.event_type = AREventType.SCAN


@dataclass
class SelectionEvent(AREvent):
    """
    User selected an object (tap, voice, gesture).
    """
    target_object_id: str = ""
    selection_method: str = "tap"  # tap, voice, gesture, gaze+dwell

    def __post_init__(self):
        self.event_type = AREventType.SELECTION


@dataclass
class GestureEvent(AREvent):
    """
    Hand or body gesture.
    """
    gesture_type: str = ""  # point, grab, swipe, wave, pinch, etc.
    hand: str = "right"  # left, right, both
    target_object_id: Optional[str] = None
    gesture_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.event_type = AREventType.GESTURE


@dataclass
class VoiceEvent(AREvent):
    """
    Voice command with AR context.

    Integrates with HoloLoom/voice/voice_agent.py patterns.
    """
    transcript: str = ""
    intent: Optional[str] = None  # Classified intent (query, navigate, select, command)
    entities: Dict[str, Any] = field(default_factory=dict)  # Extracted entities
    confidence: float = 1.0
    language: str = "en-US"

    def __post_init__(self):
        self.event_type = AREventType.VOICE
