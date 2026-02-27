"""
HoloLoom Phase 4: Hand Tracking & Gesture System
November 2025

WebXR hand tracking for natural interaction with
spatial knowledge graph.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
import json
import math


class HandSide(Enum):
    """Which hand."""
    LEFT = "left"
    RIGHT = "right"


class GestureType(Enum):
    """Types of recognized gestures."""
    # Basic gestures
    POINT = "point"              # Index finger extended
    PINCH = "pinch"              # Thumb and index together
    GRAB = "grab"                # Closed fist
    FIST = "fist"                # Alias for grab/closed fist
    OPEN = "open"                # All fingers extended
    THUMBS_UP = "thumbs_up"      # Approval gesture
    THUMBS_DOWN = "thumbs_down"  # Disapproval gesture
    OPEN_PALM = "open_palm"      # Stop/wait gesture

    # Interaction gestures
    PINCH_HOLD = "pinch_hold"    # Holding a pinch
    SPREAD = "spread"            # Fingers spreading apart
    SWIPE_LEFT = "swipe_left"    # Quick leftward movement
    SWIPE_RIGHT = "swipe_right"  # Quick rightward movement
    SWIPE_UP = "swipe_up"        # Quick upward movement
    SWIPE_DOWN = "swipe_down"    # Quick downward movement

    # Two-hand gestures
    SCALE_UP = "scale_up"        # Hands moving apart
    SCALE_DOWN = "scale_down"    # Hands coming together
    ROTATE = "rotate"            # Rotational movement

    # Custom gestures
    DRAW_CIRCLE = "draw_circle"  # Circle motion
    WAVE = "wave"                # Waving motion


class GestureState(Enum):
    """State of gesture recognition."""
    NONE = "none"
    STARTED = "started"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class JointPosition:
    """3D position of a hand joint."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: 'JointPosition') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "z": self.z}


# WebXR hand joint indices (25 joints per hand)
class HandJoint(Enum):
    WRIST = 0
    THUMB_METACARPAL = 1
    THUMB_PHALANX_PROXIMAL = 2
    THUMB_PHALANX_DISTAL = 3
    THUMB_TIP = 4
    INDEX_METACARPAL = 5
    INDEX_PHALANX_PROXIMAL = 6
    INDEX_PHALANX_INTERMEDIATE = 7
    INDEX_PHALANX_DISTAL = 8
    INDEX_TIP = 9
    MIDDLE_METACARPAL = 10
    MIDDLE_PHALANX_PROXIMAL = 11
    MIDDLE_PHALANX_INTERMEDIATE = 12
    MIDDLE_PHALANX_DISTAL = 13
    MIDDLE_TIP = 14
    RING_METACARPAL = 15
    RING_PHALANX_PROXIMAL = 16
    RING_PHALANX_INTERMEDIATE = 17
    RING_PHALANX_DISTAL = 18
    RING_TIP = 19
    PINKY_METACARPAL = 20
    PINKY_PHALANX_PROXIMAL = 21
    PINKY_PHALANX_INTERMEDIATE = 22
    PINKY_PHALANX_DISTAL = 23
    PINKY_TIP = 24


@dataclass
class HandPose:
    """Complete hand pose with all joint positions."""
    side: HandSide
    joints: Dict[int, JointPosition] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0

    def get_joint(self, joint: HandJoint) -> Optional[JointPosition]:
        """Get position of a specific joint."""
        return self.joints.get(joint.value)

    @property
    def wrist(self) -> Optional[JointPosition]:
        return self.get_joint(HandJoint.WRIST)

    @property
    def index_tip(self) -> Optional[JointPosition]:
        return self.get_joint(HandJoint.INDEX_TIP)

    @property
    def thumb_tip(self) -> Optional[JointPosition]:
        return self.get_joint(HandJoint.THUMB_TIP)

    @property
    def middle_tip(self) -> Optional[JointPosition]:
        return self.get_joint(HandJoint.MIDDLE_TIP)

    def pinch_distance(self) -> float:
        """Distance between thumb tip and index tip."""
        thumb = self.thumb_tip
        index = self.index_tip
        if thumb and index:
            return thumb.distance_to(index)
        return float('inf')

    def is_finger_extended(self, finger: str) -> bool:
        """Check if a finger is extended (straight)."""
        # Simplified heuristic based on joint positions
        tip_joints = {
            "thumb": (HandJoint.THUMB_TIP, HandJoint.THUMB_METACARPAL),
            "index": (HandJoint.INDEX_TIP, HandJoint.INDEX_METACARPAL),
            "middle": (HandJoint.MIDDLE_TIP, HandJoint.MIDDLE_METACARPAL),
            "ring": (HandJoint.RING_TIP, HandJoint.RING_METACARPAL),
            "pinky": (HandJoint.PINKY_TIP, HandJoint.PINKY_METACARPAL)
        }

        if finger not in tip_joints:
            return False

        tip_joint, base_joint = tip_joints[finger]
        tip = self.get_joint(tip_joint)
        base = self.get_joint(base_joint)

        if not tip or not base:
            return False

        # Finger is extended if tip is far from base
        distance = tip.distance_to(base)
        return distance > 0.06  # ~6cm threshold

    def to_dict(self) -> Dict:
        return {
            "side": self.side.value,
            "joints": {str(k): v.to_dict() for k, v in self.joints.items()},
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence
        }


@dataclass
class HandGesture:
    """A recognized hand gesture."""
    gesture_type: GestureType
    state: GestureState
    side: HandSide
    position: JointPosition            # World position of gesture
    confidence: float = 1.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    velocity: Optional[JointPosition] = None  # For swipe gestures
    scale: float = 1.0                        # For scale gestures
    rotation: float = 0.0                     # For rotate gestures (radians)

    @property
    def duration_ms(self) -> float:
        """Duration of gesture in milliseconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict:
        return {
            "gesture_type": self.gesture_type.value,
            "state": self.state.value,
            "side": self.side.value,
            "position": self.position.to_dict(),
            "confidence": self.confidence,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "velocity": self.velocity.to_dict() if self.velocity else None,
            "scale": self.scale,
            "rotation": self.rotation
        }


@dataclass
class GestureBinding:
    """Maps a gesture to an action."""
    gesture_type: GestureType
    action_name: str
    handler: Callable[[HandGesture], Any]
    description: str = ""
    side: Optional[HandSide] = None  # None = both hands


class HandTracker:
    """
    Tracks hand poses and recognizes gestures for XR interaction.

    Features:
    - WebXR hand tracking integration
    - Gesture recognition (point, pinch, grab, swipes, etc.)
    - Gesture-to-action binding
    - Multi-hand tracking
    - Gesture history
    """

    # Thresholds for gesture recognition
    PINCH_THRESHOLD = 0.02       # 2cm for pinch
    GRAB_THRESHOLD = 0.04        # 4cm for grab
    SWIPE_VELOCITY = 0.5         # 50cm/s for swipe
    SWIPE_MIN_DISTANCE = 0.1     # 10cm minimum swipe

    def __init__(self):
        self.left_pose: Optional[HandPose] = None
        self.right_pose: Optional[HandPose] = None

        # Current gesture states
        self.active_gestures: Dict[str, HandGesture] = {}

        # Gesture bindings
        self.bindings: Dict[GestureType, List[GestureBinding]] = {}

        # Gesture history
        self.history: List[HandGesture] = []
        self.max_history = 100

        # Position history for motion detection
        self._left_positions: List[Tuple[datetime, JointPosition]] = []
        self._right_positions: List[Tuple[datetime, JointPosition]] = []
        self._max_position_history = 30

        # Callbacks
        self.on_gesture: Optional[Callable[[HandGesture], None]] = None

    # === Hand Pose Updates ===

    def update_hand(self, side: HandSide, joints: Dict[int, JointPosition], confidence: float = 1.0):
        """
        Update hand pose from WebXR tracking data.

        Args:
            side: Which hand
            joints: Dictionary mapping joint index to position
            confidence: Tracking confidence (0-1)
        """
        pose = HandPose(
            side=side,
            joints=joints,
            confidence=confidence
        )

        if side == HandSide.LEFT:
            self.left_pose = pose
            self._update_position_history(self._left_positions, pose)
        else:
            self.right_pose = pose
            self._update_position_history(self._right_positions, pose)

        # Detect gestures
        self._detect_gestures(pose)

    def _update_position_history(self, history: List, pose: HandPose):
        """Track position history for motion detection."""
        if pose.wrist:
            history.append((pose.timestamp, pose.wrist))
            if len(history) > self._max_position_history:
                history.pop(0)

    def update_from_webxr(self, hand_data: Dict):
        """
        Update from WebXR hand tracking frame.

        Expected format:
        {
            "left": {"joints": [[x,y,z], ...], "confidence": 0.95},
            "right": {"joints": [[x,y,z], ...], "confidence": 0.98}
        }
        """
        for side_name in ["left", "right"]:
            if side_name in hand_data:
                data = hand_data[side_name]
                side = HandSide.LEFT if side_name == "left" else HandSide.RIGHT

                joints = {}
                for i, pos in enumerate(data.get("joints", [])):
                    if len(pos) >= 3:
                        joints[i] = JointPosition(x=pos[0], y=pos[1], z=pos[2])

                self.update_hand(side, joints, data.get("confidence", 1.0))

    # === Gesture Detection ===

    def _detect_gestures(self, pose: HandPose):
        """Detect gestures from current pose."""
        detected = []

        # Static gestures
        if self._is_pointing(pose):
            detected.append(self._create_gesture(GestureType.POINT, pose))

        if self._is_pinching(pose):
            detected.append(self._create_gesture(GestureType.PINCH, pose))
        elif self._is_grabbing(pose):
            detected.append(self._create_gesture(GestureType.GRAB, pose))
        elif self._is_open(pose):
            detected.append(self._create_gesture(GestureType.OPEN, pose))

        if self._is_thumbs_up(pose):
            detected.append(self._create_gesture(GestureType.THUMBS_UP, pose))

        # Motion gestures (swipes)
        swipe = self._detect_swipe(pose)
        if swipe:
            detected.append(swipe)

        # Process detected gestures
        for gesture in detected:
            self._process_gesture(gesture)

    def _is_pointing(self, pose: HandPose) -> bool:
        """Check if pointing (index extended, others curled)."""
        index_ext = pose.is_finger_extended("index")
        middle_ext = pose.is_finger_extended("middle")
        ring_ext = pose.is_finger_extended("ring")
        pinky_ext = pose.is_finger_extended("pinky")

        return index_ext and not middle_ext and not ring_ext and not pinky_ext

    def _is_pinching(self, pose: HandPose) -> bool:
        """Check if pinching (thumb and index together)."""
        return pose.pinch_distance() < self.PINCH_THRESHOLD

    def _is_grabbing(self, pose: HandPose) -> bool:
        """Check if grabbing (all fingers curled)."""
        for finger in ["index", "middle", "ring", "pinky"]:
            if pose.is_finger_extended(finger):
                return False
        return True

    def _is_open(self, pose: HandPose) -> bool:
        """Check if hand is open (all fingers extended)."""
        for finger in ["index", "middle", "ring", "pinky"]:
            if not pose.is_finger_extended(finger):
                return False
        return True

    def _is_thumbs_up(self, pose: HandPose) -> bool:
        """Check for thumbs up gesture."""
        thumb_ext = pose.is_finger_extended("thumb")
        others_curled = not any(
            pose.is_finger_extended(f)
            for f in ["index", "middle", "ring", "pinky"]
        )
        return thumb_ext and others_curled

    def _detect_swipe(self, pose: HandPose) -> Optional[HandGesture]:
        """Detect swipe gestures from motion history."""
        history = self._left_positions if pose.side == HandSide.LEFT else self._right_positions

        if len(history) < 10:
            return None

        # Calculate velocity over recent frames
        recent = history[-10:]
        start_time, start_pos = recent[0]
        end_time, end_pos = recent[-1]

        dt = (end_time - start_time).total_seconds()
        if dt < 0.05:  # Too fast, likely noise
            return None

        dx = end_pos.x - start_pos.x
        dy = end_pos.y - start_pos.y
        dz = end_pos.z - start_pos.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        if distance < self.SWIPE_MIN_DISTANCE:
            return None

        velocity = distance / dt
        if velocity < self.SWIPE_VELOCITY:
            return None

        # Determine swipe direction
        abs_dx, abs_dy = abs(dx), abs(dy)

        if abs_dx > abs_dy:
            gesture_type = GestureType.SWIPE_RIGHT if dx > 0 else GestureType.SWIPE_LEFT
        else:
            gesture_type = GestureType.SWIPE_UP if dy > 0 else GestureType.SWIPE_DOWN

        gesture = self._create_gesture(gesture_type, pose)
        gesture.velocity = JointPosition(x=dx/dt, y=dy/dt, z=dz/dt)
        gesture.state = GestureState.COMPLETED

        return gesture

    def _create_gesture(self, gesture_type: GestureType, pose: HandPose) -> HandGesture:
        """Create a gesture from pose."""
        position = pose.wrist or JointPosition()

        return HandGesture(
            gesture_type=gesture_type,
            state=GestureState.STARTED,
            side=pose.side,
            position=position,
            confidence=pose.confidence
        )

    def _process_gesture(self, gesture: HandGesture):
        """Process a detected gesture."""
        key = f"{gesture.side.value}_{gesture.gesture_type.value}"

        # Check if gesture is ongoing
        if key in self.active_gestures:
            existing = self.active_gestures[key]
            existing.state = GestureState.ONGOING
            return

        # New gesture
        self.active_gestures[key] = gesture

        # Add to history
        self.history.append(gesture)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Fire callback
        if self.on_gesture:
            self.on_gesture(gesture)

        # Execute bindings
        self._execute_bindings(gesture)

    # === Gesture Bindings ===

    def bind_gesture(
        self,
        gesture_type: GestureType,
        action_name: str,
        handler: Callable[[HandGesture], Any],
        description: str = "",
        side: Optional[HandSide] = None
    ):
        """
        Bind a gesture to an action.

        Args:
            gesture_type: Type of gesture to bind
            action_name: Name of the action
            handler: Function to call when gesture detected
            description: Human-readable description
            side: Optional hand side (None = both)
        """
        binding = GestureBinding(
            gesture_type=gesture_type,
            action_name=action_name,
            handler=handler,
            description=description,
            side=side
        )

        if gesture_type not in self.bindings:
            self.bindings[gesture_type] = []
        self.bindings[gesture_type].append(binding)

    def unbind_gesture(self, gesture_type: GestureType, action_name: str):
        """Remove a gesture binding."""
        if gesture_type in self.bindings:
            self.bindings[gesture_type] = [
                b for b in self.bindings[gesture_type]
                if b.action_name != action_name
            ]

    def _execute_bindings(self, gesture: HandGesture):
        """Execute all bindings for a gesture."""
        if gesture.gesture_type not in self.bindings:
            return

        for binding in self.bindings[gesture.gesture_type]:
            # Check hand side filter
            if binding.side and binding.side != gesture.side:
                continue

            try:
                binding.handler(gesture)
            except Exception as e:
                print(f"Gesture handler error: {e}")

    # === Two-Hand Gestures ===

    def detect_two_hand_gestures(self) -> List[HandGesture]:
        """Detect gestures that use both hands."""
        if not self.left_pose or not self.right_pose:
            return []

        gestures = []

        # Scale gesture (hands moving apart/together)
        scale = self._detect_scale_gesture()
        if scale:
            gestures.append(scale)

        return gestures

    def _detect_scale_gesture(self) -> Optional[HandGesture]:
        """Detect pinch-to-scale with both hands."""
        left = self.left_pose
        right = self.right_pose

        if not left or not right:
            return None

        if not self._is_pinching(left) or not self._is_pinching(right):
            return None

        left_pos = left.thumb_tip
        right_pos = right.thumb_tip

        if not left_pos or not right_pos:
            return None

        # Check motion history for scaling
        if len(self._left_positions) < 5 or len(self._right_positions) < 5:
            return None

        # Calculate hand distance change
        current_distance = left_pos.distance_to(right_pos)

        old_left = self._left_positions[-5][1]
        old_right = self._right_positions[-5][1]
        old_distance = old_left.distance_to(old_right)

        if old_distance < 0.01:
            return None

        scale_ratio = current_distance / old_distance

        if abs(scale_ratio - 1.0) < 0.1:  # Less than 10% change
            return None

        gesture_type = GestureType.SCALE_UP if scale_ratio > 1.0 else GestureType.SCALE_DOWN

        center = JointPosition(
            x=(left_pos.x + right_pos.x) / 2,
            y=(left_pos.y + right_pos.y) / 2,
            z=(left_pos.z + right_pos.z) / 2
        )

        gesture = HandGesture(
            gesture_type=gesture_type,
            state=GestureState.ONGOING,
            side=HandSide.LEFT,  # Use left as primary
            position=center,
            scale=scale_ratio
        )

        return gesture

    # === Queries ===

    def get_active_gesture(self, side: HandSide) -> Optional[HandGesture]:
        """Get current active gesture for a hand."""
        for key, gesture in self.active_gestures.items():
            if key.startswith(side.value):
                return gesture
        return None

    def get_pointing_direction(self, side: HandSide) -> Optional[JointPosition]:
        """Get the direction the hand is pointing."""
        pose = self.left_pose if side == HandSide.LEFT else self.right_pose
        if not pose:
            return None

        wrist = pose.wrist
        index_tip = pose.index_tip

        if not wrist or not index_tip:
            return None

        # Direction vector
        dx = index_tip.x - wrist.x
        dy = index_tip.y - wrist.y
        dz = index_tip.z - wrist.z

        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length < 0.01:
            return None

        return JointPosition(x=dx/length, y=dy/length, z=dz/length)

    def get_grab_position(self, side: HandSide) -> Optional[JointPosition]:
        """Get the position between fingers (for grabbing objects)."""
        pose = self.left_pose if side == HandSide.LEFT else self.right_pose
        if not pose:
            return None

        thumb = pose.thumb_tip
        index = pose.index_tip

        if not thumb or not index:
            return None

        return JointPosition(
            x=(thumb.x + index.x) / 2,
            y=(thumb.y + index.y) / 2,
            z=(thumb.z + index.z) / 2
        )

    def get_gesture_history(self, gesture_type: Optional[GestureType] = None) -> List[HandGesture]:
        """Get gesture history, optionally filtered by type."""
        if gesture_type:
            return [g for g in self.history if g.gesture_type == gesture_type]
        return list(self.history)

    def get_bindings_summary(self) -> List[Dict]:
        """Get summary of all gesture bindings."""
        summary = []
        for gesture_type, bindings in self.bindings.items():
            for binding in bindings:
                summary.append({
                    "gesture": gesture_type.value,
                    "action": binding.action_name,
                    "description": binding.description,
                    "side": binding.side.value if binding.side else "both"
                })
        return summary

    def export_state(self) -> str:
        """Export current tracking state as JSON."""
        data = {
            "left_pose": self.left_pose.to_dict() if self.left_pose else None,
            "right_pose": self.right_pose.to_dict() if self.right_pose else None,
            "active_gestures": {k: g.to_dict() for k, g in self.active_gestures.items()},
            "recent_history": [g.to_dict() for g in self.history[-10:]]
        }
        return json.dumps(data, indent=2)


# Quick test
if __name__ == "__main__":
    print("Testing Hand Tracker...")

    tracker = HandTracker()

    # Bind some gestures
    def on_pinch(gesture):
        print(f"Pinch detected! Side: {gesture.side.value}")

    def on_point(gesture):
        print(f"Pointing! Side: {gesture.side.value}")

    tracker.bind_gesture(GestureType.PINCH, "select", on_pinch, "Select object")
    tracker.bind_gesture(GestureType.POINT, "highlight", on_point, "Highlight object")

    # Simulate hand data
    joints = {
        HandJoint.WRIST.value: JointPosition(0.0, 0.0, 0.0),
        HandJoint.THUMB_TIP.value: JointPosition(0.02, 0.05, 0.01),
        HandJoint.INDEX_TIP.value: JointPosition(0.015, 0.06, 0.01),  # Close to thumb = pinch
        HandJoint.MIDDLE_TIP.value: JointPosition(0.0, 0.03, 0.0),
        HandJoint.RING_TIP.value: JointPosition(-0.01, 0.03, 0.0),
        HandJoint.PINKY_TIP.value: JointPosition(-0.02, 0.02, 0.0),
        # Metacarpals for extension detection
        HandJoint.THUMB_METACARPAL.value: JointPosition(0.0, 0.01, 0.0),
        HandJoint.INDEX_METACARPAL.value: JointPosition(0.01, 0.0, 0.0),
        HandJoint.MIDDLE_METACARPAL.value: JointPosition(0.0, 0.0, 0.0),
        HandJoint.RING_METACARPAL.value: JointPosition(-0.01, 0.0, 0.0),
        HandJoint.PINKY_METACARPAL.value: JointPosition(-0.02, 0.0, 0.0),
    }

    tracker.update_hand(HandSide.RIGHT, joints)

    print(f"\nActive gestures: {list(tracker.active_gestures.keys())}")
    print(f"Gesture history: {len(tracker.history)}")
    print(f"\nBindings: {tracker.get_bindings_summary()}")
