"""
Gesture Recognition Engine
===========================
Real-time hand gesture recognition using MediaPipe Hands.

This module provides comprehensive hand gesture detection and classification,
supporting 10 gesture types with both MediaPipe-based and simplified fallback modes.

Author: HoloLoom Team (Agent M)
Date: November 17, 2025
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import time
import math
import numpy as np
from collections import deque


# ============================================================================
# Gesture Types
# ============================================================================

class GestureType(Enum):
    """Recognized gesture types."""
    POINT = "point"              # Index finger pointing
    GRAB = "grab"                # Closed fist
    OPEN_PALM = "open_palm"      # All fingers extended
    SWIPE_LEFT = "swipe_left"    # Horizontal swipe left
    SWIPE_RIGHT = "swipe_right"  # Horizontal swipe right
    SWIPE_UP = "swipe_up"        # Vertical swipe up
    SWIPE_DOWN = "swipe_down"    # Vertical swipe down
    PINCH = "pinch"              # Thumb and index finger together
    CIRCLE = "circle"            # Drawing a circle in the air
    WAVE = "wave"                # Waving hand
    UNKNOWN = "unknown"          # Unrecognized gesture


# ============================================================================
# Hand Landmark Data
# ============================================================================

@dataclass
class HandLandmark:
    """Single hand landmark (joint position)."""
    x: float  # -1 to 1 (left to right in normalized coordinates)
    y: float  # -1 to 1 (top to bottom in normalized coordinates)
    z: float  # Depth (relative to wrist)
    visibility: float = 1.0  # 0 to 1 (confidence)

    def distance_to(self, other: 'HandLandmark') -> float:
        """Calculate Euclidean distance to another landmark."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


@dataclass
class Hand:
    """Detected hand with landmarks."""
    handedness: str  # "Left" or "Right"
    landmarks: List[HandLandmark]  # 21 landmarks (MediaPipe standard)
    confidence: float

    # Landmark indices (MediaPipe standard)
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def get_finger_extended(self, finger: str) -> bool:
        """Check if a finger is extended."""
        finger_tips = {
            "thumb": self.THUMB_TIP,
            "index": self.INDEX_FINGER_TIP,
            "middle": self.MIDDLE_FINGER_TIP,
            "ring": self.RING_FINGER_TIP,
            "pinky": self.PINKY_TIP
        }

        finger_bases = {
            "thumb": self.THUMB_IP,
            "index": self.INDEX_FINGER_MCP,
            "middle": self.MIDDLE_FINGER_MCP,
            "ring": self.RING_FINGER_MCP,
            "pinky": self.PINKY_MCP
        }

        if finger not in finger_tips:
            return False

        tip = self.landmarks[finger_tips[finger]]
        base = self.landmarks[finger_bases[finger]]

        # For most fingers, extended = tip.y < base.y (tip higher than base)
        # For thumb, check x-axis as well
        if finger == "thumb":
            # Thumb extension depends on handedness
            if self.handedness == "Right":
                return tip.x < base.x  # Thumb extends to the left
            else:
                return tip.x > base.x  # Thumb extends to the right
        else:
            # Other fingers extend upward
            return tip.y < base.y

    def get_palm_center(self) -> HandLandmark:
        """Calculate approximate palm center."""
        # Average of wrist and base of middle finger
        wrist = self.landmarks[self.WRIST]
        middle_base = self.landmarks[self.MIDDLE_FINGER_MCP]

        return HandLandmark(
            x=(wrist.x + middle_base.x) / 2,
            y=(wrist.y + middle_base.y) / 2,
            z=(wrist.z + middle_base.z) / 2
        )

    def get_pointing_direction(self) -> Tuple[float, float, float]:
        """Get pointing direction vector (for index finger)."""
        tip = self.landmarks[self.INDEX_FINGER_TIP]
        base = self.landmarks[self.INDEX_FINGER_MCP]

        # Direction from base to tip
        dx = tip.x - base.x
        dy = tip.y - base.y
        dz = tip.z - base.z

        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length == 0:
            return (0, 0, 1)

        return (dx / length, dy / length, dz / length)


@dataclass
class Gesture:
    """Recognized gesture."""
    type: GestureType
    hand: Hand
    confidence: float
    direction: Optional[Tuple[float, float, float]] = None  # For directional gestures
    target_position: Optional[Tuple[float, float, float]] = None  # For pointing gestures
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Gesture({self.type.value}, "
            f"hand={self.hand.handedness}, "
            f"conf={self.confidence:.2f})"
        )


# ============================================================================
# Gesture Recognizer
# ============================================================================

class GestureRecognizer:
    """Real-time hand gesture recognition."""

    def __init__(
        self,
        use_mediapipe: bool = True,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize gesture recognizer.

        Args:
            use_mediapipe: Use MediaPipe Hands if available
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.use_mediapipe = use_mediapipe
        self.mp_hands = None
        self.hands = None

        # Motion history for swipe/wave detection
        self.motion_history: Dict[str, deque] = {}  # handedness -> deque of positions
        self.motion_history_length = 30  # Keep last 30 frames (~1 second at 30fps)

        if use_mediapipe:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                print("✓ MediaPipe Hands initialized")
            except ImportError:
                print("⚠ MediaPipe not available, using simplified gesture recognition")
                self.use_mediapipe = False

    async def recognize(self, frame: np.ndarray) -> List[Gesture]:
        """
        Recognize gestures from camera frame.

        Args:
            frame: RGB image (H x W x 3)

        Returns:
            List of recognized gestures
        """
        if self.use_mediapipe and self.hands is not None:
            return await self._recognize_mediapipe(frame)
        else:
            return await self._recognize_simplified(frame)

    async def _recognize_mediapipe(self, frame: np.ndarray) -> List[Gesture]:
        """Recognize gestures using MediaPipe Hands."""
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume OpenCV BGR format, convert to RGB
            rgb_frame = frame[:, :, ::-1]
        else:
            rgb_frame = frame

        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return []

        gestures = []

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # Convert to Hand object
            hand = self._landmarks_to_hand(hand_landmarks, handedness)

            # Update motion history
            self._update_motion_history(hand)

            # Classify gesture
            gesture_type, confidence = self._classify_gesture(hand)

            if gesture_type != GestureType.UNKNOWN:
                # Calculate direction for directional gestures
                direction = self._calculate_direction(hand, gesture_type)

                # Calculate target for pointing gestures
                target = self._calculate_target(hand, gesture_type)

                gesture = Gesture(
                    type=gesture_type,
                    hand=hand,
                    confidence=confidence,
                    direction=direction,
                    target_position=target,
                    timestamp=time.time()
                )
                gestures.append(gesture)

        return gestures

    async def _recognize_simplified(self, frame: np.ndarray) -> List[Gesture]:
        """Simplified gesture recognition without MediaPipe (returns empty)."""
        # Without MediaPipe, we can't detect hands
        # This is a graceful fallback
        return []

    def _landmarks_to_hand(self, hand_landmarks, handedness) -> Hand:
        """Convert MediaPipe landmarks to Hand object."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append(HandLandmark(
                x=landmark.x * 2 - 1,  # Convert 0-1 to -1 to 1
                y=landmark.y * 2 - 1,
                z=landmark.z,
                visibility=landmark.visibility if hasattr(landmark, 'visibility') else 1.0
            ))

        return Hand(
            handedness=handedness.classification[0].label,
            landmarks=landmarks,
            confidence=handedness.classification[0].score
        )

    def _update_motion_history(self, hand: Hand) -> None:
        """Update motion history for swipe/wave detection."""
        handedness = hand.handedness

        if handedness not in self.motion_history:
            self.motion_history[handedness] = deque(maxlen=self.motion_history_length)

        # Store palm center position
        palm = hand.get_palm_center()
        self.motion_history[handedness].append((palm.x, palm.y, time.time()))

    def _classify_gesture(self, hand: Hand) -> Tuple[GestureType, float]:
        """Classify gesture from hand landmarks."""
        # Check static gestures first (faster)

        # Check for pointing (index finger extended, others closed)
        if self._is_pointing(hand):
            return GestureType.POINT, 0.9

        # Check for grab (all fingers closed)
        if self._is_grabbing(hand):
            return GestureType.GRAB, 0.85

        # Check for open palm (all fingers extended)
        if self._is_open_palm(hand):
            return GestureType.OPEN_PALM, 0.9

        # Check for pinch (thumb and index close)
        if self._is_pinching(hand):
            return GestureType.PINCH, 0.85

        # Check dynamic gestures (require motion history)

        # Check for swipes (directional motion)
        swipe_type = self._detect_swipe(hand)
        if swipe_type:
            return swipe_type, 0.8

        # Check for wave (oscillating motion)
        if self._is_waving(hand):
            return GestureType.WAVE, 0.75

        # Check for circle (circular motion)
        if self._is_circle(hand):
            return GestureType.CIRCLE, 0.7

        return GestureType.UNKNOWN, 0.0

    def _is_pointing(self, hand: Hand) -> bool:
        """Check if hand is in pointing gesture."""
        # Index finger extended, other fingers curled
        index_extended = hand.get_finger_extended("index")
        middle_curled = not hand.get_finger_extended("middle")
        ring_curled = not hand.get_finger_extended("ring")
        pinky_curled = not hand.get_finger_extended("pinky")

        return index_extended and middle_curled and ring_curled and pinky_curled

    def _is_grabbing(self, hand: Hand) -> bool:
        """Check if hand is in grabbing gesture (closed fist)."""
        # All fingers curled
        all_curled = all([
            not hand.get_finger_extended("index"),
            not hand.get_finger_extended("middle"),
            not hand.get_finger_extended("ring"),
            not hand.get_finger_extended("pinky")
        ])

        return all_curled

    def _is_open_palm(self, hand: Hand) -> bool:
        """Check if hand is in open palm gesture."""
        # All fingers extended
        all_extended = all([
            hand.get_finger_extended("index"),
            hand.get_finger_extended("middle"),
            hand.get_finger_extended("ring"),
            hand.get_finger_extended("pinky")
        ])

        return all_extended

    def _is_pinching(self, hand: Hand) -> bool:
        """Check if hand is in pinching gesture."""
        # Thumb and index finger tips close together
        thumb_tip = hand.landmarks[hand.THUMB_TIP]
        index_tip = hand.landmarks[hand.INDEX_FINGER_TIP]

        distance = thumb_tip.distance_to(index_tip)

        # Other fingers should be extended or neutral
        middle_extended = hand.get_finger_extended("middle")

        return distance < 0.05 and middle_extended

    def _detect_swipe(self, hand: Hand) -> Optional[GestureType]:
        """Detect swipe gestures from motion history."""
        handedness = hand.handedness

        if handedness not in self.motion_history:
            return None

        history = list(self.motion_history[handedness])

        if len(history) < 10:  # Need at least 10 frames
            return None

        # Get first and last positions
        first_x, first_y, first_t = history[0]
        last_x, last_y, last_t = history[-1]

        # Calculate displacement
        dx = last_x - first_x
        dy = last_y - first_y
        dt = last_t - first_t

        # Check if motion is fast enough
        if dt == 0 or dt > 0.5:  # Must complete within 0.5 seconds
            return None

        # Calculate velocity
        vx = dx / dt
        vy = dy / dt
        speed = math.sqrt(vx*vx + vy*vy)

        if speed < 1.0:  # Threshold for swipe speed
            return None

        # Determine direction
        if abs(dx) > abs(dy):
            # Horizontal swipe
            if dx > 0.2:
                return GestureType.SWIPE_RIGHT
            elif dx < -0.2:
                return GestureType.SWIPE_LEFT
        else:
            # Vertical swipe
            if dy > 0.2:
                return GestureType.SWIPE_DOWN
            elif dy < -0.2:
                return GestureType.SWIPE_UP

        return None

    def _is_waving(self, hand: Hand) -> bool:
        """Check if hand is waving (oscillating motion)."""
        handedness = hand.handedness

        if handedness not in self.motion_history:
            return False

        history = list(self.motion_history[handedness])

        if len(history) < 20:  # Need at least 20 frames (~0.66s)
            return False

        # Extract x positions
        x_positions = [pos[0] for pos in history]

        # Count direction changes
        direction_changes = 0
        for i in range(1, len(x_positions) - 1):
            # Check if direction changed
            if (x_positions[i] - x_positions[i-1]) * (x_positions[i+1] - x_positions[i]) < 0:
                direction_changes += 1

        # Wave should have at least 2 direction changes (back and forth)
        return direction_changes >= 2

    def _is_circle(self, hand: Hand) -> bool:
        """Check if hand drew a circle."""
        handedness = hand.handedness

        if handedness not in self.motion_history:
            return False

        history = list(self.motion_history[handedness])

        if len(history) < 20:
            return False

        # Check if path forms a closed loop
        first_x, first_y, _ = history[0]
        last_x, last_y, _ = history[-1]

        # Must return close to starting position
        closing_dist = math.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)

        if closing_dist > 0.1:
            return False

        # Calculate total path length
        total_length = 0
        for i in range(1, len(history)):
            x1, y1, _ = history[i-1]
            x2, y2, _ = history[i]
            total_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Circle should have significant length (not just noise)
        return total_length > 0.5

    def _calculate_direction(
        self,
        hand: Hand,
        gesture_type: GestureType
    ) -> Optional[Tuple[float, float, float]]:
        """Calculate direction vector for directional gestures."""
        if gesture_type == GestureType.POINT:
            return hand.get_pointing_direction()

        elif gesture_type in (
            GestureType.SWIPE_LEFT,
            GestureType.SWIPE_RIGHT,
            GestureType.SWIPE_UP,
            GestureType.SWIPE_DOWN
        ):
            # Get motion direction from history
            handedness = hand.handedness
            if handedness not in self.motion_history:
                return None

            history = list(self.motion_history[handedness])
            if len(history) < 2:
                return None

            first_x, first_y, _ = history[0]
            last_x, last_y, _ = history[-1]

            dx = last_x - first_x
            dy = last_y - first_y

            # Normalize
            length = math.sqrt(dx*dx + dy*dy)
            if length == 0:
                return None

            return (dx / length, dy / length, 0.0)

        return None

    def _calculate_target(
        self,
        hand: Hand,
        gesture_type: GestureType
    ) -> Optional[Tuple[float, float, float]]:
        """Calculate target position for pointing gestures."""
        if gesture_type == GestureType.POINT:
            # Project index finger direction into 3D space
            # For now, use tip position (could raycast later)
            tip = hand.landmarks[hand.INDEX_FINGER_TIP]
            return (tip.x, tip.y, tip.z)

        return None

    def close(self) -> None:
        """Release resources."""
        if self.hands is not None:
            self.hands.close()


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_hand(gesture_type: GestureType) -> Hand:
    """Create a test hand for a specific gesture type."""
    # Create 21 landmarks (MediaPipe standard)
    landmarks = [HandLandmark(0, 0, 0) for _ in range(21)]

    # Set wrist
    landmarks[0] = HandLandmark(0, 0.5, 0)

    if gesture_type == GestureType.POINT:
        # Index extended, others curled
        landmarks[8] = HandLandmark(0.1, -0.3, 0)  # Index tip (high)
        landmarks[5] = HandLandmark(0.05, 0.2, 0)  # Index base
        landmarks[12] = HandLandmark(0, 0.3, 0)    # Middle tip (low = curled)
        landmarks[16] = HandLandmark(-0.05, 0.3, 0)  # Ring tip (low = curled)
        landmarks[20] = HandLandmark(-0.1, 0.3, 0)   # Pinky tip (low = curled)
        landmarks[9] = HandLandmark(0, 0.2, 0)     # Middle base
        landmarks[13] = HandLandmark(-0.05, 0.2, 0)  # Ring base
        landmarks[17] = HandLandmark(-0.1, 0.2, 0)   # Pinky base

    elif gesture_type == GestureType.OPEN_PALM:
        # All fingers extended
        landmarks[8] = HandLandmark(0.1, -0.3, 0)   # Index tip
        landmarks[12] = HandLandmark(0, -0.35, 0)   # Middle tip
        landmarks[16] = HandLandmark(-0.1, -0.3, 0) # Ring tip
        landmarks[20] = HandLandmark(-0.15, -0.25, 0) # Pinky tip
        landmarks[5] = HandLandmark(0.1, 0.2, 0)    # Index base
        landmarks[9] = HandLandmark(0, 0.2, 0)      # Middle base
        landmarks[13] = HandLandmark(-0.1, 0.2, 0)  # Ring base
        landmarks[17] = HandLandmark(-0.15, 0.2, 0) # Pinky base

    elif gesture_type == GestureType.GRAB:
        # All fingers curled
        landmarks[8] = HandLandmark(0.1, 0.3, 0)   # Index tip (low)
        landmarks[12] = HandLandmark(0, 0.3, 0)    # Middle tip (low)
        landmarks[16] = HandLandmark(-0.1, 0.3, 0) # Ring tip (low)
        landmarks[20] = HandLandmark(-0.15, 0.3, 0) # Pinky tip (low)
        landmarks[5] = HandLandmark(0.1, 0.2, 0)   # Index base
        landmarks[9] = HandLandmark(0, 0.2, 0)     # Middle base
        landmarks[13] = HandLandmark(-0.1, 0.2, 0) # Ring base
        landmarks[17] = HandLandmark(-0.15, 0.2, 0) # Pinky base

    elif gesture_type == GestureType.PINCH:
        # Thumb and index close, others extended
        landmarks[4] = HandLandmark(0.05, -0.2, 0)  # Thumb tip
        landmarks[8] = HandLandmark(0.06, -0.2, 0)  # Index tip (close to thumb)
        landmarks[12] = HandLandmark(0, -0.3, 0)    # Middle tip (extended)
        landmarks[16] = HandLandmark(-0.1, -0.3, 0) # Ring tip (extended)
        landmarks[20] = HandLandmark(-0.15, -0.25, 0) # Pinky tip (extended)
        landmarks[3] = HandLandmark(0.05, 0.1, 0)   # Thumb IP
        landmarks[5] = HandLandmark(0.1, 0.2, 0)    # Index base
        landmarks[9] = HandLandmark(0, 0.2, 0)      # Middle base
        landmarks[13] = HandLandmark(-0.1, 0.2, 0)  # Ring base
        landmarks[17] = HandLandmark(-0.15, 0.2, 0) # Pinky base

    return Hand(
        handedness="Right",
        landmarks=landmarks,
        confidence=0.95
    )


if __name__ == "__main__":
    # Example usage
    print("Gesture Recognition Engine Demo")
    print("=" * 50)

    # Create test hands
    test_gestures = [
        GestureType.POINT,
        GestureType.OPEN_PALM,
        GestureType.GRAB,
        GestureType.PINCH
    ]

    for gesture_type in test_gestures:
        hand = create_test_hand(gesture_type)
        print(f"\nTest Hand: {gesture_type.value}")
        print(f"  Handedness: {hand.handedness}")
        print(f"  Confidence: {hand.confidence:.2f}")
        print(f"  Index extended: {hand.get_finger_extended('index')}")
        print(f"  Middle extended: {hand.get_finger_extended('middle')}")
        print(f"  Palm center: {hand.get_palm_center()}")

        if gesture_type == GestureType.POINT:
            direction = hand.get_pointing_direction()
            print(f"  Pointing direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})")
