"""
Hand Tracker - Hand pose estimation and gesture recognition

Uses MediaPipe Hands for 21-landmark hand tracking and gesture classification.

Supported Gestures:
    - point (index finger extended)
    - grab/fist (all fingers closed)
    - open_palm (all fingers extended)
    - pinch (thumb and index touching)
    - thumbs_up
    - peace (index and middle extended)

Created: 2025-11-22
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from datetime import datetime

from HoloLoom.vision.protocol import (
    HandPose,
    HandLandmark,
    HandTrackerProtocol,
)


logger = logging.getLogger(__name__)


# ============================================================================
# MediaPipe Landmarks
# ============================================================================

# MediaPipe hand landmarks (21 points)
HAND_LANDMARKS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Finger tip indices
FINGERTIP_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}


# ============================================================================
# Gesture Definitions
# ============================================================================

class Gesture:
    """Gesture classification result"""

    POINT = "point"
    GRAB = "grab"
    OPEN_PALM = "open_palm"
    PINCH = "pinch"
    THUMBS_UP = "thumbs_up"
    PEACE = "peace"
    UNKNOWN = "unknown"


# ============================================================================
# Hand Tracker
# ============================================================================

class HandTracker(HandTrackerProtocol):
    """
    Hand pose estimator and gesture recognizer.

    Uses MediaPipe Hands (Python) or MediaPipe Hands (JavaScript for web).
    """

    def __init__(
        self,
        backend: str = "mediapipe",
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize hand tracker.

        Args:
            backend: "mediapipe" or "mock"
            max_hands: Maximum number of hands to track
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.backend = backend
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = None
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize hand tracker"""
        try:
            if self.backend == "mediapipe":
                return await self._initialize_mediapipe()
            elif self.backend == "mock":
                logger.info("Using mock hand tracker")
                self.initialized = True
                return True
            else:
                logger.warning(f"Unknown backend: {self.backend}, using mock")
                self.backend = "mock"
                self.initialized = True
                return True

        except Exception as e:
            logger.error(f"Failed to initialize {self.backend}: {e}")
            logger.warning("Falling back to mock hand tracker")
            self.backend = "mock"
            self.initialized = True
            return True

    async def _initialize_mediapipe(self) -> bool:
        """Initialize MediaPipe Hands"""
        try:
            import mediapipe as mp

            mp_hands = mp.solutions.hands
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

            self.initialized = True
            logger.info("MediaPipe Hands initialized")
            return True

        except ImportError:
            logger.warning("mediapipe not installed, install with: pip install mediapipe")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            return False

    async def process_frame(self, frame: np.ndarray) -> List[HandPose]:
        """Process frame (delegates to track_hands)"""
        return await self.track_hands(frame)

    async def track_hands(self, frame: np.ndarray) -> List[HandPose]:
        """
        Track hands in frame.

        Args:
            frame: RGB image (HxWx3)

        Returns:
            List of detected hands
        """
        if not self.initialized:
            await self.initialize()

        if self.backend == "mediapipe":
            return await self._track_mediapipe(frame)
        elif self.backend == "mock":
            return self._track_mock()
        else:
            return []

    async def _track_mediapipe(self, frame: np.ndarray) -> List[HandPose]:
        """Track hands with MediaPipe"""
        if not self.hands:
            return []

        # Process frame
        results = self.hands.process(frame)

        if not results.multi_hand_landmarks:
            return []

        # Convert to HandPose objects
        hand_poses = []

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine handedness (left/right)
            if results.multi_handedness:
                handedness = results.multi_handedness[hand_idx]
                hand_id = handedness.classification[0].label.lower()  # "left" or "right"
            else:
                hand_id = f"hand_{hand_idx}"

            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(
                    HandLandmark(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0,
                    )
                )

            # Create hand pose
            hand_pose = HandPose(
                hand_id=hand_id,
                landmarks=landmarks,
                confidence=0.9,  # MediaPipe doesn't provide per-hand confidence
            )

            # Recognize gesture
            hand_pose.gesture = self.recognize_gesture(hand_pose)

            hand_poses.append(hand_pose)

        return hand_poses

    def _track_mock(self) -> List[HandPose]:
        """Mock hand tracking for testing"""
        # Return dummy hand pose
        landmarks = [
            HandLandmark(x=0.5 + i*0.01, y=0.5 + i*0.01, z=0)
            for i in range(21)
        ]

        return [
            HandPose(
                hand_id="right",
                landmarks=landmarks,
                confidence=0.9,
                gesture=Gesture.OPEN_PALM,
            )
        ]

    def recognize_gesture(self, hand_pose: HandPose) -> Optional[str]:
        """
        Recognize gesture from hand landmarks.

        Args:
            hand_pose: Hand pose with 21 landmarks

        Returns:
            Gesture name or None
        """
        if len(hand_pose.landmarks) != 21:
            return None

        # Extract finger states (extended or not)
        finger_states = self._get_finger_states(hand_pose.landmarks)

        # Classify gesture based on finger states
        if self._is_point(finger_states):
            return Gesture.POINT

        elif self._is_grab(finger_states):
            return Gesture.GRAB

        elif self._is_open_palm(finger_states):
            return Gesture.OPEN_PALM

        elif self._is_pinch(hand_pose.landmarks):
            return Gesture.PINCH

        elif self._is_thumbs_up(hand_pose.landmarks, finger_states):
            return Gesture.THUMBS_UP

        elif self._is_peace(finger_states):
            return Gesture.PEACE

        return Gesture.UNKNOWN

    def _get_finger_states(self, landmarks: List[HandLandmark]) -> Dict[str, bool]:
        """
        Determine if each finger is extended.

        Returns dict: {finger_name: is_extended}
        """
        states = {}

        # Thumb (different logic - check horizontal distance from wrist)
        thumb_tip = landmarks[FINGERTIP_INDICES["thumb"]]
        thumb_mcp = landmarks[2]  # THUMB_MCP
        wrist = landmarks[0]

        thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x)
        states["thumb"] = thumb_extended

        # Other fingers (check if tip is above MCP)
        for finger, tip_idx in FINGERTIP_INDICES.items():
            if finger == "thumb":
                continue

            tip = landmarks[tip_idx]
            mcp = landmarks[tip_idx - 3]  # MCP is 3 indices before tip

            # Extended if tip is above MCP (lower y value)
            states[finger] = tip.y < mcp.y

        return states

    def _is_point(self, finger_states: Dict[str, bool]) -> bool:
        """Index finger extended, others closed"""
        return (
            finger_states["index"] and
            not finger_states["middle"] and
            not finger_states["ring"] and
            not finger_states["pinky"]
        )

    def _is_grab(self, finger_states: Dict[str, bool]) -> bool:
        """All fingers closed (fist)"""
        return not any([
            finger_states["thumb"],
            finger_states["index"],
            finger_states["middle"],
            finger_states["ring"],
            finger_states["pinky"],
        ])

    def _is_open_palm(self, finger_states: Dict[str, bool]) -> bool:
        """All fingers extended"""
        return all([
            finger_states["thumb"],
            finger_states["index"],
            finger_states["middle"],
            finger_states["ring"],
            finger_states["pinky"],
        ])

    def _is_pinch(self, landmarks: List[HandLandmark]) -> bool:
        """Thumb and index finger close together"""
        thumb_tip = landmarks[FINGERTIP_INDICES["thumb"]]
        index_tip = landmarks[FINGERTIP_INDICES["index"]]

        # Calculate distance
        distance = (
            (thumb_tip.x - index_tip.x)**2 +
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )**0.5

        return distance < 0.05  # Threshold for "touching"

    def _is_thumbs_up(
        self,
        landmarks: List[HandLandmark],
        finger_states: Dict[str, bool]
    ) -> bool:
        """Thumb extended upward, other fingers closed"""
        thumb_tip = landmarks[FINGERTIP_INDICES["thumb"]]
        wrist = landmarks[0]

        # Thumb pointing upward (tip above wrist)
        thumb_up = thumb_tip.y < wrist.y

        # Other fingers closed
        others_closed = not any([
            finger_states["index"],
            finger_states["middle"],
            finger_states["ring"],
            finger_states["pinky"],
        ])

        return thumb_up and others_closed

    def _is_peace(self, finger_states: Dict[str, bool]) -> bool:
        """Index and middle extended, others closed"""
        return (
            finger_states["index"] and
            finger_states["middle"] and
            not finger_states["ring"] and
            not finger_states["pinky"]
        )

    async def cleanup(self) -> None:
        """Clean up resources"""
        if self.hands:
            self.hands.close()
            self.hands = None
        self.initialized = False

    def get_capabilities(self) -> Dict[str, bool]:
        """Get tracker capabilities"""
        return {
            "hand_tracking": True,
            "gesture_recognition": True,
            "21_landmarks": self.backend == "mediapipe",
            "multi_hand": self.max_hands > 1,
        }


# ============================================================================
# Factory
# ============================================================================

def create_hand_tracker(backend: str = "mock", **kwargs) -> HandTracker:
    """
    Factory for creating hand trackers.

    Args:
        backend: "mediapipe" or "mock"
        **kwargs: Tracker options

    Returns:
        HandTracker instance
    """
    return HandTracker(backend=backend, **kwargs)


# ============================================================================
# Helper Functions
# ============================================================================

def get_pointing_direction(hand_pose: HandPose) -> Optional[np.ndarray]:
    """
    Get 3D pointing direction from hand pose.

    Args:
        hand_pose: Hand pose with point gesture

    Returns:
        Normalized direction vector [x, y, z] or None
    """
    if hand_pose.gesture != Gesture.POINT:
        return None

    if len(hand_pose.landmarks) != 21:
        return None

    # Direction from MCP to tip of index finger
    index_mcp = hand_pose.landmarks[5]
    index_tip = hand_pose.landmarks[8]

    direction = np.array([
        index_tip.x - index_mcp.x,
        index_tip.y - index_mcp.y,
        index_tip.z - index_mcp.z,
    ])

    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    return direction


def get_pinch_strength(hand_pose: HandPose) -> float:
    """
    Get pinch strength (0.0 = open, 1.0 = fully pinched).

    Args:
        hand_pose: Hand pose

    Returns:
        Pinch strength
    """
    if len(hand_pose.landmarks) != 21:
        return 0.0

    thumb_tip = hand_pose.landmarks[FINGERTIP_INDICES["thumb"]]
    index_tip = hand_pose.landmarks[FINGERTIP_INDICES["index"]]

    # Distance between tips
    distance = (
        (thumb_tip.x - index_tip.x)**2 +
        (thumb_tip.y - index_tip.y)**2 +
        (thumb_tip.z - index_tip.z)**2
    )**0.5

    # Normalize to 0-1 (assuming max distance ~0.2)
    strength = 1.0 - min(distance / 0.2, 1.0)

    return strength
