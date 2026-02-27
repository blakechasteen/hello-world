"""
Vision Protocol - Interfaces for computer vision components

Defines standard interfaces for vision processors following HoloLoom's
protocol-based architecture.

Created: 2025-11-22
"""

from typing import Protocol, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


# ============================================================================
# Core Data Models
# ============================================================================

@dataclass
class BoundingBox:
    """2D bounding box in image coordinates"""
    x_min: float  # 0.0 - 1.0 (normalized)
    y_min: float
    x_max: float
    y_max: float

    def center(self) -> Tuple[float, float]:
        """Get center point"""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def area(self) -> float:
        """Get area"""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def iou(self, other: "BoundingBox") -> float:
        """Calculate intersection over union with another box"""
        x_left = max(self.x_min, other.x_min)
        y_top = max(self.y_min, other.y_min)
        x_right = min(self.x_max, other.x_max)
        y_bottom = min(self.y_max, other.y_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area() + other.area() - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class DetectedObject:
    """Object detected in image/video frame"""
    id: str
    label: str
    confidence: float
    bbox: BoundingBox
    class_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 3D information (if available)
    position_3d: Optional[Tuple[float, float, float]] = None
    depth: Optional[float] = None


@dataclass
class SceneUnderstanding:
    """Complete scene analysis"""
    objects: List[DetectedObject]
    dominant_colors: List[str] = field(default_factory=list)
    scene_type: str = "unknown"  # indoor, outdoor, workshop, etc.
    lighting_condition: str = "normal"  # bright, dim, backlit, etc.
    spatial_layout: Dict[str, Any] = field(default_factory=dict)
    relationships: List["SpatialRelationship"] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SpatialRelationship:
    """Spatial relationship between objects"""
    subject_id: str
    relation: str  # "left_of", "right_of", "above", "below", "near", "far", "contains"
    object_id: str
    confidence: float = 1.0


@dataclass
class HandLandmark:
    """3D hand landmark point"""
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class HandPose:
    """Complete hand pose with landmarks"""
    hand_id: str  # "left" or "right"
    landmarks: List[HandLandmark]  # 21 landmarks (MediaPipe)
    confidence: float
    gesture: Optional[str] = None  # Recognized gesture
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DepthMap:
    """Monocular depth estimation result"""
    depth: np.ndarray  # HxW depth map
    width: int
    height: int
    min_depth: float
    max_depth: float
    unit: str = "normalized"  # normalized, meters, disparity
    confidence_map: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Marker:
    """Detected marker (QR code, ArUco, AprilTag)"""
    id: str  # Marker ID or decoded data
    marker_type: str  # "aruco", "qr_code", "apriltag"
    corners: List[List[float]]  # 4 corner points [[x, y], ...]
    center: Tuple[float, float]  # Center point (x, y)
    position: Optional[List[float]] = None  # 3D position [x, y, z] in meters
    rotation: Optional[List[float]] = None  # Rotation vector [rx, ry, rz]
    data: Optional[str] = None  # Decoded data (for QR codes)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SegmentationMask:
    """Semantic segmentation result (Phase 5)"""
    mask: np.ndarray  # HxW segmentation mask (class IDs per pixel)
    class_names: List[str]  # Class names indexed by ID
    num_classes: int
    width: int
    height: int
    confidence_map: Optional[np.ndarray] = None  # HxW confidence per pixel
    timestamp: datetime = field(default_factory=datetime.now)

    def get_class_mask(self, class_id: int) -> np.ndarray:
        """Get binary mask for specific class"""
        return (self.mask == class_id).astype(np.uint8)

    def get_class_pixels(self, class_id: int) -> int:
        """Count pixels of specific class"""
        return np.sum(self.mask == class_id)


@dataclass
class Keypoint:
    """Single body keypoint (Phase 5)"""
    x: float  # 0.0-1.0 normalized
    y: float
    z: float  # Depth (if available)
    visibility: float  # 0.0-1.0
    presence: float = 1.0  # Whether keypoint is present in frame


@dataclass
class BodyPose:
    """Full-body pose with 33 keypoints (MediaPipe Pose) (Phase 5)"""
    keypoints: List[Keypoint]  # 33 keypoints
    world_keypoints: Optional[List[Keypoint]] = None  # 3D world coordinates
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SLAMPose:
    """SLAM camera pose estimate (Phase 5)"""
    position: Tuple[float, float, float]  # (x, y, z) in meters
    orientation: Tuple[float, float, float, float]  # Quaternion (x, y, z, w)
    covariance: Optional[np.ndarray] = None  # 6x6 pose covariance
    tracking_quality: float = 0.0  # 0.0-1.0
    num_features: int = 0  # Number of tracked features
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MapPoint:
    """3D map point from SLAM (Phase 5)"""
    id: int
    position: Tuple[float, float, float]  # (x, y, z) in meters
    descriptor: Optional[np.ndarray] = None
    num_observations: int = 0
    confidence: float = 1.0


# ============================================================================
# Vision Processor Protocol
# ============================================================================

class VisionProcessor(Protocol):
    """
    Base protocol for vision processing components.

    All vision processors follow this interface for consistency.
    """

    async def initialize(self) -> bool:
        """
        Initialize vision processor.

        Returns:
            True if initialized successfully
        """
        ...

    async def process_frame(self, frame: np.ndarray) -> Any:
        """
        Process a single video frame.

        Args:
            frame: RGB image as numpy array (HxWx3)

        Returns:
            Processing result (type depends on processor)
        """
        ...

    async def cleanup(self) -> None:
        """Clean up resources"""
        ...

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get processor capabilities.

        Returns:
            Dict of capability flags
        """
        ...


# ============================================================================
# Specific Processor Protocols
# ============================================================================

class ObjectDetectorProtocol(VisionProcessor):
    """Protocol for object detection"""

    async def detect_objects(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[DetectedObject]:
        """
        Detect objects in frame.

        Args:
            frame: RGB image
            confidence_threshold: Minimum confidence (0.0-1.0)

        Returns:
            List of detected objects
        """
        ...


class SceneAnalyzerProtocol(VisionProcessor):
    """Protocol for scene understanding"""

    async def analyze_scene(
        self,
        frame: np.ndarray,
        objects: Optional[List[DetectedObject]] = None
    ) -> SceneUnderstanding:
        """
        Analyze scene for spatial understanding.

        Args:
            frame: RGB image
            objects: Pre-detected objects (optional)

        Returns:
            Scene understanding result
        """
        ...


class HandTrackerProtocol(VisionProcessor):
    """Protocol for hand tracking"""

    async def track_hands(
        self,
        frame: np.ndarray
    ) -> List[HandPose]:
        """
        Track hands in frame.

        Args:
            frame: RGB image

        Returns:
            List of detected hands with poses
        """
        ...

    def recognize_gesture(self, hand_pose: HandPose) -> Optional[str]:
        """
        Recognize gesture from hand pose.

        Args:
            hand_pose: Hand pose with landmarks

        Returns:
            Gesture name or None
        """
        ...


class DepthEstimatorProtocol(VisionProcessor):
    """Protocol for depth estimation"""

    async def estimate_depth(
        self,
        frame: np.ndarray
    ) -> DepthMap:
        """
        Estimate depth map from single image.

        Args:
            frame: RGB image

        Returns:
            Depth map
        """
        ...


class MarkerDetectorProtocol(VisionProcessor):
    """Protocol for marker detection"""

    async def detect_markers(
        self,
        frame: np.ndarray,
        marker_types: Optional[List[str]] = None
    ) -> List[Marker]:
        """
        Detect markers in frame.

        Args:
            frame: RGB image
            marker_types: Types to detect (None = all)

        Returns:
            List of detected markers
        """
        ...


# ============================================================================
# Vision Pipeline
# ============================================================================

@dataclass
class VisionPipelineResult:
    """Complete vision pipeline result"""
    objects: List[DetectedObject]
    scene: SceneUnderstanding
    hands: List[HandPose]
    depth: Optional[DepthMap] = None
    markers: List[Marker] = field(default_factory=list)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class VisionPipeline:
    """
    Complete vision processing pipeline.

    Coordinates multiple vision processors for comprehensive scene understanding.
    """

    def __init__(
        self,
        object_detector: Optional[ObjectDetectorProtocol] = None,
        scene_analyzer: Optional[SceneAnalyzerProtocol] = None,
        hand_tracker: Optional[HandTrackerProtocol] = None,
        depth_estimator: Optional[DepthEstimatorProtocol] = None,
        marker_detector: Optional[MarkerDetectorProtocol] = None,
    ):
        self.object_detector = object_detector
        self.scene_analyzer = scene_analyzer
        self.hand_tracker = hand_tracker
        self.depth_estimator = depth_estimator
        self.marker_detector = marker_detector

    async def process(
        self,
        frame: np.ndarray,
        enable_objects: bool = True,
        enable_scene: bool = True,
        enable_hands: bool = True,
        enable_depth: bool = False,
        enable_markers: bool = False,
    ) -> VisionPipelineResult:
        """
        Process frame through complete pipeline.

        Args:
            frame: RGB image
            enable_*: Flags to enable/disable specific processors

        Returns:
            Complete vision result
        """
        import time
        start_time = time.time()

        result = VisionPipelineResult(
            objects=[],
            scene=SceneUnderstanding(objects=[]),
            hands=[],
        )

        # Object detection
        if enable_objects and self.object_detector:
            result.objects = await self.object_detector.detect_objects(frame)

        # Scene analysis
        if enable_scene and self.scene_analyzer:
            result.scene = await self.scene_analyzer.analyze_scene(
                frame,
                objects=result.objects if result.objects else None
            )

        # Hand tracking
        if enable_hands and self.hand_tracker:
            result.hands = await self.hand_tracker.track_hands(frame)

        # Depth estimation
        if enable_depth and self.depth_estimator:
            result.depth = await self.depth_estimator.estimate_depth(frame)

        # Marker detection
        if enable_markers and self.marker_detector:
            result.markers = await self.marker_detector.detect_markers(frame)

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result
