"""
HoloLoom Vision Tools - Real-time computer vision for AR

Provides object detection, scene understanding, hand tracking, and spatial analysis
for AR applications.

Components:
    - ObjectDetector: Real-time object detection (COCO-SSD, YOLO)
    - SceneAnalyzer: Scene understanding and spatial relationships
    - HandTracker: Hand gesture recognition (MediaPipe)
    - DepthEstimator: Monocular depth estimation
    - MarkerDetector: QR codes, ArUco markers
    - LayoutOptimizer: Monte Carlo spatial layout optimization

Created: 2025-11-22 (Phase 2)
"""

from HoloLoom.vision.protocol import (
    VisionProcessor,
    DetectedObject,
    SceneUnderstanding,
    HandPose,
    DepthMap,
    Marker,
)

from HoloLoom.vision.object_detector import (
    ObjectDetector,
    create_object_detector,
)

from HoloLoom.vision.scene_analyzer import (
    SceneAnalyzer,
    SpatialRelationship,
    create_scene_analyzer,
)

from HoloLoom.vision.hand_tracker import (
    HandTracker,
    Gesture,
    create_hand_tracker,
)

__all__ = [
    # Protocols
    "VisionProcessor",
    "DetectedObject",
    "SceneUnderstanding",
    "HandPose",
    "DepthMap",
    "Marker",

    # Object Detection
    "ObjectDetector",
    "create_object_detector",

    # Scene Analysis
    "SceneAnalyzer",
    "SpatialRelationship",
    "create_scene_analyzer",

    # Hand Tracking
    "HandTracker",
    "Gesture",
    "create_hand_tracker",
]
