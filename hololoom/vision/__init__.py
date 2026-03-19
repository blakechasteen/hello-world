"""
HoloLoom Vision Tools - Real-time computer vision for AR

Provides comprehensive vision capabilities for AR applications including object detection,
scene understanding, hand tracking, depth estimation, marker detection, semantic segmentation,
pose estimation, and SLAM.

Components:
    Phase 2 (Core Vision):
    - ObjectDetector: Real-time object detection (COCO-SSD, YOLO)
    - SceneAnalyzer: Scene understanding and spatial relationships
    - HandTracker: Hand gesture recognition (MediaPipe)

    Phase 4 (Depth & Markers):
    - DepthEstimator: Monocular depth estimation (MiDaS, ZoeDepth)
    - MarkerDetector: QR codes, ArUco markers, AprilTags

    Phase 5 (Advanced Vision):
    - SemanticSegmenter: Pixel-level semantic segmentation
    - PoseEstimator: Full-body pose estimation (MediaPipe Pose)
    - SLAMProcessor: Camera tracking and 3D mapping

Created: 2025-11-22 (Phase 2)
Updated: 2025-11-22 (Phase 4-5)
"""

# Phase 2: Core protocols and tools
# Phase 4: Depth estimation and marker detection
from hololoom.vision.depth_estimator import (
    DepthEstimator,
    create_depth_estimator,
    create_point_cloud,
    depth_to_3d_point,
    get_depth_at_point,
)
from hololoom.vision.hand_tracker import (
    Gesture,
    HandTracker,
    create_hand_tracker,
)
from hololoom.vision.marker_detector import (
    MarkerDetector,
    MarkerType,
    create_default_camera_matrix,
    create_marker_detector,
    marker_to_transform_matrix,
)
from hololoom.vision.object_detector import (
    ObjectDetector,
    create_object_detector,
)
from hololoom.vision.pose_estimator import (
    PoseEstimator,
    create_pose_estimator,
    detect_gesture,
    draw_pose_skeleton,
    get_body_orientation,
    get_joint_angle,
)
from hololoom.vision.protocol import (
    BodyPose,
    # Phase 4 protocols
    DepthMap,
    DetectedObject,
    HandPose,
    Keypoint,
    MapPoint,
    Marker,
    SceneUnderstanding,
    # Phase 5 protocols
    SegmentationMask,
    SLAMPose,
    VisionProcessor,
)
from hololoom.vision.scene_analyzer import (
    SceneAnalyzer,
    SpatialRelationship,
    create_scene_analyzer,
)

# Phase 5: Semantic segmentation, pose estimation, SLAM
from hololoom.vision.semantic_segmenter import (
    SegmentationDataset,
    SegmentationModel,
    SemanticSegmenter,
    create_semantic_segmenter,
    get_class_distribution,
    visualize_segmentation,
)
from hololoom.vision.slam_processor import (
    SLAMProcessor,
    create_camera_matrix,
    create_slam_processor,
    visualize_slam_tracking,
)

__all__ = [
    # ========== Protocols ==========
    "VisionProcessor",

    # Phase 2 protocols
    "DetectedObject",
    "SceneUnderstanding",
    "HandPose",

    # Phase 4 protocols
    "DepthMap",
    "Marker",

    # Phase 5 protocols
    "SegmentationMask",
    "Keypoint",
    "BodyPose",
    "SLAMPose",
    "MapPoint",

    # ========== Phase 2: Core Vision ==========

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

    # ========== Phase 4: Depth & Markers ==========

    # Depth Estimation
    "DepthEstimator",
    "create_depth_estimator",
    "get_depth_at_point",
    "depth_to_3d_point",
    "create_point_cloud",

    # Marker Detection
    "MarkerDetector",
    "MarkerType",
    "create_marker_detector",
    "create_default_camera_matrix",
    "marker_to_transform_matrix",

    # ========== Phase 5: Advanced Vision ==========

    # Semantic Segmentation
    "SemanticSegmenter",
    "SegmentationModel",
    "SegmentationDataset",
    "create_semantic_segmenter",
    "visualize_segmentation",
    "get_class_distribution",

    # Pose Estimation
    "PoseEstimator",
    "create_pose_estimator",
    "draw_pose_skeleton",
    "get_joint_angle",
    "detect_gesture",
    "get_body_orientation",

    # SLAM
    "SLAMProcessor",
    "create_slam_processor",
    "create_camera_matrix",
    "visualize_slam_tracking",
]
