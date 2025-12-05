/**
 * Vision Services - Complete AR Vision Pipeline
 *
 * Export unified vision services for AR integration:
 *
 * Phase 2 (Core Vision):
 * - Object Detection (COCO-SSD)
 * - Hand Tracking (MediaPipe Hands)
 *
 * Phase 4 (Depth & Markers):
 * - Depth Estimation (MiDaS ONNX)
 * - Marker Detection (ArUco, QR codes)
 *
 * Phase 5 (Advanced Vision):
 * - Semantic Segmentation (DeepLabV3, BodyPix)
 * - Pose Estimation (MediaPipe Pose)
 * - SLAM (Visual Odometry)
 *
 * Created: 2025-11-22
 * Updated: 2025-11-22 (Phase 4-5)
 */

// ========== Phase 2: Core Vision ==========

// Object Detection
export {
  ObjectDetectionService,
  getObjectDetectionService,
  resetObjectDetectionService,
} from './objectDetection'

export type { DetectedObject, BoundingBox, ObjectDetectionConfig } from './objectDetection'

// Hand Tracking
export {
  HandTrackingService,
  getHandTrackingService,
  resetHandTrackingService,
  getPointingDirection,
  getPinchStrength,
  isHandInCenter,
  Gesture,
} from './handTracking'

export type {
  HandPose,
  HandLandmark,
  HandTrackingConfig,
  PointingDirection,
} from './handTracking'

// ========== Phase 4: Depth & Markers ==========

// Depth Estimation
export {
  DepthEstimationService,
  getDepthEstimationService,
  resetDepthEstimationService,
} from './depthEstimation'

export type {
  DepthMap,
  Point3D,
  DepthEstimationConfig,
} from './depthEstimation'

// Marker Detection
export {
  MarkerDetectionService,
  getMarkerDetectionService,
  resetMarkerDetectionService,
  createDefaultCameraCalibration,
} from './markerDetection'

export type {
  Marker,
  MarkerDetectionConfig,
  CameraCalibration,
} from './markerDetection'

// ========== Phase 5: Advanced Vision ==========

// Semantic Segmentation
export {
  SemanticSegmentationService,
  getSemanticSegmentationService,
  resetSemanticSegmentationService,
} from './semanticSegmentation'

export type {
  SegmentationMask,
  SegmentationConfig,
} from './semanticSegmentation'

// Pose Estimation
export {
  PoseEstimationService,
  getPoseEstimationService,
  resetPoseEstimationService,
  drawPoseSkeleton,
  getJointAngle,
  detectGesture,
  POSE_LANDMARK_NAMES,
} from './poseEstimation'

export type {
  Keypoint,
  BodyPose,
  PoseEstimationConfig,
} from './poseEstimation'

// SLAM
export {
  SLAMService,
  getSLAMService,
  resetSLAMService,
  visualizeSLAMTracking,
} from './slam'

export type {
  SLAMPose,
  MapPoint,
  SLAMConfig,
  SLAMStatistics,
} from './slam'
