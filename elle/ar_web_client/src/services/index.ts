/**
 * Vision Services - Complete AR Vision Pipeline
 *
 * Export unified vision services for AR integration:
 * - Object Detection (COCO-SSD)
 * - Hand Tracking (MediaPipe Hands)
 * - Depth Estimation (MiDaS ONNX)
 * - Marker Detection (ArUco, QR codes)
 *
 * Created: 2025-11-22
 * Updated: 2025-11-22 (Phase 4)
 */

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
