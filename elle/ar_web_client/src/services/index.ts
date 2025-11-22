/**
 * Vision Services - Object Detection and Hand Tracking
 *
 * Export unified vision services for AR integration
 *
 * Created: 2025-11-22
 */

export {
  ObjectDetectionService,
  getObjectDetectionService,
  resetObjectDetectionService,
} from './objectDetection'

export type { DetectedObject, BoundingBox, ObjectDetectionConfig } from './objectDetection'

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
