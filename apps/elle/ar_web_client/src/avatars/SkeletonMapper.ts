/**
 * SkeletonMapper - Map MediaPipe pose keypoints to VRM humanoid skeleton
 *
 * Converts MediaPipe's 33 body landmarks into VRM bone rotations.
 * Handles missing keypoints, interpolates spine, and calculates natural poses.
 *
 * Created: 2025-11-22 (Phase 6.1)
 */

import * as THREE from 'three';
import { VRMHumanoid, VRMHumanBoneName } from '@pixiv/three-vrm';
import { BodyPose, Keypoint } from '../services/poseEstimation';

/**
 * Configuration for skeleton mapping
 */
export interface SkeletonMapperConfig {
  /**
   * Minimum keypoint visibility to use (0-1, default: 0.5)
   * Keypoints below this threshold are ignored
   */
  visibilityThreshold?: number;

  /**
   * Enable IK solvers (default: false for now, will be true when IK implemented)
   * Uses inverse kinematics for natural joint bending
   */
  enableIK?: boolean;

  /**
   * Hip height offset in meters (default: 0)
   * Adjusts avatar vertical position
   */
  hipHeightOffset?: number;

  /**
   * Smoothing factor for rotations (0-1, default: 0.3)
   * Higher = smoother but more lag
   */
  smoothingFactor?: number;
}

/**
 * MediaPipe landmark indices
 * https://google.github.io/mediapipe/solutions/pose.html
 */
export enum MediaPipeLandmark {
  NOSE = 0,
  LEFT_EYE_INNER = 1,
  LEFT_EYE = 2,
  LEFT_EYE_OUTER = 3,
  RIGHT_EYE_INNER = 4,
  RIGHT_EYE = 5,
  RIGHT_EYE_OUTER = 6,
  LEFT_EAR = 7,
  RIGHT_EAR = 8,
  MOUTH_LEFT = 9,
  MOUTH_RIGHT = 10,
  LEFT_SHOULDER = 11,
  RIGHT_SHOULDER = 12,
  LEFT_ELBOW = 13,
  RIGHT_ELBOW = 14,
  LEFT_WRIST = 15,
  RIGHT_WRIST = 16,
  LEFT_PINKY = 17,
  RIGHT_PINKY = 18,
  LEFT_INDEX = 19,
  RIGHT_INDEX = 20,
  LEFT_THUMB = 21,
  RIGHT_THUMB = 22,
  LEFT_HIP = 23,
  RIGHT_HIP = 24,
  LEFT_KNEE = 25,
  RIGHT_KNEE = 26,
  LEFT_ANKLE = 27,
  RIGHT_ANKLE = 28,
  LEFT_HEEL = 29,
  RIGHT_HEEL = 30,
  LEFT_FOOT_INDEX = 31,
  RIGHT_FOOT_INDEX = 32,
}

/**
 * Mapping from MediaPipe landmarks to VRM bones
 */
interface BoneMapping {
  bone: VRMHumanBoneName;
  landmarks: MediaPipeLandmark[];
  interpolate?: boolean;  // True if bone position is interpolated
}

/**
 * SkeletonMapper - Maps MediaPipe pose to VRM skeleton
 *
 * Usage:
 * ```typescript
 * const mapper = new SkeletonMapper();
 * const pose = await poseService.processFrame(video);
 * mapper.updateSkeleton(vrm.humanoid, pose);
 * ```
 */
export class SkeletonMapper {
  private config: Required<SkeletonMapperConfig>;
  private previousRotations: Map<VRMHumanBoneName, THREE.Quaternion> = new Map();

  constructor(config: SkeletonMapperConfig = {}) {
    this.config = {
      visibilityThreshold: config.visibilityThreshold ?? 0.5,
      enableIK: config.enableIK ?? false,
      hipHeightOffset: config.hipHeightOffset ?? 0,
      smoothingFactor: config.smoothingFactor ?? 0.3,
    };
  }

  /**
   * Update VRM skeleton from MediaPipe pose
   *
   * @param humanoid - VRM humanoid rig
   * @param pose - MediaPipe body pose with 33 keypoints
   */
  updateSkeleton(humanoid: VRMHumanoid, pose: BodyPose): void {
    if (!pose || !pose.keypoints || pose.keypoints.length !== 33) {
      console.warn('[SkeletonMapper] Invalid pose data');
      return;
    }

    // Convert keypoints to Vector3
    const landmarks = this.keypointsToVector3(pose.keypoints);

    // Update core skeleton
    this.updateHips(humanoid, landmarks);
    this.updateSpine(humanoid, landmarks);
    this.updateHead(humanoid, landmarks);

    // Update arms
    this.updateArm(humanoid, landmarks, 'left');
    this.updateArm(humanoid, landmarks, 'right');

    // Update legs
    this.updateLeg(humanoid, landmarks, 'left');
    this.updateLeg(humanoid, landmarks, 'right');

    // Update hands (finger rotations from landmarks)
    this.updateHand(humanoid, landmarks, 'left');
    this.updateHand(humanoid, landmarks, 'right');
  }

  /**
   * Convert MediaPipe keypoints to THREE.Vector3 array
   */
  private keypointsToVector3(keypoints: Keypoint[]): (THREE.Vector3 | null)[] {
    return keypoints.map((kp) => {
      if (kp.visibility < this.config.visibilityThreshold) {
        return null;  // Low confidence, ignore
      }

      // MediaPipe coords: x/y in [0,1], z in world units
      // Convert to Three.js world space
      return new THREE.Vector3(
        (kp.x - 0.5) * 2,  // Center at origin
        (1 - kp.y - 0.5) * 2,  // Flip Y (MediaPipe: top=0, Three: top=+1)
        -kp.z  // Z depth (negative for forward)
      );
    });
  }

  /**
   * Update hips position and rotation
   */
  private updateHips(humanoid: VRMHumanoid, landmarks: (THREE.Vector3 | null)[]): void {
    const leftHip = landmarks[MediaPipeLandmark.LEFT_HIP];
    const rightHip = landmarks[MediaPipeLandmark.RIGHT_HIP];

    if (!leftHip || !rightHip) return;

    const hipsBone = humanoid.getBoneNode('hips');
    if (!hipsBone) return;

    // Hips position: midpoint of left/right hips
    const hipsPosition = new THREE.Vector3()
      .addVectors(leftHip, rightHip)
      .multiplyScalar(0.5);

    hipsPosition.y += this.config.hipHeightOffset;

    hipsBone.position.copy(hipsPosition);

    // Hips rotation: face forward, align with hip orientation
    const hipDirection = new THREE.Vector3().subVectors(rightHip, leftHip).normalize();
    const forward = new THREE.Vector3(0, 0, 1);
    const up = new THREE.Vector3(0, 1, 0);

    const rotation = this.calculateRotationFromDirection(hipDirection, forward, up);
    this.applySmoothRotation(hipsBone, rotation, 'hips');
  }

  /**
   * Update spine bones (interpolated from hips and shoulders)
   */
  private updateSpine(humanoid: VRMHumanoid, landmarks: (THREE.Vector3 | null)[]): void {
    const leftHip = landmarks[MediaPipeLandmark.LEFT_HIP];
    const rightHip = landmarks[MediaPipeLandmark.RIGHT_HIP];
    const leftShoulder = landmarks[MediaPipeLandmark.LEFT_SHOULDER];
    const rightShoulder = landmarks[MediaPipeLandmark.RIGHT_SHOULDER];

    if (!leftHip || !rightHip || !leftShoulder || !rightShoulder) return;

    // Calculate hips and shoulders midpoints
    const hipsCenter = new THREE.Vector3()
      .addVectors(leftHip, rightHip)
      .multiplyScalar(0.5);

    const shouldersCenter = new THREE.Vector3()
      .addVectors(leftShoulder, rightShoulder)
      .multiplyScalar(0.5);

    // Interpolate spine and chest positions
    const spineBone = humanoid.getBoneNode('spine');
    const chestBone = humanoid.getBoneNode('chest');

    if (spineBone) {
      const spinePosition = new THREE.Vector3()
        .lerpVectors(hipsCenter, shouldersCenter, 0.33);  // 1/3 up from hips

      const spineDirection = new THREE.Vector3()
        .subVectors(shouldersCenter, hipsCenter)
        .normalize();

      const rotation = this.calculateRotationFromDirection(
        spineDirection,
        new THREE.Vector3(0, 1, 0),  // Up
        new THREE.Vector3(0, 0, 1)   // Forward
      );

      this.applySmoothRotation(spineBone, rotation, 'spine');
    }

    if (chestBone) {
      const chestPosition = new THREE.Vector3()
        .lerpVectors(hipsCenter, shouldersCenter, 0.67);  // 2/3 up from hips

      const chestDirection = new THREE.Vector3()
        .subVectors(shouldersCenter, hipsCenter)
        .normalize();

      const rotation = this.calculateRotationFromDirection(
        chestDirection,
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, 1)
      );

      this.applySmoothRotation(chestBone, rotation, 'chest');
    }
  }

  /**
   * Update head and neck
   */
  private updateHead(humanoid: VRMHumanoid, landmarks: (THREE.Vector3 | null)[]): void {
    const nose = landmarks[MediaPipeLandmark.NOSE];
    const leftShoulder = landmarks[MediaPipeLandmark.LEFT_SHOULDER];
    const rightShoulder = landmarks[MediaPipeLandmark.RIGHT_SHOULDER];

    if (!nose || !leftShoulder || !rightShoulder) return;

    const shouldersCenter = new THREE.Vector3()
      .addVectors(leftShoulder, rightShoulder)
      .multiplyScalar(0.5);

    // Neck rotation
    const neckBone = humanoid.getBoneNode('neck');
    if (neckBone) {
      const neckDirection = new THREE.Vector3()
        .subVectors(nose, shouldersCenter)
        .normalize();

      const rotation = this.calculateRotationFromDirection(
        neckDirection,
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, 1)
      );

      this.applySmoothRotation(neckBone, rotation, 'neck');
    }

    // Head rotation (look direction from nose + eyes)
    const headBone = humanoid.getBoneNode('head');
    if (headBone) {
      const leftEye = landmarks[MediaPipeLandmark.LEFT_EYE];
      const rightEye = landmarks[MediaPipeLandmark.RIGHT_EYE];

      let headDirection: THREE.Vector3;

      if (leftEye && rightEye) {
        // Calculate forward direction from eye line
        const eyeCenter = new THREE.Vector3()
          .addVectors(leftEye, rightEye)
          .multiplyScalar(0.5);

        headDirection = new THREE.Vector3()
          .subVectors(nose, eyeCenter)
          .normalize();
      } else {
        // Fallback: use neck direction
        headDirection = new THREE.Vector3()
          .subVectors(nose, shouldersCenter)
          .normalize();
      }

      const rotation = this.calculateRotationFromDirection(
        headDirection,
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, 1)
      );

      this.applySmoothRotation(headBone, rotation, 'head');
    }
  }

  /**
   * Update arm (left or right)
   */
  private updateArm(
    humanoid: VRMHumanoid,
    landmarks: (THREE.Vector3 | null)[],
    side: 'left' | 'right'
  ): void {
    const shoulder = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_SHOULDER : MediaPipeLandmark.RIGHT_SHOULDER];
    const elbow = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_ELBOW : MediaPipeLandmark.RIGHT_ELBOW];
    const wrist = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_WRIST : MediaPipeLandmark.RIGHT_WRIST];

    if (!shoulder || !elbow || !wrist) return;

    // Upper arm (shoulder → elbow)
    const upperArmBone = humanoid.getBoneNode(`${side}UpperArm` as VRMHumanBoneName);
    if (upperArmBone) {
      const upperArmDirection = new THREE.Vector3()
        .subVectors(elbow, shoulder)
        .normalize();

      const rotation = this.calculateRotationFromDirection(
        upperArmDirection,
        new THREE.Vector3(side === 'left' ? -1 : 1, 0, 0),  // Side direction
        new THREE.Vector3(0, 1, 0)  // Up
      );

      this.applySmoothRotation(upperArmBone, rotation, `${side}UpperArm` as VRMHumanBoneName);
    }

    // Lower arm (elbow → wrist)
    const lowerArmBone = humanoid.getBoneNode(`${side}LowerArm` as VRMHumanBoneName);
    if (lowerArmBone) {
      const lowerArmDirection = new THREE.Vector3()
        .subVectors(wrist, elbow)
        .normalize();

      const rotation = this.calculateRotationFromDirection(
        lowerArmDirection,
        new THREE.Vector3(side === 'left' ? -1 : 1, 0, 0),
        new THREE.Vector3(0, 1, 0)
      );

      this.applySmoothRotation(lowerArmBone, rotation, `${side}LowerArm` as VRMHumanBoneName);
    }
  }

  /**
   * Update leg (left or right)
   */
  private updateLeg(
    humanoid: VRMHumanoid,
    landmarks: (THREE.Vector3 | null)[],
    side: 'left' | 'right'
  ): void {
    const hip = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_HIP : MediaPipeLandmark.RIGHT_HIP];
    const knee = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_KNEE : MediaPipeLandmark.RIGHT_KNEE];
    const ankle = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_ANKLE : MediaPipeLandmark.RIGHT_ANKLE];
    const foot = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_FOOT_INDEX : MediaPipeLandmark.RIGHT_FOOT_INDEX];

    if (!hip || !knee || !ankle) return;

    // Upper leg (hip → knee)
    const upperLegBone = humanoid.getBoneNode(`${side}UpperLeg` as VRMHumanBoneName);
    if (upperLegBone) {
      const upperLegDirection = new THREE.Vector3()
        .subVectors(knee, hip)
        .normalize();

      const rotation = this.calculateRotationFromDirection(
        upperLegDirection,
        new THREE.Vector3(0, -1, 0),  // Down
        new THREE.Vector3(0, 0, 1)   // Forward
      );

      this.applySmoothRotation(upperLegBone, rotation, `${side}UpperLeg` as VRMHumanBoneName);
    }

    // Lower leg (knee → ankle)
    const lowerLegBone = humanoid.getBoneNode(`${side}LowerLeg` as VRMHumanBoneName);
    if (lowerLegBone) {
      const lowerLegDirection = new THREE.Vector3()
        .subVectors(ankle, knee)
        .normalize();

      const rotation = this.calculateRotationFromDirection(
        lowerLegDirection,
        new THREE.Vector3(0, -1, 0),
        new THREE.Vector3(0, 0, 1)
      );

      this.applySmoothRotation(lowerLegBone, rotation, `${side}LowerLeg` as VRMHumanBoneName);
    }

    // Foot (ankle → toe)
    const footBone = humanoid.getBoneNode(`${side}Foot` as VRMHumanBoneName);
    if (footBone && foot) {
      const footDirection = new THREE.Vector3()
        .subVectors(foot, ankle)
        .normalize();

      const rotation = this.calculateRotationFromDirection(
        footDirection,
        new THREE.Vector3(0, 0, 1),  // Forward
        new THREE.Vector3(0, 1, 0)   // Up
      );

      this.applySmoothRotation(footBone, rotation, `${side}Foot` as VRMHumanBoneName);
    }
  }

  /**
   * Update hand rotation from finger landmarks
   */
  private updateHand(
    humanoid: VRMHumanoid,
    landmarks: (THREE.Vector3 | null)[],
    side: 'left' | 'right'
  ): void {
    const wrist = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_WRIST : MediaPipeLandmark.RIGHT_WRIST];
    const pinky = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_PINKY : MediaPipeLandmark.RIGHT_PINKY];
    const index = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_INDEX : MediaPipeLandmark.RIGHT_INDEX];
    const thumb = landmarks[side === 'left' ? MediaPipeLandmark.LEFT_THUMB : MediaPipeLandmark.RIGHT_THUMB];

    if (!wrist || !pinky || !index || !thumb) return;

    const handBone = humanoid.getBoneNode(`${side}Hand` as VRMHumanBoneName);
    if (!handBone) return;

    // Calculate palm normal (cross product of finger vectors)
    const indexToPinky = new THREE.Vector3().subVectors(pinky, index);
    const thumbToPinky = new THREE.Vector3().subVectors(pinky, thumb);
    const palmNormal = new THREE.Vector3()
      .crossVectors(indexToPinky, thumbToPinky)
      .normalize();

    // Forward direction (wrist → center of fingers)
    const fingerCenter = new THREE.Vector3()
      .addVectors(pinky, index)
      .multiplyScalar(0.5);

    const forward = new THREE.Vector3()
      .subVectors(fingerCenter, wrist)
      .normalize();

    const rotation = this.calculateRotationFromDirection(
      forward,
      new THREE.Vector3(side === 'left' ? -1 : 1, 0, 0),
      palmNormal
    );

    this.applySmoothRotation(handBone, rotation, `${side}Hand` as VRMHumanBoneName);
  }

  /**
   * Calculate rotation quaternion from direction vectors
   *
   * @param forward - Desired forward direction
   * @param defaultForward - Default forward direction (e.g., +X for arm)
   * @param up - Up direction for orientation
   * @returns Rotation quaternion
   */
  private calculateRotationFromDirection(
    forward: THREE.Vector3,
    defaultForward: THREE.Vector3,
    up: THREE.Vector3
  ): THREE.Quaternion {
    // Normalize inputs
    forward = forward.clone().normalize();
    defaultForward = defaultForward.clone().normalize();
    up = up.clone().normalize();

    // Calculate right vector (perpendicular to forward and up)
    const right = new THREE.Vector3()
      .crossVectors(up, forward)
      .normalize();

    // Recalculate up (orthogonal to forward and right)
    const trueUp = new THREE.Vector3()
      .crossVectors(forward, right)
      .normalize();

    // Create rotation matrix
    const rotationMatrix = new THREE.Matrix4();
    rotationMatrix.makeBasis(right, trueUp, forward);

    // Extract quaternion
    const quaternion = new THREE.Quaternion();
    quaternion.setFromRotationMatrix(rotationMatrix);

    return quaternion;
  }

  /**
   * Apply rotation with smoothing
   *
   * Interpolates between previous and new rotation for smooth animation.
   */
  private applySmoothRotation(
    bone: THREE.Object3D,
    newRotation: THREE.Quaternion,
    boneName: VRMHumanBoneName
  ): void {
    const previousRotation = this.previousRotations.get(boneName);

    if (!previousRotation) {
      // First frame, no smoothing
      bone.quaternion.copy(newRotation);
      this.previousRotations.set(boneName, newRotation.clone());
      return;
    }

    // Spherical linear interpolation (SLERP)
    const smoothedRotation = new THREE.Quaternion()
      .slerpQuaternions(
        previousRotation,
        newRotation,
        this.config.smoothingFactor
      );

    bone.quaternion.copy(smoothedRotation);
    this.previousRotations.set(boneName, smoothedRotation.clone());
  }

  /**
   * Reset smoothing state
   *
   * Call this when switching avatars or starting new session.
   */
  resetSmoothing(): void {
    this.previousRotations.clear();
  }
}

/**
 * Create default skeleton mapper
 */
export function createSkeletonMapper(config?: SkeletonMapperConfig): SkeletonMapper {
  return new SkeletonMapper(config);
}
