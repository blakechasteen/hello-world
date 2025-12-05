/**
 * CCDIKSolver - Cyclic Coordinate Descent Inverse Kinematics
 *
 * Very fast IK algorithm for short chains (2-5 joints).
 * Ideal for fingers, head look-at, and other simple chains.
 *
 * Algorithm:
 * Iteratively rotate each joint toward target, starting from end.
 * Simple but effective for short chains.
 *
 * Advantages:
 * - Very fast (no vector operations, just rotations)
 * - Simple implementation
 * - Good for short chains (3-5 bones)
 *
 * Disadvantages:
 * - Can produce unnatural poses for long chains
 * - Slower convergence than FABRIK for complex chains
 *
 * Created: 2025-11-22 (Phase 6.1)
 */

import * as THREE from 'three';

/**
 * Configuration for CCD-IK solver
 */
export interface CCDIKConfig {
  /**
   * Maximum iterations (default: 10)
   */
  maxIterations?: number;

  /**
   * Convergence tolerance (default: 0.01)
   * Distance threshold for end effector to target
   */
  tolerance?: number;

  /**
   * Maximum rotation per iteration (radians, default: PI)
   * Prevents excessive joint rotation in single step
   */
  maxRotationPerStep?: number;
}

/**
 * CCDIKSolver - Cyclic Coordinate Descent IK
 *
 * Usage:
 * ```typescript
 * const solver = new CCDIKSolver();
 * const joints = [shoulder, elbow, wrist];
 * const target = new Vector3(1, 0.5, 0);
 *
 * const solved = solver.solve(joints, target);
 * ```
 */
export class CCDIKSolver {
  private config: Required<CCDIKConfig>;

  constructor(config: CCDIKConfig = {}) {
    this.config = {
      maxIterations: config.maxIterations ?? 10,
      tolerance: config.tolerance ?? 0.01,
      maxRotationPerStep: config.maxRotationPerStep ?? Math.PI,
    };
  }

  /**
   * Solve IK chain using CCD algorithm
   *
   * @param joints - Array of joint positions (root → end effector)
   * @param target - Desired end effector position
   * @returns Updated joint positions
   */
  solve(
    joints: THREE.Vector3[],
    target: THREE.Vector3
  ): THREE.Vector3[] {
    const n = joints.length;

    if (n < 2) {
      throw new Error('CCD-IK requires at least 2 joints');
    }

    // Clone joints to avoid modifying input
    const solvedJoints = joints.map((j) => j.clone());

    let iterations = 0;
    while (
      solvedJoints[n - 1].distanceTo(target) > this.config.tolerance &&
      iterations < this.config.maxIterations
    ) {
      iterations++;

      // Iterate from second-to-last joint to root
      for (let i = n - 2; i >= 0; i--) {
        const jointPos = solvedJoints[i];
        const endEffectorPos = solvedJoints[n - 1];

        // Vectors from joint to end effector and target
        const toEndEffector = new THREE.Vector3()
          .subVectors(endEffectorPos, jointPos)
          .normalize();

        const toTarget = new THREE.Vector3()
          .subVectors(target, jointPos)
          .normalize();

        // Calculate rotation angle
        let angle = Math.acos(
          THREE.MathUtils.clamp(toEndEffector.dot(toTarget), -1, 1)
        );

        // Clamp rotation per step
        if (angle > this.config.maxRotationPerStep) {
          angle = this.config.maxRotationPerStep;
        }

        // Skip if angle is negligible
        if (angle < 0.001) continue;

        // Rotation axis (cross product)
        const axis = new THREE.Vector3()
          .crossVectors(toEndEffector, toTarget)
          .normalize();

        // Handle parallel vectors (cross product is zero)
        if (axis.length() < 0.001) {
          continue;
        }

        // Create rotation quaternion
        const rotation = new THREE.Quaternion().setFromAxisAngle(axis, angle);

        // Rotate all child joints (i+1 to end)
        for (let j = i + 1; j < n; j++) {
          const relativePos = new THREE.Vector3()
            .subVectors(solvedJoints[j], jointPos)
            .applyQuaternion(rotation);

          solvedJoints[j].copy(jointPos).add(relativePos);
        }

        // Early exit if close enough
        if (solvedJoints[n - 1].distanceTo(target) < this.config.tolerance) {
          break;
        }
      }
    }

    if (iterations >= this.config.maxIterations) {
      console.warn(
        `[CCD-IK] Max iterations reached (${this.config.maxIterations}). ` +
        `Final error: ${solvedJoints[n - 1].distanceTo(target).toFixed(4)}m`
      );
    }

    return solvedJoints;
  }

  /**
   * Solve with joint angle constraints
   *
   * @param joints - Joint positions
   * @param target - Target position
   * @param constraints - Per-joint angle limits
   * @returns Constrained solution
   */
  solveWithConstraints(
    joints: THREE.Vector3[],
    target: THREE.Vector3,
    constraints: Array<{
      minAngle: number;  // Minimum rotation (radians)
      maxAngle: number;  // Maximum rotation (radians)
      axis: THREE.Vector3;  // Rotation axis
    }>
  ): THREE.Vector3[] {
    const n = joints.length;
    const solvedJoints = joints.map((j) => j.clone());

    let iterations = 0;
    while (
      solvedJoints[n - 1].distanceTo(target) > this.config.tolerance &&
      iterations < this.config.maxIterations
    ) {
      iterations++;

      for (let i = n - 2; i >= 0; i--) {
        const jointPos = solvedJoints[i];
        const endEffectorPos = solvedJoints[n - 1];

        const toEndEffector = new THREE.Vector3()
          .subVectors(endEffectorPos, jointPos)
          .normalize();

        const toTarget = new THREE.Vector3()
          .subVectors(target, jointPos)
          .normalize();

        let angle = Math.acos(
          THREE.MathUtils.clamp(toEndEffector.dot(toTarget), -1, 1)
        );

        // Apply constraint
        const constraint = constraints[i];
        if (constraint) {
          angle = THREE.MathUtils.clamp(angle, constraint.minAngle, constraint.maxAngle);
        }

        // Clamp rotation per step
        if (angle > this.config.maxRotationPerStep) {
          angle = this.config.maxRotationPerStep;
        }

        if (angle < 0.001) continue;

        let axis = new THREE.Vector3()
          .crossVectors(toEndEffector, toTarget)
          .normalize();

        if (axis.length() < 0.001) continue;

        // Use constraint axis if provided
        if (constraint && constraint.axis) {
          axis = constraint.axis.clone().normalize();
        }

        const rotation = new THREE.Quaternion().setFromAxisAngle(axis, angle);

        for (let j = i + 1; j < n; j++) {
          const relativePos = new THREE.Vector3()
            .subVectors(solvedJoints[j], jointPos)
            .applyQuaternion(rotation);

          solvedJoints[j].copy(jointPos).add(relativePos);
        }

        if (solvedJoints[n - 1].distanceTo(target) < this.config.tolerance) {
          break;
        }
      }
    }

    return solvedJoints;
  }

  /**
   * Convert joint positions to rotation quaternions
   *
   * @param joints - Solved joint positions
   * @returns Rotation quaternions for each joint
   */
  jointsToRotations(joints: THREE.Vector3[]): THREE.Quaternion[] {
    const rotations: THREE.Quaternion[] = [];

    for (let i = 0; i < joints.length - 1; i++) {
      const direction = new THREE.Vector3()
        .subVectors(joints[i + 1], joints[i])
        .normalize();

      // Assume default forward is +Y
      const defaultForward = new THREE.Vector3(0, 1, 0);

      const rotation = new THREE.Quaternion()
        .setFromUnitVectors(defaultForward, direction);

      rotations.push(rotation);
    }

    return rotations;
  }
}

/**
 * Create default CCD-IK solver
 */
export function createCCDIKSolver(config?: CCDIKConfig): CCDIKSolver {
  return new CCDIKSolver(config);
}
