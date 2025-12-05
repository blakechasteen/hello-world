/**
 * FABRIKSolver - Forward And Backward Reaching Inverse Kinematics
 *
 * Fast IK algorithm for chains with multiple joints (3+).
 * Ideal for spine, arms, and other long kinematic chains.
 *
 * Algorithm:
 * 1. BACKWARD pass: Start from end effector (target), propagate back to root
 * 2. FORWARD pass: Start from root, propagate forward to end effector
 * 3. Repeat until convergence or max iterations
 *
 * Advantages:
 * - Fast convergence (5-10 iterations typical)
 * - Stable for long chains
 * - No matrix inversions (just vector math)
 *
 * Created: 2025-11-22 (Phase 6.1)
 */

import * as THREE from 'three';

/**
 * Configuration for FABRIK solver
 */
export interface FABRIKConfig {
  /**
   * Convergence tolerance (default: 0.01)
   * Distance threshold for end effector to target
   */
  tolerance?: number;

  /**
   * Maximum iterations (default: 10)
   */
  maxIterations?: number;

  /**
   * Clamp chain to maximum reach (default: true)
   * Prevents impossible targets beyond chain length
   */
  clampToMaxReach?: boolean;
}

/**
 * FABRIKSolver - Forward And Backward Reaching IK
 *
 * Usage:
 * ```typescript
 * const solver = new FABRIKSolver();
 * const joints = [shoulder, elbow, wrist];  // Vector3 positions
 * const target = new Vector3(1, 2, 0);
 *
 * const solved = solver.solve(joints, target);
 * // Update bone positions from solved joints
 * ```
 */
export class FABRIKSolver {
  private config: Required<FABRIKConfig>;

  constructor(config: FABRIKConfig = {}) {
    this.config = {
      tolerance: config.tolerance ?? 0.01,
      maxIterations: config.maxIterations ?? 10,
      clampToMaxReach: config.clampToMaxReach ?? true,
    };
  }

  /**
   * Solve IK chain using FABRIK algorithm
   *
   * @param joints - Array of joint positions (root → end effector)
   * @param target - Desired end effector position
   * @param rootPosition - Optional fixed root position (defaults to joints[0])
   * @returns Updated joint positions
   */
  solve(
    joints: THREE.Vector3[],
    target: THREE.Vector3,
    rootPosition?: THREE.Vector3
  ): THREE.Vector3[] {
    const n = joints.length;

    if (n < 2) {
      throw new Error('FABRIK requires at least 2 joints');
    }

    // Clone joints to avoid modifying input
    const solvedJoints = joints.map((j) => j.clone());

    // Store root position (fixed during solving)
    const root = rootPosition ? rootPosition.clone() : solvedJoints[0].clone();

    // Calculate bone lengths (invariant during solving)
    const boneLengths: number[] = [];
    for (let i = 0; i < n - 1; i++) {
      boneLengths[i] = solvedJoints[i].distanceTo(solvedJoints[i + 1]);
    }

    // Total chain length
    const chainLength = boneLengths.reduce((sum, len) => sum + len, 0);

    // Check if target is reachable
    const targetDistance = root.distanceTo(target);

    if (this.config.clampToMaxReach && targetDistance > chainLength) {
      // Target beyond reach - stretch chain toward target
      const direction = new THREE.Vector3()
        .subVectors(target, root)
        .normalize();

      let currentPos = root.clone();
      for (let i = 0; i < n; i++) {
        solvedJoints[i].copy(currentPos);
        if (i < n - 1) {
          currentPos.add(direction.clone().multiplyScalar(boneLengths[i]));
        }
      }

      return solvedJoints;
    }

    // Iterative FABRIK solving
    let iterations = 0;
    while (
      solvedJoints[n - 1].distanceTo(target) > this.config.tolerance &&
      iterations < this.config.maxIterations
    ) {
      iterations++;

      // ========== BACKWARD PASS (end effector → root) ==========
      // Move end effector to target
      solvedJoints[n - 1].copy(target);

      // Iterate backwards, maintaining bone lengths
      for (let i = n - 2; i >= 0; i--) {
        // Direction from child to parent
        const direction = new THREE.Vector3()
          .subVectors(solvedJoints[i], solvedJoints[i + 1])
          .normalize();

        // Place parent at correct distance from child
        solvedJoints[i].copy(solvedJoints[i + 1])
          .add(direction.multiplyScalar(boneLengths[i]));
      }

      // ========== FORWARD PASS (root → end effector) ==========
      // Restore root position
      solvedJoints[0].copy(root);

      // Iterate forward, maintaining bone lengths
      for (let i = 0; i < n - 1; i++) {
        // Direction from parent to child
        const direction = new THREE.Vector3()
          .subVectors(solvedJoints[i + 1], solvedJoints[i])
          .normalize();

        // Place child at correct distance from parent
        solvedJoints[i + 1].copy(solvedJoints[i])
          .add(direction.multiplyScalar(boneLengths[i]));
      }
    }

    if (iterations >= this.config.maxIterations) {
      console.warn(
        `[FABRIK] Max iterations reached (${this.config.maxIterations}). ` +
        `Final error: ${solvedJoints[n - 1].distanceTo(target).toFixed(4)}m`
      );
    }

    return solvedJoints;
  }

  /**
   * Solve with constraints (angle limits)
   *
   * @param joints - Joint positions
   * @param target - Target position
   * @param constraints - Angle constraints per joint (optional)
   * @returns Constrained solution
   */
  solveWithConstraints(
    joints: THREE.Vector3[],
    target: THREE.Vector3,
    constraints?: Array<{ minAngle: number; maxAngle: number; axis: THREE.Vector3 }>
  ): THREE.Vector3[] {
    // First solve without constraints
    let solvedJoints = this.solve(joints, target);

    // If no constraints, return
    if (!constraints || constraints.length === 0) {
      return solvedJoints;
    }

    // Apply constraints (iterative approach)
    for (let iter = 0; iter < 3; iter++) {
      for (let i = 1; i < solvedJoints.length - 1; i++) {
        const constraint = constraints[i];
        if (!constraint) continue;

        // Calculate current angle
        const v1 = new THREE.Vector3()
          .subVectors(solvedJoints[i], solvedJoints[i - 1])
          .normalize();
        const v2 = new THREE.Vector3()
          .subVectors(solvedJoints[i + 1], solvedJoints[i])
          .normalize();

        const angle = Math.acos(THREE.MathUtils.clamp(v1.dot(v2), -1, 1));

        // Clamp angle
        if (angle < constraint.minAngle || angle > constraint.maxAngle) {
          const targetAngle = THREE.MathUtils.clamp(
            angle,
            constraint.minAngle,
            constraint.maxAngle
          );

          // Rotate around constraint axis to satisfy limit
          const rotation = new THREE.Quaternion()
            .setFromAxisAngle(constraint.axis, targetAngle - angle);

          // Apply rotation to child joints
          for (let j = i + 1; j < solvedJoints.length; j++) {
            const relative = new THREE.Vector3()
              .subVectors(solvedJoints[j], solvedJoints[i])
              .applyQuaternion(rotation);

            solvedJoints[j].copy(solvedJoints[i]).add(relative);
          }
        }
      }

      // Re-solve to maintain bone lengths after constraint application
      solvedJoints = this.solve(solvedJoints, target);
    }

    return solvedJoints;
  }

  /**
   * Convert joint positions to bone rotations
   *
   * Useful for applying solved positions back to skeleton.
   *
   * @param joints - Solved joint positions
   * @returns Array of rotation quaternions
   */
  jointsToRotations(joints: THREE.Vector3[]): THREE.Quaternion[] {
    const rotations: THREE.Quaternion[] = [];

    for (let i = 0; i < joints.length - 1; i++) {
      // Direction from joint to next joint
      const direction = new THREE.Vector3()
        .subVectors(joints[i + 1], joints[i])
        .normalize();

      // Default forward direction (assume +Y for now)
      const defaultForward = new THREE.Vector3(0, 1, 0);

      // Calculate rotation from default to current direction
      const rotation = new THREE.Quaternion()
        .setFromUnitVectors(defaultForward, direction);

      rotations.push(rotation);
    }

    return rotations;
  }

  /**
   * Validate bone lengths are preserved
   *
   * Debugging helper to ensure FABRIK maintains bone lengths.
   *
   * @param originalJoints - Original joint positions
   * @param solvedJoints - Solved joint positions
   * @returns True if bone lengths preserved within tolerance
   */
  validateBoneLengths(
    originalJoints: THREE.Vector3[],
    solvedJoints: THREE.Vector3[],
    tolerance: number = 0.001
  ): boolean {
    if (originalJoints.length !== solvedJoints.length) {
      return false;
    }

    for (let i = 0; i < originalJoints.length - 1; i++) {
      const originalLength = originalJoints[i].distanceTo(originalJoints[i + 1]);
      const solvedLength = solvedJoints[i].distanceTo(solvedJoints[i + 1]);

      const error = Math.abs(originalLength - solvedLength);
      if (error > tolerance) {
        console.error(
          `[FABRIK] Bone ${i} length mismatch: ` +
          `${originalLength.toFixed(4)} → ${solvedLength.toFixed(4)} ` +
          `(error: ${error.toFixed(4)})`
        );
        return false;
      }
    }

    return true;
  }
}

/**
 * Create default FABRIK solver
 */
export function createFABRIKSolver(config?: FABRIKConfig): FABRIKSolver {
  return new FABRIKSolver(config);
}
