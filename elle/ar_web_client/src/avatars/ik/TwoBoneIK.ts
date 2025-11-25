/**
 * TwoBoneIK - Analytical Two-Bone Inverse Kinematics
 *
 * Fast and exact IK solver for 2-bone chains (3 joints).
 * Ideal for elbows and knees.
 *
 * Algorithm:
 * Uses law of cosines to calculate exact joint angles.
 * No iteration needed - direct trigonometric solution.
 *
 * Advantages:
 * - Exact solution (no approximation)
 * - Fastest IK method (single trigonometric calculation)
 * - Always converges (no iteration)
 * - Perfect for elbows/knees
 *
 * Math:
 * For triangle with sides a, b, c and angle A opposite side a:
 * cos(A) = (b² + c² - a²) / (2bc)
 *
 * Created: 2025-11-22 (Phase 6.1)
 */

import * as THREE from 'three';

/**
 * Configuration for Two-Bone IK
 */
export interface TwoBoneIKConfig {
  /**
   * Minimum bend angle (radians, default: 0.1)
   * Prevents joint from locking straight
   */
  minBendAngle?: number;

  /**
   * Maximum bend angle (radians, default: PI - 0.1)
   * Prevents joint from bending backwards
   */
  maxBendAngle?: number;

  /**
   * Pole vector weight (0-1, default: 1.0)
   * How much to respect pole target direction
   */
  poleVectorWeight?: number;
}

/**
 * Two-Bone IK solution result
 */
export interface TwoBoneSolution {
  /**
   * Middle joint position (e.g., elbow, knee)
   */
  mid: THREE.Vector3;

  /**
   * End effector position (e.g., wrist, ankle)
   */
  end: THREE.Vector3;

  /**
   * Bend angle at middle joint (radians)
   */
  bendAngle: number;

  /**
   * Whether target was reachable
   */
  reachable: boolean;

  /**
   * If unreachable, how far beyond max reach
   */
  overreach?: number;
}

/**
 * TwoBoneIK - Analytical IK for 2-bone chains
 *
 * Usage:
 * ```typescript
 * const solver = new TwoBoneIK();
 *
 * // Solve arm IK
 * const result = solver.solve(
 *   shoulder,  // Root
 *   elbow,     // Mid
 *   wrist,     // End
 *   targetPos, // Desired wrist position
 *   new Vector3(0, 0, 1)  // Pole target (elbow should point forward)
 * );
 *
 * // Apply solution
 * elbowBone.position.copy(result.mid);
 * wristBone.position.copy(result.end);
 * ```
 */
export class TwoBoneIK {
  private config: Required<TwoBoneIKConfig>;

  constructor(config: TwoBoneIKConfig = {}) {
    this.config = {
      minBendAngle: config.minBendAngle ?? 0.1,  // ~5.7°
      maxBendAngle: config.maxBendAngle ?? Math.PI - 0.1,  // ~174°
      poleVectorWeight: config.poleVectorWeight ?? 1.0,
    };
  }

  /**
   * Solve two-bone IK using law of cosines
   *
   * @param root - Root joint position (e.g., shoulder, hip)
   * @param mid - Middle joint position (e.g., elbow, knee)
   * @param end - End effector position (e.g., wrist, ankle)
   * @param target - Desired end position
   * @param poleTarget - Direction for mid joint (e.g., elbow should point forward)
   * @returns Solution with updated mid and end positions
   */
  solve(
    root: THREE.Vector3,
    mid: THREE.Vector3,
    end: THREE.Vector3,
    target: THREE.Vector3,
    poleTarget?: THREE.Vector3
  ): TwoBoneSolution {
    // Calculate bone lengths
    const upperLength = root.distanceTo(mid);
    const lowerLength = mid.distanceTo(end);
    const maxReach = upperLength + lowerLength;
    const minReach = Math.abs(upperLength - lowerLength);

    // Distance from root to target
    const targetDistance = root.distanceTo(target);

    // Check if target is reachable
    let reachable = true;
    let overreach = 0;
    let clampedDistance = targetDistance;

    if (targetDistance > maxReach) {
      // Beyond max reach - stretch toward target
      reachable = false;
      overreach = targetDistance - maxReach;
      clampedDistance = maxReach;
    } else if (targetDistance < minReach) {
      // Within min reach - clamp to minimum
      clampedDistance = minReach;
    }

    // Direction from root to target
    const rootToTarget = new THREE.Vector3()
      .subVectors(target, root)
      .normalize();

    // ========== Law of Cosines ==========
    // Triangle sides:
    // a = lowerLength (opposite to angle A at root)
    // b = upperLength (adjacent to angle A)
    // c = clampedDistance (opposite to angle C at mid)
    //
    // cos(A) = (b² + c² - a²) / (2bc)

    const a = lowerLength;
    const b = upperLength;
    const c = clampedDistance;

    // Angle at root (upper bone angle)
    const cosAngleA = (b * b + c * c - a * a) / (2 * b * c);
    const angleA = Math.acos(THREE.MathUtils.clamp(cosAngleA, -1, 1));

    // Angle at mid (bend angle)
    const cosAngleB = (a * a + b * b - c * c) / (2 * a * b);
    const angleB = Math.acos(THREE.MathUtils.clamp(cosAngleB, -1, 1));

    // Clamp bend angle
    const clampedAngleB = THREE.MathUtils.clamp(
      angleB,
      this.config.minBendAngle,
      this.config.maxBendAngle
    );

    // ========== Calculate Pole Direction ==========
    let poleDirection: THREE.Vector3;

    if (poleTarget) {
      // Use provided pole target
      poleDirection = new THREE.Vector3()
        .subVectors(poleTarget, root)
        .projectOnPlane(rootToTarget)  // Project onto plane perpendicular to root→target
        .normalize();
    } else {
      // Default: use current mid position as pole
      poleDirection = new THREE.Vector3()
        .subVectors(mid, root)
        .projectOnPlane(rootToTarget)
        .normalize();
    }

    // Fallback if projection is zero (pole parallel to root→target)
    if (poleDirection.length() < 0.001) {
      poleDirection = new THREE.Vector3(0, 1, 0)  // Default up
        .projectOnPlane(rootToTarget)
        .normalize();
    }

    // ========== Calculate Mid Joint Position ==========
    // Rotate upper bone around root by angleA in pole direction
    const upperRotation = new THREE.Quaternion()
      .setFromAxisAngle(poleDirection, angleA);

    const newMid = root.clone()
      .add(
        rootToTarget.clone()
          .applyQuaternion(upperRotation)
          .multiplyScalar(upperLength)
      );

    // ========== Calculate End Effector Position ==========
    // If unreachable, place at max reach
    let newEnd: THREE.Vector3;

    if (!reachable) {
      // Stretch fully toward target
      newEnd = root.clone()
        .add(rootToTarget.clone().multiplyScalar(maxReach));
    } else {
      // Place at target (clamped distance)
      newEnd = root.clone()
        .add(rootToTarget.clone().multiplyScalar(clampedDistance));
    }

    return {
      mid: newMid,
      end: newEnd,
      bendAngle: clampedAngleB,
      reachable,
      overreach: reachable ? undefined : overreach,
    };
  }

  /**
   * Solve for elbow/knee with natural bending direction
   *
   * Convenience method that automatically determines pole direction
   * based on anatomical constraints.
   *
   * @param root - Shoulder or hip
   * @param mid - Elbow or knee
   * @param end - Wrist or ankle
   * @param target - Desired end position
   * @param side - 'left' or 'right'
   * @param joint - 'arm' or 'leg'
   * @returns Solution
   */
  solveNatural(
    root: THREE.Vector3,
    mid: THREE.Vector3,
    end: THREE.Vector3,
    target: THREE.Vector3,
    side: 'left' | 'right',
    joint: 'arm' | 'leg'
  ): TwoBoneSolution {
    // Determine natural pole direction
    let poleDirection: THREE.Vector3;

    if (joint === 'arm') {
      // Arms: elbows should point forward
      poleDirection = new THREE.Vector3(0, 0, 1);
    } else {
      // Legs: knees should point forward
      poleDirection = new THREE.Vector3(0, 0, 1);
    }

    // Mirror for right side
    if (side === 'right') {
      poleDirection.x *= -1;
    }

    return this.solve(root, mid, end, target, poleDirection);
  }

  /**
   * Convert solution to bone rotations
   *
   * @param solution - Two-bone IK solution
   * @param root - Root position
   * @returns Upper and lower bone rotations
   */
  solutionToRotations(
    solution: TwoBoneSolution,
    root: THREE.Vector3
  ): { upper: THREE.Quaternion; lower: THREE.Quaternion } {
    // Upper bone rotation (root → mid)
    const upperDirection = new THREE.Vector3()
      .subVectors(solution.mid, root)
      .normalize();

    const upperRotation = new THREE.Quaternion()
      .setFromUnitVectors(new THREE.Vector3(0, 1, 0), upperDirection);

    // Lower bone rotation (mid → end)
    const lowerDirection = new THREE.Vector3()
      .subVectors(solution.end, solution.mid)
      .normalize();

    const lowerRotation = new THREE.Quaternion()
      .setFromUnitVectors(new THREE.Vector3(0, 1, 0), lowerDirection);

    return {
      upper: upperRotation,
      lower: lowerRotation,
    };
  }
}

/**
 * Create default Two-Bone IK solver
 */
export function createTwoBoneIK(config?: TwoBoneIKConfig): TwoBoneIK {
  return new TwoBoneIK(config);
}
