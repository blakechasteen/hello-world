/**
 * Unit Tests for IK Solvers
 *
 * Comprehensive test suite for all three IK algorithms:
 * - FABRIKSolver (Forward And Backward Reaching IK)
 * - CCDIKSolver (Cyclic Coordinate Descent)
 * - TwoBoneIK (Analytical two-bone solution)
 *
 * Test Coverage:
 * - Convergence accuracy
 * - Bone length preservation
 * - Unreachable target handling
 * - Constraint application
 * - Edge cases
 *
 * Created: 2025-11-22 (Phase 6.1)
 */

import * as THREE from 'three';
import { FABRIKSolver } from '../FABRIKSolver';
import { CCDIKSolver } from '../CCDIKSolver';
import { TwoBoneIK } from '../TwoBoneIK';

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * Check if two vectors are approximately equal
 */
function expectVector3Near(
  actual: THREE.Vector3,
  expected: THREE.Vector3,
  tolerance: number = 0.01
): void {
  const distance = actual.distanceTo(expected);
  if (distance > tolerance) {
    throw new Error(
      `Vector3 not near: expected ${expected.toArray()}, ` +
      `got ${actual.toArray()}, distance: ${distance.toFixed(4)}`
    );
  }
}

/**
 * Check if bone lengths are preserved
 */
function expectBoneLengthsPreserved(
  originalJoints: THREE.Vector3[],
  solvedJoints: THREE.Vector3[],
  tolerance: number = 0.001
): void {
  if (originalJoints.length !== solvedJoints.length) {
    throw new Error('Joint arrays must have same length');
  }

  for (let i = 0; i < originalJoints.length - 1; i++) {
    const originalLength = originalJoints[i].distanceTo(originalJoints[i + 1]);
    const solvedLength = solvedJoints[i].distanceTo(solvedJoints[i + 1]);
    const error = Math.abs(originalLength - solvedLength);

    if (error > tolerance) {
      throw new Error(
        `Bone ${i} length not preserved: original ${originalLength.toFixed(4)}, ` +
        `solved ${solvedLength.toFixed(4)}, error: ${error.toFixed(4)}`
      );
    }
  }
}

/**
 * Create a simple 3-joint arm chain
 */
function createSimpleArmChain(): THREE.Vector3[] {
  return [
    new THREE.Vector3(0, 0, 0),     // Shoulder
    new THREE.Vector3(0, 1, 0),     // Elbow
    new THREE.Vector3(0, 2, 0),     // Wrist
  ];
}

/**
 * Create a 5-joint spine chain
 */
function createSpineChain(): THREE.Vector3[] {
  return [
    new THREE.Vector3(0, 0, 0),     // Hips
    new THREE.Vector3(0, 0.3, 0),   // Lower spine
    new THREE.Vector3(0, 0.6, 0),   // Mid spine
    new THREE.Vector3(0, 0.9, 0),   // Upper spine
    new THREE.Vector3(0, 1.2, 0),   // Neck
  ];
}

// ============================================================================
// FABRIK Solver Tests
// ============================================================================

describe('FABRIKSolver', () => {
  let solver: FABRIKSolver;

  beforeEach(() => {
    solver = new FABRIKSolver({
      tolerance: 0.01,
      maxIterations: 10,
      clampToMaxReach: true,
    });
  });

  // --------------------------------------------------------------------------
  // Basic Convergence
  // --------------------------------------------------------------------------

  test('should converge to reachable target', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(0.5, 1.8, 0);  // Reachable
    const solved = solver.solve(joints, target);

    // End effector should reach target
    expectVector3Near(solved[solved.length - 1], target, 0.01);
  });

  test('should preserve bone lengths', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(0.5, 1.5, 0);
    const solved = solver.solve(joints, target);

    expectBoneLengthsPreserved(joints, solved, 0.001);
  });

  test('should maintain root position', () => {
    const joints = createSimpleArmChain();
    const root = joints[0].clone();
    const target = new THREE.Vector3(0.5, 1.5, 0);
    const solved = solver.solve(joints, target, root);

    expectVector3Near(solved[0], root, 0.001);
  });

  // --------------------------------------------------------------------------
  // Unreachable Targets
  // --------------------------------------------------------------------------

  test('should handle unreachable target (beyond max reach)', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(0, 5, 0);  // Beyond max reach (2.0)
    const solved = solver.solve(joints, target);

    // Should stretch toward target
    const direction = new THREE.Vector3().subVectors(target, joints[0]).normalize();
    const expectedEnd = joints[0].clone().add(direction.multiplyScalar(2.0));

    expectVector3Near(solved[solved.length - 1], expectedEnd, 0.1);
  });

  test('should stretch all joints toward unreachable target', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(0, 10, 0);  // Far beyond reach
    const solved = solver.solve(joints, target);

    // All joints should be collinear along direction to target
    const direction = new THREE.Vector3().subVectors(target, joints[0]).normalize();

    for (let i = 1; i < solved.length; i++) {
      const jointDirection = new THREE.Vector3()
        .subVectors(solved[i], solved[i - 1])
        .normalize();

      const dot = jointDirection.dot(direction);
      expect(dot).toBeGreaterThan(0.99);  // Nearly parallel
    }
  });

  // --------------------------------------------------------------------------
  // Long Chains (5+ joints)
  // --------------------------------------------------------------------------

  test('should solve long chain (spine)', () => {
    const joints = createSpineChain();
    const target = new THREE.Vector3(0.3, 1.0, 0);  // Reachable
    const solved = solver.solve(joints, target);

    expectVector3Near(solved[solved.length - 1], target, 0.01);
    expectBoneLengthsPreserved(joints, solved, 0.001);
  });

  test('should converge within max iterations', () => {
    const joints = createSpineChain();
    const target = new THREE.Vector3(0.5, 0.8, 0.2);

    const solver = new FABRIKSolver({
      tolerance: 0.01,
      maxIterations: 5,  // Very few iterations
    });

    const solved = solver.solve(joints, target);

    // Should still converge (maybe not exactly at target)
    const finalDistance = solved[solved.length - 1].distanceTo(target);
    expect(finalDistance).toBeLessThan(0.1);  // Within 10cm
  });

  // --------------------------------------------------------------------------
  // Edge Cases
  // --------------------------------------------------------------------------

  test('should handle 2-joint chain (minimum)', () => {
    const joints = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 1, 0),
    ];
    const target = new THREE.Vector3(0.5, 0.8, 0);
    const solved = solver.solve(joints, target);

    expectVector3Near(solved[1], target, 0.01);
  });

  test('should throw error for single joint', () => {
    const joints = [new THREE.Vector3(0, 0, 0)];
    const target = new THREE.Vector3(1, 0, 0);

    expect(() => solver.solve(joints, target)).toThrow();
  });

  test('should handle target at exact max reach', () => {
    const joints = createSimpleArmChain();
    const maxReach = 2.0;
    const target = new THREE.Vector3(0, maxReach, 0);
    const solved = solver.solve(joints, target);

    expectVector3Near(solved[solved.length - 1], target, 0.01);
  });

  // --------------------------------------------------------------------------
  // Bone Length Validation
  // --------------------------------------------------------------------------

  test('validateBoneLengths should pass for preserved lengths', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(0.5, 1.5, 0);
    const solved = solver.solve(joints, target);

    const isValid = solver.validateBoneLengths(joints, solved, 0.001);
    expect(isValid).toBe(true);
  });

  test('validateBoneLengths should fail for modified lengths', () => {
    const joints = createSimpleArmChain();
    const solved = joints.map((j) => j.clone());
    solved[1].y += 0.5;  // Break bone length

    const isValid = solver.validateBoneLengths(joints, solved, 0.001);
    expect(isValid).toBe(false);
  });

  // --------------------------------------------------------------------------
  // Rotation Conversion
  // --------------------------------------------------------------------------

  test('jointsToRotations should produce valid quaternions', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(0.5, 1.5, 0);
    const solved = solver.solve(joints, target);

    const rotations = solver.jointsToRotations(solved);

    expect(rotations.length).toBe(solved.length - 1);
    rotations.forEach((rotation) => {
      expect(rotation.length()).toBeCloseTo(1.0, 3);  // Unit quaternion
    });
  });
});

// ============================================================================
// CCD-IK Solver Tests
// ============================================================================

describe('CCDIKSolver', () => {
  let solver: CCDIKSolver;

  beforeEach(() => {
    solver = new CCDIKSolver({
      tolerance: 0.01,
      maxIterations: 10,
      maxRotationPerStep: Math.PI / 4,  // 45 degrees
    });
  });

  // --------------------------------------------------------------------------
  // Basic Convergence
  // --------------------------------------------------------------------------

  test('should converge to reachable target', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(0.5, 1.8, 0);
    const solved = solver.solve(joints, target);

    expectVector3Near(solved[solved.length - 1], target, 0.01);
  });

  test('should preserve bone lengths', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(0.5, 1.5, 0);
    const solved = solver.solve(joints, target);

    expectBoneLengthsPreserved(joints, solved, 0.001);
  });

  // --------------------------------------------------------------------------
  // Short Chains (2-3 joints)
  // --------------------------------------------------------------------------

  test('should solve short chain (2 joints)', () => {
    const joints = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 1, 0),
    ];
    const target = new THREE.Vector3(0.7, 0.7, 0);
    const solved = solver.solve(joints, target);

    expectVector3Near(solved[1], target, 0.01);
  });

  test('should solve short chain (3 joints)', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(1, 1, 0);
    const solved = solver.solve(joints, target);

    expectVector3Near(solved[2], target, 0.01);
  });

  // --------------------------------------------------------------------------
  // Rotation Clamping
  // --------------------------------------------------------------------------

  test('should respect maxRotationPerStep', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(2, 0, 0);  // 90 degree rotation needed

    const solver = new CCDIKSolver({
      maxRotationPerStep: Math.PI / 8,  // Only 22.5 degrees per step
      maxIterations: 1,  // Only 1 iteration
    });

    const solved = solver.solve(joints, target);

    // Should not reach target in 1 iteration due to rotation limit
    const distance = solved[solved.length - 1].distanceTo(target);
    expect(distance).toBeGreaterThan(0.1);
  });

  // --------------------------------------------------------------------------
  // Early Termination
  // --------------------------------------------------------------------------

  test('should terminate early when target reached', () => {
    const joints = createSimpleArmChain();
    const target = new THREE.Vector3(0, 2, 0);  // Already aligned

    const solver = new CCDIKSolver({
      tolerance: 0.01,
      maxIterations: 100,  // Large number
    });

    const solved = solver.solve(joints, target);

    // Should converge in very few iterations (target already aligned)
    expectVector3Near(solved[solved.length - 1], target, 0.01);
  });

  // --------------------------------------------------------------------------
  // Edge Cases
  // --------------------------------------------------------------------------

  test('should handle target at root position', () => {
    const joints = createSimpleArmChain();
    const target = joints[0].clone();  // Target at root
    const solved = solver.solve(joints, target);

    // End effector should collapse toward root
    const distance = solved[solved.length - 1].distanceTo(target);
    expect(distance).toBeLessThan(0.5);  // Close to root
  });

  test('should handle collinear joints', () => {
    const joints = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 2, 0),
      new THREE.Vector3(0, 3, 0),
    ];
    const target = new THREE.Vector3(1, 2, 0);
    const solved = solver.solve(joints, target);

    expectVector3Near(solved[solved.length - 1], target, 0.1);
  });
});

// ============================================================================
// Two-Bone IK Solver Tests
// ============================================================================

describe('TwoBoneIK', () => {
  let solver: TwoBoneIK;

  beforeEach(() => {
    solver = new TwoBoneIK({
      minBendAngle: 0.1,  // ~5.7 degrees
      maxBendAngle: Math.PI - 0.1,  // ~174 degrees
      poleVectorWeight: 1.0,
    });
  });

  // --------------------------------------------------------------------------
  // Exact Solution (Reachable Targets)
  // --------------------------------------------------------------------------

  test('should produce exact solution for reachable target', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(0.5, 1.5, 0);

    const result = solver.solve(root, mid, end, target);

    expectVector3Near(result.end, target, 0.01);
    expect(result.reachable).toBe(true);
  });

  test('should preserve bone lengths', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(1, 1, 0);

    const upperLength = root.distanceTo(mid);
    const lowerLength = mid.distanceTo(end);

    const result = solver.solve(root, mid, end, target);

    const newUpperLength = root.distanceTo(result.mid);
    const newLowerLength = result.mid.distanceTo(result.end);

    expect(newUpperLength).toBeCloseTo(upperLength, 3);
    expect(newLowerLength).toBeCloseTo(lowerLength, 3);
  });

  // --------------------------------------------------------------------------
  // Unreachable Targets
  // --------------------------------------------------------------------------

  test('should detect unreachable target', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(0, 5, 0);  // Beyond max reach (2.0)

    const result = solver.solve(root, mid, end, target);

    expect(result.reachable).toBe(false);
    expect(result.overreach).toBeGreaterThan(0);
  });

  test('should stretch toward unreachable target', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(0, 10, 0);  // Far beyond reach

    const result = solver.solve(root, mid, end, target);

    // Should be at max reach along direction to target
    const expectedDistance = 2.0;  // Upper + lower length
    const actualDistance = root.distanceTo(result.end);

    expect(actualDistance).toBeCloseTo(expectedDistance, 2);
  });

  // --------------------------------------------------------------------------
  // Pole Vector
  // --------------------------------------------------------------------------

  test('should respect pole vector direction', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(1, 1, 0);

    // Pole vector pointing forward (+Z)
    const poleTarget = new THREE.Vector3(0, 1, 1);

    const result = solver.solve(root, mid, end, target, poleTarget);

    // Mid joint should bend toward +Z
    expect(result.mid.z).toBeGreaterThan(0);
  });

  test('should handle opposite pole vectors', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(1, 1, 0);

    // Pole vector pointing backward (-Z)
    const poleTarget1 = new THREE.Vector3(0, 1, 1);
    const result1 = solver.solve(root, mid, end, target, poleTarget1);

    // Pole vector pointing forward (+Z)
    const poleTarget2 = new THREE.Vector3(0, 1, -1);
    const result2 = solver.solve(root, mid, end, target, poleTarget2);

    // Mid joints should bend in opposite directions
    expect(result1.mid.z).toBeGreaterThan(0);
    expect(result2.mid.z).toBeLessThan(0);
  });

  // --------------------------------------------------------------------------
  // Anatomical Constraints
  // --------------------------------------------------------------------------

  test('should respect minBendAngle', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(0, 1.99, 0);  // Almost straight

    const solver = new TwoBoneIK({
      minBendAngle: 0.5,  // ~28 degrees minimum
    });

    const result = solver.solve(root, mid, end, target);

    // Bend angle should be at least 0.5 radians
    expect(result.bendAngle).toBeGreaterThanOrEqual(0.5);
  });

  test('should respect maxBendAngle', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(0, 0.1, 0);  // Extreme bend

    const solver = new TwoBoneIK({
      maxBendAngle: 2.5,  // ~143 degrees maximum
    });

    const result = solver.solve(root, mid, end, target);

    // Bend angle should be at most 2.5 radians
    expect(result.bendAngle).toBeLessThanOrEqual(2.5);
  });

  // --------------------------------------------------------------------------
  // Natural Solve (Arms/Legs)
  // --------------------------------------------------------------------------

  test('solveNatural should solve left arm', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(1, 1, 0);

    const result = solver.solveNatural(root, mid, end, target, 'left', 'arm');

    expectVector3Near(result.end, target, 0.01);
    expect(result.reachable).toBe(true);
  });

  test('solveNatural should solve right leg', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(0.5, 1.2, 0);

    const result = solver.solveNatural(root, mid, end, target, 'right', 'leg');

    expectVector3Near(result.end, target, 0.01);
    expect(result.reachable).toBe(true);
  });

  // --------------------------------------------------------------------------
  // Rotation Conversion
  // --------------------------------------------------------------------------

  test('solutionToRotations should produce valid quaternions', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(1, 1, 0);

    const solution = solver.solve(root, mid, end, target);
    const rotations = solver.solutionToRotations(solution, root);

    expect(rotations.upper.length()).toBeCloseTo(1.0, 3);  // Unit quaternion
    expect(rotations.lower.length()).toBeCloseTo(1.0, 3);
  });

  // --------------------------------------------------------------------------
  // Edge Cases
  // --------------------------------------------------------------------------

  test('should handle target at exact max reach', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const maxReach = 2.0;
    const target = new THREE.Vector3(0, maxReach, 0);

    const result = solver.solve(root, mid, end, target);

    expect(result.reachable).toBe(true);
    expectVector3Near(result.end, target, 0.01);
  });

  test('should handle target at root position', () => {
    const root = new THREE.Vector3(0, 0, 0);
    const mid = new THREE.Vector3(0, 1, 0);
    const end = new THREE.Vector3(0, 2, 0);
    const target = root.clone();

    const result = solver.solve(root, mid, end, target);

    // Should collapse toward root
    const distance = result.end.distanceTo(root);
    expect(distance).toBeLessThan(0.5);
  });
});
