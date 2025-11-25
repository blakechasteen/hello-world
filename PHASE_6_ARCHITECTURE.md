# Phase 6: 3D Avatar Integration - Complete Architecture

**Created**: 2025-11-22
**Status**: ✅ Architecture Complete, Ready for Implementation
**Timeline**: 8-12 weeks (3 sub-phases)
**Complexity**: Advanced - Real-time 3D graphics + networking + physics

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Avatar Model Formats & Standards](#2-avatar-model-formats--standards)
3. [Pose-to-Skeleton Mapping](#3-pose-to-skeleton-mapping)
4. [Inverse Kinematics (IK) Solvers](#4-inverse-kinematics-ik-solvers)
5. [Rendering Pipeline](#5-rendering-pipeline)
6. [Segmentation & Compositing](#6-segmentation--compositing)
7. [Spatial Positioning & Multi-User](#7-spatial-positioning--multi-user)
8. [Interaction Systems](#8-interaction-systems)
9. [Performance Optimization](#9-performance-optimization)
10. [Backend Infrastructure](#10-backend-infrastructure)
11. [Implementation Phases](#11-implementation-phases)
12. [File Structure](#12-file-structure)
13. [Integration with Phase 5](#13-integration-with-phase-5)
14. [Testing Strategy](#14-testing-strategy)
15. [Performance Benchmarks](#15-performance-benchmarks)
16. [Success Criteria](#16-success-criteria)

---

## 1. Architecture Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Phase 6: 3D Avatar System                    │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Phase 5    │      │  Avatar Core │      │  Multi-User  │
│   Services   │──────▶│    System    │◀─────│     Sync     │
└──────────────┘      └──────────────┘      └──────────────┘
     │ │ │                    │                      │
     │ │ │                    │                      │
     │ │ └────────┐           │           ┌──────────┘
     │ │          │           │           │
     ▼ ▼          ▼           ▼           ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  Pose   │  │Segment  │  │ Avatar  │  │ WebRTC  │
│  (33    │  │ (Body   │  │ Render  │  │  P2P    │
│  pts)   │  │  Pix)   │  │ (R3F)   │  │  Sync   │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
     │            │             │             │
     └────────────┴─────────────┴─────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │   AR Scene with      │
            │  Animated Avatars    │
            └──────────────────────┘
```

### Data Flow

```
Video Frame
    │
    ├──▶ PoseEstimationService (Phase 5)
    │         │
    │         └──▶ 33 Keypoints (MediaPipe)
    │                  │
    │                  ▼
    │         SkeletonMapper
    │                  │
    │                  ├──▶ VRM Bone Mapping
    │                  │
    │                  └──▶ IK Solvers
    │                         │
    │                         ├──▶ FABRIK (spine, arms)
    │                         ├──▶ CCD-IK (fingers, head)
    │                         └──▶ Two-Bone IK (elbows, knees)
    │                                  │
    │                                  ▼
    │                         Avatar Skeleton Update
    │                                  │
    │                                  ▼
    │                         React Three Fiber Render
    │
    └──▶ SemanticSegmentationService (Phase 5)
              │
              └──▶ Person Segmentation Mask
                       │
                       ▼
              MaskRefiner (erode/dilate/blur)
                       │
                       ▼
              AvatarCompositor
                       │
                       ├──▶ Alpha Mask Generation
                       │
                       └──▶ Composite with Background
                                  │
                                  ▼
                         Final AR Scene

Multi-User Sync (Parallel):
    SLAM Pose (Phase 5)
         │
         └──▶ Avatar Position in World Space
                  │
                  ▼
         WebRTC DataChannel
                  │
                  ├──▶ Broadcast Local Transform
                  │
                  └──▶ Receive Remote Transforms
                           │
                           ▼
                  Update Remote Avatars
```

### Core Components

1. **Avatar Manager** - Lifecycle management for all avatars
2. **VRM Loader** - Load and parse VRM 1.0 humanoid models
3. **Skeleton Mapper** - MediaPipe keypoints → VRM bones
4. **IK Solvers** - Natural pose calculation (FABRIK, CCD-IK, Two-Bone)
5. **Motion Smoother** - Reduce jitter with exponential filter + Kalman prediction
6. **Avatar Compositor** - Combine avatar with alpha mask for transparency
7. **Multi-User Sync** - WebRTC P2P avatar state synchronization
8. **Spatial Anchor Manager** - Position avatars in shared world space
9. **LOD Manager** - Level-of-detail optimization for performance
10. **Physics World** - Collision detection and spring bones (hair/cloth)

---

## 2. Avatar Model Formats & Standards

### VRM 1.0 Specification

VRM (Virtual Reality Model) is the **standard format for humanoid avatars**. It extends glTF 2.0 with:

- **Mandated humanoid bone mapping** (54 required bones)
- **MToon shader** for toon/cel-shading
- **Spring bones** for dynamic hair/cloth physics
- **Blend shapes** for facial expressions
- **First-person view settings** (invisible head for VR)

**Why VRM?**
- Industry standard for VR/AR avatars
- Guaranteed humanoid rig compatibility
- Cross-platform (VRChat, Cluster, Mozilla Hubs)
- Extensive tooling (VRoid Studio, UniVRM, three-vrm)

### VRM Humanoid Bone Hierarchy

```
hips (root)
├── spine
│   └── chest
│       ├── neck
│       │   └── head
│       │       ├── leftEye
│       │       └── rightEye
│       ├── leftShoulder
│       │   └── leftUpperArm
│       │       └── leftLowerArm
│       │           └── leftHand
│       │               ├── leftThumbProximal → leftThumbIntermediate → leftThumbDistal
│       │               ├── leftIndexProximal → leftIndexIntermediate → leftIndexDistal
│       │               ├── leftMiddleProximal → leftMiddleIntermediate → leftMiddleDistal
│       │               ├── leftRingProximal → leftRingIntermediate → leftRingDistal
│       │               └── leftLittleProximal → leftLittleIntermediate → leftLittleDistal
│       └── rightShoulder
│           └── rightUpperArm
│               └── rightLowerArm
│                   └── rightHand
│                       ├── rightThumb... (5 fingers)
│                       └── ...
├── leftUpperLeg
│   └── leftLowerLeg
│       └── leftFoot
│           └── leftToes
└── rightUpperLeg
    └── rightLowerLeg
        └── rightFoot
            └── rightToes
```

**Total**: 54 bones (Hips-based hierarchy)

### Ready Player Me Integration

**Ready Player Me** provides a free avatar customization API with:
- Web-based character creator (face, hair, clothing)
- Exports to GLB (glTF binary) format
- Automatic humanoid rig (compatible with VRM mapping)
- ~500 customization options
- API for programmatic avatar generation

**Integration Steps:**

1. **Embed Creator**:
   ```html
   <iframe
     src="https://demo.readyplayer.me/avatar?frameApi"
     allow="camera *; microphone *"
   ></iframe>
   ```

2. **Get Avatar URL**:
   ```typescript
   window.addEventListener('message', (event) => {
     if (event.data.source === 'readyplayerme') {
       const avatarUrl = event.data.url; // .glb file
       loadAvatar(avatarUrl);
     }
   });
   ```

3. **Load in Three.js**:
   ```typescript
   import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

   const loader = new GLTFLoader();
   loader.load(avatarUrl, (gltf) => {
     const avatar = gltf.scene;
     scene.add(avatar);
   });
   ```

### VRoid Studio

**VRoid Studio** is a free desktop app for creating anime-style avatars:
- Exports to VRM 1.0 format (native)
- Extensive customization (hair, face, clothing)
- Procedural hair generation
- Community asset marketplace

**Workflow:**
1. Create character in VRoid Studio
2. Export as VRM 1.0 (.vrm file)
3. Upload to CDN or bundle with app
4. Load with `@pixiv/three-vrm` library

### Fallback: glTF/GLB with Auto-Rigging

For non-humanoid or generic 3D models:

1. **Load GLB**:
   ```typescript
   loader.load('model.glb', (gltf) => {
     const model = gltf.scene;
     // Auto-detect skeleton
     const skeleton = model.getObjectByProperty('type', 'SkinnedMesh')?.skeleton;
   });
   ```

2. **Map Bones** (heuristic):
   ```typescript
   function detectHumanoidBones(skeleton: THREE.Skeleton): VRMHumanoidBones {
     // Search for bone names like "Hips", "Spine", "Head", etc.
     const bones: Partial<VRMHumanoidBones> = {};
     skeleton.bones.forEach((bone) => {
       const name = bone.name.toLowerCase();
       if (name.includes('hips')) bones.hips = bone;
       if (name.includes('spine')) bones.spine = bone;
       // ... (54 bones)
     });
     return bones as VRMHumanoidBones;
   }
   ```

---

## 3. Pose-to-Skeleton Mapping

### MediaPipe 33 Keypoints

**MediaPipe Pose** (Phase 5) provides 33 body landmarks:

```
0:  NOSE
1:  LEFT_EYE_INNER
2:  LEFT_EYE
3:  LEFT_EYE_OUTER
4:  RIGHT_EYE_INNER
5:  RIGHT_EYE
6:  RIGHT_EYE_OUTER
7:  LEFT_EAR
8:  RIGHT_EAR
9:  MOUTH_LEFT
10: MOUTH_RIGHT
11: LEFT_SHOULDER
12: RIGHT_SHOULDER
13: LEFT_ELBOW
14: RIGHT_ELBOW
15: LEFT_WRIST
16: RIGHT_WRIST
17: LEFT_PINKY
18: RIGHT_PINKY
19: LEFT_INDEX
20: RIGHT_INDEX
21: LEFT_THUMB
22: RIGHT_THUMB
23: LEFT_HIP
24: RIGHT_HIP
25: LEFT_KNEE
26: RIGHT_KNEE
27: LEFT_ANKLE
28: RIGHT_ANKLE
29: LEFT_HEEL
30: RIGHT_HEEL
31: LEFT_FOOT_INDEX
32: RIGHT_FOOT_INDEX
```

### VRM Bone Mapping Strategy

**Challenge**: MediaPipe provides **points**, VRM requires **bone rotations**.

**Solution**: Calculate bone rotations from directional vectors between keypoints.

```typescript
interface MediaPipeToVRMMapping {
  // Core skeleton
  hips: [23, 24];              // Average of left/right hip
  spine: [23, 24, 11, 12];     // Interpolate hips → shoulders
  chest: [11, 12];             // Average of shoulders
  neck: [11, 12, 0];           // Shoulders → nose
  head: [0];                   // Nose (forward direction from neck)

  // Left arm
  leftShoulder: [11];
  leftUpperArm: [11, 13];      // Shoulder → elbow
  leftLowerArm: [13, 15];      // Elbow → wrist
  leftHand: [15];              // Wrist (rotation from pinky/index/thumb)

  // Right arm
  rightShoulder: [12];
  rightUpperArm: [12, 14];
  rightLowerArm: [14, 16];
  rightHand: [16];

  // Left leg
  leftUpperLeg: [23, 25];      // Hip → knee
  leftLowerLeg: [25, 27];      // Knee → ankle
  leftFoot: [27, 31];          // Ankle → foot index
  leftToes: [29, 31];          // Heel → foot index

  // Right leg
  rightUpperLeg: [24, 26];
  rightLowerLeg: [26, 28];
  rightFoot: [28, 32];
  rightToes: [30, 32];

  // Fingers (no MediaPipe data - use procedural animation)
  leftThumbProximal: null;     // Procedural from hand rotation
  leftIndexProximal: [19];     // MediaPipe left index
  // ... (other fingers)
}
```

### Bone Rotation Calculation

```typescript
function calculateBoneRotation(
  boneStart: THREE.Vector3,
  boneEnd: THREE.Vector3,
  upVector: THREE.Vector3 = new THREE.Vector3(0, 1, 0)
): THREE.Quaternion {
  // 1. Calculate forward direction (bone direction)
  const forward = new THREE.Vector3()
    .subVectors(boneEnd, boneStart)
    .normalize();

  // 2. Calculate right direction (cross product with up)
  const right = new THREE.Vector3()
    .crossVectors(upVector, forward)
    .normalize();

  // 3. Recalculate up (orthogonal to forward and right)
  const up = new THREE.Vector3()
    .crossVectors(forward, right)
    .normalize();

  // 4. Create rotation matrix
  const rotationMatrix = new THREE.Matrix4();
  rotationMatrix.makeBasis(right, up, forward);

  // 5. Extract quaternion
  const quaternion = new THREE.Quaternion();
  quaternion.setFromRotationMatrix(rotationMatrix);

  return quaternion;
}
```

### Spine Interpolation

MediaPipe only provides hips (23, 24) and shoulders (11, 12). VRM needs intermediate spine and chest bones.

```typescript
function interpolateSpine(
  hips: THREE.Vector3,
  shoulders: THREE.Vector3
): { spine: THREE.Vector3; chest: THREE.Vector3 } {
  // Spine at 1/3 distance from hips to shoulders
  const spine = new THREE.Vector3().lerpVectors(hips, shoulders, 0.33);

  // Chest at 2/3 distance
  const chest = new THREE.Vector3().lerpVectors(hips, shoulders, 0.67);

  return { spine, chest };
}
```

### Hand Rotation from Finger Keypoints

MediaPipe provides pinky (17/18), index (19/20), and thumb (21/22) for each hand.

```typescript
function calculateHandRotation(
  wrist: THREE.Vector3,
  pinky: THREE.Vector3,
  index: THREE.Vector3,
  thumb: THREE.Vector3
): THREE.Quaternion {
  // 1. Palm normal (cross product of index→pinky and thumb→pinky)
  const indexToPinky = new THREE.Vector3().subVectors(pinky, index);
  const thumbToPinky = new THREE.Vector3().subVectors(pinky, thumb);
  const palmNormal = new THREE.Vector3()
    .crossVectors(indexToPinky, thumbToPinky)
    .normalize();

  // 2. Forward direction (wrist → middle of fingers)
  const fingerCenter = new THREE.Vector3()
    .addVectors(pinky, index)
    .multiplyScalar(0.5);
  const forward = new THREE.Vector3()
    .subVectors(fingerCenter, wrist)
    .normalize();

  // 3. Right direction (cross product)
  const right = new THREE.Vector3()
    .crossVectors(palmNormal, forward)
    .normalize();

  // 4. Create rotation
  const rotationMatrix = new THREE.Matrix4();
  rotationMatrix.makeBasis(right, palmNormal, forward);

  const quaternion = new THREE.Quaternion();
  quaternion.setFromRotationMatrix(rotationMatrix);

  return quaternion;
}
```

---

## 4. Inverse Kinematics (IK) Solvers

### Why IK?

**Problem**: Direct bone rotation from keypoints can produce **unnatural poses**:
- Elbow bending backwards
- Limbs stretching beyond natural length
- Shoulder twisting unnaturally

**Solution**: Inverse Kinematics constrains poses to **anatomically plausible** configurations.

### IK Solver Selection Strategy

Different body parts need different IK approaches:

| Body Part | IK Solver | Reason |
|-----------|-----------|--------|
| **Spine** | FABRIK | Long chain (3-4 bones), needs stability |
| **Arms** | FABRIK | Long chain (shoulder → elbow → wrist) |
| **Legs** | Two-Bone IK | Simple 2-bone chain (hip → knee → ankle) |
| **Fingers** | CCD-IK | Short chains (3 bones), needs speed |
| **Head Look-At** | CCD-IK | Single target (eyes → target) |

### FABRIK (Forward And Backward Reaching IK)

**Algorithm**: Iteratively reach from both ends toward target.

**Advantages**:
- Fast convergence (5-10 iterations typical)
- Stable for long chains
- No matrix inversions (just vector math)

**Code**:

```typescript
class FABRIKSolver {
  /**
   * Solve IK chain using FABRIK algorithm
   * @param joints - Array of joint positions (root → end effector)
   * @param target - Desired end effector position
   * @param tolerance - Distance threshold for convergence (default: 0.01)
   * @param maxIterations - Maximum iterations (default: 10)
   * @returns Updated joint positions
   */
  solve(
    joints: THREE.Vector3[],
    target: THREE.Vector3,
    tolerance: number = 0.01,
    maxIterations: number = 10
  ): THREE.Vector3[] {
    const n = joints.length;
    const rootPos = joints[0].clone();

    // Store bone lengths (invariant)
    const boneLengths: number[] = [];
    for (let i = 0; i < n - 1; i++) {
      boneLengths[i] = joints[i].distanceTo(joints[i + 1]);
    }

    let iterations = 0;
    while (joints[n - 1].distanceTo(target) > tolerance && iterations < maxIterations) {
      iterations++;

      // ========== BACKWARD PASS (end effector → root) ==========
      // Move end effector to target
      joints[n - 1].copy(target);

      // Iterate backwards, maintaining bone lengths
      for (let i = n - 2; i >= 0; i--) {
        // Direction from child to parent
        const direction = new THREE.Vector3()
          .subVectors(joints[i], joints[i + 1])
          .normalize();

        // Place parent at correct distance from child
        joints[i].copy(joints[i + 1])
          .add(direction.multiplyScalar(boneLengths[i]));
      }

      // ========== FORWARD PASS (root → end effector) ==========
      // Restore root position
      joints[0].copy(rootPos);

      // Iterate forward, maintaining bone lengths
      for (let i = 0; i < n - 1; i++) {
        // Direction from parent to child
        const direction = new THREE.Vector3()
          .subVectors(joints[i + 1], joints[i])
          .normalize();

        // Place child at correct distance from parent
        joints[i + 1].copy(joints[i])
          .add(direction.multiplyScalar(boneLengths[i]));
      }
    }

    return joints;
  }
}
```

**Usage Example**:

```typescript
// Spine IK: hips → spine → chest → neck
const spineChain = [hipsPos, spinePos, chestPos, neckPos];
const shoulderTarget = new THREE.Vector3().lerpVectors(
  leftShoulderPos,
  rightShoulderPos,
  0.5
);

const fabrik = new FABRIKSolver();
const solvedSpine = fabrik.solve(spineChain, shoulderTarget);

// Update VRM bones
vrmHumanoid.getBoneNode('hips').position.copy(solvedSpine[0]);
vrmHumanoid.getBoneNode('spine').position.copy(solvedSpine[1]);
vrmHumanoid.getBoneNode('chest').position.copy(solvedSpine[2]);
vrmHumanoid.getBoneNode('neck').position.copy(solvedSpine[3]);
```

### CCD-IK (Cyclic Coordinate Descent)

**Algorithm**: Iteratively rotate each joint toward target, starting from end.

**Advantages**:
- Very fast (no vector operations, just rotations)
- Simple implementation
- Good for short chains (3-5 bones)

**Code**:

```typescript
class CCDIKSolver {
  /**
   * Solve IK chain using CCD algorithm
   * @param joints - Array of joint positions
   * @param target - Desired end effector position
   * @param maxIterations - Maximum iterations (default: 10)
   * @returns Updated joint positions
   */
  solve(
    joints: THREE.Vector3[],
    target: THREE.Vector3,
    maxIterations: number = 10
  ): THREE.Vector3[] {
    const n = joints.length;

    for (let iteration = 0; iteration < maxIterations; iteration++) {
      // Iterate from second-to-last joint to root
      for (let i = n - 2; i >= 0; i--) {
        const jointPos = joints[i];
        const endEffectorPos = joints[n - 1];

        // Vectors from joint to end effector and target
        const toEndEffector = new THREE.Vector3()
          .subVectors(endEffectorPos, jointPos)
          .normalize();
        const toTarget = new THREE.Vector3()
          .subVectors(target, jointPos)
          .normalize();

        // Calculate rotation angle
        const angle = Math.acos(
          THREE.MathUtils.clamp(toEndEffector.dot(toTarget), -1, 1)
        );

        // Rotation axis (cross product)
        const axis = new THREE.Vector3()
          .crossVectors(toEndEffector, toTarget)
          .normalize();

        // Create rotation quaternion
        const rotation = new THREE.Quaternion().setFromAxisAngle(axis, angle);

        // Rotate all child joints
        for (let j = i + 1; j < n; j++) {
          const relativePos = new THREE.Vector3()
            .subVectors(joints[j], jointPos)
            .applyQuaternion(rotation);
          joints[j].copy(jointPos).add(relativePos);
        }

        // Early exit if close enough
        if (joints[n - 1].distanceTo(target) < 0.01) {
          return joints;
        }
      }
    }

    return joints;
  }
}
```

### Two-Bone IK (Analytical Solution)

**Algorithm**: Solve triangle geometry for 2-bone chains (e.g., elbow/knee).

**Advantages**:
- Exact solution (no iteration)
- Fastest (single trigonometric calculation)
- Always converges

**Code**:

```typescript
class TwoBoneIK {
  /**
   * Solve 2-bone IK using law of cosines
   * @param root - Root joint position (e.g., shoulder)
   * @param mid - Middle joint position (e.g., elbow)
   * @param end - End effector position (e.g., wrist)
   * @param target - Desired end position
   * @param poleTarget - Direction for mid joint (e.g., elbow should point forward)
   * @returns Updated mid and end positions
   */
  solve(
    root: THREE.Vector3,
    mid: THREE.Vector3,
    end: THREE.Vector3,
    target: THREE.Vector3,
    poleTarget?: THREE.Vector3
  ): { mid: THREE.Vector3; end: THREE.Vector3 } {
    // Bone lengths
    const upperLength = root.distanceTo(mid);
    const lowerLength = mid.distanceTo(end);
    const targetDistance = root.distanceTo(target);

    // Clamp target to reachable distance
    const maxReach = upperLength + lowerLength;
    const minReach = Math.abs(upperLength - lowerLength);
    const clampedDistance = THREE.MathUtils.clamp(
      targetDistance,
      minReach,
      maxReach
    );

    // Direction from root to target
    const rootToTarget = new THREE.Vector3()
      .subVectors(target, root)
      .normalize();

    // Law of cosines: cos(A) = (b² + c² - a²) / (2bc)
    const a = lowerLength;   // Opposite to angle A
    const b = upperLength;   // Adjacent to angle A
    const c = clampedDistance;

    const angleA = Math.acos(
      THREE.MathUtils.clamp(
        (b * b + c * c - a * a) / (2 * b * c),
        -1,
        1
      )
    );

    // Calculate pole direction (perpendicular to root→target)
    let poleDirection: THREE.Vector3;
    if (poleTarget) {
      poleDirection = new THREE.Vector3()
        .subVectors(poleTarget, root)
        .projectOnPlane(rootToTarget)
        .normalize();
    } else {
      // Default: use current mid position as pole
      poleDirection = new THREE.Vector3()
        .subVectors(mid, root)
        .projectOnPlane(rootToTarget)
        .normalize();
    }

    // Rotate upper bone toward target
    const upperRotation = new THREE.Quaternion()
      .setFromAxisAngle(poleDirection, angleA);

    const newMid = root.clone()
      .add(
        rootToTarget.clone()
          .applyQuaternion(upperRotation)
          .multiplyScalar(upperLength)
      );

    // End effector at target (clamped distance)
    const newEnd = root.clone()
      .add(rootToTarget.multiplyScalar(clampedDistance));

    return { mid: newMid, end: newEnd };
  }
}
```

**Usage Example**:

```typescript
// Solve left elbow IK
const shoulderPos = getPoseKeypoint(11);  // LEFT_SHOULDER
const elbowPos = getPoseKeypoint(13);     // LEFT_ELBOW
const wristPos = getPoseKeypoint(15);     // LEFT_WRIST

const solver = new TwoBoneIK();
const { mid, end } = solver.solve(
  shoulderPos,
  elbowPos,
  wristPos,
  wristPos,  // Target is current wrist position
  new THREE.Vector3(0, 0, 1)  // Elbow should point forward
);

// Update VRM bones
vrmHumanoid.getBoneNode('leftLowerArm').position.copy(mid);
vrmHumanoid.getBoneNode('leftHand').position.copy(end);
```

### Joint Constraints

**Problem**: IK solvers can produce **anatomically impossible poses** (e.g., elbow bending backwards).

**Solution**: Apply joint angle constraints after IK solve.

```typescript
interface JointConstraint {
  bone: string;
  type: 'hinge' | 'ball-socket';
  minAngle?: THREE.Euler;  // Minimum rotation (degrees)
  maxAngle?: THREE.Euler;  // Maximum rotation
  twistAxis?: THREE.Vector3;  // Axis for twist limits
}

const HUMANOID_CONSTRAINTS: JointConstraint[] = [
  // Elbow: hinge joint, 0-150° bend only
  {
    bone: 'leftLowerArm',
    type: 'hinge',
    minAngle: new THREE.Euler(0, 0, 0),
    maxAngle: new THREE.Euler(0, 0, Math.PI * 0.83),  // 150°
  },

  // Knee: hinge joint, 0-150° bend only
  {
    bone: 'leftLowerLeg',
    type: 'hinge',
    minAngle: new THREE.Euler(0, 0, 0),
    maxAngle: new THREE.Euler(Math.PI * 0.83, 0, 0),
  },

  // Shoulder: ball-socket, limited range
  {
    bone: 'leftUpperArm',
    type: 'ball-socket',
    minAngle: new THREE.Euler(-Math.PI * 0.5, -Math.PI * 0.5, -Math.PI),
    maxAngle: new THREE.Euler(Math.PI * 0.5, Math.PI * 0.5, Math.PI * 0.5),
  },

  // ... (other joints)
];

function applyJointConstraints(
  bone: THREE.Bone,
  constraint: JointConstraint
): void {
  const rotation = bone.rotation;

  if (constraint.type === 'hinge') {
    // Clamp to single axis
    rotation.x = THREE.MathUtils.clamp(
      rotation.x,
      constraint.minAngle!.x,
      constraint.maxAngle!.x
    );
    rotation.y = 0;  // No twist
    rotation.z = 0;
  } else if (constraint.type === 'ball-socket') {
    // Clamp all axes
    rotation.x = THREE.MathUtils.clamp(
      rotation.x,
      constraint.minAngle!.x,
      constraint.maxAngle!.x
    );
    rotation.y = THREE.MathUtils.clamp(
      rotation.y,
      constraint.minAngle!.y,
      constraint.maxAngle!.y
    );
    rotation.z = THREE.MathUtils.clamp(
      rotation.z,
      constraint.minAngle!.z,
      constraint.maxAngle!.z
    );
  }
}
```

---

## 5. Rendering Pipeline

### React Three Fiber Architecture

**React Three Fiber (R3F)** is a React renderer for Three.js with:
- Declarative 3D scenes
- Automatic disposal and cleanup
- Hooks for animation (`useFrame`)
- Integration with React ecosystem

**Basic Scene Setup**:

```typescript
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';

export function AvatarScene() {
  return (
    <Canvas>
      {/* Camera */}
      <PerspectiveCamera
        makeDefault
        position={[0, 1.6, 3]}
        fov={60}
      />

      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight
        position={[5, 5, 5]}
        intensity={1.0}
        castShadow
      />

      {/* Avatar */}
      <Avatar
        url="/avatars/default.vrm"
        position={[0, 0, 0]}
        rotation={[0, 0, 0, 1]}
      />

      {/* Controls */}
      <OrbitControls />
    </Canvas>
  );
}
```

### Avatar Component

```typescript
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import { VRM, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import * as THREE from 'three';

interface AvatarProps {
  url: string;
  pose: BodyPose | null;
  position: [number, number, number];
  rotation: [number, number, number, number];  // Quaternion [x, y, z, w]
  userId: string;
  enablePhysics?: boolean;
}

export function Avatar({
  url,
  pose,
  position,
  rotation,
  userId,
  enablePhysics = true,
}: AvatarProps) {
  const groupRef = useRef<THREE.Group>(null);
  const vrmRef = useRef<VRM | null>(null);
  const skeletonMapper = useRef(new SkeletonMapper());
  const smoother = useRef(new MotionSmoother());

  // Load VRM model
  useEffect(() => {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));

    loader.load(url, (gltf) => {
      const vrm = gltf.userData.vrm as VRM;

      // Rotate model 180° (VRM faces -Z by default)
      VRMUtils.rotateVRM0(vrm);

      vrmRef.current = vrm;

      if (groupRef.current) {
        groupRef.current.add(vrm.scene);
      }
    });

    return () => {
      // Cleanup
      if (vrmRef.current) {
        VRMUtils.deepDispose(vrmRef.current.scene);
      }
    };
  }, [url]);

  // Update skeleton from pose (every frame)
  useFrame((state, delta) => {
    if (!vrmRef.current || !pose) return;

    const vrm = vrmRef.current;

    // 1. Smooth keypoints to reduce jitter
    const smoothedKeypoints = smoother.current.smooth(
      pose.keypoints.map((kp) => new THREE.Vector3(kp.x, kp.y, kp.z))
    );

    // 2. Update skeleton from smoothed pose
    skeletonMapper.current.updateSkeleton(vrm.humanoid, {
      ...pose,
      keypoints: smoothedKeypoints.map((v, i) => ({
        x: v.x,
        y: v.y,
        z: v.z,
        visibility: pose.keypoints[i].visibility,
        presence: pose.keypoints[i].presence,
      })),
    });

    // 3. Update spring bones (hair/cloth physics)
    if (enablePhysics && vrm.springBoneManager) {
      vrm.springBoneManager.update(delta);
    }

    // 4. Update blend shapes (facial expressions - future)
    // vrm.expressionManager?.update();
  });

  // Update position/rotation from SLAM or WebRTC
  useEffect(() => {
    if (groupRef.current) {
      groupRef.current.position.set(...position);
      groupRef.current.quaternion.set(...rotation);
    }
  }, [position, rotation]);

  return (
    <group ref={groupRef}>
      {/* VRM model added in useEffect */}
    </group>
  );
}
```

### PBR Materials & MToon Shader

**PBR (Physically Based Rendering)** uses metalness/roughness for realistic materials:

```typescript
const material = new THREE.MeshStandardMaterial({
  map: colorTexture,
  normalMap: normalTexture,
  metalnessMap: metalnessTexture,
  roughnessMap: roughnessTexture,
  metalness: 0.0,    // Non-metallic (skin, cloth)
  roughness: 0.8,    // Rough surface
});
```

**MToon Shader** (VRM standard) for toon/cel-shading:

```typescript
// Automatically applied by VRM loader
// Configurable via VRM.materials
vrm.materials.forEach((material) => {
  if (material.type === 'MToon') {
    material.shadeColor = new THREE.Color(0x7f7f7f);  // Shadow color
    material.shadeShift = 0.0;   // Shadow threshold
    material.shadeToony = 0.9;   // Sharpness (0 = PBR, 1 = toon)
  }
});
```

### Lighting Setup

```typescript
// Ambient light (base illumination)
<ambientLight intensity={0.5} color="#ffffff" />

// Directional light (sun)
<directionalLight
  position={[5, 5, 5]}
  intensity={1.0}
  castShadow
  shadow-mapSize-width={2048}
  shadow-mapSize-height={2048}
  shadow-camera-near={0.5}
  shadow-camera-far={50}
/>

// Hemisphere light (sky + ground)
<hemisphereLight
  skyColor="#87CEEB"    // Sky blue
  groundColor="#8B4513" // Brown ground
  intensity={0.3}
/>

// Point light (fill light)
<pointLight
  position={[-3, 2, 3]}
  intensity={0.5}
  distance={10}
  decay={2}
/>
```

### LOD (Level of Detail)

Optimize performance by reducing polygon count for distant avatars:

```typescript
import { useMemo } from 'react';
import { SimplifyModifier } from 'three/examples/jsm/modifiers/SimplifyModifier';

function createLODMeshes(
  originalMesh: THREE.Mesh
): { high: THREE.Mesh; medium: THREE.Mesh; low: THREE.Mesh } {
  const simplifier = new SimplifyModifier();

  // High detail (original)
  const high = originalMesh.clone();

  // Medium detail (50% polygons)
  const mediumGeometry = originalMesh.geometry.clone();
  simplifier.modify(mediumGeometry, Math.floor(mediumGeometry.attributes.position.count * 0.5));
  const medium = new THREE.Mesh(mediumGeometry, originalMesh.material);

  // Low detail (25% polygons)
  const lowGeometry = originalMesh.geometry.clone();
  simplifier.modify(lowGeometry, Math.floor(lowGeometry.attributes.position.count * 0.25));
  const low = new THREE.Mesh(lowGeometry, originalMesh.material);

  return { high, medium, low };
}

function selectLOD(
  cameraPosition: THREE.Vector3,
  avatarPosition: THREE.Vector3,
  lodMeshes: { high: THREE.Mesh; medium: THREE.Mesh; low: THREE.Mesh }
): THREE.Mesh {
  const distance = cameraPosition.distanceTo(avatarPosition);

  if (distance < 3) return lodMeshes.high;       // <3m: high detail
  if (distance < 10) return lodMeshes.medium;    // 3-10m: medium
  return lodMeshes.low;                          // >10m: low detail
}
```

### Post-Processing

Optional visual effects for enhanced quality:

```typescript
import { EffectComposer, Bloom, ChromaticAberration, SSAO } from '@react-three/postprocessing';

<EffectComposer>
  {/* Bloom (glow effect) */}
  <Bloom
    intensity={0.5}
    luminanceThreshold={0.9}
    luminanceSmoothing={0.9}
  />

  {/* SSAO (ambient occlusion) */}
  <SSAO
    samples={16}
    radius={5}
    intensity={0.5}
  />

  {/* Chromatic aberration (lens effect) */}
  <ChromaticAberration
    offset={new THREE.Vector2(0.001, 0.001)}
  />
</EffectComposer>
```

---

## 6. Segmentation & Compositing

### BodyPix Integration (Phase 5)

**Phase 5** provides `SemanticSegmentationService` with BodyPix backend for person segmentation.

**Get Segmentation Mask**:

```typescript
import { getSemanticSegmentationService } from '../services';

const segService = getSemanticSegmentationService();

// Process video frame
const segmentation = await segService.segmentImage(videoElement);

// segmentation.data: Uint8Array with class IDs
// - 0: background
// - 15: person (COCO class ID)
```

### Alpha Mask Generation

Convert segmentation to binary/gradient alpha mask:

```typescript
class AlphaMaskGenerator {
  generateMask(
    segmentation: SegmentationMask,
    targetClass: number = 15  // Person class
  ): ImageData {
    const { width, height, data } = segmentation;
    const alphaData = new Uint8ClampedArray(width * height * 4);  // RGBA

    for (let i = 0; i < data.length; i++) {
      const classId = data[i];
      const alpha = classId === targetClass ? 255 : 0;

      // RGBA (set all to white with alpha)
      alphaData[i * 4 + 0] = 255;  // R
      alphaData[i * 4 + 1] = 255;  // G
      alphaData[i * 4 + 2] = 255;  // B
      alphaData[i * 4 + 3] = alpha; // A (0 = transparent, 255 = opaque)
    }

    return new ImageData(alphaData, width, height);
  }
}
```

### Mask Refinement

Improve edge quality with morphological operations:

```typescript
class MaskRefiner {
  /**
   * Refine alpha mask with morphological operations
   * @param mask - Binary alpha mask
   * @param erodeRadius - Erosion radius (shrink mask, default: 2)
   * @param dilateRadius - Dilation radius (grow mask, default: 2)
   * @param blurRadius - Gaussian blur radius (smooth edges, default: 3)
   * @returns Refined mask
   */
  refine(
    mask: ImageData,
    erodeRadius: number = 2,
    dilateRadius: number = 2,
    blurRadius: number = 3
  ): ImageData {
    // 1. Erode (shrink) to remove noise
    const eroded = this.erode(mask, erodeRadius);

    // 2. Dilate (grow) to restore size
    const dilated = this.dilate(eroded, dilateRadius);

    // 3. Gaussian blur for soft edges
    const blurred = this.gaussianBlur(dilated, blurRadius);

    return blurred;
  }

  private erode(mask: ImageData, radius: number): ImageData {
    const { width, height, data } = mask;
    const output = new Uint8ClampedArray(data.length);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let minAlpha = 255;

        // Check neighborhood
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx;
            const ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const idx = (ny * width + nx) * 4 + 3;  // Alpha channel
              minAlpha = Math.min(minAlpha, data[idx]);
            }
          }
        }

        const idx = (y * width + x) * 4;
        output[idx + 0] = 255;
        output[idx + 1] = 255;
        output[idx + 2] = 255;
        output[idx + 3] = minAlpha;
      }
    }

    return new ImageData(output, width, height);
  }

  private dilate(mask: ImageData, radius: number): ImageData {
    const { width, height, data } = mask;
    const output = new Uint8ClampedArray(data.length);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let maxAlpha = 0;

        // Check neighborhood
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx;
            const ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const idx = (ny * width + nx) * 4 + 3;
              maxAlpha = Math.max(maxAlpha, data[idx]);
            }
          }
        }

        const idx = (y * width + x) * 4;
        output[idx + 0] = 255;
        output[idx + 1] = 255;
        output[idx + 2] = 255;
        output[idx + 3] = maxAlpha;
      }
    }

    return new ImageData(output, width, height);
  }

  private gaussianBlur(mask: ImageData, radius: number): ImageData {
    // Gaussian kernel
    const kernel = this.createGaussianKernel(radius);
    const { width, height, data } = mask;
    const output = new Uint8ClampedArray(data.length);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        let weightSum = 0;

        for (let ky = -radius; ky <= radius; ky++) {
          for (let kx = -radius; kx <= radius; kx++) {
            const nx = x + kx;
            const ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const idx = (ny * width + nx) * 4 + 3;
              const weight = kernel[ky + radius][kx + radius];
              sum += data[idx] * weight;
              weightSum += weight;
            }
          }
        }

        const idx = (y * width + x) * 4;
        output[idx + 0] = 255;
        output[idx + 1] = 255;
        output[idx + 2] = 255;
        output[idx + 3] = sum / weightSum;
      }
    }

    return new ImageData(output, width, height);
  }

  private createGaussianKernel(radius: number): number[][] {
    const size = radius * 2 + 1;
    const kernel: number[][] = [];
    const sigma = radius / 2;

    for (let y = 0; y < size; y++) {
      kernel[y] = [];
      for (let x = 0; x < size; x++) {
        const dx = x - radius;
        const dy = y - radius;
        kernel[y][x] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      }
    }

    return kernel;
  }
}
```

### Compositing Strategies

**1. Direct Alpha Blending** (simplest):

```typescript
// Fragment shader
precision highp float;

uniform sampler2D avatarTexture;
uniform sampler2D backgroundTexture;
uniform sampler2D alphaMask;

varying vec2 vUv;

void main() {
  vec4 avatar = texture2D(avatarTexture, vUv);
  vec4 background = texture2D(backgroundTexture, vUv);
  float alpha = texture2D(alphaMask, vUv).a;

  // Blend
  vec3 color = mix(background.rgb, avatar.rgb, alpha);
  gl_FragColor = vec4(color, 1.0);
}
```

**2. Premultiplied Alpha** (more accurate):

```typescript
// Premultiply avatar RGB by alpha
vec4 avatar = texture2D(avatarTexture, vUv);
float alpha = texture2D(alphaMask, vUv).a;
vec3 premultiplied = avatar.rgb * alpha;

// Blend
vec3 color = premultiplied + background.rgb * (1.0 - alpha);
gl_FragColor = vec4(color, 1.0);
```

**3. Chroma Key (Green Screen)**:

```typescript
// Remove green background
vec4 avatar = texture2D(avatarTexture, vUv);
vec3 greenKey = vec3(0.0, 1.0, 0.0);
float chromaDist = distance(avatar.rgb, greenKey);
float alpha = smoothstep(0.0, 0.3, chromaDist);  // Threshold: 0.3

vec3 color = mix(background.rgb, avatar.rgb, alpha);
gl_FragColor = vec4(color, 1.0);
```

---

## 7. Spatial Positioning & Multi-User

### SLAM Integration (Phase 5)

**Phase 5** provides `SLAMService` with 6-DOF camera tracking.

**Get Camera Pose**:

```typescript
import { getSLAMService } from '../services';

const slamService = getSLAMService();

// In XR rendering loop
const slamPose = await slamService.processFrame(videoElement, xrFrame);

// slamPose.position: [x, y, z]
// slamPose.orientation: [x, y, z, w] (quaternion)
```

### Avatar Positioning in World Space

```typescript
class AvatarPositionManager {
  /**
   * Update avatar position/rotation from SLAM pose
   * @param userId - User ID
   * @param slamPose - 6-DOF pose from SLAM
   * @param height - User height offset (default: 1.6m)
   */
  updateFromSLAM(
    userId: string,
    slamPose: SLAMPose,
    height: number = 1.6
  ): void {
    const avatar = this.avatars.get(userId);
    if (!avatar) return;

    // Position at floor (Y = 0), offset by height
    const position = new THREE.Vector3(
      slamPose.position[0],
      0,  // Floor level (avatar hips at Y=0)
      slamPose.position[2]
    );

    // Rotation from SLAM quaternion
    const rotation = new THREE.Quaternion(
      slamPose.orientation[0],
      slamPose.orientation[1],
      slamPose.orientation[2],
      slamPose.orientation[3]
    );

    avatar.position.copy(position);
    avatar.quaternion.copy(rotation);
  }
}
```

### WebXR Spatial Anchors

**Goal**: Place avatars at persistent world positions (e.g., "avatar stands here").

**Limitation**: WebXR spatial anchors are **local-only** (not network-shareable).

**Solution**: Each client creates anchor at same world position using shared coordinate system.

```typescript
class SpatialAnchorManager {
  private anchors: Map<string, XRAnchor> = new Map();

  /**
   * Create spatial anchor at world position
   * @param id - Anchor ID (shared across network)
   * @param position - World position [x, y, z]
   * @param xrFrame - Current XR frame
   * @returns XRAnchor (local to this device)
   */
  async createAnchor(
    id: string,
    position: [number, number, number],
    xrFrame: XRFrame
  ): Promise<XRAnchor | null> {
    const session = xrFrame.session;

    // Create anchor at position
    const anchorPose = new XRRigidTransform({
      x: position[0],
      y: position[1],
      z: position[2],
    });

    try {
      const anchor = await session.requestAnchor(anchorPose, xrFrame.referenceSpace);
      if (anchor) {
        this.anchors.set(id, anchor);
      }
      return anchor;
    } catch (error) {
      console.warn('Failed to create anchor:', error);
      return null;
    }
  }

  /**
   * Get anchor pose (updates each frame as device refines world understanding)
   * @param id - Anchor ID
   * @param xrFrame - Current XR frame
   * @returns Anchor pose or null
   */
  getAnchorPose(
    id: string,
    xrFrame: XRFrame
  ): XRPose | null {
    const anchor = this.anchors.get(id);
    if (!anchor) return null;

    return xrFrame.getPose(anchor.anchorSpace, xrFrame.referenceSpace);
  }
}
```

### WebRTC Multi-User Synchronization

**Goal**: Synchronize avatar transforms across network with low latency (<50ms).

**Architecture**:
- **WebSocket** for signaling (peer connection establishment)
- **WebRTC DataChannel** for P2P avatar state (low latency)
- **Fallback** to server relay if P2P fails

**Signaling Server** (Python):

```python
# HoloLoom/server/signaling_server.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
from typing import Dict, Set

app = FastAPI()

# Active connections: room_id -> set of WebSocket connections
rooms: Dict[str, Set[WebSocket]] = {}

@app.websocket("/signaling/{room_id}")
async def signaling_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()

    # Join room
    if room_id not in rooms:
        rooms[room_id] = set()
    rooms[room_id].add(websocket)

    try:
        while True:
            # Receive signaling message
            data = await websocket.receive_json()

            # Forward to all peers in room (except sender)
            for peer in rooms[room_id]:
                if peer != websocket:
                    await peer.send_json(data)

    except WebSocketDisconnect:
        # Leave room
        rooms[room_id].remove(websocket)
        if not rooms[room_id]:
            del rooms[room_id]
```

**WebRTC Manager** (TypeScript):

```typescript
class WebRTCManager {
  private peerConnections: Map<string, RTCPeerConnection> = new Map();
  private dataChannels: Map<string, RTCDataChannel> = new Map();
  private signalingWs: WebSocket | null = null;

  /**
   * Connect to signaling server
   * @param roomId - Room ID to join
   */
  async connectToRoom(roomId: string): Promise<void> {
    this.signalingWs = new WebSocket(`ws://localhost:8001/signaling/${roomId}`);

    this.signalingWs.onmessage = async (event) => {
      const message = JSON.parse(event.data);
      await this.handleSignalingMessage(message);
    };
  }

  /**
   * Create peer connection to remote user
   * @param remotePeerId - Remote user ID
   */
  async createPeerConnection(remotePeerId: string): Promise<void> {
    const peerConnection = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' },
      ],
    });

    this.peerConnections.set(remotePeerId, peerConnection);

    // Create data channel for avatar state
    const dataChannel = peerConnection.createDataChannel('avatar', {
      ordered: false,        // Lower latency (allow out-of-order)
      maxRetransmits: 0,     // No retransmissions (drop old packets)
    });

    dataChannel.onopen = () => {
      console.log(`DataChannel opened to ${remotePeerId}`);
      this.dataChannels.set(remotePeerId, dataChannel);
    };

    dataChannel.onmessage = (event) => {
      const transform: AvatarTransform = JSON.parse(event.data);
      this.onRemoteAvatarUpdate(remotePeerId, transform);
    };

    // ICE candidate handling
    peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        this.sendSignalingMessage({
          type: 'ice-candidate',
          targetPeerId: remotePeerId,
          candidate: event.candidate,
        });
      }
    };

    // Create offer
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);

    // Send offer via signaling
    this.sendSignalingMessage({
      type: 'offer',
      targetPeerId: remotePeerId,
      sdp: offer.sdp,
    });
  }

  /**
   * Handle incoming signaling message
   */
  private async handleSignalingMessage(message: any): Promise<void> {
    const { type, sourcePeerId, targetPeerId } = message;

    if (type === 'offer') {
      // Received offer, create answer
      const peerConnection = new RTCPeerConnection(/* ... */);
      this.peerConnections.set(sourcePeerId, peerConnection);

      await peerConnection.setRemoteDescription({
        type: 'offer',
        sdp: message.sdp,
      });

      const answer = await peerConnection.createAnswer();
      await peerConnection.setLocalDescription(answer);

      this.sendSignalingMessage({
        type: 'answer',
        targetPeerId: sourcePeerId,
        sdp: answer.sdp,
      });

    } else if (type === 'answer') {
      // Received answer, set remote description
      const peerConnection = this.peerConnections.get(sourcePeerId);
      if (peerConnection) {
        await peerConnection.setRemoteDescription({
          type: 'answer',
          sdp: message.sdp,
        });
      }

    } else if (type === 'ice-candidate') {
      // Received ICE candidate
      const peerConnection = this.peerConnections.get(sourcePeerId);
      if (peerConnection) {
        await peerConnection.addIceCandidate(message.candidate);
      }
    }
  }

  /**
   * Broadcast local avatar transform to all peers
   */
  broadcastTransform(transform: AvatarTransform): void {
    const message = JSON.stringify(transform);

    this.dataChannels.forEach((channel, peerId) => {
      if (channel.readyState === 'open') {
        channel.send(message);
      }
    });
  }

  private sendSignalingMessage(message: any): void {
    if (this.signalingWs && this.signalingWs.readyState === WebSocket.OPEN) {
      this.signalingWs.send(JSON.stringify(message));
    }
  }

  private onRemoteAvatarUpdate(peerId: string, transform: AvatarTransform): void {
    // Update remote avatar (position, rotation, pose)
    // Implemented in AvatarManager
  }
}
```

**Avatar Transform Message**:

```typescript
interface AvatarTransform {
  userId: string;
  timestamp: number;
  position: [number, number, number];
  rotation: [number, number, number, number];  // Quaternion
  pose?: {
    keypoints: Array<{ x: number; y: number; z: number }>;
  };
}
```

### Colocated Mode (Experimental - Quest Only)

**Goal**: Multiple users in **same physical space** see each other's avatars at correct positions.

**Approach**: WebXR Shared Anchors API (currently experimental).

```typescript
// Create shared anchor (Quest 3+ experimental feature)
const sharedAnchor = await session.requestAnchor(
  anchorPose,
  xrFrame.referenceSpace,
  { isShared: true }  // Experimental
);

// Share anchor UUID via network
const anchorUuid = sharedAnchor.anchorUuid;

// Other users restore anchor by UUID
const restoredAnchor = await session.restoreAnchor(anchorUuid);
```

**Fallback**: Use marker detection (ArUco/QR) to establish shared coordinate system.

---

## 8. Interaction Systems

### Gesture Detection

Map hand poses to avatar gestures (e.g., wave, thumbs up, peace sign).

```typescript
enum Gesture {
  NEUTRAL = 'neutral',
  WAVE = 'wave',
  THUMBS_UP = 'thumbs_up',
  PEACE = 'peace',
  POINT = 'point',
  FIST = 'fist',
}

class GestureDetector {
  detectGesture(handPose: HandPose): Gesture {
    // Wave: rapid horizontal wrist movement
    if (this.isWaving(handPose)) {
      return Gesture.WAVE;
    }

    // Thumbs up: thumb extended, fingers curled
    if (this.isThumbsUp(handPose)) {
      return Gesture.THUMBS_UP;
    }

    // Peace: index + middle extended, others curled
    if (this.isPeace(handPose)) {
      return Gesture.PEACE;
    }

    // Point: index extended, others curled
    if (this.isPointing(handPose)) {
      return Gesture.POINT;
    }

    // Fist: all fingers curled
    if (this.isFist(handPose)) {
      return Gesture.FIST;
    }

    return Gesture.NEUTRAL;
  }

  private isThumbsUp(handPose: HandPose): boolean {
    // Thumb tip above thumb base (extended)
    const thumbExtended = handPose.landmarks[4].y < handPose.landmarks[2].y;

    // Other fingers curled (tip below base)
    const fingersCurled = [8, 12, 16, 20].every((tipIdx) => {
      const baseIdx = tipIdx - 2;
      return handPose.landmarks[tipIdx].y > handPose.landmarks[baseIdx].y;
    });

    return thumbExtended && fingersCurled;
  }

  // ... (other gesture detection methods)
}
```

### Gesture-to-Animation Mapping

Trigger VRM animations based on detected gestures:

```typescript
class GestureAnimator {
  private currentGesture: Gesture = Gesture.NEUTRAL;
  private animations: Map<Gesture, THREE.AnimationClip> = new Map();

  playGestureAnimation(
    vrm: VRM,
    gesture: Gesture
  ): void {
    if (gesture === this.currentGesture) return;

    this.currentGesture = gesture;

    // Get animation clip
    const clip = this.animations.get(gesture);
    if (!clip) return;

    // Play animation
    const mixer = new THREE.AnimationMixer(vrm.scene);
    const action = mixer.clipAction(clip);
    action.reset();
    action.play();
  }

  /**
   * Load gesture animations from VRM
   */
  loadAnimations(vrm: VRM): void {
    // VRM 1.0 animations are stored in vrm.expressionManager
    // Custom animations can be loaded from external files

    // Example: Wave animation (procedural)
    const waveClip = this.createWaveAnimation(vrm);
    this.animations.set(Gesture.WAVE, waveClip);
  }

  private createWaveAnimation(vrm: VRM): THREE.AnimationClip {
    const times = [0, 0.5, 1.0, 1.5, 2.0];
    const values = [
      0, 0, 0,           // 0s: neutral
      0, 0, Math.PI/4,   // 0.5s: rotate hand 45°
      0, 0, 0,           // 1.0s: neutral
      0, 0, Math.PI/4,   // 1.5s: rotate hand 45°
      0, 0, 0,           // 2.0s: neutral
    ];

    const track = new THREE.QuaternionKeyframeTrack(
      '.bones[leftHand].quaternion',
      times,
      values
    );

    return new THREE.AnimationClip('wave', 2.0, [track]);
  }
}
```

### Raycasting (Point & Click)

Select avatars by pointing or clicking:

```typescript
class AvatarRaycaster {
  private raycaster = new THREE.Raycaster();

  /**
   * Get avatar at mouse/pointer position
   * @param pointer - Normalized pointer coords (-1 to 1)
   * @param camera - Active camera
   * @param avatars - All avatars in scene
   * @returns Avatar under pointer or null
   */
  getAvatarAtPointer(
    pointer: { x: number; y: number },
    camera: THREE.Camera,
    avatars: THREE.Object3D[]
  ): THREE.Object3D | null {
    this.raycaster.setFromCamera(pointer, camera);

    const intersects = this.raycaster.intersectObjects(avatars, true);

    if (intersects.length > 0) {
      // Find root avatar object
      let object = intersects[0].object;
      while (object.parent && !avatars.includes(object)) {
        object = object.parent;
      }
      return object;
    }

    return null;
  }

  /**
   * Raycast from XR controller
   * @param xrInputSource - XR controller
   * @param xrFrame - Current XR frame
   * @param avatars - All avatars
   * @returns Hit avatar or null
   */
  getAvatarFromController(
    xrInputSource: XRInputSource,
    xrFrame: XRFrame,
    avatars: THREE.Object3D[]
  ): THREE.Object3D | null {
    const targetRayPose = xrFrame.getPose(
      xrInputSource.targetRaySpace,
      xrFrame.referenceSpace
    );

    if (!targetRayPose) return null;

    // Set ray from controller
    const origin = new THREE.Vector3(
      targetRayPose.transform.position.x,
      targetRayPose.transform.position.y,
      targetRayPose.transform.position.z
    );

    const direction = new THREE.Vector3(0, 0, -1)
      .applyQuaternion(
        new THREE.Quaternion(
          targetRayPose.transform.orientation.x,
          targetRayPose.transform.orientation.y,
          targetRayPose.transform.orientation.z,
          targetRayPose.transform.orientation.w
        )
      );

    this.raycaster.set(origin, direction);

    const intersects = this.raycaster.intersectObjects(avatars, true);

    if (intersects.length > 0) {
      let object = intersects[0].object;
      while (object.parent && !avatars.includes(object)) {
        object = object.parent;
      }
      return object;
    }

    return null;
  }
}
```

### Haptic Feedback

Provide tactile feedback on avatar interactions:

```typescript
class HapticFeedbackManager {
  /**
   * Trigger haptic pulse on XR controller
   * @param xrInputSource - XR controller
   * @param intensity - Vibration intensity (0-1)
   * @param duration - Duration in milliseconds
   */
  triggerPulse(
    xrInputSource: XRInputSource,
    intensity: number = 0.5,
    duration: number = 100
  ): void {
    const gamepad = xrInputSource.gamepad;
    if (!gamepad || !gamepad.hapticActuators || gamepad.hapticActuators.length === 0) {
      return;
    }

    const actuator = gamepad.hapticActuators[0];

    actuator.pulse(intensity, duration);
  }

  /**
   * Trigger feedback when selecting avatar
   */
  onAvatarSelect(xrInputSource: XRInputSource): void {
    this.triggerPulse(xrInputSource, 0.7, 50);  // Strong, short pulse
  }

  /**
   * Trigger feedback when avatar speaks (future voice integration)
   */
  onAvatarSpeak(xrInputSource: XRInputSource): void {
    this.triggerPulse(xrInputSource, 0.3, 200);  // Weak, long pulse
  }
}
```

---

## 9. Performance Optimization

### Target Performance

| Metric | Target | Platform |
|--------|--------|----------|
| **Frame Rate** | 60 FPS | Desktop |
| **Frame Rate** | 72 FPS | Quest 2/3 |
| **Frame Time** | <16.6ms | Desktop |
| **Frame Time** | <13.9ms | Quest 2/3 |
| **Memory** | <500MB | 4 avatars |
| **Network Latency** | <50ms | Avatar sync |

### LOD System Implementation

```typescript
class LODManager {
  private lodLevels: Map<string, {
    high: THREE.Mesh;
    medium: THREE.Mesh;
    low: THREE.Mesh;
  }> = new Map();

  /**
   * Create LOD meshes from original
   * @param avatarId - Avatar ID
   * @param original - Original high-poly mesh
   */
  createLODMeshes(avatarId: string, original: THREE.Mesh): void {
    const simplifier = new SimplifyModifier();

    // High detail (original)
    const high = original.clone();

    // Medium detail (50% polygons)
    const mediumGeometry = original.geometry.clone();
    simplifier.modify(
      mediumGeometry,
      Math.floor(mediumGeometry.attributes.position.count * 0.5)
    );
    const medium = new THREE.Mesh(mediumGeometry, original.material);

    // Low detail (25% polygons)
    const lowGeometry = original.geometry.clone();
    simplifier.modify(
      lowGeometry,
      Math.floor(lowGeometry.attributes.position.count * 0.25)
    );
    const low = new THREE.Mesh(lowGeometry, original.material);

    this.lodLevels.set(avatarId, { high, medium, low });
  }

  /**
   * Update LOD based on camera distance
   * @param camera - Active camera
   */
  updateLODs(camera: THREE.Camera): void {
    this.lodLevels.forEach((lods, avatarId) => {
      const avatar = this.getAvatar(avatarId);
      if (!avatar) return;

      const distance = camera.position.distanceTo(avatar.position);

      let activeMesh: THREE.Mesh;
      if (distance < 3) {
        activeMesh = lods.high;
      } else if (distance < 10) {
        activeMesh = lods.medium;
      } else {
        activeMesh = lods.low;
      }

      // Swap mesh (hide others, show active)
      lods.high.visible = activeMesh === lods.high;
      lods.medium.visible = activeMesh === lods.medium;
      lods.low.visible = activeMesh === lods.low;
    });
  }
}
```

### Occlusion Culling

Don't render avatars behind walls or outside camera view:

```typescript
class OcclusionCuller {
  private frustum = new THREE.Frustum();
  private cameraViewProjectionMatrix = new THREE.Matrix4();

  /**
   * Update frustum from camera
   */
  updateFrustum(camera: THREE.Camera): void {
    camera.updateMatrixWorld();
    this.cameraViewProjectionMatrix.multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
    this.frustum.setFromProjectionMatrix(this.cameraViewProjectionMatrix);
  }

  /**
   * Check if avatar is visible
   * @param avatar - Avatar object
   * @returns True if visible (in frustum), false otherwise
   */
  isVisible(avatar: THREE.Object3D): boolean {
    // Bounding sphere check (fast)
    avatar.updateMatrixWorld();
    const boundingSphere = new THREE.Sphere();
    avatar.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry.computeBoundingSphere();
        if (child.geometry.boundingSphere) {
          boundingSphere.union(child.geometry.boundingSphere);
        }
      }
    });

    return this.frustum.intersectsSphere(boundingSphere);
  }

  /**
   * Cull avatars (set visible=false for offscreen)
   * @param avatars - All avatars
   * @param camera - Active camera
   */
  cullAvatars(avatars: THREE.Object3D[], camera: THREE.Camera): void {
    this.updateFrustum(camera);

    avatars.forEach((avatar) => {
      avatar.visible = this.isVisible(avatar);
    });
  }
}
```

### Texture Atlasing

Combine textures to reduce draw calls:

```typescript
class TextureAtlaser {
  /**
   * Create texture atlas from individual textures
   * @param textures - Array of texture URLs
   * @param atlasSize - Atlas dimensions (default: 2048x2048)
   * @returns Atlas texture + UV offsets for each texture
   */
  async createAtlas(
    textures: string[],
    atlasSize: number = 2048
  ): Promise<{
    atlas: THREE.Texture;
    uvOffsets: Array<{ u: number; v: number; width: number; height: number }>;
  }> {
    const canvas = document.createElement('canvas');
    canvas.width = atlasSize;
    canvas.height = atlasSize;
    const ctx = canvas.getContext('2d')!;

    const uvOffsets: Array<{ u: number; v: number; width: number; height: number }> = [];
    let x = 0;
    let y = 0;
    let rowHeight = 0;

    for (const textureUrl of textures) {
      const img = await this.loadImage(textureUrl);

      // Place texture in atlas
      if (x + img.width > atlasSize) {
        // New row
        x = 0;
        y += rowHeight;
        rowHeight = 0;
      }

      ctx.drawImage(img, x, y);

      // Store UV offset
      uvOffsets.push({
        u: x / atlasSize,
        v: y / atlasSize,
        width: img.width / atlasSize,
        height: img.height / atlasSize,
      });

      x += img.width;
      rowHeight = Math.max(rowHeight, img.height);
    }

    const atlas = new THREE.CanvasTexture(canvas);

    return { atlas, uvOffsets };
  }

  private loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  }
}
```

### Animation Compression

Reduce keyframe data size for network transmission:

```typescript
class AnimationCompressor {
  /**
   * Quantize keyframe positions/rotations to reduce size
   * @param keyframes - Array of keyframe data
   * @param precision - Decimal places (default: 3)
   * @returns Compressed keyframes
   */
  compressKeyframes(
    keyframes: Array<{ time: number; value: number[] }>,
    precision: number = 3
  ): Array<{ time: number; value: number[] }> {
    return keyframes.map((kf) => ({
      time: Math.round(kf.time * 1000) / 1000,  // 1ms precision
      value: kf.value.map((v) => Math.round(v * 10 ** precision) / 10 ** precision),
    }));
  }

  /**
   * Decimate keyframes (remove redundant frames)
   * @param keyframes - Original keyframes
   * @param threshold - Position/rotation change threshold
   * @returns Decimated keyframes
   */
  decimateKeyframes(
    keyframes: Array<{ time: number; value: number[] }>,
    threshold: number = 0.001
  ): Array<{ time: number; value: number[] }> {
    if (keyframes.length <= 2) return keyframes;

    const decimated = [keyframes[0]];  // Always keep first

    for (let i = 1; i < keyframes.length - 1; i++) {
      const prev = keyframes[i - 1];
      const curr = keyframes[i];
      const next = keyframes[i + 1];

      // Check if current frame is significantly different from interpolation
      const interpolated = prev.value.map((v, j) =>
        v + (next.value[j] - v) * ((curr.time - prev.time) / (next.time - prev.time))
      );

      const diff = curr.value.reduce((sum, v, j) => sum + Math.abs(v - interpolated[j]), 0);

      if (diff > threshold) {
        decimated.push(curr);  // Keyframe is important
      }
    }

    decimated.push(keyframes[keyframes.length - 1]);  // Always keep last

    return decimated;
  }
}
```

---

## 10. Backend Infrastructure

### Avatar State Synchronization Server

**Goal**: Relay avatar transforms when WebRTC P2P fails.

```python
# HoloLoom/server/avatar_state_sync.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio

app = FastAPI()

# Room state: room_id -> list of (user_id, websocket)
rooms: Dict[str, List[tuple[str, WebSocket]]] = {}

@app.websocket("/avatar-sync/{room_id}/{user_id}")
async def avatar_sync_endpoint(websocket: WebSocket, room_id: str, user_id: str):
    await websocket.accept()

    # Join room
    if room_id not in rooms:
        rooms[room_id] = []
    rooms[room_id].append((user_id, websocket))

    # Notify others of new user
    await broadcast_to_room(room_id, user_id, {
        "type": "user-joined",
        "userId": user_id,
    })

    try:
        while True:
            # Receive avatar transform
            data = await websocket.receive_json()

            # Add sender user ID
            data["userId"] = user_id

            # Broadcast to all others in room
            await broadcast_to_room(room_id, user_id, data)

    except WebSocketDisconnect:
        # Remove from room
        rooms[room_id] = [(uid, ws) for uid, ws in rooms[room_id] if uid != user_id]
        if not rooms[room_id]:
            del rooms[room_id]

        # Notify others
        await broadcast_to_room(room_id, None, {
            "type": "user-left",
            "userId": user_id,
        })

async def broadcast_to_room(room_id: str, exclude_user_id: str | None, message: dict):
    """Send message to all users in room except excluded user"""
    if room_id not in rooms:
        return

    for user_id, websocket in rooms[room_id]:
        if user_id != exclude_user_id:
            try:
                await websocket.send_json(message)
            except:
                pass  # Ignore send errors
```

### Avatar CDN Management

**Goal**: Host avatar models on CDN for fast loading.

```python
# HoloLoom/server/cdn_management.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import hashlib
from pathlib import Path

app = FastAPI()

# CDN storage directory
CDN_DIR = Path("./cdn/avatars")
CDN_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/avatars/upload")
async def upload_avatar(
    file: UploadFile = File(...),
    user_id: str = None
):
    """
    Upload avatar model (VRM/GLB)
    Returns CDN URL
    """
    # Generate content hash for deduplication
    contents = await file.read()
    file_hash = hashlib.sha256(contents).hexdigest()

    # File extension
    ext = Path(file.filename).suffix

    # Save to CDN
    file_path = CDN_DIR / f"{file_hash}{ext}"

    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(contents)

    # Return CDN URL
    cdn_url = f"http://localhost:8001/cdn/avatars/{file_hash}{ext}"

    return {
        "url": cdn_url,
        "hash": file_hash,
        "size": len(contents),
    }

@app.get("/cdn/avatars/{filename}")
async def get_avatar(filename: str):
    """
    Serve avatar file from CDN
    """
    file_path = CDN_DIR / filename

    if not file_path.exists():
        return {"error": "Avatar not found"}, 404

    return FileResponse(file_path)

@app.delete("/avatars/{file_hash}")
async def delete_avatar(file_hash: str):
    """
    Delete avatar from CDN (admin only in production)
    """
    files = list(CDN_DIR.glob(f"{file_hash}.*"))

    for file_path in files:
        file_path.unlink()

    return {"deleted": len(files)}
```

### Avatar Analytics

Track usage metrics for optimization:

```python
# HoloLoom/server/avatar_analytics.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import json
from datetime import datetime

app = FastAPI()

# Metrics storage (use Redis/Prometheus in production)
metrics: Dict[str, list] = {
    "frame_times": [],
    "network_latency": [],
    "avatar_loads": [],
}

class PerformanceMetric(BaseModel):
    user_id: str
    avatar_id: str
    metric_type: str  # "frame_time", "network_latency", "load_time"
    value: float
    timestamp: str

@app.post("/metrics/report")
async def report_metric(metric: PerformanceMetric):
    """
    Report performance metric from client
    """
    if metric.metric_type not in metrics:
        metrics[metric.metric_type] = []

    metrics[metric.metric_type].append({
        "user_id": metric.user_id,
        "avatar_id": metric.avatar_id,
        "value": metric.value,
        "timestamp": metric.timestamp,
    })

    # Keep last 10,000 metrics
    if len(metrics[metric.metric_type]) > 10000:
        metrics[metric.metric_type] = metrics[metric.metric_type][-10000:]

    return {"status": "recorded"}

@app.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get aggregated metrics
    """
    summary = {}

    for metric_type, data in metrics.items():
        if not data:
            continue

        values = [m["value"] for m in data]
        summary[metric_type] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p50": sorted(values)[len(values) // 2],
            "p95": sorted(values)[int(len(values) * 0.95)],
            "p99": sorted(values)[int(len(values) * 0.99)],
        }

    return summary
```

---

## 11. Implementation Phases

### Phase 6.1: Core Avatar System (Weeks 1-4)

**Goal**: Single-user avatar with pose-driven animation.

**Tasks**:

1. **Setup Dependencies** (Day 1)
   - Install `@pixiv/three-vrm` (^2.0.0)
   - Install `cannon-es` (^0.20.0) for physics
   - Update TypeScript config for Three.js types

2. **VRM Loader** (Days 2-3)
   - Implement `VRMLoader.ts` (~300 lines)
   - Load VRM 1.0 models
   - Parse humanoid rig
   - Test with Ready Player Me avatar

3. **Skeleton Mapper** (Days 4-7)
   - Implement `SkeletonMapper.ts` (~500 lines)
   - Map MediaPipe 33 keypoints → VRM bones
   - Calculate bone rotations from directional vectors
   - Handle missing keypoints (interpolation)

4. **IK Solvers** (Days 8-12)
   - Implement `FABRIKSolver.ts` (~250 lines)
   - Implement `CCDIKSolver.ts` (~200 lines)
   - Implement `TwoBoneIK.ts` (~150 lines)
   - Unit tests for convergence and accuracy

5. **Joint Constraints** (Days 13-14)
   - Implement `JointConstraints.ts` (~300 lines)
   - Define anatomical limits for all 54 bones
   - Apply constraints after IK solve

6. **Motion Smoothing** (Days 15-16)
   - Implement `MotionSmoother.ts` (~200 lines)
   - Exponential filter for keypoint smoothing
   - Kalman filter for prediction (optional)

7. **Avatar Component** (Days 17-20)
   - Implement `Avatar.tsx` (~400 lines)
   - Integrate VRM loader + skeleton mapper + IK
   - Test with Phase 5 pose estimation
   - Performance testing (60 FPS with 1 avatar)

**Deliverables**:
- Working single-user avatar with pose-driven animation
- 60 FPS on desktop
- Unit tests for IK solvers (accuracy <1cm error)

---

### Phase 6.2: Segmentation & Multi-User (Weeks 5-8)

**Goal**: Person segmentation compositing + 2-4 user multi-user sync.

**Tasks**:

1. **Avatar Compositor** (Days 21-24)
   - Implement `AvatarCompositor.ts` (~450 lines)
   - Alpha mask generation from BodyPix
   - Mask refinement (erode/dilate/blur)
   - Compositing with background

2. **WebRTC Manager** (Days 25-28)
   - Implement `WebRTCManager.ts` (~400 lines)
   - Peer connection establishment
   - DataChannel for avatar state (<50ms latency)

3. **Signaling Server** (Days 29-30)
   - Implement `signaling_server.py` (~400 lines)
   - WebSocket signaling for WebRTC
   - Room management

4. **Spatial Anchor Manager** (Days 31-34)
   - Implement `SpatialAnchorManager.ts` (~300 lines)
   - WebXR anchor creation
   - SLAM-based positioning

5. **Multi-User Avatar Scene** (Days 35-40)
   - Implement `MultiUserAvatarScene.tsx` (~500 lines)
   - Integrate WebRTC sync
   - Render 2-4 remote avatars
   - Test latency (<50ms)

**Deliverables**:
- Multi-user avatar sync (2-4 users)
- Person segmentation compositing
- <50ms network latency
- 60 FPS with 4 avatars

---

### Phase 6.3: Optimization & Polish (Weeks 9-12)

**Goal**: Production-ready with LOD, physics, gestures.

**Tasks**:

1. **LOD Manager** (Days 41-45)
   - Implement `LODManager.ts` (~300 lines)
   - 3-level LOD (high/medium/low)
   - Mesh decimation
   - Distance-based switching

2. **Occlusion Culling** (Days 46-48)
   - Implement `OcclusionCuller.ts` (~200 lines)
   - Frustum culling
   - Offscreen avatar hiding

3. **Physics World** (Days 49-52)
   - Implement `PhysicsWorld.ts` (~400 lines)
   - Spring bones (hair/cloth)
   - Collision detection

4. **Gesture Mapper** (Days 53-56)
   - Implement `GestureMapper.ts` (~350 lines)
   - Detect hand gestures (wave, thumbs up, etc.)
   - Map to VRM animations

5. **Performance Dashboard** (Days 57-60)
   - Avatar analytics integration
   - Real-time FPS/latency monitoring
   - Optimization recommendations

**Deliverables**:
- 72 FPS on Quest 2/3 (with 4 avatars)
- <500MB total memory
- Physics simulation (spring bones)
- Gesture-to-animation mapping

---

## 12. File Structure

```
elle/ar_web_client/src/
├── avatars/                    # NEW - Core avatar system
│   ├── AvatarManager.ts (~400 lines)
│   │   - Lifecycle management for all avatars
│   │   - Position/rotation updates
│   │   - LOD coordination
│   │
│   ├── VRMLoader.ts (~300 lines)
│   │   - Load VRM 1.0 models
│   │   - Parse humanoid rig
│   │   - Ready Player Me integration
│   │
│   ├── SkeletonMapper.ts (~500 lines)
│   │   - MediaPipe 33 keypoints → VRM bones
│   │   - Bone rotation calculation
│   │   - Handle missing keypoints
│   │
│   ├── ik/
│   │   ├── FABRIKSolver.ts (~250 lines)
│   │   │   - Forward And Backward Reaching IK
│   │   │   - For spine and arms
│   │   │
│   │   ├── CCDIKSolver.ts (~200 lines)
│   │   │   - Cyclic Coordinate Descent IK
│   │   │   - For fingers and head look-at
│   │   │
│   │   └── TwoBoneIK.ts (~150 lines)
│   │       - Analytical 2-bone IK
│   │       - For elbows and knees
│   │
│   ├── constraints/
│   │   └── JointConstraints.ts (~300 lines)
│   │       - Anatomical angle limits
│   │       - Hinge vs ball-socket joints
│   │
│   └── motion/
│       └── MotionSmoother.ts (~200 lines)
│           - Exponential filter
│           - Kalman prediction (optional)
│
├── compositing/                # NEW - Segmentation & alpha
│   ├── AvatarCompositor.ts (~450 lines)
│   │   - Main compositing pipeline
│   │   - Shader-based blending
│   │
│   ├── AlphaCompositor.ts (~300 lines)
│   │   - Generate alpha masks from BodyPix
│   │   - Premultiplied alpha blending
│   │
│   └── MaskRefiner.ts (~250 lines)
│       - Erode/dilate/blur
│       - Edge smoothing
│
├── multiplayer/                # NEW - Multi-user sync
│   ├── MultiUserSync.ts (~500 lines)
│   │   - Orchestrates WebRTC + signaling
│   │   - Room management
│   │
│   ├── WebRTCManager.ts (~400 lines)
│   │   - Peer connections
│   │   - DataChannel for avatar state
│   │   - ICE candidate handling
│   │
│   ├── SpatialAnchorManager.ts (~300 lines)
│   │   - WebXR anchor creation
│   │   - Shared coordinate system
│   │
│   └── AvatarPositionManager.ts (~350 lines)
│       - SLAM-based positioning
│       - World-space transforms
│
├── optimization/               # NEW - Performance
│   ├── LODManager.ts (~300 lines)
│   │   - 3-level LOD (high/medium/low)
│   │   - Mesh decimation
│   │   - Distance-based switching
│   │
│   ├── MeshDecimator.ts (~250 lines)
│   │   - Polygon reduction
│   │   - SimplifyModifier integration
│   │
│   └── OcclusionCuller.ts (~200 lines)
│       - Frustum culling
│       - Visibility testing
│
├── physics/                    # NEW - Physics simulation
│   ├── PhysicsWorld.ts (~400 lines)
│   │   - cannon-es integration
│   │   - Spring bone simulation
│   │
│   └── CollisionDetector.ts (~300 lines)
│       - Avatar-avatar collisions
│       - Avatar-environment collisions
│
├── interactions/               # NEW - Gestures & input
│   ├── GestureMapper.ts (~350 lines)
│   │   - Detect hand gestures
│   │   - Map to VRM animations
│   │
│   ├── Raycaster.ts (~250 lines)
│   │   - Point & click selection
│   │   - XR controller raycasting
│   │
│   └── HapticFeedback.ts (~150 lines)
│       - WebXR haptics API
│       - Vibration on interactions
│
└── components/
    ├── Avatar.tsx (~400 lines)
    │   - Main Avatar React component
    │   - Integrates all systems
    │   - useFrame for animation loop
    │
    └── MultiUserAvatarScene.tsx (~500 lines)
        - Full multi-user scene
        - WebRTC sync integration
        - Render 2-4 remote avatars

HoloLoom/server/
├── avatar_api.py (~600 lines)
│   - FastAPI endpoints for avatar operations
│   - Upload/download avatars
│   - Metadata management
│
├── signaling_server.py (~400 lines)
│   - WebSocket signaling for WebRTC
│   - Room management
│   - Peer discovery
│
├── avatar_state_sync.py (~300 lines)
│   - Fallback server relay (when WebRTC P2P fails)
│   - Broadcast avatar transforms
│
├── avatar_analytics.py (~400 lines)
│   - Performance metrics collection
│   - FPS, latency, load time tracking
│   - Aggregated statistics
│
└── cdn_management.py (~300 lines)
    - Avatar model hosting
    - Content-addressable storage
    - Deduplication

HoloLoom/vision/
├── avatar_integration.py (~500 lines)
│   - Integration helpers for Phase 5 → Phase 6
│   - Pose → VRM bone mapping (Python side)
│   - Segmentation → alpha mask conversion
│
└── tests/
    └── test_avatar_integration.py (~400 lines)
        - End-to-end tests
        - Pose → skeleton → render pipeline

**Total New Code**: ~13,700 lines (TypeScript) + ~2,000 lines (Python)
```

---

## 13. Integration with Phase 5

### Pose Estimation Integration

```typescript
import { getPoseEstimationService } from '../services';
import { SkeletonMapper } from '../avatars/SkeletonMapper';
import { VRM } from '@pixiv/three-vrm';

async function updateAvatarFromPose(
  videoElement: HTMLVideoElement,
  vrm: VRM
): Promise<void> {
  // 1. Get pose from Phase 5
  const poseService = getPoseEstimationService();
  const pose = await poseService.processFrame(videoElement);

  if (!pose) return;

  // 2. Map pose to VRM skeleton
  const skeletonMapper = new SkeletonMapper();
  skeletonMapper.updateSkeleton(vrm.humanoid, pose);
}
```

### Segmentation Integration

```typescript
import { getSemanticSegmentationService } from '../services';
import { AlphaMaskGenerator } from '../compositing/AlphaMaskGenerator';
import { MaskRefiner } from '../compositing/MaskRefiner';

async function getPersonAlphaMask(
  videoElement: HTMLVideoElement
): Promise<ImageData> {
  // 1. Get segmentation from Phase 5
  const segService = getSemanticSegmentationService();
  const segmentation = await segService.segmentImage(videoElement);

  // 2. Generate alpha mask
  const maskGenerator = new AlphaMaskGenerator();
  const rawMask = maskGenerator.generateMask(segmentation, 15);  // Person class

  // 3. Refine mask
  const refiner = new MaskRefiner();
  const refinedMask = refiner.refine(rawMask);

  return refinedMask;
}
```

### SLAM Integration

```typescript
import { getSLAMService } from '../services';
import { AvatarPositionManager } from '../multiplayer/AvatarPositionManager';

async function updateAvatarPosition(
  videoElement: HTMLVideoElement,
  xrFrame: XRFrame,
  userId: string
): Promise<void> {
  // 1. Get SLAM pose from Phase 5
  const slamService = getSLAMService();
  const slamPose = await slamService.processFrame(videoElement, xrFrame);

  if (!slamPose) return;

  // 2. Update avatar position
  const positionManager = new AvatarPositionManager();
  positionManager.updateFromSLAM(userId, slamPose);
}
```

### Complete Pipeline Example

```typescript
import { useFrame } from '@react-three/fiber';
import { getPoseEstimationService, getSemanticSegmentationService, getSLAMService } from '../services';
import { Avatar } from '../components/Avatar';

export function ARScene() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [pose, setPose] = useState<BodyPose | null>(null);
  const [alphaMask, setAlphaMask] = useState<ImageData | null>(null);
  const [avatarPosition, setAvatarPosition] = useState<[number, number, number]>([0, 0, 0]);
  const [avatarRotation, setAvatarRotation] = useState<[number, number, number, number]>([0, 0, 0, 1]);

  // Update loop (every frame)
  useFrame(async (state, delta) => {
    if (!videoRef.current) return;

    // 1. Update pose
    const poseService = getPoseEstimationService();
    const newPose = await poseService.processFrame(videoRef.current);
    if (newPose) setPose(newPose);

    // 2. Update segmentation (every 5 frames for performance)
    if (state.frameloop.count % 5 === 0) {
      const segService = getSemanticSegmentationService();
      const segmentation = await segService.segmentImage(videoRef.current);

      const maskGenerator = new AlphaMaskGenerator();
      const refiner = new MaskRefiner();
      const mask = refiner.refine(maskGenerator.generateMask(segmentation));
      setAlphaMask(mask);
    }

    // 3. Update position from SLAM (in XR mode)
    const xrFrame = state.gl.xr.getFrame();
    if (xrFrame) {
      const slamService = getSLAMService();
      const slamPose = await slamService.processFrame(videoRef.current, xrFrame);

      if (slamPose) {
        setAvatarPosition(slamPose.position);
        setAvatarRotation(slamPose.orientation);
      }
    }
  });

  return (
    <>
      <video ref={videoRef} autoPlay style={{ display: 'none' }} />

      <Avatar
        url="/avatars/default.vrm"
        pose={pose}
        position={avatarPosition}
        rotation={avatarRotation}
        userId="local"
      />
    </>
  );
}
```

---

## 14. Testing Strategy

### Unit Tests

**IK Solver Tests** (`ik/tests/test_ik_solvers.ts`):

```typescript
import { FABRIKSolver } from '../ik/FABRIKSolver';
import { CCDIKSolver } from '../ik/CCDIKSolver';
import { TwoBoneIK } from '../ik/TwoBoneIK';
import * as THREE from 'three';

describe('FABRIK Solver', () => {
  test('should converge to target within tolerance', () => {
    const solver = new FABRIKSolver();

    const joints = [
      new THREE.Vector3(0, 0, 0),    // Root
      new THREE.Vector3(0, 1, 0),    // Joint 1
      new THREE.Vector3(0, 2, 0),    // Joint 2
      new THREE.Vector3(0, 3, 0),    // End effector
    ];

    const target = new THREE.Vector3(1, 2, 0);

    const solved = solver.solve(joints, target, 0.01, 10);

    // Check convergence
    const distance = solved[solved.length - 1].distanceTo(target);
    expect(distance).toBeLessThan(0.01);

    // Check bone lengths preserved
    for (let i = 0; i < solved.length - 1; i++) {
      const originalLength = joints[i].distanceTo(joints[i + 1]);
      const solvedLength = solved[i].distanceTo(solved[i + 1]);
      expect(Math.abs(originalLength - solvedLength)).toBeLessThan(0.001);
    }
  });
});

describe('Two-Bone IK', () => {
  test('should solve elbow position for reachable target', () => {
    const solver = new TwoBoneIK();

    const shoulder = new THREE.Vector3(0, 0, 0);
    const elbow = new THREE.Vector3(0, 1, 0);
    const wrist = new THREE.Vector3(0, 2, 0);
    const target = new THREE.Vector3(1, 1, 0);

    const { mid, end } = solver.solve(shoulder, elbow, wrist, target);

    // Check end effector reaches target
    expect(end.distanceTo(target)).toBeLessThan(0.001);

    // Check bone lengths preserved
    const upperLength = shoulder.distanceTo(elbow);
    const lowerLength = elbow.distanceTo(wrist);
    expect(shoulder.distanceTo(mid)).toBeCloseTo(upperLength, 2);
    expect(mid.distanceTo(end)).toBeCloseTo(lowerLength, 2);
  });
});
```

### Integration Tests

**Pose → Skeleton Pipeline** (`tests/test_avatar_integration.ts`):

```typescript
import { SkeletonMapper } from '../avatars/SkeletonMapper';
import { VRMLoader } from '../avatars/VRMLoader';
import { BodyPose } from '../services/poseEstimation';

describe('Pose to Skeleton Mapping', () => {
  let vrm: VRM;
  let skeletonMapper: SkeletonMapper;

  beforeAll(async () => {
    const loader = new VRMLoader();
    vrm = await loader.load('/test-assets/avatar.vrm');
    skeletonMapper = new SkeletonMapper();
  });

  test('should map T-pose correctly', async () => {
    // Mock T-pose (arms extended)
    const tPose: BodyPose = {
      keypoints: [
        { x: 0, y: 0, z: 0, visibility: 1.0, presence: 1.0 },  // Nose
        // ... (33 keypoints in T-pose configuration)
      ],
    };

    skeletonMapper.updateSkeleton(vrm.humanoid, tPose);

    // Check arm extension
    const leftUpperArm = vrm.humanoid.getBoneNode('leftUpperArm');
    const rightUpperArm = vrm.humanoid.getBoneNode('rightUpperArm');

    expect(leftUpperArm.rotation.z).toBeCloseTo(Math.PI / 2, 1);  // Left arm extended
    expect(rightUpperArm.rotation.z).toBeCloseTo(-Math.PI / 2, 1); // Right arm extended
  });

  test('should handle missing keypoints gracefully', () => {
    const incompletePose: BodyPose = {
      keypoints: new Array(33).fill(null).map((_, i) => ({
        x: 0,
        y: 0,
        z: 0,
        visibility: i < 15 ? 1.0 : 0.0,  // Only first 15 keypoints visible
        presence: i < 15 ? 1.0 : 0.0,
      })),
    };

    expect(() => {
      skeletonMapper.updateSkeleton(vrm.humanoid, incompletePose);
    }).not.toThrow();
  });
});
```

### Performance Tests

**Frame Rate Benchmarks** (`tests/test_performance.ts`):

```typescript
import { Avatar } from '../components/Avatar';
import { render } from '@testing-library/react';
import { Canvas } from '@react-three/fiber';

describe('Performance Benchmarks', () => {
  test('should maintain 60 FPS with 1 avatar', async () => {
    const frameTimes: number[] = [];
    let lastTime = performance.now();

    const { rerender } = render(
      <Canvas>
        <Avatar
          url="/test-assets/avatar.vrm"
          pose={mockPose}
          position={[0, 0, 0]}
          rotation={[0, 0, 0, 1]}
          userId="test"
        />
      </Canvas>
    );

    // Measure 100 frames
    for (let i = 0; i < 100; i++) {
      rerender(
        <Canvas>
          <Avatar
            url="/test-assets/avatar.vrm"
            pose={mockPose}
            position={[0, 0, 0]}
            rotation={[0, 0, 0, 1]}
            userId="test"
          />
        </Canvas>
      );

      const now = performance.now();
      frameTimes.push(now - lastTime);
      lastTime = now;

      await new Promise((resolve) => setTimeout(resolve, 16));  // ~60 FPS
    }

    const avgFrameTime = frameTimes.reduce((a, b) => a + b) / frameTimes.length;
    const fps = 1000 / avgFrameTime;

    expect(fps).toBeGreaterThanOrEqual(60);
  });

  test('should use <500MB memory with 4 avatars', async () => {
    if (!performance.memory) {
      console.warn('performance.memory not available (Chrome only)');
      return;
    }

    const { rerender } = render(
      <Canvas>
        <Avatar url="/test-assets/avatar.vrm" /* ... */ userId="user1" />
        <Avatar url="/test-assets/avatar.vrm" /* ... */ userId="user2" />
        <Avatar url="/test-assets/avatar.vrm" /* ... */ userId="user3" />
        <Avatar url="/test-assets/avatar.vrm" /* ... */ userId="user4" />
      </Canvas>
    );

    const memoryUsage = performance.memory.usedJSHeapSize / (1024 * 1024);  // MB

    expect(memoryUsage).toBeLessThan(500);
  });
});
```

### End-to-End Tests

**Multi-User Sync** (`tests/e2e/test_multiuser.ts`):

```typescript
import { MultiUserSync } from '../multiplayer/MultiUserSync';

describe('Multi-User Sync E2E', () => {
  test('should synchronize 2 avatars with <50ms latency', async () => {
    const user1 = new MultiUserSync('room1', 'user1');
    const user2 = new MultiUserSync('room1', 'user2');

    await user1.connect();
    await user2.connect();

    // Wait for WebRTC connection
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // User 1 sends transform
    const transform = {
      userId: 'user1',
      timestamp: Date.now(),
      position: [1, 0, 0],
      rotation: [0, 0, 0, 1],
    };

    const receivedPromise = new Promise<number>((resolve) => {
      user2.on('avatar-update', (data) => {
        const latency = Date.now() - data.timestamp;
        resolve(latency);
      });
    });

    user1.broadcastTransform(transform);

    const latency = await receivedPromise;

    expect(latency).toBeLessThan(50);  // <50ms latency
  });
});
```

---

## 15. Performance Benchmarks

### Target Metrics

| Metric | Desktop | Quest 2/3 |
|--------|---------|-----------|
| **Frame Rate** | 60 FPS | 72 FPS |
| **Frame Time** | <16.6ms | <13.9ms |
| **Avatar Count** | 8 max | 4 max |
| **Memory Usage** | <1GB | <500MB |
| **Network Latency** | <50ms | <50ms |
| **Pose Update Rate** | 30 Hz | 30 Hz |

### Frame Time Breakdown (Target)

| Stage | Desktop | Quest 2/3 |
|-------|---------|-----------|
| Pose Estimation (Phase 5) | 3-5ms | 5-7ms |
| Skeleton Mapping + IK | 2-3ms | 3-4ms |
| Rendering (R3F) | 8-10ms | 5-6ms |
| Physics (Spring Bones) | 1-2ms | 1-2ms |
| **Total** | **14-20ms** | **14-19ms** |

### Memory Breakdown (4 Avatars)

| Component | Size |
|-----------|------|
| VRM Model (×4) | 40MB |
| Textures (×4) | 120MB |
| Geometry LODs (×4) | 80MB |
| Animation State | 20MB |
| WebRTC Buffers | 10MB |
| Three.js Runtime | 30MB |
| **Total** | **300MB** |

### Network Bandwidth (Per User)

| Data | Rate | Bandwidth |
|------|------|-----------|
| Avatar Transform (16 floats) | 30 Hz | 1.9 KB/s |
| Pose Keypoints (33×3 floats) | 30 Hz | 11.9 KB/s |
| Total (3 remote users) | - | **41.4 KB/s** |

---

## 16. Success Criteria

### Phase 6.1 (Core Avatar System)

✅ **Technical**:
- [ ] VRM loader successfully loads Ready Player Me avatars
- [ ] Skeleton mapping accuracy: <5° joint rotation error
- [ ] IK solvers converge in <10 iterations
- [ ] 60 FPS with 1 avatar on desktop

✅ **Functional**:
- [ ] Avatar animates from MediaPipe pose in real-time
- [ ] Smooth motion (no jitter or stuttering)
- [ ] Natural poses (no anatomically impossible positions)

---

### Phase 6.2 (Segmentation & Multi-User)

✅ **Technical**:
- [ ] Person segmentation compositing with clean alpha edges
- [ ] WebRTC P2P connection established <2 seconds
- [ ] Network latency <50ms for avatar state sync
- [ ] 60 FPS with 4 avatars on desktop

✅ **Functional**:
- [ ] Remote avatars visible and animated correctly
- [ ] Spatial positioning correct (avatars at world positions)
- [ ] Segmentation removes background cleanly

---

### Phase 6.3 (Optimization & Polish)

✅ **Technical**:
- [ ] 72 FPS with 4 avatars on Quest 2/3
- [ ] <500MB total memory usage
- [ ] LOD system switches smoothly (no popping)
- [ ] Occlusion culling improves FPS by >20%

✅ **Functional**:
- [ ] Spring bones animate hair/cloth realistically
- [ ] Gesture detection works for 5+ gestures
- [ ] Haptic feedback on avatar interactions
- [ ] Production deployment successful

---

## Conclusion

Phase 6 represents the **culmination of all Phase 5 vision capabilities**, bringing them together into a unified 3D avatar system. The architecture is designed to be:

- **Extensible**: Modular IK solvers, compositing strategies, networking layers
- **Exhaustive**: Complete coverage of avatar formats, rendering, physics, multi-user sync
- **Production-Ready**: Performance-optimized for 72 FPS on Quest, <500MB memory
- **Deeply Integrated**: Seamless use of Phase 5 pose, segmentation, and SLAM services

**Timeline**: 8-12 weeks (3 sub-phases)
**Complexity**: Advanced (real-time 3D graphics + networking + physics)
**Impact**: Transforms AR from flat overlays to **living, interactive 3D avatars**

---

**Architecture Status**: ✅ **COMPLETE AND APPROVED**
**Next Step**: Begin Phase 6.1 implementation with VRM loader
