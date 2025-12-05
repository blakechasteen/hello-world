/**
 * Avatar - React Three Fiber component for VRM avatars
 *
 * Main avatar component that integrates:
 * - VRM loading (VRMLoader)
 * - Pose-driven animation (SkeletonMapper)
 * - Position/rotation from SLAM
 * - Spring bones physics
 * - Motion smoothing
 *
 * Created: 2025-11-22 (Phase 6.1)
 */

import { useEffect, useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { VRM, VRMUtils } from '@pixiv/three-vrm';
import { VRMLoader } from '../avatars/VRMLoader';
import { SkeletonMapper } from '../avatars/SkeletonMapper';
import { BodyPose } from '../services/poseEstimation';
import { TextureAtlas, AtlasEntry } from '../optimization/TextureAtlas';

/**
 * Avatar component props
 */
export interface AvatarProps {
  /**
   * URL to VRM or GLB avatar file
   */
  url: string;

  /**
   * Current body pose from MediaPipe (Phase 5)
   * If null, avatar will be in default T-pose
   */
  pose: BodyPose | null;

  /**
   * Avatar position in world space [x, y, z]
   * From SLAM or manual placement
   */
  position: [number, number, number];

  /**
   * Avatar rotation as quaternion [x, y, z, w]
   * From SLAM or manual rotation
   */
  rotation: [number, number, number, number];

  /**
   * User ID (for multi-user scenarios)
   */
  userId: string;

  /**
   * Enable spring bones physics (default: true)
   * Hair and cloth will react to movement
   */
  enablePhysics?: boolean;

  /**
   * Enable motion smoothing (default: true)
   * Reduces jitter from pose estimation
   */
  enableSmoothing?: boolean;

  /**
   * Smoothing factor (0-1, default: 0.3)
   * Higher = smoother but more lag
   */
  smoothingFactor?: number;

  /**
   * Visibility threshold for pose keypoints (0-1, default: 0.5)
   * Keypoints below this confidence are ignored
   */
  visibilityThreshold?: number;

  /**
   * Scale factor (default: 1.0)
   * Adjust avatar size
   */
  scale?: number;

  /**
   * Enable shadows (default: true)
   */
  castShadow?: boolean;

  /**
   * Receive shadows (default: true)
   */
  receiveShadow?: boolean;

  /**
   * Callback when avatar loads
   */
  onLoad?: (vrm: VRM) => void;

  /**
   * Callback on load error
   */
  onError?: (error: Error) => void;

  /**
   * Optional texture atlas for batching
   * When provided, avatar textures are registered with atlas
   */
  textureAtlas?: TextureAtlas;

  /**
   * Unique ID for this avatar in the texture atlas
   * Used to retrieve UV transforms
   */
  atlasEntryId?: string;

  /**
   * Callback when textures are added to atlas
   * Returns map of material name -> atlas entry ID
   */
  onTexturesRegistered?: (textureIds: Map<string, string>) => void;
}

/**
 * Avatar component
 *
 * Renders a VRM avatar with pose-driven animation.
 *
 * Usage:
 * ```tsx
 * <Avatar
 *   url="/avatars/default.vrm"
 *   pose={currentPose}
 *   position={[0, 0, 0]}
 *   rotation={[0, 0, 0, 1]}
 *   userId="local"
 * />
 * ```
 */
export function Avatar({
  url,
  pose,
  position,
  rotation,
  userId,
  enablePhysics = true,
  enableSmoothing = true,
  smoothingFactor = 0.3,
  visibilityThreshold = 0.5,
  scale = 1.0,
  castShadow = true,
  receiveShadow = true,
  onLoad,
  onError,
  textureAtlas,
  atlasEntryId,
  onTexturesRegistered,
}: AvatarProps) {
  const groupRef = useRef<THREE.Group>(null);
  const vrmRef = useRef<VRM | null>(null);
  const skeletonMapperRef = useRef<SkeletonMapper | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<Error | null>(null);

  // Load VRM model
  useEffect(() => {
    const loader = new VRMLoader({
      enableCache: true,
      autoRotateVRM0: true,
      validateHumanoidRig: true,
    });

    let mounted = true;

    async function loadAvatar() {
      try {
        setLoading(true);
        setLoadError(null);

        const result = await loader.load(url);

        if (!mounted) return;

        const vrm = result.vrm;
        vrmRef.current = vrm;

        // Add VRM scene to group
        if (groupRef.current) {
          groupRef.current.add(vrm.scene);
        }

        // Enable shadows
        vrm.scene.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            object.castShadow = castShadow;
            object.receiveShadow = receiveShadow;

            // Enable frustum culling
            object.frustumCulled = true;
          }
        });

        // Register textures with atlas (if provided)
        if (textureAtlas && atlasEntryId) {
          const textureIds = new Map<string, string>();

          vrm.scene.traverse((object) => {
            if (object instanceof THREE.Mesh) {
              const materials = Array.isArray(object.material)
                ? object.material
                : [object.material];

              materials.forEach((material, index) => {
                if ('map' in material && material.map instanceof THREE.Texture) {
                  // Generate unique ID for this texture
                  const textureId = `${atlasEntryId}_${material.name || index}`;

                  // Add to atlas
                  const entry = textureAtlas.addTexture(textureId, material.map);

                  if (entry) {
                    textureIds.set(material.name || `material_${index}`, textureId);
                  }
                }
              });
            }
          });

          if (onTexturesRegistered) {
            onTexturesRegistered(textureIds);
          }

          console.log(`[Avatar] Registered ${textureIds.size} textures with atlas for ${userId}`);
        }

        // Create skeleton mapper
        skeletonMapperRef.current = new SkeletonMapper({
          visibilityThreshold,
          enableIK: false,  // IK solvers not yet integrated
          smoothingFactor: enableSmoothing ? smoothingFactor : 0,
        });

        console.log(`[Avatar] Loaded ${userId}: ${url} (${result.loadTime.toFixed(1)}ms)`);

        setLoading(false);

        if (onLoad) {
          onLoad(vrm);
        }
      } catch (error) {
        if (!mounted) return;

        console.error(`[Avatar] Failed to load ${userId}:`, error);
        setLoadError(error as Error);
        setLoading(false);

        if (onError) {
          onError(error as Error);
        }
      }
    }

    loadAvatar();

    return () => {
      mounted = false;

      // Cleanup VRM on unmount
      if (vrmRef.current) {
        VRMUtils.deepDispose(vrmRef.current.scene);
        vrmRef.current = null;
      }

      skeletonMapperRef.current = null;
    };
  }, [url, userId]);  // Reload if URL or user ID changes

  // Apply texture atlas UV transforms (when atlas is built)
  useEffect(() => {
    const vrm = vrmRef.current;
    if (!vrm || !textureAtlas || !atlasEntryId) return;

    // Get atlas texture
    const atlasTexture = textureAtlas.getAtlasTexture();

    // Apply atlas texture and UV transforms to all materials
    vrm.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];

        materials.forEach((material, index) => {
          if ('map' in material && material.map instanceof THREE.Texture) {
            // Get texture ID
            const textureId = `${atlasEntryId}_${material.name || index}`;

            // Get atlas entry
            const entry = textureAtlas.getEntry(textureId);

            if (entry) {
              // Replace material texture with atlas
              material.map = atlasTexture;

              // Apply UV offset and scale
              material.map.offset.set(entry.uvOffset.x, entry.uvOffset.y);
              material.map.repeat.set(entry.uvScale.x, entry.uvScale.y);
              material.map.needsUpdate = true;

              // Update material
              material.needsUpdate = true;
            }
          }
        });
      }
    });

    console.log(`[Avatar] Applied texture atlas transforms for ${userId}`);
  }, [textureAtlas, atlasEntryId, userId]);  // Reapply if atlas changes

  // Update skeleton from pose (every frame)
  useFrame((state, delta) => {
    const vrm = vrmRef.current;
    const skeletonMapper = skeletonMapperRef.current;

    if (!vrm || !skeletonMapper) return;

    // Update skeleton from pose
    if (pose) {
      try {
        skeletonMapper.updateSkeleton(vrm.humanoid, pose);
      } catch (error) {
        console.warn(`[Avatar] Skeleton update error for ${userId}:`, error);
      }
    }

    // Update spring bones (hair/cloth physics)
    if (enablePhysics && vrm.springBoneManager) {
      vrm.springBoneManager.update(delta);
    }

    // Update blend shapes (facial expressions - future)
    // if (vrm.expressionManager) {
    //   vrm.expressionManager.update();
    // }
  });

  // Update position/rotation from props (SLAM or manual)
  useEffect(() => {
    if (groupRef.current) {
      groupRef.current.position.set(...position);
      groupRef.current.quaternion.set(...rotation);
      groupRef.current.scale.setScalar(scale);
    }
  }, [position, rotation, scale]);

  // Render loading/error states (optional - could show placeholder)
  if (loadError) {
    return (
      <group ref={groupRef}>
        {/* Error placeholder - red sphere */}
        <mesh position={[0, 1, 0]}>
          <sphereGeometry args={[0.2, 16, 16]} />
          <meshStandardMaterial color="red" />
        </mesh>
      </group>
    );
  }

  if (loading) {
    return (
      <group ref={groupRef}>
        {/* Loading placeholder - gray sphere */}
        <mesh position={[0, 1, 0]}>
          <sphereGeometry args={[0.2, 16, 16]} />
          <meshStandardMaterial color="gray" />
        </mesh>
      </group>
    );
  }

  return (
    <group ref={groupRef}>
      {/* VRM scene added in useEffect */}
    </group>
  );
}

/**
 * AvatarDebugHelper - Show skeleton debug visualization
 *
 * Optional component for debugging skeleton mapping.
 * Renders spheres at each bone position.
 */
export interface AvatarDebugHelperProps {
  vrm: VRM;
  visible?: boolean;
}

export function AvatarDebugHelper({ vrm, visible = true }: AvatarDebugHelperProps) {
  if (!visible) return null;

  // Get all humanoid bones
  const bones = [
    'hips',
    'spine',
    'chest',
    'neck',
    'head',
    'leftShoulder',
    'leftUpperArm',
    'leftLowerArm',
    'leftHand',
    'rightShoulder',
    'rightUpperArm',
    'rightLowerArm',
    'rightHand',
    'leftUpperLeg',
    'leftLowerLeg',
    'leftFoot',
    'rightUpperLeg',
    'rightLowerLeg',
    'rightFoot',
  ] as const;

  return (
    <group>
      {bones.map((boneName) => {
        const bone = vrm.humanoid.getBoneNode(boneName);
        if (!bone) return null;

        // Get world position
        const worldPos = new THREE.Vector3();
        bone.getWorldPosition(worldPos);

        return (
          <mesh key={boneName} position={worldPos}>
            <sphereGeometry args={[0.02, 8, 8]} />
            <meshBasicMaterial color="lime" />
          </mesh>
        );
      })}
    </group>
  );
}

/**
 * AvatarNameTag - Floating name tag above avatar
 *
 * Optional component for multi-user scenarios.
 * Shows username above avatar head.
 */
export interface AvatarNameTagProps {
  vrm: VRM;
  name: string;
  offset?: number;  // Height offset above head (default: 0.3)
  fontSize?: number;  // Font size (default: 0.1)
  color?: string;  // Text color (default: white)
}

export function AvatarNameTag({
  vrm,
  name,
  offset = 0.3,
  fontSize = 0.1,
  color = 'white',
}: AvatarNameTagProps) {
  const headBone = vrm.humanoid.getBoneNode('head');
  if (!headBone) return null;

  // Get head world position
  const headPos = new THREE.Vector3();
  headBone.getWorldPosition(headPos);
  headPos.y += offset;

  return (
    <group position={headPos}>
      {/* Billboard (always face camera) */}
      <sprite scale={[fontSize * name.length * 0.5, fontSize, 1]}>
        <spriteMaterial color={color} transparent opacity={0.8} />
      </sprite>
      {/* TODO: Add text rendering with troika-three-text */}
    </group>
  );
}
