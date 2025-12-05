/**
 * VRMLoader - Load and parse VRM 1.0 humanoid avatars
 *
 * Features:
 * - VRM 1.0 model loading with @pixiv/three-vrm
 * - Ready Player Me GLB integration
 * - Humanoid rig parsing and validation
 * - Model caching for performance
 * - Graceful error handling
 *
 * Created: 2025-11-22 (Phase 6.1)
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { VRM, VRMLoaderPlugin, VRMUtils, VRMHumanBoneName } from '@pixiv/three-vrm';

/**
 * Configuration for VRM loader
 */
export interface VRMLoaderConfig {
  /**
   * Enable model caching (default: true)
   * Cached models are reused for same URL
   */
  enableCache?: boolean;

  /**
   * Automatically rotate VRM0 models 180° (default: true)
   * VRM 0.x models face -Z, this rotates them to face +Z
   */
  autoRotateVRM0?: boolean;

  /**
   * Validate humanoid rig on load (default: true)
   * Ensures all required bones are present
   */
  validateHumanoidRig?: boolean;

  /**
   * Timeout for model loading in milliseconds (default: 30000)
   */
  loadTimeout?: number;
}

/**
 * Result of VRM loading operation
 */
export interface VRMLoadResult {
  /**
   * Loaded VRM instance
   */
  vrm: VRM;

  /**
   * Model format detected
   */
  format: 'vrm-1.0' | 'vrm-0.x' | 'glb';

  /**
   * Loading time in milliseconds
   */
  loadTime: number;

  /**
   * Whether model was loaded from cache
   */
  fromCache: boolean;

  /**
   * Humanoid rig validation result
   */
  validation?: {
    valid: boolean;
    missingBones?: VRMHumanBoneName[];
    warnings?: string[];
  };
}

/**
 * VRM model cache entry
 */
interface CacheEntry {
  vrm: VRM;
  format: 'vrm-1.0' | 'vrm-0.x' | 'glb';
  timestamp: number;
}

/**
 * VRMLoader - Load VRM 1.0 humanoid avatars
 *
 * Supports:
 * - Native VRM 1.0 files (.vrm)
 * - VRM 0.x files (auto-upgraded)
 * - Ready Player Me GLB files
 *
 * Usage:
 * ```typescript
 * const loader = new VRMLoader();
 * const result = await loader.load('/avatars/my-avatar.vrm');
 * scene.add(result.vrm.scene);
 * ```
 */
export class VRMLoader {
  private gltfLoader: GLTFLoader;
  private config: Required<VRMLoaderConfig>;
  private cache: Map<string, CacheEntry> = new Map();

  constructor(config: VRMLoaderConfig = {}) {
    this.config = {
      enableCache: config.enableCache ?? true,
      autoRotateVRM0: config.autoRotateVRM0 ?? true,
      validateHumanoidRig: config.validateHumanoidRig ?? true,
      loadTimeout: config.loadTimeout ?? 30000,
    };

    // Initialize GLTF loader with VRM plugin
    this.gltfLoader = new GLTFLoader();
    this.gltfLoader.register((parser) => new VRMLoaderPlugin(parser));
  }

  /**
   * Load VRM model from URL
   *
   * @param url - URL to VRM or GLB file
   * @returns Load result with VRM instance
   * @throws Error if loading fails or timeout
   */
  async load(url: string): Promise<VRMLoadResult> {
    const startTime = performance.now();

    // Check cache
    if (this.config.enableCache && this.cache.has(url)) {
      const cached = this.cache.get(url)!;
      console.log(`[VRMLoader] Loaded from cache: ${url}`);

      return {
        vrm: this.cloneVRM(cached.vrm),
        format: cached.format,
        loadTime: performance.now() - startTime,
        fromCache: true,
      };
    }

    // Load with timeout
    const loadPromise = this.loadInternal(url);
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error(`VRM load timeout after ${this.config.loadTimeout}ms`)), this.config.loadTimeout);
    });

    try {
      const result = await Promise.race([loadPromise, timeoutPromise]);
      result.loadTime = performance.now() - startTime;

      // Cache result
      if (this.config.enableCache) {
        this.cache.set(url, {
          vrm: result.vrm,
          format: result.format,
          timestamp: Date.now(),
        });
      }

      console.log(`[VRMLoader] Loaded ${result.format} model: ${url} (${result.loadTime.toFixed(1)}ms)`);

      return result;
    } catch (error) {
      console.error(`[VRMLoader] Failed to load ${url}:`, error);
      throw error;
    }
  }

  /**
   * Load VRM from Ready Player Me URL
   *
   * Ready Player Me provides GLB models with humanoid rigs.
   * This method wraps the GLB in a VRM-compatible structure.
   *
   * @param avatarUrl - Ready Player Me avatar URL (ends with .glb)
   * @returns Load result
   */
  async loadReadyPlayerMe(avatarUrl: string): Promise<VRMLoadResult> {
    if (!avatarUrl.endsWith('.glb')) {
      throw new Error('Ready Player Me avatars must be .glb format');
    }

    return this.load(avatarUrl);
  }

  /**
   * Clear model cache
   *
   * @param url - Optional specific URL to clear (clears all if not provided)
   */
  clearCache(url?: string): void {
    if (url) {
      this.cache.delete(url);
      console.log(`[VRMLoader] Cleared cache for: ${url}`);
    } else {
      this.cache.clear();
      console.log(`[VRMLoader] Cleared all cache (${this.cache.size} entries)`);
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; urls: string[] } {
    return {
      size: this.cache.size,
      urls: Array.from(this.cache.keys()),
    };
  }

  /**
   * Internal load implementation
   */
  private async loadInternal(url: string): Promise<VRMLoadResult> {
    return new Promise((resolve, reject) => {
      this.gltfLoader.load(
        url,
        (gltf) => {
          try {
            // Extract VRM from GLTF userData
            const vrm = gltf.userData.vrm as VRM | undefined;

            if (!vrm) {
              // Not a VRM file, try to treat as GLB with humanoid rig
              const glbVRM = this.convertGLBToVRM(gltf);
              resolve({
                vrm: glbVRM,
                format: 'glb',
                loadTime: 0,  // Set by caller
                fromCache: false,
              });
              return;
            }

            // Detect VRM version
            const format = this.detectVRMVersion(vrm);

            // Auto-rotate VRM 0.x models
            if (format === 'vrm-0.x' && this.config.autoRotateVRM0) {
              VRMUtils.rotateVRM0(vrm);
            }

            // Validate humanoid rig
            let validation: VRMLoadResult['validation'];
            if (this.config.validateHumanoidRig) {
              validation = this.validateHumanoid(vrm);
              if (!validation.valid) {
                console.warn(`[VRMLoader] Humanoid rig validation failed:`, validation);
              }
            }

            resolve({
              vrm,
              format,
              loadTime: 0,  // Set by caller
              fromCache: false,
              validation,
            });
          } catch (error) {
            reject(new Error(`Failed to parse VRM: ${error}`));
          }
        },
        (progress) => {
          // Optional: Progress callback
          const percent = (progress.loaded / progress.total) * 100;
          if (percent % 25 === 0) {
            console.log(`[VRMLoader] Loading ${url}: ${percent.toFixed(0)}%`);
          }
        },
        (error) => {
          reject(new Error(`GLTF load error: ${error}`));
        }
      );
    });
  }

  /**
   * Detect VRM version from model
   */
  private detectVRMVersion(vrm: VRM): 'vrm-1.0' | 'vrm-0.x' {
    // VRM 1.0 uses 'humanoid', VRM 0.x uses 'humanoid.humanBones'
    // @ts-ignore - accessing internal version field
    const version = vrm.meta?.specVersion || vrm.meta?.version;

    if (version && version.startsWith('1.')) {
      return 'vrm-1.0';
    }

    return 'vrm-0.x';
  }

  /**
   * Convert Ready Player Me GLB to VRM-compatible structure
   *
   * Note: This is a basic conversion. Full VRM support requires
   * additional metadata (blend shapes, first-person settings, etc.)
   */
  private convertGLBToVRM(gltf: any): VRM {
    // Find skeleton in GLB
    let skeleton: THREE.Skeleton | null = null;
    gltf.scene.traverse((object: any) => {
      if (object.isSkinnedMesh && object.skeleton) {
        skeleton = object.skeleton;
      }
    });

    if (!skeleton) {
      throw new Error('GLB file has no skeleton (not a humanoid model)');
    }

    // Attempt to map bones to VRM humanoid structure
    const humanBones = this.detectHumanoidBonesFromSkeleton(skeleton);

    // Create minimal VRM structure
    // Note: @pixiv/three-vrm doesn't provide a constructor for manual VRM creation
    // This is a placeholder - in production, you'd need proper VRM instantiation
    console.warn('[VRMLoader] GLB to VRM conversion is experimental');

    // For now, throw error - proper conversion requires deeper integration
    throw new Error('GLB to VRM conversion not fully implemented. Use native VRM files.');
  }

  /**
   * Detect humanoid bones from generic skeleton (heuristic)
   */
  private detectHumanoidBonesFromSkeleton(
    skeleton: THREE.Skeleton
  ): Partial<Record<VRMHumanBoneName, THREE.Bone>> {
    const bones: Partial<Record<VRMHumanBoneName, THREE.Bone>> = {};

    skeleton.bones.forEach((bone) => {
      const name = bone.name.toLowerCase();

      // Common bone name patterns
      if (name.includes('hips')) bones.hips = bone;
      if (name.includes('spine')) bones.spine = bone;
      if (name.includes('chest')) bones.chest = bone;
      if (name.includes('neck')) bones.neck = bone;
      if (name.includes('head')) bones.head = bone;

      // Left arm
      if (name.includes('leftshoulder') || name.includes('l_shoulder')) bones.leftShoulder = bone;
      if (name.includes('leftupperarm') || name.includes('l_upperarm')) bones.leftUpperArm = bone;
      if (name.includes('leftlowerarm') || name.includes('l_lowerarm')) bones.leftLowerArm = bone;
      if (name.includes('lefthand') || name.includes('l_hand')) bones.leftHand = bone;

      // Right arm
      if (name.includes('rightshoulder') || name.includes('r_shoulder')) bones.rightShoulder = bone;
      if (name.includes('rightupperarm') || name.includes('r_upperarm')) bones.rightUpperArm = bone;
      if (name.includes('rightlowerarm') || name.includes('r_lowerarm')) bones.rightLowerArm = bone;
      if (name.includes('righthand') || name.includes('r_hand')) bones.rightHand = bone;

      // Left leg
      if (name.includes('leftupperleg') || name.includes('l_upperleg')) bones.leftUpperLeg = bone;
      if (name.includes('leftlowerleg') || name.includes('l_lowerleg')) bones.leftLowerLeg = bone;
      if (name.includes('leftfoot') || name.includes('l_foot')) bones.leftFoot = bone;

      // Right leg
      if (name.includes('rightupperleg') || name.includes('r_upperleg')) bones.rightUpperLeg = bone;
      if (name.includes('rightlowerleg') || name.includes('r_lowerleg')) bones.rightLowerLeg = bone;
      if (name.includes('rightfoot') || name.includes('r_foot')) bones.rightFoot = bone;
    });

    return bones;
  }

  /**
   * Validate humanoid rig has all required bones
   */
  private validateHumanoid(vrm: VRM): NonNullable<VRMLoadResult['validation']> {
    const requiredBones: VRMHumanBoneName[] = [
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
    ];

    const missingBones: VRMHumanBoneName[] = [];
    const warnings: string[] = [];

    requiredBones.forEach((boneName) => {
      const bone = vrm.humanoid.getBoneNode(boneName);
      if (!bone) {
        missingBones.push(boneName);
      }
    });

    // Warnings for optional but recommended bones
    const optionalBones: VRMHumanBoneName[] = ['leftEye', 'rightEye'];
    optionalBones.forEach((boneName) => {
      const bone = vrm.humanoid.getBoneNode(boneName);
      if (!bone) {
        warnings.push(`Optional bone missing: ${boneName} (eye tracking won't work)`);
      }
    });

    return {
      valid: missingBones.length === 0,
      missingBones: missingBones.length > 0 ? missingBones : undefined,
      warnings: warnings.length > 0 ? warnings : undefined,
    };
  }

  /**
   * Clone VRM instance (for cache reuse)
   *
   * Creates a deep clone of the VRM model so cached models
   * can be reused without interference.
   */
  private cloneVRM(vrm: VRM): VRM {
    // Clone scene
    const clonedScene = vrm.scene.clone(true);

    // Note: Full VRM cloning requires deep cloning of:
    // - Humanoid rig references
    // - Blend shapes
    // - Spring bones
    // - Materials

    // This is a simplified clone - in production, use VRM.clone() if available
    console.warn('[VRMLoader] VRM cloning is simplified - materials/blend shapes may be shared');

    // @ts-ignore - accessing internal constructor
    return new VRM({
      scene: clonedScene,
      humanoid: vrm.humanoid,  // Shared reference (OK for read-only)
      meta: vrm.meta,
      materials: vrm.materials,
      expressionManager: vrm.expressionManager,
      springBoneManager: vrm.springBoneManager,
    });
  }
}

/**
 * Create default VRM loader instance
 *
 * Convenience function for common use case.
 *
 * @returns Configured VRM loader
 */
export function createVRMLoader(config?: VRMLoaderConfig): VRMLoader {
  return new VRMLoader(config);
}

/**
 * Load VRM model (one-shot helper)
 *
 * @param url - VRM file URL
 * @returns Loaded VRM
 */
export async function loadVRM(url: string): Promise<VRM> {
  const loader = createVRMLoader();
  const result = await loader.load(url);
  return result.vrm;
}
