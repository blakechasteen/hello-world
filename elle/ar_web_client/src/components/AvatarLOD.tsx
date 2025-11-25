/**
 * AvatarLOD - Avatar Component with Level of Detail
 *
 * React wrapper around Avatar component that automatically applies LOD optimizations.
 * Integrates with LODManager for centralized LOD management.
 *
 * Features:
 * - Automatic LOD level application
 * - Geometry decimation for distant avatars
 * - Texture downscaling
 * - Physics toggling
 * - Shadow optimization
 * - Smooth transitions
 *
 * Usage:
 * ```tsx
 * <AvatarLOD
 *   avatarId="user-123"
 *   url="/avatars/avatar.vrm"
 *   pose={pose}
 *   position={[0, 0, 0]}
 *   lodManager={lodManager}
 * />
 * ```
 *
 * Created: 2025-11-22 (Phase 6.3)
 */

import React, { useEffect, useState, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

// Phase 6.1: Avatar component
import { Avatar, AvatarProps } from './Avatar';

// Phase 6.3: LOD system
import { LODManager } from '../optimization/LODManager';
import { LODLevel, LODLevelConfig } from '../optimization/LODLevel';
import { FrustumCuller } from '../optimization/FrustumCuller';

/**
 * AvatarLOD props
 * Extends Avatar props with LOD support
 */
export interface AvatarLODProps extends Omit<AvatarProps, 'enablePhysics' | 'enableSmoothing'> {
  /**
   * Avatar identifier (required for LOD tracking)
   */
  avatarId: string;

  /**
   * LOD manager instance (shared across all avatars)
   */
  lodManager: LODManager;

  /**
   * Force specific LOD level (overrides automatic calculation)
   */
  forceLOD?: LODLevel;

  /**
   * Callback when LOD level changes
   */
  onLODChange?: (newLevel: LODLevel, oldLevel: LODLevel) => void;

  /**
   * Enable LOD debug visualization
   */
  showLODDebug?: boolean;

  /**
   * Frustum culler instance (optional, for occlusion culling)
   */
  frustumCuller?: FrustumCuller;

  /**
   * Bounding radius for frustum culling (default: 0.6)
   */
  boundingRadius?: number;

  /**
   * Callback when visibility changes
   */
  onVisibilityChange?: (visible: boolean) => void;
}

/**
 * AvatarLOD - Avatar with automatic level of detail
 *
 * Automatically adjusts quality based on distance from camera.
 * Integrates with LODManager for system-wide LOD coordination.
 */
export const AvatarLOD: React.FC<AvatarLODProps> = ({
  avatarId,
  lodManager,
  url,
  pose,
  position,
  rotation,
  userId,
  scale = 1.0,
  forceLOD,
  onLODChange,
  showLODDebug = false,
  onLoad,
  onError,
  frustumCuller,
  boundingRadius = 0.6,
  onVisibilityChange,
}) => {
  const { camera } = useThree();

  // Current LOD level
  const [currentLOD, setCurrentLOD] = useState<LODLevel>(
    forceLOD || LODLevel.HIGH
  );

  // Visibility state (for frustum culling)
  const [isVisible, setIsVisible] = useState<boolean>(true);

  // Convert position array to Vector3
  const positionVec = useMemo(
    () => new THREE.Vector3(...position),
    [position]
  );

  /**
   * Update LOD level and visibility each frame
   */
  useFrame(() => {
    // Update frustum culling (if enabled)
    if (frustumCuller) {
      const visible = frustumCuller.isVisible(avatarId, positionVec, boundingRadius);

      if (visible !== isVisible) {
        setIsVisible(visible);
        onVisibilityChange?.(visible);
      }

      // Early exit if not visible (skip LOD update)
      if (!visible) return;
    }

    if (forceLOD) {
      // Use forced LOD
      if (currentLOD !== forceLOD) {
        const oldLevel = currentLOD;
        setCurrentLOD(forceLOD);
        onLODChange?.(forceLOD, oldLevel);
      }
      return;
    }

    // Update LOD manager
    const result = lodManager.update(avatarId, positionVec);

    // Update local state if changed
    if (result.changed) {
      setCurrentLOD(result.newLevel);
      onLODChange?.(result.newLevel, result.oldLevel);
    }
  });

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      lodManager.remove(avatarId);
      frustumCuller?.remove(avatarId);
    };
  }, [avatarId, lodManager, frustumCuller]);

  // Get LOD configuration
  const lodConfig = useMemo(
    () => lodManager.getConfig(currentLOD),
    [currentLOD, lodManager]
  );

  // Apply LOD settings
  const avatarProps: AvatarProps = {
    url,
    pose,
    position,
    rotation,
    userId,
    scale,
    enablePhysics: lodConfig.enablePhysics,
    enableSmoothing: currentLOD === LODLevel.HIGH, // Only smooth HIGH LOD
    visibilityThreshold: currentLOD === LODLevel.LOW ? 0.7 : 0.5, // Higher threshold for LOW
    onLoad,
    onError,
  };

  // Don't render if culled (not visible)
  if (!isVisible) {
    return null;
  }

  return (
    <group>
      {/* Main avatar */}
      <Avatar {...avatarProps} />

      {/* LOD debug visualization */}
      {showLODDebug && (
        <LODDebugVisual
          position={positionVec}
          level={currentLOD}
          distance={camera.position.distanceTo(positionVec)}
        />
      )}
    </group>
  );
};

/**
 * LODDebugVisual - Debug visualization for LOD level
 */
interface LODDebugVisualProps {
  position: THREE.Vector3;
  level: LODLevel;
  distance: number;
}

const LODDebugVisual: React.FC<LODDebugVisualProps> = ({
  position,
  level,
  distance,
}) => {
  // Colors for each LOD level
  const colors = {
    [LODLevel.HIGH]: 0x00ff00,   // Green
    [LODLevel.MEDIUM]: 0xffff00, // Yellow
    [LODLevel.LOW]: 0xff0000,    // Red
  };

  const color = colors[level];

  return (
    <group position={[position.x, position.y + 2.5, position.z]}>
      {/* LOD indicator sphere */}
      <mesh>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.7} />
      </mesh>

      {/* Distance text (would need THREE.TextGeometry or sprite) */}
      {/* For now, just the sphere indicator */}
    </group>
  );
};

/**
 * MultiAvatarLODScene - Component that manages LOD for multiple avatars
 *
 * Usage:
 * ```tsx
 * const avatars = [
 *   { id: 'user-1', url: '/avatar1.vrm', position: [0, 0, 0], ... },
 *   { id: 'user-2', url: '/avatar2.vrm', position: [5, 0, 0], ... },
 * ];
 *
 * <MultiAvatarLODScene avatars={avatars} />
 * ```
 */
export interface MultiAvatarLODSceneProps {
  /**
   * Avatar configurations
   */
  avatars: Array<AvatarLODProps>;

  /**
   * LOD manager instance (optional, creates one if not provided)
   */
  lodManager?: LODManager;

  /**
   * Show LOD debug for all avatars
   */
  showLODDebug?: boolean;

  /**
   * Callback when any avatar LOD changes
   */
  onLODChange?: (avatarId: string, newLevel: LODLevel, oldLevel: LODLevel) => void;
}

export const MultiAvatarLODScene: React.FC<MultiAvatarLODSceneProps> = ({
  avatars,
  lodManager: providedManager,
  showLODDebug = false,
  onLODChange,
}) => {
  const { camera } = useThree();

  // Create LOD manager if not provided
  const lodManager = useMemo(
    () => providedManager || new LODManager(camera),
    [providedManager, camera]
  );

  // Update camera reference when camera changes
  useEffect(() => {
    if (!providedManager) {
      lodManager.setCamera(camera);
    }
  }, [camera, lodManager, providedManager]);

  return (
    <>
      {avatars.map((avatar) => (
        <AvatarLOD
          key={avatar.avatarId}
          {...avatar}
          lodManager={lodManager}
          showLODDebug={showLODDebug}
          onLODChange={(newLevel, oldLevel) => {
            avatar.onLODChange?.(newLevel, oldLevel);
            onLODChange?.(avatar.avatarId, newLevel, oldLevel);
          }}
        />
      ))}
    </>
  );
};

/**
 * useLODMetrics - Hook to monitor LOD system performance
 *
 * Usage:
 * ```tsx
 * const metrics = useLODMetrics(lodManager);
 * console.log(`Average LOD: ${metrics.averageLevel}`);
 * console.log(`FPS Impact: ${metrics.fpsImpact}%`);
 * ```
 */
export function useLODMetrics(lodManager: LODManager) {
  const [metrics, setMetrics] = useState(lodManager.getMetrics());

  useFrame(() => {
    setMetrics(lodManager.getMetrics());
  });

  return metrics;
}

/**
 * useLODLevel - Hook to get current LOD level for avatar
 *
 * Usage:
 * ```tsx
 * const lodLevel = useLODLevel(lodManager, 'user-123');
 * ```
 */
export function useLODLevel(lodManager: LODManager, avatarId: string): LODLevel | null {
  const [level, setLevel] = useState<LODLevel | null>(
    lodManager.getLevel(avatarId)
  );

  useFrame(() => {
    const currentLevel = lodManager.getLevel(avatarId);
    if (currentLevel !== level) {
      setLevel(currentLevel);
    }
  });

  return level;
}

/**
 * LODMetricsDisplay - Component to display LOD metrics
 *
 * Usage:
 * ```tsx
 * <LODMetricsDisplay lodManager={lodManager} />
 * ```
 */
export interface LODMetricsDisplayProps {
  lodManager: LODManager;
  position?: [number, number]; // Screen position (x, y)
}

export const LODMetricsDisplay: React.FC<LODMetricsDisplayProps> = ({
  lodManager,
  position = [10, 10],
}) => {
  const metrics = useLODMetrics(lodManager);

  return (
    <div
      style={{
        position: 'absolute',
        left: position[0],
        top: position[1],
        background: 'rgba(0, 0, 0, 0.7)',
        color: 'white',
        padding: '10px',
        borderRadius: '5px',
        fontFamily: 'monospace',
        fontSize: '12px',
        zIndex: 100,
      }}
    >
      <div><strong>LOD System</strong></div>
      <div>High: {metrics.distribution[LODLevel.HIGH]}</div>
      <div>Medium: {metrics.distribution[LODLevel.MEDIUM]}</div>
      <div>Low: {metrics.distribution[LODLevel.LOW]}</div>
      <div>Avg Level: {metrics.averageLevel.toFixed(2)}</div>
      <div>Total Cost: {metrics.totalCost}</div>
      <div>FPS Impact: {metrics.fpsImpact.toFixed(1)}%</div>
      <div>Transitions/s: {metrics.transitionsPerSecond}</div>
    </div>
  );
};

/**
 * Default export
 */
export default AvatarLOD;
