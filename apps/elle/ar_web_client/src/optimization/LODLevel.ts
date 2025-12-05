/**
 * LODLevel - Level of Detail Definitions
 *
 * Defines the three LOD levels used throughout the avatar system.
 * Each level has specific quality settings and performance characteristics.
 *
 * Design Philosophy:
 * - HIGH: Full quality for nearby avatars (<5m)
 * - MEDIUM: Balanced quality for mid-range (5-15m)
 * - LOW: Minimal detail for distant avatars (>15m)
 *
 * Created: 2025-11-22 (Phase 6.3)
 */

import * as THREE from 'three';

/**
 * LOD quality levels
 */
export enum LODLevel {
  /**
   * Highest quality - full detail
   * - Original mesh geometry
   * - Full textures (1024x1024+)
   * - Spring bone physics enabled
   * - Shadow casting enabled
   * - Target: <5m distance
   */
  HIGH = 'high',

  /**
   * Medium quality - balanced
   * - 50% geometry reduction
   * - Half-resolution textures (512x512)
   * - Spring bones disabled
   * - Shadow receiving only
   * - Target: 5-15m distance
   */
  MEDIUM = 'medium',

  /**
   * Lowest quality - minimal detail
   * - 75% geometry reduction
   * - Quarter-resolution textures (256x256)
   * - All physics disabled
   * - No shadows
   * - Target: >15m distance
   */
  LOW = 'low',
}

/**
 * Configuration for each LOD level
 */
export interface LODLevelConfig {
  /**
   * LOD level identifier
   */
  level: LODLevel;

  /**
   * Maximum distance for this LOD (meters)
   * Avatars beyond this distance switch to next lower LOD
   */
  maxDistance: number;

  /**
   * Geometry decimation factor (0-1)
   * 0 = remove all geometry, 1 = keep all geometry
   */
  geometryQuality: number;

  /**
   * Texture resolution scale (0-1)
   * Applied to original texture dimensions
   */
  textureScale: number;

  /**
   * Enable spring bone physics
   */
  enablePhysics: boolean;

  /**
   * Shadow settings
   */
  shadows: {
    cast: boolean;      // Cast shadows
    receive: boolean;   // Receive shadows
  };

  /**
   * Animation update rate (Hz)
   * Lower rates for distant avatars
   */
  animationRate: number;

  /**
   * Estimated performance cost (1-10)
   * Higher = more expensive to render
   */
  performanceCost: number;
}

/**
 * Default LOD configurations
 */
export const DEFAULT_LOD_CONFIGS: Record<LODLevel, LODLevelConfig> = {
  [LODLevel.HIGH]: {
    level: LODLevel.HIGH,
    maxDistance: 5.0,
    geometryQuality: 1.0,      // 100% geometry
    textureScale: 1.0,         // Full resolution
    enablePhysics: true,
    shadows: {
      cast: true,
      receive: true,
    },
    animationRate: 60,         // 60 Hz
    performanceCost: 10,
  },

  [LODLevel.MEDIUM]: {
    level: LODLevel.MEDIUM,
    maxDistance: 15.0,
    geometryQuality: 0.5,      // 50% geometry
    textureScale: 0.5,         // Half resolution
    enablePhysics: false,
    shadows: {
      cast: false,
      receive: true,
    },
    animationRate: 30,         // 30 Hz
    performanceCost: 5,
  },

  [LODLevel.LOW]: {
    level: LODLevel.LOW,
    maxDistance: Infinity,
    geometryQuality: 0.25,     // 25% geometry
    textureScale: 0.25,        // Quarter resolution
    enablePhysics: false,
    shadows: {
      cast: false,
      receive: false,
    },
    animationRate: 15,         // 15 Hz
    performanceCost: 2,
  },
};

/**
 * LOD transition settings
 */
export interface LODTransitionConfig {
  /**
   * Enable smooth transitions between LOD levels
   * Uses cross-fade when switching levels
   */
  enableSmoothing: boolean;

  /**
   * Transition duration in milliseconds
   */
  transitionDuration: number;

  /**
   * Hysteresis factor (0-1)
   * Prevents rapid LOD switching near boundaries
   * Higher values = more stable but less responsive
   */
  hysteresis: number;
}

/**
 * Default transition configuration
 */
export const DEFAULT_LOD_TRANSITION: LODTransitionConfig = {
  enableSmoothing: true,
  transitionDuration: 300,    // 300ms fade
  hysteresis: 0.2,            // 20% hysteresis (e.g., 5m ± 1m)
};

/**
 * LOD metrics for performance monitoring
 */
export interface LODMetrics {
  /**
   * Current LOD distribution
   * Counts how many avatars at each level
   */
  distribution: Record<LODLevel, number>;

  /**
   * Average LOD level (1=LOW, 2=MEDIUM, 3=HIGH)
   */
  averageLevel: number;

  /**
   * Total performance cost (sum of all avatar costs)
   */
  totalCost: number;

  /**
   * Number of LOD transitions in last second
   */
  transitionsPerSecond: number;

  /**
   * Estimated FPS impact (percentage)
   */
  fpsImpact: number;
}

/**
 * Calculate LOD level based on distance
 *
 * @param distance - Distance from camera to avatar (meters)
 * @param configs - LOD configurations (optional, uses defaults)
 * @returns Appropriate LOD level for distance
 */
export function calculateLODLevel(
  distance: number,
  configs: Record<LODLevel, LODLevelConfig> = DEFAULT_LOD_CONFIGS
): LODLevel {
  if (distance <= configs[LODLevel.HIGH].maxDistance) {
    return LODLevel.HIGH;
  } else if (distance <= configs[LODLevel.MEDIUM].maxDistance) {
    return LODLevel.MEDIUM;
  } else {
    return LODLevel.LOW;
  }
}

/**
 * Calculate distance with hysteresis to prevent flickering
 *
 * @param currentDistance - Current measured distance
 * @param previousLevel - Previously assigned LOD level
 * @param hysteresis - Hysteresis factor (0-1)
 * @param configs - LOD configurations
 * @returns LOD level with hysteresis applied
 */
export function calculateLODWithHysteresis(
  currentDistance: number,
  previousLevel: LODLevel,
  hysteresis: number = 0.2,
  configs: Record<LODLevel, LODLevelConfig> = DEFAULT_LOD_CONFIGS
): LODLevel {
  // Get base LOD level
  const baseLevel = calculateLODLevel(currentDistance, configs);

  // If no previous level, return base
  if (!previousLevel) {
    return baseLevel;
  }

  // Calculate hysteresis buffer
  const previousConfig = configs[previousLevel];
  const buffer = previousConfig.maxDistance * hysteresis;

  // If switching to lower quality (further away)
  if (baseLevel === LODLevel.MEDIUM && previousLevel === LODLevel.HIGH) {
    if (currentDistance < configs[LODLevel.HIGH].maxDistance + buffer) {
      return LODLevel.HIGH; // Stay HIGH until firmly in MEDIUM range
    }
  } else if (baseLevel === LODLevel.LOW && previousLevel === LODLevel.MEDIUM) {
    if (currentDistance < configs[LODLevel.MEDIUM].maxDistance + buffer) {
      return LODLevel.MEDIUM; // Stay MEDIUM until firmly in LOW range
    }
  }

  // If switching to higher quality (closer)
  if (baseLevel === LODLevel.MEDIUM && previousLevel === LODLevel.LOW) {
    if (currentDistance > configs[LODLevel.MEDIUM].maxDistance - buffer) {
      return LODLevel.LOW; // Stay LOW until firmly in MEDIUM range
    }
  } else if (baseLevel === LODLevel.HIGH && previousLevel === LODLevel.MEDIUM) {
    if (currentDistance > configs[LODLevel.HIGH].maxDistance - buffer) {
      return LODLevel.MEDIUM; // Stay MEDIUM until firmly in HIGH range
    }
  }

  return baseLevel;
}

/**
 * Get performance cost for LOD level
 *
 * @param level - LOD level
 * @param configs - LOD configurations
 * @returns Performance cost (1-10)
 */
export function getLODCost(
  level: LODLevel,
  configs: Record<LODLevel, LODLevelConfig> = DEFAULT_LOD_CONFIGS
): number {
  return configs[level].performanceCost;
}

/**
 * Estimate FPS impact for given LOD distribution
 *
 * @param distribution - Avatar count per LOD level
 * @param configs - LOD configurations
 * @returns Estimated FPS impact percentage (0-100)
 */
export function estimateFPSImpact(
  distribution: Record<LODLevel, number>,
  configs: Record<LODLevel, LODLevelConfig> = DEFAULT_LOD_CONFIGS
): number {
  const totalCost =
    distribution[LODLevel.HIGH] * configs[LODLevel.HIGH].performanceCost +
    distribution[LODLevel.MEDIUM] * configs[LODLevel.MEDIUM].performanceCost +
    distribution[LODLevel.LOW] * configs[LODLevel.LOW].performanceCost;

  // Baseline: 1 HIGH avatar = 10 cost ≈ 60 FPS
  // Target: 8 avatars at mixed LOD ≈ 60 FPS
  // Each cost point ≈ 1% FPS impact
  const fpsImpact = Math.min(100, (totalCost / 80) * 100);

  return fpsImpact;
}

/**
 * Create LOD metrics from avatar set
 *
 * @param avatarLevels - Map of avatar ID to current LOD level
 * @param recentTransitions - Number of transitions in last second
 * @param configs - LOD configurations
 * @returns LOD metrics
 */
export function createLODMetrics(
  avatarLevels: Map<string, LODLevel>,
  recentTransitions: number = 0,
  configs: Record<LODLevel, LODLevelConfig> = DEFAULT_LOD_CONFIGS
): LODMetrics {
  const distribution: Record<LODLevel, number> = {
    [LODLevel.HIGH]: 0,
    [LODLevel.MEDIUM]: 0,
    [LODLevel.LOW]: 0,
  };

  // Count avatars at each level
  avatarLevels.forEach((level) => {
    distribution[level]++;
  });

  // Calculate average level (1=LOW, 2=MEDIUM, 3=HIGH)
  const levelValues = {
    [LODLevel.LOW]: 1,
    [LODLevel.MEDIUM]: 2,
    [LODLevel.HIGH]: 3,
  };

  const totalAvatars = avatarLevels.size;
  const sumLevels = Array.from(avatarLevels.values()).reduce(
    (sum, level) => sum + levelValues[level],
    0
  );

  const averageLevel = totalAvatars > 0 ? sumLevels / totalAvatars : 0;

  // Calculate total cost
  const totalCost =
    distribution[LODLevel.HIGH] * configs[LODLevel.HIGH].performanceCost +
    distribution[LODLevel.MEDIUM] * configs[LODLevel.MEDIUM].performanceCost +
    distribution[LODLevel.LOW] * configs[LODLevel.LOW].performanceCost;

  // Estimate FPS impact
  const fpsImpact = estimateFPSImpact(distribution, configs);

  return {
    distribution,
    averageLevel,
    totalCost,
    transitionsPerSecond: recentTransitions,
    fpsImpact,
  };
}
