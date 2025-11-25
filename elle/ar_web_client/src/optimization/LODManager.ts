/**
 * LODManager - Level of Detail Management System
 *
 * Manages LOD levels for multiple avatars based on camera distance.
 * Provides automatic LOD switching with hysteresis and performance monitoring.
 *
 * Features:
 * - Distance-based LOD calculation
 * - Smooth transitions with hysteresis
 * - Performance metrics tracking
 * - Custom LOD strategies
 * - Event notifications
 *
 * Design Philosophy:
 * - Single source of truth for all avatar LOD states
 * - Extensible strategy pattern for custom LOD logic
 * - Performance-first: minimal overhead per frame
 *
 * Created: 2025-11-22 (Phase 6.3)
 */

import * as THREE from 'three';
import {
  LODLevel,
  LODLevelConfig,
  LODTransitionConfig,
  LODMetrics,
  DEFAULT_LOD_CONFIGS,
  DEFAULT_LOD_TRANSITION,
  calculateLODWithHysteresis,
  createLODMetrics,
} from './LODLevel';

/**
 * LOD state for single avatar
 */
export interface AvatarLODState {
  /**
   * Avatar identifier
   */
  avatarId: string;

  /**
   * Current LOD level
   */
  currentLevel: LODLevel;

  /**
   * Previous LOD level (for transition detection)
   */
  previousLevel: LODLevel | null;

  /**
   * Current distance from camera
   */
  distance: number;

  /**
   * Timestamp of last LOD change
   */
  lastTransition: number;

  /**
   * Transition progress (0-1) for smoothing
   */
  transitionProgress: number;
}

/**
 * LOD update result
 */
export interface LODUpdateResult {
  /**
   * Avatar ID
   */
  avatarId: string;

  /**
   * New LOD level
   */
  newLevel: LODLevel;

  /**
   * Previous LOD level
   */
  oldLevel: LODLevel;

  /**
   * Whether level changed
   */
  changed: boolean;

  /**
   * Distance from camera
   */
  distance: number;
}

/**
 * LOD strategy interface
 * Allows custom LOD calculation logic
 */
export interface LODStrategy {
  /**
   * Calculate appropriate LOD level
   *
   * @param distance - Distance from camera to avatar
   * @param currentLevel - Current LOD level (for hysteresis)
   * @param avatarId - Avatar identifier (for context)
   * @returns New LOD level
   */
  calculateLevel(
    distance: number,
    currentLevel: LODLevel | null,
    avatarId: string
  ): LODLevel;
}

/**
 * Default distance-based LOD strategy
 */
export class DistanceBasedLODStrategy implements LODStrategy {
  constructor(
    private configs: Record<LODLevel, LODLevelConfig> = DEFAULT_LOD_CONFIGS,
    private hysteresis: number = 0.2
  ) {}

  calculateLevel(
    distance: number,
    currentLevel: LODLevel | null,
    avatarId: string
  ): LODLevel {
    if (currentLevel) {
      return calculateLODWithHysteresis(distance, currentLevel, this.hysteresis, this.configs);
    } else {
      // First time, no hysteresis
      return calculateLODWithHysteresis(distance, LODLevel.HIGH, 0, this.configs);
    }
  }
}

/**
 * Performance-aware LOD strategy
 * Dynamically adjusts LOD based on current FPS
 */
export class PerformanceAwareLODStrategy implements LODStrategy {
  private targetFPS: number = 60;
  private currentFPS: number = 60;
  private performanceBudget: number = 80; // Max total cost

  constructor(
    private configs: Record<LODLevel, LODLevelConfig> = DEFAULT_LOD_CONFIGS,
    private hysteresis: number = 0.2
  ) {}

  /**
   * Update current FPS for adaptive behavior
   */
  updateFPS(fps: number): void {
    this.currentFPS = fps;
  }

  calculateLevel(
    distance: number,
    currentLevel: LODLevel | null,
    avatarId: string
  ): LODLevel {
    // Start with distance-based calculation
    const baseLevel = calculateLODWithHysteresis(
      distance,
      currentLevel || LODLevel.HIGH,
      this.hysteresis,
      this.configs
    );

    // If performance is good, return base level
    if (this.currentFPS >= this.targetFPS * 0.9) {
      return baseLevel;
    }

    // If performance is poor, downgrade
    if (this.currentFPS < this.targetFPS * 0.7) {
      // Severe performance issues - aggressive downgrade
      if (baseLevel === LODLevel.HIGH) return LODLevel.MEDIUM;
      if (baseLevel === LODLevel.MEDIUM) return LODLevel.LOW;
    } else if (this.currentFPS < this.targetFPS * 0.8) {
      // Moderate performance issues - conservative downgrade
      if (baseLevel === LODLevel.HIGH && distance > this.configs[LODLevel.HIGH].maxDistance * 0.7) {
        return LODLevel.MEDIUM;
      }
    }

    return baseLevel;
  }
}

/**
 * LODManager - Centralized LOD management
 *
 * Usage:
 * ```typescript
 * const manager = new LODManager(camera);
 *
 * // Update all avatars each frame
 * const updates = manager.updateAll(avatarPositions);
 *
 * // Check if avatar needs LOD change
 * updates.forEach(update => {
 *   if (update.changed) {
 *     applyLOD(update.avatarId, update.newLevel);
 *   }
 * });
 *
 * // Get metrics
 * const metrics = manager.getMetrics();
 * ```
 */
export class LODManager {
  private camera: THREE.Camera;
  private strategy: LODStrategy;
  private configs: Record<LODLevel, LODLevelConfig>;
  private transitionConfig: LODTransitionConfig;

  // Avatar states
  private states: Map<string, AvatarLODState> = new Map();

  // Metrics tracking
  private transitionCount: number = 0;
  private lastMetricsUpdate: number = 0;
  private recentTransitions: number = 0;

  // Event handlers
  private eventHandlers: Map<string, Function[]> = new Map();

  constructor(
    camera: THREE.Camera,
    strategy?: LODStrategy,
    configs: Record<LODLevel, LODLevelConfig> = DEFAULT_LOD_CONFIGS,
    transitionConfig: LODTransitionConfig = DEFAULT_LOD_TRANSITION
  ) {
    this.camera = camera;
    this.strategy = strategy || new DistanceBasedLODStrategy(configs);
    this.configs = configs;
    this.transitionConfig = transitionConfig;
  }

  /**
   * Update LOD for single avatar
   *
   * @param avatarId - Avatar identifier
   * @param position - Avatar position in world space
   * @returns Update result
   */
  update(avatarId: string, position: THREE.Vector3): LODUpdateResult {
    // Calculate distance from camera
    const distance = this.camera.position.distanceTo(position);

    // Get current state
    const currentState = this.states.get(avatarId);
    const currentLevel = currentState?.currentLevel || null;

    // Calculate new level using strategy
    const newLevel = this.strategy.calculateLevel(distance, currentLevel, avatarId);

    // Check if level changed
    const changed = currentLevel !== newLevel;

    // Update state
    const newState: AvatarLODState = {
      avatarId,
      currentLevel: newLevel,
      previousLevel: currentLevel,
      distance,
      lastTransition: changed ? Date.now() : currentState?.lastTransition || Date.now(),
      transitionProgress: changed ? 0 : 1,
    };

    this.states.set(avatarId, newState);

    // Track transition
    if (changed) {
      this.transitionCount++;
      this.recentTransitions++;

      // Emit event
      this.emit('lod-changed', {
        avatarId,
        oldLevel: currentLevel,
        newLevel,
        distance,
      });
    }

    return {
      avatarId,
      newLevel,
      oldLevel: currentLevel || newLevel,
      changed,
      distance,
    };
  }

  /**
   * Update LOD for all avatars
   *
   * @param avatarPositions - Map of avatar ID to position
   * @returns Array of update results
   */
  updateAll(avatarPositions: Map<string, THREE.Vector3>): LODUpdateResult[] {
    const results: LODUpdateResult[] = [];

    // Update each avatar
    avatarPositions.forEach((position, avatarId) => {
      const result = this.update(avatarId, position);
      results.push(result);
    });

    // Remove avatars that no longer exist
    const currentAvatars = new Set(avatarPositions.keys());
    this.states.forEach((state, avatarId) => {
      if (!currentAvatars.has(avatarId)) {
        this.remove(avatarId);
      }
    });

    // Update transition progress for smooth transitions
    if (this.transitionConfig.enableSmoothing) {
      this.updateTransitions();
    }

    return results;
  }

  /**
   * Update transition progress for all avatars
   */
  private updateTransitions(): void {
    const now = Date.now();
    const duration = this.transitionConfig.transitionDuration;

    this.states.forEach((state) => {
      if (state.transitionProgress < 1) {
        const elapsed = now - state.lastTransition;
        state.transitionProgress = Math.min(1, elapsed / duration);
      }
    });
  }

  /**
   * Remove avatar from tracking
   *
   * @param avatarId - Avatar identifier
   */
  remove(avatarId: string): void {
    this.states.delete(avatarId);
  }

  /**
   * Get current LOD level for avatar
   *
   * @param avatarId - Avatar identifier
   * @returns Current LOD level or null if not tracked
   */
  getLevel(avatarId: string): LODLevel | null {
    return this.states.get(avatarId)?.currentLevel || null;
  }

  /**
   * Get full state for avatar
   *
   * @param avatarId - Avatar identifier
   * @returns Avatar LOD state or null
   */
  getState(avatarId: string): AvatarLODState | null {
    return this.states.get(avatarId) || null;
  }

  /**
   * Get configuration for LOD level
   *
   * @param level - LOD level
   * @returns LOD configuration
   */
  getConfig(level: LODLevel): LODLevelConfig {
    return this.configs[level];
  }

  /**
   * Get performance metrics
   *
   * @returns LOD metrics
   */
  getMetrics(): LODMetrics {
    const now = Date.now();

    // Reset transition counter every second
    if (now - this.lastMetricsUpdate >= 1000) {
      this.recentTransitions = 0;
      this.lastMetricsUpdate = now;
    }

    // Get current level distribution
    const levelMap = new Map<string, LODLevel>();
    this.states.forEach((state, avatarId) => {
      levelMap.set(avatarId, state.currentLevel);
    });

    return createLODMetrics(levelMap, this.recentTransitions, this.configs);
  }

  /**
   * Get all tracked avatar IDs
   *
   * @returns Array of avatar IDs
   */
  getAvatarIds(): string[] {
    return Array.from(this.states.keys());
  }

  /**
   * Get avatar count
   *
   * @returns Number of tracked avatars
   */
  getAvatarCount(): number {
    return this.states.size;
  }

  /**
   * Update camera reference
   * Call this if camera changes
   *
   * @param camera - New camera
   */
  setCamera(camera: THREE.Camera): void {
    this.camera = camera;
  }

  /**
   * Update LOD strategy
   * Allows runtime strategy switching
   *
   * @param strategy - New LOD strategy
   */
  setStrategy(strategy: LODStrategy): void {
    this.strategy = strategy;
  }

  /**
   * Event emitter
   */
  on(event: string, handler: Function): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  private emit(event: string, ...args: any[]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach((handler) => handler(...args));
    }
  }

  /**
   * Reset all tracking
   */
  reset(): void {
    this.states.clear();
    this.transitionCount = 0;
    this.recentTransitions = 0;
  }
}

/**
 * Create LOD manager with default settings
 *
 * @param camera - Camera for distance calculation
 * @returns LOD manager instance
 */
export function createLODManager(camera: THREE.Camera): LODManager {
  return new LODManager(camera);
}

/**
 * Create performance-aware LOD manager
 *
 * @param camera - Camera for distance calculation
 * @returns LOD manager with performance-aware strategy
 */
export function createPerformanceAwareLODManager(camera: THREE.Camera): LODManager {
  const strategy = new PerformanceAwareLODStrategy();
  return new LODManager(camera, strategy);
}
