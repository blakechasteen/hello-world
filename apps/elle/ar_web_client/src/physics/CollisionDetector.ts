/**
 * CollisionDetector - Avatar Collision Detection System
 *
 * Detects collisions between avatars using sphere-sphere intersection.
 * Provides collision events and proximity information for multi-user interactions.
 *
 * Features:
 * - Sphere-based collision detection (efficient)
 * - Configurable collision radius per avatar
 * - Proximity zones (near, medium, far)
 * - Collision event callbacks
 * - Performance tracking (collisions per second)
 *
 * Usage:
 * ```typescript
 * const detector = new CollisionDetector({
 *   defaultRadius: 0.5,
 *   enableProximity: true,
 * });
 *
 * // Register avatars
 * detector.registerAvatar('user1', position1, { radius: 0.5 });
 * detector.registerAvatar('user2', position2, { radius: 0.5 });
 *
 * // Check collisions (call every frame)
 * const collisions = detector.checkCollisions();
 *
 * // Listen for events
 * detector.on('collision', (event) => {
 *   console.log(`${event.avatarA} collided with ${event.avatarB}`);
 * });
 * ```
 *
 * Created: 2025-11-22 (Phase 6.3 Task 4)
 */

import * as THREE from 'three';

/**
 * Proximity zone
 */
export enum ProximityZone {
  NEAR = 'near',       // <1m
  MEDIUM = 'medium',   // 1-2m
  FAR = 'far',         // 2-5m
  DISTANT = 'distant', // >5m
}

/**
 * Collision event
 */
export interface CollisionEvent {
  /**
   * First avatar ID
   */
  avatarA: string;

  /**
   * Second avatar ID
   */
  avatarB: string;

  /**
   * Collision point (midpoint between centers)
   */
  point: THREE.Vector3;

  /**
   * Distance between centers
   */
  distance: number;

  /**
   * Penetration depth (how much they overlap)
   */
  penetration: number;

  /**
   * Collision normal (from A to B)
   */
  normal: THREE.Vector3;

  /**
   * Timestamp
   */
  timestamp: number;
}

/**
 * Proximity event
 */
export interface ProximityEvent {
  /**
   * First avatar ID
   */
  avatarA: string;

  /**
   * Second avatar ID
   */
  avatarB: string;

  /**
   * Proximity zone
   */
  zone: ProximityZone;

  /**
   * Distance between avatars
   */
  distance: number;

  /**
   * Previous zone (for zone change detection)
   */
  previousZone: ProximityZone | null;

  /**
   * Timestamp
   */
  timestamp: number;
}

/**
 * Avatar collision data
 */
interface AvatarCollisionData {
  id: string;
  position: THREE.Vector3;
  radius: number;
  lastUpdate: number;
}

/**
 * Collision pair state (for detecting enter/exit)
 */
interface CollisionPairState {
  colliding: boolean;
  proximity: ProximityZone;
  lastDistance: number;
}

/**
 * Collision detector configuration
 */
export interface CollisionDetectorConfig {
  /**
   * Default collision radius (meters)
   * Default: 0.5
   */
  defaultRadius?: number;

  /**
   * Enable proximity zones
   * Default: true
   */
  enableProximity?: boolean;

  /**
   * Proximity zone thresholds (meters)
   */
  proximityThresholds?: {
    near: number;      // Default: 1.0
    medium: number;    // Default: 2.0
    far: number;       // Default: 5.0
  };

  /**
   * Maximum avatars to check (performance limit)
   * Default: 50
   */
  maxAvatars?: number;

  /**
   * Enable debug logging
   * Default: false
   */
  debug?: boolean;
}

/**
 * Default configuration
 */
export const DEFAULT_COLLISION_CONFIG: Required<CollisionDetectorConfig> = {
  defaultRadius: 0.5,
  enableProximity: true,
  proximityThresholds: {
    near: 1.0,
    medium: 2.0,
    far: 5.0,
  },
  maxAvatars: 50,
  debug: false,
};

/**
 * CollisionDetector - Detects collisions between avatars
 *
 * Uses sphere-sphere intersection for efficient collision detection.
 * Tracks collision state to detect enter/exit events.
 */
export class CollisionDetector {
  private config: Required<CollisionDetectorConfig>;

  // Avatar data
  private avatars: Map<string, AvatarCollisionData> = new Map();

  // Collision state (for tracking enter/exit)
  private pairStates: Map<string, CollisionPairState> = new Map();

  // Event handlers
  private eventHandlers: Map<string, Function[]> = new Map();

  // Metrics
  private collisionCount: number = 0;
  private proximityChangeCount: number = 0;
  private lastMetricsReset: number = Date.now();

  constructor(config: CollisionDetectorConfig = {}) {
    this.config = {
      ...DEFAULT_COLLISION_CONFIG,
      ...config,
      proximityThresholds: {
        ...DEFAULT_COLLISION_CONFIG.proximityThresholds,
        ...config.proximityThresholds,
      },
    };
  }

  /**
   * Register avatar for collision detection
   */
  registerAvatar(
    id: string,
    position: THREE.Vector3,
    options?: { radius?: number }
  ): void {
    if (this.avatars.size >= this.config.maxAvatars) {
      console.warn(`[CollisionDetector] Max avatars reached (${this.config.maxAvatars})`);
      return;
    }

    this.avatars.set(id, {
      id,
      position: position.clone(),
      radius: options?.radius ?? this.config.defaultRadius,
      lastUpdate: Date.now(),
    });

    if (this.config.debug) {
      console.log(`[CollisionDetector] Registered avatar ${id} at (${position.x.toFixed(2)}, ${position.y.toFixed(2)}, ${position.z.toFixed(2)})`);
    }
  }

  /**
   * Update avatar position
   */
  updateAvatar(id: string, position: THREE.Vector3): void {
    const avatar = this.avatars.get(id);

    if (!avatar) {
      console.warn(`[CollisionDetector] Avatar ${id} not registered`);
      return;
    }

    avatar.position.copy(position);
    avatar.lastUpdate = Date.now();
  }

  /**
   * Unregister avatar
   */
  unregisterAvatar(id: string): void {
    this.avatars.delete(id);

    // Remove all pair states involving this avatar
    const keysToRemove: string[] = [];
    this.pairStates.forEach((state, key) => {
      if (key.includes(id)) {
        keysToRemove.push(key);
      }
    });
    keysToRemove.forEach((key) => this.pairStates.delete(key));

    if (this.config.debug) {
      console.log(`[CollisionDetector] Unregistered avatar ${id}`);
    }
  }

  /**
   * Check all collisions
   * Call this every frame
   */
  checkCollisions(): CollisionEvent[] {
    const collisions: CollisionEvent[] = [];
    const avatarArray = Array.from(this.avatars.values());

    // Check all pairs (N^2 but acceptable for small N)
    for (let i = 0; i < avatarArray.length; i++) {
      for (let j = i + 1; j < avatarArray.length; j++) {
        const avatarA = avatarArray[i];
        const avatarB = avatarArray[j];

        // Check collision
        const result = this.checkPairCollision(avatarA, avatarB);

        if (result) {
          collisions.push(result);
        }
      }
    }

    return collisions;
  }

  /**
   * Check collision between two avatars
   */
  private checkPairCollision(
    avatarA: AvatarCollisionData,
    avatarB: AvatarCollisionData
  ): CollisionEvent | null {
    const pairKey = this.getPairKey(avatarA.id, avatarB.id);

    // Calculate distance
    const distance = avatarA.position.distanceTo(avatarB.position);

    // Get or create pair state
    let pairState = this.pairStates.get(pairKey);
    if (!pairState) {
      pairState = {
        colliding: false,
        proximity: this.getProximityZone(distance),
        lastDistance: distance,
      };
      this.pairStates.set(pairKey, pairState);
    }

    // Check collision (sphere-sphere intersection)
    const combinedRadius = avatarA.radius + avatarB.radius;
    const isColliding = distance < combinedRadius;

    // Detect collision enter/exit
    if (isColliding && !pairState.colliding) {
      // Collision entered
      pairState.colliding = true;
      this.collisionCount++;

      const event: CollisionEvent = {
        avatarA: avatarA.id,
        avatarB: avatarB.id,
        point: avatarA.position.clone().lerp(avatarB.position, 0.5),
        distance,
        penetration: combinedRadius - distance,
        normal: avatarB.position.clone().sub(avatarA.position).normalize(),
        timestamp: Date.now(),
      };

      this.emit('collision-enter', event);
      this.emit('collision', event);

      if (this.config.debug) {
        console.log(`[CollisionDetector] Collision: ${avatarA.id} <-> ${avatarB.id} (distance: ${distance.toFixed(2)}m, penetration: ${event.penetration.toFixed(2)}m)`);
      }

      return event;
    } else if (!isColliding && pairState.colliding) {
      // Collision exited
      pairState.colliding = false;

      const event: CollisionEvent = {
        avatarA: avatarA.id,
        avatarB: avatarB.id,
        point: avatarA.position.clone().lerp(avatarB.position, 0.5),
        distance,
        penetration: 0,
        normal: avatarB.position.clone().sub(avatarA.position).normalize(),
        timestamp: Date.now(),
      };

      this.emit('collision-exit', event);

      if (this.config.debug) {
        console.log(`[CollisionDetector] Collision exit: ${avatarA.id} <-> ${avatarB.id}`);
      }
    }

    // Check proximity zone changes
    if (this.config.enableProximity) {
      const currentZone = this.getProximityZone(distance);
      const previousZone = pairState.proximity;

      if (currentZone !== previousZone) {
        pairState.proximity = currentZone;
        this.proximityChangeCount++;

        const event: ProximityEvent = {
          avatarA: avatarA.id,
          avatarB: avatarB.id,
          zone: currentZone,
          distance,
          previousZone,
          timestamp: Date.now(),
        };

        this.emit('proximity-change', event);

        if (this.config.debug) {
          console.log(`[CollisionDetector] Proximity change: ${avatarA.id} <-> ${avatarB.id}: ${previousZone} → ${currentZone}`);
        }
      }
    }

    pairState.lastDistance = distance;

    return isColliding
      ? {
          avatarA: avatarA.id,
          avatarB: avatarB.id,
          point: avatarA.position.clone().lerp(avatarB.position, 0.5),
          distance,
          penetration: combinedRadius - distance,
          normal: avatarB.position.clone().sub(avatarA.position).normalize(),
          timestamp: Date.now(),
        }
      : null;
  }

  /**
   * Get proximity zone for distance
   */
  private getProximityZone(distance: number): ProximityZone {
    const { near, medium, far } = this.config.proximityThresholds;

    if (distance < near) return ProximityZone.NEAR;
    if (distance < medium) return ProximityZone.MEDIUM;
    if (distance < far) return ProximityZone.FAR;
    return ProximityZone.DISTANT;
  }

  /**
   * Get pair key (sorted to ensure uniqueness)
   */
  private getPairKey(idA: string, idB: string): string {
    return idA < idB ? `${idA}:${idB}` : `${idB}:${idA}`;
  }

  /**
   * Get avatars in proximity to target
   */
  getAvatarsInProximity(
    targetId: string,
    zone: ProximityZone
  ): string[] {
    const target = this.avatars.get(targetId);
    if (!target) return [];

    const threshold = this.getZoneThreshold(zone);
    const nearbyAvatars: string[] = [];

    this.avatars.forEach((avatar, id) => {
      if (id === targetId) return;

      const distance = target.position.distanceTo(avatar.position);
      if (distance < threshold) {
        nearbyAvatars.push(id);
      }
    });

    return nearbyAvatars;
  }

  /**
   * Get threshold for proximity zone
   */
  private getZoneThreshold(zone: ProximityZone): number {
    const { near, medium, far } = this.config.proximityThresholds;

    switch (zone) {
      case ProximityZone.NEAR:
        return near;
      case ProximityZone.MEDIUM:
        return medium;
      case ProximityZone.FAR:
        return far;
      case ProximityZone.DISTANT:
        return Infinity;
    }
  }

  /**
   * Get metrics
   */
  getMetrics(): {
    avatarCount: number;
    collisionsPerSecond: number;
    proximityChangesPerSecond: number;
    activePairs: number;
  } {
    const now = Date.now();
    const elapsedSeconds = (now - this.lastMetricsReset) / 1000;

    const collisionsPerSecond =
      elapsedSeconds > 0 ? this.collisionCount / elapsedSeconds : 0;
    const proximityChangesPerSecond =
      elapsedSeconds > 0 ? this.proximityChangeCount / elapsedSeconds : 0;

    // Reset counters every second
    if (elapsedSeconds >= 1.0) {
      this.collisionCount = 0;
      this.proximityChangeCount = 0;
      this.lastMetricsReset = now;
    }

    return {
      avatarCount: this.avatars.size,
      collisionsPerSecond,
      proximityChangesPerSecond,
      activePairs: this.pairStates.size,
    };
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.avatars.clear();
    this.pairStates.clear();
    this.collisionCount = 0;
    this.proximityChangeCount = 0;
    this.lastMetricsReset = Date.now();
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
}

/**
 * Create collision detector
 */
export function createCollisionDetector(
  config?: CollisionDetectorConfig
): CollisionDetector {
  return new CollisionDetector(config);
}
