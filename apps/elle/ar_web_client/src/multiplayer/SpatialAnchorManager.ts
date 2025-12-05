/**
 * SpatialAnchorManager - Coordinate System Alignment for Multi-User AR
 *
 * Handles spatial anchor placement and coordinate transformation between users.
 * Ensures all users see avatars in consistent world positions.
 *
 * Features:
 * - QR code-based anchor placement
 * - Coordinate system transformation
 * - Anchor persistence
 * - Multi-user synchronization
 *
 * Workflow:
 * 1. Host places anchor (QR code or manual position)
 * 2. Anchor position shared via WebRTC
 * 3. Guests scan same anchor to align coordinate systems
 * 4. All positions transformed to anchor-relative coordinates
 *
 * Created: 2025-11-22 (Phase 6.2)
 */

import * as THREE from 'three';

/**
 * Spatial anchor data
 */
export interface SpatialAnchor {
  /**
   * Unique anchor ID
   */
  id: string;

  /**
   * Anchor type
   */
  type: 'qr-code' | 'manual' | 'image-marker';

  /**
   * Position in world space
   */
  position: THREE.Vector3;

  /**
   * Rotation in world space (quaternion)
   */
  rotation: THREE.Quaternion;

  /**
   * Scale (usually 1.0)
   */
  scale: number;

  /**
   * Anchor data (QR code string, image URL, etc.)
   */
  data: string;

  /**
   * Creation timestamp
   */
  timestamp: number;

  /**
   * Creator user ID
   */
  createdBy: string;
}

/**
 * Coordinate transformation
 */
export interface CoordinateTransform {
  /**
   * Translation vector
   */
  translation: THREE.Vector3;

  /**
   * Rotation quaternion
   */
  rotation: THREE.Quaternion;

  /**
   * Scale factor
   */
  scale: number;
}

/**
 * Spatial anchor manager configuration
 */
export interface SpatialAnchorConfig {
  /**
   * Enable anchor persistence (localStorage)
   */
  enablePersistence?: boolean;

  /**
   * Storage key for persistence
   */
  storageKey?: string;

  /**
   * Enable automatic anchor detection
   */
  enableAutoDetection?: boolean;

  /**
   * QR code detection frequency (Hz)
   */
  detectionRate?: number;
}

/**
 * SpatialAnchorManager - Manages spatial anchors for multi-user AR
 *
 * Usage:
 * ```typescript
 * const anchorManager = new SpatialAnchorManager({
 *   enablePersistence: true,
 *   enableAutoDetection: true,
 * });
 *
 * // Host: Create anchor
 * const anchor = await anchorManager.createAnchor({
 *   type: 'qr-code',
 *   position: new THREE.Vector3(0, 0, 0),
 *   rotation: new THREE.Quaternion(),
 *   data: 'anchor-code-123',
 * });
 *
 * // Guest: Detect anchor
 * const detected = await anchorManager.detectAnchor(videoElement);
 * if (detected) {
 *   await anchorManager.alignToAnchor(detected.id);
 * }
 *
 * // Transform coordinates
 * const worldPos = new THREE.Vector3(1, 2, 3);
 * const anchorRelative = anchorManager.worldToAnchor(worldPos);
 * ```
 */
export class SpatialAnchorManager {
  private config: Required<SpatialAnchorConfig>;

  // Anchors
  private anchors: Map<string, SpatialAnchor> = new Map();
  private activeAnchor: SpatialAnchor | null = null;

  // Transformation
  private anchorTransform: CoordinateTransform | null = null;

  // QR detection (lazy-loaded)
  private qrScanner: any = null;

  // Event handlers
  private eventHandlers: Map<string, Function[]> = new Map();

  constructor(config: SpatialAnchorConfig = {}) {
    this.config = {
      enablePersistence: config.enablePersistence ?? true,
      storageKey: config.storageKey ?? 'spatial-anchors',
      enableAutoDetection: config.enableAutoDetection ?? false,
      detectionRate: config.detectionRate ?? 5,  // 5 Hz
    };

    // Load persisted anchors
    if (this.config.enablePersistence) {
      this.loadAnchors();
    }
  }

  /**
   * Create new spatial anchor
   */
  async createAnchor(params: {
    type: SpatialAnchor['type'];
    position: THREE.Vector3;
    rotation: THREE.Quaternion;
    data: string;
    createdBy: string;
  }): Promise<SpatialAnchor> {
    const anchor: SpatialAnchor = {
      id: this.generateAnchorId(),
      type: params.type,
      position: params.position.clone(),
      rotation: params.rotation.clone(),
      scale: 1.0,
      data: params.data,
      timestamp: Date.now(),
      createdBy: params.createdBy,
    };

    this.anchors.set(anchor.id, anchor);

    // Persist
    if (this.config.enablePersistence) {
      this.saveAnchors();
    }

    console.log(`[SpatialAnchorManager] Created anchor ${anchor.id}`);

    this.emit('anchor-created', anchor);

    return anchor;
  }

  /**
   * Set active anchor for coordinate transformation
   */
  async alignToAnchor(anchorId: string): Promise<void> {
    const anchor = this.anchors.get(anchorId);

    if (!anchor) {
      throw new Error(`Anchor ${anchorId} not found`);
    }

    this.activeAnchor = anchor;

    // Calculate transformation from world to anchor space
    this.anchorTransform = {
      translation: anchor.position.clone().negate(),
      rotation: anchor.rotation.clone().invert(),
      scale: 1.0 / anchor.scale,
    };

    console.log(`[SpatialAnchorManager] Aligned to anchor ${anchorId}`);

    this.emit('anchor-aligned', anchor);
  }

  /**
   * Detect anchor from video feed (QR code)
   */
  async detectAnchor(videoElement: HTMLVideoElement): Promise<SpatialAnchor | null> {
    // Lazy-load QR scanner
    if (!this.qrScanner) {
      try {
        const jsQR = await import('jsqr');
        this.qrScanner = jsQR.default;
      } catch (error) {
        console.warn('[SpatialAnchorManager] QR scanner not available:', error);
        return null;
      }
    }

    // Get image data from video
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(videoElement, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Scan for QR code
    const code = this.qrScanner(
      imageData.data,
      imageData.width,
      imageData.height,
      {
        inversionAttempts: 'dontInvert',
      }
    );

    if (code) {
      console.log('[SpatialAnchorManager] QR code detected:', code.data);

      // Look for matching anchor
      for (const anchor of this.anchors.values()) {
        if (anchor.type === 'qr-code' && anchor.data === code.data) {
          return anchor;
        }
      }
    }

    return null;
  }

  /**
   * Start automatic anchor detection
   */
  startAutoDetection(videoElement: HTMLVideoElement): void {
    if (!this.config.enableAutoDetection) {
      console.warn('[SpatialAnchorManager] Auto-detection not enabled');
      return;
    }

    const interval = 1000 / this.config.detectionRate;

    const detect = async () => {
      const anchor = await this.detectAnchor(videoElement);

      if (anchor) {
        console.log('[SpatialAnchorManager] Auto-detected anchor:', anchor.id);
        await this.alignToAnchor(anchor.id);

        this.emit('anchor-detected', anchor);
      }

      setTimeout(detect, interval);
    };

    detect();
  }

  /**
   * Transform position from world space to anchor space
   */
  worldToAnchor(worldPos: THREE.Vector3): THREE.Vector3 {
    if (!this.anchorTransform) {
      console.warn('[SpatialAnchorManager] No active anchor, returning world position');
      return worldPos.clone();
    }

    const transformed = worldPos.clone();

    // Apply translation
    transformed.add(this.anchorTransform.translation);

    // Apply rotation
    transformed.applyQuaternion(this.anchorTransform.rotation);

    // Apply scale
    transformed.multiplyScalar(this.anchorTransform.scale);

    return transformed;
  }

  /**
   * Transform position from anchor space to world space
   */
  anchorToWorld(anchorPos: THREE.Vector3): THREE.Vector3 {
    if (!this.anchorTransform || !this.activeAnchor) {
      console.warn('[SpatialAnchorManager] No active anchor, returning anchor position');
      return anchorPos.clone();
    }

    const transformed = anchorPos.clone();

    // Reverse scale
    transformed.multiplyScalar(1.0 / this.anchorTransform.scale);

    // Reverse rotation
    const inverseRotation = this.anchorTransform.rotation.clone().invert();
    transformed.applyQuaternion(inverseRotation);

    // Reverse translation
    transformed.sub(this.anchorTransform.translation);

    return transformed;
  }

  /**
   * Transform rotation from world space to anchor space
   */
  worldRotationToAnchor(worldRot: THREE.Quaternion): THREE.Quaternion {
    if (!this.anchorTransform) {
      return worldRot.clone();
    }

    const transformed = worldRot.clone();
    transformed.multiply(this.anchorTransform.rotation);

    return transformed;
  }

  /**
   * Transform rotation from anchor space to world space
   */
  anchorRotationToWorld(anchorRot: THREE.Quaternion): THREE.Quaternion {
    if (!this.anchorTransform) {
      return anchorRot.clone();
    }

    const inverseRotation = this.anchorTransform.rotation.clone().invert();
    const transformed = anchorRot.clone();
    transformed.multiply(inverseRotation);

    return transformed;
  }

  /**
   * Get active anchor
   */
  getActiveAnchor(): SpatialAnchor | null {
    return this.activeAnchor;
  }

  /**
   * Get all anchors
   */
  getAllAnchors(): SpatialAnchor[] {
    return Array.from(this.anchors.values());
  }

  /**
   * Remove anchor
   */
  removeAnchor(anchorId: string): void {
    this.anchors.delete(anchorId);

    if (this.activeAnchor && this.activeAnchor.id === anchorId) {
      this.activeAnchor = null;
      this.anchorTransform = null;
    }

    if (this.config.enablePersistence) {
      this.saveAnchors();
    }

    console.log(`[SpatialAnchorManager] Removed anchor ${anchorId}`);
  }

  /**
   * Generate unique anchor ID
   */
  private generateAnchorId(): string {
    return `anchor-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Save anchors to localStorage
   */
  private saveAnchors(): void {
    try {
      const serialized = JSON.stringify(
        Array.from(this.anchors.entries()),
        (key, value) => {
          // Serialize Vector3 and Quaternion
          if (value instanceof THREE.Vector3) {
            return { __type: 'Vector3', x: value.x, y: value.y, z: value.z };
          }
          if (value instanceof THREE.Quaternion) {
            return { __type: 'Quaternion', x: value.x, y: value.y, z: value.z, w: value.w };
          }
          return value;
        }
      );

      localStorage.setItem(this.config.storageKey, serialized);
    } catch (error) {
      console.error('[SpatialAnchorManager] Failed to save anchors:', error);
    }
  }

  /**
   * Load anchors from localStorage
   */
  private loadAnchors(): void {
    try {
      const serialized = localStorage.getItem(this.config.storageKey);

      if (!serialized) return;

      const entries = JSON.parse(serialized, (key, value) => {
        // Deserialize Vector3 and Quaternion
        if (value && value.__type === 'Vector3') {
          return new THREE.Vector3(value.x, value.y, value.z);
        }
        if (value && value.__type === 'Quaternion') {
          return new THREE.Quaternion(value.x, value.y, value.z, value.w);
        }
        return value;
      });

      this.anchors = new Map(entries);

      console.log(`[SpatialAnchorManager] Loaded ${this.anchors.size} anchors from storage`);
    } catch (error) {
      console.error('[SpatialAnchorManager] Failed to load anchors:', error);
    }
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
 * Create spatial anchor manager
 */
export function createSpatialAnchorManager(
  config?: SpatialAnchorConfig
): SpatialAnchorManager {
  return new SpatialAnchorManager(config);
}
