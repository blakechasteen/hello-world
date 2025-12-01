/**
 * Cursor Presence System - Phase 3: Collaborative Intelligence
 *
 * Real-time cursor tracking and presence broadcasting via WebSocket.
 * Allows users to see each other's cursors and activity in the graph.
 *
 * Features:
 * - Smooth cursor interpolation for fluid movement
 * - Adaptive broadcast rate based on movement speed
 * - Focus tracking (what node/edge is being viewed)
 * - Connection status and reconnection handling
 *
 * @module collaboration/presence
 */

// ============================================================================
// Types
// ============================================================================

/**
 * Cursor position with metadata
 */
export interface CursorState {
  x: number;
  y: number;
  nodeId?: string;
  timestamp: number;
}

/**
 * User focus state
 */
export interface FocusState {
  type: 'node' | 'edge' | 'region' | 'none';
  targetId?: string;
  bounds?: { x: number; y: number; width: number; height: number };
}

/**
 * Complete presence state for a user
 */
export interface PresenceState {
  userId: string;
  name: string;
  color: string;
  avatar?: string;
  cursor?: CursorState;
  focus?: FocusState;
  status: 'active' | 'idle' | 'away';
  lastActivity: number;
}

/**
 * Interpolated cursor for smooth rendering
 */
export interface InterpolatedCursor {
  current: { x: number; y: number };
  target: { x: number; y: number };
  user: {
    userId: string;
    name: string;
    color: string;
    avatar?: string;
  };
  nodeId?: string;
  visible: boolean;
}

/**
 * Presence system configuration
 */
export interface PresenceConfig {
  /** WebSocket server URL */
  serverUrl: string;
  /** Room/session ID */
  roomId: string;
  /** User information */
  user: {
    userId: string;
    name: string;
    color: string;
    avatar?: string;
  };
  /** Cursor update throttle (ms) */
  cursorThrottleMs?: number;
  /** Idle timeout (ms) */
  idleTimeoutMs?: number;
  /** Away timeout (ms) */
  awayTimeoutMs?: number;
  /** Cursor interpolation smoothness (0-1) */
  interpolationFactor?: number;
  /** Maximum cursor age before hiding (ms) */
  maxCursorAge?: number;
}

/**
 * WebSocket message types
 */
type PresenceMessage =
  | { type: 'join'; payload: { user: PresenceState } }
  | { type: 'leave'; payload: { userId: string } }
  | { type: 'cursor'; payload: { userId: string; cursor: CursorState } }
  | { type: 'focus'; payload: { userId: string; focus: FocusState } }
  | { type: 'status'; payload: { userId: string; status: 'active' | 'idle' | 'away' } }
  | { type: 'sync'; payload: { users: PresenceState[] } }
  | { type: 'ping' }
  | { type: 'pong'; payload: { timestamp: number } };

/**
 * Event types emitted by PresenceManager
 */
export type PresenceEvent =
  | { type: 'connected' }
  | { type: 'disconnected' }
  | { type: 'userJoined'; user: PresenceState }
  | { type: 'userLeft'; userId: string }
  | { type: 'cursorMoved'; userId: string; cursor: CursorState }
  | { type: 'focusChanged'; userId: string; focus: FocusState }
  | { type: 'statusChanged'; userId: string; status: 'active' | 'idle' | 'away' }
  | { type: 'usersSync'; users: PresenceState[] }
  | { type: 'error'; error: Error };

type PresenceEventHandler = (event: PresenceEvent) => void;

// ============================================================================
// Presence Manager
// ============================================================================

/**
 * Manages real-time cursor presence via WebSocket
 */
export class PresenceManager {
  private config: Required<PresenceConfig>;
  private ws: WebSocket | null = null;
  private handlers: Set<PresenceEventHandler> = new Set();

  // State
  private users: Map<string, PresenceState> = new Map();
  private interpolatedCursors: Map<string, InterpolatedCursor> = new Map();

  // Throttling
  private lastCursorBroadcast = 0;
  private pendingCursor: CursorState | null = null;
  private cursorBroadcastTimer: ReturnType<typeof setTimeout> | null = null;

  // Activity tracking
  private lastActivity = Date.now();
  private activityTimer: ReturnType<typeof setInterval> | null = null;
  private currentStatus: 'active' | 'idle' | 'away' = 'active';

  // Animation
  private animationFrame: number | null = null;
  private lastInterpolationTime = 0;

  // Connection
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(config: PresenceConfig) {
    this.config = {
      serverUrl: config.serverUrl,
      roomId: config.roomId,
      user: config.user,
      cursorThrottleMs: config.cursorThrottleMs ?? 50,
      idleTimeoutMs: config.idleTimeoutMs ?? 60000,
      awayTimeoutMs: config.awayTimeoutMs ?? 300000,
      interpolationFactor: config.interpolationFactor ?? 0.15,
      maxCursorAge: config.maxCursorAge ?? 5000,
    };
  }

  /**
   * Connect to the presence server
   */
  connect(): void {
    if (this.ws) {
      this.disconnect();
    }

    try {
      const url = `${this.config.serverUrl}?room=${this.config.roomId}&userId=${this.config.user.userId}`;
      this.ws = new WebSocket(url);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);
    } catch (error) {
      this.emit({ type: 'error', error: error as Error });
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from the presence server
   */
  disconnect(): void {
    if (this.activityTimer) {
      clearInterval(this.activityTimer);
      this.activityTimer = null;
    }

    if (this.cursorBroadcastTimer) {
      clearTimeout(this.cursorBroadcastTimer);
      this.cursorBroadcastTimer = null;
    }

    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.users.clear();
    this.interpolatedCursors.clear();
    this.emit({ type: 'disconnected' });
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get all users
   */
  getUsers(): PresenceState[] {
    return Array.from(this.users.values());
  }

  /**
   * Get other users (excluding self)
   */
  getOtherUsers(): PresenceState[] {
    return this.getUsers().filter((u) => u.userId !== this.config.user.userId);
  }

  /**
   * Get user by ID
   */
  getUser(userId: string): PresenceState | undefined {
    return this.users.get(userId);
  }

  /**
   * Get interpolated cursors for rendering
   */
  getInterpolatedCursors(): InterpolatedCursor[] {
    const now = Date.now();
    const cursors: InterpolatedCursor[] = [];

    this.interpolatedCursors.forEach((cursor, userId) => {
      if (userId === this.config.user.userId) return;

      const user = this.users.get(userId);
      if (!user?.cursor) return;

      // Hide stale cursors
      const age = now - user.cursor.timestamp;
      const visible = age < this.config.maxCursorAge;

      cursors.push({ ...cursor, visible });
    });

    return cursors;
  }

  // ============================================================================
  // Cursor Broadcasting
  // ============================================================================

  /**
   * Update local cursor position
   */
  updateCursor(x: number, y: number, nodeId?: string): void {
    this.markActivity();

    const cursor: CursorState = {
      x,
      y,
      nodeId,
      timestamp: Date.now(),
    };

    this.pendingCursor = cursor;
    this.scheduleCursorBroadcast();
  }

  /**
   * Update local focus
   */
  updateFocus(focus: FocusState): void {
    this.markActivity();
    this.send({ type: 'focus', payload: { userId: this.config.user.userId, focus } });
  }

  /**
   * Schedule throttled cursor broadcast
   */
  private scheduleCursorBroadcast(): void {
    if (this.cursorBroadcastTimer) return;

    const timeSinceLastBroadcast = Date.now() - this.lastCursorBroadcast;
    const delay = Math.max(0, this.config.cursorThrottleMs - timeSinceLastBroadcast);

    this.cursorBroadcastTimer = setTimeout(() => {
      this.cursorBroadcastTimer = null;

      if (this.pendingCursor) {
        this.send({
          type: 'cursor',
          payload: { userId: this.config.user.userId, cursor: this.pendingCursor },
        });
        this.lastCursorBroadcast = Date.now();
        this.pendingCursor = null;
      }
    }, delay);
  }

  // ============================================================================
  // Activity Tracking
  // ============================================================================

  /**
   * Mark user as active
   */
  private markActivity(): void {
    this.lastActivity = Date.now();

    if (this.currentStatus !== 'active') {
      this.currentStatus = 'active';
      this.send({
        type: 'status',
        payload: { userId: this.config.user.userId, status: 'active' },
      });
    }
  }

  /**
   * Start activity monitoring
   */
  private startActivityMonitoring(): void {
    this.activityTimer = setInterval(() => {
      const timeSinceActivity = Date.now() - this.lastActivity;

      let newStatus: 'active' | 'idle' | 'away' = 'active';
      if (timeSinceActivity > this.config.awayTimeoutMs) {
        newStatus = 'away';
      } else if (timeSinceActivity > this.config.idleTimeoutMs) {
        newStatus = 'idle';
      }

      if (newStatus !== this.currentStatus) {
        this.currentStatus = newStatus;
        this.send({
          type: 'status',
          payload: { userId: this.config.user.userId, status: newStatus },
        });
      }
    }, 10000);
  }

  // ============================================================================
  // Cursor Interpolation
  // ============================================================================

  /**
   * Start interpolation animation loop
   */
  private startInterpolation(): void {
    const animate = (time: number) => {
      const deltaTime = time - this.lastInterpolationTime;
      this.lastInterpolationTime = time;

      // Interpolate each cursor
      this.interpolatedCursors.forEach((cursor, userId) => {
        const user = this.users.get(userId);
        if (!user?.cursor) return;

        // Calculate adaptive interpolation based on distance
        const dx = cursor.target.x - cursor.current.x;
        const dy = cursor.target.y - cursor.current.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // Faster interpolation for larger distances
        const adaptiveFactor =
          distance > 100
            ? Math.min(1, this.config.interpolationFactor * 3)
            : this.config.interpolationFactor;

        // Time-based interpolation for consistent movement
        const timeScale = Math.min(1, deltaTime / 16.67); // Normalize to 60fps

        cursor.current.x += dx * adaptiveFactor * timeScale;
        cursor.current.y += dy * adaptiveFactor * timeScale;
      });

      this.animationFrame = requestAnimationFrame(animate);
    };

    this.lastInterpolationTime = performance.now();
    this.animationFrame = requestAnimationFrame(animate);
  }

  /**
   * Update interpolation target for a user
   */
  private updateInterpolationTarget(userId: string, cursor: CursorState): void {
    const existing = this.interpolatedCursors.get(userId);
    const user = this.users.get(userId);

    if (!user) return;

    if (existing) {
      existing.target.x = cursor.x;
      existing.target.y = cursor.y;
      existing.nodeId = cursor.nodeId;
    } else {
      this.interpolatedCursors.set(userId, {
        current: { x: cursor.x, y: cursor.y },
        target: { x: cursor.x, y: cursor.y },
        user: {
          userId: user.userId,
          name: user.name,
          color: user.color,
          avatar: user.avatar,
        },
        nodeId: cursor.nodeId,
        visible: true,
      });
    }
  }

  // ============================================================================
  // WebSocket Handlers
  // ============================================================================

  private handleOpen(): void {
    this.reconnectAttempts = 0;

    // Send join message
    const presence: PresenceState = {
      ...this.config.user,
      status: 'active',
      lastActivity: Date.now(),
    };

    this.send({ type: 'join', payload: { user: presence } });

    // Start activity monitoring and interpolation
    this.startActivityMonitoring();
    this.startInterpolation();

    this.emit({ type: 'connected' });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data) as PresenceMessage;

      switch (message.type) {
        case 'join':
          this.handleUserJoined(message.payload.user);
          break;

        case 'leave':
          this.handleUserLeft(message.payload.userId);
          break;

        case 'cursor':
          this.handleCursorUpdate(message.payload.userId, message.payload.cursor);
          break;

        case 'focus':
          this.handleFocusUpdate(message.payload.userId, message.payload.focus);
          break;

        case 'status':
          this.handleStatusUpdate(message.payload.userId, message.payload.status);
          break;

        case 'sync':
          this.handleSync(message.payload.users);
          break;

        case 'ping':
          this.send({ type: 'pong', payload: { timestamp: Date.now() } });
          break;
      }
    } catch (error) {
      console.error('Error parsing presence message:', error);
    }
  }

  private handleClose(): void {
    this.emit({ type: 'disconnected' });
    this.scheduleReconnect();
  }

  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    this.emit({ type: 'error', error: new Error('WebSocket error') });
  }

  // ============================================================================
  // Message Handlers
  // ============================================================================

  private handleUserJoined(user: PresenceState): void {
    this.users.set(user.userId, user);
    this.emit({ type: 'userJoined', user });
  }

  private handleUserLeft(userId: string): void {
    this.users.delete(userId);
    this.interpolatedCursors.delete(userId);
    this.emit({ type: 'userLeft', userId });
  }

  private handleCursorUpdate(userId: string, cursor: CursorState): void {
    const user = this.users.get(userId);
    if (user) {
      user.cursor = cursor;
      user.lastActivity = Date.now();
      this.updateInterpolationTarget(userId, cursor);
    }
    this.emit({ type: 'cursorMoved', userId, cursor });
  }

  private handleFocusUpdate(userId: string, focus: FocusState): void {
    const user = this.users.get(userId);
    if (user) {
      user.focus = focus;
      user.lastActivity = Date.now();
    }
    this.emit({ type: 'focusChanged', userId, focus });
  }

  private handleStatusUpdate(userId: string, status: 'active' | 'idle' | 'away'): void {
    const user = this.users.get(userId);
    if (user) {
      user.status = status;
    }
    this.emit({ type: 'statusChanged', userId, status });
  }

  private handleSync(users: PresenceState[]): void {
    this.users.clear();
    users.forEach((user) => {
      this.users.set(user.userId, user);
      if (user.cursor) {
        this.updateInterpolationTarget(user.userId, user.cursor);
      }
    });
    this.emit({ type: 'usersSync', users });
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  private send(message: PresenceMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.emit({
        type: 'error',
        error: new Error('Max reconnection attempts reached'),
      });
      return;
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  /**
   * Subscribe to presence events
   */
  subscribe(handler: PresenceEventHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  private emit(event: PresenceEvent): void {
    this.handlers.forEach((handler) => {
      try {
        handler(event);
      } catch (error) {
        console.error('Presence event handler error:', error);
      }
    });
  }
}

// ============================================================================
// Cursor Renderer Utilities
// ============================================================================

/**
 * Generate SVG path for cursor shape
 */
export function getCursorPath(size = 24): string {
  // Classic pointer cursor shape
  return `M0,0 L${size},${size * 0.75} L${size * 0.5},${size * 0.75} L${
    size * 0.5
  },${size} L0,0`;
}

/**
 * Generate user initials for avatar fallback
 */
export function getInitials(name: string): string {
  return name
    .split(' ')
    .map((part) => part[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
}

/**
 * Generate a color from user ID (consistent colors)
 */
export function getUserColor(userId: string): string {
  // Predefined palette of distinguishable colors
  const colors = [
    '#EF4444', // Red
    '#F97316', // Orange
    '#EAB308', // Yellow
    '#22C55E', // Green
    '#14B8A6', // Teal
    '#3B82F6', // Blue
    '#8B5CF6', // Purple
    '#EC4899', // Pink
    '#6366F1', // Indigo
    '#F59E0B', // Amber
  ];

  // Hash the user ID to get a consistent color
  let hash = 0;
  for (let i = 0; i < userId.length; i++) {
    hash = (hash << 5) - hash + userId.charCodeAt(i);
    hash |= 0;
  }

  return colors[Math.abs(hash) % colors.length];
}

// ============================================================================
// React Hooks (for convenience)
// ============================================================================

/**
 * Create presence manager instance
 */
export function createPresenceManager(config: PresenceConfig): PresenceManager {
  return new PresenceManager(config);
}