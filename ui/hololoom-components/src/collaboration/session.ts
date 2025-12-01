/**
 * Collaborative Session Manager - Phase 3: Collaborative Intelligence
 *
 * Manages collaborative sessions for the Shared Memory Palace.
 * Coordinates CRDT sync, presence, and annotations into a unified experience.
 *
 * @module collaboration/session
 */

import { CRDTSyncManager, type SyncConfig, type SyncEvent } from './sync';
import { PresenceManager, type PresenceConfig, type PresenceEvent, getUserColor } from './presence';
import { AnnotationManager, type AnnotationEvent } from './attribution';

// ============================================================================
// Types
// ============================================================================

/**
 * Session participant
 */
export interface SessionParticipant {
  userId: string;
  name: string;
  color: string;
  avatar?: string;
  role: 'owner' | 'editor' | 'viewer';
  joinedAt: number;
  status: 'active' | 'idle' | 'away';
}

/**
 * Session information
 */
export interface SessionInfo {
  sessionId: string;
  name: string;
  description?: string;
  ownerId: string;
  ownerName: string;
  createdAt: number;
  lastActivity: number;
  mode: 'edit' | 'view' | 'present';
  locked: boolean;
  participantCount: number;
  maxParticipants: number;
}

/**
 * Session configuration
 */
export interface SessionConfig {
  /** WebSocket server URL for sync */
  syncServerUrl: string;
  /** WebSocket server URL for presence */
  presenceServerUrl: string;
  /** Current user information */
  user: {
    userId: string;
    name: string;
    avatar?: string;
  };
  /** Session settings */
  settings?: {
    maxParticipants?: number;
    autoSave?: boolean;
    autoSaveInterval?: number;
  };
}

/**
 * Session state
 */
export interface SessionState {
  /** Connection status */
  status: 'disconnected' | 'connecting' | 'connected' | 'syncing' | 'ready';
  /** Current session info */
  session: SessionInfo | null;
  /** Session participants */
  participants: Map<string, SessionParticipant>;
  /** Current user's role */
  role: 'owner' | 'editor' | 'viewer' | null;
  /** Error message if any */
  error: string | null;
  /** Last sync timestamp */
  lastSync: number | null;
}

/**
 * Session events
 */
export type SessionEvent =
  | { type: 'statusChanged'; status: SessionState['status'] }
  | { type: 'sessionJoined'; session: SessionInfo }
  | { type: 'sessionLeft' }
  | { type: 'participantJoined'; participant: SessionParticipant }
  | { type: 'participantLeft'; userId: string }
  | { type: 'participantUpdated'; participant: SessionParticipant }
  | { type: 'modeChanged'; mode: 'edit' | 'view' | 'present' }
  | { type: 'error'; error: string }
  | { type: 'synced' }
  | SyncEvent
  | PresenceEvent
  | AnnotationEvent;

type SessionEventHandler = (event: SessionEvent) => void;

// ============================================================================
// Collaborative Session Manager
// ============================================================================

/**
 * Manages a collaborative session
 */
export class CollaborativeSessionManager {
  private config: SessionConfig;
  private state: SessionState;
  private handlers: Set<SessionEventHandler> = new Set();

  // Sub-managers
  private syncManager: CRDTSyncManager | null = null;
  private presenceManager: PresenceManager | null = null;
  private annotationManager: AnnotationManager;

  // Unsubscribe functions
  private syncUnsubscribe: (() => void) | null = null;
  private presenceUnsubscribe: (() => void) | null = null;
  private annotationUnsubscribe: (() => void) | null = null;

  constructor(config: SessionConfig) {
    this.config = {
      ...config,
      settings: {
        maxParticipants: config.settings?.maxParticipants ?? 10,
        autoSave: config.settings?.autoSave ?? true,
        autoSaveInterval: config.settings?.autoSaveInterval ?? 30000,
      },
    };

    this.state = {
      status: 'disconnected',
      session: null,
      participants: new Map(),
      role: null,
      error: null,
      lastSync: null,
    };

    // Initialize annotation manager with user info
    this.annotationManager = new AnnotationManager({
      userId: config.user.userId,
      name: config.user.name,
      color: getUserColor(config.user.userId),
    });
  }

  // ============================================================================
  // Session Lifecycle
  // ============================================================================

  /**
   * Create and join a new session
   */
  async createSession(name: string, description?: string): Promise<SessionInfo> {
    this.updateStatus('connecting');

    try {
      const sessionId = this.generateSessionId();
      const now = Date.now();

      const session: SessionInfo = {
        sessionId,
        name,
        description,
        ownerId: this.config.user.userId,
        ownerName: this.config.user.name,
        createdAt: now,
        lastActivity: now,
        mode: 'edit',
        locked: false,
        participantCount: 1,
        maxParticipants: this.config.settings!.maxParticipants!,
      };

      // Initialize managers
      await this.initializeManagers(sessionId);

      // Add self as participant
      const selfParticipant: SessionParticipant = {
        userId: this.config.user.userId,
        name: this.config.user.name,
        color: getUserColor(this.config.user.userId),
        avatar: this.config.user.avatar,
        role: 'owner',
        joinedAt: now,
        status: 'active',
      };

      this.state.participants.set(this.config.user.userId, selfParticipant);
      this.state.session = session;
      this.state.role = 'owner';

      this.updateStatus('ready');
      this.emit({ type: 'sessionJoined', session });

      return session;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create session';
      this.state.error = errorMessage;
      this.updateStatus('disconnected');
      this.emit({ type: 'error', error: errorMessage });
      throw error;
    }
  }

  /**
   * Join an existing session
   */
  async joinSession(sessionId: string): Promise<SessionInfo> {
    this.updateStatus('connecting');

    try {
      // Initialize managers
      await this.initializeManagers(sessionId);

      // Wait for sync
      this.updateStatus('syncing');

      // Get session info from sync (in real implementation)
      // For now, create placeholder
      const now = Date.now();
      const session: SessionInfo = {
        sessionId,
        name: `Session ${sessionId.slice(0, 8)}`,
        ownerId: 'unknown',
        ownerName: 'Unknown',
        createdAt: now,
        lastActivity: now,
        mode: 'edit',
        locked: false,
        participantCount: 1,
        maxParticipants: this.config.settings!.maxParticipants!,
      };

      // Add self as participant
      const selfParticipant: SessionParticipant = {
        userId: this.config.user.userId,
        name: this.config.user.name,
        color: getUserColor(this.config.user.userId),
        avatar: this.config.user.avatar,
        role: 'editor',
        joinedAt: now,
        status: 'active',
      };

      this.state.participants.set(this.config.user.userId, selfParticipant);
      this.state.session = session;
      this.state.role = 'editor';

      this.updateStatus('ready');
      this.emit({ type: 'sessionJoined', session });

      return session;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to join session';
      this.state.error = errorMessage;
      this.updateStatus('disconnected');
      this.emit({ type: 'error', error: errorMessage });
      throw error;
    }
  }

  /**
   * Leave the current session
   */
  async leaveSession(): Promise<void> {
    if (!this.state.session) return;

    try {
      // Disconnect managers
      if (this.presenceManager) {
        this.presenceManager.disconnect();
        this.presenceUnsubscribe?.();
        this.presenceManager = null;
      }

      if (this.syncManager) {
        this.syncManager.disconnect();
        this.syncUnsubscribe?.();
        this.syncManager = null;
      }

      // Clear state
      this.state.session = null;
      this.state.participants.clear();
      this.state.role = null;
      this.state.lastSync = null;

      this.updateStatus('disconnected');
      this.emit({ type: 'sessionLeft' });
    } catch (error) {
      console.error('Error leaving session:', error);
    }
  }

  // ============================================================================
  // Session Management
  // ============================================================================

  /**
   * Change session mode
   */
  setMode(mode: 'edit' | 'view' | 'present'): void {
    if (!this.state.session) return;
    if (this.state.role !== 'owner') {
      console.warn('Only session owner can change mode');
      return;
    }

    this.state.session.mode = mode;
    this.emit({ type: 'modeChanged', mode });

    // Broadcast mode change via sync
    if (this.syncManager) {
      this.syncManager.setViewport({ followUserId: mode === 'present' ? this.config.user.userId : undefined });
    }
  }

  /**
   * Lock/unlock session
   */
  setLocked(locked: boolean): void {
    if (!this.state.session) return;
    if (this.state.role !== 'owner') {
      console.warn('Only session owner can lock/unlock');
      return;
    }

    this.state.session.locked = locked;
  }

  /**
   * Change participant role
   */
  setParticipantRole(userId: string, role: 'editor' | 'viewer'): void {
    if (!this.state.session) return;
    if (this.state.role !== 'owner') {
      console.warn('Only session owner can change roles');
      return;
    }

    const participant = this.state.participants.get(userId);
    if (participant && participant.role !== 'owner') {
      participant.role = role;
      this.emit({ type: 'participantUpdated', participant });
    }
  }

  /**
   * Remove participant from session
   */
  removeParticipant(userId: string): void {
    if (!this.state.session) return;
    if (this.state.role !== 'owner') {
      console.warn('Only session owner can remove participants');
      return;
    }

    if (userId === this.config.user.userId) {
      console.warn('Cannot remove self');
      return;
    }

    this.state.participants.delete(userId);
    this.emit({ type: 'participantLeft', userId });
  }

  // ============================================================================
  // State Access
  // ============================================================================

  /**
   * Get current state
   */
  getState(): Readonly<SessionState> {
    return this.state;
  }

  /**
   * Get current session
   */
  getSession(): SessionInfo | null {
    return this.state.session;
  }

  /**
   * Get all participants
   */
  getParticipants(): SessionParticipant[] {
    return Array.from(this.state.participants.values());
  }

  /**
   * Get participant by ID
   */
  getParticipant(userId: string): SessionParticipant | undefined {
    return this.state.participants.get(userId);
  }

  /**
   * Get sync manager (for advanced operations)
   */
  getSyncManager(): CRDTSyncManager | null {
    return this.syncManager;
  }

  /**
   * Get presence manager (for advanced operations)
   */
  getPresenceManager(): PresenceManager | null {
    return this.presenceManager;
  }

  /**
   * Get annotation manager
   */
  getAnnotationManager(): AnnotationManager {
    return this.annotationManager;
  }

  /**
   * Check if current user can edit
   */
  canEdit(): boolean {
    return (
      this.state.status === 'ready' &&
      this.state.session?.mode === 'edit' &&
      (this.state.role === 'owner' || this.state.role === 'editor')
    );
  }

  // ============================================================================
  // Event Subscription
  // ============================================================================

  /**
   * Subscribe to session events
   */
  subscribe(handler: SessionEventHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  private emit(event: SessionEvent): void {
    this.handlers.forEach((handler) => {
      try {
        handler(event);
      } catch (error) {
        console.error('Session event handler error:', error);
      }
    });
  }

  // ============================================================================
  // Internal Methods
  // ============================================================================

  private async initializeManagers(sessionId: string): Promise<void> {
    const userColor = getUserColor(this.config.user.userId);

    // Initialize CRDT sync
    const syncConfig: SyncConfig = {
      serverUrl: this.config.syncServerUrl,
      roomName: sessionId,
    };

    this.syncManager = new CRDTSyncManager(syncConfig);
    this.syncUnsubscribe = this.syncManager.subscribe((event) => {
      this.handleSyncEvent(event);
    });
    await this.syncManager.connect();

    // Initialize presence
    const presenceConfig: PresenceConfig = {
      serverUrl: this.config.presenceServerUrl,
      roomId: sessionId,
      user: {
        userId: this.config.user.userId,
        name: this.config.user.name,
        color: userColor,
        avatar: this.config.user.avatar,
      },
    };

    this.presenceManager = new PresenceManager(presenceConfig);
    this.presenceUnsubscribe = this.presenceManager.subscribe((event) => {
      this.handlePresenceEvent(event);
    });
    this.presenceManager.connect();

    // Subscribe to annotation events
    this.annotationUnsubscribe = this.annotationManager.subscribe((event) => {
      this.emit(event);
    });
  }

  private handleSyncEvent(event: SyncEvent): void {
    switch (event.type) {
      case 'connected':
        if (this.state.status === 'connecting') {
          this.updateStatus('syncing');
        }
        break;

      case 'synced':
        this.state.lastSync = Date.now();
        if (this.state.status === 'syncing') {
          this.updateStatus('ready');
        }
        this.emit({ type: 'synced' });
        break;

      case 'disconnected':
        if (this.state.status !== 'disconnected') {
          this.updateStatus('connecting');
        }
        break;

      case 'error':
        this.state.error = event.error.message;
        this.emit({ type: 'error', error: event.error.message });
        break;

      case 'annotationsChanged':
        // Sync annotations from CRDT
        event.changes.forEach((change) => {
          if (change.action === 'add' || change.action === 'update') {
            if (change.annotation) {
              // Update local annotation manager
              // (handled by CRDT sync)
            }
          }
        });
        break;
    }

    // Forward event
    this.emit(event);
  }

  private handlePresenceEvent(event: PresenceEvent): void {
    switch (event.type) {
      case 'userJoined':
        const newParticipant: SessionParticipant = {
          userId: event.user.userId,
          name: event.user.name,
          color: event.user.color,
          avatar: event.user.avatar,
          role: 'editor', // Default role
          joinedAt: Date.now(),
          status: 'active',
        };
        this.state.participants.set(event.user.userId, newParticipant);
        if (this.state.session) {
          this.state.session.participantCount = this.state.participants.size;
        }
        this.emit({ type: 'participantJoined', participant: newParticipant });
        break;

      case 'userLeft':
        this.state.participants.delete(event.userId);
        if (this.state.session) {
          this.state.session.participantCount = this.state.participants.size;
        }
        this.emit({ type: 'participantLeft', userId: event.userId });
        break;

      case 'statusChanged':
        const participant = this.state.participants.get(event.userId);
        if (participant) {
          participant.status = event.status;
          this.emit({ type: 'participantUpdated', participant });
        }
        break;

      case 'usersSync':
        event.users.forEach((user) => {
          if (!this.state.participants.has(user.userId)) {
            const syncedParticipant: SessionParticipant = {
              userId: user.userId,
              name: user.name,
              color: user.color,
              avatar: user.avatar,
              role: 'editor',
              joinedAt: Date.now(),
              status: user.status,
            };
            this.state.participants.set(user.userId, syncedParticipant);
          }
        });
        break;
    }

    // Forward event
    this.emit(event);
  }

  private updateStatus(status: SessionState['status']): void {
    if (this.state.status !== status) {
      this.state.status = status;
      this.emit({ type: 'statusChanged', status });
    }
  }

  private generateSessionId(): string {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
  }
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Create collaborative session manager
 */
export function createCollaborativeSession(config: SessionConfig): CollaborativeSessionManager {
  return new CollaborativeSessionManager(config);
}

// ============================================================================
// Session Utilities
// ============================================================================

/**
 * Generate shareable session URL
 */
export function getSessionShareUrl(sessionId: string, baseUrl: string): string {
  return `${baseUrl}/session/${sessionId}`;
}

/**
 * Parse session ID from URL
 */
export function parseSessionIdFromUrl(url: string): string | null {
  const match = url.match(/\/session\/([a-zA-Z0-9-]+)/);
  return match ? match[1] : null;
}

/**
 * Format participant count
 */
export function formatParticipantCount(count: number): string {
  if (count === 1) return '1 participant';
  return `${count} participants`;
}

/**
 * Get role display name
 */
export function getRoleDisplayName(role: 'owner' | 'editor' | 'viewer'): string {
  switch (role) {
    case 'owner':
      return 'Owner';
    case 'editor':
      return 'Editor';
    case 'viewer':
      return 'Viewer';
  }
}

/**
 * Get role color
 */
export function getRoleColor(role: 'owner' | 'editor' | 'viewer'): string {
  switch (role) {
    case 'owner':
      return '#F59E0B'; // Amber
    case 'editor':
      return '#3B82F6'; // Blue
    case 'viewer':
      return '#6B7280'; // Gray
  }
}