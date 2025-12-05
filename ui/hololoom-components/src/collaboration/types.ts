/**
 * Collaboration Types - Shared type definitions
 *
 * Phase 3: Collaborative Intelligence
 */

// ============================================================================
// User Presence Types
// ============================================================================

export interface CursorPosition {
  x: number;
  y: number;
  nodeId?: string;
  timestamp: number;
}

export interface UserFocus {
  type: 'node' | 'edge' | 'viewport' | 'none';
  targetId?: string;
  label?: string;
}

export interface UserPresence {
  userId: string;
  name: string;
  avatar?: string;
  color: string;
  cursor?: CursorPosition;
  focus?: UserFocus;
  lastActive: number;
  status: 'online' | 'idle' | 'offline';
}

// ============================================================================
// Session Types
// ============================================================================

export type SessionMode = 'edit' | 'view' | 'present';

export interface CollaborativeSession {
  sessionId: string;
  name: string;
  description?: string;
  hostId: string;
  participants: string[];
  mode: SessionMode;
  createdAt: number;
  settings: SessionSettings;
}

export interface SessionSettings {
  allowVoice: boolean;
  allowAnnotations: boolean;
  allowEditing: boolean;
  maxParticipants: number;
  isPublic: boolean;
}

export interface SessionInfo {
  sessionId: string;
  name: string;
  description?: string;
  hostName: string;
  participantCount: number;
  mode: SessionMode;
  isPublic: boolean;
}

// ============================================================================
// Annotation Types
// ============================================================================

export type AnnotationType = 'note' | 'question' | 'insight' | 'warning' | 'task';
export type AnnotationStatus = 'open' | 'resolved' | 'archived';
export type ReactionEmoji = '👍' | '💡' | '❓' | '⚠️' | '✅' | '🔥' | '👀' | '💬';

export interface Annotation {
  id: string;
  type: AnnotationType;
  targetType: 'node' | 'edge' | 'viewport' | 'selection';
  targetId: string;
  content: string;
  authorId: string;
  authorName: string;
  createdAt: number;
  updatedAt: number;
  status: AnnotationStatus;
  replies: AnnotationReply[];
  position?: { x: number; y: number };
  metadata?: Record<string, unknown>;
}

export interface AnnotationReply {
  id: string;
  content: string;
  authorId: string;
  authorName: string;
  createdAt: number;
}

export interface Reaction {
  id: string;
  emoji: ReactionEmoji;
  targetType: 'node' | 'edge' | 'annotation';
  targetId: string;
  userId: string;
  userName: string;
  timestamp: number;
}

// ============================================================================
// Sync Types
// ============================================================================

export type SyncStatus = 'disconnected' | 'connecting' | 'synced' | 'error';

export interface SyncState {
  status: SyncStatus;
  lastSyncAt: number;
  pendingChanges: number;
  error?: string;
}

export interface SharedNode {
  id: string;
  label: string;
  type: string;
  x: number;
  y: number;
  metadata?: Record<string, unknown>;
  lastModifiedBy?: string;
  lastModifiedAt?: number;
}

export interface SharedEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  weight?: number;
  metadata?: Record<string, unknown>;
}

export interface UserAwareness {
  cursor?: CursorPosition;
  focus?: UserFocus;
  viewport?: ViewportState;
  selection?: string[];
}

export interface ViewportState {
  x: number;
  y: number;
  width: number;
  height: number;
  scale: number;
}

// ============================================================================
// Voice Types
// ============================================================================

export interface VoiceParticipant {
  peerId: string;
  name: string;
  isMuted: boolean;
  isSpeaking: boolean;
  volume: number;
  audioLevel: number;
}

export interface VoiceSettings {
  inputDevice?: string;
  outputDevice?: string;
  noiseSuppression: boolean;
  echoCancellation: boolean;
  autoGainControl: boolean;
  voiceActivityThreshold: number;
}

// ============================================================================
// Event Types
// ============================================================================

export type CollaborationEventType =
  | 'user_joined'
  | 'user_left'
  | 'cursor_moved'
  | 'focus_changed'
  | 'node_added'
  | 'node_updated'
  | 'node_deleted'
  | 'edge_added'
  | 'edge_deleted'
  | 'annotation_added'
  | 'annotation_updated'
  | 'reaction_added'
  | 'reaction_removed'
  | 'voice_state_changed'
  | 'session_mode_changed'
  | 'sync_status_changed';

export interface CollaborationEvent {
  type: CollaborationEventType;
  userId: string;
  timestamp: number;
  payload: Record<string, unknown>;
}

// ============================================================================
// Filter Types
// ============================================================================

export interface AnnotationFilter {
  types?: AnnotationType[];
  status?: AnnotationStatus[];
  authorIds?: string[];
  targetTypes?: ('node' | 'edge' | 'viewport' | 'selection')[];
  targetIds?: string[];
  since?: number;
  searchText?: string;
}

// ============================================================================
// Statistics Types
// ============================================================================

export interface AnnotationStats {
  total: number;
  byType: Record<AnnotationType, number>;
  byStatus: Record<AnnotationStatus, number>;
  byAuthor: Record<string, number>;
  recentActivity: number;
}

export interface SessionStats {
  participantCount: number;
  annotationCount: number;
  reactionCount: number;
  editCount: number;
  duration: number;
}