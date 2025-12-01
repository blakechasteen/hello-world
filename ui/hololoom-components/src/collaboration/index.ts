/**
 * Collaboration Module - Phase 3: Collaborative Intelligence
 *
 * Real-time collaboration infrastructure for HoloLoom's Shared Memory Palace.
 *
 * Components:
 * - CRDT Sync (Y.js) - Conflict-free concurrent editing
 * - Cursor Presence - Real-time cursor tracking
 * - Annotations - Notes and reactions on graph elements
 * - Sessions - Multi-user session management
 * - Voice - WebRTC voice communication
 *
 * @module collaboration
 */

// CRDT Sync
export {
  CRDTSyncManager,
  type SyncConfig,
  type SyncEvent,
  type SharedGraphState,
  type SharedNode,
  type SharedEdge,
  type SharedAnnotation,
  type SharedViewport,
  type UserAwareness,
  type NodeChange,
  type EdgeChange,
  type AnnotationChange,
  type YDoc,
  type YMap,
  type YArray,
  type YText,
  type WebSocketProvider,
  type AwarenessProtocol,
} from './sync';

// Cursor Presence
export {
  PresenceManager,
  createPresenceManager,
  getCursorPath,
  getInitials,
  getUserColor,
  type PresenceConfig,
  type PresenceState,
  type PresenceEvent,
  type CursorState,
  type FocusState,
  type InterpolatedCursor,
} from './presence';

// Annotations
export {
  AnnotationManager,
  createAnnotationManager,
  getAnnotationTypeIcon,
  getAnnotationTypeColor,
  getAnnotationTypeLabel,
  REACTION_EMOJIS,
  type Annotation,
  type AnnotationReply,
  type AnnotationReaction,
  type QuickReaction,
  type AnnotationType,
  type ReactionEmoji,
  type AnnotationTarget,
  type AnnotationFilter,
  type AnnotationStats,
  type AnnotationEvent,
} from './attribution';

// Sessions
export {
  CollaborativeSessionManager,
  createCollaborativeSession,
  getSessionShareUrl,
  parseSessionIdFromUrl,
  formatParticipantCount,
  getRoleDisplayName,
  getRoleColor,
  type SessionConfig,
  type SessionInfo,
  type SessionParticipant,
  type SessionState,
  type SessionEvent,
} from './session';

// Voice
export {
  VoiceChannelManager,
  createVoiceChannel,
  type VoiceConfig,
  type VoiceState,
  type VoiceEvent,
  type VoiceParticipant,
} from './voice';
