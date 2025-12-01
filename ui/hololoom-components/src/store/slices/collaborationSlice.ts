/**
 * Collaboration Slice - Phase 3: Collaborative Intelligence
 *
 * Manages real-time collaboration state:
 * - User presence and cursors
 * - Collaborative sessions
 * - Shared annotations and reactions
 * - CRDT synchronization state
 *
 * @module store/slices/collaborationSlice
 */

import type { StateCreator } from 'zustand';

// ============================================================================
// Types
// ============================================================================

/**
 * User presence information
 */
export interface UserPresence {
  /** Unique user identifier */
  userId: string;
  /** Display name */
  name: string;
  /** Avatar URL or initials */
  avatar?: string;
  /** User's assigned color (hex) */
  color: string;
  /** Current cursor position in graph space */
  cursor?: CursorPosition;
  /** What the user is currently viewing/focusing on */
  focus?: UserFocus;
  /** Last activity timestamp */
  lastActive: number;
  /** Connection status */
  status: 'online' | 'idle' | 'offline';
}

/**
 * Cursor position in the knowledge graph
 */
export interface CursorPosition {
  /** X coordinate in graph space */
  x: number;
  /** Y coordinate in graph space */
  y: number;
  /** Node ID if hovering over a node */
  nodeId?: string;
  /** Timestamp of position update */
  timestamp: number;
}

/**
 * What a user is currently focused on
 */
export interface UserFocus {
  /** Type of focus */
  type: 'node' | 'edge' | 'region' | 'annotation' | 'none';
  /** Target ID (node ID, edge ID, annotation ID) */
  targetId?: string;
  /** Selection bounds for region focus */
  bounds?: { x: number; y: number; width: number; height: number };
}

/**
 * Collaborative annotation on a node or edge
 */
export interface Annotation {
  /** Unique annotation ID */
  id: string;
  /** Target type */
  targetType: 'node' | 'edge' | 'region';
  /** Target ID */
  targetId: string;
  /** Annotation content */
  content: string;
  /** Author user ID */
  authorId: string;
  /** Author display name */
  authorName: string;
  /** Creation timestamp */
  createdAt: number;
  /** Last update timestamp */
  updatedAt: number;
  /** Position offset for region annotations */
  position?: { x: number; y: number };
  /** Annotation type */
  type: 'note' | 'question' | 'insight' | 'warning';
  /** Resolved status */
  resolved: boolean;
  /** Thread replies */
  replies: AnnotationReply[];
}

/**
 * Reply to an annotation
 */
export interface AnnotationReply {
  id: string;
  content: string;
  authorId: string;
  authorName: string;
  createdAt: number;
}

/**
 * Reaction to a node, edge, or annotation
 */
export interface Reaction {
  /** Unique reaction ID */
  id: string;
  /** Target type */
  targetType: 'node' | 'edge' | 'annotation';
  /** Target ID */
  targetId: string;
  /** Emoji reaction */
  emoji: '👍' | '💡' | '❓' | '⚠️' | '✅' | '🔥' | '👀' | '💬';
  /** User ID who reacted */
  userId: string;
  /** Timestamp */
  timestamp: number;
}

/**
 * Collaborative session information
 */
export interface CollaborativeSession {
  /** Session ID */
  sessionId: string;
  /** Session name */
  name: string;
  /** Session creator */
  creatorId: string;
  /** Active users in session */
  participants: string[];
  /** Session creation time */
  createdAt: number;
  /** Last activity time */
  lastActivity: number;
  /** Session mode */
  mode: 'view' | 'edit' | 'present';
  /** Whether session is locked to new participants */
  locked: boolean;
}

/**
 * CRDT sync status
 */
export interface SyncStatus {
  /** Whether connected to sync server */
  connected: boolean;
  /** Number of pending changes */
  pendingChanges: number;
  /** Last sync timestamp */
  lastSync: number;
  /** Sync error if any */
  error?: string;
  /** Latency to sync server (ms) */
  latency?: number;
}

/**
 * Collaboration state
 */
export interface CollaborationState {
  /** Current user ID */
  currentUserId: string | null;
  /** Current user info */
  currentUser: UserPresence | null;
  /** All users in current session */
  users: Record<string, UserPresence>;
  /** User cursors (optimized for frequent updates) */
  cursors: Record<string, CursorPosition>;
  /** Current session */
  session: CollaborativeSession | null;
  /** Available sessions */
  availableSessions: CollaborativeSession[];
  /** Annotations */
  annotations: Record<string, Annotation>;
  /** Reactions grouped by target */
  reactions: Record<string, Reaction[]>;
  /** CRDT sync status */
  syncStatus: SyncStatus;
  /** UI settings */
  settings: {
    showCursors: boolean;
    showAnnotations: boolean;
    showReactions: boolean;
    cursorSmoothness: number; // 0-1, interpolation factor
  };
  /** Loading states */
  isConnecting: boolean;
  isJoiningSession: boolean;
  /** Error state */
  error: string | null;
}

/**
 * Collaboration actions
 */
export interface CollaborationActions {
  // User management
  setCurrentUser: (user: UserPresence) => void;
  updateUserPresence: (userId: string, updates: Partial<UserPresence>) => void;
  setUserCursor: (userId: string, cursor: CursorPosition) => void;
  setUserFocus: (userId: string, focus: UserFocus) => void;
  addUser: (user: UserPresence) => void;
  removeUser: (userId: string) => void;
  setUserStatus: (userId: string, status: 'online' | 'idle' | 'offline') => void;

  // Session management
  createSession: (name: string, mode?: 'view' | 'edit' | 'present') => void;
  joinSession: (sessionId: string) => void;
  leaveSession: () => void;
  setSessionMode: (mode: 'view' | 'edit' | 'present') => void;
  lockSession: (locked: boolean) => void;
  setAvailableSessions: (sessions: CollaborativeSession[]) => void;

  // Annotations
  addAnnotation: (annotation: Omit<Annotation, 'id' | 'createdAt' | 'updatedAt' | 'replies'>) => string;
  updateAnnotation: (id: string, updates: Partial<Pick<Annotation, 'content' | 'resolved'>>) => void;
  deleteAnnotation: (id: string) => void;
  addReply: (annotationId: string, reply: Omit<AnnotationReply, 'id' | 'createdAt'>) => void;

  // Reactions
  addReaction: (reaction: Omit<Reaction, 'id' | 'timestamp'>) => void;
  removeReaction: (reactionId: string, targetId: string) => void;
  toggleReaction: (targetType: 'node' | 'edge' | 'annotation', targetId: string, emoji: Reaction['emoji']) => void;

  // Sync status
  setSyncStatus: (status: Partial<SyncStatus>) => void;
  setSyncConnected: (connected: boolean) => void;
  setSyncError: (error: string | null) => void;

  // Settings
  setShowCursors: (show: boolean) => void;
  setShowAnnotations: (show: boolean) => void;
  setShowReactions: (show: boolean) => void;
  setCursorSmoothness: (smoothness: number) => void;

  // Connection state
  setConnecting: (connecting: boolean) => void;
  setJoiningSession: (joining: boolean) => void;
  setError: (error: string | null) => void;

  // Bulk operations (for CRDT sync)
  syncUsers: (users: UserPresence[]) => void;
  syncAnnotations: (annotations: Annotation[]) => void;
  syncReactions: (reactions: Reaction[]) => void;
}

export type CollaborationSlice = CollaborationState & CollaborationActions;

// ============================================================================
// Default State
// ============================================================================

const defaultSyncStatus: SyncStatus = {
  connected: false,
  pendingChanges: 0,
  lastSync: 0,
  error: undefined,
  latency: undefined,
};

const defaultSettings = {
  showCursors: true,
  showAnnotations: true,
  showReactions: true,
  cursorSmoothness: 0.3,
};

// ============================================================================
// Slice Creator
// ============================================================================

/**
 * Generate unique ID
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

/**
 * Create collaboration slice
 */
export const createCollaborationSlice: StateCreator<
  CollaborationSlice,
  [],
  [],
  CollaborationSlice
> = (set, get) => ({
  // Initial state
  currentUserId: null,
  currentUser: null,
  users: {},
  cursors: {},
  session: null,
  availableSessions: [],
  annotations: {},
  reactions: {},
  syncStatus: defaultSyncStatus,
  settings: defaultSettings,
  isConnecting: false,
  isJoiningSession: false,
  error: null,

  // User management
  setCurrentUser: (user) =>
    set({
      currentUserId: user.userId,
      currentUser: user,
      users: { ...get().users, [user.userId]: user },
    }),

  updateUserPresence: (userId, updates) =>
    set((state) => {
      const user = state.users[userId];
      if (!user) return state;

      const updatedUser = { ...user, ...updates, lastActive: Date.now() };
      const newUsers = { ...state.users, [userId]: updatedUser };

      return {
        users: newUsers,
        currentUser: userId === state.currentUserId ? updatedUser : state.currentUser,
      };
    }),

  setUserCursor: (userId, cursor) =>
    set((state) => ({
      cursors: { ...state.cursors, [userId]: cursor },
    })),

  setUserFocus: (userId, focus) =>
    set((state) => {
      const user = state.users[userId];
      if (!user) return state;

      return {
        users: {
          ...state.users,
          [userId]: { ...user, focus, lastActive: Date.now() },
        },
      };
    }),

  addUser: (user) =>
    set((state) => ({
      users: { ...state.users, [user.userId]: user },
    })),

  removeUser: (userId) =>
    set((state) => {
      const { [userId]: removed, ...remainingUsers } = state.users;
      const { [userId]: removedCursor, ...remainingCursors } = state.cursors;
      return { users: remainingUsers, cursors: remainingCursors };
    }),

  setUserStatus: (userId, status) =>
    set((state) => {
      const user = state.users[userId];
      if (!user) return state;

      return {
        users: {
          ...state.users,
          [userId]: { ...user, status, lastActive: Date.now() },
        },
      };
    }),

  // Session management
  createSession: (name, mode = 'edit') => {
    const currentUser = get().currentUser;
    if (!currentUser) return;

    const session: CollaborativeSession = {
      sessionId: generateId(),
      name,
      creatorId: currentUser.userId,
      participants: [currentUser.userId],
      createdAt: Date.now(),
      lastActivity: Date.now(),
      mode,
      locked: false,
    };

    set({ session, isJoiningSession: false });
  },

  joinSession: (sessionId) => {
    const session = get().availableSessions.find((s) => s.sessionId === sessionId);
    const currentUser = get().currentUser;

    if (!session || !currentUser) return;

    const updatedSession = {
      ...session,
      participants: [...session.participants, currentUser.userId],
      lastActivity: Date.now(),
    };

    set({ session: updatedSession, isJoiningSession: false });
  },

  leaveSession: () => {
    const currentUserId = get().currentUserId;
    const session = get().session;

    if (session && currentUserId) {
      const updatedParticipants = session.participants.filter((id) => id !== currentUserId);

      if (updatedParticipants.length > 0) {
        // Session continues without this user
        set({
          session: null,
          users: { [currentUserId]: get().users[currentUserId] },
          cursors: {},
          annotations: {},
          reactions: {},
        });
      } else {
        // Session ends (last participant)
        set({
          session: null,
          users: { [currentUserId]: get().users[currentUserId] },
          cursors: {},
          annotations: {},
          reactions: {},
        });
      }
    }
  },

  setSessionMode: (mode) =>
    set((state) => ({
      session: state.session ? { ...state.session, mode, lastActivity: Date.now() } : null,
    })),

  lockSession: (locked) =>
    set((state) => ({
      session: state.session ? { ...state.session, locked, lastActivity: Date.now() } : null,
    })),

  setAvailableSessions: (sessions) => set({ availableSessions: sessions }),

  // Annotations
  addAnnotation: (annotation) => {
    const id = generateId();
    const now = Date.now();

    const newAnnotation: Annotation = {
      ...annotation,
      id,
      createdAt: now,
      updatedAt: now,
      replies: [],
    };

    set((state) => ({
      annotations: { ...state.annotations, [id]: newAnnotation },
      syncStatus: {
        ...state.syncStatus,
        pendingChanges: state.syncStatus.pendingChanges + 1,
      },
    }));

    return id;
  },

  updateAnnotation: (id, updates) =>
    set((state) => {
      const annotation = state.annotations[id];
      if (!annotation) return state;

      return {
        annotations: {
          ...state.annotations,
          [id]: { ...annotation, ...updates, updatedAt: Date.now() },
        },
        syncStatus: {
          ...state.syncStatus,
          pendingChanges: state.syncStatus.pendingChanges + 1,
        },
      };
    }),

  deleteAnnotation: (id) =>
    set((state) => {
      const { [id]: deleted, ...remaining } = state.annotations;
      return {
        annotations: remaining,
        syncStatus: {
          ...state.syncStatus,
          pendingChanges: state.syncStatus.pendingChanges + 1,
        },
      };
    }),

  addReply: (annotationId, reply) =>
    set((state) => {
      const annotation = state.annotations[annotationId];
      if (!annotation) return state;

      const newReply: AnnotationReply = {
        ...reply,
        id: generateId(),
        createdAt: Date.now(),
      };

      return {
        annotations: {
          ...state.annotations,
          [annotationId]: {
            ...annotation,
            replies: [...annotation.replies, newReply],
            updatedAt: Date.now(),
          },
        },
        syncStatus: {
          ...state.syncStatus,
          pendingChanges: state.syncStatus.pendingChanges + 1,
        },
      };
    }),

  // Reactions
  addReaction: (reaction) => {
    const newReaction: Reaction = {
      ...reaction,
      id: generateId(),
      timestamp: Date.now(),
    };

    set((state) => {
      const targetReactions = state.reactions[reaction.targetId] || [];
      return {
        reactions: {
          ...state.reactions,
          [reaction.targetId]: [...targetReactions, newReaction],
        },
      };
    });
  },

  removeReaction: (reactionId, targetId) =>
    set((state) => {
      const targetReactions = state.reactions[targetId] || [];
      return {
        reactions: {
          ...state.reactions,
          [targetId]: targetReactions.filter((r) => r.id !== reactionId),
        },
      };
    }),

  toggleReaction: (targetType, targetId, emoji) => {
    const state = get();
    const currentUserId = state.currentUserId;
    if (!currentUserId) return;

    const targetReactions = state.reactions[targetId] || [];
    const existingReaction = targetReactions.find(
      (r) => r.userId === currentUserId && r.emoji === emoji
    );

    if (existingReaction) {
      // Remove existing reaction
      set({
        reactions: {
          ...state.reactions,
          [targetId]: targetReactions.filter((r) => r.id !== existingReaction.id),
        },
      });
    } else {
      // Add new reaction
      const newReaction: Reaction = {
        id: generateId(),
        targetType,
        targetId,
        emoji,
        userId: currentUserId,
        timestamp: Date.now(),
      };

      set({
        reactions: {
          ...state.reactions,
          [targetId]: [...targetReactions, newReaction],
        },
      });
    }
  },

  // Sync status
  setSyncStatus: (status) =>
    set((state) => ({
      syncStatus: { ...state.syncStatus, ...status },
    })),

  setSyncConnected: (connected) =>
    set((state) => ({
      syncStatus: {
        ...state.syncStatus,
        connected,
        lastSync: connected ? Date.now() : state.syncStatus.lastSync,
      },
    })),

  setSyncError: (error) =>
    set((state) => ({
      syncStatus: { ...state.syncStatus, error: error ?? undefined },
    })),

  // Settings
  setShowCursors: (show) =>
    set((state) => ({
      settings: { ...state.settings, showCursors: show },
    })),

  setShowAnnotations: (show) =>
    set((state) => ({
      settings: { ...state.settings, showAnnotations: show },
    })),

  setShowReactions: (show) =>
    set((state) => ({
      settings: { ...state.settings, showReactions: show },
    })),

  setCursorSmoothness: (smoothness) =>
    set((state) => ({
      settings: { ...state.settings, cursorSmoothness: Math.max(0, Math.min(1, smoothness)) },
    })),

  // Connection state
  setConnecting: (connecting) => set({ isConnecting: connecting }),
  setJoiningSession: (joining) => set({ isJoiningSession: joining }),
  setError: (error) => set({ error }),

  // Bulk sync operations
  syncUsers: (users) => {
    const currentUserId = get().currentUserId;
    const usersMap: Record<string, UserPresence> = {};

    users.forEach((user) => {
      usersMap[user.userId] = user;
    });

    // Preserve current user if exists
    if (currentUserId && get().currentUser) {
      usersMap[currentUserId] = get().currentUser!;
    }

    set({ users: usersMap });
  },

  syncAnnotations: (annotations) => {
    const annotationsMap: Record<string, Annotation> = {};
    annotations.forEach((a) => {
      annotationsMap[a.id] = a;
    });
    set({ annotations: annotationsMap });
  },

  syncReactions: (reactions) => {
    const reactionsMap: Record<string, Reaction[]> = {};
    reactions.forEach((r) => {
      if (!reactionsMap[r.targetId]) {
        reactionsMap[r.targetId] = [];
      }
      reactionsMap[r.targetId].push(r);
    });
    set({ reactions: reactionsMap });
  },
});

// ============================================================================
// Selectors
// ============================================================================

/**
 * Select current user
 */
export const selectCurrentUser = (state: CollaborationSlice) => state.currentUser;

/**
 * Select all online users
 */
export const selectOnlineUsers = (state: CollaborationSlice) =>
  Object.values(state.users).filter((u) => u.status === 'online');

/**
 * Select other users (not current user)
 */
export const selectOtherUsers = (state: CollaborationSlice) =>
  Object.values(state.users).filter((u) => u.userId !== state.currentUserId);

/**
 * Select user by ID
 */
export const selectUserById = (state: CollaborationSlice, userId: string) =>
  state.users[userId];

/**
 * Select all cursors (excluding current user)
 */
export const selectOtherCursors = (state: CollaborationSlice) => {
  const otherCursors: Array<CursorPosition & { user: UserPresence }> = [];

  Object.entries(state.cursors).forEach(([userId, cursor]) => {
    if (userId !== state.currentUserId && state.users[userId]) {
      otherCursors.push({ ...cursor, user: state.users[userId] });
    }
  });

  return otherCursors;
};

/**
 * Select current session
 */
export const selectSession = (state: CollaborationSlice) => state.session;

/**
 * Select whether in collaborative mode
 */
export const selectIsCollaborating = (state: CollaborationSlice) =>
  state.session !== null && Object.keys(state.users).length > 1;

/**
 * Select annotations for a target
 */
export const selectAnnotationsForTarget = (
  state: CollaborationSlice,
  targetId: string
) => Object.values(state.annotations).filter((a) => a.targetId === targetId);

/**
 * Select reactions for a target
 */
export const selectReactionsForTarget = (
  state: CollaborationSlice,
  targetId: string
) => state.reactions[targetId] || [];

/**
 * Select reaction counts for a target
 */
export const selectReactionCounts = (
  state: CollaborationSlice,
  targetId: string
): Record<string, number> => {
  const reactions = state.reactions[targetId] || [];
  const counts: Record<string, number> = {};

  reactions.forEach((r) => {
    counts[r.emoji] = (counts[r.emoji] || 0) + 1;
  });

  return counts;
};

/**
 * Select if current user has reacted with emoji
 */
export const selectHasUserReacted = (
  state: CollaborationSlice,
  targetId: string,
  emoji: Reaction['emoji']
) => {
  const reactions = state.reactions[targetId] || [];
  return reactions.some(
    (r) => r.userId === state.currentUserId && r.emoji === emoji
  );
};

/**
 * Select sync status
 */
export const selectSyncStatus = (state: CollaborationSlice) => state.syncStatus;

/**
 * Select collaboration settings
 */
export const selectCollaborationSettings = (state: CollaborationSlice) =>
  state.settings;

/**
 * Select connection state
 */
export const selectIsConnected = (state: CollaborationSlice) =>
  state.syncStatus.connected && !state.isConnecting;

/**
 * Select users focused on a specific node
 */
export const selectUsersFocusedOnNode = (
  state: CollaborationSlice,
  nodeId: string
) =>
  Object.values(state.users).filter(
    (u) =>
      u.userId !== state.currentUserId &&
      u.focus?.type === 'node' &&
      u.focus?.targetId === nodeId
  );

/**
 * Select all unresolved annotations
 */
export const selectUnresolvedAnnotations = (state: CollaborationSlice) =>
  Object.values(state.annotations).filter((a) => !a.resolved);

/**
 * Select annotation count by type
 */
export const selectAnnotationCounts = (state: CollaborationSlice) => {
  const counts = { note: 0, question: 0, insight: 0, warning: 0 };

  Object.values(state.annotations).forEach((a) => {
    counts[a.type]++;
  });

  return counts;
};
