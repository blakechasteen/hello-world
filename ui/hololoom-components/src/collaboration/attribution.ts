/**
 * Collaborative Annotations - Phase 3: Collaborative Intelligence
 *
 * System for adding notes, reactions, and comments to knowledge graph elements.
 * Supports real-time collaboration with conflict-free updates.
 *
 * Features:
 * - Notes on nodes and edges
 * - Emoji reactions (👍💡❓⚠️✅🔥👀💬)
 * - Threaded discussions
 * - Annotation types (note, question, insight, warning)
 *
 * @module collaboration/attribution
 */

// ============================================================================
// Types
// ============================================================================

/**
 * Annotation type categories
 */
export type AnnotationType = 'note' | 'question' | 'insight' | 'warning';

/**
 * Supported emoji reactions
 */
export type ReactionEmoji = '👍' | '💡' | '❓' | '⚠️' | '✅' | '🔥' | '👀' | '💬';

/**
 * Target types that can have annotations
 */
export type AnnotationTarget = 'node' | 'edge' | 'region';

/**
 * Annotation data structure
 */
export interface Annotation {
  id: string;
  targetType: AnnotationTarget;
  targetId: string;
  content: string;
  authorId: string;
  authorName: string;
  authorColor: string;
  type: AnnotationType;
  resolved: boolean;
  createdAt: number;
  updatedAt: number;
  position?: { x: number; y: number }; // For region annotations
  replies: AnnotationReply[];
  reactions: AnnotationReaction[];
}

/**
 * Reply to an annotation (threaded discussion)
 */
export interface AnnotationReply {
  id: string;
  content: string;
  authorId: string;
  authorName: string;
  authorColor: string;
  createdAt: number;
  reactions: AnnotationReaction[];
}

/**
 * Reaction on an annotation or reply
 */
export interface AnnotationReaction {
  id: string;
  emoji: ReactionEmoji;
  userId: string;
  userName: string;
  timestamp: number;
}

/**
 * Node/edge reaction (quick reactions without full annotation)
 */
export interface QuickReaction {
  id: string;
  targetType: 'node' | 'edge';
  targetId: string;
  emoji: ReactionEmoji;
  userId: string;
  userName: string;
  timestamp: number;
}

/**
 * Annotation filter options
 */
export interface AnnotationFilter {
  types?: AnnotationType[];
  authors?: string[];
  resolved?: boolean;
  targetTypes?: AnnotationTarget[];
  targetIds?: string[];
  dateRange?: { start: number; end: number };
  hasReplies?: boolean;
  searchText?: string;
}

/**
 * Annotation statistics
 */
export interface AnnotationStats {
  total: number;
  byType: Record<AnnotationType, number>;
  byAuthor: Record<string, number>;
  resolved: number;
  unresolved: number;
  totalReplies: number;
  totalReactions: number;
  reactionCounts: Record<ReactionEmoji, number>;
}

// ============================================================================
// Annotation Manager
// ============================================================================

/**
 * Event types emitted by AnnotationManager
 */
export type AnnotationEvent =
  | { type: 'annotationAdded'; annotation: Annotation }
  | { type: 'annotationUpdated'; annotation: Annotation }
  | { type: 'annotationDeleted'; annotationId: string }
  | { type: 'replyAdded'; annotationId: string; reply: AnnotationReply }
  | { type: 'reactionToggled'; annotationId: string; reaction: AnnotationReaction; added: boolean }
  | { type: 'quickReactionToggled'; reaction: QuickReaction; added: boolean };

type AnnotationEventHandler = (event: AnnotationEvent) => void;

/**
 * Manages collaborative annotations
 */
export class AnnotationManager {
  private annotations: Map<string, Annotation> = new Map();
  private quickReactions: Map<string, QuickReaction[]> = new Map();
  private handlers: Set<AnnotationEventHandler> = new Set();
  private currentUserId: string;
  private currentUserName: string;
  private currentUserColor: string;

  constructor(user: { userId: string; name: string; color: string }) {
    this.currentUserId = user.userId;
    this.currentUserName = user.name;
    this.currentUserColor = user.color;
  }

  // ============================================================================
  // Annotation CRUD
  // ============================================================================

  /**
   * Create a new annotation
   */
  createAnnotation(params: {
    targetType: AnnotationTarget;
    targetId: string;
    content: string;
    type: AnnotationType;
    position?: { x: number; y: number };
  }): Annotation {
    const now = Date.now();
    const annotation: Annotation = {
      id: this.generateId(),
      targetType: params.targetType,
      targetId: params.targetId,
      content: params.content,
      authorId: this.currentUserId,
      authorName: this.currentUserName,
      authorColor: this.currentUserColor,
      type: params.type,
      resolved: false,
      createdAt: now,
      updatedAt: now,
      position: params.position,
      replies: [],
      reactions: [],
    };

    this.annotations.set(annotation.id, annotation);
    this.emit({ type: 'annotationAdded', annotation });

    return annotation;
  }

  /**
   * Update an annotation
   */
  updateAnnotation(
    annotationId: string,
    updates: Partial<Pick<Annotation, 'content' | 'type' | 'resolved'>>
  ): Annotation | null {
    const annotation = this.annotations.get(annotationId);
    if (!annotation) return null;

    // Only author can update
    if (annotation.authorId !== this.currentUserId) {
      console.warn('Cannot update annotation: not the author');
      return null;
    }

    const updated: Annotation = {
      ...annotation,
      ...updates,
      updatedAt: Date.now(),
    };

    this.annotations.set(annotationId, updated);
    this.emit({ type: 'annotationUpdated', annotation: updated });

    return updated;
  }

  /**
   * Delete an annotation
   */
  deleteAnnotation(annotationId: string): boolean {
    const annotation = this.annotations.get(annotationId);
    if (!annotation) return false;

    // Only author can delete
    if (annotation.authorId !== this.currentUserId) {
      console.warn('Cannot delete annotation: not the author');
      return false;
    }

    this.annotations.delete(annotationId);
    this.emit({ type: 'annotationDeleted', annotationId });

    return true;
  }

  /**
   * Resolve/unresolve an annotation
   */
  toggleResolved(annotationId: string): boolean {
    const annotation = this.annotations.get(annotationId);
    if (!annotation) return false;

    return this.updateAnnotation(annotationId, { resolved: !annotation.resolved }) !== null;
  }

  // ============================================================================
  // Replies
  // ============================================================================

  /**
   * Add a reply to an annotation
   */
  addReply(annotationId: string, content: string): AnnotationReply | null {
    const annotation = this.annotations.get(annotationId);
    if (!annotation) return null;

    const reply: AnnotationReply = {
      id: this.generateId(),
      content,
      authorId: this.currentUserId,
      authorName: this.currentUserName,
      authorColor: this.currentUserColor,
      createdAt: Date.now(),
      reactions: [],
    };

    annotation.replies.push(reply);
    annotation.updatedAt = Date.now();

    this.emit({ type: 'replyAdded', annotationId, reply });

    return reply;
  }

  /**
   * Delete a reply
   */
  deleteReply(annotationId: string, replyId: string): boolean {
    const annotation = this.annotations.get(annotationId);
    if (!annotation) return false;

    const replyIndex = annotation.replies.findIndex((r) => r.id === replyId);
    if (replyIndex === -1) return false;

    const reply = annotation.replies[replyIndex];
    if (reply.authorId !== this.currentUserId) {
      console.warn('Cannot delete reply: not the author');
      return false;
    }

    annotation.replies.splice(replyIndex, 1);
    annotation.updatedAt = Date.now();

    return true;
  }

  // ============================================================================
  // Reactions
  // ============================================================================

  /**
   * Toggle reaction on an annotation
   */
  toggleAnnotationReaction(annotationId: string, emoji: ReactionEmoji): boolean {
    const annotation = this.annotations.get(annotationId);
    if (!annotation) return false;

    const existingIndex = annotation.reactions.findIndex(
      (r) => r.userId === this.currentUserId && r.emoji === emoji
    );

    let added: boolean;
    let reaction: AnnotationReaction;

    if (existingIndex >= 0) {
      // Remove existing reaction
      reaction = annotation.reactions[existingIndex];
      annotation.reactions.splice(existingIndex, 1);
      added = false;
    } else {
      // Add new reaction
      reaction = {
        id: this.generateId(),
        emoji,
        userId: this.currentUserId,
        userName: this.currentUserName,
        timestamp: Date.now(),
      };
      annotation.reactions.push(reaction);
      added = true;
    }

    this.emit({ type: 'reactionToggled', annotationId, reaction, added });

    return added;
  }

  /**
   * Toggle reaction on a reply
   */
  toggleReplyReaction(
    annotationId: string,
    replyId: string,
    emoji: ReactionEmoji
  ): boolean {
    const annotation = this.annotations.get(annotationId);
    if (!annotation) return false;

    const reply = annotation.replies.find((r) => r.id === replyId);
    if (!reply) return false;

    const existingIndex = reply.reactions.findIndex(
      (r) => r.userId === this.currentUserId && r.emoji === emoji
    );

    if (existingIndex >= 0) {
      reply.reactions.splice(existingIndex, 1);
      return false;
    } else {
      reply.reactions.push({
        id: this.generateId(),
        emoji,
        userId: this.currentUserId,
        userName: this.currentUserName,
        timestamp: Date.now(),
      });
      return true;
    }
  }

  /**
   * Toggle quick reaction on a node or edge
   */
  toggleQuickReaction(
    targetType: 'node' | 'edge',
    targetId: string,
    emoji: ReactionEmoji
  ): boolean {
    const key = `${targetType}:${targetId}`;
    const reactions = this.quickReactions.get(key) || [];

    const existingIndex = reactions.findIndex(
      (r) => r.userId === this.currentUserId && r.emoji === emoji
    );

    let added: boolean;
    let reaction: QuickReaction;

    if (existingIndex >= 0) {
      // Remove existing reaction
      reaction = reactions[existingIndex];
      reactions.splice(existingIndex, 1);
      added = false;
    } else {
      // Add new reaction
      reaction = {
        id: this.generateId(),
        targetType,
        targetId,
        emoji,
        userId: this.currentUserId,
        userName: this.currentUserName,
        timestamp: Date.now(),
      };
      reactions.push(reaction);
      added = true;
    }

    this.quickReactions.set(key, reactions);
    this.emit({ type: 'quickReactionToggled', reaction, added });

    return added;
  }

  // ============================================================================
  // Queries
  // ============================================================================

  /**
   * Get annotation by ID
   */
  getAnnotation(annotationId: string): Annotation | undefined {
    return this.annotations.get(annotationId);
  }

  /**
   * Get all annotations
   */
  getAllAnnotations(): Annotation[] {
    return Array.from(this.annotations.values());
  }

  /**
   * Get annotations for a specific target
   */
  getAnnotationsForTarget(targetType: AnnotationTarget, targetId: string): Annotation[] {
    return this.getAllAnnotations().filter(
      (a) => a.targetType === targetType && a.targetId === targetId
    );
  }

  /**
   * Get quick reactions for a target
   */
  getQuickReactions(targetType: 'node' | 'edge', targetId: string): QuickReaction[] {
    const key = `${targetType}:${targetId}`;
    return this.quickReactions.get(key) || [];
  }

  /**
   * Get reaction counts for a target
   */
  getReactionCounts(
    targetType: 'node' | 'edge',
    targetId: string
  ): Record<ReactionEmoji, number> {
    const reactions = this.getQuickReactions(targetType, targetId);
    const counts: Record<string, number> = {};

    reactions.forEach((r) => {
      counts[r.emoji] = (counts[r.emoji] || 0) + 1;
    });

    return counts as Record<ReactionEmoji, number>;
  }

  /**
   * Check if current user has reacted with emoji
   */
  hasUserReacted(
    targetType: 'node' | 'edge',
    targetId: string,
    emoji: ReactionEmoji
  ): boolean {
    const reactions = this.getQuickReactions(targetType, targetId);
    return reactions.some((r) => r.userId === this.currentUserId && r.emoji === emoji);
  }

  /**
   * Filter annotations
   */
  filterAnnotations(filter: AnnotationFilter): Annotation[] {
    let results = this.getAllAnnotations();

    if (filter.types?.length) {
      results = results.filter((a) => filter.types!.includes(a.type));
    }

    if (filter.authors?.length) {
      results = results.filter((a) => filter.authors!.includes(a.authorId));
    }

    if (filter.resolved !== undefined) {
      results = results.filter((a) => a.resolved === filter.resolved);
    }

    if (filter.targetTypes?.length) {
      results = results.filter((a) => filter.targetTypes!.includes(a.targetType));
    }

    if (filter.targetIds?.length) {
      results = results.filter((a) => filter.targetIds!.includes(a.targetId));
    }

    if (filter.dateRange) {
      results = results.filter(
        (a) =>
          a.createdAt >= filter.dateRange!.start && a.createdAt <= filter.dateRange!.end
      );
    }

    if (filter.hasReplies !== undefined) {
      results = results.filter((a) =>
        filter.hasReplies ? a.replies.length > 0 : a.replies.length === 0
      );
    }

    if (filter.searchText) {
      const search = filter.searchText.toLowerCase();
      results = results.filter(
        (a) =>
          a.content.toLowerCase().includes(search) ||
          a.replies.some((r) => r.content.toLowerCase().includes(search))
      );
    }

    return results;
  }

  /**
   * Get annotation statistics
   */
  getStats(): AnnotationStats {
    const annotations = this.getAllAnnotations();
    const stats: AnnotationStats = {
      total: annotations.length,
      byType: { note: 0, question: 0, insight: 0, warning: 0 },
      byAuthor: {},
      resolved: 0,
      unresolved: 0,
      totalReplies: 0,
      totalReactions: 0,
      reactionCounts: {
        '👍': 0,
        '💡': 0,
        '❓': 0,
        '⚠️': 0,
        '✅': 0,
        '🔥': 0,
        '👀': 0,
        '💬': 0,
      },
    };

    annotations.forEach((a) => {
      stats.byType[a.type]++;
      stats.byAuthor[a.authorId] = (stats.byAuthor[a.authorId] || 0) + 1;

      if (a.resolved) {
        stats.resolved++;
      } else {
        stats.unresolved++;
      }

      stats.totalReplies += a.replies.length;
      stats.totalReactions += a.reactions.length;

      a.reactions.forEach((r) => {
        stats.reactionCounts[r.emoji]++;
      });

      a.replies.forEach((reply) => {
        stats.totalReactions += reply.reactions.length;
        reply.reactions.forEach((r) => {
          stats.reactionCounts[r.emoji]++;
        });
      });
    });

    // Add quick reactions
    this.quickReactions.forEach((reactions) => {
      reactions.forEach((r) => {
        stats.totalReactions++;
        stats.reactionCounts[r.emoji]++;
      });
    });

    return stats;
  }

  // ============================================================================
  // Bulk Operations
  // ============================================================================

  /**
   * Load annotations from external source (e.g., CRDT sync)
   */
  loadAnnotations(annotations: Annotation[]): void {
    annotations.forEach((a) => {
      this.annotations.set(a.id, a);
    });
  }

  /**
   * Load quick reactions from external source
   */
  loadQuickReactions(reactions: QuickReaction[]): void {
    reactions.forEach((r) => {
      const key = `${r.targetType}:${r.targetId}`;
      const existing = this.quickReactions.get(key) || [];
      existing.push(r);
      this.quickReactions.set(key, existing);
    });
  }

  /**
   * Clear all annotations
   */
  clear(): void {
    this.annotations.clear();
    this.quickReactions.clear();
  }

  /**
   * Export all annotations
   */
  export(): { annotations: Annotation[]; quickReactions: QuickReaction[] } {
    return {
      annotations: this.getAllAnnotations(),
      quickReactions: Array.from(this.quickReactions.values()).flat(),
    };
  }

  // ============================================================================
  // Event Subscription
  // ============================================================================

  /**
   * Subscribe to annotation events
   */
  subscribe(handler: AnnotationEventHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  private emit(event: AnnotationEvent): void {
    this.handlers.forEach((handler) => {
      try {
        handler(event);
      } catch (error) {
        console.error('Annotation event handler error:', error);
      }
    });
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
  }

  /**
   * Update current user info
   */
  setCurrentUser(user: { userId: string; name: string; color: string }): void {
    this.currentUserId = user.userId;
    this.currentUserName = user.name;
    this.currentUserColor = user.color;
  }
}

// ============================================================================
// Annotation Type Utilities
// ============================================================================

/**
 * Get icon for annotation type
 */
export function getAnnotationTypeIcon(type: AnnotationType): string {
  switch (type) {
    case 'note':
      return '📝';
    case 'question':
      return '❓';
    case 'insight':
      return '💡';
    case 'warning':
      return '⚠️';
  }
}

/**
 * Get color for annotation type
 */
export function getAnnotationTypeColor(type: AnnotationType): string {
  switch (type) {
    case 'note':
      return '#6B7280'; // Gray
    case 'question':
      return '#3B82F6'; // Blue
    case 'insight':
      return '#F59E0B'; // Amber
    case 'warning':
      return '#EF4444'; // Red
  }
}

/**
 * Get label for annotation type
 */
export function getAnnotationTypeLabel(type: AnnotationType): string {
  switch (type) {
    case 'note':
      return 'Note';
    case 'question':
      return 'Question';
    case 'insight':
      return 'Insight';
    case 'warning':
      return 'Warning';
  }
}

/**
 * Available reaction emojis with labels
 */
export const REACTION_EMOJIS: Array<{ emoji: ReactionEmoji; label: string }> = [
  { emoji: '👍', label: 'Agree' },
  { emoji: '💡', label: 'Great idea' },
  { emoji: '❓', label: 'Question' },
  { emoji: '⚠️', label: 'Concern' },
  { emoji: '✅', label: 'Done' },
  { emoji: '🔥', label: 'Important' },
  { emoji: '👀', label: 'Looking into' },
  { emoji: '💬', label: 'Discuss' },
];

// ============================================================================
// Factory
// ============================================================================

/**
 * Create annotation manager instance
 */
export function createAnnotationManager(user: {
  userId: string;
  name: string;
  color: string;
}): AnnotationManager {
  return new AnnotationManager(user);
}