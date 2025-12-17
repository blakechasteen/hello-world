/**
 * DetailPanel Type Definitions
 * HoloLoom Agent Manager UI - Phase 4
 */

/**
 * Step/Task node representation for step history
 */
export interface TaskNode {
  id: string;
  threadId: string;
  parentId?: string;
  childrenIds: string[];
  depth: number;

  // Step info
  stepType: StepType;
  name: string;
  query?: string;

  // Status & Progress
  status: StepStatus;
  progressPct: number;
  elapsedTimeMs: number;
  tokensUsed: number;
  confidence: number;

  // Dependencies
  dependsOn: string[];
  blocks: string[];

  // Injection
  mrfEligible: boolean;
  mctsEligible: boolean;
  injectionApplied?: 'mrf' | 'mcts' | null;
  injectionStrategy?: string;

  // Results
  response?: string;
  toolSelected?: string;
}

/**
 * Step types in the execution pipeline
 */
export type StepType =
  | 'query'
  | 'retrieval'
  | 'reasoning'
  | 'synthesis'
  | 'verification'
  | 'research'
  | 'planning'
  | 'execution'
  | 'reflection';

/**
 * Step execution status
 */
export type StepStatus =
  | 'idle'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'skipped';

/**
 * Memory node accessed during execution
 */
export interface MemoryNode {
  id: string;
  content: string;
  relevance: number;
  sourceType: MemorySourceType;
  accessedAt: string;
  stepId: string;
  metadata?: Record<string, unknown>;
}

/**
 * Memory source types
 */
export type MemorySourceType =
  | 'graph'      // Knowledge graph (Yarn Graph)
  | 'vector'     // Vector similarity search
  | 'cache'      // Query cache
  | 'hot_pattern' // Hot pattern feedback
  | 'awareness'  // Awareness graph activation
  | 'spring'     // Spring dynamics propagation
  | 'wave';      // Multi-wave engine

/**
 * Source type display configuration
 */
export const MEMORY_SOURCE_CONFIG: Record<MemorySourceType, { icon: string; label: string; color: string }> = {
  graph: { icon: '🔗', label: 'Graph', color: '#3B82F6' },
  vector: { icon: '📊', label: 'Vector', color: '#10B981' },
  cache: { icon: '⚡', label: 'Cache', color: '#F59E0B' },
  hot_pattern: { icon: '🔥', label: 'Hot', color: '#EF4444' },
  awareness: { icon: '👁', label: 'Awareness', color: '#8B5CF6' },
  spring: { icon: '🌊', label: 'Spring', color: '#06B6D4' },
  wave: { icon: '〰', label: 'Wave', color: '#EC4899' },
};

/**
 * File node in the file tree
 */
export interface FileNode {
  path: string;
  name: string;
  isDirectory: boolean;
  children?: FileNode[];
  modifiedBy?: string;
  status: FileStatus;
  lastModified: string;
}

/**
 * File modification status
 */
export type FileStatus =
  | 'modified'  // File was modified
  | 'read'      // File was read only
  | 'created'   // New file created
  | 'deleted';  // File was deleted

/**
 * File status display configuration
 */
export const FILE_STATUS_CONFIG: Record<FileStatus, { icon: string; label: string; color: string }> = {
  modified: { icon: '●', label: 'Modified', color: '#F59E0B' },
  read: { icon: '○', label: 'Read', color: '#64748B' },
  created: { icon: '+', label: 'Created', color: '#10B981' },
  deleted: { icon: '-', label: 'Deleted', color: '#EF4444' },
};

/**
 * Agent color palette (up to 8 distinct agents)
 */
export const AGENT_COLORS = [
  '#3B82F6', // blue
  '#10B981', // emerald
  '#F59E0B', // amber
  '#EF4444', // red
  '#8B5CF6', // violet
  '#EC4899', // pink
  '#06B6D4', // cyan
  '#84CC16', // lime
] as const;

/**
 * Get agent color by index (wraps around)
 */
export function getAgentColor(index: number): string {
  return AGENT_COLORS[index % AGENT_COLORS.length];
}

/**
 * Detail panel tab options
 */
export type DetailTab = 'history' | 'memory' | 'files' | 'injections';

/**
 * Detail panel tab configuration
 */
export const DETAIL_TABS: { id: DetailTab; label: string; icon: string }[] = [
  { id: 'history', label: 'History', icon: '📜' },
  { id: 'memory', label: 'Memory', icon: '🧠' },
  { id: 'files', label: 'Files', icon: '📁' },
  { id: 'injections', label: 'Injections', icon: '💉' },
];

/**
 * Sort options for step history
 */
export type StepSortOption = 'chronological' | 'status' | 'confidence';

/**
 * Sort options for memory nodes
 */
export type MemorySortOption = 'relevance' | 'recency' | 'source';

/**
 * Filter options for step status
 */
export type StepFilterOption = 'all' | 'completed' | 'running' | 'failed';
