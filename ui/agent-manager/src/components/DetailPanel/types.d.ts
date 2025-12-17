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
    stepType: StepType;
    name: string;
    query?: string;
    status: StepStatus;
    progressPct: number;
    elapsedTimeMs: number;
    tokensUsed: number;
    confidence: number;
    dependsOn: string[];
    blocks: string[];
    mrfEligible: boolean;
    mctsEligible: boolean;
    injectionApplied?: 'mrf' | 'mcts' | null;
    injectionStrategy?: string;
    response?: string;
    toolSelected?: string;
}
/**
 * Step types in the execution pipeline
 */
export type StepType = 'query' | 'retrieval' | 'reasoning' | 'synthesis' | 'verification' | 'research' | 'planning' | 'execution' | 'reflection';
/**
 * Step execution status
 */
export type StepStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled' | 'skipped';
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
export type MemorySourceType = 'graph' | 'vector' | 'cache' | 'hot_pattern' | 'awareness' | 'spring' | 'wave';
/**
 * Source type display configuration
 */
export declare const MEMORY_SOURCE_CONFIG: Record<MemorySourceType, {
    icon: string;
    label: string;
    color: string;
}>;
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
export type FileStatus = 'modified' | 'read' | 'created' | 'deleted';
/**
 * File status display configuration
 */
export declare const FILE_STATUS_CONFIG: Record<FileStatus, {
    icon: string;
    label: string;
    color: string;
}>;
/**
 * Agent color palette (up to 8 distinct agents)
 */
export declare const AGENT_COLORS: readonly ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"];
/**
 * Get agent color by index (wraps around)
 */
export declare function getAgentColor(index: number): string;
/**
 * Detail panel tab options
 */
export type DetailTab = 'history' | 'memory' | 'files';
/**
 * Detail panel tab configuration
 */
export declare const DETAIL_TABS: {
    id: DetailTab;
    label: string;
    icon: string;
}[];
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
//# sourceMappingURL=types.d.ts.map