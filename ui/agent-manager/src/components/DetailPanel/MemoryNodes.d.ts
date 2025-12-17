import React from 'react';
/**
 * MemoryNode represents a memory element accessed during thread execution
 * Source types indicate where the memory came from:
 * - graph: Knowledge graph (symbolic relationships)
 * - vector: Vector database (semantic similarity)
 * - cache: Query cache (fast retrieval)
 * - hot_pattern: Hot pattern feedback (frequently accessed)
 */
interface MemoryNode {
    id: string;
    content: string;
    relevance: number;
    sourceType: 'graph' | 'vector' | 'cache' | 'hot_pattern';
    accessedAt: string;
    stepId: string;
    metadata?: Record<string, unknown>;
}
interface MemoryNodesProps {
    /** Array of memory nodes to display */
    nodes: MemoryNode[];
    /** Group nodes by step (if true) or show all together */
    groupByStep?: boolean;
    /** Callback when a node ID is clicked (for copying) */
    onNodeClick?: (nodeId: string) => void;
    /** Optional CSS class for styling */
    className?: string;
}
/**
 * MemoryNodes Component
 *
 * Displays memory nodes accessed during thread execution with:
 * - Relevance-based heat map coloring
 * - Sortable by relevance, recency, or access count
 * - Expandable nodes with full content
 * - Search/filter by content
 * - Click-to-copy node IDs
 * - Visual source type badges
 */
export declare const MemoryNodes: React.FC<MemoryNodesProps>;
/**
 * Default export
 */
export default MemoryNodes;
//# sourceMappingURL=MemoryNodes.d.ts.map