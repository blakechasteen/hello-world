/**
 * ConnectedKnowledgeGraph
 *
 * Connected wrapper that wires KnowledgeGraph2D to the graphSlice.
 * Subscribes to graph state and dispatches actions for user interactions.
 *
 * Part of HoloLoom Phase 1 Frontend (November 2025)
 */

import React, { useCallback, useMemo } from 'react';
import { KnowledgeGraph2D } from '../graph/KnowledgeGraph2D';
import { useHoloLoomStore } from '../../store';
import type { GraphNode, GraphEdge, GraphViewport } from '../../store/slices/graphSlice';

// ============================================================================
// Props
// ============================================================================

export interface ConnectedKnowledgeGraphProps {
  /** Whether to animate node transitions */
  animated?: boolean;
  /** Dark mode */
  darkMode?: boolean;
  /** Show node labels */
  showLabels?: boolean;
  /** Show edge type labels */
  showEdgeTypes?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Width override */
  width?: number;
  /** Height override */
  height?: number;
  /** Callback when query changes (for external control) */
  onQueryChange?: (query: string) => void;
}

// ============================================================================
// Store Selectors (inline for performance)
// ============================================================================

const selectNodesArray = (state: ReturnType<typeof useHoloLoomStore.getState>): GraphNode[] => {
  return Array.from((state as any).graphNodes?.values() ?? []);
};

const selectEdgesArray = (state: ReturnType<typeof useHoloLoomStore.getState>): GraphEdge[] => {
  return Array.from((state as any).graphEdges?.values() ?? []);
};

// ============================================================================
// Component
// ============================================================================

export const ConnectedKnowledgeGraph: React.FC<ConnectedKnowledgeGraphProps> = ({
  animated = true,
  darkMode = false,
  showLabels = true,
  showEdgeTypes = false,
  className,
  width,
  height,
}) => {
  // Select state from store
  // Note: We need to handle the case where graphSlice might be combined differently
  const contextChunks = useHoloLoomStore(state => state.contextChunks);
  const confidence = useHoloLoomStore(state => state.confidence);
  const query = useHoloLoomStore(state => state.query);

  // Derive nodes from context chunks (simplified mapping)
  // In a full implementation, this would come from graphSlice
  const nodes = useMemo((): GraphNode[] => {
    const nodeList: GraphNode[] = [];

    // Add query node if we have a query
    if (query) {
      nodeList.push({
        id: 'query',
        label: query.slice(0, 30) + (query.length > 30 ? '...' : ''),
        type: 'query',
        x: 400,
        y: 50,
        importance: 1.0,
        activation: 1.0,
        isOnPath: true,
      });
    }

    // Add nodes from context chunks
    contextChunks.forEach((chunk, index) => {
      // Position nodes in a radial layout around query
      const angle = (index / Math.max(contextChunks.length, 1)) * Math.PI * 2 - Math.PI / 2;
      const radius = 150 + chunk.hopDistance * 80;
      const x = 400 + Math.cos(angle) * radius;
      const y = 250 + Math.sin(angle) * radius;

      nodeList.push({
        id: chunk.id,
        label: chunk.content.slice(0, 25) + '...',
        type: chunk.hopDistance === 0 ? 'concept' : chunk.hopDistance === 1 ? 'entity' : 'memory',
        x,
        y,
        importance: chunk.relevance,
        activation: chunk.relevance * 0.8,
        isOnPath: chunk.hopDistance <= 1,
      });
    });

    return nodeList;
  }, [contextChunks, query]);

  // Derive edges connecting context to query
  const edges = useMemo((): GraphEdge[] => {
    if (!query) return [];

    return contextChunks.slice(0, 5).map((chunk, index) => ({
      id: `edge-query-${chunk.id}`,
      source: 'query',
      target: chunk.id,
      type: 'RELATES_TO' as const,
      weight: chunk.relevance,
      isOnPath: chunk.hopDistance <= 1,
    }));
  }, [contextChunks, query]);

  // Viewport state
  const [viewport, setViewport] = React.useState<GraphViewport>({
    zoom: 1,
    panX: 0,
    panY: 0,
    width: width || 800,
    height: height || 400,
  });

  // Selection state
  const [selectedNodeId, setSelectedNodeId] = React.useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = React.useState<string | null>(null);

  // Update viewport dimensions
  React.useEffect(() => {
    if (width || height) {
      setViewport(prev => ({
        ...prev,
        width: width || prev.width,
        height: height || prev.height,
      }));
    }
  }, [width, height]);

  // Handle node click
  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNodeId(prev => (prev === node.id ? null : node.id));
  }, []);

  // Handle node hover
  const handleNodeHover = useCallback((node: GraphNode | null) => {
    setHoveredNodeId(node?.id ?? null);
  }, []);

  // Handle viewport change
  const handleViewportChange = useCallback((changes: Partial<GraphViewport>) => {
    setViewport(prev => ({ ...prev, ...changes }));
  }, []);

  return (
    <KnowledgeGraph2D
      nodes={nodes}
      edges={edges}
      viewport={viewport}
      selectedNodeId={selectedNodeId}
      hoveredNodeId={hoveredNodeId}
      queryNodeId={query ? 'query' : null}
      animated={animated}
      darkMode={darkMode}
      showLabels={showLabels}
      showEdgeTypes={showEdgeTypes}
      onNodeClick={handleNodeClick}
      onNodeHover={handleNodeHover}
      onViewportChange={handleViewportChange}
      className={className}
    />
  );
};

ConnectedKnowledgeGraph.displayName = 'ConnectedKnowledgeGraph';

export default ConnectedKnowledgeGraph;
