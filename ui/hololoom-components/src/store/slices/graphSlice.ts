/**
 * Graph Slice for HoloLoom Store
 *
 * Manages state for the Knowledge Graph Animation component.
 * Tracks nodes, edges, activation levels, and layout positions.
 *
 * @module store/slices/graphSlice
 */

import type { StateCreator } from 'zustand';

// ============================================================================
// Types
// ============================================================================

/**
 * Node types in the knowledge graph
 */
export type GraphNodeType = 'query' | 'concept' | 'entity' | 'fact' | 'memory';

/**
 * Edge relationship types
 */
export type GraphEdgeType =
  | 'IS_A'
  | 'USES'
  | 'MENTIONS'
  | 'LEADS_TO'
  | 'PART_OF'
  | 'IN_TIME'
  | 'OCCURRED_AT'
  | 'RELATES_TO'
  | 'INCLUDES';

/**
 * A node in the knowledge graph
 */
export interface GraphNode {
  id: string;
  label: string;
  type: GraphNodeType;
  importance: number; // 0.0-1.0
  /** Layout position (pixels) */
  x: number;
  y: number;
  /** Velocity for force-directed layout */
  vx: number;
  vy: number;
  /** Current activation level (0.0-1.0) */
  activation: number;
  /** Target activation for animation */
  targetActivation: number;
  /** Whether this node is part of the retrieval path */
  isOnPath: boolean;
  /** Node metadata */
  metadata?: Record<string, unknown>;
}

/**
 * An edge in the knowledge graph
 */
export interface GraphEdge {
  id: string; // source:target:type
  source: string;
  target: string;
  type: GraphEdgeType;
  weight: number; // 0.0-1.0
  /** Whether this edge is part of the retrieval path */
  isOnPath: boolean;
}

/**
 * Activation event from backend
 */
export interface ActivationEvent {
  nodeId: string;
  activationLevel: number;
  sourceNodeId: string | null;
  hopDistance: number;
  relevanceToQuery: number;
  timestamp: number;
}

/**
 * Retrieval path through the graph
 */
export interface RetrievalPath {
  nodes: string[];
  edges: Array<{ source: string; target: string; type: string }>;
  totalHops: number;
  finalRelevance: number;
}

/**
 * Graph viewport state
 */
export interface GraphViewport {
  /** Zoom level (1.0 = 100%) */
  zoom: number;
  /** Pan offset X */
  panX: number;
  /** Pan offset Y */
  panY: number;
  /** Viewport width */
  width: number;
  /** Viewport height */
  height: number;
}

/**
 * Layout algorithm settings
 */
export interface LayoutSettings {
  /** Force strength for node repulsion */
  repulsionStrength: number;
  /** Spring strength for edge attraction */
  attractionStrength: number;
  /** Damping factor (0.0-1.0) */
  damping: number;
  /** Center gravity strength */
  centerGravity: number;
  /** Whether layout is currently simulating */
  isSimulating: boolean;
  /** Maximum iterations for layout */
  maxIterations: number;
}

/**
 * Graph state
 */
export interface GraphState {
  /** All nodes in the graph */
  nodes: Map<string, GraphNode>;
  /** All edges in the graph */
  edges: Map<string, GraphEdge>;
  /** Currently selected node (if any) */
  selectedNodeId: string | null;
  /** Currently hovered node (if any) */
  hoveredNodeId: string | null;
  /** Query node ID (if any) */
  queryNodeId: string | null;
  /** Activation event history */
  activationHistory: ActivationEvent[];
  /** Current retrieval path */
  retrievalPath: RetrievalPath | null;
  /** Viewport state */
  viewport: GraphViewport;
  /** Layout settings */
  layout: LayoutSettings;
  /** Whether graph is loading initial data */
  isLoading: boolean;
  /** Error message (if any) */
  error: string | null;
}

/**
 * Graph actions
 */
export interface GraphActions {
  /** Set the full graph from a snapshot */
  setGraphSnapshot: (
    nodes: Array<{
      id: string;
      label: string;
      type: string;
      importance: number;
      x?: number;
      y?: number;
      metadata?: Record<string, unknown>;
    }>,
    edges: Array<{
      source: string;
      target: string;
      type: string;
      weight: number;
    }>,
    queryNodeId?: string
  ) => void;

  /** Process an activation event (animate node activation) */
  processActivation: (event: {
    nodeId: string;
    activationLevel: number;
    sourceNodeId?: string | null;
    hopDistance: number;
    relevanceToQuery: number;
  }) => void;

  /** Set the retrieval path */
  setRetrievalPath: (path: RetrievalPath) => void;

  /** Clear the retrieval path */
  clearRetrievalPath: () => void;

  /** Update node position (from force layout) */
  updateNodePosition: (nodeId: string, x: number, y: number) => void;

  /** Update multiple node positions (batch) */
  updateNodePositions: (positions: Array<{ id: string; x: number; y: number }>) => void;

  /** Set node selection */
  selectNode: (nodeId: string | null) => void;

  /** Set node hover */
  hoverNode: (nodeId: string | null) => void;

  /** Update viewport */
  setViewport: (viewport: Partial<GraphViewport>) => void;

  /** Zoom in/out */
  zoom: (delta: number, centerX?: number, centerY?: number) => void;

  /** Pan the viewport */
  pan: (deltaX: number, deltaY: number) => void;

  /** Fit graph to viewport */
  fitToView: () => void;

  /** Update layout settings */
  setLayoutSettings: (settings: Partial<LayoutSettings>) => void;

  /** Start/stop layout simulation */
  setSimulating: (simulating: boolean) => void;

  /** Decay all activations by a factor */
  decayActivations: (factor: number) => void;

  /** Reset graph to initial state */
  resetGraph: () => void;

  /** Set loading state */
  setLoading: (loading: boolean) => void;

  /** Set error */
  setError: (error: string | null) => void;
}

/**
 * Combined graph slice
 */
export type GraphSlice = GraphState & GraphActions;

// ============================================================================
// Initial State
// ============================================================================

const initialViewport: GraphViewport = {
  zoom: 1.0,
  panX: 0,
  panY: 0,
  width: 800,
  height: 600,
};

const initialLayout: LayoutSettings = {
  repulsionStrength: 1000,
  attractionStrength: 0.1,
  damping: 0.9,
  centerGravity: 0.05,
  isSimulating: false,
  maxIterations: 300,
};

const initialGraphState: GraphState = {
  nodes: new Map(),
  edges: new Map(),
  selectedNodeId: null,
  hoveredNodeId: null,
  queryNodeId: null,
  activationHistory: [],
  retrievalPath: null,
  viewport: initialViewport,
  layout: initialLayout,
  isLoading: false,
  error: null,
};

// ============================================================================
// Helpers
// ============================================================================

const MAX_ACTIVATION_HISTORY = 100;
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 5.0;

function createEdgeId(source: string, target: string, type: string): string {
  return `${source}:${target}:${type}`;
}

// ============================================================================
// Slice Creator
// ============================================================================

/**
 * Create the graph slice for Zustand store.
 *
 * Usage with existing store:
 * ```typescript
 * import { create } from 'zustand';
 * import { immer } from 'zustand/middleware/immer';
 * import { createGraphSlice } from './slices/graphSlice';
 *
 * const useStore = create<HoloLoomStore & GraphSlice>()(
 *   immer((...a) => ({
 *     ...existingSlices(...a),
 *     ...createGraphSlice(...a),
 *   }))
 * );
 * ```
 */
export const createGraphSlice: StateCreator<
  GraphSlice,
  [['zustand/immer', never]],
  [],
  GraphSlice
> = (set, get) => ({
  ...initialGraphState,

  setGraphSnapshot: (nodes, edges, queryNodeId) =>
    set((state) => {
      // Clear existing graph
      state.nodes = new Map();
      state.edges = new Map();
      state.activationHistory = [];
      state.retrievalPath = null;
      state.queryNodeId = queryNodeId ?? null;

      // Add nodes
      for (const node of nodes) {
        const graphNode: GraphNode = {
          id: node.id,
          label: node.label,
          type: (node.type as GraphNodeType) || 'concept',
          importance: node.importance ?? 0.5,
          x: node.x ?? Math.random() * state.viewport.width,
          y: node.y ?? Math.random() * state.viewport.height,
          vx: 0,
          vy: 0,
          activation: node.id === queryNodeId ? 1.0 : 0.0,
          targetActivation: node.id === queryNodeId ? 1.0 : 0.0,
          isOnPath: false,
          metadata: node.metadata,
        };
        state.nodes.set(node.id, graphNode);
      }

      // Add edges
      for (const edge of edges) {
        const edgeId = createEdgeId(edge.source, edge.target, edge.type);
        const graphEdge: GraphEdge = {
          id: edgeId,
          source: edge.source,
          target: edge.target,
          type: (edge.type as GraphEdgeType) || 'RELATES_TO',
          weight: edge.weight ?? 0.5,
          isOnPath: false,
        };
        state.edges.set(edgeId, graphEdge);
      }

      state.isLoading = false;
    }),

  processActivation: (event) =>
    set((state) => {
      const node = state.nodes.get(event.nodeId);
      if (node) {
        node.targetActivation = event.activationLevel;
        // Smooth animation - current activation moves toward target
        node.activation = node.activation + (event.activationLevel - node.activation) * 0.3;
      }

      // Record activation event
      const activationEvent: ActivationEvent = {
        nodeId: event.nodeId,
        activationLevel: event.activationLevel,
        sourceNodeId: event.sourceNodeId ?? null,
        hopDistance: event.hopDistance,
        relevanceToQuery: event.relevanceToQuery,
        timestamp: Date.now(),
      };

      state.activationHistory.push(activationEvent);

      // Keep history bounded
      if (state.activationHistory.length > MAX_ACTIVATION_HISTORY) {
        state.activationHistory.shift();
      }
    }),

  setRetrievalPath: (path) =>
    set((state) => {
      state.retrievalPath = path;

      // Mark nodes on path
      for (const [, node] of state.nodes) {
        node.isOnPath = path.nodes.includes(node.id);
      }

      // Mark edges on path
      for (const [, edge] of state.edges) {
        edge.isOnPath = path.edges.some(
          (e) => e.source === edge.source && e.target === edge.target
        );
      }
    }),

  clearRetrievalPath: () =>
    set((state) => {
      state.retrievalPath = null;

      // Clear path markers
      for (const [, node] of state.nodes) {
        node.isOnPath = false;
      }
      for (const [, edge] of state.edges) {
        edge.isOnPath = false;
      }
    }),

  updateNodePosition: (nodeId, x, y) =>
    set((state) => {
      const node = state.nodes.get(nodeId);
      if (node) {
        node.x = x;
        node.y = y;
      }
    }),

  updateNodePositions: (positions) =>
    set((state) => {
      for (const pos of positions) {
        const node = state.nodes.get(pos.id);
        if (node) {
          node.x = pos.x;
          node.y = pos.y;
        }
      }
    }),

  selectNode: (nodeId) =>
    set((state) => {
      state.selectedNodeId = nodeId;
    }),

  hoverNode: (nodeId) =>
    set((state) => {
      state.hoveredNodeId = nodeId;
    }),

  setViewport: (viewport) =>
    set((state) => {
      Object.assign(state.viewport, viewport);
    }),

  zoom: (delta, centerX, centerY) =>
    set((state) => {
      const oldZoom = state.viewport.zoom;
      const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, oldZoom * (1 + delta)));

      // Zoom toward center point
      const cx = centerX ?? state.viewport.width / 2;
      const cy = centerY ?? state.viewport.height / 2;

      const scale = newZoom / oldZoom;
      state.viewport.panX = cx - (cx - state.viewport.panX) * scale;
      state.viewport.panY = cy - (cy - state.viewport.panY) * scale;
      state.viewport.zoom = newZoom;
    }),

  pan: (deltaX, deltaY) =>
    set((state) => {
      state.viewport.panX += deltaX;
      state.viewport.panY += deltaY;
    }),

  fitToView: () =>
    set((state) => {
      if (state.nodes.size === 0) return;

      // Calculate bounding box
      let minX = Infinity,
        minY = Infinity,
        maxX = -Infinity,
        maxY = -Infinity;

      for (const [, node] of state.nodes) {
        minX = Math.min(minX, node.x);
        minY = Math.min(minY, node.y);
        maxX = Math.max(maxX, node.x);
        maxY = Math.max(maxY, node.y);
      }

      const padding = 50;
      const graphWidth = maxX - minX + padding * 2;
      const graphHeight = maxY - minY + padding * 2;

      const scaleX = state.viewport.width / graphWidth;
      const scaleY = state.viewport.height / graphHeight;
      const scale = Math.min(scaleX, scaleY, MAX_ZOOM);

      state.viewport.zoom = Math.max(MIN_ZOOM, scale);
      state.viewport.panX = (state.viewport.width - graphWidth * state.viewport.zoom) / 2 - minX * state.viewport.zoom + padding * state.viewport.zoom;
      state.viewport.panY = (state.viewport.height - graphHeight * state.viewport.zoom) / 2 - minY * state.viewport.zoom + padding * state.viewport.zoom;
    }),

  setLayoutSettings: (settings) =>
    set((state) => {
      Object.assign(state.layout, settings);
    }),

  setSimulating: (simulating) =>
    set((state) => {
      state.layout.isSimulating = simulating;
    }),

  decayActivations: (factor) =>
    set((state) => {
      for (const [, node] of state.nodes) {
        // Don't decay query node
        if (node.id !== state.queryNodeId) {
          node.activation *= factor;
          node.targetActivation *= factor;
        }
      }
    }),

  resetGraph: () =>
    set(() => ({
      ...initialGraphState,
      nodes: new Map(),
      edges: new Map(),
    })),

  setLoading: (loading) =>
    set((state) => {
      state.isLoading = loading;
    }),

  setError: (error) =>
    set((state) => {
      state.error = error;
    }),
});

// ============================================================================
// Selectors
// ============================================================================

export const selectNodes = (state: GraphSlice) => state.nodes;
export const selectEdges = (state: GraphSlice) => state.edges;
export const selectSelectedNode = (state: GraphSlice) =>
  state.selectedNodeId ? state.nodes.get(state.selectedNodeId) : null;
export const selectHoveredNode = (state: GraphSlice) =>
  state.hoveredNodeId ? state.nodes.get(state.hoveredNodeId) : null;
export const selectQueryNode = (state: GraphSlice) =>
  state.queryNodeId ? state.nodes.get(state.queryNodeId) : null;
export const selectViewport = (state: GraphSlice) => state.viewport;
export const selectLayout = (state: GraphSlice) => state.layout;
export const selectRetrievalPath = (state: GraphSlice) => state.retrievalPath;
export const selectIsLoading = (state: GraphSlice) => state.isLoading;
export const selectError = (state: GraphSlice) => state.error;

/**
 * Get nodes as array (for rendering)
 */
export const selectNodesArray = (state: GraphSlice): GraphNode[] =>
  Array.from(state.nodes.values());

/**
 * Get edges as array (for rendering)
 */
export const selectEdgesArray = (state: GraphSlice): GraphEdge[] =>
  Array.from(state.edges.values());

/**
 * Get node count
 */
export const selectNodeCount = (state: GraphSlice): number => state.nodes.size;

/**
 * Get edge count
 */
export const selectEdgeCount = (state: GraphSlice): number => state.edges.size;

/**
 * Get active nodes (activation > threshold)
 */
export const selectActiveNodes = (state: GraphSlice, threshold = 0.1): GraphNode[] =>
  Array.from(state.nodes.values()).filter((node) => node.activation > threshold);

/**
 * Get path nodes
 */
export const selectPathNodes = (state: GraphSlice): GraphNode[] =>
  Array.from(state.nodes.values()).filter((node) => node.isOnPath);

/**
 * Get neighbors of a node
 */
export const selectNodeNeighbors = (state: GraphSlice, nodeId: string): GraphNode[] => {
  const neighbors = new Set<string>();

  for (const [, edge] of state.edges) {
    if (edge.source === nodeId) neighbors.add(edge.target);
    if (edge.target === nodeId) neighbors.add(edge.source);
  }

  return Array.from(neighbors)
    .map((id) => state.nodes.get(id))
    .filter((node): node is GraphNode => node !== undefined);
};

export default createGraphSlice;
