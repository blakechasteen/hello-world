/**
 * KnowledgeGraph2D Renderer
 *
 * High-performance Canvas 2D renderer for the knowledge graph visualization.
 * Handles node rendering, edge drawing, activation glow, and path highlighting.
 *
 * @module components/graph/KnowledgeGraph2DRenderer
 */

import type { GraphNode, GraphEdge, GraphNodeType, GraphEdgeType } from '../../store/slices/graphSlice';

// ============================================================================
// Types
// ============================================================================

export interface GraphRenderConfig {
  /** Base node radius */
  nodeRadius: number;
  /** Edge line width */
  edgeWidth: number;
  /** Path highlight width */
  pathWidth: number;
  /** Font for labels */
  font: string;
  /** Show node labels */
  showLabels: boolean;
  /** Show edge types */
  showEdgeTypes: boolean;
  /** Animation speed for activation glow */
  glowAnimationSpeed: number;
  /** Dark mode */
  darkMode: boolean;
}

export interface GraphRenderState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  viewport: {
    zoom: number;
    panX: number;
    panY: number;
    width: number;
    height: number;
  };
  selectedNodeId: string | null;
  hoveredNodeId: string | null;
  queryNodeId: string | null;
  animationTime: number;
}

// ============================================================================
// Color Schemes
// ============================================================================

const NODE_TYPE_COLORS: Record<GraphNodeType, { light: string; dark: string }> = {
  query: { light: '#6366f1', dark: '#818cf8' }, // Indigo
  concept: { light: '#3b82f6', dark: '#60a5fa' }, // Blue
  entity: { light: '#10b981', dark: '#34d399' }, // Emerald
  fact: { light: '#f59e0b', dark: '#fbbf24' }, // Amber
  memory: { light: '#8b5cf6', dark: '#a78bfa' }, // Purple
};

const EDGE_TYPE_COLORS: Record<GraphEdgeType | 'default', { light: string; dark: string }> = {
  IS_A: { light: '#3b82f6', dark: '#60a5fa' }, // Blue
  USES: { light: '#10b981', dark: '#34d399' }, // Green
  MENTIONS: { light: '#6b7280', dark: '#9ca3af' }, // Gray
  LEADS_TO: { light: '#f59e0b', dark: '#fbbf24' }, // Amber
  PART_OF: { light: '#8b5cf6', dark: '#a78bfa' }, // Purple
  IN_TIME: { light: '#06b6d4', dark: '#22d3ee' }, // Cyan
  OCCURRED_AT: { light: '#14b8a6', dark: '#2dd4bf' }, // Teal
  RELATES_TO: { light: '#6b7280', dark: '#9ca3af' }, // Gray
  INCLUDES: { light: '#ec4899', dark: '#f472b6' }, // Pink
  default: { light: '#6b7280', dark: '#9ca3af' }, // Gray
};

const COLORS = {
  light: {
    background: '#ffffff',
    text: '#1f2937',
    textSecondary: '#6b7280',
    selection: '#fbbf24',
    hover: '#fef3c7',
    path: '#ef4444',
    glow: 'rgba(99, 102, 241, 0.3)',
    edge: '#d1d5db',
  },
  dark: {
    background: '#1f2937',
    text: '#f3f4f6',
    textSecondary: '#9ca3af',
    selection: '#fbbf24',
    hover: '#374151',
    path: '#f87171',
    glow: 'rgba(129, 140, 248, 0.4)',
    edge: '#4b5563',
  },
};

// ============================================================================
// Renderer
// ============================================================================

export class KnowledgeGraph2DRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private config: GraphRenderConfig;
  private dpr: number;

  constructor(canvas: HTMLCanvasElement, config: Partial<GraphRenderConfig> = {}) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get 2D context');
    this.ctx = ctx;

    this.dpr = window.devicePixelRatio || 1;
    this.config = {
      nodeRadius: 8,
      edgeWidth: 1.5,
      pathWidth: 3,
      font: '11px Inter, system-ui, sans-serif',
      showLabels: true,
      showEdgeTypes: false,
      glowAnimationSpeed: 2,
      darkMode: false,
      ...config,
    };

    this.setupCanvas();
  }

  private setupCanvas(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
    this.ctx.scale(this.dpr, this.dpr);
  }

  updateConfig(config: Partial<GraphRenderConfig>): void {
    Object.assign(this.config, config);
  }

  resize(): void {
    this.setupCanvas();
  }

  /**
   * Main render method
   */
  render(state: GraphRenderState): void {
    const { ctx, config } = this;
    const { viewport, animationTime } = state;
    const colors = config.darkMode ? COLORS.dark : COLORS.light;

    // Clear canvas
    ctx.fillStyle = colors.background;
    ctx.fillRect(0, 0, this.canvas.width / this.dpr, this.canvas.height / this.dpr);

    // Apply viewport transform
    ctx.save();
    ctx.translate(viewport.panX, viewport.panY);
    ctx.scale(viewport.zoom, viewport.zoom);

    // Draw edges first (below nodes)
    this.renderEdges(state, colors);

    // Draw nodes
    this.renderNodes(state, colors, animationTime);

    ctx.restore();
  }

  /**
   * Render all edges
   */
  private renderEdges(state: GraphRenderState, colors: typeof COLORS.light): void {
    const { ctx, config } = this;
    const { edges, nodes } = state;
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    for (const edge of edges) {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) continue;

      const edgeColor = EDGE_TYPE_COLORS[edge.type] || EDGE_TYPE_COLORS.default;
      const color = config.darkMode ? edgeColor.dark : edgeColor.light;

      // Calculate activation-based opacity
      const avgActivation = (source.activation + target.activation) / 2;
      const baseOpacity = 0.3 + avgActivation * 0.7;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);

      if (edge.isOnPath) {
        // Highlighted path edge
        ctx.strokeStyle = colors.path;
        ctx.lineWidth = config.pathWidth;
        ctx.globalAlpha = 1;
      } else {
        ctx.strokeStyle = color;
        ctx.lineWidth = config.edgeWidth * edge.weight;
        ctx.globalAlpha = baseOpacity;
      }

      ctx.stroke();
      ctx.globalAlpha = 1;

      // Draw arrow for directed edges
      this.drawArrow(source.x, source.y, target.x, target.y, color, edge.isOnPath ? config.pathWidth : config.edgeWidth);

      // Draw edge type label
      if (config.showEdgeTypes && edge.weight > 0.5) {
        const midX = (source.x + target.x) / 2;
        const midY = (source.y + target.y) / 2;
        ctx.font = '9px Inter, sans-serif';
        ctx.fillStyle = colors.textSecondary;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(edge.type, midX, midY - 5);
      }
    }
  }

  /**
   * Draw an arrow at the target end of an edge
   */
  private drawArrow(
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    color: string,
    lineWidth: number
  ): void {
    const { ctx, config } = this;
    const arrowSize = 6 + lineWidth;
    const angle = Math.atan2(y2 - y1, x2 - x1);

    // Position arrow at edge of target node
    const nodeRadius = config.nodeRadius;
    const arrowX = x2 - Math.cos(angle) * nodeRadius;
    const arrowY = y2 - Math.sin(angle) * nodeRadius;

    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(
      arrowX - arrowSize * Math.cos(angle - Math.PI / 6),
      arrowY - arrowSize * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      arrowX - arrowSize * Math.cos(angle + Math.PI / 6),
      arrowY - arrowSize * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  /**
   * Render all nodes
   */
  private renderNodes(
    state: GraphRenderState,
    colors: typeof COLORS.light,
    animationTime: number
  ): void {
    const { ctx, config } = this;
    const { nodes, selectedNodeId, hoveredNodeId, queryNodeId } = state;

    for (const node of nodes) {
      const nodeColor = NODE_TYPE_COLORS[node.type] || NODE_TYPE_COLORS.concept;
      const color = config.darkMode ? nodeColor.dark : nodeColor.light;

      const isSelected = node.id === selectedNodeId;
      const isHovered = node.id === hoveredNodeId;
      const isQuery = node.id === queryNodeId;
      const isOnPath = node.isOnPath;

      // Calculate radius based on importance
      const radius = config.nodeRadius * (0.8 + node.importance * 0.6);

      // Draw activation glow
      if (node.activation > 0.1) {
        const glowRadius = radius + 10 + Math.sin(animationTime * config.glowAnimationSpeed) * 3;
        const gradient = ctx.createRadialGradient(
          node.x,
          node.y,
          radius,
          node.x,
          node.y,
          glowRadius
        );
        gradient.addColorStop(0, colors.glow);
        gradient.addColorStop(1, 'transparent');

        ctx.beginPath();
        ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.globalAlpha = node.activation;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // Draw path highlight ring
      if (isOnPath) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = colors.path;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw selection/hover ring
      if (isSelected || isHovered) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + 3, 0, Math.PI * 2);
        ctx.strokeStyle = isSelected ? colors.selection : colors.textSecondary;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);

      // Fill with gradient based on activation
      if (node.activation > 0.1) {
        const gradient = ctx.createRadialGradient(
          node.x - radius * 0.3,
          node.y - radius * 0.3,
          0,
          node.x,
          node.y,
          radius
        );
        gradient.addColorStop(0, this.lightenColor(color, 30));
        gradient.addColorStop(1, color);
        ctx.fillStyle = gradient;
      } else {
        ctx.fillStyle = color;
      }

      // Query node has special styling
      if (isQuery) {
        ctx.lineWidth = 3;
        ctx.strokeStyle = colors.selection;
      } else {
        ctx.lineWidth = 1;
        ctx.strokeStyle = config.darkMode ? '#374151' : '#e5e7eb';
      }

      ctx.fill();
      ctx.stroke();

      // Draw label
      if (config.showLabels) {
        ctx.font = config.font;
        ctx.fillStyle = colors.text;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        const label = node.label.length > 15 ? node.label.slice(0, 15) + '…' : node.label;
        ctx.fillText(label, node.x, node.y + radius + 4);
      }

      // Draw activation indicator (small arc showing activation level)
      if (node.activation > 0.1) {
        ctx.beginPath();
        ctx.arc(
          node.x,
          node.y,
          radius - 2,
          -Math.PI / 2,
          -Math.PI / 2 + Math.PI * 2 * node.activation
        );
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }

  /**
   * Lighten a hex color
   */
  private lightenColor(hex: string, percent: number): string {
    const num = parseInt(hex.replace('#', ''), 16);
    const amt = Math.round(2.55 * percent);
    const R = Math.min(255, (num >> 16) + amt);
    const G = Math.min(255, ((num >> 8) & 0x00ff) + amt);
    const B = Math.min(255, (num & 0x0000ff) + amt);
    return `#${((1 << 24) | (R << 16) | (G << 8) | B).toString(16).slice(1)}`;
  }

  /**
   * Hit test - find node at a given point
   */
  getNodeAtPoint(
    x: number,
    y: number,
    state: GraphRenderState
  ): GraphNode | null {
    const { viewport, nodes } = state;
    const { config } = this;

    // Transform point to graph coordinates
    const graphX = (x - viewport.panX) / viewport.zoom;
    const graphY = (y - viewport.panY) / viewport.zoom;

    // Check nodes in reverse order (top-most first)
    for (let i = nodes.length - 1; i >= 0; i--) {
      const node = nodes[i];
      const radius = config.nodeRadius * (0.8 + node.importance * 0.6);
      const dx = graphX - node.x;
      const dy = graphY - node.y;
      if (dx * dx + dy * dy <= radius * radius) {
        return node;
      }
    }

    return null;
  }

  /**
   * Transform screen coordinates to graph coordinates
   */
  screenToGraph(
    screenX: number,
    screenY: number,
    viewport: GraphRenderState['viewport']
  ): { x: number; y: number } {
    return {
      x: (screenX - viewport.panX) / viewport.zoom,
      y: (screenY - viewport.panY) / viewport.zoom,
    };
  }

  /**
   * Transform graph coordinates to screen coordinates
   */
  graphToScreen(
    graphX: number,
    graphY: number,
    viewport: GraphRenderState['viewport']
  ): { x: number; y: number } {
    return {
      x: graphX * viewport.zoom + viewport.panX,
      y: graphY * viewport.zoom + viewport.panY,
    };
  }

  dispose(): void {
    // Clean up resources if needed
  }
}

export default KnowledgeGraph2DRenderer;
