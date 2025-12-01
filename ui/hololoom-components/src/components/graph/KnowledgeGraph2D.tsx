/**
 * KnowledgeGraph2D Component
 *
 * Interactive 2D visualization of the HoloLoom knowledge graph.
 * Features real-time activation animation, pan/zoom, and path highlighting.
 *
 * @module components/graph/KnowledgeGraph2D
 */

import React, {
  useRef,
  useEffect,
  useCallback,
  useState,
  useMemo,
} from 'react';
import { KnowledgeGraph2DRenderer, type GraphRenderConfig, type GraphRenderState } from './KnowledgeGraph2DRenderer';
import { useAnimationFrame } from '../../hooks/useAnimationFrame';
import type { GraphNode, GraphEdge, GraphViewport } from '../../store/slices/graphSlice';

// ============================================================================
// Types
// ============================================================================

export interface KnowledgeGraph2DProps {
  /** Nodes to display */
  nodes: GraphNode[];
  /** Edges to display */
  edges: GraphEdge[];
  /** Viewport state */
  viewport: GraphViewport;
  /** Currently selected node ID */
  selectedNodeId?: string | null;
  /** Currently hovered node ID */
  hoveredNodeId?: string | null;
  /** Query node ID */
  queryNodeId?: string | null;
  /** Whether to animate */
  animated?: boolean;
  /** Dark mode */
  darkMode?: boolean;
  /** Show node labels */
  showLabels?: boolean;
  /** Show edge type labels */
  showEdgeTypes?: boolean;
  /** Callback when a node is clicked */
  onNodeClick?: (node: GraphNode) => void;
  /** Callback when a node is hovered */
  onNodeHover?: (node: GraphNode | null) => void;
  /** Callback when viewport changes (pan/zoom) */
  onViewportChange?: (viewport: Partial<GraphViewport>) => void;
  /** Additional CSS class */
  className?: string;
}

// ============================================================================
// Tooltip Component
// ============================================================================

interface TooltipProps {
  node: GraphNode;
  x: number;
  y: number;
  darkMode: boolean;
}

const Tooltip: React.FC<TooltipProps> = ({ node, x, y, darkMode }) => {
  const bgColor = darkMode ? 'bg-gray-800' : 'bg-white';
  const textColor = darkMode ? 'text-gray-100' : 'text-gray-900';
  const borderColor = darkMode ? 'border-gray-700' : 'border-gray-200';

  return (
    <div
      className={`absolute z-50 pointer-events-none ${bgColor} ${textColor} ${borderColor} border rounded-lg shadow-lg p-2 text-sm max-w-xs`}
      style={{
        left: x + 10,
        top: y + 10,
        transform: 'translateY(-50%)',
      }}
    >
      <div className="font-semibold">{node.label}</div>
      <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        Type: {node.type}
      </div>
      <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        Importance: {(node.importance * 100).toFixed(0)}%
      </div>
      {node.activation > 0.1 && (
        <div className={`text-xs ${darkMode ? 'text-indigo-400' : 'text-indigo-600'}`}>
          Activation: {(node.activation * 100).toFixed(0)}%
        </div>
      )}
      {node.isOnPath && (
        <div className={`text-xs ${darkMode ? 'text-red-400' : 'text-red-600'}`}>
          ✓ On retrieval path
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Legend Component
// ============================================================================

interface LegendProps {
  darkMode: boolean;
}

const Legend: React.FC<LegendProps> = ({ darkMode }) => {
  const bgColor = darkMode ? 'bg-gray-800/80' : 'bg-white/80';
  const textColor = darkMode ? 'text-gray-300' : 'text-gray-600';

  const nodeTypes = [
    { type: 'query', color: darkMode ? '#818cf8' : '#6366f1', label: 'Query' },
    { type: 'concept', color: darkMode ? '#60a5fa' : '#3b82f6', label: 'Concept' },
    { type: 'entity', color: darkMode ? '#34d399' : '#10b981', label: 'Entity' },
    { type: 'fact', color: darkMode ? '#fbbf24' : '#f59e0b', label: 'Fact' },
    { type: 'memory', color: darkMode ? '#a78bfa' : '#8b5cf6', label: 'Memory' },
  ];

  return (
    <div className={`absolute bottom-2 left-2 ${bgColor} backdrop-blur rounded-lg p-2 ${textColor} text-xs`}>
      <div className="font-medium mb-1">Node Types</div>
      <div className="flex flex-wrap gap-2">
        {nodeTypes.map(({ type, color, label }) => (
          <div key={type} className="flex items-center gap-1">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span>{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const KnowledgeGraph2D: React.FC<KnowledgeGraph2DProps> = ({
  nodes,
  edges,
  viewport,
  selectedNodeId = null,
  hoveredNodeId = null,
  queryNodeId = null,
  animated = true,
  darkMode = false,
  showLabels = true,
  showEdgeTypes = false,
  onNodeClick,
  onNodeHover,
  onViewportChange,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<KnowledgeGraph2DRenderer | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Pan state
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef<{ x: number; y: number; panX: number; panY: number } | null>(null);

  // Tooltip state
  const [tooltipNode, setTooltipNode] = useState<GraphNode | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

  // Animation time
  const animationTimeRef = useRef(0);

  // Check for reduced motion preference
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  const shouldAnimate = animated && !prefersReducedMotion;

  // Initialize renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const config: Partial<GraphRenderConfig> = {
      darkMode,
      showLabels,
      showEdgeTypes,
    };

    rendererRef.current = new KnowledgeGraph2DRenderer(canvas, config);

    // Initial render
    const state: GraphRenderState = {
      nodes,
      edges,
      viewport,
      selectedNodeId,
      hoveredNodeId,
      queryNodeId,
      animationTime: 0,
    };
    rendererRef.current.render(state);

    return () => {
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  // Update renderer config when props change
  useEffect(() => {
    rendererRef.current?.updateConfig({
      darkMode,
      showLabels,
      showEdgeTypes,
    });
  }, [darkMode, showLabels, showEdgeTypes]);

  // Handle resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver(() => {
      rendererRef.current?.resize();
    });

    resizeObserver.observe(container);
    return () => resizeObserver.disconnect();
  }, []);

  // Animation frame callback
  const onFrame = useCallback(
    (deltaTime: number) => {
      if (!rendererRef.current) return;

      animationTimeRef.current += deltaTime / 1000;

      const state: GraphRenderState = {
        nodes,
        edges,
        viewport,
        selectedNodeId,
        hoveredNodeId,
        queryNodeId,
        animationTime: shouldAnimate ? animationTimeRef.current : 0,
      };

      rendererRef.current.render(state);
    },
    [nodes, edges, viewport, selectedNodeId, hoveredNodeId, queryNodeId, shouldAnimate]
  );

  // Use animation frame hook
  useAnimationFrame(onFrame, {
    enabled: shouldAnimate,
    priority: 1,
  });

  // Render once when not animating
  useEffect(() => {
    if (!shouldAnimate) {
      onFrame(0);
    }
  }, [shouldAnimate, onFrame]);

  // Mouse handlers
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (e.button !== 0) return; // Left click only

      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Check if clicking on a node
      if (rendererRef.current) {
        const state: GraphRenderState = {
          nodes,
          edges,
          viewport,
          selectedNodeId,
          hoveredNodeId,
          queryNodeId,
          animationTime: 0,
        };
        const node = rendererRef.current.getNodeAtPoint(x, y, state);
        if (node) {
          onNodeClick?.(node);
          return;
        }
      }

      // Start panning
      setIsPanning(true);
      panStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        panX: viewport.panX,
        panY: viewport.panY,
      };
    },
    [nodes, edges, viewport, selectedNodeId, hoveredNodeId, queryNodeId, onNodeClick]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (isPanning && panStartRef.current) {
        const dx = e.clientX - panStartRef.current.x;
        const dy = e.clientY - panStartRef.current.y;
        onViewportChange?.({
          panX: panStartRef.current.panX + dx,
          panY: panStartRef.current.panY + dy,
        });
      } else {
        // Check for hover
        if (rendererRef.current) {
          const state: GraphRenderState = {
            nodes,
            edges,
            viewport,
            selectedNodeId,
            hoveredNodeId,
            queryNodeId,
            animationTime: 0,
          };
          const node = rendererRef.current.getNodeAtPoint(x, y, state);
          onNodeHover?.(node);

          // Update tooltip
          if (node) {
            setTooltipNode(node);
            setTooltipPosition({ x, y });
          } else {
            setTooltipNode(null);
          }
        }
      }
    },
    [isPanning, nodes, edges, viewport, selectedNodeId, hoveredNodeId, queryNodeId, onNodeHover, onViewportChange]
  );

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
    panStartRef.current = null;
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsPanning(false);
    panStartRef.current = null;
    setTooltipNode(null);
    onNodeHover?.(null);
  }, [onNodeHover]);

  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLCanvasElement>) => {
      e.preventDefault();

      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const centerX = e.clientX - rect.left;
      const centerY = e.clientY - rect.top;

      // Zoom direction (scroll up = zoom in)
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      const oldZoom = viewport.zoom;
      const newZoom = Math.max(0.1, Math.min(5, oldZoom * (1 + delta)));

      // Zoom toward cursor
      const scale = newZoom / oldZoom;
      const newPanX = centerX - (centerX - viewport.panX) * scale;
      const newPanY = centerY - (centerY - viewport.panY) * scale;

      onViewportChange?.({
        zoom: newZoom,
        panX: newPanX,
        panY: newPanY,
      });
    },
    [viewport, onViewportChange]
  );

  // Keyboard handlers for accessibility
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLCanvasElement>) => {
      const panAmount = 20;
      const zoomAmount = 0.1;

      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          onViewportChange?.({ panY: viewport.panY + panAmount });
          break;
        case 'ArrowDown':
          e.preventDefault();
          onViewportChange?.({ panY: viewport.panY - panAmount });
          break;
        case 'ArrowLeft':
          e.preventDefault();
          onViewportChange?.({ panX: viewport.panX + panAmount });
          break;
        case 'ArrowRight':
          e.preventDefault();
          onViewportChange?.({ panX: viewport.panX - panAmount });
          break;
        case '+':
        case '=':
          e.preventDefault();
          onViewportChange?.({ zoom: Math.min(5, viewport.zoom * (1 + zoomAmount)) });
          break;
        case '-':
          e.preventDefault();
          onViewportChange?.({ zoom: Math.max(0.1, viewport.zoom * (1 - zoomAmount)) });
          break;
        case '0':
          e.preventDefault();
          onViewportChange?.({ zoom: 1, panX: 0, panY: 0 });
          break;
      }
    },
    [viewport, onViewportChange]
  );

  // ARIA description
  const ariaDescription = useMemo(() => {
    const activeNodes = nodes.filter((n) => n.activation > 0.1);
    const pathNodes = nodes.filter((n) => n.isOnPath);
    return `Knowledge graph with ${nodes.length} nodes and ${edges.length} edges. ${activeNodes.length} nodes currently active. ${pathNodes.length} nodes on retrieval path.`;
  }, [nodes, edges]);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden ${className}`}
      style={{ width: viewport.width, height: viewport.height }}
    >
      <canvas
        ref={canvasRef}
        className={`w-full h-full ${isPanning ? 'cursor-grabbing' : 'cursor-grab'}`}
        tabIndex={0}
        role="img"
        aria-label="Knowledge graph visualization"
        aria-description={ariaDescription}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
        onKeyDown={handleKeyDown}
      />

      {/* Legend */}
      <Legend darkMode={darkMode} />

      {/* Tooltip */}
      {tooltipNode && (
        <Tooltip
          node={tooltipNode}
          x={tooltipPosition.x}
          y={tooltipPosition.y}
          darkMode={darkMode}
        />
      )}

      {/* Screen reader summary */}
      <div className="sr-only" aria-live="polite">
        {selectedNodeId && `Selected: ${nodes.find((n) => n.id === selectedNodeId)?.label}`}
        {hoveredNodeId && `Hovering: ${nodes.find((n) => n.id === hoveredNodeId)?.label}`}
      </div>

      {/* Controls hint */}
      <div
        className={`absolute bottom-2 right-2 text-xs ${
          darkMode ? 'text-gray-500' : 'text-gray-400'
        }`}
      >
        Scroll to zoom • Drag to pan • Click nodes to select
      </div>
    </div>
  );
};

export default KnowledgeGraph2D;
