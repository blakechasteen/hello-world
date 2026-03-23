'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Card, CardHeader, CardBody, Button, Badge } from '@hololoom/design-system';
import { useMemoryGraph as useMemoryGraphData } from '@hololoom/api-client';
import type { MemoryNode as ApiMemoryNode, MemoryEdge as ApiMemoryEdge } from '@hololoom/api-client';

interface MemoryGraphProps {
  searchQuery: string;
  onNodeSelect: (nodeId: string) => void;
  selectedNodeId: string | null;
  highlightedNodeIds?: Set<string>;
}

interface GraphNode {
  id: string;
  label: string;
  type: 'concept' | 'entity' | 'memory' | 'relationship';
  x: number;
  y: number;
  activation: number;
  connections: number;
}

interface GraphEdge {
  source: string;
  target: string;
  type: 'IS_A' | 'USES' | 'MENTIONS' | 'LEADS_TO' | 'PART_OF';
  weight: number;
}

const EDGE_COLORS: Record<string, string> = {
  IS_A: '#6366f1',      // Indigo
  USES: '#22c55e',      // Green
  MENTIONS: '#94a3b8',  // Gray
  LEADS_TO: '#f97316',  // Orange
  PART_OF: '#a855f7',   // Purple
};

const NODE_COLORS: Record<string, string> = {
  concept: '#8b5cf6',   // Violet
  entity: '#3b82f6',    // Blue
  memory: '#10b981',    // Emerald
  relationship: '#f59e0b', // Amber
};

export function MemoryGraph({
  searchQuery,
  onNodeSelect,
  selectedNodeId,
  highlightedNodeIds,
}: MemoryGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Fetch graph from backend via DataSource protocol (degrades: live → stale → mock → empty)
  const { data: graphData, source: dataSource } = useMemoryGraphData();

  // Mock fallback data for when backend is unavailable
  const MOCK_NODES: GraphNode[] = [
    { id: 'thompson', label: 'Thompson Sampling', type: 'concept', x: 300, y: 200, activation: 0.9, connections: 5 },
    { id: 'bayesian', label: 'Bayesian', type: 'concept', x: 450, y: 150, activation: 0.8, connections: 4 },
    { id: 'bandit', label: 'Multi-Armed Bandit', type: 'concept', x: 450, y: 250, activation: 0.7, connections: 3 },
    { id: 'exploration', label: 'Exploration', type: 'entity', x: 200, y: 100, activation: 0.6, connections: 3 },
    { id: 'exploitation', label: 'Exploitation', type: 'entity', x: 200, y: 300, activation: 0.6, connections: 3 },
    { id: 'rl', label: 'Reinforcement Learning', type: 'concept', x: 550, y: 200, activation: 0.75, connections: 6 },
    { id: 'policy', label: 'Policy', type: 'entity', x: 650, y: 150, activation: 0.65, connections: 4 },
    { id: 'reward', label: 'Reward', type: 'entity', x: 650, y: 250, activation: 0.55, connections: 3 },
    { id: 'ucb', label: 'UCB Algorithm', type: 'concept', x: 350, y: 350, activation: 0.5, connections: 2 },
    { id: 'neural', label: 'Neural Network', type: 'concept', x: 550, y: 350, activation: 0.7, connections: 5 },
  ];
  const MOCK_EDGES: GraphEdge[] = [
    { source: 'thompson', target: 'bayesian', type: 'USES', weight: 0.9 },
    { source: 'thompson', target: 'bandit', type: 'IS_A', weight: 0.8 },
    { source: 'thompson', target: 'exploration', type: 'USES', weight: 0.7 },
    { source: 'thompson', target: 'exploitation', type: 'USES', weight: 0.7 },
    { source: 'bandit', target: 'rl', type: 'PART_OF', weight: 0.6 },
    { source: 'rl', target: 'policy', type: 'USES', weight: 0.8 },
    { source: 'rl', target: 'reward', type: 'USES', weight: 0.8 },
    { source: 'ucb', target: 'bandit', type: 'IS_A', weight: 0.7 },
    { source: 'neural', target: 'policy', type: 'USES', weight: 0.6 },
    { source: 'neural', target: 'rl', type: 'PART_OF', weight: 0.5 },
  ];

  // Adapt API data → canvas rendering types, or fallback to mock
  const { nodes, edges } = useMemo(() => {
    if (!graphData || dataSource === 'empty') {
      return { nodes: MOCK_NODES, edges: MOCK_EDGES };
    }

    // Map API MemoryNode → GraphNode with circular layout
    const apiNodes: GraphNode[] = graphData.nodes.map((n: ApiMemoryNode, i: number) => {
      const angle = (i / graphData.nodes.length) * Math.PI * 2;
      const radius = 200;
      return {
        id: n.id,
        label: n.content.slice(0, 30),
        type: (n.type === 'entity' || n.type === 'concept' || n.type === 'memory' || n.type === 'relationship')
          ? n.type as GraphNode['type']
          : 'memory',
        x: 400 + Math.cos(angle) * radius,
        y: 250 + Math.sin(angle) * radius,
        activation: n.activation,
        connections: n.connections?.length ?? 0,
      };
    });

    const apiEdges: GraphEdge[] = graphData.edges.map((e: ApiMemoryEdge) => ({
      source: e.source,
      target: e.target,
      type: e.type as GraphEdge['type'],
      weight: e.weight,
    }));

    return { nodes: apiNodes, edges: apiEdges };
  }, [graphData, dataSource]);

  // Filter by search query
  const filteredNodes = useMemo(() => {
    if (!searchQuery) return nodes;
    const query = searchQuery.toLowerCase();
    const matched = nodes.filter(
      (n) => n.label.toLowerCase().includes(query) || n.type.includes(query)
    );
    const nodeIds = new Set(matched.map((n) => n.id));
    edges.forEach((e) => {
      if (nodeIds.has(e.source) || nodeIds.has(e.target)) {
        nodeIds.add(e.source);
        nodeIds.add(e.target);
      }
    });
    return nodes.filter((n) => nodeIds.has(n.id));
  }, [nodes, edges, searchQuery]);

  const filteredEdges = useMemo(() => {
    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    return edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));
  }, [edges, filteredNodes]);

  // Draw the graph
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      // Clear
      ctx.fillStyle = 'transparent';
      ctx.fillRect(0, 0, rect.width, rect.height);

      ctx.save();
      ctx.translate(transform.x + rect.width / 2, transform.y + rect.height / 2);
      ctx.scale(transform.scale, transform.scale);
      ctx.translate(-rect.width / 2, -rect.height / 2);

      // Draw edges
      filteredEdges.forEach((edge) => {
        const source = filteredNodes.find((n) => n.id === edge.source);
        const target = filteredNodes.find((n) => n.id === edge.target);
        if (!source || !target) return;

        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        ctx.strokeStyle = EDGE_COLORS[edge.type] || '#666';
        ctx.lineWidth = edge.weight * 2;
        ctx.globalAlpha = 0.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
      });

      // Draw nodes
      filteredNodes.forEach((node) => {
        const isSelected = node.id === selectedNodeId;
        const isHovered = node.id === hoveredNode;
        const nodeRadius = 8 + node.connections * 2;
        const activationRadius = nodeRadius + node.activation * 10;

        // Activation glow
        if (node.activation > 0.3) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, activationRadius, 0, Math.PI * 2);
          const gradient = ctx.createRadialGradient(
            node.x, node.y, nodeRadius,
            node.x, node.y, activationRadius
          );
          gradient.addColorStop(0, `${NODE_COLORS[node.type]}40`);
          gradient.addColorStop(1, 'transparent');
          ctx.fillStyle = gradient;
          ctx.fill();
        }

        // Selection ring
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, nodeRadius + 4, 0, Math.PI * 2);
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Highlight ring (from consciousness shell)
        if (highlightedNodeIds?.has(node.id) && !isSelected) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, nodeRadius + 6, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Hover ring
        if (isHovered && !isSelected) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, nodeRadius + 3, 0, Math.PI * 2);
          ctx.strokeStyle = '#ffffff80';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2);
        ctx.fillStyle = NODE_COLORS[node.type];
        ctx.fill();

        // Label
        ctx.font = '11px Inter, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#fff';
        ctx.fillText(node.label, node.x, node.y + nodeRadius + 14);
      });

      ctx.restore();
    };

    draw();
  }, [filteredNodes, filteredEdges, transform, selectedNodeId, hoveredNode, highlightedNodeIds]);

  // Handle mouse events
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - transform.x, y: e.clientY - transform.y });
  };

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - rect.width / 2 - transform.x) / transform.scale + rect.width / 2;
    const y = (e.clientY - rect.top - rect.height / 2 - transform.y) / transform.scale + rect.height / 2;

    // Check for hover
    let foundHover = false;
    for (const node of filteredNodes) {
      const nodeRadius = 8 + node.connections * 2;
      const dist = Math.sqrt(Math.pow(x - node.x, 2) + Math.pow(y - node.y, 2));
      if (dist < nodeRadius + 5) {
        setHoveredNode(node.id);
        foundHover = true;
        break;
      }
    }
    if (!foundHover) setHoveredNode(null);

    // Handle drag
    if (isDragging) {
      setTransform((prev) => ({
        ...prev,
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      }));
    }
  }, [isDragging, dragStart, transform, filteredNodes]);

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleClick = (e: React.MouseEvent) => {
    if (hoveredNode) {
      onNodeSelect(hoveredNode);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setTransform((prev) => ({
      ...prev,
      scale: Math.min(2, Math.max(0.5, prev.scale * delta)),
    }));
  };

  const resetView = () => {
    setTransform({ x: 0, y: 0, scale: 1 });
  };

  return (
    <Card className="h-[600px] flex flex-col">
      <CardHeader className="flex flex-row items-center justify-between flex-shrink-0">
        <div>
          <h3 className="text-lg font-semibold text-fg-primary">Knowledge Graph</h3>
          <p className="text-sm text-fg-tertiary">
            {filteredNodes.length} nodes, {filteredEdges.length} connections
            {dataSource !== 'live' && dataSource !== 'empty' && (
              <span className="ml-2 text-xs text-cosmic-aurora">({dataSource})</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={resetView}>
            <ResetIcon />
            <span className="ml-1">Reset</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setTransform((p) => ({ ...p, scale: p.scale * 1.2 }))}>
            <ZoomInIcon />
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setTransform((p) => ({ ...p, scale: p.scale * 0.8 }))}>
            <ZoomOutIcon />
          </Button>
        </div>
      </CardHeader>

      <CardBody className="flex-1 p-0 relative">
        <div ref={containerRef} className="absolute inset-0">
          <canvas
            ref={canvasRef}
            className="w-full h-full cursor-grab active:cursor-grabbing"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onClick={handleClick}
            onWheel={handleWheel}
          />
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-bg-primary/90 backdrop-blur-sm border border-border-primary rounded-lg p-3">
          <p className="text-xs font-medium text-fg-tertiary mb-2">Node Types</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(NODE_COLORS).map(([type, color]) => (
              <div key={type} className="flex items-center gap-1.5">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="text-xs text-fg-tertiary capitalize">{type}</span>
              </div>
            ))}
          </div>
          <p className="text-xs font-medium text-fg-tertiary mt-3 mb-2">Edge Types</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(EDGE_COLORS).slice(0, 3).map(([type, color]) => (
              <div key={type} className="flex items-center gap-1.5">
                <div
                  className="w-4 h-0.5 rounded"
                  style={{ backgroundColor: color }}
                />
                <span className="text-xs text-fg-tertiary">{type}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Hover tooltip */}
        {hoveredNode && (
          <div className="absolute top-4 right-4 bg-bg-primary/90 backdrop-blur-sm border border-border-primary rounded-lg p-3">
            <p className="text-sm font-medium text-fg-primary">
              {filteredNodes.find((n) => n.id === hoveredNode)?.label}
            </p>
            <p className="text-xs text-fg-tertiary mt-1">
              Click to view details
            </p>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function ResetIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  );
}

function ZoomInIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
      <path d="M11 8v6" />
      <path d="M8 11h6" />
    </svg>
  );
}

function ZoomOutIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
      <path d="M8 11h6" />
    </svg>
  );
}
