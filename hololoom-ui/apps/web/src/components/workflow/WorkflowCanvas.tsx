'use client';

import { useRef, useState, useEffect, useCallback } from 'react';
import type { WorkflowNode, WorkflowConnection, AgentType } from '../../app/workflow/page';

interface WorkflowCanvasProps {
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  onUpdateNode: (nodeId: string, updates: Partial<WorkflowNode>) => void;
  onDeleteNode: (nodeId: string) => void;
  onConnect: (sourceId: string, targetId: string) => void;
  onDeleteConnection: (connectionId: string) => void;
  onAddNode: (type: AgentType, x: number, y: number) => void;
}

const NODE_WIDTH = 180;
const NODE_HEIGHT = 80;
const PORT_RADIUS = 6;

export function WorkflowCanvas({
  nodes,
  connections,
  selectedNodeId,
  onSelectNode,
  onUpdateNode,
  onDeleteNode,
  onConnect,
  onDeleteConnection,
  onAddNode,
}: WorkflowCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [scale, setScale] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [draggingNode, setDraggingNode] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [connectingFrom, setConnectingFrom] = useState<string | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Handle drop from palette
  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const agentType = e.dataTransfer.getData('agentType') as AgentType;
      if (!agentType || !canvasRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left - offset.x) / scale;
      const y = (e.clientY - rect.top - offset.y) / scale;

      onAddNode(agentType, x - NODE_WIDTH / 2, y - NODE_HEIGHT / 2);
    },
    [offset, scale, onAddNode]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }, []);

  // Pan handling
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button === 1 || (e.button === 0 && e.altKey)) {
        // Middle click or Alt+click to pan
        setIsPanning(true);
        setPanStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
        e.preventDefault();
      } else if (e.button === 0 && (e.target as HTMLElement).classList.contains('canvas-bg')) {
        // Click on background to deselect
        onSelectNode(null);
      }
    },
    [offset, onSelectNode]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      setMousePos({ x: e.clientX, y: e.clientY });

      if (isPanning) {
        setOffset({
          x: e.clientX - panStart.x,
          y: e.clientY - panStart.y,
        });
      } else if (draggingNode && canvasRef.current) {
        const rect = canvasRef.current.getBoundingClientRect();
        const x = (e.clientX - rect.left - offset.x) / scale - dragOffset.x;
        const y = (e.clientY - rect.top - offset.y) / scale - dragOffset.y;
        onUpdateNode(draggingNode, { x, y });
      }
    },
    [isPanning, panStart, draggingNode, dragOffset, offset, scale, onUpdateNode]
  );

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
    setDraggingNode(null);
    setConnectingFrom(null);
  }, []);

  // Zoom handling
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.min(Math.max(scale * delta, 0.25), 2);

      // Zoom toward cursor
      if (canvasRef.current) {
        const rect = canvasRef.current.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const newOffset = {
          x: mouseX - (mouseX - offset.x) * (newScale / scale),
          y: mouseY - (mouseY - offset.y) * (newScale / scale),
        };

        setScale(newScale);
        setOffset(newOffset);
      }
    },
    [scale, offset]
  );

  // Node drag handling
  const handleNodeMouseDown = useCallback(
    (e: React.MouseEvent, nodeId: string) => {
      if (e.button !== 0) return;
      e.stopPropagation();

      const node = nodes.find((n) => n.id === nodeId);
      if (!node || !canvasRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const mouseX = (e.clientX - rect.left - offset.x) / scale;
      const mouseY = (e.clientY - rect.top - offset.y) / scale;

      setDraggingNode(nodeId);
      setDragOffset({ x: mouseX - node.x, y: mouseY - node.y });
      onSelectNode(nodeId);
    },
    [nodes, offset, scale, onSelectNode]
  );

  // Port click handling (for connections)
  const handlePortClick = useCallback(
    (e: React.MouseEvent, nodeId: string, portType: 'input' | 'output') => {
      e.stopPropagation();

      if (portType === 'output') {
        setConnectingFrom(nodeId);
      } else if (connectingFrom && connectingFrom !== nodeId) {
        onConnect(connectingFrom, nodeId);
        setConnectingFrom(null);
      }
    },
    [connectingFrom, onConnect]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedNodeId && document.activeElement?.tagName !== 'INPUT') {
          onDeleteNode(selectedNodeId);
        }
      } else if (e.key === 'Escape') {
        onSelectNode(null);
        setConnectingFrom(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNodeId, onDeleteNode, onSelectNode]);

  // Get node position for connections
  const getNodeCenter = (nodeId: string, port: 'input' | 'output') => {
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return { x: 0, y: 0 };

    return {
      x: node.x + (port === 'output' ? NODE_WIDTH : 0),
      y: node.y + NODE_HEIGHT / 2,
    };
  };

  // Generate bezier path for connection
  const getConnectionPath = (sourceId: string, targetId: string) => {
    const source = getNodeCenter(sourceId, 'output');
    const target = getNodeCenter(targetId, 'input');
    const dx = Math.abs(target.x - source.x) / 2;

    return `M ${source.x} ${source.y} C ${source.x + dx} ${source.y}, ${target.x - dx} ${target.y}, ${target.x} ${target.y}`;
  };

  return (
    <div
      ref={canvasRef}
      className="w-full h-full overflow-hidden bg-bg-secondary cursor-grab relative"
      style={{
        backgroundImage: `
          radial-gradient(circle, var(--color-border-primary) 1px, transparent 1px)
        `,
        backgroundSize: `${20 * scale}px ${20 * scale}px`,
        backgroundPosition: `${offset.x}px ${offset.y}px`,
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      {/* Canvas background (for click detection) */}
      <div className="canvas-bg absolute inset-0" />

      {/* Transform container */}
      <div
        style={{
          transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
          transformOrigin: '0 0',
        }}
      >
        {/* Connections */}
        <svg className="absolute inset-0 pointer-events-none overflow-visible">
          {connections.map((conn) => (
            <g key={conn.id}>
              <path
                d={getConnectionPath(conn.sourceId, conn.targetId)}
                fill="none"
                stroke="var(--color-border-primary)"
                strokeWidth={2}
                className="transition-colors"
              />
              {/* Click area for deleting connection */}
              <path
                d={getConnectionPath(conn.sourceId, conn.targetId)}
                fill="none"
                stroke="transparent"
                strokeWidth={10}
                className="pointer-events-auto cursor-pointer"
                onClick={() => onDeleteConnection(conn.id)}
              />
            </g>
          ))}

          {/* Connection preview while dragging */}
          {connectingFrom && canvasRef.current && (
            <path
              d={(() => {
                const source = getNodeCenter(connectingFrom, 'output');
                const rect = canvasRef.current!.getBoundingClientRect();
                const targetX = (mousePos.x - rect.left - offset.x) / scale;
                const targetY = (mousePos.y - rect.top - offset.y) / scale;
                const dx = Math.abs(targetX - source.x) / 2;

                return `M ${source.x} ${source.y} C ${source.x + dx} ${source.y}, ${targetX - dx} ${targetY}, ${targetX} ${targetY}`;
              })()}
              fill="none"
              stroke="var(--color-cosmic-nebula)"
              strokeWidth={2}
              strokeDasharray="5,5"
              className="animate-pulse"
            />
          )}
        </svg>

        {/* Nodes */}
        {nodes.map((node) => (
          <WorkflowNodeComponent
            key={node.id}
            node={node}
            isSelected={selectedNodeId === node.id}
            isConnectingFrom={connectingFrom === node.id}
            onMouseDown={(e) => handleNodeMouseDown(e, node.id)}
            onPortClick={(e, portType) => handlePortClick(e, node.id, portType)}
          />
        ))}
      </div>

      {/* Zoom controls */}
      <div className="absolute bottom-4 right-4 flex items-center gap-2 bg-bg-primary border border-border-primary rounded-lg p-1">
        <button
          onClick={() => setScale(Math.max(scale - 0.1, 0.25))}
          className="p-1.5 hover:bg-bg-secondary rounded transition-colors"
        >
          <MinusIcon />
        </button>
        <span className="text-sm text-fg-secondary w-12 text-center">
          {Math.round(scale * 100)}%
        </span>
        <button
          onClick={() => setScale(Math.min(scale + 0.1, 2))}
          className="p-1.5 hover:bg-bg-secondary rounded transition-colors"
        >
          <PlusIcon />
        </button>
        <button
          onClick={() => {
            setScale(1);
            setOffset({ x: 0, y: 0 });
          }}
          className="p-1.5 hover:bg-bg-secondary rounded transition-colors"
        >
          <ResetIcon />
        </button>
      </div>

      {/* Empty state */}
      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <DropIcon className="w-12 h-12 mx-auto text-fg-tertiary/50 mb-3" />
            <p className="text-fg-tertiary">
              Drag agents from the palette to get started
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// Node component
interface WorkflowNodeComponentProps {
  node: WorkflowNode;
  isSelected: boolean;
  isConnectingFrom: boolean;
  onMouseDown: (e: React.MouseEvent) => void;
  onPortClick: (e: React.MouseEvent, portType: 'input' | 'output') => void;
}

function WorkflowNodeComponent({
  node,
  isSelected,
  isConnectingFrom,
  onMouseDown,
  onPortClick,
}: WorkflowNodeComponentProps) {
  const statusColors = {
    idle: 'bg-fg-tertiary/30',
    running: 'bg-cosmic-nebula animate-pulse',
    success: 'bg-confidence-high',
    error: 'bg-safety-critical',
  };

  const categoryColors: Record<string, string> = {
    hololoom_query: 'cosmic-nebula',
    memory_search: 'cosmic-nebula',
    multi_query: 'cosmic-nebula',
    matryoshka_embedder: 'cosmic-aurora',
    synthesizer: 'cosmic-aurora',
    recursive_refiner: 'cosmic-aurora',
    memory_store: 'cosmic-starlight',
    context_retriever: 'cosmic-starlight',
    knowledge_fusion: 'cosmic-starlight',
    thompson_sampler: 'confidence-high',
    convergence_engine: 'confidence-high',
    safety_guardrails: 'confidence-high',
    response_generator: 'confidence-medium',
    format_converter: 'confidence-medium',
    conditional_branch: 'fg-tertiary',
    loop_iterator: 'fg-tertiary',
    parallel_executor: 'fg-tertiary',
    aggregator: 'fg-tertiary',
  };

  const color = categoryColors[node.type] || 'fg-tertiary';

  return (
    <div
      className={`absolute cursor-move select-none transition-shadow ${
        isSelected ? 'z-10' : 'z-0'
      }`}
      style={{
        left: node.x,
        top: node.y,
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
      }}
      onMouseDown={onMouseDown}
    >
      {/* Node body */}
      <div
        className={`w-full h-full rounded-xl border-2 bg-bg-primary shadow-lg transition-colors ${
          isSelected
            ? `border-${color} shadow-${color}/20`
            : isConnectingFrom
            ? 'border-cosmic-nebula border-dashed'
            : 'border-border-primary hover:border-border-secondary'
        }`}
      >
        {/* Header */}
        <div className={`h-7 px-3 flex items-center gap-2 border-b border-border-primary rounded-t-xl bg-${color}/10`}>
          <div className={`w-2 h-2 rounded-full ${statusColors[node.status || 'idle']}`} />
          <span className="text-xs font-medium text-fg-primary truncate">
            {node.label}
          </span>
        </div>

        {/* Body */}
        <div className="px-3 py-2">
          <p className="text-xs text-fg-tertiary truncate">
            {getNodeSubtitle(node)}
          </p>
        </div>
      </div>

      {/* Input port */}
      <div
        className="absolute -left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 border-border-primary bg-bg-primary hover:border-cosmic-nebula hover:bg-cosmic-nebula/20 transition-colors cursor-crosshair"
        onClick={(e) => onPortClick(e, 'input')}
      />

      {/* Output port */}
      <div
        className={`absolute -right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full border-2 transition-colors cursor-crosshair ${
          isConnectingFrom
            ? 'border-cosmic-nebula bg-cosmic-nebula'
            : 'border-border-primary bg-bg-primary hover:border-cosmic-nebula hover:bg-cosmic-nebula/20'
        }`}
        onClick={(e) => onPortClick(e, 'output')}
      />
    </div>
  );
}

function getNodeSubtitle(node: WorkflowNode): string {
  switch (node.type) {
    case 'hololoom_query':
      return `Mode: ${node.config.mode}`;
    case 'memory_search':
      return `k=${node.config.k}, threshold=${node.config.threshold}`;
    case 'matryoshka_embedder':
      return `Scale: ${node.config.scale}`;
    case 'recursive_refiner':
      return `Strategy: ${node.config.strategy}`;
    case 'thompson_sampler':
      return `Exploration: ${(node.config.exploration as number) * 100}%`;
    case 'safety_guardrails':
      return `Risk: ${node.config.riskThreshold}`;
    case 'conditional_branch':
      return node.config.condition ? `if: ${node.config.condition}` : 'Configure condition';
    case 'loop_iterator':
      return `Max: ${node.config.maxIterations} iterations`;
    default:
      return node.type.replace(/_/g, ' ');
  }
}

// Icons
function MinusIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14" />
      <path d="M12 5v14" />
    </svg>
  );
}

function ResetIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  );
}

function DropIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  );
}
