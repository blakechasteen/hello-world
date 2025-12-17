'use client';

import { useState, useCallback } from 'react';
import { Navigation } from '../../components/Navigation';
import { Button, Badge } from '@hololoom/design-system';
import {
  WorkflowCanvas,
  AgentPalette,
  WorkflowToolbar,
  NodeInspector,
  ExecutionPanel,
} from '../../components/workflow';

export interface WorkflowNode {
  id: string;
  type: AgentType;
  label: string;
  x: number;
  y: number;
  config: Record<string, unknown>;
  status?: 'idle' | 'running' | 'success' | 'error';
}

export interface WorkflowConnection {
  id: string;
  sourceId: string;
  targetId: string;
  sourcePort: 'output';
  targetPort: 'input';
}

export type AgentType =
  // Query Agents
  | 'hololoom_query'
  | 'memory_search'
  | 'multi_query'
  // Processing Agents
  | 'matryoshka_embedder'
  | 'synthesizer'
  | 'recursive_refiner'
  // Memory Agents
  | 'memory_store'
  | 'context_retriever'
  | 'knowledge_fusion'
  // Decision Agents
  | 'thompson_sampler'
  | 'convergence_engine'
  | 'safety_guardrails'
  // Output Agents
  | 'response_generator'
  | 'format_converter'
  // Control Flow
  | 'conditional_branch'
  | 'loop_iterator'
  | 'parallel_executor'
  | 'aggregator';

export interface Workflow {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  createdAt: Date;
  updatedAt: Date;
}

export default function WorkflowPage() {
  const [workflow, setWorkflow] = useState<Workflow>({
    id: 'new-workflow',
    name: 'Untitled Workflow',
    description: '',
    nodes: [],
    connections: [],
    createdAt: new Date(),
    updatedAt: new Date(),
  });

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResults, setExecutionResults] = useState<Record<string, unknown> | null>(null);
  const [showPalette, setShowPalette] = useState(true);
  const [showInspector, setShowInspector] = useState(true);

  const selectedNode = workflow.nodes.find((n) => n.id === selectedNodeId) || null;

  const handleAddNode = useCallback((type: AgentType, x: number, y: number) => {
    const newNode: WorkflowNode = {
      id: `node-${Date.now()}`,
      type,
      label: getDefaultLabel(type),
      x,
      y,
      config: getDefaultConfig(type),
      status: 'idle',
    };

    setWorkflow((prev) => ({
      ...prev,
      nodes: [...prev.nodes, newNode],
      updatedAt: new Date(),
    }));

    setSelectedNodeId(newNode.id);
  }, []);

  const handleUpdateNode = useCallback((nodeId: string, updates: Partial<WorkflowNode>) => {
    setWorkflow((prev) => ({
      ...prev,
      nodes: prev.nodes.map((n) =>
        n.id === nodeId ? { ...n, ...updates } : n
      ),
      updatedAt: new Date(),
    }));
  }, []);

  const handleDeleteNode = useCallback((nodeId: string) => {
    setWorkflow((prev) => ({
      ...prev,
      nodes: prev.nodes.filter((n) => n.id !== nodeId),
      connections: prev.connections.filter(
        (c) => c.sourceId !== nodeId && c.targetId !== nodeId
      ),
      updatedAt: new Date(),
    }));

    if (selectedNodeId === nodeId) {
      setSelectedNodeId(null);
    }
  }, [selectedNodeId]);

  const handleConnect = useCallback((sourceId: string, targetId: string) => {
    // Prevent self-connections
    if (sourceId === targetId) return;

    // Prevent duplicate connections
    const exists = workflow.connections.some(
      (c) => c.sourceId === sourceId && c.targetId === targetId
    );
    if (exists) return;

    const newConnection: WorkflowConnection = {
      id: `conn-${Date.now()}`,
      sourceId,
      targetId,
      sourcePort: 'output',
      targetPort: 'input',
    };

    setWorkflow((prev) => ({
      ...prev,
      connections: [...prev.connections, newConnection],
      updatedAt: new Date(),
    }));
  }, [workflow.connections]);

  const handleDeleteConnection = useCallback((connectionId: string) => {
    setWorkflow((prev) => ({
      ...prev,
      connections: prev.connections.filter((c) => c.id !== connectionId),
      updatedAt: new Date(),
    }));
  }, []);

  const handleExecute = useCallback(async () => {
    setIsExecuting(true);
    setExecutionResults(null);

    // Reset all node statuses
    setWorkflow((prev) => ({
      ...prev,
      nodes: prev.nodes.map((n) => ({ ...n, status: 'idle' as const })),
    }));

    try {
      // Simulate workflow execution
      for (const node of workflow.nodes) {
        setWorkflow((prev) => ({
          ...prev,
          nodes: prev.nodes.map((n) =>
            n.id === node.id ? { ...n, status: 'running' as const } : n
          ),
        }));

        await new Promise((resolve) => setTimeout(resolve, 500 + Math.random() * 500));

        setWorkflow((prev) => ({
          ...prev,
          nodes: prev.nodes.map((n) =>
            n.id === node.id ? { ...n, status: 'success' as const } : n
          ),
        }));
      }

      setExecutionResults({
        success: true,
        output: 'Workflow completed successfully',
        duration: `${(workflow.nodes.length * 0.75).toFixed(1)}s`,
      });
    } catch (error) {
      setExecutionResults({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      setIsExecuting(false);
    }
  }, [workflow.nodes]);

  const handleSave = useCallback(() => {
    const json = JSON.stringify(workflow, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${workflow.name.toLowerCase().replace(/\s+/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [workflow]);

  const handleLoad = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const loaded = JSON.parse(e.target?.result as string);
        setWorkflow({
          ...loaded,
          createdAt: new Date(loaded.createdAt),
          updatedAt: new Date(loaded.updatedAt),
        });
        setSelectedNodeId(null);
      } catch (error) {
        console.error('Failed to load workflow:', error);
      }
    };
    reader.readAsText(file);
  }, []);

  return (
    <div className="min-h-screen bg-bg-primary flex flex-col">
      <Navigation />

      <main className="flex-1 flex flex-col">
        {/* Toolbar */}
        <WorkflowToolbar
          workflow={workflow}
          isExecuting={isExecuting}
          onExecute={handleExecute}
          onSave={handleSave}
          onLoad={handleLoad}
          onNameChange={(name) => setWorkflow((prev) => ({ ...prev, name }))}
          onTogglePalette={() => setShowPalette(!showPalette)}
          onToggleInspector={() => setShowInspector(!showInspector)}
          showPalette={showPalette}
          showInspector={showInspector}
        />

        {/* Main Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Agent Palette */}
          {showPalette && (
            <AgentPalette onDragStart={handleAddNode} />
          )}

          {/* Canvas */}
          <div className="flex-1 relative">
            <WorkflowCanvas
              nodes={workflow.nodes}
              connections={workflow.connections}
              selectedNodeId={selectedNodeId}
              onSelectNode={setSelectedNodeId}
              onUpdateNode={handleUpdateNode}
              onDeleteNode={handleDeleteNode}
              onConnect={handleConnect}
              onDeleteConnection={handleDeleteConnection}
              onAddNode={handleAddNode}
            />

            {/* Execution Panel */}
            {executionResults && (
              <ExecutionPanel
                results={executionResults}
                onClose={() => setExecutionResults(null)}
              />
            )}
          </div>

          {/* Inspector */}
          {showInspector && (
            <NodeInspector
              node={selectedNode}
              onUpdateNode={(updates) => {
                if (selectedNodeId) {
                  handleUpdateNode(selectedNodeId, updates);
                }
              }}
              onDeleteNode={() => {
                if (selectedNodeId) {
                  handleDeleteNode(selectedNodeId);
                }
              }}
            />
          )}
        </div>
      </main>
    </div>
  );
}

function getDefaultLabel(type: AgentType): string {
  const labels: Record<AgentType, string> = {
    hololoom_query: 'HoloLoom Query',
    memory_search: 'Memory Search',
    multi_query: 'Multi-Query',
    matryoshka_embedder: 'Matryoshka Embedder',
    synthesizer: 'Synthesizer',
    recursive_refiner: 'Recursive Refiner',
    memory_store: 'Memory Store',
    context_retriever: 'Context Retriever',
    knowledge_fusion: 'Knowledge Fusion',
    thompson_sampler: 'Thompson Sampler',
    convergence_engine: 'Convergence Engine',
    safety_guardrails: 'Safety Guardrails',
    response_generator: 'Response Generator',
    format_converter: 'Format Converter',
    conditional_branch: 'Conditional Branch',
    loop_iterator: 'Loop Iterator',
    parallel_executor: 'Parallel Executor',
    aggregator: 'Aggregator',
  };
  return labels[type];
}

function getDefaultConfig(type: AgentType): Record<string, unknown> {
  const configs: Record<AgentType, Record<string, unknown>> = {
    hololoom_query: { mode: 'FAST', maxTokens: 2000 },
    memory_search: { k: 10, threshold: 0.7 },
    multi_query: { maxQueries: 5 },
    matryoshka_embedder: { scale: '384D' },
    synthesizer: { extractEntities: true, extractMotifs: true },
    recursive_refiner: { strategy: 'elegance', maxIterations: 3 },
    memory_store: { persist: true },
    context_retriever: { maxHops: 3 },
    knowledge_fusion: { fusionStrategy: 'weighted' },
    thompson_sampler: { exploration: 0.15 },
    convergence_engine: { strategy: 'bayesian_blend' },
    safety_guardrails: { riskThreshold: 'medium' },
    response_generator: { format: 'markdown' },
    format_converter: { outputFormat: 'json' },
    conditional_branch: { condition: '' },
    loop_iterator: { maxIterations: 10 },
    parallel_executor: { maxConcurrency: 5 },
    aggregator: { aggregationMethod: 'merge' },
  };
  return configs[type];
}
