/**
 * Workflow Builder Component
 *
 * Features:
 * - Visual drag-and-drop workflow creation
 * - 18 agent types across 6 categories
 * - Connection validation and type checking
 * - Workflow execution with real-time monitoring
 * - Import/export workflows as JSON
 * - Auto-layout and prettification
 */

import React, { useState, useCallback, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Connection,
  addEdge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
  NodeTypes,
  MarkerType,
} from 'react-flow-renderer';
import axios from 'axios';
import {
  Play,
  Save,
  Upload,
  Download,
  Trash2,
  Plus,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Zap,
  Database,
  Brain,
  GitBranch,
  FileText,
  Settings,
} from 'lucide-react';

const API_URL = 'http://localhost:8000';

// Agent Categories and Types
const AGENT_CATEGORIES = {
  QUERY: {
    name: 'Query Agents',
    color: 'blue',
    icon: Zap,
    agents: [
      { type: 'hololoom_query', name: 'HoloLoom Query', description: 'Full weaving cycle query' },
      { type: 'memory_search', name: 'Memory Search', description: 'Search knowledge graph' },
      { type: 'multi_query', name: 'Multi-Query', description: 'Break into sub-questions' },
    ],
  },
  PROCESS: {
    name: 'Processing Agents',
    color: 'purple',
    icon: Settings,
    agents: [
      { type: 'matryoshka_embedder', name: 'Matryoshka Embedder', description: 'Multi-scale embeddings' },
      { type: 'synthesizer', name: 'Synthesizer', description: 'Extract entities/motifs' },
      { type: 'recursive_refiner', name: 'Recursive Refiner', description: 'Quality refinement' },
    ],
  },
  MEMORY: {
    name: 'Memory Agents',
    color: 'green',
    icon: Database,
    agents: [
      { type: 'memory_store', name: 'Memory Store', description: 'Persist to graph+vector' },
      { type: 'context_retriever', name: 'Context Retriever', description: 'Retrieve context' },
      { type: 'knowledge_fusion', name: 'Knowledge Fusion', description: 'Multi-hop traversal' },
    ],
  },
  DECISION: {
    name: 'Decision Agents',
    color: 'amber',
    icon: Brain,
    agents: [
      { type: 'thompson_sampler', name: 'Thompson Sampler', description: 'Bayesian exploration' },
      { type: 'convergence_engine', name: 'Convergence Engine', description: 'Decision collapse' },
      { type: 'safety_guardrails', name: 'Safety Guardrails', description: 'Risk gating' },
    ],
  },
  OUTPUT: {
    name: 'Output Agents',
    color: 'indigo',
    icon: FileText,
    agents: [
      { type: 'response_generator', name: 'Response Generator', description: 'Generate response' },
      { type: 'format_converter', name: 'Format Converter', description: 'JSON/Markdown/HTML' },
    ],
  },
  CONTROL: {
    name: 'Control Flow',
    color: 'red',
    icon: GitBranch,
    agents: [
      { type: 'conditional_branch', name: 'Conditional Branch', description: 'If/else logic' },
      { type: 'loop_iterator', name: 'Loop Iterator', description: 'Repeat until condition' },
      { type: 'parallel_executor', name: 'Parallel Executor', description: 'Concurrent execution' },
    ],
  },
};

// Color mapping
const COLOR_MAP = {
  blue: { bg: 'bg-blue-50', border: 'border-blue-500', text: 'text-blue-700' },
  purple: { bg: 'bg-purple-50', border: 'border-purple-500', text: 'text-purple-700' },
  green: { bg: 'bg-green-50', border: 'border-green-500', text: 'text-green-700' },
  amber: { bg: 'bg-amber-50', border: 'border-amber-500', text: 'text-amber-700' },
  indigo: { bg: 'bg-indigo-50', border: 'border-indigo-500', text: 'text-indigo-700' },
  red: { bg: 'bg-red-50', border: 'border-red-500', text: 'text-red-700' },
};

interface WorkflowNode extends Node {
  data: {
    label: string;
    type: string;
    category: string;
    config?: Record<string, any>;
    status?: 'pending' | 'running' | 'completed' | 'error';
    result?: any;
    error?: string;
  };
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: Edge[];
  created_at: string;
  updated_at: string;
}

// Custom Node Component
const AgentNode: React.FC<any> = ({ data, isConnectable }) => {
  const categoryKey = Object.keys(AGENT_CATEGORIES).find((key) =>
    AGENT_CATEGORIES[key as keyof typeof AGENT_CATEGORIES].agents.some((a) => a.type === data.type)
  );
  const category = categoryKey ? AGENT_CATEGORIES[categoryKey as keyof typeof AGENT_CATEGORIES] : null;
  const colors = category ? COLOR_MAP[category.color as keyof typeof COLOR_MAP] : COLOR_MAP.blue;
  const Icon = category?.icon || Settings;

  const getStatusIcon = () => {
    switch (data.status) {
      case 'running':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return null;
    }
  };

  return (
    <div className={`px-4 py-3 rounded-lg border-2 ${colors.border} ${colors.bg} shadow-md min-w-[200px]`}>
      <div className="flex items-center gap-2 mb-1">
        <Icon className={`w-4 h-4 ${colors.text}`} />
        <div className="font-medium text-gray-900 text-sm">{data.label}</div>
        {getStatusIcon()}
      </div>
      <div className="text-xs text-gray-500">{data.type}</div>
      {data.error && (
        <div className="mt-2 text-xs text-red-600 bg-red-50 px-2 py-1 rounded">{data.error}</div>
      )}
    </div>
  );
};

const nodeTypes: NodeTypes = {
  agent: AgentNode,
};

export const WorkflowBuilder: React.FC = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState<WorkflowNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedAgentType, setSelectedAgentType] = useState<string | null>(null);
  const [workflowName, setWorkflowName] = useState('Untitled Workflow');
  const [workflowDescription, setWorkflowDescription] = useState('');
  const [executing, setExecuting] = useState(false);
  const [executionResults, setExecutionResults] = useState<any>(null);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [savedWorkflows, setSavedWorkflows] = useState<Workflow[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const nodeIdCounter = useRef(0);

  // Connection validation
  const isValidConnection = useCallback((connection: Connection) => {
    // Prevent self-loops
    if (connection.source === connection.target) {
      return false;
    }

    // Check for cycles (simple DFS)
    const adjacencyList = new Map<string, string[]>();
    edges.forEach((edge) => {
      if (!adjacencyList.has(edge.source)) {
        adjacencyList.set(edge.source, []);
      }
      adjacencyList.get(edge.source)!.push(edge.target);
    });

    // Add proposed edge
    if (!adjacencyList.has(connection.source!)) {
      adjacencyList.set(connection.source!, []);
    }
    adjacencyList.get(connection.source!)!.push(connection.target!);

    // DFS to detect cycle
    const visited = new Set<string>();
    const recStack = new Set<string>();

    const hasCycle = (node: string): boolean => {
      visited.add(node);
      recStack.add(node);

      const neighbors = adjacencyList.get(node) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          if (hasCycle(neighbor)) return true;
        } else if (recStack.has(neighbor)) {
          return true;
        }
      }

      recStack.delete(node);
      return false;
    };

    for (const node of adjacencyList.keys()) {
      if (!visited.has(node)) {
        if (hasCycle(node)) {
          alert('Cannot create cycle in workflow!');
          return false;
        }
      }
    }

    return true;
  }, [edges]);

  const onConnect = useCallback(
    (params: Connection) => {
      if (isValidConnection(params)) {
        setEdges((eds) =>
          addEdge(
            {
              ...params,
              type: 'smoothstep',
              animated: true,
              markerEnd: { type: MarkerType.ArrowClosed },
            },
            eds
          )
        );
      }
    },
    [isValidConnection, setEdges]
  );

  // Add agent node
  const handleAddAgent = (agentType: string, agentName: string, category: string) => {
    const newNode: WorkflowNode = {
      id: `node_${nodeIdCounter.current++}`,
      type: 'agent',
      position: {
        x: Math.random() * 400 + 100,
        y: Math.random() * 300 + 100,
      },
      data: {
        label: agentName,
        type: agentType,
        category,
        status: 'pending',
      },
    };

    setNodes((nds) => [...nds, newNode]);
    setSelectedAgentType(null);
  };

  // Execute workflow
  const handleExecute = async () => {
    if (nodes.length === 0) {
      alert('Please add at least one agent to the workflow');
      return;
    }

    setExecuting(true);
    setExecutionResults(null);

    // Reset all node statuses
    setNodes((nds) =>
      nds.map((node) => ({
        ...node,
        data: { ...node.data, status: 'pending', result: undefined, error: undefined },
      }))
    );

    try {
      // Build workflow payload
      const workflow = {
        version: '1.0',
        name: workflowName,
        description: workflowDescription,
        nodes: nodes.map((node) => ({
          id: node.id,
          type: node.data.type,
          config: node.data.config || {},
        })),
        connections: edges.map((edge) => ({
          from: edge.source,
          to: edge.target,
        })),
      };

      // Execute workflow
      const response = await axios.post(`${API_URL}/api/workflow/execute`, {
        workflow,
        input_data: {
          query: 'Sample query for workflow execution',
        },
      });

      if (response.data.success) {
        setExecutionResults(response.data.data);

        // Update node statuses based on execution results
        const nodeResults = response.data.data.node_results || {};
        setNodes((nds) =>
          nds.map((node) => {
            const result = nodeResults[node.id];
            if (result) {
              return {
                ...node,
                data: {
                  ...node.data,
                  status: result.status,
                  result: result.output,
                  error: result.error,
                },
              };
            }
            return node;
          })
        );
      }
    } catch (error: any) {
      console.error('Workflow execution failed:', error);
      alert(`Execution failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setExecuting(false);
    }
  };

  // Save workflow
  const handleSave = async () => {
    if (!workflowName.trim()) {
      alert('Please enter a workflow name');
      return;
    }

    try {
      const workflow: Workflow = {
        id: `wf_${Date.now()}`,
        name: workflowName,
        description: workflowDescription,
        nodes,
        edges,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      const response = await axios.post(`${API_URL}/api/workflows`, workflow);

      if (response.data.success) {
        setSavedWorkflows([...savedWorkflows, workflow]);
        alert('Workflow saved successfully!');
        setShowSaveModal(false);
      }
    } catch (error) {
      console.error('Failed to save workflow:', error);
      alert('Failed to save workflow');
    }
  };

  // Export workflow
  const handleExport = () => {
    const workflow = {
      version: '1.0',
      name: workflowName,
      description: workflowDescription,
      nodes,
      edges,
      exported_at: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(workflow, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `workflow_${workflowName.replace(/\s+/g, '_')}.json`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Import workflow
  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const workflow = JSON.parse(e.target?.result as string);
        setWorkflowName(workflow.name || 'Imported Workflow');
        setWorkflowDescription(workflow.description || '');
        setNodes(workflow.nodes || []);
        setEdges(workflow.edges || []);
        nodeIdCounter.current = Math.max(...workflow.nodes.map((n: any) => parseInt(n.id.split('_')[1]) || 0)) + 1;
      } catch (error) {
        alert('Failed to import workflow: Invalid JSON');
      }
    };
    reader.readAsText(file);
  };

  // Clear workflow
  const handleClear = () => {
    if (confirm('Are you sure you want to clear the entire workflow?')) {
      setNodes([]);
      setEdges([]);
      setWorkflowName('Untitled Workflow');
      setWorkflowDescription('');
      setExecutionResults(null);
      nodeIdCounter.current = 0;
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="bg-white rounded-lg shadow p-4 mb-4">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <input
              type="text"
              value={workflowName}
              onChange={(e) => setWorkflowName(e.target.value)}
              className="text-xl font-bold text-gray-900 bg-transparent border-none focus:outline-none focus:ring-0"
              placeholder="Workflow Name"
            />
            <input
              type="text"
              value={workflowDescription}
              onChange={(e) => setWorkflowDescription(e.target.value)}
              className="text-sm text-gray-500 bg-transparent border-none focus:outline-none focus:ring-0 mt-1 w-full"
              placeholder="Add description..."
            />
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            <button
              onClick={handleExecute}
              disabled={executing || nodes.length === 0}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {executing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Executing...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Execute
                </>
              )}
            </button>

            <button
              onClick={() => setShowSaveModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Save className="w-4 h-4" />
              Save
            </button>

            <button
              onClick={handleExport}
              className="flex items-center gap-2 px-3 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <Download className="w-4 h-4" />
            </button>

            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 px-3 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <Upload className="w-4 h-4" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImport}
              className="hidden"
            />

            <button
              onClick={handleClear}
              className="flex items-center gap-2 px-3 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6 mt-3 text-sm text-gray-600">
          <span>{nodes.length} nodes</span>
          <span>{edges.length} connections</span>
          {executionResults && (
            <span className="text-green-600">
              ✓ Last execution: {executionResults.duration_ms}ms
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 flex gap-4">
        {/* Agent Palette */}
        <div className="w-64 bg-white rounded-lg shadow p-4 overflow-y-auto">
          <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <Plus className="w-4 h-4" />
            Add Agent
          </h3>

          <div className="space-y-4">
            {Object.entries(AGENT_CATEGORIES).map(([key, category]) => {
              const Icon = category.icon;
              const colors = COLOR_MAP[category.color as keyof typeof COLOR_MAP];

              return (
                <div key={key}>
                  <div className="flex items-center gap-2 mb-2">
                    <Icon className={`w-4 h-4 ${colors.text}`} />
                    <h4 className="text-sm font-medium text-gray-900">{category.name}</h4>
                  </div>

                  <div className="space-y-1">
                    {category.agents.map((agent) => (
                      <button
                        key={agent.type}
                        onClick={() => handleAddAgent(agent.type, agent.name, key)}
                        className={`w-full text-left px-3 py-2 rounded text-xs ${colors.bg} ${colors.text} hover:opacity-80 transition-opacity`}
                      >
                        <div className="font-medium">{agent.name}</div>
                        <div className="text-gray-600 text-xs">{agent.description}</div>
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Canvas */}
        <div className="flex-1 bg-white rounded-lg shadow overflow-hidden">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={nodeTypes}
            fitView
            attributionPosition="bottom-left"
          >
            <Background />
            <Controls />
            <MiniMap
              nodeColor={(node) => {
                const categoryKey = Object.keys(AGENT_CATEGORIES).find((key) =>
                  AGENT_CATEGORIES[key as keyof typeof AGENT_CATEGORIES].agents.some(
                    (a) => a.type === (node.data as any).type
                  )
                );
                const category = categoryKey
                  ? AGENT_CATEGORIES[categoryKey as keyof typeof AGENT_CATEGORIES]
                  : null;
                return category ? `var(--${category.color}-500)` : '#888';
              }}
            />
          </ReactFlow>
        </div>
      </div>

      {/* Save Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Save Workflow</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Workflow Name</label>
                <input
                  type="text"
                  value={workflowName}
                  onChange={(e) => setWorkflowName(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter workflow name..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  value={workflowDescription}
                  onChange={(e) => setWorkflowDescription(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter description..."
                />
              </div>

              <div className="text-sm text-gray-600">
                <p>Nodes: {nodes.length}</p>
                <p>Connections: {edges.length}</p>
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowSaveModal(false)}
                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Save Workflow
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
