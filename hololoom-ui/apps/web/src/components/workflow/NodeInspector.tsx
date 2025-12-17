'use client';

import { Button, Badge, Input } from '@hololoom/design-system';
import type { WorkflowNode, AgentType } from '../../app/workflow/page';

interface NodeInspectorProps {
  node: WorkflowNode | null;
  onUpdateNode: (updates: Partial<WorkflowNode>) => void;
  onDeleteNode: () => void;
}

export function NodeInspector({
  node,
  onUpdateNode,
  onDeleteNode,
}: NodeInspectorProps) {
  if (!node) {
    return (
      <div className="w-72 flex-shrink-0 border-l border-border-primary bg-bg-primary p-4">
        <div className="h-full flex flex-col items-center justify-center text-center">
          <InspectorEmptyIcon className="w-12 h-12 text-fg-tertiary/50 mb-3" />
          <p className="text-fg-tertiary text-sm">
            Select a node to view and edit its configuration
          </p>
        </div>
      </div>
    );
  }

  const configFields = getConfigFields(node.type);

  return (
    <div className="w-72 flex-shrink-0 border-l border-border-primary bg-bg-primary overflow-y-auto">
      {/* Header */}
      <div className="p-4 border-b border-border-primary">
        <div className="flex items-center justify-between mb-2">
          <Badge variant="secondary" size="sm">
            {getCategoryName(node.type)}
          </Badge>
          <StatusBadge status={node.status || 'idle'} />
        </div>
        <Input
          value={node.label}
          onChange={(e) => onUpdateNode({ label: e.target.value })}
          className="text-lg font-semibold"
        />
        <p className="text-xs text-fg-tertiary mt-1">
          ID: {node.id}
        </p>
      </div>

      {/* Configuration */}
      <div className="p-4 space-y-4">
        <h4 className="text-sm font-medium text-fg-secondary">Configuration</h4>

        {configFields.map((field) => (
          <div key={field.key}>
            <label className="block text-xs font-medium text-fg-tertiary mb-1.5">
              {field.label}
            </label>

            {field.type === 'select' ? (
              <select
                value={node.config[field.key] as string}
                onChange={(e) =>
                  onUpdateNode({
                    config: { ...node.config, [field.key]: e.target.value },
                  })
                }
                className="w-full px-3 py-2 text-sm bg-bg-secondary border border-border-primary rounded-lg text-fg-primary focus:outline-none focus:ring-1 focus:ring-cosmic-nebula"
              >
                {field.options?.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            ) : field.type === 'number' ? (
              <Input
                type="number"
                value={node.config[field.key] as number}
                onChange={(e) =>
                  onUpdateNode({
                    config: {
                      ...node.config,
                      [field.key]: parseFloat(e.target.value) || 0,
                    },
                  })
                }
                min={field.min}
                max={field.max}
                step={field.step}
              />
            ) : field.type === 'boolean' ? (
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={node.config[field.key] as boolean}
                  onChange={(e) =>
                    onUpdateNode({
                      config: { ...node.config, [field.key]: e.target.checked },
                    })
                  }
                  className="w-4 h-4 rounded border-border-primary bg-bg-secondary text-cosmic-nebula focus:ring-cosmic-nebula"
                />
                <span className="text-sm text-fg-secondary">{field.description}</span>
              </label>
            ) : (
              <Input
                value={node.config[field.key] as string}
                onChange={(e) =>
                  onUpdateNode({
                    config: { ...node.config, [field.key]: e.target.value },
                  })
                }
                placeholder={field.placeholder}
              />
            )}

            {field.description && field.type !== 'boolean' && (
              <p className="text-xs text-fg-tertiary mt-1">{field.description}</p>
            )}
          </div>
        ))}
      </div>

      {/* Position */}
      <div className="p-4 border-t border-border-primary space-y-4">
        <h4 className="text-sm font-medium text-fg-secondary">Position</h4>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs font-medium text-fg-tertiary mb-1.5">
              X
            </label>
            <Input
              type="number"
              value={Math.round(node.x)}
              onChange={(e) =>
                onUpdateNode({ x: parseInt(e.target.value) || 0 })
              }
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-fg-tertiary mb-1.5">
              Y
            </label>
            <Input
              type="number"
              value={Math.round(node.y)}
              onChange={(e) =>
                onUpdateNode({ y: parseInt(e.target.value) || 0 })
              }
            />
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="p-4 border-t border-border-primary">
        <Button
          variant="secondary"
          size="sm"
          onClick={onDeleteNode}
          className="w-full text-safety-critical hover:bg-safety-critical/10"
        >
          <TrashIcon />
          <span className="ml-1.5">Delete Node</span>
        </Button>
      </div>
    </div>
  );
}

interface ConfigField {
  key: string;
  label: string;
  type: 'text' | 'number' | 'select' | 'boolean';
  placeholder?: string;
  description?: string;
  options?: Array<{ value: string; label: string }>;
  min?: number;
  max?: number;
  step?: number;
}

function getConfigFields(type: AgentType): ConfigField[] {
  const fields: Record<AgentType, ConfigField[]> = {
    hololoom_query: [
      {
        key: 'mode',
        label: 'Mode',
        type: 'select',
        options: [
          { value: 'BARE', label: 'BARE (Fast)' },
          { value: 'FAST', label: 'FAST (Balanced)' },
          { value: 'FUSED', label: 'FUSED (Full)' },
        ],
        description: 'Processing complexity mode',
      },
      {
        key: 'maxTokens',
        label: 'Max Tokens',
        type: 'number',
        min: 100,
        max: 10000,
        step: 100,
        description: 'Maximum tokens in response',
      },
    ],
    memory_search: [
      {
        key: 'k',
        label: 'Top K',
        type: 'number',
        min: 1,
        max: 100,
        description: 'Number of results to retrieve',
      },
      {
        key: 'threshold',
        label: 'Similarity Threshold',
        type: 'number',
        min: 0,
        max: 1,
        step: 0.1,
        description: 'Minimum similarity score',
      },
    ],
    multi_query: [
      {
        key: 'maxQueries',
        label: 'Max Queries',
        type: 'number',
        min: 2,
        max: 10,
        description: 'Maximum number of sub-queries',
      },
    ],
    matryoshka_embedder: [
      {
        key: 'scale',
        label: 'Embedding Scale',
        type: 'select',
        options: [
          { value: '384D', label: '384D (Full)' },
          { value: '256D', label: '256D (Medium)' },
          { value: '128D', label: '128D (Compact)' },
        ],
        description: 'Embedding dimensionality',
      },
    ],
    synthesizer: [
      {
        key: 'extractEntities',
        label: 'Extract Entities',
        type: 'boolean',
        description: 'Extract named entities',
      },
      {
        key: 'extractMotifs',
        label: 'Extract Motifs',
        type: 'boolean',
        description: 'Extract semantic motifs',
      },
    ],
    recursive_refiner: [
      {
        key: 'strategy',
        label: 'Refinement Strategy',
        type: 'select',
        options: [
          { value: 'elegance', label: 'Elegance' },
          { value: 'verify', label: 'Verify' },
          { value: 'critique', label: 'Critique' },
          { value: 'hofstadter', label: 'Hofstadter' },
        ],
        description: 'How to refine the response',
      },
      {
        key: 'maxIterations',
        label: 'Max Iterations',
        type: 'number',
        min: 1,
        max: 10,
        description: 'Maximum refinement passes',
      },
    ],
    memory_store: [
      {
        key: 'persist',
        label: 'Persist',
        type: 'boolean',
        description: 'Save to persistent storage',
      },
    ],
    context_retriever: [
      {
        key: 'maxHops',
        label: 'Max Hops',
        type: 'number',
        min: 1,
        max: 10,
        description: 'Graph traversal depth',
      },
    ],
    knowledge_fusion: [
      {
        key: 'fusionStrategy',
        label: 'Fusion Strategy',
        type: 'select',
        options: [
          { value: 'weighted', label: 'Weighted' },
          { value: 'attention', label: 'Attention' },
          { value: 'concat', label: 'Concatenate' },
        ],
        description: 'How to fuse knowledge',
      },
    ],
    thompson_sampler: [
      {
        key: 'exploration',
        label: 'Exploration Rate',
        type: 'number',
        min: 0,
        max: 1,
        step: 0.05,
        description: 'Exploration vs exploitation balance',
      },
    ],
    convergence_engine: [
      {
        key: 'strategy',
        label: 'Collapse Strategy',
        type: 'select',
        options: [
          { value: 'argmax', label: 'Argmax' },
          { value: 'epsilon_greedy', label: 'Epsilon Greedy' },
          { value: 'bayesian_blend', label: 'Bayesian Blend' },
          { value: 'pure_thompson', label: 'Pure Thompson' },
        ],
        description: 'Decision collapse method',
      },
    ],
    safety_guardrails: [
      {
        key: 'riskThreshold',
        label: 'Risk Threshold',
        type: 'select',
        options: [
          { value: 'low', label: 'Low (Strict)' },
          { value: 'medium', label: 'Medium (Balanced)' },
          { value: 'high', label: 'High (Permissive)' },
        ],
        description: 'Maximum allowed risk level',
      },
    ],
    response_generator: [
      {
        key: 'format',
        label: 'Output Format',
        type: 'select',
        options: [
          { value: 'markdown', label: 'Markdown' },
          { value: 'html', label: 'HTML' },
          { value: 'plain', label: 'Plain Text' },
        ],
        description: 'Response format',
      },
    ],
    format_converter: [
      {
        key: 'outputFormat',
        label: 'Output Format',
        type: 'select',
        options: [
          { value: 'json', label: 'JSON' },
          { value: 'yaml', label: 'YAML' },
          { value: 'xml', label: 'XML' },
          { value: 'csv', label: 'CSV' },
        ],
        description: 'Target format',
      },
    ],
    conditional_branch: [
      {
        key: 'condition',
        label: 'Condition',
        type: 'text',
        placeholder: 'e.g., confidence > 0.8',
        description: 'JavaScript expression',
      },
    ],
    loop_iterator: [
      {
        key: 'maxIterations',
        label: 'Max Iterations',
        type: 'number',
        min: 1,
        max: 100,
        description: 'Maximum loop iterations',
      },
    ],
    parallel_executor: [
      {
        key: 'maxConcurrency',
        label: 'Max Concurrency',
        type: 'number',
        min: 2,
        max: 10,
        description: 'Parallel execution limit',
      },
    ],
    aggregator: [
      {
        key: 'aggregationMethod',
        label: 'Aggregation Method',
        type: 'select',
        options: [
          { value: 'merge', label: 'Merge' },
          { value: 'concat', label: 'Concatenate' },
          { value: 'first', label: 'First' },
          { value: 'best', label: 'Best Score' },
        ],
        description: 'How to combine inputs',
      },
    ],
  };

  return fields[type] || [];
}

function getCategoryName(type: AgentType): string {
  const categories: Record<AgentType, string> = {
    hololoom_query: 'Query',
    memory_search: 'Query',
    multi_query: 'Query',
    matryoshka_embedder: 'Processing',
    synthesizer: 'Processing',
    recursive_refiner: 'Processing',
    memory_store: 'Memory',
    context_retriever: 'Memory',
    knowledge_fusion: 'Memory',
    thompson_sampler: 'Decision',
    convergence_engine: 'Decision',
    safety_guardrails: 'Decision',
    response_generator: 'Output',
    format_converter: 'Output',
    conditional_branch: 'Control Flow',
    loop_iterator: 'Control Flow',
    parallel_executor: 'Control Flow',
    aggregator: 'Control Flow',
  };
  return categories[type];
}

function StatusBadge({ status }: { status: 'idle' | 'running' | 'success' | 'error' }) {
  const variants: Record<string, { variant: 'default' | 'secondary' | 'success' | 'warning' | 'accent'; label: string }> = {
    idle: { variant: 'secondary', label: 'Idle' },
    running: { variant: 'warning', label: 'Running' },
    success: { variant: 'success', label: 'Success' },
    error: { variant: 'accent', label: 'Error' },
  };

  const { variant, label } = variants[status];
  return <Badge variant={variant} size="sm">{label}</Badge>;
}

function InspectorEmptyIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect width="18" height="18" x="3" y="3" rx="2" />
      <path d="M9 3v18" />
      <path d="M14 9h.01" />
      <path d="M14 13h.01" />
      <path d="M14 17h.01" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
      <line x1="10" x2="10" y1="11" y2="17" />
      <line x1="14" x2="14" y1="11" y2="17" />
    </svg>
  );
}
