/**
 * DetailPanel Component
 * Main expanded view container that shows detailed information about a selected thread
 *
 * Features:
 * - Displays selected thread's full details (name, status, agent type, reasoning mode)
 * - Multi-dimensional progress tracking (steps, time, tokens) with ProgressBars component
 * - Confidence and epistemic confidence displays
 * - Tabbed interface for History, Memory, and Files views
 * - Shows dependencies (depends_on, blocks) with visual indicators
 * - Smooth slide-in animation from right
 * - Editable thread name with inline editing
 * - Responsive layout with min/max widths
 *
 * Theme:
 * - Dark theme: bg-slate-900 for panel, bg-slate-800 for cards
 * - Borders: border-slate-700
 * - Text: text-slate-100 for headings, text-slate-400 for secondary
 * - Smooth animations and transitions
 *
 * Dependencies:
 * - useAgentManagerStore: Zustand store for agent state
 * - StatusBadge: Component for status display
 * - ProgressBars: Multi-dimensional progress visualization
 *
 * Tab Layout:
 * - History: Shows step-by-step reasoning trace (StepHistory component)
 * - Memory: Shows memory nodes and knowledge graph access (MemoryNodes component)
 * - Files: Shows files accessed/modified during execution (FileTreeViewer component)
 *
 * Usage:
 * ```tsx
 * const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null);
 *
 * {selectedThreadId && (
 *   <DetailPanel
 *     threadId={selectedThreadId}
 *     onClose={() => setSelectedThreadId(null)}
 *   />
 * )}
 * ```
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useAgentManagerStore, AgentThread } from '../../stores/agentManagerStore';
import { useStepStore, MRFStrategy } from '../../stores/stepStore';
import { useInjectionFeedback } from '../../hooks';
import type { MCTSConfig } from '../../lib/api';
import { StatusBadge } from '../common/StatusBadge';
import { ProgressBars } from '../OutlineView';
import type { DetailTab, TaskNode, MemoryNode, FileNode, StepStatus, MemorySourceType, FileStatus } from './types';
import { StepHistory } from './StepHistory';
import { MemoryNodes } from './MemoryNodes';
import { FileTreeViewer } from './FileTreeViewer';
import { InjectionHistoryLog } from './InjectionHistoryLog';

/**
 * Generate mock step data based on thread state
 * This creates sample steps that demonstrate the StepHistory component
 * In production, this data would come from the API
 */
const generateMockSteps = (thread: AgentThread): TaskNode[] => {
  if (thread.totalSteps === 0) return [];

  const stepStatuses: StepStatus[] = ['completed', 'running', 'pending', 'failed', 'skipped'];
  const stepTypes = ['retrieval', 'reasoning', 'synthesis', 'verification', 'execution'];
  const baseTime = new Date(thread.createdAt).getTime();

  return Array.from({ length: thread.totalSteps }, (_, i) => {
    const stepNum = i + 1;
    let status: StepStatus;

    if (stepNum < thread.currentStep) {
      status = 'completed';
    } else if (stepNum === thread.currentStep) {
      status = thread.status === 'running' ? 'running' :
               thread.status === 'failed' ? 'failed' : 'completed';
    } else {
      status = 'pending';
    }

    const startTime = baseTime + i * 1500; // 1.5s per step simulated
    const duration = status === 'completed' ? Math.floor(Math.random() * 2000) + 500 :
                     status === 'running' ? Date.now() - startTime : 0;

    return {
      id: `step-${thread.id}-${stepNum}`,
      stepNumber: stepNum,
      name: `Step ${stepNum}: ${stepTypes[i % stepTypes.length]}`,
      status,
      startTime: new Date(startTime).toISOString(),
      endTime: status === 'completed' ? new Date(startTime + duration).toISOString() : undefined,
      durationMs: duration,
      description: `${stepTypes[i % stepTypes.length].charAt(0).toUpperCase() + stepTypes[i % stepTypes.length].slice(1)} step for ${thread.name}`,
      inputSummary: i === 0 ? 'Initial query' : `Output from Step ${stepNum - 1}`,
      outputSummary: status === 'completed' ? `Processed ${stepTypes[i % stepTypes.length]} data` : undefined,
      tokensUsed: status === 'completed' ? Math.floor(thread.tokensUsed / thread.totalSteps) :
                  status === 'running' ? Math.floor(thread.tokensUsed / thread.totalSteps * 0.5) : 0,
      confidence: status === 'completed' ? 0.7 + Math.random() * 0.3 :
                  status === 'running' ? 0.5 + Math.random() * 0.3 : 0,
      memoryNodesAccessed: status !== 'pending' ? Math.floor(Math.random() * 5) + 1 : 0,
      toolsUsed: status === 'completed' ? ['weave', 'recall'] : [],
      // MRF/MCTS injection eligibility - reasoning and verification steps are eligible
      mrfEligible: ['reasoning', 'verification', 'synthesis'].includes(stepTypes[i % stepTypes.length]),
      mctsEligible: ['reasoning', 'execution'].includes(stepTypes[i % stepTypes.length]),
      metadata: {
        reasoning_mode: thread.reasoningMode,
        agent_type: thread.agentType
      }
    };
  });
};

/**
 * Generate mock memory node data based on thread state
 * This creates sample memory nodes that demonstrate the MemoryNodes component
 * In production, this data would come from the API
 */
const generateMockMemoryNodes = (thread: AgentThread): MemoryNode[] => {
  if (thread.currentStep === 0) return [];

  const sourceTypes: MemorySourceType[] = ['graph', 'vector', 'cache', 'hot_pattern', 'awareness'];
  const nodeNames = [
    'Thompson Sampling', 'Bayesian Methods', 'Exploration Strategy',
    'Knowledge Graph', 'Memory Retrieval', 'Context Packing',
    'Neural Policy', 'Decision Engine', 'Weaving Cycle'
  ];

  const numNodes = Math.min(thread.currentStep * 2, 12);

  return Array.from({ length: numNodes }, (_, i) => ({
    id: `mem-${thread.id}-${i}`,
    name: nodeNames[i % nodeNames.length],
    content: `Memory content for ${nodeNames[i % nodeNames.length]}`,
    source: sourceTypes[i % sourceTypes.length],
    relevance: 0.5 + Math.random() * 0.5,
    accessedAtStep: Math.ceil((i + 1) / 2),
    timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
    metadata: {
      thread_id: thread.id,
      source_type: sourceTypes[i % sourceTypes.length]
    }
  }));
};

/**
 * Generate mock file tree data based on thread state
 * This creates sample files that demonstrate the FileTreeViewer component
 * In production, this data would come from the API
 */
const generateMockFiles = (thread: AgentThread): FileNode[] => {
  if (thread.status === 'idle' || thread.currentStep === 0) return [];

  const fileStatuses: FileStatus[] = ['read', 'modified', 'created', 'unchanged'];

  // Create a sample file tree structure
  return [
    {
      id: 'file-1',
      name: 'src',
      path: '/src',
      type: 'directory',
      status: 'unchanged',
      children: [
        {
          id: 'file-2',
          name: 'components',
          path: '/src/components',
          type: 'directory',
          status: 'unchanged',
          children: [
            {
              id: 'file-3',
              name: 'DetailPanel.tsx',
              path: '/src/components/DetailPanel.tsx',
              type: 'file',
              status: 'read',
              agentId: thread.id,
              accessTime: new Date().toISOString()
            },
            {
              id: 'file-4',
              name: 'StepHistory.tsx',
              path: '/src/components/StepHistory.tsx',
              type: 'file',
              status: 'modified',
              agentId: thread.id,
              accessTime: new Date().toISOString()
            }
          ]
        },
        {
          id: 'file-5',
          name: 'stores',
          path: '/src/stores',
          type: 'directory',
          status: 'unchanged',
          children: [
            {
              id: 'file-6',
              name: 'agentManagerStore.ts',
              path: '/src/stores/agentManagerStore.ts',
              type: 'file',
              status: 'read',
              agentId: thread.id,
              accessTime: new Date().toISOString()
            }
          ]
        }
      ]
    },
    {
      id: 'file-7',
      name: 'output.log',
      path: '/output.log',
      type: 'file',
      status: 'created',
      agentId: thread.id,
      accessTime: new Date().toISOString()
    }
  ];
};

/**
 * DetailPanelProps
 * Props for DetailPanel component
 */
export interface DetailPanelProps {
  /** ID of the thread to display details for */
  threadId: string;
  /** Callback when user closes the detail panel */
  onClose: () => void;
}

/**
 * DetailPanel Component
 * Shows comprehensive details for a selected agent thread
 *
 * Component structure:
 * 1. Header - Title, status badge, close button
 * 2. Confidence section - Confidence and epistemic confidence bars
 * 3. Progress section - Multi-dimensional progress (steps, time, tokens)
 * 4. Dependencies - Shows depends_on and blocks relationships
 * 5. Tabs - History, Memory, Files navigation
 * 6. Tab content - Context-specific content for each tab
 * 7. Footer - Metadata timestamps
 */
export const DetailPanel: React.FC<DetailPanelProps> = ({ threadId, onClose }) => {
  // State
  const [activeTab, setActiveTab] = useState<DetailTab>('history');
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState('');
  const editInputRef = useRef<HTMLInputElement>(null);

  // Store
  const thread = useAgentManagerStore((state) => state.getThreadById(threadId));
  const updateThread = useAgentManagerStore((state) => state.updateThread);
  const getThreadDependencies = useAgentManagerStore((state) => state.getThreadDependencies);

  // Step store for injections
  const injectMRF = useStepStore((state) => state.injectMRF);
  const injectMCTS = useStepStore((state) => state.injectMCTS);

  // WebSocket for injection feedback - subscribes to thread-specific events
  const { isConnected, lastInjectionFeedback } = useInjectionFeedback(threadId);

  // Get dependencies
  const dependencies = thread ? getThreadDependencies(threadId) : { dependsOn: [], blocks: [] };

  // Initialize edit name
  useEffect(() => {
    if (thread) {
      setEditedName(thread.name);
    }
  }, [thread]);

  // Focus edit input when editing starts
  useEffect(() => {
    if (isEditingName && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [isEditingName]);

  // Handle name save
  const handleSaveName = () => {
    if (editedName.trim() && thread) {
      updateThread(threadId, { name: editedName.trim() });
    }
    setIsEditingName(false);
  };

  // Handle name cancel
  const handleCancelEdit = () => {
    if (thread) {
      setEditedName(thread.name);
    }
    setIsEditingName(false);
  };

  // Handle keyboard events in edit mode
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSaveName();
    } else if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };

  // Injection callbacks - connect to step store
  const handleInjectMRF = useCallback(
    async (stepId: string, strategy: MRFStrategy) => {
      if (!thread) return;
      try {
        await injectMRF(thread.id, stepId, strategy);
      } catch (error) {
        console.error('MRF injection failed:', error);
      }
    },
    [thread, injectMRF]
  );

  const handleInjectMCTS = useCallback(
    async (stepId: string, config: MCTSConfig) => {
      if (!thread) return;
      try {
        await injectMCTS(thread.id, stepId, config);
      } catch (error) {
        console.error('MCTS injection failed:', error);
      }
    },
    [thread, injectMCTS]
  );

  // Return null if thread not found
  if (!thread) {
    return (
      <div className="w-full h-full bg-slate-900 border-l border-slate-700 flex items-center justify-center">
        <div className="text-center">
          <p className="text-slate-400 mb-2">Thread not found</p>
          <button
            onClick={onClose}
            className="text-slate-500 hover:text-slate-300 transition-colors"
          >
            Close panel
          </button>
        </div>
      </div>
    );
  }

  /**
   * Format milliseconds as human-readable time string
   */
  const formatTime = (ms: number): string => {
    if (ms < 1000) {
      return `${Math.round(ms)}ms`;
    }
    const seconds = (ms / 1000).toFixed(1);
    return `${seconds}s`;
  };

  return (
    <div className="w-full h-full bg-slate-900 border-l border-slate-700 flex flex-col overflow-hidden animate-slide-in">
      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-800/50 px-4 py-4 flex-shrink-0">
        {/* Title Row with Close Button */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex-1 min-w-0">
            {isEditingName ? (
              <div className="flex gap-2 items-center">
                <input
                  ref={editInputRef}
                  type="text"
                  value={editedName}
                  onChange={(e) => setEditedName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="flex-1 px-2 py-1 bg-slate-700 border border-slate-600 rounded text-slate-100 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                  placeholder="Thread name"
                />
                <button
                  onClick={handleSaveName}
                  className="px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors"
                  title="Save thread name"
                  aria-label="Save"
                >
                  ✓
                </button>
                <button
                  onClick={handleCancelEdit}
                  className="px-2 py-1 bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded transition-colors"
                  title="Cancel editing"
                  aria-label="Cancel"
                >
                  ✕
                </button>
              </div>
            ) : (
              <div
                onClick={() => setIsEditingName(true)}
                className="group cursor-pointer"
                title="Click to edit thread name"
              >
                <h2 className="text-lg font-semibold text-slate-100 group-hover:text-blue-400 transition-colors truncate">
                  {thread.name}
                </h2>
                <p className="text-xs text-slate-500 group-hover:text-slate-400 truncate">
                  ID: {thread.id}
                </p>
              </div>
            )}
          </div>

          <button
            onClick={onClose}
            className="flex-shrink-0 text-slate-500 hover:text-slate-300 hover:bg-slate-700 rounded p-1 transition-colors"
            title="Close detail panel"
            aria-label="Close detail panel"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Metadata Row */}
        <div className="flex items-center gap-3 flex-wrap">
          <StatusBadge status={thread.status} size="sm" />
          <div className="text-xs text-slate-400 bg-slate-700 px-2 py-1 rounded">
            {thread.agentType}
          </div>
          <div className="text-xs text-slate-400 bg-slate-700 px-2 py-1 rounded">
            {thread.reasoningMode}
          </div>
          <div className="text-xs text-slate-400">
            Priority: <span className="text-slate-200 font-medium">{thread.priority}</span>
          </div>
        </div>
      </div>

      {/* Confidence Section */}
      <div className="border-b border-slate-700 bg-slate-800/30 px-4 py-3 flex-shrink-0">
        <div className="grid grid-cols-2 gap-4">
          {/* Confidence */}
          <div>
            <div className="text-xs text-slate-400 font-semibold uppercase tracking-wide mb-1">
              Confidence
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-lg font-bold text-blue-400">
                {(thread.confidence * 100).toFixed(0)}%
              </div>
              <div className="flex-1 bg-slate-700 rounded-full h-1.5">
                <div
                  className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${thread.confidence * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Epistemic Confidence */}
          <div>
            <div className="text-xs text-slate-400 font-semibold uppercase tracking-wide mb-1">
              Epistemic Confidence
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-lg font-bold text-amber-400">
                {(thread.epistemicConfidence * 100).toFixed(0)}%
              </div>
              <div className="flex-1 bg-slate-700 rounded-full h-1.5">
                <div
                  className="bg-amber-500 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${thread.epistemicConfidence * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Confidence Interpretation */}
        {thread.epistemicConfidence < 0.3 && (
          <div className="mt-2 text-xs text-amber-400 bg-amber-900/20 border border-amber-700/30 rounded px-2 py-1">
            ⚠ Low epistemic confidence: System uncertainty about its knowledge is high
          </div>
        )}
      </div>

      {/* Progress Section */}
      <div className="border-b border-slate-700 bg-slate-800/30 px-4 py-3 flex-shrink-0">
        <div className="text-xs text-slate-400 font-semibold uppercase tracking-wide mb-2">
          Execution Progress
        </div>
        <ProgressBars
          currentStep={thread.currentStep}
          totalSteps={thread.totalSteps}
          elapsedTimeMs={thread.elapsedTimeMs}
          timeBudgetMs={
            thread.totalSteps > 0
              ? Math.max(thread.elapsedTimeMs * 2, 5000) // Estimate budget as 2x elapsed or 5s min
              : undefined
          }
          tokensUsed={thread.tokensUsed}
          tokenBudget={thread.tokenBudget}
          variant="detailed"
          size="sm"
          showValues={true}
        />
      </div>

      {/* Dependencies Section */}
      {(dependencies.dependsOn.length > 0 || dependencies.blocks.length > 0) && (
        <div className="border-b border-slate-700 bg-slate-800/30 px-4 py-3 flex-shrink-0">
          <div className="space-y-2">
            {/* Depends On */}
            {dependencies.dependsOn.length > 0 && (
              <div>
                <div className="text-xs text-slate-400 font-semibold uppercase tracking-wide mb-1">
                  Depends On ({dependencies.dependsOn.length})
                </div>
                <div className="flex flex-wrap gap-1">
                  {dependencies.dependsOn.map((dep) => (
                    <div
                      key={dep.id}
                      className="text-xs bg-blue-900/30 border border-blue-700/50 text-blue-300 px-2 py-1 rounded hover:bg-blue-900/50 transition-colors cursor-pointer"
                      title={`Depends on: ${dep.name}`}
                    >
                      ← {dep.name}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Blocks */}
            {dependencies.blocks.length > 0 && (
              <div>
                <div className="text-xs text-slate-400 font-semibold uppercase tracking-wide mb-1">
                  Blocks ({dependencies.blocks.length})
                </div>
                <div className="flex flex-wrap gap-1">
                  {dependencies.blocks.map((blocked) => (
                    <div
                      key={blocked.id}
                      className="text-xs bg-amber-900/30 border border-amber-700/50 text-amber-300 px-2 py-1 rounded hover:bg-amber-900/50 transition-colors cursor-pointer"
                      title={`Blocks: ${blocked.name}`}
                    >
                      → {blocked.name}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 flex gap-0 bg-slate-800/30 px-4 flex-shrink-0">
        {(['history', 'memory', 'files', 'injections'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors capitalize ${
              activeTab === tab
                ? 'text-blue-400 border-blue-500'
                : 'text-slate-400 border-transparent hover:text-slate-300 hover:border-slate-600'
            }`}
            aria-selected={activeTab === tab}
            role="tab"
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-auto">
        {/* History Tab */}
        {activeTab === 'history' && (
          <div className="p-4">
            <div className="bg-slate-800 rounded border border-slate-700 p-4">
              <div className="text-slate-400 text-sm space-y-2">
                <p className="font-semibold text-slate-300">Reasoning History</p>
                <p>
                  Step {thread.currentStep} of {thread.totalSteps}
                </p>
                <p>Elapsed: {formatTime(thread.elapsedTimeMs)}</p>
                <p>Tokens used: {thread.tokensUsed.toLocaleString()}</p>

                {/* Status-specific info */}
                {thread.status === 'completed' && thread.finalResponse && (
                  <div className="mt-3 pt-3 border-t border-slate-700">
                    <p className="font-semibold text-slate-300 mb-2">Final Response:</p>
                    <p className="text-slate-400 text-sm line-clamp-4">{thread.finalResponse}</p>
                  </div>
                )}

                {thread.status === 'failed' && (
                  <div className="mt-3 pt-3 border-t border-slate-700">
                    <p className="text-red-400">
                      Status: Failed
                      {thread.updatedAt && ` - ${new Date(thread.updatedAt).toLocaleTimeString()}`}
                    </p>
                  </div>
                )}
              </div>

              {/* StepHistory Component */}
              <div className="mt-4 border-t border-slate-700 pt-4">
                <StepHistory
                  steps={generateMockSteps(thread)}
                  currentStepIndex={thread.currentStep - 1}
                  threadId={thread.id}
                  onStepSelect={(stepId) => console.log('Selected step:', stepId)}
                  onInjectMRF={handleInjectMRF}
                  onInjectMCTS={handleInjectMCTS}
                />
              </div>
            </div>
          </div>
        )}

        {/* Memory Tab */}
        {activeTab === 'memory' && (
          <div className="p-4">
            <div className="bg-slate-800 rounded border border-slate-700 p-4">
              <div className="text-slate-400 text-sm">
                <p className="font-semibold text-slate-300 mb-2">Memory Access</p>
                {thread.swarmId && <p>Swarm ID: {thread.swarmId}</p>}
                {thread.parentThreadId && <p>Parent Thread: {thread.parentThreadId}</p>}
                {thread.childThreadIds.length > 0 && (
                  <p>Child Threads: {thread.childThreadIds.length}</p>
                )}

                {/* MemoryNodes Component */}
                <div className="mt-4 border-t border-slate-700 pt-4">
                  <MemoryNodes
                    nodes={generateMockMemoryNodes(thread)}
                    groupByStep={true}
                    onNodeClick={(nodeId) => console.log('Clicked memory node:', nodeId)}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Files Tab */}
        {activeTab === 'files' && (
          <div className="p-4">
            <div className="bg-slate-800 rounded border border-slate-700 p-4">
              <div className="text-slate-400 text-sm">
                <p className="font-semibold text-slate-300 mb-2">Files Accessed</p>
                <p>No files accessed in this execution</p>

                {/* FileTreeViewer Component */}
                <div className="mt-4 border-t border-slate-700 pt-4">
                  <FileTreeViewer
                    files={generateMockFiles(thread)}
                    activeAgentId={thread.id}
                    onFileSelect={(path) => console.log('Selected file:', path)}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Injections Tab */}
        {activeTab === 'injections' && (
          <InjectionHistoryLog
            threadId={thread.id}
            onStepClick={(stepId) => console.log('Navigate to step:', stepId)}
            className="h-full"
          />
        )}
      </div>

      {/* Footer with metadata */}
      <div className="border-t border-slate-700 bg-slate-800/30 px-4 py-2 flex-shrink-0 text-xs text-slate-500">
        <div className="grid grid-cols-2 gap-4">
          <div>
            Created: {new Date(thread.createdAt).toLocaleString()}
          </div>
          <div className="text-right">
            Updated: {new Date(thread.updatedAt).toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
};

DetailPanel.displayName = 'DetailPanel';

export default DetailPanel;
