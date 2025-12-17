import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Edit2, Check, X, Zap, Target } from 'lucide-react';
import { AgentThread, useAgentManagerStore } from '../../stores/agentManagerStore';
import { StatusBadge } from '../common/StatusBadge';

/**
 * ThreadCard Component
 * Displays a collapsible card for an agent thread with summary and detailed step list
 *
 * Features:
 * - Status indicator with color-coding
 * - Editable thread name (click to edit)
 * - Priority controls (up/down vote buttons)
 * - Expanded view showing step list with MRF/MCTS injection buttons
 * - Visual indicators for dependencies
 * - Status-based border colors with animations
 */

interface ThreadCardProps {
  thread: AgentThread;
  isActive?: boolean;
  onSelect?: (threadId: string) => void;
}

export const ThreadCard: React.FC<ThreadCardProps> = ({
  thread,
  isActive = false,
  onSelect,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState(thread.name);

  // Store actions
  const {
    updateThread,
    setActiveThread,
    upvoteThread,
    downvoteThread,
    pauseThread,
    resumeThread,
    cancelThread,
  } = useAgentManagerStore();

  // Get dependencies for display
  const { dependsOn, blocks } = useAgentManagerStore((state) =>
    state.getThreadDependencies(thread.id)
  );

  // Get child threads for hierarchy display
  const childThreads = useAgentManagerStore((state) =>
    state.getChildThreads(thread.id)
  );

  // Handlers
  const handleNameEdit = () => {
    if (isEditingName && editedName !== thread.name) {
      updateThread(thread.id, { name: editedName });
    }
    setIsEditingName(!isEditingName);
  };

  const handleNameCancel = () => {
    setEditedName(thread.name);
    setIsEditingName(false);
  };

  const handleSelectThread = () => {
    setActiveThread(thread.id);
    onSelect?.(thread.id);
  };

  // Format elapsed time
  const formatTime = (ms: number): string => {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    const seconds = (ms / 1000).toFixed(1);
    return `${seconds}s`;
  };

  // Format tokens with K suffix for thousands
  const formatTokens = (tokens: number): string => {
    if (tokens < 1000) return tokens.toString();
    return `${(tokens / 1000).toFixed(1)}k`;
  };

  // Get border color based on status
  const getBorderColor = (): string => {
    switch (thread.status) {
      case 'running':
        return 'border-blue-500';
      case 'paused':
        return 'border-amber-500';
      case 'completed':
        return 'border-emerald-500';
      case 'failed':
        return 'border-red-500';
      case 'cancelled':
        return 'border-slate-600';
      case 'idle':
      default:
        return 'border-slate-700';
    }
  };

  // Get background color based on active state
  const getBackgroundColor = (): string => {
    if (isActive) {
      return 'bg-slate-800';
    }
    return 'bg-slate-900 hover:bg-slate-850';
  };

  // Get animation classes
  const getAnimationClasses = (): string => {
    if (thread.status === 'running') {
      return 'animate-pulse';
    }
    return '';
  };

  return (
    <div
      className={`
        border-l-4 rounded-lg transition-all duration-200 cursor-pointer
        ${getBorderColor()}
        ${getBackgroundColor()}
        ${isActive ? 'ring-2 ring-blue-500' : ''}
        ${getAnimationClasses()}
      `}
      onClick={handleSelectThread}
    >
      {/* Header */}
      <div className="p-4 space-y-3">
        {/* Top row: Expand button, Status, Name, Priority controls */}
        <div className="flex items-center justify-between gap-3">
          {/* Expand/Collapse button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsExpanded(!isExpanded);
            }}
            className="p-1 text-slate-400 hover:text-slate-300 hover:bg-slate-800 rounded transition-colors"
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? (
              <ChevronDown size={16} />
            ) : (
              <ChevronRight size={16} />
            )}
          </button>

          {/* Status badge */}
          <StatusBadge status={thread.status} size="sm" showLabel={false} />

          {/* Thread name (editable) */}
          {isEditingName ? (
            <div
              className="flex-1 flex items-center gap-2"
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="text"
                value={editedName}
                onChange={(e) => setEditedName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleNameEdit();
                  if (e.key === 'Escape') handleNameCancel();
                }}
                autoFocus
                className="flex-1 px-2 py-1 bg-slate-800 border border-blue-500 rounded text-slate-100 text-sm font-medium placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-400"
                placeholder="Enter thread name"
              />
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleNameEdit();
                }}
                className="p-1 text-emerald-500 hover:text-emerald-400 hover:bg-slate-800 rounded transition-colors"
                title="Save"
              >
                <Check size={16} />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleNameCancel();
                }}
                className="p-1 text-red-500 hover:text-red-400 hover:bg-slate-800 rounded transition-colors"
                title="Cancel"
              >
                <X size={16} />
              </button>
            </div>
          ) : (
            <h3
              className="flex-1 font-semibold text-slate-100 text-sm cursor-text hover:text-blue-300 transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                setIsEditingName(true);
              }}
              title="Click to edit name"
            >
              {thread.name}
            </h3>
          )}

          {/* Priority controls */}
          <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                downvoteThread(thread.id);
              }}
              className="p-1 text-slate-500 hover:text-slate-300 hover:bg-slate-800 rounded transition-colors"
              title="Lower priority"
            >
              −
            </button>
            <div className="w-8 text-center text-xs font-semibold text-slate-400">
              {thread.priority}
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                upvoteThread(thread.id);
              }}
              className="p-1 text-slate-500 hover:text-slate-300 hover:bg-slate-800 rounded transition-colors"
              title="Raise priority"
            >
              +
            </button>
          </div>
        </div>

        {/* Summary line: Step X/Y | Xs | Xk tokens | Confidence% */}
        <div className="flex items-center justify-between">
          <div className="text-xs text-slate-400 font-mono space-x-4 flex">
            <span>
              Step {thread.currentStep}/{thread.totalSteps}
            </span>
            <span>{formatTime(thread.elapsedTimeMs)}</span>
            <span>{formatTokens(thread.tokensUsed)} tokens</span>
            {thread.tokenBudget && (
              <span className="text-slate-500">
                ({Math.round((thread.tokensUsed / thread.tokenBudget) * 100)}% of budget)
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* Confidence indicator */}
            <div className="flex items-center gap-1">
              <div className="text-xs text-slate-400">
                {Math.round(thread.confidence * 100)}%
              </div>
              <div
                className="w-2 h-2 rounded-full"
                style={{
                  backgroundColor: thread.confidence > 0.7
                    ? '#10b981'
                    : thread.confidence > 0.4
                    ? '#f59e0b'
                    : '#ef4444',
                }}
                title={`Confidence: ${Math.round(thread.confidence * 100)}%`}
              />
            </div>

            {/* Epistemic confidence indicator */}
            <div className="flex items-center gap-1">
              <div className="text-xs text-slate-500">E:</div>
              <div
                className="w-2 h-2 rounded-full"
                style={{
                  backgroundColor: thread.epistemicConfidence > 0.7
                    ? '#06b6d4'
                    : thread.epistemicConfidence > 0.4
                    ? '#84cc16'
                    : '#f43f5e',
                }}
                title={`Epistemic Confidence: ${Math.round(thread.epistemicConfidence * 100)}%`}
              />
            </div>
          </div>
        </div>

        {/* Dependencies indicator */}
        {(dependsOn.length > 0 || blocks.length > 0) && (
          <div className="text-xs space-y-1 pt-2 border-t border-slate-800">
            {dependsOn.length > 0 && (
              <div className="text-slate-400">
                <span className="text-slate-500">Waiting on:</span>{' '}
                {dependsOn.map((t) => t.name).join(', ')}
              </div>
            )}
            {blocks.length > 0 && (
              <div className="text-slate-400">
                <span className="text-slate-500">Blocks:</span>{' '}
                {blocks.map((t) => t.name).join(', ')}
              </div>
            )}
          </div>
        )}

        {/* Reasoning mode indicator */}
        <div className="flex items-center gap-2 text-xs">
          <span className="px-2 py-1 bg-slate-800 text-slate-300 rounded font-mono">
            {thread.reasoningMode}
          </span>
          <span className="px-2 py-1 bg-slate-800 text-slate-300 rounded text-xs">
            {thread.agentType}
          </span>
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div
          className="border-t border-slate-800 bg-slate-950 px-4 py-3 space-y-3"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Step list */}
          {thread.totalSteps > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-slate-400 uppercase">
                Steps ({thread.currentStep}/{thread.totalSteps})
              </h4>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {Array.from({ length: thread.totalSteps }).map((_, i) => {
                  const stepNum = i + 1;
                  const isCurrentStep = stepNum === thread.currentStep;
                  const isPastStep = stepNum < thread.currentStep;

                  return (
                    <div
                      key={i}
                      className={`
                        px-2 py-1 text-xs rounded transition-colors
                        ${isCurrentStep
                          ? 'bg-blue-900 text-blue-100'
                          : isPastStep
                          ? 'bg-emerald-900 text-emerald-100'
                          : 'bg-slate-800 text-slate-400'
                        }
                      `}
                    >
                      <span className="font-mono font-semibold">Step {stepNum}</span>
                      {isCurrentStep && (
                        <span className="ml-2 animate-pulse">● Running</span>
                      )}
                      {isPastStep && (
                        <span className="ml-2">✓ Completed</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Control buttons */}
          <div className="flex flex-wrap gap-2 pt-2 border-t border-slate-800">
            {/* Pause/Resume button */}
            {thread.status === 'running' ? (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  pauseThread(thread.id);
                }}
                className="px-3 py-1 bg-amber-600 hover:bg-amber-700 text-white text-xs font-medium rounded transition-colors"
                title="Pause thread"
              >
                ⏸ Pause
              </button>
            ) : thread.status === 'paused' ? (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  resumeThread(thread.id);
                }}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded transition-colors"
                title="Resume thread"
              >
                ▶ Resume
              </button>
            ) : null}

            {/* Cancel button */}
            {(thread.status === 'running' || thread.status === 'paused') && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  cancelThread(thread.id);
                }}
                className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded transition-colors"
                title="Cancel thread"
              >
                ✕ Cancel
              </button>
            )}

            {/* MRF Injection button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                // TODO: Implement MRF injection
                console.log('MRF injection triggered for thread:', thread.id);
              }}
              className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-xs font-medium rounded transition-colors inline-flex items-center gap-1"
              title="Inject Metaprompting Refinement Framework"
            >
              <Zap size={12} />
              MRF
            </button>

            {/* MCTS Injection button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                // TODO: Implement MCTS injection
                console.log('MCTS injection triggered for thread:', thread.id);
              }}
              className="px-3 py-1 bg-cyan-600 hover:bg-cyan-700 text-white text-xs font-medium rounded transition-colors inline-flex items-center gap-1"
              title="Inject Monte Carlo Tree Search"
            >
              <Target size={12} />
              MCTS
            </button>
          </div>

          {/* Child threads indicator */}
          {childThreads.length > 0 && (
            <div className="text-xs text-slate-400 pt-2 border-t border-slate-800">
              <span className="text-slate-500">Spawned threads:</span> {childThreads.length}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ThreadCard;
