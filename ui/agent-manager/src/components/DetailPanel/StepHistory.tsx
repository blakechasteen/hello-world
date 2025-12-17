/**
 * StepHistory Component
 * Scrollable list showing all steps executed in a thread with full details
 * Features: Virtualized scrolling, filtering, sorting, detail expansion
 *
 * HoloLoom Agent Manager UI - Phase 4
 */

import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { TaskNode, StepStatus, StepFilterOption } from './types';
import { StatusIndicator } from '../common/StatusBadge';
import { InjectMenu } from '../OutlineView/InjectMenu';
import type { MCTSConfig } from '../../lib/api';
import type { MRFStrategy } from '../../stores/stepStore';

/**
 * Props for StepHistory component
 */
interface StepHistoryProps {
  /** All steps in chronological order */
  steps: TaskNode[];

  /** Currently selected/executing step index */
  currentStepIndex: number;

  /** Thread ID for injection context */
  threadId: string;

  /** Callback when step is selected */
  onStepSelect?: (stepId: string) => void;

  /** Optional callback for injecting MRF with strategy */
  onInjectMRF?: (stepId: string, strategy: MRFStrategy) => void;

  /** Optional callback for injecting MCTS with config */
  onInjectMCTS?: (stepId: string, config: MCTSConfig) => void;

  /** Custom CSS class */
  className?: string;
}

/**
 * Single expandable step row for step history list
 */
interface StepHistoryRowProps {
  step: TaskNode;
  index: number;
  threadId: string;
  isSelected: boolean;
  onSelect: (stepId: string) => void;
  onInjectMRF?: (stepId: string, strategy: MRFStrategy) => void;
  onInjectMCTS?: (stepId: string, config: MCTSConfig) => void;
}

/**
 * Get status color classes
 */
const getStatusColorClasses = (status: StepStatus): string => {
  switch (status) {
    case 'idle':
      return 'text-slate-400';
    case 'running':
      return 'text-blue-400 animate-pulse';
    case 'paused':
      return 'text-amber-400';
    case 'completed':
      return 'text-emerald-400';
    case 'failed':
      return 'text-red-400';
    case 'cancelled':
      return 'text-slate-500';
    case 'skipped':
      return 'text-slate-500';
    default:
      return 'text-slate-400';
  }
};

/**
 * Get status display text and icon
 */
const getStatusDisplay = (status: StepStatus): { icon: string; label: string } => {
  const statusMap: Record<StepStatus, { icon: string; label: string }> = {
    idle: { icon: '○', label: 'Idle' },
    running: { icon: '▶', label: 'Running' },
    paused: { icon: '⏸', label: 'Paused' },
    completed: { icon: '✓', label: 'Completed' },
    failed: { icon: '✕', label: 'Failed' },
    cancelled: { icon: '⊗', label: 'Cancelled' },
    skipped: { icon: '—', label: 'Skipped' },
  };
  return statusMap[status];
};

/**
 * Get step type emoji
 */
const getStepTypeEmoji = (stepType: string): string => {
  const typeMap: Record<string, string> = {
    query: '🔍',
    retrieval: '📥',
    reasoning: '🧠',
    synthesis: '🔀',
    verification: '✓',
    research: '📚',
    planning: '📋',
    execution: '⚙️',
    reflection: '💭',
  };
  return typeMap[stepType] || '◆';
};

/**
 * Format elapsed time for display (ms → human readable)
 */
const formatTime = (ms: number): string => {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
};

/**
 * Format token count for display
 */
const formatTokens = (tokens: number): string => {
  if (tokens < 1000) return tokens.toString();
  if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}k`;
  return `${(tokens / 1000000).toFixed(1)}m`;
};

/**
 * Individual step row component with expand/collapse
 */
const StepHistoryRow: React.FC<StepHistoryRowProps> = ({
  step,
  index,
  threadId,
  isSelected,
  onSelect,
  onInjectMRF,
  onInjectMCTS,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const statusDisplay = getStatusDisplay(step.status);
  const statusColorClasses = getStatusColorClasses(step.status);

  // Confidence color coding
  const confidenceColor =
    step.confidence >= 0.8
      ? 'text-emerald-400'
      : step.confidence >= 0.5
        ? 'text-amber-400'
        : 'text-red-400';

  return (
    <div
      className={`
        border-b border-slate-700/50 transition-colors
        ${isSelected ? 'bg-slate-700/50 border-l-2 border-l-blue-500' : 'bg-slate-800/30 hover:bg-slate-750/30'}
      `}
    >
      {/* Main row */}
      <div
        onClick={() => onSelect(step.id)}
        className="px-4 py-2 cursor-pointer flex items-center gap-3"
      >
        {/* Expand button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            setIsExpanded(!isExpanded);
          }}
          className="flex-shrink-0 w-5 h-5 flex items-center justify-center text-slate-400 hover:text-slate-200 transition-colors"
        >
          <span className={`text-sm transition-transform ${isExpanded ? 'rotate-90' : ''}`}>
            ▶
          </span>
        </button>

        {/* Status indicator */}
        <div className={`flex-shrink-0 w-5 h-5 flex items-center justify-center ${statusColorClasses}`}>
          {statusDisplay.icon}
        </div>

        {/* Step index */}
        <div className="flex-shrink-0 w-8 text-right text-xs text-slate-500 font-mono">
          {index + 1}
        </div>

        {/* Step type emoji + name */}
        <div className="flex-1 min-w-0 flex items-center gap-2">
          <span className="flex-shrink-0 text-base">{getStepTypeEmoji(step.stepType)}</span>
          <div className="min-w-0">
            <div className="text-sm font-medium text-slate-100 truncate">
              {step.name || step.stepType}
            </div>
            {step.query && (
              <div className="text-xs text-slate-400 truncate max-w-xs">
                {step.query}
              </div>
            )}
          </div>
        </div>

        {/* Confidence score */}
        <div className={`flex-shrink-0 text-xs font-mono ${confidenceColor}`}>
          {step.confidence.toFixed(2)}
        </div>

        {/* Tokens used (visible on wider screens) */}
        <div className="hidden sm:block flex-shrink-0 text-xs text-slate-400 font-mono min-w-[50px] text-right">
          {formatTokens(step.tokensUsed)}t
        </div>

        {/* Duration */}
        <div className="hidden md:block flex-shrink-0 text-xs text-slate-400 font-mono min-w-[60px] text-right">
          {formatTime(step.elapsedTimeMs)}
        </div>

        {/* Status label (visible on wider screens) */}
        <div className="hidden lg:flex flex-shrink-0">
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${statusColorClasses}`}>
            {statusDisplay.label}
          </span>
        </div>

        {/* MRF/MCTS injection indicators */}
        {(step.mrfEligible || step.mctsEligible || step.injectionApplied) && (
          <div className="flex-shrink-0 flex items-center gap-1">
            {step.injectionApplied && (
              <span className="text-xs px-2 py-0.5 rounded bg-purple-900/40 text-purple-300 font-medium">
                {step.injectionApplied.toUpperCase()}
              </span>
            )}
            {!step.injectionApplied && step.mrfEligible && (
              <span className="text-xs text-emerald-400 opacity-60">mrf</span>
            )}
            {!step.injectionApplied && step.mctsEligible && (
              <span className="text-xs text-cyan-400 opacity-60">mcts</span>
            )}
          </div>
        )}
      </div>

      {/* Expanded details section */}
      {isExpanded && (
        <div className="px-4 py-3 bg-slate-900/40 border-t border-slate-700/30">
          <div className="space-y-3">
            {/* Query/Input */}
            {step.query && (
              <div>
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                  Query
                </div>
                <div className="text-sm text-slate-200 bg-slate-900 rounded px-3 py-2 font-mono break-words">
                  {step.query}
                </div>
              </div>
            )}

            {/* Response preview */}
            {step.response && (
              <div>
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                  Response
                </div>
                <div className="text-sm text-slate-200 bg-slate-900 rounded px-3 py-2 max-h-32 overflow-y-auto break-words">
                  {step.response}
                </div>
              </div>
            )}

            {/* Tool selected */}
            {step.toolSelected && (
              <div className="flex items-center gap-2">
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  Tool:
                </div>
                <span className="text-sm text-blue-300 bg-blue-900/20 px-2 py-0.5 rounded font-mono">
                  {step.toolSelected}
                </span>
              </div>
            )}

            {/* Detailed metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-2 border-t border-slate-700/30">
              <div className="bg-slate-800/50 rounded px-2 py-2">
                <div className="text-xs text-slate-500 uppercase tracking-wider">Tokens</div>
                <div className="text-sm font-mono text-slate-100">{formatTokens(step.tokensUsed)}</div>
              </div>

              <div className="bg-slate-800/50 rounded px-2 py-2">
                <div className="text-xs text-slate-500 uppercase tracking-wider">Time</div>
                <div className="text-sm font-mono text-slate-100">{formatTime(step.elapsedTimeMs)}</div>
              </div>

              <div className="bg-slate-800/50 rounded px-2 py-2">
                <div className="text-xs text-slate-500 uppercase tracking-wider">Confidence</div>
                <div className={`text-sm font-mono ${confidenceColor}`}>
                  {(step.confidence * 100).toFixed(0)}%
                </div>
              </div>

              <div className="bg-slate-800/50 rounded px-2 py-2">
                <div className="text-xs text-slate-500 uppercase tracking-wider">Status</div>
                <div className={`text-sm font-mono ${statusColorClasses}`}>
                  {statusDisplay.label}
                </div>
              </div>
            </div>

            {/* Dependencies */}
            {(step.dependsOn.length > 0 || step.blocks.length > 0) && (
              <div className="pt-2 border-t border-slate-700/30 space-y-1">
                {step.dependsOn.length > 0 && (
                  <div className="text-xs">
                    <span className="text-slate-500">Depends on:</span>
                    <span className="text-slate-300 ml-2">{step.dependsOn.join(', ')}</span>
                  </div>
                )}
                {step.blocks.length > 0 && (
                  <div className="text-xs">
                    <span className="text-slate-500">Blocks:</span>
                    <span className="text-slate-300 ml-2">{step.blocks.join(', ')}</span>
                  </div>
                )}
              </div>
            )}

            {/* Injection menu - uses InjectMenu component for strategy/config selection */}
            {(step.mrfEligible || step.mctsEligible) && (
              <div className="flex items-center gap-3 pt-2 border-t border-slate-700/30">
                <span className="text-xs text-slate-400">Inject:</span>
                <InjectMenu
                  threadId={threadId}
                  stepId={step.id}
                  mrfEligible={step.mrfEligible}
                  mctsEligible={step.mctsEligible}
                  onInjectMRF={(strategy) => onInjectMRF?.(step.id, strategy as MRFStrategy)}
                  onInjectMCTS={(config) => onInjectMCTS?.(step.id, config)}
                  injected={step.injectionApplied}
                  appliedStrategy={step.injectionStrategy}
                  size="md"
                />
                {step.injectionApplied && (
                  <span className="text-xs text-purple-300">
                    {step.injectionApplied.toUpperCase()}
                    {step.injectionStrategy && ` (${step.injectionStrategy})`}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * StepHistory - Main component
 * Displays all steps in a virtualized scrollable list with filtering and sorting
 */
export const StepHistory: React.FC<StepHistoryProps> = ({
  steps,
  currentStepIndex,
  threadId,
  onStepSelect,
  onInjectMRF,
  onInjectMCTS,
  className = '',
}) => {
  const [filterStatus, setFilterStatus] = useState<StepFilterOption>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOption, setSortOption] = useState<'chronological' | 'status'>('chronological');
  const containerRef = useRef<HTMLDivElement>(null);
  const currentStepRef = useRef<HTMLDivElement>(null);

  // Filter steps based on status and search query
  const filteredSteps = useMemo(() => {
    let filtered = steps;

    // Filter by status
    if (filterStatus !== 'all') {
      filtered = filtered.filter((step) => step.status === filterStatus);
    }

    // Filter by search query (searches name, query, and type)
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (step) =>
          step.name.toLowerCase().includes(query) ||
          step.query?.toLowerCase().includes(query) ||
          step.stepType.toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [steps, filterStatus, searchQuery]);

  // Sort steps
  const sortedSteps = useMemo(() => {
    if (sortOption === 'chronological') {
      return [...filteredSteps].sort((a, b) => steps.indexOf(a) - steps.indexOf(b));
    }

    // Sort by status priority
    const statusPriority: Record<StepStatus, number> = {
      running: 0,
      paused: 1,
      failed: 2,
      completed: 3,
      idle: 4,
      cancelled: 5,
      skipped: 6,
    };

    return [...filteredSteps].sort((a, b) => {
      const priorityDiff = statusPriority[a.status] - statusPriority[b.status];
      if (priorityDiff !== 0) return priorityDiff;
      return steps.indexOf(a) - steps.indexOf(b);
    });
  }, [filteredSteps, sortOption, steps]);

  // Auto-scroll to current step
  useEffect(() => {
    if (currentStepRef.current && containerRef.current) {
      currentStepRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }
  }, [currentStepIndex]);

  const currentStepId = steps[currentStepIndex]?.id;

  // Status counts for filter buttons
  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = { all: steps.length };
    steps.forEach((step) => {
      counts[step.status] = (counts[step.status] ?? 0) + 1;
    });
    return counts;
  }, [steps]);

  return (
    <div className={`flex flex-col h-full bg-slate-800 rounded-lg overflow-hidden ${className}`}>
      {/* Header with filters */}
      <div className="flex-shrink-0 border-b border-slate-700 bg-slate-850 px-4 py-3 space-y-3">
        {/* Title and stats */}
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-100">Execution History</h3>
          <div className="text-xs text-slate-400">
            {sortedSteps.length} of {steps.length} steps
          </div>
        </div>

        {/* Search bar */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search steps..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-3 py-1.5 text-sm bg-slate-700 text-slate-100 placeholder-slate-500 rounded border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
            >
              ✕
            </button>
          )}
        </div>

        {/* Filter buttons */}
        <div className="flex flex-wrap gap-2">
          {['all', 'completed', 'running', 'failed'].map((status) => (
            <button
              key={status}
              onClick={() => setFilterStatus(status as StepFilterOption)}
              className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                filterStatus === status
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
              <span className="ml-1 text-slate-400">({statusCounts[status] ?? 0})</span>
            </button>
          ))}

          {/* Divider */}
          <div className="w-px h-6 bg-slate-700 mx-1" />

          {/* Sort selector */}
          <select
            value={sortOption}
            onChange={(e) => setSortOption(e.target.value as 'chronological' | 'status')}
            className="px-2.5 py-1 text-xs bg-slate-700 text-slate-300 rounded border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="chronological">Chronological</option>
            <option value="status">By Status</option>
          </select>
        </div>
      </div>

      {/* Steps list container with scroll */}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto overscroll-contain"
      >
        {sortedSteps.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500 text-sm">
            {steps.length === 0 ? 'No steps yet' : 'No steps match filter'}
          </div>
        ) : (
          <div className="divide-y divide-slate-700/50">
            {sortedSteps.map((step, index) => {
              const isSelected = step.id === currentStepId;
              return (
                <div
                  key={step.id}
                  ref={isSelected ? currentStepRef : undefined}
                >
                  <StepHistoryRow
                    step={step}
                    index={steps.indexOf(step)}
                    threadId={threadId}
                    isSelected={isSelected}
                    onSelect={(stepId) => onStepSelect?.(stepId)}
                    onInjectMRF={onInjectMRF}
                    onInjectMCTS={onInjectMCTS}
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer with stats */}
      {sortedSteps.length > 0 && (
        <div className="flex-shrink-0 border-t border-slate-700 bg-slate-850 px-4 py-2 text-xs text-slate-400 space-y-1">
          <div>
            Total time: <span className="text-slate-200 font-mono">{formatTime(steps.reduce((sum, s) => sum + s.elapsedTimeMs, 0))}</span>
          </div>
          <div>
            Total tokens: <span className="text-slate-200 font-mono">{formatTokens(steps.reduce((sum, s) => sum + s.tokensUsed, 0))}</span>
          </div>
          <div>
            Avg confidence: <span className="text-slate-200 font-mono">{steps.length > 0 ? (steps.reduce((sum, s) => sum + s.confidence, 0) / steps.length).toFixed(2) : '—'}</span>
          </div>
        </div>
      )}
    </div>
  );
};

StepHistory.displayName = 'StepHistory';

export default StepHistory;
