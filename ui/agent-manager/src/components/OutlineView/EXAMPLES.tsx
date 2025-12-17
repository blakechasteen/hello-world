/**
 * Example Usage of Control Components
 * Shows common patterns for using PriorityControls, ThreadControls, and InjectMenu
 */

import React, { useState } from 'react';
import { PriorityControls, ThreadControls, InjectMenu } from './index';

// ============================================================================
// Example 1: Simple Thread Row with All Controls
// ============================================================================

interface ThreadRowExample1Props {
  threadId: string;
  name: string;
  status: 'running' | 'paused' | 'idle' | 'completed' | 'failed' | 'cancelled';
  priority: number;
}

export const ThreadRowExample1: React.FC<ThreadRowExample1Props> = ({
  threadId,
  name,
  status,
  priority,
}) => {
  return (
    <div className="flex items-center gap-2 p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors">
      {/* Status indicator */}
      <div className="w-3 h-3 rounded-full bg-blue-400" />

      {/* Thread name */}
      <span className="flex-1 font-medium">{name}</span>

      {/* Priority controls */}
      <PriorityControls
        threadId={threadId}
        priority={priority}
        size="md"
        orientation="vertical"
      />

      {/* Thread controls */}
      <ThreadControls
        threadId={threadId}
        status={status}
        size="md"
        showCancelConfirm={true}
      />

      {/* Injection menu */}
      <InjectMenu
        threadId={threadId}
        stepId="0"
        mrfEligible={status === 'running' || status === 'paused'}
        mctsEligible={status === 'running' || status === 'paused'}
        size="md"
      />
    </div>
  );
};

// ============================================================================
// Example 2: Horizontal Layout for Thread List
// ============================================================================

export const ThreadRowHorizontalExample: React.FC<ThreadRowExample1Props> = ({
  threadId,
  name,
  status,
  priority,
}) => {
  return (
    <div className="flex items-center justify-between p-3 bg-slate-800 rounded-lg mb-2 border border-slate-700">
      <div className="flex-1">
        <h3 className="font-semibold text-white">{name}</h3>
        <p className="text-xs text-slate-400 mt-1">
          Status: <span className="capitalize">{status}</span>
        </p>
      </div>

      <div className="flex items-center gap-3">
        {/* Priority in horizontal layout */}
        <PriorityControls
          threadId={threadId}
          priority={priority}
          size="md"
          orientation="horizontal"
        />

        {/* Thread controls */}
        <ThreadControls
          threadId={threadId}
          status={status}
          size="md"
        />

        {/* Inject menu */}
        <InjectMenu
          threadId={threadId}
          stepId="0"
          mrfEligible={true}
          mctsEligible={true}
          size="md"
        />
      </div>
    </div>
  );
};

// ============================================================================
// Example 3: Detailed Thread View with Step-Level Injection
// ============================================================================

interface Step {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed';
}

interface ThreadViewExampleProps {
  threadId: string;
  name: string;
  status: 'running' | 'paused' | 'completed';
  priority: number;
  steps: Step[];
}

export const ThreadDetailViewExample: React.FC<ThreadViewExampleProps> = ({
  threadId,
  name,
  status,
  priority,
  steps,
}) => {
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());

  const toggleStepExpanded = (stepId: string) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId);
    } else {
      newExpanded.add(stepId);
    }
    setExpandedSteps(newExpanded);
  };

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-700 overflow-hidden">
      {/* Header with thread controls */}
      <div className="p-4 bg-slate-800 border-b border-slate-700">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-bold text-white">{name}</h2>
          <div className="flex gap-2">
            <PriorityControls
              threadId={threadId}
              priority={priority}
              size="md"
              orientation="horizontal"
            />
            <ThreadControls
              threadId={threadId}
              status={status}
              size="md"
              showCancelConfirm={true}
            />
          </div>
        </div>
        <p className="text-sm text-slate-400">
          Status: <span className="capitalize font-medium">{status}</span>
        </p>
      </div>

      {/* Steps list */}
      <div className="divide-y divide-slate-700">
        {steps.map((step) => (
          <div key={step.id}>
            {/* Step header */}
            <button
              onClick={() => toggleStepExpanded(step.id)}
              className="w-full px-4 py-3 hover:bg-slate-800 transition-colors text-left flex items-center justify-between"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`text-sm ${
                    step.status === 'completed'
                      ? 'text-emerald-400'
                      : 'text-slate-400'
                  }`}
                >
                  {step.status === 'completed' ? '✓' : '○'}
                </span>
                <span className="font-medium text-white">{step.name}</span>
              </div>
              <span className="text-slate-400">
                {expandedSteps.has(step.id) ? '▼' : '▶'}
              </span>
            </button>

            {/* Step details (when expanded) */}
            {expandedSteps.has(step.id) && (
              <div className="px-4 py-3 bg-slate-800/50 border-t border-slate-700">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-slate-300">
                    Injection point for step {step.id}
                  </p>
                  <InjectMenu
                    threadId={threadId}
                    stepId={step.id}
                    mrfEligible={step.status === 'running'}
                    mctsEligible={step.status === 'running'}
                    size="md"
                    onInjectMRF={(strategy) =>
                      console.log(`MRF ${strategy} injected at step ${step.id}`)
                    }
                    onInjectMCTS={({ budget, exploration }) =>
                      console.log(
                        `MCTS ${budget}/${exploration} injected at step ${step.id}`
                      )
                    }
                  />
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// Example 4: Compact Controls for Dense UI
// ============================================================================

export const CompactThreadRowExample: React.FC<ThreadRowExample1Props> = ({
  threadId,
  name,
  status,
  priority,
}) => {
  return (
    <div className="flex items-center gap-1 px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600">
      {/* Compact status */}
      <span className="w-1.5 h-1.5 rounded-full bg-blue-400 flex-shrink-0" />

      {/* Truncated name */}
      <span className="flex-1 truncate">{name}</span>

      {/* Small priority controls */}
      <PriorityControls
        threadId={threadId}
        priority={priority}
        size="sm"
        orientation="horizontal"
        className="mx-1"
      />

      {/* Small thread controls */}
      <ThreadControls threadId={threadId} status={status} size="sm" />

      {/* Small inject menu */}
      <InjectMenu
        threadId={threadId}
        stepId="0"
        mrfEligible={true}
        mctsEligible={false}
        size="sm"
      />
    </div>
  );
};

// ============================================================================
// Example 5: With Callbacks and State Management
// ============================================================================

interface ThreadWithCallbacksProps {
  threadId: string;
  name: string;
  status: 'running' | 'paused' | 'idle';
  priority: number;
  onPriorityChange?: (newPriority: number) => void;
  onStatusChange?: (newStatus: string) => void;
  onInject?: (type: 'mrf' | 'mcts', value: any) => void;
}

export const ThreadRowWithCallbacksExample: React.FC<
  ThreadWithCallbacksProps
> = ({
  threadId,
  name,
  status,
  priority,
  onPriorityChange,
  onStatusChange,
  onInject,
}) => {
  return (
    <div className="flex items-center gap-2 p-2 bg-slate-700 rounded">
      <span className="flex-1">{name}</span>

      <PriorityControls
        threadId={threadId}
        priority={priority}
        onChange={(newPriority) => {
          onPriorityChange?.(newPriority);
          console.log(`Priority changed to ${newPriority}`);
        }}
      />

      <ThreadControls
        threadId={threadId}
        status={status}
        onAction={(action) => {
          onStatusChange?.(action);
          console.log(`Thread action: ${action}`);
        }}
      />

      <InjectMenu
        threadId={threadId}
        stepId="0"
        mrfEligible={status === 'running'}
        mctsEligible={status === 'running'}
        onInjectMRF={(strategy) => {
          onInject?.('mrf', strategy);
          console.log(`Injected MRF: ${strategy}`);
        }}
        onInjectMCTS={(config) => {
          onInject?.('mcts', config);
          console.log(`Injected MCTS:`, config);
        }}
      />
    </div>
  );
};

// ============================================================================
// Example 6: Grid Layout for Thread Overview
// ============================================================================

interface ThreadGridExampleProps {
  threads: ThreadRowExample1Props[];
}

export const ThreadGridExample: React.FC<ThreadGridExampleProps> = ({
  threads,
}) => {
  return (
    <div className="grid grid-cols-1 gap-2">
      {/* Header row */}
      <div className="grid grid-cols-12 gap-2 px-2 py-1 text-xs font-semibold text-slate-400 uppercase">
        <div className="col-span-4">Thread</div>
        <div className="col-span-2">Priority</div>
        <div className="col-span-2">Status</div>
        <div className="col-span-2">Controls</div>
        <div className="col-span-2">Inject</div>
      </div>

      {/* Data rows */}
      <div className="space-y-1">
        {threads.map((thread) => (
          <div
            key={thread.threadId}
            className="grid grid-cols-12 gap-2 items-center p-2 bg-slate-700 rounded hover:bg-slate-600"
          >
            <div className="col-span-4 truncate font-medium">{thread.name}</div>

            <div className="col-span-2 flex justify-center">
              <PriorityControls
                threadId={thread.threadId}
                priority={thread.priority}
                size="sm"
                orientation="horizontal"
              />
            </div>

            <div className="col-span-2">
              <span
                className={`text-xs px-2 py-1 rounded ${
                  thread.status === 'running'
                    ? 'bg-blue-700 text-blue-200'
                    : thread.status === 'paused'
                      ? 'bg-amber-700 text-amber-200'
                      : 'bg-slate-700 text-slate-300'
                }`}
              >
                {thread.status}
              </span>
            </div>

            <div className="col-span-2 flex justify-center">
              <ThreadControls
                threadId={thread.threadId}
                status={thread.status}
                size="sm"
              />
            </div>

            <div className="col-span-2 flex justify-center">
              <InjectMenu
                threadId={thread.threadId}
                stepId="0"
                mrfEligible={thread.status === 'running'}
                mctsEligible={thread.status === 'running'}
                size="sm"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// Example 7: Responsive Layout (Mobile/Desktop)
// ============================================================================

export const ResponsiveThreadRowExample: React.FC<ThreadRowExample1Props> = ({
  threadId,
  name,
  status,
  priority,
}) => {
  return (
    <div className="bg-slate-700 rounded-lg p-3 space-y-2 md:space-y-0 md:flex md:items-center md:gap-2">
      {/* Mobile: full width; Desktop: flex */}

      {/* Thread info */}
      <div className="flex-1">
        <h3 className="font-semibold text-white">{name}</h3>
        <p className="text-xs text-slate-400 md:hidden capitalize">{status}</p>
      </div>

      {/* Controls - wrap on mobile, row on desktop */}
      <div className="flex items-center justify-between gap-2 md:gap-1">
        {/* Hide priority label on mobile */}
        <div className="hidden md:flex gap-1 items-center">
          <span className="text-xs text-slate-400">P:</span>
          <PriorityControls
            threadId={threadId}
            priority={priority}
            size="sm"
            orientation="horizontal"
          />
        </div>

        {/* Show all controls */}
        <ThreadControls threadId={threadId} status={status} size="sm" />
        <InjectMenu
          threadId={threadId}
          stepId="0"
          mrfEligible={true}
          mctsEligible={true}
          size="sm"
        />
      </div>
    </div>
  );
};

// ============================================================================
// Example 8: Custom Styling
// ============================================================================

export const StyledThreadRowExample: React.FC<ThreadRowExample1Props> = ({
  threadId,
  name,
  status,
  priority,
}) => {
  // Custom styling based on priority
  const priorityBgClass =
    priority >= 75
      ? 'bg-red-900/20 border-red-700'
      : priority >= 50
        ? 'bg-amber-900/20 border-amber-700'
        : 'bg-blue-900/20 border-blue-700';

  return (
    <div
      className={`
        flex items-center gap-3 p-3 rounded-lg border
        transition-all duration-150 hover:shadow-lg
        ${priorityBgClass}
      `}
    >
      {/* Status indicator with custom styling */}
      <div
        className={`
          w-4 h-4 rounded-full flex-shrink-0
          ${
            status === 'running'
              ? 'bg-green-500 animate-pulse'
              : status === 'paused'
                ? 'bg-amber-500'
                : 'bg-slate-500'
          }
        `}
      />

      {/* Thread info */}
      <div className="flex-1 min-w-0">
        <h3 className="font-semibold text-white truncate">{name}</h3>
        <div className="flex gap-2 mt-1 text-xs">
          <span className="text-slate-400">Status:</span>
          <span className="text-blue-300 capitalize">{status}</span>
          <span className="text-slate-400">Priority:</span>
          <span className="text-amber-300 font-bold">{priority}</span>
        </div>
      </div>

      {/* Controls with custom arrangement */}
      <div className="flex items-center gap-1 flex-shrink-0">
        <PriorityControls
          threadId={threadId}
          priority={priority}
          size="md"
          orientation="vertical"
          className="px-1"
        />

        <div className="w-px h-6 bg-slate-600" />

        <ThreadControls threadId={threadId} status={status} size="md" />

        <div className="w-px h-6 bg-slate-600" />

        <InjectMenu
          threadId={threadId}
          stepId="0"
          mrfEligible={status !== 'idle'}
          mctsEligible={status !== 'idle'}
          size="md"
        />
      </div>
    </div>
  );
};

export default {
  ThreadRowExample1,
  ThreadRowHorizontalExample,
  ThreadDetailViewExample,
  CompactThreadRowExample,
  ThreadRowWithCallbacksExample,
  ThreadGridExample,
  ResponsiveThreadRowExample,
  StyledThreadRowExample,
};
