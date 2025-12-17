/**
 * StepList Component
 * Container for a list of task steps within a thread
 * Manages step hierarchy, visual connectors, and interaction state
 */

import React, { useState, useCallback, useMemo } from 'react';
import StepRow from './StepRow';
import { TaskNode } from './types';

interface StepListProps {
  /** Array of task nodes to display */
  steps: TaskNode[];

  /** ID of the parent thread */
  threadId: string;

  /** Root task node (optional, for header display) */
  rootTask?: TaskNode;

  /** Currently hovered step ID */
  hoveredStepId?: string | null;

  /** Currently selected step ID */
  selectedStepId?: string | null;

  /** Callback when hovering over a step */
  onStepHover?: (stepId: string | null) => void;

  /** Callback when clicking a step */
  onStepSelect?: (stepId: string) => void;

  /** Callback when injecting MRF */
  onInjectMRF?: (stepId: string) => void;

  /** Callback when injecting MCTS */
  onInjectMCTS?: (stepId: string) => void;

  /** Whether to show query preview on hover */
  showQueryPreview?: boolean;

  /** Custom className for the container */
  className?: string;
}

/**
 * ConnectorLine - Vertical line connecting steps
 */
interface ConnectorLineProps {
  isLast: boolean;
  depth: number;
}

const ConnectorLine: React.FC<ConnectorLineProps> = ({ isLast, depth }) => {
  const indentPx = depth * 12 + 10; // Align with step icon
  const heightClass = isLast ? 'h-4' : 'h-8';

  return (
    <div
      className={`absolute ${heightClass} w-px bg-slate-700/40 pointer-events-none`}
      style={{
        left: `${indentPx}px`,
        top: '100%',
      }}
    />
  );
};

/**
 * StepListHeader - Optional header showing thread or root task info
 */
const StepListHeader: React.FC<{ rootTask?: TaskNode; threadId: string }> = ({
  rootTask,
  threadId,
}) => {
  if (!rootTask) return null;

  return (
    <div className="px-3 py-2 border-b border-slate-700 bg-slate-800/30">
      <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
        {rootTask.name}
      </h4>
      {rootTask.query && (
        <p className="text-xs text-slate-400 mt-1 truncate">
          {rootTask.query}
        </p>
      )}
    </div>
  );
};

/**
 * StepList Component
 */
export const StepList: React.FC<StepListProps> = ({
  steps,
  threadId,
  rootTask,
  hoveredStepId,
  selectedStepId,
  onStepHover,
  onStepSelect,
  onInjectMRF,
  onInjectMCTS,
  showQueryPreview = true,
  className = '',
}) => {
  const [internalHoveredId, setInternalHoveredId] = useState<string | null>(null);
  const [internalSelectedId, setInternalSelectedId] = useState<string | null>(null);

  // Use provided props or internal state
  const currentHoveredId = hoveredStepId !== undefined ? hoveredStepId : internalHoveredId;
  const currentSelectedId = selectedStepId !== undefined ? selectedStepId : internalSelectedId;

  const handleStepHover = useCallback(
    (stepId: string | null) => {
      if (hoveredStepId === undefined) {
        setInternalHoveredId(stepId);
      }
      onStepHover?.(stepId);
    },
    [hoveredStepId, onStepHover]
  );

  const handleStepSelect = useCallback(
    (stepId: string) => {
      if (selectedStepId === undefined) {
        setInternalSelectedId(stepId);
      }
      onStepSelect?.(stepId);
    },
    [selectedStepId, onStepSelect]
  );

  // Build a map of step IDs to their indices for faster lookups
  const stepIndexMap = useMemo(() => {
    const map: Record<string, number> = {};
    steps.forEach((step, index) => {
      map[step.id] = index;
    });
    return map;
  }, [steps]);

  // Sort steps by depth and execution order while maintaining parent-child relationships
  const sortedSteps = useMemo(() => {
    const sorted = [...steps];
    sorted.sort((a, b) => {
      // Primary: sort by parent-child relationship (maintain tree structure)
      if (a.parentId === b.id) return 1; // a is child of b
      if (b.parentId === a.id) return -1; // b is child of a

      // Secondary: sort by depth (shallower first)
      if (a.depth !== b.depth) return a.depth - b.depth;

      // Tertiary: maintain original order for same-depth siblings
      return stepIndexMap[a.id] - stepIndexMap[b.id];
    });
    return sorted;
  }, [steps, stepIndexMap]);

  // Calculate which steps should show connector lines
  const shouldShowConnector = useCallback(
    (stepId: string): boolean => {
      const stepIndex = sortedSteps.findIndex((s) => s.id === stepId);
      if (stepIndex === -1 || stepIndex === sortedSteps.length - 1) {
        return false;
      }

      const currentStep = sortedSteps[stepIndex];
      const nextStep = sortedSteps[stepIndex + 1];

      // Only show connector if next step is at same or greater depth
      // (indicates continuation of execution flow)
      return nextStep.depth >= currentStep.depth;
    },
    [sortedSteps]
  );

  // Calculate completion stats
  const stats = useMemo(() => {
    const total = steps.length;
    const completed = steps.filter((s) => s.status === 'completed').length;
    const running = steps.filter((s) => s.status === 'running').length;
    const failed = steps.filter((s) => s.status === 'failed').length;

    return { total, completed, running, failed };
  }, [steps]);

  // Empty state
  if (steps.length === 0) {
    return (
      <div className={`bg-slate-800/30 border border-slate-700 rounded overflow-hidden ${className}`}>
        <StepListHeader rootTask={rootTask} threadId={threadId} />
        <div className="px-4 py-8 text-center text-slate-400 text-sm">
          <p>No steps in this thread</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-slate-800/30 border border-slate-700 rounded overflow-hidden ${className}`}>
      {/* Header */}
      {rootTask && <StepListHeader rootTask={rootTask} threadId={threadId} />}

      {/* Progress bar */}
      <div className="h-1 bg-slate-700">
        <div
          className="h-full bg-gradient-to-r from-blue-500 via-cyan-500 to-emerald-500 transition-all duration-300"
          style={{
            width: `${stats.total > 0 ? (stats.completed / stats.total) * 100 : 0}%`,
          }}
        />
      </div>

      {/* Stats bar */}
      <div className="px-3 py-1 border-b border-slate-700/50 bg-slate-800/20 flex items-center gap-4 text-xs text-slate-400">
        <span>
          Steps:{' '}
          <span className="text-slate-200 font-mono">
            {stats.completed}/{stats.total}
          </span>
        </span>

        {stats.running > 0 && (
          <span>
            Running:{' '}
            <span className="text-blue-400 font-mono">{stats.running}</span>
          </span>
        )}

        {stats.failed > 0 && (
          <span>
            Failed:{' '}
            <span className="text-red-400 font-mono">{stats.failed}</span>
          </span>
        )}

        <div className="flex-1" />

        {/* Thread ID (on wide screens) */}
        <span className="hidden lg:inline text-slate-500 font-mono text-xs">
          {threadId.substring(0, 8)}...
        </span>
      </div>

      {/* Steps container with scroll */}
      <div className="overflow-y-auto max-h-96 divide-y divide-slate-700/30">
        {sortedSteps.map((step, index) => (
          <div
            key={step.id}
            className="relative"
          >
            {/* Connector line to next step */}
            {shouldShowConnector(step.id) && (
              <ConnectorLine
                isLast={index === sortedSteps.length - 1}
                depth={step.depth}
              />
            )}

            {/* Step row */}
            <StepRow
              step={step}
              depth={step.depth}
              isHovered={currentHoveredId === step.id}
              isSelected={currentSelectedId === step.id}
              onHover={handleStepHover}
              onClick={handleStepSelect}
              onInjectMRF={onInjectMRF}
              onInjectMCTS={onInjectMCTS}
              showQueryPreview={showQueryPreview}
            />
          </div>
        ))}
      </div>

      {/* Footer with summary */}
      <div className="px-3 py-2 border-t border-slate-700/50 bg-slate-800/20 text-xs text-slate-400 flex items-center justify-between">
        <span>
          {stats.failed > 0
            ? `${stats.failed} failed`
            : stats.running > 0
              ? `${stats.running} running`
              : 'All steps completed'}
        </span>

        {/* MRF/MCTS eligibility info */}
        {steps.some((s) => s.mrfEligible || s.mctsEligible) && (
          <span className="text-slate-500">
            {steps.filter((s) => s.mrfEligible).length > 0 && (
              <span className="text-emerald-400 ml-2">MRF available</span>
            )}
            {steps.filter((s) => s.mctsEligible).length > 0 && (
              <span className="text-cyan-400 ml-2">MCTS available</span>
            )}
          </span>
        )}
      </div>
    </div>
  );
};

StepList.displayName = 'StepList';

export default StepList;
