/**
 * StepRow Component
 * Represents a single task step in the execution outline
 * Shows status, progress, confidence, and action buttons
 */

import React, { useState, useMemo } from 'react';
import { TaskNode, StepStatus } from './types';

interface StepRowProps {
  /** The task node to display */
  step: TaskNode;

  /** Depth for indentation (in levels, 12px per level) */
  depth: number;

  /** Whether this row is currently hovered */
  isHovered?: boolean;

  /** Whether this row is selected */
  isSelected?: boolean;

  /** Callback when hovering */
  onHover?: (stepId: string | null) => void;

  /** Callback when clicking */
  onClick?: (stepId: string) => void;

  /** Callback when injecting MRF */
  onInjectMRF?: (stepId: string) => void;

  /** Callback when injecting MCTS */
  onInjectMCTS?: (stepId: string) => void;

  /** Whether to show detailed query on hover */
  showQueryPreview?: boolean;
}

/**
 * StatusIcon - Renders the status indicator
 */
const StatusIcon: React.FC<{ status: StepStatus }> = ({ status }) => {
  const baseClasses = 'flex-shrink-0 w-5 h-5 flex items-center justify-center';

  switch (status) {
    case 'pending':
      return (
        <div className={`${baseClasses} text-slate-400`}>
          <span className="text-sm">○</span>
        </div>
      );

    case 'running':
      return (
        <div className={`${baseClasses} text-blue-400 animate-spin`}>
          <span className="text-sm">◐</span>
        </div>
      );

    case 'completed':
      return (
        <div className={`${baseClasses} text-emerald-500`}>
          <svg
            className="w-4 h-4"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      );

    case 'failed':
      return (
        <div className={`${baseClasses} text-red-500`}>
          <svg
            className="w-4 h-4"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      );

    case 'skipped':
      return (
        <div className={`${baseClasses} text-slate-500`}>
          <span className="text-sm">—</span>
        </div>
      );
  }
};

/**
 * ProgressBar - Renders inline progress indicator
 */
const ProgressBar: React.FC<{ progress: number }> = ({ progress }) => {
  return (
    <div className="flex-shrink-0 w-12 h-1.5 bg-slate-700 rounded-full overflow-hidden">
      <div
        className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 transition-all duration-300"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
};

/**
 * ConfidenceIndicator - Renders confidence score with color coding
 */
const ConfidenceIndicator: React.FC<{ confidence: number }> = ({ confidence }) => {
  let colorClass = 'text-red-400';
  if (confidence > 0.8) {
    colorClass = 'text-emerald-400';
  } else if (confidence > 0.5) {
    colorClass = 'text-amber-400';
  }

  return (
    <span className={`${colorClass} text-xs font-mono`}>
      {confidence.toFixed(2)}
    </span>
  );
};

/**
 * InjectionButton - Small action button for MRF/MCTS injection
 */
const InjectionButton: React.FC<{
  type: 'mrf' | 'mcts';
  onClick: () => void;
}> = ({ type, onClick }) => {
  const isExpanded = window.innerWidth > 1536; // 2xl breakpoint

  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={`
        px-2 py-0.5 rounded text-xs font-medium transition-colors
        ${
          type === 'mrf'
            ? 'bg-emerald-900/30 text-emerald-300 hover:bg-emerald-800/50'
            : 'bg-cyan-900/30 text-cyan-300 hover:bg-cyan-800/50'
        }
        focus:outline-none focus:ring-2 focus:ring-offset-0
        ${type === 'mrf' ? 'focus:ring-emerald-500' : 'focus:ring-cyan-500'}
      `}
      title={type === 'mrf' ? 'Inject MRF enhancement' : 'Inject MCTS exploration'}
    >
      {isExpanded ? (type === 'mrf' ? 'MRF' : 'MCTS') : type.toUpperCase()}
    </button>
  );
};

/**
 * QueryPreview - Tooltip showing full query text
 */
const QueryPreview: React.FC<{ query: string }> = ({ query }) => {
  return (
    <div className="absolute bottom-full left-0 mb-1 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-slate-100 whitespace-nowrap max-w-xs truncate pointer-events-none z-10">
      {query}
    </div>
  );
};

/**
 * StepRow Component
 */
export const StepRow: React.FC<StepRowProps> = ({
  step,
  depth,
  isHovered = false,
  isSelected = false,
  onHover,
  onClick,
  onInjectMRF,
  onInjectMCTS,
  showQueryPreview = true,
}) => {
  const [showPreview, setShowPreview] = useState(false);
  const indentPx = useMemo(() => depth * 12, [depth]);
  const isCompleted = step.status === 'completed';
  const isRunning = step.status === 'running';

  return (
    <div
      onClick={() => onClick?.(step.id)}
      onMouseEnter={() => {
        onHover?.(step.id);
        setShowPreview(true);
      }}
      onMouseLeave={() => {
        onHover?.(null);
        setShowPreview(false);
      }}
      className={`
        group relative h-8 flex items-center gap-2 px-2 transition-colors
        ${
          isSelected
            ? 'bg-slate-700 border-l-2 border-blue-500'
            : isHovered
              ? 'bg-slate-700/50'
              : 'bg-slate-800/50'
        }
        ${isCompleted ? 'opacity-75' : ''}
        ${isRunning ? 'animate-pulse' : ''}
        hover:bg-slate-700 cursor-pointer
      `}
      style={{
        paddingLeft: `${indentPx + 8}px`,
      }}
    >
      {/* Status Icon */}
      <StatusIcon status={step.status} />

      {/* Step Name */}
      <div className="flex-1 min-w-0">
        <div className="relative">
          <span className="text-sm text-slate-100 font-medium truncate block">
            {step.stepType === 'query' && '🔍 '}
            {step.stepType === 'research' && '📚 '}
            {step.stepType === 'verify' && '✓ '}
            {step.stepType === 'synthesize' && '🔀 '}
            {step.stepType === 'execute' && '⚙️ '}
            {step.name}
          </span>

          {/* Query Preview Tooltip */}
          {showPreview && showQueryPreview && step.query && (
            <QueryPreview query={step.query} />
          )}

          {/* Injection Badge */}
          {step.injectionApplied && (
            <span className="absolute right-0 top-0 text-xs px-1 py-0.5 rounded bg-purple-900/40 text-purple-300 whitespace-nowrap">
              {step.injectionApplied}
            </span>
          )}
        </div>
      </div>

      {/* Progress Bar (visible only if in progress) */}
      {step.status === 'running' && (
        <ProgressBar progress={step.progressPct} />
      )}

      {/* Confidence Score */}
      <div className="flex-shrink-0">
        <ConfidenceIndicator confidence={step.confidence} />
      </div>

      {/* Token Usage (on wide screens) */}
      <div className="hidden lg:block flex-shrink-0">
        <span className="text-xs text-slate-400">
          {step.tokensUsed > 0 && `${step.tokensUsed.toLocaleString()}t`}
        </span>
      </div>

      {/* Elapsed Time (on wide screens) */}
      <div className="hidden sm:block flex-shrink-0">
        <span className="text-xs text-slate-400">
          {step.elapsedTimeMs > 0 &&
            `${(step.elapsedTimeMs / 1000).toFixed(1)}s`}
        </span>
      </div>

      {/* Injection Buttons */}
      <div className="flex-shrink-0 gap-1 hidden group-hover:flex">
        {step.mrfEligible && !step.injectionApplied && (
          <InjectionButton
            type="mrf"
            onClick={() => onInjectMRF?.(step.id)}
          />
        )}
        {step.mctsEligible && !step.injectionApplied && (
          <InjectionButton
            type="mcts"
            onClick={() => onInjectMCTS?.(step.id)}
          />
        )}
      </div>

      {/* Dependency Indicators (visual feedback for blocks/dependsOn) */}
      {(step.dependsOn.length > 0 || step.blocks.length > 0) && (
        <div className="flex-shrink-0 hidden xl:flex gap-0.5">
          {step.dependsOn.length > 0 && (
            <div
              className="w-1.5 h-1.5 rounded-full bg-amber-500"
              title={`Depends on ${step.dependsOn.length} task(s)`}
            />
          )}
          {step.blocks.length > 0 && (
            <div
              className="w-1.5 h-1.5 rounded-full bg-red-500"
              title={`Blocks ${step.blocks.length} task(s)`}
            />
          )}
        </div>
      )}

      {/* Subtle border on bottom for visual separation */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-slate-700/30 pointer-events-none" />
    </div>
  );
};

StepRow.displayName = 'StepRow';

export default StepRow;
