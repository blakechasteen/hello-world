/**
 * HoloLoom Design System - Memory Node Component
 *
 * Visual representation of a memory in the knowledge graph.
 * Shows activation state, heat, and relevance.
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type MemoryState = 'active' | 'recent' | 'dormant' | 'hot';
export type MemoryNodeSize = 'sm' | 'md' | 'lg';

export interface MemoryNodeProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Memory state */
  state?: MemoryState;
  /** Size preset */
  size?: MemoryNodeSize;
  /** Memory content preview */
  content?: string;
  /** Activation level 0-1 */
  activation?: number;
  /** Heat score 0-1 */
  heat?: number;
  /** Relevance score 0-1 */
  relevance?: number;
  /** Node ID */
  nodeId?: string;
  /** Edge count */
  edgeCount?: number;
  /** Is selected */
  selected?: boolean;
  /** Is highlighted (part of path) */
  highlighted?: boolean;
  /** Show metrics */
  showMetrics?: boolean;
  /** Click handler */
  onNodeClick?: () => void;
}

// =============================================================================
// STYLES
// =============================================================================

const stateStyles: Record<MemoryState, { bg: string; border: string; glow: string }> = {
  active: {
    bg: 'bg-memory-active-bg',
    border: 'border-memory-active',
    glow: 'shadow-[0_0_12px_rgba(99,102,241,0.4)]',
  },
  recent: {
    bg: 'bg-memory-recent-bg',
    border: 'border-memory-recent',
    glow: 'shadow-[0_0_8px_rgba(34,211,238,0.3)]',
  },
  dormant: {
    bg: 'bg-memory-dormant-bg',
    border: 'border-memory-dormant',
    glow: '',
  },
  hot: {
    bg: 'bg-memory-hot-bg',
    border: 'border-memory-hot',
    glow: 'shadow-[0_0_16px_rgba(251,191,36,0.5)]',
  },
};

const sizeStyles: Record<MemoryNodeSize, { container: string; text: string; metrics: string }> = {
  sm: {
    container: 'min-w-[120px] p-2',
    text: 'text-xs',
    metrics: 'text-2xs',
  },
  md: {
    container: 'min-w-[160px] p-3',
    text: 'text-sm',
    metrics: 'text-xs',
  },
  lg: {
    container: 'min-w-[200px] p-4',
    text: 'text-base',
    metrics: 'text-sm',
  },
};

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

function MetricBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-fg-tertiary w-12 text-right">{label}</span>
      <div className="flex-1 h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-300', color)}
          style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
        />
      </div>
      <span className="text-fg-secondary font-mono w-8">
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

// =============================================================================
// COMPONENT
// =============================================================================

export const MemoryNode = forwardRef<HTMLDivElement, MemoryNodeProps>(
  (
    {
      className,
      state = 'dormant',
      size = 'md',
      content,
      activation = 0,
      heat = 0,
      relevance = 0,
      nodeId,
      edgeCount,
      selected = false,
      highlighted = false,
      showMetrics = false,
      onNodeClick,
      ...props
    },
    ref
  ) => {
    const styles = stateStyles[state];
    const sizes = sizeStyles[size];

    // Determine visual state based on metrics if not explicitly set
    const effectiveState = heat > 0.7 ? 'hot' : activation > 0.5 ? 'active' : state;
    const effectiveStyles = stateStyles[effectiveState];

    return (
      <div
        ref={ref}
        className={cn(
          'rounded-lg border-2 transition-all duration-200',
          sizes.container,
          effectiveStyles.bg,
          effectiveStyles.border,
          effectiveStyles.glow,
          selected && 'ring-2 ring-cosmic-nebula ring-offset-2 ring-offset-bg-primary',
          highlighted && 'border-cosmic-aurora',
          onNodeClick && 'cursor-pointer hover:scale-105',
          className
        )}
        onClick={onNodeClick}
        role={onNodeClick ? 'button' : undefined}
        tabIndex={onNodeClick ? 0 : undefined}
        {...props}
      >
        {/* Header */}
        <div className="flex items-center justify-between gap-2 mb-2">
          {nodeId && (
            <span className={cn('font-mono text-fg-tertiary truncate', sizes.metrics)}>
              {nodeId}
            </span>
          )}
          {edgeCount !== undefined && (
            <span className={cn(
              'flex items-center gap-1 text-fg-tertiary',
              sizes.metrics
            )}>
              <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                <path d="M7 3a1 1 0 000 2h6a1 1 0 100-2H7zM4 7a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1zM2 11a2 2 0 012-2h12a2 2 0 012 2v4a2 2 0 01-2 2H4a2 2 0 01-2-2v-4z" />
              </svg>
              {edgeCount}
            </span>
          )}
        </div>

        {/* Content preview */}
        {content && (
          <p className={cn(
            'text-fg-primary line-clamp-2 mb-2',
            sizes.text
          )}>
            {content}
          </p>
        )}

        {/* State indicator */}
        <div className="flex items-center gap-2 mb-2">
          <span className={cn(
            'inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-medium capitalize',
            effectiveStyles.bg,
            'text-fg-primary'
          )}>
            <span className={cn(
              'w-1.5 h-1.5 rounded-full',
              effectiveState === 'hot' && 'bg-memory-hot animate-pulse',
              effectiveState === 'active' && 'bg-memory-active',
              effectiveState === 'recent' && 'bg-memory-recent',
              effectiveState === 'dormant' && 'bg-memory-dormant'
            )} />
            {effectiveState}
          </span>
        </div>

        {/* Metrics */}
        {showMetrics && (
          <div className={cn('space-y-1.5 pt-2 border-t border-border-secondary', sizes.metrics)}>
            <MetricBar label="Act" value={activation} color="bg-memory-active" />
            <MetricBar label="Heat" value={heat} color="bg-memory-hot" />
            <MetricBar label="Rel" value={relevance} color="bg-cosmic-nebula" />
          </div>
        )}
      </div>
    );
  }
);

MemoryNode.displayName = 'MemoryNode';
