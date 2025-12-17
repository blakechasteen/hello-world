/**
 * HoloLoom Design System - Thread Item Component
 *
 * Represents a thread or conversation item in the weaving interface.
 * Shows status, confidence, and thread metadata.
 */

import React, { forwardRef } from 'react';
import { cn } from '../utils/cn';

// =============================================================================
// TYPES
// =============================================================================

export type ThreadStatus = 'pending' | 'weaving' | 'complete' | 'error' | 'cached';

export interface ThreadItemProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Thread title or query text */
  title: string;
  /** Thread status */
  status?: ThreadStatus;
  /** Confidence score 0-1 */
  confidence?: number;
  /** Timestamp */
  timestamp?: string | Date;
  /** Is active/selected */
  active?: boolean;
  /** Tool used */
  tool?: string;
  /** Reasoning mode */
  mode?: string;
  /** Latency in ms */
  latencyMs?: number;
  /** Token count */
  tokens?: number;
  /** Click handler */
  onSelect?: () => void;
  /** Delete handler */
  onDelete?: () => void;
}

// =============================================================================
// HELPERS
// =============================================================================

const statusConfig: Record<ThreadStatus, { color: string; icon: string; label: string }> = {
  pending: {
    color: 'text-fg-tertiary',
    icon: '○',
    label: 'Pending',
  },
  weaving: {
    color: 'text-cosmic-aurora',
    icon: '◉',
    label: 'Weaving',
  },
  complete: {
    color: 'text-confidence-high',
    icon: '●',
    label: 'Complete',
  },
  error: {
    color: 'text-safety-danger',
    icon: '✕',
    label: 'Error',
  },
  cached: {
    color: 'text-cosmic-nebula',
    icon: '◆',
    label: 'Cached',
  },
};

function formatTime(timestamp: string | Date): string {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  // Less than 1 minute
  if (diff < 60000) return 'just now';
  // Less than 1 hour
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  // Less than 24 hours
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  // Less than 7 days
  if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
  // Otherwise show date
  return date.toLocaleDateString();
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.75) return 'text-confidence-high';
  if (confidence >= 0.5) return 'text-confidence-medium';
  if (confidence > 0) return 'text-confidence-low';
  return 'text-confidence-unknown';
}

// =============================================================================
// COMPONENT
// =============================================================================

export const ThreadItem = forwardRef<HTMLDivElement, ThreadItemProps>(
  (
    {
      className,
      title,
      status = 'complete',
      confidence,
      timestamp,
      active = false,
      tool,
      mode,
      latencyMs,
      tokens,
      onSelect,
      onDelete,
      ...props
    },
    ref
  ) => {
    const statusInfo = statusConfig[status];

    return (
      <div
        ref={ref}
        className={cn(
          'group relative flex items-start gap-3 p-3 rounded-lg',
          'transition-all duration-fast ease-out',
          'hover:bg-bg-tertiary',
          active && 'bg-bg-tertiary border-l-3 border-cosmic-nebula',
          !active && 'border-l-3 border-transparent',
          onSelect && 'cursor-pointer',
          className
        )}
        onClick={onSelect}
        role={onSelect ? 'button' : undefined}
        tabIndex={onSelect ? 0 : undefined}
        {...props}
      >
        {/* Status indicator */}
        <div className="flex-shrink-0 pt-0.5">
          <span
            className={cn(
              'text-lg leading-none',
              statusInfo.color,
              status === 'weaving' && 'animate-pulse'
            )}
            aria-label={statusInfo.label}
          >
            {statusInfo.icon}
          </span>
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Title */}
          <p className={cn(
            'text-sm font-medium text-fg-primary truncate',
            active && 'text-cosmic-starlight'
          )}>
            {title}
          </p>

          {/* Metadata row */}
          <div className="flex items-center gap-2 mt-1 text-xs text-fg-tertiary">
            {timestamp && (
              <span>{formatTime(timestamp)}</span>
            )}

            {confidence !== undefined && (
              <>
                <span>•</span>
                <span className={cn('font-mono', getConfidenceColor(confidence))}>
                  {Math.round(confidence * 100)}%
                </span>
              </>
            )}

            {tool && (
              <>
                <span>•</span>
                <span className="px-1 py-0.5 rounded bg-bg-elevated text-fg-secondary">
                  {tool}
                </span>
              </>
            )}

            {mode && (
              <>
                <span>•</span>
                <span className="text-cosmic-aurora capitalize">{mode}</span>
              </>
            )}
          </div>

          {/* Performance metrics */}
          {(latencyMs !== undefined || tokens !== undefined) && (
            <div className="flex items-center gap-3 mt-1.5 text-2xs text-fg-tertiary font-mono">
              {latencyMs !== undefined && (
                <span className="flex items-center gap-1">
                  <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                  </svg>
                  {latencyMs.toFixed(0)}ms
                </span>
              )}
              {tokens !== undefined && (
                <span className="flex items-center gap-1">
                  <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
                  </svg>
                  {tokens.toLocaleString()}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Delete button */}
        {onDelete && (
          <button
            className={cn(
              'flex-shrink-0 p-1 rounded opacity-0 group-hover:opacity-100',
              'text-fg-tertiary hover:text-safety-danger hover:bg-safety-danger-bg',
              'transition-all duration-fast'
            )}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            aria-label="Delete thread"
          >
            <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </button>
        )}
      </div>
    );
  }
);

ThreadItem.displayName = 'ThreadItem';
