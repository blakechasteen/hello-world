/**
 * PriorityControls Component
 * Upvote/downvote controls for thread priority management
 * Supports both vertical and horizontal layouts with keyboard accessibility
 */

import React, { useCallback } from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';

interface PriorityControlsProps {
  /** Thread ID to control */
  threadId: string;

  /** Current priority value (0-100) */
  priority: number;

  /** Size of the controls */
  size?: 'sm' | 'md';

  /** Layout orientation */
  orientation?: 'horizontal' | 'vertical';

  /** Custom CSS class */
  className?: string;

  /** Optional callback when priority changes */
  onChange?: (newPriority: number) => void;
}

export const PriorityControls: React.FC<PriorityControlsProps> = ({
  threadId,
  priority,
  size = 'md',
  orientation = 'vertical',
  className = '',
  onChange,
}) => {
  const { upvoteThread, downvoteThread } = useAgentManagerStore();

  // Compute button size classes
  const buttonSizeClasses = {
    sm: 'w-7 h-7 text-xs',
    md: 'w-8 h-8 text-sm',
  };

  // Compute priority display size classes
  const prioritySizeClasses = {
    sm: 'text-xs font-semibold min-w-[24px]',
    md: 'text-sm font-semibold min-w-[28px]',
  };

  // Check if buttons should be disabled
  const isUpvoteDisabled = priority >= 100;
  const isDownvoteDisabled = priority <= 0;

  // Handle upvote
  const handleUpvote = useCallback(() => {
    if (!isUpvoteDisabled) {
      upvoteThread(threadId);
      onChange?.(Math.min(100, priority + 1));
    }
  }, [threadId, priority, isUpvoteDisabled, upvoteThread, onChange]);

  // Handle downvote
  const handleDownvote = useCallback(() => {
    if (!isDownvoteDisabled) {
      downvoteThread(threadId);
      onChange?.(Math.max(0, priority - 1));
    }
  }, [threadId, priority, isDownvoteDisabled, downvoteThread, onChange]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'ArrowUp') {
        event.preventDefault();
        handleUpvote();
      } else if (event.key === 'ArrowDown') {
        event.preventDefault();
        handleDownvote();
      }
    },
    [handleUpvote, handleDownvote]
  );

  // Compute priority color based on value
  const getPriorityColor = () => {
    if (priority >= 75) return 'text-red-400';
    if (priority >= 50) return 'text-amber-400';
    if (priority >= 25) return 'text-blue-400';
    return 'text-slate-400';
  };

  // Vertical layout
  if (orientation === 'vertical') {
    return (
      <div
        className={`flex flex-col items-center gap-1 ${className}`}
        onKeyDown={handleKeyDown}
        role="group"
        aria-label="Priority controls"
      >
        {/* Upvote button */}
        <button
          onClick={handleUpvote}
          disabled={isUpvoteDisabled}
          className={`
            ${buttonSizeClasses[size]}
            flex items-center justify-center rounded
            transition-all duration-150 ease-out
            font-bold
            ${
              isUpvoteDisabled
                ? 'bg-slate-700 text-slate-600 cursor-not-allowed opacity-50'
                : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800'
            }
          `}
          title="Increase priority (↑)"
          aria-label="Increase priority"
          aria-disabled={isUpvoteDisabled}
        >
          ▲
        </button>

        {/* Priority display */}
        <div
          className={`
            text-center
            ${prioritySizeClasses[size]}
            ${getPriorityColor()}
            tabindex="0"
            aria-live="polite"
            aria-atomic="true"
          `}
        >
          {priority}
        </div>

        {/* Downvote button */}
        <button
          onClick={handleDownvote}
          disabled={isDownvoteDisabled}
          className={`
            ${buttonSizeClasses[size]}
            flex items-center justify-center rounded
            transition-all duration-150 ease-out
            font-bold
            ${
              isDownvoteDisabled
                ? 'bg-slate-700 text-slate-600 cursor-not-allowed opacity-50'
                : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800'
            }
          `}
          title="Decrease priority (↓)"
          aria-label="Decrease priority"
          aria-disabled={isDownvoteDisabled}
        >
          ▼
        </button>
      </div>
    );
  }

  // Horizontal layout
  return (
    <div
      className={`flex items-center gap-2 ${className}`}
      onKeyDown={handleKeyDown}
      role="group"
      aria-label="Priority controls"
    >
      {/* Downvote button */}
      <button
        onClick={handleDownvote}
        disabled={isDownvoteDisabled}
        className={`
          ${buttonSizeClasses[size]}
          flex items-center justify-center rounded
          transition-all duration-150 ease-out
          font-bold
          ${
            isDownvoteDisabled
              ? 'bg-slate-700 text-slate-600 cursor-not-allowed opacity-50'
              : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800'
          }
        `}
        title="Decrease priority (←)"
        aria-label="Decrease priority"
        aria-disabled={isDownvoteDisabled}
      >
        ▼
      </button>

      {/* Priority display */}
      <div
        className={`
          text-center
          ${prioritySizeClasses[size]}
          ${getPriorityColor()}
          tabindex="0"
          aria-live="polite"
          aria-atomic="true"
        `}
      >
        {priority}
      </div>

      {/* Upvote button */}
      <button
        onClick={handleUpvote}
        disabled={isUpvoteDisabled}
        className={`
          ${buttonSizeClasses[size]}
          flex items-center justify-center rounded
          transition-all duration-150 ease-out
          font-bold
          ${
            isUpvoteDisabled
              ? 'bg-slate-700 text-slate-600 cursor-not-allowed opacity-50'
              : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800'
          }
        `}
        title="Increase priority (→)"
        aria-label="Increase priority"
        aria-disabled={isUpvoteDisabled}
      >
        ▲
      </button>
    </div>
  );
};

export default PriorityControls;
