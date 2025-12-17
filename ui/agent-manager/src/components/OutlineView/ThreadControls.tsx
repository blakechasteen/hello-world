/**
 * ThreadControls Component
 * Pause/Resume/Cancel controls for thread lifecycle management
 * Shows contextual buttons based on thread status
 */

import React, { useCallback, useState } from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';

interface ThreadControlsProps {
  /** Thread ID to control */
  threadId: string;

  /** Current thread status */
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

  /** Size of the controls */
  size?: 'sm' | 'md';

  /** Custom CSS class */
  className?: string;

  /** Show confirmation dialog before cancelling */
  showCancelConfirm?: boolean;

  /** Optional callback when action is performed */
  onAction?: (action: 'pause' | 'resume' | 'cancel' | 'retry') => void;
}

interface ConfirmDialogState {
  isOpen: boolean;
  pendingAction: 'cancel' | null;
}

export const ThreadControls: React.FC<ThreadControlsProps> = ({
  threadId,
  status,
  size = 'md',
  className = '',
  showCancelConfirm = true,
  onAction,
}) => {
  const { pauseThread, resumeThread, cancelThread } = useAgentManagerStore();
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState>({
    isOpen: false,
    pendingAction: null,
  });

  // Compute button size classes
  const buttonSizeClasses = {
    sm: 'w-7 h-7 text-xs p-1',
    md: 'w-8 h-8 text-sm p-1',
  };

  // Compute tooltip size classes
  const tooltipSizeClasses = {
    sm: 'text-xs px-2 py-1',
    md: 'text-sm px-2.5 py-1.5',
  };

  // Handle pause
  const handlePause = useCallback(() => {
    pauseThread(threadId);
    onAction?.('pause');
  }, [threadId, pauseThread, onAction]);

  // Handle resume
  const handleResume = useCallback(() => {
    resumeThread(threadId);
    onAction?.('resume');
  }, [threadId, resumeThread, onAction]);

  // Handle cancel (with optional confirmation)
  const handleCancelClick = useCallback(() => {
    if (showCancelConfirm) {
      setConfirmDialog({ isOpen: true, pendingAction: 'cancel' });
    } else {
      performCancel();
    }
  }, [showCancelConfirm]);

  // Perform the actual cancel
  const performCancel = useCallback(() => {
    cancelThread(threadId);
    setConfirmDialog({ isOpen: false, pendingAction: null });
    onAction?.('cancel');
  }, [threadId, cancelThread, onAction]);

  // Confirm dialog action
  const handleConfirmCancel = useCallback(() => {
    performCancel();
  }, [performCancel]);

  // Close confirmation dialog
  const handleCancelConfirm = useCallback(() => {
    setConfirmDialog({ isOpen: false, pendingAction: null });
  }, []);

  // Determine which buttons to show based on status
  const showPause = status === 'running';
  const showResume = status === 'paused';
  const showCancel = status === 'running' || status === 'paused';
  const showRetry = status === 'failed' || status === 'completed';

  // If no buttons should be shown
  if (!showPause && !showResume && !showCancel && !showRetry) {
    return (
      <div
        className={`flex items-center gap-1 ${className}`}
        role="group"
        aria-label="Thread controls"
      >
        {/* Empty state - show status is terminal (idle or cancelled only here) */}
        <div
          className="text-xs text-slate-500"
          aria-live="polite"
        >
          {status === 'cancelled' && 'Cancelled'}
          {status === 'idle' && 'Idle'}
        </div>
      </div>
    );
  }

  return (
    <>
      <div
        className={`flex items-center gap-1 ${className}`}
        role="group"
        aria-label="Thread controls"
      >
        {/* Pause button */}
        {showPause && (
          <div className="relative group">
            <button
              onClick={handlePause}
              className={`
                ${buttonSizeClasses[size]}
                flex items-center justify-center rounded
                transition-all duration-150 ease-out
                bg-amber-700 text-amber-200 hover:bg-amber-600 hover:text-white
                active:scale-95
                focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-1 focus:ring-offset-slate-800
              `}
              title="Pause this thread"
              aria-label="Pause thread"
            >
              ⏸
            </button>
            <div
              className={`
                absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                bg-slate-900 text-white rounded px-2 py-1 text-xs
                opacity-0 group-hover:opacity-100 transition-opacity
                pointer-events-none whitespace-nowrap
                border border-slate-700
              `}
            >
              Pause thread
            </div>
          </div>
        )}

        {/* Resume button */}
        {showResume && (
          <div className="relative group">
            <button
              onClick={handleResume}
              className={`
                ${buttonSizeClasses[size]}
                flex items-center justify-center rounded
                transition-all duration-150 ease-out
                bg-emerald-700 text-emerald-200 hover:bg-emerald-600 hover:text-white
                active:scale-95
                focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-1 focus:ring-offset-slate-800
              `}
              title="Resume this thread"
              aria-label="Resume thread"
            >
              ▶
            </button>
            <div
              className={`
                absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                bg-slate-900 text-white rounded px-2 py-1 text-xs
                opacity-0 group-hover:opacity-100 transition-opacity
                pointer-events-none whitespace-nowrap
                border border-slate-700
              `}
            >
              Resume thread
            </div>
          </div>
        )}

        {/* Cancel button */}
        {showCancel && (
          <div className="relative group">
            <button
              onClick={handleCancelClick}
              className={`
                ${buttonSizeClasses[size]}
                flex items-center justify-center rounded
                transition-all duration-150 ease-out
                bg-red-700 text-red-200 hover:bg-red-600 hover:text-white
                active:scale-95
                focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1 focus:ring-offset-slate-800
              `}
              title="Cancel this thread"
              aria-label="Cancel thread"
            >
              ✕
            </button>
            <div
              className={`
                absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                bg-slate-900 text-white rounded px-2 py-1 text-xs
                opacity-0 group-hover:opacity-100 transition-opacity
                pointer-events-none whitespace-nowrap
                border border-slate-700
              `}
            >
              Cancel thread
            </div>
          </div>
        )}

        {/* Retry button (for completed/failed) */}
        {showRetry && (
          <div className="relative group">
            <button
              disabled
              className={`
                ${buttonSizeClasses[size]}
                flex items-center justify-center rounded
                transition-all duration-150 ease-out
                bg-slate-700 text-slate-600 cursor-not-allowed opacity-50
              `}
              title="Retry is not yet available"
              aria-label="Retry thread (disabled)"
              aria-disabled="true"
            >
              ↻
            </button>
            <div
              className={`
                absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                bg-slate-900 text-slate-300 rounded px-2 py-1 text-xs
                opacity-0 group-hover:opacity-100 transition-opacity
                pointer-events-none whitespace-nowrap
                border border-slate-700
              `}
            >
              Coming soon
            </div>
          </div>
        )}
      </div>

      {/* Confirmation Dialog */}
      {showCancelConfirm && confirmDialog.isOpen && confirmDialog.pendingAction === 'cancel' && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          role="dialog"
          aria-modal="true"
          aria-labelledby="confirm-dialog-title"
        >
          <div
            className={`
              bg-slate-800 border border-slate-700 rounded-lg
              shadow-xl p-4
              ${tooltipSizeClasses[size]}
              max-w-sm
            `}
          >
            <h3
              id="confirm-dialog-title"
              className="font-semibold text-white mb-2"
            >
              Cancel Thread?
            </h3>
            <p className="text-slate-300 text-sm mb-4">
              Are you sure you want to cancel this thread? This action cannot be undone.
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={handleCancelConfirm}
                className={`
                  px-3 py-1.5 rounded
                  bg-slate-700 text-slate-200
                  hover:bg-slate-600 hover:text-white
                  transition-all duration-150
                  text-sm font-medium
                  focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-1 focus:ring-offset-slate-800
                `}
              >
                No, keep it
              </button>
              <button
                onClick={handleConfirmCancel}
                className={`
                  px-3 py-1.5 rounded
                  bg-red-700 text-red-200
                  hover:bg-red-600 hover:text-white
                  transition-all duration-150
                  text-sm font-medium
                  focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1 focus:ring-offset-slate-800
                `}
              >
                Yes, cancel it
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ThreadControls;
