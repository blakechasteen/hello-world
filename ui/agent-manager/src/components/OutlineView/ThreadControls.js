import { jsxs as _jsxs, jsx as _jsx, Fragment as _Fragment } from "react/jsx-runtime";
/**
 * ThreadControls Component
 * Pause/Resume/Cancel controls for thread lifecycle management
 * Shows contextual buttons based on thread status
 */
import { useCallback, useState } from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';
export const ThreadControls = ({ threadId, status, size = 'md', className = '', showCancelConfirm = true, onAction, }) => {
    const { pauseThread, resumeThread, cancelThread } = useAgentManagerStore();
    const [confirmDialog, setConfirmDialog] = useState({
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
        }
        else {
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
        return (_jsx("div", { className: `flex items-center gap-1 ${className}`, role: "group", "aria-label": "Thread controls", children: _jsxs("div", { className: "text-xs text-slate-500", "aria-live": "polite", children: [status === 'cancelled' && 'Cancelled', status === 'idle' && 'Idle'] }) }));
    }
    return (_jsxs(_Fragment, { children: [_jsxs("div", { className: `flex items-center gap-1 ${className}`, role: "group", "aria-label": "Thread controls", children: [showPause && (_jsxs("div", { className: "relative group", children: [_jsx("button", { onClick: handlePause, className: `
                ${buttonSizeClasses[size]}
                flex items-center justify-center rounded
                transition-all duration-150 ease-out
                bg-amber-700 text-amber-200 hover:bg-amber-600 hover:text-white
                active:scale-95
                focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-1 focus:ring-offset-slate-800
              `, title: "Pause this thread", "aria-label": "Pause thread", children: "\u23F8" }), _jsx("div", { className: `
                absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                bg-slate-900 text-white rounded px-2 py-1 text-xs
                opacity-0 group-hover:opacity-100 transition-opacity
                pointer-events-none whitespace-nowrap
                border border-slate-700
              `, children: "Pause thread" })] })), showResume && (_jsxs("div", { className: "relative group", children: [_jsx("button", { onClick: handleResume, className: `
                ${buttonSizeClasses[size]}
                flex items-center justify-center rounded
                transition-all duration-150 ease-out
                bg-emerald-700 text-emerald-200 hover:bg-emerald-600 hover:text-white
                active:scale-95
                focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-1 focus:ring-offset-slate-800
              `, title: "Resume this thread", "aria-label": "Resume thread", children: "\u25B6" }), _jsx("div", { className: `
                absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                bg-slate-900 text-white rounded px-2 py-1 text-xs
                opacity-0 group-hover:opacity-100 transition-opacity
                pointer-events-none whitespace-nowrap
                border border-slate-700
              `, children: "Resume thread" })] })), showCancel && (_jsxs("div", { className: "relative group", children: [_jsx("button", { onClick: handleCancelClick, className: `
                ${buttonSizeClasses[size]}
                flex items-center justify-center rounded
                transition-all duration-150 ease-out
                bg-red-700 text-red-200 hover:bg-red-600 hover:text-white
                active:scale-95
                focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1 focus:ring-offset-slate-800
              `, title: "Cancel this thread", "aria-label": "Cancel thread", children: "\u2715" }), _jsx("div", { className: `
                absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                bg-slate-900 text-white rounded px-2 py-1 text-xs
                opacity-0 group-hover:opacity-100 transition-opacity
                pointer-events-none whitespace-nowrap
                border border-slate-700
              `, children: "Cancel thread" })] })), showRetry && (_jsxs("div", { className: "relative group", children: [_jsx("button", { disabled: true, className: `
                ${buttonSizeClasses[size]}
                flex items-center justify-center rounded
                transition-all duration-150 ease-out
                bg-slate-700 text-slate-600 cursor-not-allowed opacity-50
              `, title: "Retry is not yet available", "aria-label": "Retry thread (disabled)", "aria-disabled": "true", children: "\u21BB" }), _jsx("div", { className: `
                absolute bottom-full left-1/2 -translate-x-1/2 mb-2
                bg-slate-900 text-slate-300 rounded px-2 py-1 text-xs
                opacity-0 group-hover:opacity-100 transition-opacity
                pointer-events-none whitespace-nowrap
                border border-slate-700
              `, children: "Coming soon" })] }))] }), showCancelConfirm && confirmDialog.isOpen && confirmDialog.pendingAction === 'cancel' && (_jsx("div", { className: "fixed inset-0 bg-black/50 flex items-center justify-center z-50", role: "dialog", "aria-modal": "true", "aria-labelledby": "confirm-dialog-title", children: _jsxs("div", { className: `
              bg-slate-800 border border-slate-700 rounded-lg
              shadow-xl p-4
              ${tooltipSizeClasses[size]}
              max-w-sm
            `, children: [_jsx("h3", { id: "confirm-dialog-title", className: "font-semibold text-white mb-2", children: "Cancel Thread?" }), _jsx("p", { className: "text-slate-300 text-sm mb-4", children: "Are you sure you want to cancel this thread? This action cannot be undone." }), _jsxs("div", { className: "flex gap-2 justify-end", children: [_jsx("button", { onClick: handleCancelConfirm, className: `
                  px-3 py-1.5 rounded
                  bg-slate-700 text-slate-200
                  hover:bg-slate-600 hover:text-white
                  transition-all duration-150
                  text-sm font-medium
                  focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-1 focus:ring-offset-slate-800
                `, children: "No, keep it" }), _jsx("button", { onClick: handleConfirmCancel, className: `
                  px-3 py-1.5 rounded
                  bg-red-700 text-red-200
                  hover:bg-red-600 hover:text-white
                  transition-all duration-150
                  text-sm font-medium
                  focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-1 focus:ring-offset-slate-800
                `, children: "Yes, cancel it" })] })] }) }))] }));
};
export default ThreadControls;
//# sourceMappingURL=ThreadControls.js.map