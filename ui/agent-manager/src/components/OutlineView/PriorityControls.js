import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * PriorityControls Component
 * Upvote/downvote controls for thread priority management
 * Supports both vertical and horizontal layouts with keyboard accessibility
 */
import { useCallback } from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';
export const PriorityControls = ({ threadId, priority, size = 'md', orientation = 'vertical', className = '', onChange, }) => {
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
    const handleKeyDown = useCallback((event) => {
        if (event.key === 'ArrowUp') {
            event.preventDefault();
            handleUpvote();
        }
        else if (event.key === 'ArrowDown') {
            event.preventDefault();
            handleDownvote();
        }
    }, [handleUpvote, handleDownvote]);
    // Compute priority color based on value
    const getPriorityColor = () => {
        if (priority >= 75)
            return 'text-red-400';
        if (priority >= 50)
            return 'text-amber-400';
        if (priority >= 25)
            return 'text-blue-400';
        return 'text-slate-400';
    };
    // Vertical layout
    if (orientation === 'vertical') {
        return (_jsxs("div", { className: `flex flex-col items-center gap-1 ${className}`, onKeyDown: handleKeyDown, role: "group", "aria-label": "Priority controls", children: [_jsx("button", { onClick: handleUpvote, disabled: isUpvoteDisabled, className: `
            ${buttonSizeClasses[size]}
            flex items-center justify-center rounded
            transition-all duration-150 ease-out
            font-bold
            ${isUpvoteDisabled
                        ? 'bg-slate-700 text-slate-600 cursor-not-allowed opacity-50'
                        : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800'}
          `, title: "Increase priority (\u2191)", "aria-label": "Increase priority", "aria-disabled": isUpvoteDisabled, children: "\u25B2" }), _jsx("div", { className: `
            text-center
            ${prioritySizeClasses[size]}
            ${getPriorityColor()}
            tabindex="0"
            aria-live="polite"
            aria-atomic="true"
          `, children: priority }), _jsx("button", { onClick: handleDownvote, disabled: isDownvoteDisabled, className: `
            ${buttonSizeClasses[size]}
            flex items-center justify-center rounded
            transition-all duration-150 ease-out
            font-bold
            ${isDownvoteDisabled
                        ? 'bg-slate-700 text-slate-600 cursor-not-allowed opacity-50'
                        : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800'}
          `, title: "Decrease priority (\u2193)", "aria-label": "Decrease priority", "aria-disabled": isDownvoteDisabled, children: "\u25BC" })] }));
    }
    // Horizontal layout
    return (_jsxs("div", { className: `flex items-center gap-2 ${className}`, onKeyDown: handleKeyDown, role: "group", "aria-label": "Priority controls", children: [_jsx("button", { onClick: handleDownvote, disabled: isDownvoteDisabled, className: `
          ${buttonSizeClasses[size]}
          flex items-center justify-center rounded
          transition-all duration-150 ease-out
          font-bold
          ${isDownvoteDisabled
                    ? 'bg-slate-700 text-slate-600 cursor-not-allowed opacity-50'
                    : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800'}
        `, title: "Decrease priority (\u2190)", "aria-label": "Decrease priority", "aria-disabled": isDownvoteDisabled, children: "\u25BC" }), _jsx("div", { className: `
          text-center
          ${prioritySizeClasses[size]}
          ${getPriorityColor()}
          tabindex="0"
          aria-live="polite"
          aria-atomic="true"
        `, children: priority }), _jsx("button", { onClick: handleUpvote, disabled: isUpvoteDisabled, className: `
          ${buttonSizeClasses[size]}
          flex items-center justify-center rounded
          transition-all duration-150 ease-out
          font-bold
          ${isUpvoteDisabled
                    ? 'bg-slate-700 text-slate-600 cursor-not-allowed opacity-50'
                    : 'bg-slate-700 text-slate-200 hover:bg-slate-600 hover:text-white active:scale-95 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-slate-800'}
        `, title: "Increase priority (\u2192)", "aria-label": "Increase priority", "aria-disabled": isUpvoteDisabled, children: "\u25B2" })] }));
};
export default PriorityControls;
//# sourceMappingURL=PriorityControls.js.map