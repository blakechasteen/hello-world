import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * Example Usage of Control Components
 * Shows common patterns for using PriorityControls, ThreadControls, and InjectMenu
 */
import { useState } from 'react';
import { PriorityControls, ThreadControls, InjectMenu } from './index';
export const ThreadRowExample1 = ({ threadId, name, status, priority, }) => {
    return (_jsxs("div", { className: "flex items-center gap-2 p-2 bg-slate-700 rounded-lg hover:bg-slate-600 transition-colors", children: [_jsx("div", { className: "w-3 h-3 rounded-full bg-blue-400" }), _jsx("span", { className: "flex-1 font-medium", children: name }), _jsx(PriorityControls, { threadId: threadId, priority: priority, size: "md", orientation: "vertical" }), _jsx(ThreadControls, { threadId: threadId, status: status, size: "md", showCancelConfirm: true }), _jsx(InjectMenu, { threadId: threadId, stepId: "0", mrfEligible: status === 'running' || status === 'paused', mctsEligible: status === 'running' || status === 'paused', size: "md" })] }));
};
// ============================================================================
// Example 2: Horizontal Layout for Thread List
// ============================================================================
export const ThreadRowHorizontalExample = ({ threadId, name, status, priority, }) => {
    return (_jsxs("div", { className: "flex items-center justify-between p-3 bg-slate-800 rounded-lg mb-2 border border-slate-700", children: [_jsxs("div", { className: "flex-1", children: [_jsx("h3", { className: "font-semibold text-white", children: name }), _jsxs("p", { className: "text-xs text-slate-400 mt-1", children: ["Status: ", _jsx("span", { className: "capitalize", children: status })] })] }), _jsxs("div", { className: "flex items-center gap-3", children: [_jsx(PriorityControls, { threadId: threadId, priority: priority, size: "md", orientation: "horizontal" }), _jsx(ThreadControls, { threadId: threadId, status: status, size: "md" }), _jsx(InjectMenu, { threadId: threadId, stepId: "0", mrfEligible: true, mctsEligible: true, size: "md" })] })] }));
};
export const ThreadDetailViewExample = ({ threadId, name, status, priority, steps, }) => {
    const [expandedSteps, setExpandedSteps] = useState(new Set());
    const toggleStepExpanded = (stepId) => {
        const newExpanded = new Set(expandedSteps);
        if (newExpanded.has(stepId)) {
            newExpanded.delete(stepId);
        }
        else {
            newExpanded.add(stepId);
        }
        setExpandedSteps(newExpanded);
    };
    return (_jsxs("div", { className: "bg-slate-900 rounded-lg border border-slate-700 overflow-hidden", children: [_jsxs("div", { className: "p-4 bg-slate-800 border-b border-slate-700", children: [_jsxs("div", { className: "flex items-center justify-between mb-2", children: [_jsx("h2", { className: "text-lg font-bold text-white", children: name }), _jsxs("div", { className: "flex gap-2", children: [_jsx(PriorityControls, { threadId: threadId, priority: priority, size: "md", orientation: "horizontal" }), _jsx(ThreadControls, { threadId: threadId, status: status, size: "md", showCancelConfirm: true })] })] }), _jsxs("p", { className: "text-sm text-slate-400", children: ["Status: ", _jsx("span", { className: "capitalize font-medium", children: status })] })] }), _jsx("div", { className: "divide-y divide-slate-700", children: steps.map((step) => (_jsxs("div", { children: [_jsxs("button", { onClick: () => toggleStepExpanded(step.id), className: "w-full px-4 py-3 hover:bg-slate-800 transition-colors text-left flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: `text-sm ${step.status === 'completed'
                                                ? 'text-emerald-400'
                                                : 'text-slate-400'}`, children: step.status === 'completed' ? '✓' : '○' }), _jsx("span", { className: "font-medium text-white", children: step.name })] }), _jsx("span", { className: "text-slate-400", children: expandedSteps.has(step.id) ? '▼' : '▶' })] }), expandedSteps.has(step.id) && (_jsx("div", { className: "px-4 py-3 bg-slate-800/50 border-t border-slate-700", children: _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("p", { className: "text-sm text-slate-300", children: ["Injection point for step ", step.id] }), _jsx(InjectMenu, { threadId: threadId, stepId: step.id, mrfEligible: step.status === 'running', mctsEligible: step.status === 'running', size: "md", onInjectMRF: (strategy) => console.log(`MRF ${strategy} injected at step ${step.id}`), onInjectMCTS: ({ budget, exploration }) => console.log(`MCTS ${budget}/${exploration} injected at step ${step.id}`) })] }) }))] }, step.id))) })] }));
};
// ============================================================================
// Example 4: Compact Controls for Dense UI
// ============================================================================
export const CompactThreadRowExample = ({ threadId, name, status, priority, }) => {
    return (_jsxs("div", { className: "flex items-center gap-1 px-2 py-1 text-xs bg-slate-700 rounded hover:bg-slate-600", children: [_jsx("span", { className: "w-1.5 h-1.5 rounded-full bg-blue-400 flex-shrink-0" }), _jsx("span", { className: "flex-1 truncate", children: name }), _jsx(PriorityControls, { threadId: threadId, priority: priority, size: "sm", orientation: "horizontal", className: "mx-1" }), _jsx(ThreadControls, { threadId: threadId, status: status, size: "sm" }), _jsx(InjectMenu, { threadId: threadId, stepId: "0", mrfEligible: true, mctsEligible: false, size: "sm" })] }));
};
export const ThreadRowWithCallbacksExample = ({ threadId, name, status, priority, onPriorityChange, onStatusChange, onInject, }) => {
    return (_jsxs("div", { className: "flex items-center gap-2 p-2 bg-slate-700 rounded", children: [_jsx("span", { className: "flex-1", children: name }), _jsx(PriorityControls, { threadId: threadId, priority: priority, onChange: (newPriority) => {
                    onPriorityChange?.(newPriority);
                    console.log(`Priority changed to ${newPriority}`);
                } }), _jsx(ThreadControls, { threadId: threadId, status: status, onAction: (action) => {
                    onStatusChange?.(action);
                    console.log(`Thread action: ${action}`);
                } }), _jsx(InjectMenu, { threadId: threadId, stepId: "0", mrfEligible: status === 'running', mctsEligible: status === 'running', onInjectMRF: (strategy) => {
                    onInject?.('mrf', strategy);
                    console.log(`Injected MRF: ${strategy}`);
                }, onInjectMCTS: (config) => {
                    onInject?.('mcts', config);
                    console.log(`Injected MCTS:`, config);
                } })] }));
};
export const ThreadGridExample = ({ threads, }) => {
    return (_jsxs("div", { className: "grid grid-cols-1 gap-2", children: [_jsxs("div", { className: "grid grid-cols-12 gap-2 px-2 py-1 text-xs font-semibold text-slate-400 uppercase", children: [_jsx("div", { className: "col-span-4", children: "Thread" }), _jsx("div", { className: "col-span-2", children: "Priority" }), _jsx("div", { className: "col-span-2", children: "Status" }), _jsx("div", { className: "col-span-2", children: "Controls" }), _jsx("div", { className: "col-span-2", children: "Inject" })] }), _jsx("div", { className: "space-y-1", children: threads.map((thread) => (_jsxs("div", { className: "grid grid-cols-12 gap-2 items-center p-2 bg-slate-700 rounded hover:bg-slate-600", children: [_jsx("div", { className: "col-span-4 truncate font-medium", children: thread.name }), _jsx("div", { className: "col-span-2 flex justify-center", children: _jsx(PriorityControls, { threadId: thread.threadId, priority: thread.priority, size: "sm", orientation: "horizontal" }) }), _jsx("div", { className: "col-span-2", children: _jsx("span", { className: `text-xs px-2 py-1 rounded ${thread.status === 'running'
                                    ? 'bg-blue-700 text-blue-200'
                                    : thread.status === 'paused'
                                        ? 'bg-amber-700 text-amber-200'
                                        : 'bg-slate-700 text-slate-300'}`, children: thread.status }) }), _jsx("div", { className: "col-span-2 flex justify-center", children: _jsx(ThreadControls, { threadId: thread.threadId, status: thread.status, size: "sm" }) }), _jsx("div", { className: "col-span-2 flex justify-center", children: _jsx(InjectMenu, { threadId: thread.threadId, stepId: "0", mrfEligible: thread.status === 'running', mctsEligible: thread.status === 'running', size: "sm" }) })] }, thread.threadId))) })] }));
};
// ============================================================================
// Example 7: Responsive Layout (Mobile/Desktop)
// ============================================================================
export const ResponsiveThreadRowExample = ({ threadId, name, status, priority, }) => {
    return (_jsxs("div", { className: "bg-slate-700 rounded-lg p-3 space-y-2 md:space-y-0 md:flex md:items-center md:gap-2", children: [_jsxs("div", { className: "flex-1", children: [_jsx("h3", { className: "font-semibold text-white", children: name }), _jsx("p", { className: "text-xs text-slate-400 md:hidden capitalize", children: status })] }), _jsxs("div", { className: "flex items-center justify-between gap-2 md:gap-1", children: [_jsxs("div", { className: "hidden md:flex gap-1 items-center", children: [_jsx("span", { className: "text-xs text-slate-400", children: "P:" }), _jsx(PriorityControls, { threadId: threadId, priority: priority, size: "sm", orientation: "horizontal" })] }), _jsx(ThreadControls, { threadId: threadId, status: status, size: "sm" }), _jsx(InjectMenu, { threadId: threadId, stepId: "0", mrfEligible: true, mctsEligible: true, size: "sm" })] })] }));
};
// ============================================================================
// Example 8: Custom Styling
// ============================================================================
export const StyledThreadRowExample = ({ threadId, name, status, priority, }) => {
    // Custom styling based on priority
    const priorityBgClass = priority >= 75
        ? 'bg-red-900/20 border-red-700'
        : priority >= 50
            ? 'bg-amber-900/20 border-amber-700'
            : 'bg-blue-900/20 border-blue-700';
    return (_jsxs("div", { className: `
        flex items-center gap-3 p-3 rounded-lg border
        transition-all duration-150 hover:shadow-lg
        ${priorityBgClass}
      `, children: [_jsx("div", { className: `
          w-4 h-4 rounded-full flex-shrink-0
          ${status === 'running'
                    ? 'bg-green-500 animate-pulse'
                    : status === 'paused'
                        ? 'bg-amber-500'
                        : 'bg-slate-500'}
        ` }), _jsxs("div", { className: "flex-1 min-w-0", children: [_jsx("h3", { className: "font-semibold text-white truncate", children: name }), _jsxs("div", { className: "flex gap-2 mt-1 text-xs", children: [_jsx("span", { className: "text-slate-400", children: "Status:" }), _jsx("span", { className: "text-blue-300 capitalize", children: status }), _jsx("span", { className: "text-slate-400", children: "Priority:" }), _jsx("span", { className: "text-amber-300 font-bold", children: priority })] })] }), _jsxs("div", { className: "flex items-center gap-1 flex-shrink-0", children: [_jsx(PriorityControls, { threadId: threadId, priority: priority, size: "md", orientation: "vertical", className: "px-1" }), _jsx("div", { className: "w-px h-6 bg-slate-600" }), _jsx(ThreadControls, { threadId: threadId, status: status, size: "md" }), _jsx("div", { className: "w-px h-6 bg-slate-600" }), _jsx(InjectMenu, { threadId: threadId, stepId: "0", mrfEligible: status !== 'idle', mctsEligible: status !== 'idle', size: "md" })] })] }));
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
//# sourceMappingURL=EXAMPLES.js.map