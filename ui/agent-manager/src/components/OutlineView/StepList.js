import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * StepList Component
 * Container for a list of task steps within a thread
 * Manages step hierarchy, visual connectors, and interaction state
 */
import { useState, useCallback, useMemo } from 'react';
import StepRow from './StepRow';
const ConnectorLine = ({ isLast, depth }) => {
    const indentPx = depth * 12 + 10; // Align with step icon
    const heightClass = isLast ? 'h-4' : 'h-8';
    return (_jsx("div", { className: `absolute ${heightClass} w-px bg-slate-700/40 pointer-events-none`, style: {
            left: `${indentPx}px`,
            top: '100%',
        } }));
};
/**
 * StepListHeader - Optional header showing thread or root task info
 */
const StepListHeader = ({ rootTask, threadId, }) => {
    if (!rootTask)
        return null;
    return (_jsxs("div", { className: "px-3 py-2 border-b border-slate-700 bg-slate-800/30", children: [_jsx("h4", { className: "text-xs font-semibold text-slate-300 uppercase tracking-wider", children: rootTask.name }), rootTask.query && (_jsx("p", { className: "text-xs text-slate-400 mt-1 truncate", children: rootTask.query }))] }));
};
/**
 * StepList Component
 */
export const StepList = ({ steps, threadId, rootTask, hoveredStepId, selectedStepId, onStepHover, onStepSelect, onInjectMRF, onInjectMCTS, showQueryPreview = true, className = '', }) => {
    const [internalHoveredId, setInternalHoveredId] = useState(null);
    const [internalSelectedId, setInternalSelectedId] = useState(null);
    // Use provided props or internal state
    const currentHoveredId = hoveredStepId !== undefined ? hoveredStepId : internalHoveredId;
    const currentSelectedId = selectedStepId !== undefined ? selectedStepId : internalSelectedId;
    const handleStepHover = useCallback((stepId) => {
        if (hoveredStepId === undefined) {
            setInternalHoveredId(stepId);
        }
        onStepHover?.(stepId);
    }, [hoveredStepId, onStepHover]);
    const handleStepSelect = useCallback((stepId) => {
        if (selectedStepId === undefined) {
            setInternalSelectedId(stepId);
        }
        onStepSelect?.(stepId);
    }, [selectedStepId, onStepSelect]);
    // Build a map of step IDs to their indices for faster lookups
    const stepIndexMap = useMemo(() => {
        const map = {};
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
            if (a.parentId === b.id)
                return 1; // a is child of b
            if (b.parentId === a.id)
                return -1; // b is child of a
            // Secondary: sort by depth (shallower first)
            if (a.depth !== b.depth)
                return a.depth - b.depth;
            // Tertiary: maintain original order for same-depth siblings
            return stepIndexMap[a.id] - stepIndexMap[b.id];
        });
        return sorted;
    }, [steps, stepIndexMap]);
    // Calculate which steps should show connector lines
    const shouldShowConnector = useCallback((stepId) => {
        const stepIndex = sortedSteps.findIndex((s) => s.id === stepId);
        if (stepIndex === -1 || stepIndex === sortedSteps.length - 1) {
            return false;
        }
        const currentStep = sortedSteps[stepIndex];
        const nextStep = sortedSteps[stepIndex + 1];
        // Only show connector if next step is at same or greater depth
        // (indicates continuation of execution flow)
        return nextStep.depth >= currentStep.depth;
    }, [sortedSteps]);
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
        return (_jsxs("div", { className: `bg-slate-800/30 border border-slate-700 rounded overflow-hidden ${className}`, children: [_jsx(StepListHeader, { rootTask: rootTask, threadId: threadId }), _jsx("div", { className: "px-4 py-8 text-center text-slate-400 text-sm", children: _jsx("p", { children: "No steps in this thread" }) })] }));
    }
    return (_jsxs("div", { className: `bg-slate-800/30 border border-slate-700 rounded overflow-hidden ${className}`, children: [rootTask && _jsx(StepListHeader, { rootTask: rootTask, threadId: threadId }), _jsx("div", { className: "h-1 bg-slate-700", children: _jsx("div", { className: "h-full bg-gradient-to-r from-blue-500 via-cyan-500 to-emerald-500 transition-all duration-300", style: {
                        width: `${stats.total > 0 ? (stats.completed / stats.total) * 100 : 0}%`,
                    } }) }), _jsxs("div", { className: "px-3 py-1 border-b border-slate-700/50 bg-slate-800/20 flex items-center gap-4 text-xs text-slate-400", children: [_jsxs("span", { children: ["Steps:", ' ', _jsxs("span", { className: "text-slate-200 font-mono", children: [stats.completed, "/", stats.total] })] }), stats.running > 0 && (_jsxs("span", { children: ["Running:", ' ', _jsx("span", { className: "text-blue-400 font-mono", children: stats.running })] })), stats.failed > 0 && (_jsxs("span", { children: ["Failed:", ' ', _jsx("span", { className: "text-red-400 font-mono", children: stats.failed })] })), _jsx("div", { className: "flex-1" }), _jsxs("span", { className: "hidden lg:inline text-slate-500 font-mono text-xs", children: [threadId.substring(0, 8), "..."] })] }), _jsx("div", { className: "overflow-y-auto max-h-96 divide-y divide-slate-700/30", children: sortedSteps.map((step, index) => (_jsxs("div", { className: "relative", children: [shouldShowConnector(step.id) && (_jsx(ConnectorLine, { isLast: index === sortedSteps.length - 1, depth: step.depth })), _jsx(StepRow, { step: step, depth: step.depth, isHovered: currentHoveredId === step.id, isSelected: currentSelectedId === step.id, onHover: handleStepHover, onClick: handleStepSelect, onInjectMRF: onInjectMRF, onInjectMCTS: onInjectMCTS, showQueryPreview: showQueryPreview })] }, step.id))) }), _jsxs("div", { className: "px-3 py-2 border-t border-slate-700/50 bg-slate-800/20 text-xs text-slate-400 flex items-center justify-between", children: [_jsx("span", { children: stats.failed > 0
                            ? `${stats.failed} failed`
                            : stats.running > 0
                                ? `${stats.running} running`
                                : 'All steps completed' }), steps.some((s) => s.mrfEligible || s.mctsEligible) && (_jsxs("span", { className: "text-slate-500", children: [steps.filter((s) => s.mrfEligible).length > 0 && (_jsx("span", { className: "text-emerald-400 ml-2", children: "MRF available" })), steps.filter((s) => s.mctsEligible).length > 0 && (_jsx("span", { className: "text-cyan-400 ml-2", children: "MCTS available" }))] }))] })] }));
};
StepList.displayName = 'StepList';
export default StepList;
//# sourceMappingURL=StepList.js.map