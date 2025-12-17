import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { ChevronDown, ChevronRight, Check, X, Zap, Target } from 'lucide-react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';
import { StatusBadge } from '../common/StatusBadge';
export const ThreadCard = ({ thread, isActive = false, onSelect, }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [isEditingName, setIsEditingName] = useState(false);
    const [editedName, setEditedName] = useState(thread.name);
    // Store actions
    const { updateThread, setActiveThread, upvoteThread, downvoteThread, pauseThread, resumeThread, cancelThread, } = useAgentManagerStore();
    // Get dependencies for display
    const { dependsOn, blocks } = useAgentManagerStore((state) => state.getThreadDependencies(thread.id));
    // Get child threads for hierarchy display
    const childThreads = useAgentManagerStore((state) => state.getChildThreads(thread.id));
    // Handlers
    const handleNameEdit = () => {
        if (isEditingName && editedName !== thread.name) {
            updateThread(thread.id, { name: editedName });
        }
        setIsEditingName(!isEditingName);
    };
    const handleNameCancel = () => {
        setEditedName(thread.name);
        setIsEditingName(false);
    };
    const handleSelectThread = () => {
        setActiveThread(thread.id);
        onSelect?.(thread.id);
    };
    // Format elapsed time
    const formatTime = (ms) => {
        if (ms < 1000)
            return `${Math.round(ms)}ms`;
        const seconds = (ms / 1000).toFixed(1);
        return `${seconds}s`;
    };
    // Format tokens with K suffix for thousands
    const formatTokens = (tokens) => {
        if (tokens < 1000)
            return tokens.toString();
        return `${(tokens / 1000).toFixed(1)}k`;
    };
    // Get border color based on status
    const getBorderColor = () => {
        switch (thread.status) {
            case 'running':
                return 'border-blue-500';
            case 'paused':
                return 'border-amber-500';
            case 'completed':
                return 'border-emerald-500';
            case 'failed':
                return 'border-red-500';
            case 'cancelled':
                return 'border-slate-600';
            case 'idle':
            default:
                return 'border-slate-700';
        }
    };
    // Get background color based on active state
    const getBackgroundColor = () => {
        if (isActive) {
            return 'bg-slate-800';
        }
        return 'bg-slate-900 hover:bg-slate-850';
    };
    // Get animation classes
    const getAnimationClasses = () => {
        if (thread.status === 'running') {
            return 'animate-pulse';
        }
        return '';
    };
    return (_jsxs("div", { className: `
        border-l-4 rounded-lg transition-all duration-200 cursor-pointer
        ${getBorderColor()}
        ${getBackgroundColor()}
        ${isActive ? 'ring-2 ring-blue-500' : ''}
        ${getAnimationClasses()}
      `, onClick: handleSelectThread, children: [_jsxs("div", { className: "p-4 space-y-3", children: [_jsxs("div", { className: "flex items-center justify-between gap-3", children: [_jsx("button", { onClick: (e) => {
                                    e.stopPropagation();
                                    setIsExpanded(!isExpanded);
                                }, className: "p-1 text-slate-400 hover:text-slate-300 hover:bg-slate-800 rounded transition-colors", title: isExpanded ? 'Collapse' : 'Expand', children: isExpanded ? (_jsx(ChevronDown, { size: 16 })) : (_jsx(ChevronRight, { size: 16 })) }), _jsx(StatusBadge, { status: thread.status, size: "sm", showLabel: false }), isEditingName ? (_jsxs("div", { className: "flex-1 flex items-center gap-2", onClick: (e) => e.stopPropagation(), children: [_jsx("input", { type: "text", value: editedName, onChange: (e) => setEditedName(e.target.value), onKeyDown: (e) => {
                                            if (e.key === 'Enter')
                                                handleNameEdit();
                                            if (e.key === 'Escape')
                                                handleNameCancel();
                                        }, autoFocus: true, className: "flex-1 px-2 py-1 bg-slate-800 border border-blue-500 rounded text-slate-100 text-sm font-medium placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-400", placeholder: "Enter thread name" }), _jsx("button", { onClick: (e) => {
                                            e.stopPropagation();
                                            handleNameEdit();
                                        }, className: "p-1 text-emerald-500 hover:text-emerald-400 hover:bg-slate-800 rounded transition-colors", title: "Save", children: _jsx(Check, { size: 16 }) }), _jsx("button", { onClick: (e) => {
                                            e.stopPropagation();
                                            handleNameCancel();
                                        }, className: "p-1 text-red-500 hover:text-red-400 hover:bg-slate-800 rounded transition-colors", title: "Cancel", children: _jsx(X, { size: 16 }) })] })) : (_jsx("h3", { className: "flex-1 font-semibold text-slate-100 text-sm cursor-text hover:text-blue-300 transition-colors", onClick: (e) => {
                                    e.stopPropagation();
                                    setIsEditingName(true);
                                }, title: "Click to edit name", children: thread.name })), _jsxs("div", { className: "flex items-center gap-1", onClick: (e) => e.stopPropagation(), children: [_jsx("button", { onClick: (e) => {
                                            e.stopPropagation();
                                            downvoteThread(thread.id);
                                        }, className: "p-1 text-slate-500 hover:text-slate-300 hover:bg-slate-800 rounded transition-colors", title: "Lower priority", children: "\u2212" }), _jsx("div", { className: "w-8 text-center text-xs font-semibold text-slate-400", children: thread.priority }), _jsx("button", { onClick: (e) => {
                                            e.stopPropagation();
                                            upvoteThread(thread.id);
                                        }, className: "p-1 text-slate-500 hover:text-slate-300 hover:bg-slate-800 rounded transition-colors", title: "Raise priority", children: "+" })] })] }), _jsxs("div", { className: "flex items-center justify-between", children: [_jsxs("div", { className: "text-xs text-slate-400 font-mono space-x-4 flex", children: [_jsxs("span", { children: ["Step ", thread.currentStep, "/", thread.totalSteps] }), _jsx("span", { children: formatTime(thread.elapsedTimeMs) }), _jsxs("span", { children: [formatTokens(thread.tokensUsed), " tokens"] }), thread.tokenBudget && (_jsxs("span", { className: "text-slate-500", children: ["(", Math.round((thread.tokensUsed / thread.tokenBudget) * 100), "% of budget)"] }))] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsxs("div", { className: "flex items-center gap-1", children: [_jsxs("div", { className: "text-xs text-slate-400", children: [Math.round(thread.confidence * 100), "%"] }), _jsx("div", { className: "w-2 h-2 rounded-full", style: {
                                                    backgroundColor: thread.confidence > 0.7
                                                        ? '#10b981'
                                                        : thread.confidence > 0.4
                                                            ? '#f59e0b'
                                                            : '#ef4444',
                                                }, title: `Confidence: ${Math.round(thread.confidence * 100)}%` })] }), _jsxs("div", { className: "flex items-center gap-1", children: [_jsx("div", { className: "text-xs text-slate-500", children: "E:" }), _jsx("div", { className: "w-2 h-2 rounded-full", style: {
                                                    backgroundColor: thread.epistemicConfidence > 0.7
                                                        ? '#06b6d4'
                                                        : thread.epistemicConfidence > 0.4
                                                            ? '#84cc16'
                                                            : '#f43f5e',
                                                }, title: `Epistemic Confidence: ${Math.round(thread.epistemicConfidence * 100)}%` })] })] })] }), (dependsOn.length > 0 || blocks.length > 0) && (_jsxs("div", { className: "text-xs space-y-1 pt-2 border-t border-slate-800", children: [dependsOn.length > 0 && (_jsxs("div", { className: "text-slate-400", children: [_jsx("span", { className: "text-slate-500", children: "Waiting on:" }), ' ', dependsOn.map((t) => t.name).join(', ')] })), blocks.length > 0 && (_jsxs("div", { className: "text-slate-400", children: [_jsx("span", { className: "text-slate-500", children: "Blocks:" }), ' ', blocks.map((t) => t.name).join(', ')] }))] })), _jsxs("div", { className: "flex items-center gap-2 text-xs", children: [_jsx("span", { className: "px-2 py-1 bg-slate-800 text-slate-300 rounded font-mono", children: thread.reasoningMode }), _jsx("span", { className: "px-2 py-1 bg-slate-800 text-slate-300 rounded text-xs", children: thread.agentType })] })] }), isExpanded && (_jsxs("div", { className: "border-t border-slate-800 bg-slate-950 px-4 py-3 space-y-3", onClick: (e) => e.stopPropagation(), children: [thread.totalSteps > 0 && (_jsxs("div", { className: "space-y-2", children: [_jsxs("h4", { className: "text-xs font-semibold text-slate-400 uppercase", children: ["Steps (", thread.currentStep, "/", thread.totalSteps, ")"] }), _jsx("div", { className: "space-y-1 max-h-48 overflow-y-auto", children: Array.from({ length: thread.totalSteps }).map((_, i) => {
                                    const stepNum = i + 1;
                                    const isCurrentStep = stepNum === thread.currentStep;
                                    const isPastStep = stepNum < thread.currentStep;
                                    return (_jsxs("div", { className: `
                        px-2 py-1 text-xs rounded transition-colors
                        ${isCurrentStep
                                            ? 'bg-blue-900 text-blue-100'
                                            : isPastStep
                                                ? 'bg-emerald-900 text-emerald-100'
                                                : 'bg-slate-800 text-slate-400'}
                      `, children: [_jsxs("span", { className: "font-mono font-semibold", children: ["Step ", stepNum] }), isCurrentStep && (_jsx("span", { className: "ml-2 animate-pulse", children: "\u25CF Running" })), isPastStep && (_jsx("span", { className: "ml-2", children: "\u2713 Completed" }))] }, i));
                                }) })] })), _jsxs("div", { className: "flex flex-wrap gap-2 pt-2 border-t border-slate-800", children: [thread.status === 'running' ? (_jsx("button", { onClick: (e) => {
                                    e.stopPropagation();
                                    pauseThread(thread.id);
                                }, className: "px-3 py-1 bg-amber-600 hover:bg-amber-700 text-white text-xs font-medium rounded transition-colors", title: "Pause thread", children: "\u23F8 Pause" })) : thread.status === 'paused' ? (_jsx("button", { onClick: (e) => {
                                    e.stopPropagation();
                                    resumeThread(thread.id);
                                }, className: "px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded transition-colors", title: "Resume thread", children: "\u25B6 Resume" })) : null, (thread.status === 'running' || thread.status === 'paused') && (_jsx("button", { onClick: (e) => {
                                    e.stopPropagation();
                                    cancelThread(thread.id);
                                }, className: "px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded transition-colors", title: "Cancel thread", children: "\u2715 Cancel" })), _jsxs("button", { onClick: (e) => {
                                    e.stopPropagation();
                                    // TODO: Implement MRF injection
                                    console.log('MRF injection triggered for thread:', thread.id);
                                }, className: "px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white text-xs font-medium rounded transition-colors inline-flex items-center gap-1", title: "Inject Metaprompting Refinement Framework", children: [_jsx(Zap, { size: 12 }), "MRF"] }), _jsxs("button", { onClick: (e) => {
                                    e.stopPropagation();
                                    // TODO: Implement MCTS injection
                                    console.log('MCTS injection triggered for thread:', thread.id);
                                }, className: "px-3 py-1 bg-cyan-600 hover:bg-cyan-700 text-white text-xs font-medium rounded transition-colors inline-flex items-center gap-1", title: "Inject Monte Carlo Tree Search", children: [_jsx(Target, { size: 12 }), "MCTS"] })] }), childThreads.length > 0 && (_jsxs("div", { className: "text-xs text-slate-400 pt-2 border-t border-slate-800", children: [_jsx("span", { className: "text-slate-500", children: "Spawned threads:" }), " ", childThreads.length] }))] }))] }));
};
export default ThreadCard;
//# sourceMappingURL=ThreadCard.js.map