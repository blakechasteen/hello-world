import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * StepHistory Component
 * Scrollable list showing all steps executed in a thread with full details
 * Features: Virtualized scrolling, filtering, sorting, detail expansion
 *
 * HoloLoom Agent Manager UI - Phase 4
 */
import { useState, useMemo, useRef, useEffect } from 'react';
/**
 * Get status color classes
 */
const getStatusColorClasses = (status) => {
    switch (status) {
        case 'idle':
            return 'text-slate-400';
        case 'running':
            return 'text-blue-400 animate-pulse';
        case 'paused':
            return 'text-amber-400';
        case 'completed':
            return 'text-emerald-400';
        case 'failed':
            return 'text-red-400';
        case 'cancelled':
            return 'text-slate-500';
        case 'skipped':
            return 'text-slate-500';
        default:
            return 'text-slate-400';
    }
};
/**
 * Get status display text and icon
 */
const getStatusDisplay = (status) => {
    const statusMap = {
        idle: { icon: '○', label: 'Idle' },
        running: { icon: '▶', label: 'Running' },
        paused: { icon: '⏸', label: 'Paused' },
        completed: { icon: '✓', label: 'Completed' },
        failed: { icon: '✕', label: 'Failed' },
        cancelled: { icon: '⊗', label: 'Cancelled' },
        skipped: { icon: '—', label: 'Skipped' },
    };
    return statusMap[status];
};
/**
 * Get step type emoji
 */
const getStepTypeEmoji = (stepType) => {
    const typeMap = {
        query: '🔍',
        retrieval: '📥',
        reasoning: '🧠',
        synthesis: '🔀',
        verification: '✓',
        research: '📚',
        planning: '📋',
        execution: '⚙️',
        reflection: '💭',
    };
    return typeMap[stepType] || '◆';
};
/**
 * Format elapsed time for display (ms → human readable)
 */
const formatTime = (ms) => {
    if (ms < 1000)
        return `${Math.round(ms)}ms`;
    if (ms < 60000)
        return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
};
/**
 * Format token count for display
 */
const formatTokens = (tokens) => {
    if (tokens < 1000)
        return tokens.toString();
    if (tokens < 1000000)
        return `${(tokens / 1000).toFixed(1)}k`;
    return `${(tokens / 1000000).toFixed(1)}m`;
};
/**
 * Individual step row component with expand/collapse
 */
const StepHistoryRow = ({ step, index, isSelected, onSelect, onInjectMRF, onInjectMCTS, }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const statusDisplay = getStatusDisplay(step.status);
    const statusColorClasses = getStatusColorClasses(step.status);
    // Confidence color coding
    const confidenceColor = step.confidence >= 0.8
        ? 'text-emerald-400'
        : step.confidence >= 0.5
            ? 'text-amber-400'
            : 'text-red-400';
    return (_jsxs("div", { className: `
        border-b border-slate-700/50 transition-colors
        ${isSelected ? 'bg-slate-700/50 border-l-2 border-l-blue-500' : 'bg-slate-800/30 hover:bg-slate-750/30'}
      `, children: [_jsxs("div", { onClick: () => onSelect(step.id), className: "px-4 py-2 cursor-pointer flex items-center gap-3", children: [_jsx("button", { onClick: (e) => {
                            e.stopPropagation();
                            setIsExpanded(!isExpanded);
                        }, className: "flex-shrink-0 w-5 h-5 flex items-center justify-center text-slate-400 hover:text-slate-200 transition-colors", children: _jsx("span", { className: `text-sm transition-transform ${isExpanded ? 'rotate-90' : ''}`, children: "\u25B6" }) }), _jsx("div", { className: `flex-shrink-0 w-5 h-5 flex items-center justify-center ${statusColorClasses}`, children: statusDisplay.icon }), _jsx("div", { className: "flex-shrink-0 w-8 text-right text-xs text-slate-500 font-mono", children: index + 1 }), _jsxs("div", { className: "flex-1 min-w-0 flex items-center gap-2", children: [_jsx("span", { className: "flex-shrink-0 text-base", children: getStepTypeEmoji(step.stepType) }), _jsxs("div", { className: "min-w-0", children: [_jsx("div", { className: "text-sm font-medium text-slate-100 truncate", children: step.name || step.stepType }), step.query && (_jsx("div", { className: "text-xs text-slate-400 truncate max-w-xs", children: step.query }))] })] }), _jsx("div", { className: `flex-shrink-0 text-xs font-mono ${confidenceColor}`, children: step.confidence.toFixed(2) }), _jsxs("div", { className: "hidden sm:block flex-shrink-0 text-xs text-slate-400 font-mono min-w-[50px] text-right", children: [formatTokens(step.tokensUsed), "t"] }), _jsx("div", { className: "hidden md:block flex-shrink-0 text-xs text-slate-400 font-mono min-w-[60px] text-right", children: formatTime(step.elapsedTimeMs) }), _jsx("div", { className: "hidden lg:flex flex-shrink-0", children: _jsx("span", { className: `text-xs font-medium px-2 py-0.5 rounded-full ${statusColorClasses}`, children: statusDisplay.label }) }), (step.mrfEligible || step.mctsEligible || step.injectionApplied) && (_jsxs("div", { className: "flex-shrink-0 flex items-center gap-1", children: [step.injectionApplied && (_jsx("span", { className: "text-xs px-2 py-0.5 rounded bg-purple-900/40 text-purple-300 font-medium", children: step.injectionApplied.toUpperCase() })), !step.injectionApplied && step.mrfEligible && (_jsx("span", { className: "text-xs text-emerald-400 opacity-60", children: "mrf" })), !step.injectionApplied && step.mctsEligible && (_jsx("span", { className: "text-xs text-cyan-400 opacity-60", children: "mcts" }))] }))] }), isExpanded && (_jsx("div", { className: "px-4 py-3 bg-slate-900/40 border-t border-slate-700/30", children: _jsxs("div", { className: "space-y-3", children: [step.query && (_jsxs("div", { children: [_jsx("div", { className: "text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1", children: "Query" }), _jsx("div", { className: "text-sm text-slate-200 bg-slate-900 rounded px-3 py-2 font-mono break-words", children: step.query })] })), step.response && (_jsxs("div", { children: [_jsx("div", { className: "text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1", children: "Response" }), _jsx("div", { className: "text-sm text-slate-200 bg-slate-900 rounded px-3 py-2 max-h-32 overflow-y-auto break-words", children: step.response })] })), step.toolSelected && (_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "text-xs font-semibold text-slate-400 uppercase tracking-wider", children: "Tool:" }), _jsx("span", { className: "text-sm text-blue-300 bg-blue-900/20 px-2 py-0.5 rounded font-mono", children: step.toolSelected })] })), _jsxs("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-2 pt-2 border-t border-slate-700/30", children: [_jsxs("div", { className: "bg-slate-800/50 rounded px-2 py-2", children: [_jsx("div", { className: "text-xs text-slate-500 uppercase tracking-wider", children: "Tokens" }), _jsx("div", { className: "text-sm font-mono text-slate-100", children: formatTokens(step.tokensUsed) })] }), _jsxs("div", { className: "bg-slate-800/50 rounded px-2 py-2", children: [_jsx("div", { className: "text-xs text-slate-500 uppercase tracking-wider", children: "Time" }), _jsx("div", { className: "text-sm font-mono text-slate-100", children: formatTime(step.elapsedTimeMs) })] }), _jsxs("div", { className: "bg-slate-800/50 rounded px-2 py-2", children: [_jsx("div", { className: "text-xs text-slate-500 uppercase tracking-wider", children: "Confidence" }), _jsxs("div", { className: `text-sm font-mono ${confidenceColor}`, children: [(step.confidence * 100).toFixed(0), "%"] })] }), _jsxs("div", { className: "bg-slate-800/50 rounded px-2 py-2", children: [_jsx("div", { className: "text-xs text-slate-500 uppercase tracking-wider", children: "Status" }), _jsx("div", { className: `text-sm font-mono ${statusColorClasses}`, children: statusDisplay.label })] })] }), (step.dependsOn.length > 0 || step.blocks.length > 0) && (_jsxs("div", { className: "pt-2 border-t border-slate-700/30 space-y-1", children: [step.dependsOn.length > 0 && (_jsxs("div", { className: "text-xs", children: [_jsx("span", { className: "text-slate-500", children: "Depends on:" }), _jsx("span", { className: "text-slate-300 ml-2", children: step.dependsOn.join(', ') })] })), step.blocks.length > 0 && (_jsxs("div", { className: "text-xs", children: [_jsx("span", { className: "text-slate-500", children: "Blocks:" }), _jsx("span", { className: "text-slate-300 ml-2", children: step.blocks.join(', ') })] }))] })), (step.mrfEligible || step.mctsEligible) && !step.injectionApplied && (_jsxs("div", { className: "flex gap-2 pt-2 border-t border-slate-700/30", children: [step.mrfEligible && (_jsx("button", { onClick: (e) => {
                                        e.stopPropagation();
                                        onInjectMRF?.(step.id);
                                    }, className: "px-3 py-1.5 rounded text-xs font-medium bg-emerald-900/30 text-emerald-300 hover:bg-emerald-800/50 transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500", children: "Inject MRF" })), step.mctsEligible && (_jsx("button", { onClick: (e) => {
                                        e.stopPropagation();
                                        onInjectMCTS?.(step.id);
                                    }, className: "px-3 py-1.5 rounded text-xs font-medium bg-cyan-900/30 text-cyan-300 hover:bg-cyan-800/50 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500", children: "Inject MCTS" }))] }))] }) }))] }));
};
/**
 * StepHistory - Main component
 * Displays all steps in a virtualized scrollable list with filtering and sorting
 */
export const StepHistory = ({ steps, currentStepIndex, onStepSelect, onInjectMRF, onInjectMCTS, className = '', }) => {
    const [filterStatus, setFilterStatus] = useState('all');
    const [searchQuery, setSearchQuery] = useState('');
    const [sortOption, setSortOption] = useState('chronological');
    const containerRef = useRef(null);
    const currentStepRef = useRef(null);
    // Filter steps based on status and search query
    const filteredSteps = useMemo(() => {
        let filtered = steps;
        // Filter by status
        if (filterStatus !== 'all') {
            filtered = filtered.filter((step) => step.status === filterStatus);
        }
        // Filter by search query (searches name, query, and type)
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter((step) => step.name.toLowerCase().includes(query) ||
                step.query?.toLowerCase().includes(query) ||
                step.stepType.toLowerCase().includes(query));
        }
        return filtered;
    }, [steps, filterStatus, searchQuery]);
    // Sort steps
    const sortedSteps = useMemo(() => {
        if (sortOption === 'chronological') {
            return [...filteredSteps].sort((a, b) => steps.indexOf(a) - steps.indexOf(b));
        }
        // Sort by status priority
        const statusPriority = {
            running: 0,
            paused: 1,
            failed: 2,
            completed: 3,
            idle: 4,
            cancelled: 5,
            skipped: 6,
        };
        return [...filteredSteps].sort((a, b) => {
            const priorityDiff = statusPriority[a.status] - statusPriority[b.status];
            if (priorityDiff !== 0)
                return priorityDiff;
            return steps.indexOf(a) - steps.indexOf(b);
        });
    }, [filteredSteps, sortOption, steps]);
    // Auto-scroll to current step
    useEffect(() => {
        if (currentStepRef.current && containerRef.current) {
            currentStepRef.current.scrollIntoView({
                behavior: 'smooth',
                block: 'nearest',
            });
        }
    }, [currentStepIndex]);
    const currentStepId = steps[currentStepIndex]?.id;
    // Status counts for filter buttons
    const statusCounts = useMemo(() => {
        const counts = { all: steps.length };
        steps.forEach((step) => {
            counts[step.status] = (counts[step.status] ?? 0) + 1;
        });
        return counts;
    }, [steps]);
    return (_jsxs("div", { className: `flex flex-col h-full bg-slate-800 rounded-lg overflow-hidden ${className}`, children: [_jsxs("div", { className: "flex-shrink-0 border-b border-slate-700 bg-slate-850 px-4 py-3 space-y-3", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("h3", { className: "text-sm font-semibold text-slate-100", children: "Execution History" }), _jsxs("div", { className: "text-xs text-slate-400", children: [sortedSteps.length, " of ", steps.length, " steps"] })] }), _jsxs("div", { className: "relative", children: [_jsx("input", { type: "text", placeholder: "Search steps...", value: searchQuery, onChange: (e) => setSearchQuery(e.target.value), className: "w-full px-3 py-1.5 text-sm bg-slate-700 text-slate-100 placeholder-slate-500 rounded border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent" }), searchQuery && (_jsx("button", { onClick: () => setSearchQuery(''), className: "absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors", children: "\u2715" }))] }), _jsxs("div", { className: "flex flex-wrap gap-2", children: [['all', 'completed', 'running', 'failed'].map((status) => (_jsxs("button", { onClick: () => setFilterStatus(status), className: `px-2.5 py-1 text-xs font-medium rounded transition-colors ${filterStatus === status
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`, children: [status.charAt(0).toUpperCase() + status.slice(1), _jsxs("span", { className: "ml-1 text-slate-400", children: ["(", statusCounts[status] ?? 0, ")"] })] }, status))), _jsx("div", { className: "w-px h-6 bg-slate-700 mx-1" }), _jsxs("select", { value: sortOption, onChange: (e) => setSortOption(e.target.value), className: "px-2.5 py-1 text-xs bg-slate-700 text-slate-300 rounded border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500", children: [_jsx("option", { value: "chronological", children: "Chronological" }), _jsx("option", { value: "status", children: "By Status" })] })] })] }), _jsx("div", { ref: containerRef, className: "flex-1 overflow-y-auto overscroll-contain", children: sortedSteps.length === 0 ? (_jsx("div", { className: "h-full flex items-center justify-center text-slate-500 text-sm", children: steps.length === 0 ? 'No steps yet' : 'No steps match filter' })) : (_jsx("div", { className: "divide-y divide-slate-700/50", children: sortedSteps.map((step, index) => {
                        const isSelected = step.id === currentStepId;
                        return (_jsx("div", { ref: isSelected ? currentStepRef : undefined, children: _jsx(StepHistoryRow, { step: step, index: steps.indexOf(step), isSelected: isSelected, onSelect: (stepId) => onStepSelect?.(stepId), onInjectMRF: onInjectMRF, onInjectMCTS: onInjectMCTS }) }, step.id));
                    }) })) }), sortedSteps.length > 0 && (_jsxs("div", { className: "flex-shrink-0 border-t border-slate-700 bg-slate-850 px-4 py-2 text-xs text-slate-400 space-y-1", children: [_jsxs("div", { children: ["Total time: ", _jsx("span", { className: "text-slate-200 font-mono", children: formatTime(steps.reduce((sum, s) => sum + s.elapsedTimeMs, 0)) })] }), _jsxs("div", { children: ["Total tokens: ", _jsx("span", { className: "text-slate-200 font-mono", children: formatTokens(steps.reduce((sum, s) => sum + s.tokensUsed, 0)) })] }), _jsxs("div", { children: ["Avg confidence: ", _jsx("span", { className: "text-slate-200 font-mono", children: steps.length > 0 ? (steps.reduce((sum, s) => sum + s.confidence, 0) / steps.length).toFixed(2) : '—' })] })] }))] }));
};
StepHistory.displayName = 'StepHistory';
export default StepHistory;
//# sourceMappingURL=StepHistory.js.map