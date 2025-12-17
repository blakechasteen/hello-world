import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * DetailPanel Component
 * Main expanded view container that shows detailed information about a selected thread
 *
 * Features:
 * - Displays selected thread's full details (name, status, agent type, reasoning mode)
 * - Multi-dimensional progress tracking (steps, time, tokens) with ProgressBars component
 * - Confidence and epistemic confidence displays
 * - Tabbed interface for History, Memory, and Files views
 * - Shows dependencies (depends_on, blocks) with visual indicators
 * - Smooth slide-in animation from right
 * - Editable thread name with inline editing
 * - Responsive layout with min/max widths
 *
 * Theme:
 * - Dark theme: bg-slate-900 for panel, bg-slate-800 for cards
 * - Borders: border-slate-700
 * - Text: text-slate-100 for headings, text-slate-400 for secondary
 * - Smooth animations and transitions
 *
 * Dependencies:
 * - useAgentManagerStore: Zustand store for agent state
 * - StatusBadge: Component for status display
 * - ProgressBars: Multi-dimensional progress visualization
 *
 * Tab Layout:
 * - History: Shows step-by-step reasoning trace (StepHistory component)
 * - Memory: Shows memory nodes and knowledge graph access (MemoryNodes component)
 * - Files: Shows files accessed/modified during execution (FileTreeViewer component)
 *
 * Usage:
 * ```tsx
 * const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null);
 *
 * {selectedThreadId && (
 *   <DetailPanel
 *     threadId={selectedThreadId}
 *     onClose={() => setSelectedThreadId(null)}
 *   />
 * )}
 * ```
 */
import { useState, useRef, useEffect } from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';
import { StatusBadge } from '../common/StatusBadge';
import { ProgressBars } from '../OutlineView';
/**
 * DetailPanel Component
 * Shows comprehensive details for a selected agent thread
 *
 * Component structure:
 * 1. Header - Title, status badge, close button
 * 2. Confidence section - Confidence and epistemic confidence bars
 * 3. Progress section - Multi-dimensional progress (steps, time, tokens)
 * 4. Dependencies - Shows depends_on and blocks relationships
 * 5. Tabs - History, Memory, Files navigation
 * 6. Tab content - Context-specific content for each tab
 * 7. Footer - Metadata timestamps
 */
export const DetailPanel = ({ threadId, onClose }) => {
    // State
    const [activeTab, setActiveTab] = useState('history');
    const [isEditingName, setIsEditingName] = useState(false);
    const [editedName, setEditedName] = useState('');
    const editInputRef = useRef(null);
    // Store
    const thread = useAgentManagerStore((state) => state.getThreadById(threadId));
    const updateThread = useAgentManagerStore((state) => state.updateThread);
    const getThreadDependencies = useAgentManagerStore((state) => state.getThreadDependencies);
    // Get dependencies
    const dependencies = thread ? getThreadDependencies(threadId) : { dependsOn: [], blocks: [] };
    // Initialize edit name
    useEffect(() => {
        if (thread) {
            setEditedName(thread.name);
        }
    }, [thread]);
    // Focus edit input when editing starts
    useEffect(() => {
        if (isEditingName && editInputRef.current) {
            editInputRef.current.focus();
            editInputRef.current.select();
        }
    }, [isEditingName]);
    // Handle name save
    const handleSaveName = () => {
        if (editedName.trim() && thread) {
            updateThread(threadId, { name: editedName.trim() });
        }
        setIsEditingName(false);
    };
    // Handle name cancel
    const handleCancelEdit = () => {
        if (thread) {
            setEditedName(thread.name);
        }
        setIsEditingName(false);
    };
    // Handle keyboard events in edit mode
    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleSaveName();
        }
        else if (e.key === 'Escape') {
            handleCancelEdit();
        }
    };
    // Return null if thread not found
    if (!thread) {
        return (_jsx("div", { className: "w-full h-full bg-slate-900 border-l border-slate-700 flex items-center justify-center", children: _jsxs("div", { className: "text-center", children: [_jsx("p", { className: "text-slate-400 mb-2", children: "Thread not found" }), _jsx("button", { onClick: onClose, className: "text-slate-500 hover:text-slate-300 transition-colors", children: "Close panel" })] }) }));
    }
    /**
     * Format milliseconds as human-readable time string
     */
    const formatTime = (ms) => {
        if (ms < 1000) {
            return `${Math.round(ms)}ms`;
        }
        const seconds = (ms / 1000).toFixed(1);
        return `${seconds}s`;
    };
    return (_jsxs("div", { className: "w-full h-full bg-slate-900 border-l border-slate-700 flex flex-col overflow-hidden animate-slide-in", children: [_jsxs("div", { className: "border-b border-slate-700 bg-slate-800/50 px-4 py-4 flex-shrink-0", children: [_jsxs("div", { className: "flex items-start justify-between gap-3 mb-3", children: [_jsx("div", { className: "flex-1 min-w-0", children: isEditingName ? (_jsxs("div", { className: "flex gap-2 items-center", children: [_jsx("input", { ref: editInputRef, type: "text", value: editedName, onChange: (e) => setEditedName(e.target.value), onKeyDown: handleKeyDown, className: "flex-1 px-2 py-1 bg-slate-700 border border-slate-600 rounded text-slate-100 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500", placeholder: "Thread name" }), _jsx("button", { onClick: handleSaveName, className: "px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded transition-colors", title: "Save thread name", "aria-label": "Save", children: "\u2713" }), _jsx("button", { onClick: handleCancelEdit, className: "px-2 py-1 bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs rounded transition-colors", title: "Cancel editing", "aria-label": "Cancel", children: "\u2715" })] })) : (_jsxs("div", { onClick: () => setIsEditingName(true), className: "group cursor-pointer", title: "Click to edit thread name", children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 group-hover:text-blue-400 transition-colors truncate", children: thread.name }), _jsxs("p", { className: "text-xs text-slate-500 group-hover:text-slate-400 truncate", children: ["ID: ", thread.id] })] })) }), _jsx("button", { onClick: onClose, className: "flex-shrink-0 text-slate-500 hover:text-slate-300 hover:bg-slate-700 rounded p-1 transition-colors", title: "Close detail panel", "aria-label": "Close detail panel", children: _jsx("svg", { className: "w-5 h-5", fill: "none", stroke: "currentColor", viewBox: "0 0 24 24", children: _jsx("path", { strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: 2, d: "M6 18L18 6M6 6l12 12" }) }) })] }), _jsxs("div", { className: "flex items-center gap-3 flex-wrap", children: [_jsx(StatusBadge, { status: thread.status, size: "sm" }), _jsx("div", { className: "text-xs text-slate-400 bg-slate-700 px-2 py-1 rounded", children: thread.agentType }), _jsx("div", { className: "text-xs text-slate-400 bg-slate-700 px-2 py-1 rounded", children: thread.reasoningMode }), _jsxs("div", { className: "text-xs text-slate-400", children: ["Priority: ", _jsx("span", { className: "text-slate-200 font-medium", children: thread.priority })] })] })] }), _jsxs("div", { className: "border-b border-slate-700 bg-slate-800/30 px-4 py-3 flex-shrink-0", children: [_jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("div", { className: "text-xs text-slate-400 font-semibold uppercase tracking-wide mb-1", children: "Confidence" }), _jsxs("div", { className: "flex items-baseline gap-2", children: [_jsxs("div", { className: "text-lg font-bold text-blue-400", children: [(thread.confidence * 100).toFixed(0), "%"] }), _jsx("div", { className: "flex-1 bg-slate-700 rounded-full h-1.5", children: _jsx("div", { className: "bg-blue-500 h-1.5 rounded-full transition-all duration-300", style: { width: `${thread.confidence * 100}%` } }) })] })] }), _jsxs("div", { children: [_jsx("div", { className: "text-xs text-slate-400 font-semibold uppercase tracking-wide mb-1", children: "Epistemic Confidence" }), _jsxs("div", { className: "flex items-baseline gap-2", children: [_jsxs("div", { className: "text-lg font-bold text-amber-400", children: [(thread.epistemicConfidence * 100).toFixed(0), "%"] }), _jsx("div", { className: "flex-1 bg-slate-700 rounded-full h-1.5", children: _jsx("div", { className: "bg-amber-500 h-1.5 rounded-full transition-all duration-300", style: { width: `${thread.epistemicConfidence * 100}%` } }) })] })] })] }), thread.epistemicConfidence < 0.3 && (_jsx("div", { className: "mt-2 text-xs text-amber-400 bg-amber-900/20 border border-amber-700/30 rounded px-2 py-1", children: "\u26A0 Low epistemic confidence: System uncertainty about its knowledge is high" }))] }), _jsxs("div", { className: "border-b border-slate-700 bg-slate-800/30 px-4 py-3 flex-shrink-0", children: [_jsx("div", { className: "text-xs text-slate-400 font-semibold uppercase tracking-wide mb-2", children: "Execution Progress" }), _jsx(ProgressBars, { currentStep: thread.currentStep, totalSteps: thread.totalSteps, elapsedTimeMs: thread.elapsedTimeMs, timeBudgetMs: thread.totalSteps > 0
                            ? Math.max(thread.elapsedTimeMs * 2, 5000) // Estimate budget as 2x elapsed or 5s min
                            : undefined, tokensUsed: thread.tokensUsed, tokenBudget: thread.tokenBudget, variant: "detailed", size: "sm", showValues: true })] }), (dependencies.dependsOn.length > 0 || dependencies.blocks.length > 0) && (_jsx("div", { className: "border-b border-slate-700 bg-slate-800/30 px-4 py-3 flex-shrink-0", children: _jsxs("div", { className: "space-y-2", children: [dependencies.dependsOn.length > 0 && (_jsxs("div", { children: [_jsxs("div", { className: "text-xs text-slate-400 font-semibold uppercase tracking-wide mb-1", children: ["Depends On (", dependencies.dependsOn.length, ")"] }), _jsx("div", { className: "flex flex-wrap gap-1", children: dependencies.dependsOn.map((dep) => (_jsxs("div", { className: "text-xs bg-blue-900/30 border border-blue-700/50 text-blue-300 px-2 py-1 rounded hover:bg-blue-900/50 transition-colors cursor-pointer", title: `Depends on: ${dep.name}`, children: ["\u2190 ", dep.name] }, dep.id))) })] })), dependencies.blocks.length > 0 && (_jsxs("div", { children: [_jsxs("div", { className: "text-xs text-slate-400 font-semibold uppercase tracking-wide mb-1", children: ["Blocks (", dependencies.blocks.length, ")"] }), _jsx("div", { className: "flex flex-wrap gap-1", children: dependencies.blocks.map((blocked) => (_jsxs("div", { className: "text-xs bg-amber-900/30 border border-amber-700/50 text-amber-300 px-2 py-1 rounded hover:bg-amber-900/50 transition-colors cursor-pointer", title: `Blocks: ${blocked.name}`, children: ["\u2192 ", blocked.name] }, blocked.id))) })] }))] }) })), _jsx("div", { className: "border-b border-slate-700 flex gap-0 bg-slate-800/30 px-4 flex-shrink-0", children: ['history', 'memory', 'files'].map((tab) => (_jsx("button", { onClick: () => setActiveTab(tab), className: `px-4 py-3 text-sm font-medium border-b-2 transition-colors capitalize ${activeTab === tab
                        ? 'text-blue-400 border-blue-500'
                        : 'text-slate-400 border-transparent hover:text-slate-300 hover:border-slate-600'}`, "aria-selected": activeTab === tab, role: "tab", children: tab }, tab))) }), _jsxs("div", { className: "flex-1 overflow-auto", children: [activeTab === 'history' && (_jsx("div", { className: "p-4", children: _jsxs("div", { className: "bg-slate-800 rounded border border-slate-700 p-4", children: [_jsxs("div", { className: "text-slate-400 text-sm space-y-2", children: [_jsx("p", { className: "font-semibold text-slate-300", children: "Reasoning History" }), _jsxs("p", { children: ["Step ", thread.currentStep, " of ", thread.totalSteps] }), _jsxs("p", { children: ["Elapsed: ", formatTime(thread.elapsedTimeMs)] }), _jsxs("p", { children: ["Tokens used: ", thread.tokensUsed.toLocaleString()] }), thread.status === 'completed' && thread.finalResponse && (_jsxs("div", { className: "mt-3 pt-3 border-t border-slate-700", children: [_jsx("p", { className: "font-semibold text-slate-300 mb-2", children: "Final Response:" }), _jsx("p", { className: "text-slate-400 text-sm line-clamp-4", children: thread.finalResponse })] })), thread.status === 'failed' && (_jsx("div", { className: "mt-3 pt-3 border-t border-slate-700", children: _jsxs("p", { className: "text-red-400", children: ["Status: Failed", thread.updatedAt && ` - ${new Date(thread.updatedAt).toLocaleTimeString()}`] }) }))] }), _jsx("div", { className: "mt-4 text-xs text-slate-500 border-t border-slate-700 pt-4", children: _jsx("p", { className: "italic", children: "Step-by-step history will appear here" }) })] }) })), activeTab === 'memory' && (_jsx("div", { className: "p-4", children: _jsx("div", { className: "bg-slate-800 rounded border border-slate-700 p-4", children: _jsxs("div", { className: "text-slate-400 text-sm", children: [_jsx("p", { className: "font-semibold text-slate-300 mb-2", children: "Memory Access" }), thread.swarmId && _jsxs("p", { children: ["Swarm ID: ", thread.swarmId] }), thread.parentThreadId && _jsxs("p", { children: ["Parent Thread: ", thread.parentThreadId] }), thread.childThreadIds.length > 0 && (_jsxs("p", { children: ["Child Threads: ", thread.childThreadIds.length] })), _jsx("div", { className: "mt-4 text-xs text-slate-500 border-t border-slate-700 pt-4", children: _jsx("p", { className: "italic", children: "Memory nodes and graph access will appear here" }) })] }) }) })), activeTab === 'files' && (_jsx("div", { className: "p-4", children: _jsx("div", { className: "bg-slate-800 rounded border border-slate-700 p-4", children: _jsxs("div", { className: "text-slate-400 text-sm", children: [_jsx("p", { className: "font-semibold text-slate-300 mb-2", children: "Files Accessed" }), _jsx("p", { children: "No files accessed in this execution" }), _jsx("div", { className: "mt-4 text-xs text-slate-500 border-t border-slate-700 pt-4", children: _jsx("p", { className: "italic", children: "File tree will appear here" }) })] }) }) }))] }), _jsx("div", { className: "border-t border-slate-700 bg-slate-800/30 px-4 py-2 flex-shrink-0 text-xs text-slate-500", children: _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: ["Created: ", new Date(thread.createdAt).toLocaleString()] }), _jsxs("div", { className: "text-right", children: ["Updated: ", new Date(thread.updatedAt).toLocaleString()] })] }) })] }));
};
DetailPanel.displayName = 'DetailPanel';
export default DetailPanel;
//# sourceMappingURL=DetailPanel.js.map