import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo } from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';
import ThreadCard from './ThreadCard';
/**
 * OutlineView Component
 * Displays all agent threads in outline mode, sorted by priority (highest first)
 *
 * Features:
 * - Lists ThreadCards in priority order
 * - Virtualized scrolling for performance (using CSS overflow)
 * - Empty state when no threads
 * - Filter indicator showing current filter
 * - Real-time updates via Zustand store
 */
export const OutlineView = () => {
    // Get filtered threads and active thread ID
    const { threads, activeThreadId, filter } = useAgentManagerStore((state) => ({
        threads: state.getFilteredThreads(),
        activeThreadId: state.activeThreadId,
        filter: state.filter,
    }));
    // Sort threads by priority (highest first), then by creation time (newest first)
    const sortedThreads = useMemo(() => {
        return [...threads].sort((a, b) => {
            // Primary sort: priority (descending)
            if (a.priority !== b.priority) {
                return b.priority - a.priority;
            }
            // Secondary sort: creation time (newest first)
            return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
        });
    }, [threads]);
    // Get thread count by status for header
    const threadStats = useMemo(() => {
        return {
            total: sortedThreads.length,
            running: sortedThreads.filter((t) => t.status === 'running').length,
            paused: sortedThreads.filter((t) => t.status === 'paused').length,
            completed: sortedThreads.filter((t) => t.status === 'completed').length,
            failed: sortedThreads.filter((t) => t.status === 'failed' || t.status === 'cancelled')
                .length,
        };
    }, [sortedThreads]);
    // Get filter label
    const getFilterLabel = () => {
        switch (filter) {
            case 'active':
                return 'Active Threads';
            case 'completed':
                return 'Completed Threads';
            case 'failed':
                return 'Failed Threads';
            case 'all':
            default:
                return 'All Threads';
        }
    };
    // Empty state
    if (sortedThreads.length === 0) {
        return (_jsxs("div", { className: "min-h-full bg-slate-950 p-8", children: [_jsxs("div", { className: "space-y-2 mb-8", children: [_jsx("h2", { className: "text-2xl font-bold text-slate-100", children: "Thread Outline" }), _jsx("p", { className: "text-slate-400", children: "Hierarchical view of agent threads and their dependencies" })] }), _jsxs("div", { className: "flex flex-col items-center justify-center py-16 text-center", children: [_jsx("div", { className: "text-6xl mb-4 text-slate-700", children: "\u2261" }), _jsx("h3", { className: "text-lg font-semibold text-slate-300 mb-2", children: "No Threads Yet" }), _jsx("p", { className: "text-slate-500 max-w-md", children: "Create a new thread from the sidebar to see it appear in the outline view. Threads will be sorted by priority and displayed here." })] })] }));
    }
    return (_jsx("div", { className: "min-h-full bg-slate-950", children: _jsxs("div", { className: "p-8 space-y-6", children: [_jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("h2", { className: "text-2xl font-bold text-slate-100", children: "Thread Outline" }), _jsx("p", { className: "text-slate-400", children: "Hierarchical view of agent threads and their dependencies" })] }), _jsxs("div", { className: "flex items-center justify-between bg-slate-900 border border-slate-800 rounded-lg px-4 py-3", children: [_jsx("div", { className: "flex items-center gap-4", children: _jsxs("div", { children: [_jsx("h3", { className: "text-sm font-semibold text-slate-300", children: getFilterLabel() }), _jsx("p", { className: "text-xs text-slate-500 mt-1", children: "Sorted by priority (highest first)" })] }) }), _jsxs("div", { className: "flex items-center gap-4 text-xs", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "w-2 h-2 rounded-full bg-slate-500" }), _jsxs("span", { className: "text-slate-400", children: [threadStats.total, " total"] })] }), threadStats.running > 0 && (_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "w-2 h-2 rounded-full bg-blue-500 animate-pulse" }), _jsxs("span", { className: "text-slate-400", children: [threadStats.running, " running"] })] })), threadStats.paused > 0 && (_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "w-2 h-2 rounded-full bg-amber-500" }), _jsxs("span", { className: "text-slate-400", children: [threadStats.paused, " paused"] })] })), threadStats.completed > 0 && (_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "w-2 h-2 rounded-full bg-emerald-500" }), _jsxs("span", { className: "text-slate-400", children: [threadStats.completed, " completed"] })] })), threadStats.failed > 0 && (_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "w-2 h-2 rounded-full bg-red-500" }), _jsxs("span", { className: "text-slate-400", children: [threadStats.failed, " failed"] })] }))] })] })] }), _jsxs("div", { className: "space-y-3 max-h-[calc(100vh-400px)] overflow-y-auto pr-2", children: [_jsx("style", { children: `
            .outline-view-scroll::-webkit-scrollbar {
              width: 8px;
            }
            .outline-view-scroll::-webkit-scrollbar-track {
              background: transparent;
            }
            .outline-view-scroll::-webkit-scrollbar-thumb {
              background: #475569;
              border-radius: 4px;
            }
            .outline-view-scroll::-webkit-scrollbar-thumb:hover {
              background: #64748b;
            }
          ` }), _jsx("div", { className: "outline-view-scroll space-y-3 max-h-[calc(100vh-400px)] overflow-y-auto pr-2", children: sortedThreads.map((thread) => (_jsx(ThreadCard, { thread: thread, isActive: activeThreadId === thread.id, onSelect: () => {
                                    // Selection handled by store within ThreadCard
                                } }, thread.id))) })] }), _jsx("div", { className: "text-xs text-slate-500 border-t border-slate-800 pt-4", children: _jsx("p", { children: "\uD83D\uDCA1 Tip: Click on a thread name to edit it. Click the expand button to see detailed steps and controls. Use the +/- buttons to adjust priority." }) })] }) }));
};
export default OutlineView;
//# sourceMappingURL=OutlineView.js.map