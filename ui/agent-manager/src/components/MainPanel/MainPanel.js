import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCallback } from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';
import { OutlineView } from '../OutlineView';
import { DetailPanel } from '../DetailPanel';
/**
 * MainPanel Component
 * Renders different views based on viewMode:
 * - outline: Agent thread outline view (Phase 3 - Implemented)
 * - tree: Hierarchical thread tree view (Phase 6 - Coming soon)
 * - swarm: Swarm visualization (Phase 7 - Coming soon)
 *
 * Also includes a slide-in DetailPanel for expanded thread view (Phase 4)
 */
export const MainPanel = () => {
    const { viewMode, threads, activeThreadId, setActiveThread } = useAgentManagerStore((state) => ({
        viewMode: state.viewMode,
        threads: state.getFilteredThreads(),
        activeThreadId: state.activeThreadId,
        setActiveThread: state.setActiveThread,
    }));
    // Handle detail panel close - clears active thread
    const handleDetailClose = useCallback(() => {
        setActiveThread(null);
    }, [setActiveThread]);
    const threadCount = threads.length;
    const renderView = () => {
        switch (viewMode) {
            case 'outline':
                return _jsx(OutlineView, {});
            case 'tree':
                return (_jsxs("div", { className: "p-8 space-y-6", children: [_jsxs("div", { className: "space-y-2", children: [_jsx("h2", { className: "text-2xl font-bold text-slate-100", children: "Thread Tree" }), _jsx("p", { className: "text-slate-400", children: "Dependency tree showing parent-child relationships" })] }), threadCount === 0 ? (_jsxs("div", { className: "flex flex-col items-center justify-center py-16 text-center", children: [_jsx("div", { className: "text-6xl mb-4", children: "\u22A2" }), _jsx("h3", { className: "text-lg font-semibold text-slate-300 mb-2", children: "No Threads Yet" }), _jsx("p", { className: "text-slate-500 max-w-md", children: "Create a new thread from the sidebar to see it appear in the tree view" })] })) : (_jsxs("div", { className: "bg-slate-800 border border-slate-700 rounded-lg p-6", children: [_jsxs("div", { className: "text-slate-400", children: [threadCount, " thread", threadCount !== 1 ? 's' : '', " loaded"] }), _jsx("p", { className: "text-slate-500 text-sm mt-2", children: "Tree rendering coming in Phase 3" })] }))] }));
            case 'swarm':
                return (_jsxs("div", { className: "p-8 space-y-6", children: [_jsxs("div", { className: "space-y-2", children: [_jsx("h2", { className: "text-2xl font-bold text-slate-100", children: "Swarm Visualization" }), _jsx("p", { className: "text-slate-400", children: "Force-directed graph of agent interactions and data flow" })] }), threadCount === 0 ? (_jsxs("div", { className: "flex flex-col items-center justify-center py-16 text-center", children: [_jsx("div", { className: "text-6xl mb-4", children: "\u25C6" }), _jsx("h3", { className: "text-lg font-semibold text-slate-300 mb-2", children: "No Threads Yet" }), _jsx("p", { className: "text-slate-500 max-w-md", children: "Create a new thread from the sidebar to see it appear in the swarm visualization" })] })) : (_jsxs("div", { className: "bg-slate-800 border border-slate-700 rounded-lg p-6", children: [_jsxs("div", { className: "text-slate-400", children: [threadCount, " thread", threadCount !== 1 ? 's' : '', " loaded"] }), _jsx("p", { className: "text-slate-500 text-sm mt-2", children: "Swarm visualization coming in Phase 3" })] }))] }));
            default:
                return null;
        }
    };
    return (_jsxs("div", { className: "min-h-full bg-slate-950 flex", children: [_jsx("div", { className: `flex-1 transition-all duration-300 ease-out ${activeThreadId ? 'mr-[500px]' : ''}`, children: renderView() }), _jsx("div", { className: `
          fixed top-0 right-0 h-full w-[500px] z-50
          transform transition-transform duration-300 ease-out
          ${activeThreadId ? 'translate-x-0' : 'translate-x-full'}
        `, children: activeThreadId && (_jsx(DetailPanel, { threadId: activeThreadId, onClose: handleDetailClose })) }), activeThreadId && (_jsx("div", { className: "fixed inset-0 bg-black/50 z-40 lg:hidden", onClick: handleDetailClose, "aria-hidden": "true" }))] }));
};
export default MainPanel;
//# sourceMappingURL=MainPanel.js.map