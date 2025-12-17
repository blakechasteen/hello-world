import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { useAgents, useSidebarOpen, useAppStore } from '@stores/appStore';
/**
 * Sidebar Component
 *
 * Displays:
 * - List of active agents with state badges
 * - Agent filtering by type/state
 * - Collapse/expand toggle
 * - Agent quick actions
 */
export default function Sidebar() {
    const agents = useAgents();
    const sidebarOpen = useSidebarOpen();
    const { setSidebarOpen, setSelectedAgent } = useAppStore();
    const [filterState, setFilterState] = useState('all');
    const filteredAgents = filterState === 'all'
        ? agents
        : agents.filter((agent) => agent.state === filterState);
    const _stateColors = {
        idle: 'bg-gray-900 text-gray-300',
        ready: 'bg-emerald-950 text-emerald-200',
        running: 'bg-cyan-950 text-cyan-200',
        success: 'bg-emerald-900 text-emerald-100',
        warning: 'bg-amber-950 text-amber-200',
        error: 'bg-red-950 text-red-200',
        paused: 'bg-orange-950 text-orange-200',
    };
    const stateLabels = {
        idle: 'Idle',
        ready: 'Ready',
        running: 'Running',
        success: 'Success',
        warning: 'Warning',
        error: 'Error',
        paused: 'Paused',
    };
    return (_jsxs("aside", { className: `bg-surface-secondary border-r border-dark-light transition-all duration-300 overflow-y-auto flex flex-col ${sidebarOpen ? 'w-64' : 'w-16'}`, children: [_jsx("button", { onClick: () => setSidebarOpen(!sidebarOpen), className: "p-4 hover:bg-surface-tertiary transition-colors", title: sidebarOpen ? 'Collapse' : 'Expand', children: _jsx("span", { className: "text-xl", children: sidebarOpen ? '◀️' : '▶️' }) }), sidebarOpen && (_jsxs("div", { className: "flex-1 flex flex-col p-4 gap-4 min-h-0", children: [_jsxs("div", { className: "space-y-2", children: [_jsx("h3", { className: "text-xs font-semibold text-text-secondary uppercase tracking-wider", children: "Filter" }), _jsxs("select", { value: filterState, onChange: (e) => setFilterState(e.target.value), className: "w-full bg-surface-tertiary border border-dark-light rounded px-2 py-1 text-xs text-text-primary focus:outline-none focus:ring-2 focus:ring-holo-primary", children: [_jsx("option", { value: "all", children: "All Agents" }), _jsx("option", { value: "idle", children: "Idle" }), _jsx("option", { value: "ready", children: "Ready" }), _jsx("option", { value: "running", children: "Running" }), _jsx("option", { value: "success", children: "Success" }), _jsx("option", { value: "error", children: "Error" })] })] }), _jsxs("div", { className: "space-y-2 flex-1 overflow-y-auto", children: [_jsxs("h3", { className: "text-xs font-semibold text-text-secondary uppercase tracking-wider", children: ["Agents (", filteredAgents.length, ")"] }), filteredAgents.length === 0 ? (_jsx("div", { className: "text-xs text-text-tertiary text-center py-4", children: "No agents found" })) : (_jsx("div", { className: "space-y-2", children: filteredAgents.map((agent) => (_jsxs("button", { onClick: () => setSelectedAgent(agent.id), className: "card-interactive w-full p-3 text-left group", children: [_jsxs("div", { className: "flex items-start gap-2 min-w-0", children: [_jsx("span", { className: "text-lg flex-shrink-0 mt-0.5", children: agent.type === 'query'
                                                        ? '❓'
                                                        : agent.type === 'process'
                                                            ? '⚙️'
                                                            : agent.type === 'memory'
                                                                ? '🧠'
                                                                : agent.type === 'decision'
                                                                    ? '🎯'
                                                                    : agent.type === 'output'
                                                                        ? '📤'
                                                                        : '🔄' }), _jsxs("div", { className: "min-w-0 flex-1", children: [_jsx("div", { className: "text-sm font-medium text-text-primary truncate", children: agent.name }), _jsx("div", { className: "text-xs text-text-tertiary", children: agent.type.charAt(0).toUpperCase() +
                                                                agent.type.slice(1) })] })] }), _jsxs("div", { className: "mt-2 flex items-center gap-2", children: [_jsx("span", { className: `badge-agent-${agent.state}`, children: stateLabels[agent.state] }), _jsxs("span", { className: "text-xs text-text-tertiary", children: [(agent.confidence * 100).toFixed(0), "%"] })] })] }, agent.id))) }))] }), _jsxs("div", { className: "border-t border-dark-light pt-4 space-y-2 text-xs", children: [_jsxs("div", { className: "flex justify-between text-text-secondary", children: [_jsx("span", { children: "Total Agents:" }), _jsx("span", { className: "font-semibold", children: agents.length })] }), _jsxs("div", { className: "flex justify-between text-text-secondary", children: [_jsx("span", { children: "Running:" }), _jsx("span", { className: "font-semibold text-cyan-400", children: agents.filter((a) => a.state === 'running').length })] }), _jsxs("div", { className: "flex justify-between text-text-secondary", children: [_jsx("span", { children: "Errors:" }), _jsx("span", { className: "font-semibold text-red-400", children: agents.filter((a) => a.state === 'error').length })] })] })] }))] }));
}
//# sourceMappingURL=Sidebar.js.map