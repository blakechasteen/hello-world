import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useCurrentView, useAppStore } from '@stores/appStore';
import { DashboardView } from '../../types';
/**
 * Header Component
 *
 * Top navigation bar with:
 * - HoloLoom branding
 * - View switcher (Overview, Agents, Tasks, Metrics, Logs)
 * - Connection status indicator
 * - Quick actions
 */
export default function Header({ isConnected }) {
    const currentView = useCurrentView();
    const { setCurrentView } = useAppStore();
    const views = [
        { id: DashboardView.OVERVIEW, label: 'Overview', icon: '📊' },
        { id: DashboardView.AGENTS, label: 'Agents', icon: '🤖' },
        { id: DashboardView.TASKS, label: 'Tasks', icon: '📋' },
        { id: DashboardView.METRICS, label: 'Metrics', icon: '📈' },
        { id: DashboardView.LOGS, label: 'Logs', icon: '📝' },
        { id: DashboardView.SETTINGS, label: 'Settings', icon: '⚙️' },
    ];
    return (_jsxs("header", { className: "bg-surface-secondary border-b border-dark-light px-6 py-4 shadow-dark-md", children: [_jsxs("div", { className: "flex items-center justify-between mb-4", children: [_jsxs("div", { className: "flex items-center gap-3", children: [_jsx("div", { className: "w-8 h-8 bg-gradient-to-br from-holo-primary to-holo-secondary rounded-lg flex items-center justify-center", children: _jsx("span", { className: "text-white font-bold", children: "HL" }) }), _jsx("h1", { className: "text-xl font-bold text-text-primary", children: "Agent Manager" })] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: `w-2 h-2 rounded-full ${isConnected ? 'bg-agent-success glow-success' : 'bg-agent-error'}` }), _jsx("span", { className: "text-xs text-text-secondary", children: isConnected ? 'Connected' : 'Disconnected' })] })] }), _jsx("nav", { className: "flex gap-1 overflow-x-auto", children: views.map((view) => (_jsxs("button", { onClick: () => setCurrentView(view.id), className: `px-3 py-2 rounded-md text-sm font-medium whitespace-nowrap transition-all ${currentView === view.id
                        ? 'bg-holo-primary text-white shadow-glow-primary'
                        : 'text-text-secondary hover:bg-surface-tertiary hover:text-text-primary'}`, children: [view.icon, " ", view.label] }, view.id))) })] }));
}
//# sourceMappingURL=Header.js.map