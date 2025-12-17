import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Suspense, lazy } from 'react';
import { useCurrentView } from '@stores/appStore';
/**
 * Dashboard Component
 *
 * Routes between different views:
 * - Overview: System-wide metrics and quick stats
 * - Agents: Detailed agent management
 * - Tasks: Task queue and execution tracking
 * - Metrics: Performance and historical data
 * - Logs: System logs and diagnostics
 * - Settings: Configuration options
 *
 * Uses code splitting for better performance
 */
// Lazy load view components
const OverviewView = lazy(() => import('./views/OverviewView'));
const AgentsView = lazy(() => import('./views/AgentsView'));
const TasksView = lazy(() => import('./views/TasksView'));
const MetricsView = lazy(() => import('./views/MetricsView'));
const LogsView = lazy(() => import('./views/LogsView'));
const SettingsView = lazy(() => import('./views/SettingsView'));
const LoadingSpinner = () => (_jsx("div", { className: "flex items-center justify-center h-full", children: _jsxs("div", { className: "text-center space-y-4", children: [_jsx("div", { className: "inline-block", children: _jsx("div", { className: "w-12 h-12 border-4 border-surface-tertiary border-t-holo-primary rounded-full animate-spin" }) }), _jsx("p", { className: "text-text-secondary", children: "Loading..." })] }) }));
export default function Dashboard() {
    const currentView = useCurrentView();
    const renderView = () => {
        switch (currentView) {
            case 'overview':
                return _jsx(OverviewView, {});
            case 'agents':
                return _jsx(AgentsView, {});
            case 'tasks':
                return _jsx(TasksView, {});
            case 'metrics':
                return _jsx(MetricsView, {});
            case 'logs':
                return _jsx(LogsView, {});
            case 'settings':
                return _jsx(SettingsView, {});
            default:
                return _jsx(OverviewView, {});
        }
    };
    return (_jsx("main", { className: "flex-1 overflow-y-auto bg-surface-primary", children: _jsx(Suspense, { fallback: _jsx(LoadingSpinner, {}), children: _jsx("div", { className: "p-6 min-h-full", children: renderView() }) }) }));
}
//# sourceMappingURL=Dashboard.js.map