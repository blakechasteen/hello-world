import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useSystemMetrics, useAgents, useTasks } from '@stores/appStore';
/**
 * Overview View
 *
 * Displays system-wide metrics and quick stats:
 * - Total agents / running / errors
 * - System latency and cache hit rate
 * - Recent activity feed
 * - Quick action buttons
 */
export default function OverviewView() {
    const systemMetrics = useSystemMetrics();
    const agents = useAgents();
    const tasks = useTasks();
    const stats = [
        { label: 'Total Agents', value: agents.length, color: 'holo-primary' },
        {
            label: 'Running',
            value: agents.filter((a) => a.state === 'running').length,
            color: 'holo-accent',
        },
        {
            label: 'Errors',
            value: agents.filter((a) => a.state === 'error').length,
            color: 'holo-danger',
        },
        { label: 'Pending Tasks', value: tasks.filter((t) => t.status === 'pending').length, color: 'holo-warning' },
    ];
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { children: [_jsx("h1", { className: "text-3xl font-bold text-text-primary", children: "Dashboard Overview" }), _jsx("p", { className: "text-text-secondary mt-1", children: "Real-time monitoring of HoloLoom agent system" })] }), _jsx("div", { className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4", children: stats.map((stat) => (_jsxs("div", { className: "card p-6", children: [_jsx("div", { className: "text-text-secondary text-sm font-medium mb-2", children: stat.label }), _jsx("div", { className: `text-3xl font-bold text-${stat.color}`, children: stat.value })] }, stat.label))) }), systemMetrics && (_jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-3 gap-6", children: [_jsxs("div", { className: "card p-6", children: [_jsx("div", { className: "text-text-secondary text-sm font-medium mb-4", children: "System Latency" }), _jsxs("div", { className: "text-4xl font-bold text-holo-accent", children: [systemMetrics.systemLatencyMs.toFixed(1), _jsx("span", { className: "text-lg ml-2", children: "ms" })] }), _jsx("div", { className: "text-xs text-text-tertiary mt-2", children: "Average response time" })] }), _jsxs("div", { className: "card p-6", children: [_jsx("div", { className: "text-text-secondary text-sm font-medium mb-4", children: "Cache Hit Rate" }), _jsxs("div", { className: "text-4xl font-bold text-holo-success", children: [(systemMetrics.cacheHitRate * 100).toFixed(1), _jsx("span", { className: "text-lg ml-2", children: "%" })] }), _jsx("div", { className: "text-xs text-text-tertiary mt-2", children: "Query cache efficiency" })] }), _jsxs("div", { className: "card p-6", children: [_jsx("div", { className: "text-text-secondary text-sm font-medium mb-4", children: "Avg Confidence" }), _jsxs("div", { className: "text-4xl font-bold text-holo-secondary", children: [(systemMetrics.avgConfidence * 100).toFixed(1), _jsx("span", { className: "text-lg ml-2", children: "%" })] }), _jsx("div", { className: "text-xs text-text-tertiary mt-2", children: "System-wide confidence" })] })] })), _jsxs("div", { className: "card p-6", children: [_jsx("h2", { className: "text-xl font-semibold text-text-primary mb-4", children: "Recent Activity" }), _jsxs("div", { className: "space-y-2 text-center text-text-secondary", children: [_jsx("p", { children: "\uD83D\uDE80 Dashboard views in development" }), _jsx("p", { className: "text-sm", children: "Detailed charts and activity feed coming soon" })] })] })] }));
}
//# sourceMappingURL=OverviewView.js.map