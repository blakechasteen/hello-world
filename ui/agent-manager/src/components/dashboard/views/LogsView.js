import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { useRecentLogs, useErrorLogs } from '@stores/appStore';
/**
 * Logs View
 *
 * Displays:
 * - System logs with filtering
 * - Error tracking and diagnostics
 * - Log search and export
 */
export default function LogsView() {
    const logs = useRecentLogs(100);
    const errorLogs = useErrorLogs();
    const [logLevel, setLogLevel] = useState('all');
    const filteredLogs = logLevel === 'all'
        ? logs
        : logs.filter((log) => log.level === logLevel);
    const levelColors = {
        info: 'text-blue-400',
        success: 'text-emerald-400',
        warning: 'text-amber-400',
        error: 'text-red-400',
    };
    const levelIcons = {
        info: 'ℹ️',
        success: '✅',
        warning: '⚠️',
        error: '❌',
    };
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { children: [_jsx("h1", { className: "text-3xl font-bold text-text-primary", children: "System Logs" }), _jsx("p", { className: "text-text-secondary mt-1", children: "Real-time logging and diagnostics" })] }), _jsxs("div", { className: "grid grid-cols-2 md:grid-cols-4 gap-4", children: [_jsxs("div", { className: "card p-4 text-center", children: [_jsx("div", { className: "text-2xl font-bold text-blue-400", children: logs.filter((l) => l.level === 'info').length }), _jsx("div", { className: "text-xs text-text-secondary mt-1", children: "Info" })] }), _jsxs("div", { className: "card p-4 text-center", children: [_jsx("div", { className: "text-2xl font-bold text-emerald-400", children: logs.filter((l) => l.level === 'success').length }), _jsx("div", { className: "text-xs text-text-secondary mt-1", children: "Success" })] }), _jsxs("div", { className: "card p-4 text-center", children: [_jsx("div", { className: "text-2xl font-bold text-amber-400", children: logs.filter((l) => l.level === 'warning').length }), _jsx("div", { className: "text-xs text-text-secondary mt-1", children: "Warning" })] }), _jsxs("div", { className: "card p-4 text-center", children: [_jsx("div", { className: "text-2xl font-bold text-red-400", children: errorLogs.length }), _jsx("div", { className: "text-xs text-text-secondary mt-1", children: "Error" })] })] }), _jsxs("div", { className: "card p-6", children: [_jsxs("div", { className: "flex items-center gap-4 mb-4", children: [_jsx("label", { className: "text-sm font-medium text-text-secondary", children: "Filter by level:" }), _jsxs("select", { value: logLevel, onChange: (e) => setLogLevel(e.target.value), className: "bg-surface-tertiary border border-dark-light rounded px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-holo-primary", children: [_jsx("option", { value: "all", children: "All Levels" }), _jsx("option", { value: "info", children: "Info" }), _jsx("option", { value: "success", children: "Success" }), _jsx("option", { value: "warning", children: "Warning" }), _jsx("option", { value: "error", children: "Error" })] })] }), _jsx("div", { className: "space-y-2 max-h-96 overflow-y-auto", children: filteredLogs.length === 0 ? (_jsx("div", { className: "text-center py-8 text-text-tertiary", children: "No logs found" })) : (filteredLogs.map((log) => (_jsx("div", { className: "bg-surface-tertiary rounded px-4 py-3 text-sm border-l-4 border-surface-tertiary hover:border-holo-primary transition-colors", children: _jsxs("div", { className: "flex items-start gap-3", children: [_jsx("span", { className: "text-lg flex-shrink-0", children: levelIcons[log.level] || '📝' }), _jsxs("div", { className: "min-w-0 flex-1", children: [_jsxs("div", { className: "flex items-center gap-2 flex-wrap", children: [_jsx("span", { className: `font-semibold ${levelColors[log.level]}`, children: log.level.toUpperCase() }), _jsx("span", { className: "text-text-secondary", children: log.source }), _jsx("span", { className: "text-text-tertiary text-xs", children: new Date(log.timestamp).toLocaleTimeString() })] }), _jsx("div", { className: "text-text-primary mt-1 break-words", children: log.message })] })] }) }, log.id)))) })] })] }));
}
//# sourceMappingURL=LogsView.js.map