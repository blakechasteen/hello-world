import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { useAutoRefresh, useRefreshInterval, useAppStore } from '@stores/appStore';
/**
 * Settings View
 *
 * Configurable options for:
 * - Auto-refresh interval
 * - Display preferences
 * - Data retention
 * - Backend connection settings
 */
export default function SettingsView() {
    const autoRefresh = useAutoRefresh();
    const refreshInterval = useRefreshInterval();
    const { setAutoRefresh, setRefreshInterval } = useAppStore();
    const [settings, setSettings] = useState({
        autoRefresh,
        refreshInterval,
        theme: 'dark',
        compactMode: false,
        showNotifications: true,
        logRetention: 7, // days
    });
    const handleSave = () => {
        setAutoRefresh(settings.autoRefresh);
        setRefreshInterval(settings.refreshInterval);
        // Additional settings would be saved here
    };
    return (_jsxs("div", { className: "space-y-6 max-w-2xl", children: [_jsxs("div", { children: [_jsx("h1", { className: "text-3xl font-bold text-text-primary", children: "Settings" }), _jsx("p", { className: "text-text-secondary mt-1", children: "Configure dashboard behavior and preferences" })] }), _jsxs("div", { className: "card p-6 space-y-6", children: [_jsx("h2", { className: "text-xl font-semibold text-text-primary", children: "General" }), _jsxs("div", { className: "space-y-2", children: [_jsxs("label", { className: "flex items-center gap-3 cursor-pointer", children: [_jsx("input", { type: "checkbox", checked: settings.autoRefresh, onChange: (e) => setSettings({ ...settings, autoRefresh: e.target.checked }), className: "w-4 h-4 accent-holo-primary" }), _jsx("span", { className: "text-text-primary font-medium", children: "Auto-Refresh Data" })] }), _jsx("p", { className: "text-xs text-text-tertiary ml-7", children: "Automatically update metrics and logs" })] }), settings.autoRefresh && (_jsxs("div", { className: "space-y-2 pl-7 border-l border-dark-light", children: [_jsx("label", { className: "text-sm font-medium text-text-secondary", children: "Refresh Interval" }), _jsxs("select", { value: settings.refreshInterval, onChange: (e) => setSettings({
                                    ...settings,
                                    refreshInterval: parseInt(e.target.value),
                                }), className: "w-full bg-surface-tertiary border border-dark-light rounded px-3 py-2 text-text-primary focus:outline-none focus:ring-2 focus:ring-holo-primary", children: [_jsx("option", { value: 1000, children: "1 second" }), _jsx("option", { value: 5000, children: "5 seconds" }), _jsx("option", { value: 10000, children: "10 seconds" }), _jsx("option", { value: 30000, children: "30 seconds" }), _jsx("option", { value: 60000, children: "1 minute" })] })] })), _jsxs("div", { className: "space-y-2", children: [_jsxs("label", { className: "flex items-center gap-3 cursor-pointer", children: [_jsx("input", { type: "checkbox", checked: settings.compactMode, onChange: (e) => setSettings({ ...settings, compactMode: e.target.checked }), className: "w-4 h-4 accent-holo-primary" }), _jsx("span", { className: "text-text-primary font-medium", children: "Compact Mode" })] }), _jsx("p", { className: "text-xs text-text-tertiary ml-7", children: "Reduce padding and spacing in tables and lists" })] }), _jsxs("div", { className: "space-y-2", children: [_jsxs("label", { className: "flex items-center gap-3 cursor-pointer", children: [_jsx("input", { type: "checkbox", checked: settings.showNotifications, onChange: (e) => setSettings({
                                            ...settings,
                                            showNotifications: e.target.checked,
                                        }), className: "w-4 h-4 accent-holo-primary" }), _jsx("span", { className: "text-text-primary font-medium", children: "Show Notifications" })] }), _jsx("p", { className: "text-xs text-text-tertiary ml-7", children: "Display alerts for errors and important events" })] }), _jsxs("div", { className: "space-y-2 border-t border-dark-light pt-6", children: [_jsx("label", { className: "text-sm font-medium text-text-secondary", children: "Log Retention (days)" }), _jsx("input", { type: "number", min: "1", max: "90", value: settings.logRetention, onChange: (e) => setSettings({
                                    ...settings,
                                    logRetention: parseInt(e.target.value),
                                }), className: "w-full bg-surface-tertiary border border-dark-light rounded px-3 py-2 text-text-primary focus:outline-none focus:ring-2 focus:ring-holo-primary" }), _jsx("p", { className: "text-xs text-text-tertiary", children: "Logs older than this will be archived" })] })] }), _jsxs("div", { className: "card p-6 space-y-6", children: [_jsx("h2", { className: "text-xl font-semibold text-text-primary", children: "Backend" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "text-sm font-medium text-text-secondary block mb-1", children: "Backend URL" }), _jsx("input", { type: "text", value: "http://localhost:8000", disabled: true, className: "w-full bg-surface-tertiary border border-dark-light rounded px-3 py-2 text-text-secondary cursor-not-allowed opacity-75" }), _jsx("p", { className: "text-xs text-text-tertiary mt-1", children: "Currently configured in vite.config.ts" })] }), _jsxs("div", { children: [_jsx("label", { className: "text-sm font-medium text-text-secondary block mb-1", children: "Agent Manager URL" }), _jsx("input", { type: "text", value: "http://localhost:8002", disabled: true, className: "w-full bg-surface-tertiary border border-dark-light rounded px-3 py-2 text-text-secondary cursor-not-allowed opacity-75" }), _jsx("p", { className: "text-xs text-text-tertiary mt-1", children: "Currently configured in vite.config.ts" })] })] })] }), _jsxs("div", { className: "flex gap-4", children: [_jsx("button", { onClick: handleSave, className: "px-6 py-2 bg-holo-primary text-white rounded-lg font-medium hover:bg-holo-primary-light transition-colors shadow-glow-primary", children: "Save Settings" }), _jsx("button", { onClick: () => {
                            /* Reset to defaults */
                        }, className: "px-6 py-2 bg-surface-tertiary text-text-primary rounded-lg font-medium hover:bg-dark-light transition-colors", children: "Reset to Defaults" })] }), _jsx("div", { className: "card p-4 bg-surface-tertiary text-sm text-text-secondary", children: _jsx("p", { children: "\uD83D\uDCA1 Settings are stored in browser localStorage and will persist across sessions." }) })] }));
}
//# sourceMappingURL=SettingsView.js.map