import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React from 'react';
export const Sidebar = () => {
    const [config, setConfig] = React.useState({
        agentType: 'weaving',
        reasoningMode: 'DIRECT',
        tokenBudget: undefined,
        priority: 5,
    });
    const agentTypes = [
        { value: 'weaving', label: 'Weaving', icon: '◆' },
        { value: 'rag', label: 'RAG', icon: '🔍' },
        { value: 'agentic', label: 'Agentic', icon: '🤖' },
        { value: 'custom', label: 'Custom', icon: '⚙' },
    ];
    const reasoningModes = [
        { value: 'DIRECT', label: 'Direct', description: '~150ms' },
        { value: 'VERIFY', label: 'Verify', description: '~600ms' },
        { value: 'RESEARCH', label: 'Research', description: '~900ms' },
        {
            value: 'PLAN_EXECUTE',
            label: 'Plan & Execute',
            description: '~750ms',
        },
    ];
    const handleCreateThread = () => {
        console.log('Creating thread with config:', config);
        // TODO: Implement thread creation via WebSocket
    };
    return (_jsxs("div", { className: "flex flex-col h-full bg-slate-900 border-r border-slate-800 p-4 gap-6 overflow-y-auto", children: [_jsxs("div", { className: "space-y-3", children: [_jsx("div", { className: "px-2 py-1.5", children: _jsx("label", { className: "block text-xs font-semibold text-slate-300 uppercase tracking-wide", children: "Agent Type" }) }), _jsx("div", { className: "space-y-2", children: agentTypes.map((type) => (_jsxs("label", { className: "flex items-center gap-3 p-2.5 rounded-md cursor-pointer hover:bg-slate-800 transition-colors group", children: [_jsx("input", { type: "radio", name: "agent-type", value: type.value, checked: config.agentType === type.value, onChange: (e) => setConfig({ ...config, agentType: e.target.value }), className: "w-4 h-4 accent-emerald-600 cursor-pointer" }), _jsxs("div", { className: "flex-1 flex items-center gap-2", children: [_jsx("span", { className: "text-lg", children: type.icon }), _jsx("span", { className: "text-sm font-medium text-slate-300 group-hover:text-slate-100", children: type.label })] })] }, type.value))) })] }), _jsx("div", { className: "h-px bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800" }), _jsxs("div", { className: "space-y-3", children: [_jsx("div", { className: "px-2 py-1.5", children: _jsx("label", { className: "block text-xs font-semibold text-slate-300 uppercase tracking-wide", children: "Reasoning Mode" }) }), _jsx("div", { className: "space-y-2", children: reasoningModes.map((mode) => (_jsxs("label", { className: "flex items-center gap-3 p-2.5 rounded-md cursor-pointer hover:bg-slate-800 transition-colors group", children: [_jsx("input", { type: "radio", name: "reasoning-mode", value: mode.value, checked: config.reasoningMode === mode.value, onChange: (e) => setConfig({
                                        ...config,
                                        reasoningMode: e.target.value,
                                    }), className: "w-4 h-4 accent-emerald-600 cursor-pointer" }), _jsxs("div", { className: "flex-1 flex flex-col gap-0.5", children: [_jsx("span", { className: "text-sm font-medium text-slate-300 group-hover:text-slate-100", children: mode.label }), _jsx("span", { className: "text-xs text-slate-500 group-hover:text-slate-400", children: mode.description })] })] }, mode.value))) })] }), _jsx("div", { className: "h-px bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800" }), _jsxs("div", { className: "space-y-3", children: [_jsx("div", { className: "px-2 py-1.5", children: _jsx("label", { htmlFor: "token-budget", className: "block text-xs font-semibold text-slate-300 uppercase tracking-wide", children: "Token Budget (optional)" }) }), _jsx("input", { id: "token-budget", type: "number", min: "100", step: "100", placeholder: "e.g., 2000", value: config.tokenBudget ?? '', onChange: (e) => setConfig({
                            ...config,
                            tokenBudget: e.target.value ? parseInt(e.target.value) : undefined,
                        }), className: "w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-md text-slate-100 placeholder-slate-600 text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors" }), _jsx("p", { className: "text-xs text-slate-500 px-2", children: "Leave empty for unlimited" })] }), _jsx("div", { className: "h-px bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800" }), _jsxs("div", { className: "space-y-3", children: [_jsxs("div", { className: "px-2 py-1.5 flex items-center justify-between", children: [_jsx("label", { htmlFor: "priority", className: "block text-xs font-semibold text-slate-300 uppercase tracking-wide", children: "Priority" }), _jsx("span", { className: "text-sm font-semibold text-emerald-400", children: config.priority })] }), _jsx("div", { className: "px-1 py-2", children: _jsx("input", { id: "priority", type: "range", min: "1", max: "10", value: config.priority, onChange: (e) => setConfig({ ...config, priority: parseInt(e.target.value) }), className: "w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-600" }) }), _jsxs("div", { className: "flex justify-between text-xs text-slate-500 px-2", children: [_jsx("span", { children: "Low" }), _jsx("span", { children: "High" })] })] }), _jsx("div", { className: "flex-1" }), _jsxs("button", { onClick: handleCreateThread, className: "w-full px-4 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-md transition-colors flex items-center justify-center gap-2 text-sm", children: [_jsx("span", { children: "\u2192" }), _jsx("span", { children: "Create Thread" })] }), _jsx("p", { className: "text-xs text-slate-500 text-center px-2", children: "WebSocket connection required" })] }));
};
export default Sidebar;
//# sourceMappingURL=Sidebar.js.map