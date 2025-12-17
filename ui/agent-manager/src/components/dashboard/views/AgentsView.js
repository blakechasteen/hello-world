import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useAgents } from '@stores/appStore';
/**
 * Agents View
 *
 * Displays:
 * - Detailed agent list with type/state/metrics
 * - Agent filtering and search
 * - Individual agent controls
 * - Agent performance graphs
 */
export default function AgentsView() {
    const agents = useAgents();
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { children: [_jsx("h1", { className: "text-3xl font-bold text-text-primary", children: "Agents" }), _jsx("p", { className: "text-text-secondary mt-1", children: "Manage and monitor individual agents in the swarm" })] }), _jsxs("div", { className: "card p-6", children: [_jsxs("h2", { className: "text-xl font-semibold text-text-primary mb-4", children: ["Agent List (", agents.length, ")"] }), agents.length === 0 ? (_jsxs("div", { className: "text-center py-12 text-text-secondary", children: [_jsx("p", { className: "text-lg mb-2", children: "No agents connected" }), _jsx("p", { className: "text-sm", children: "Agents will appear here when they connect to the backend" })] })) : (_jsx("div", { className: "space-y-3", children: agents.map((agent) => (_jsxs("div", { className: "card-interactive p-4 flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center gap-4 flex-1 min-w-0", children: [_jsx("span", { className: "text-2xl flex-shrink-0", children: agent.type === 'query'
                                                ? '❓'
                                                : agent.type === 'process'
                                                    ? '⚙️'
                                                    : agent.type === 'memory'
                                                        ? '🧠'
                                                        : agent.type === 'decision'
                                                            ? '🎯'
                                                            : agent.type === 'output'
                                                                ? '📤'
                                                                : '🔄' }), _jsxs("div", { className: "min-w-0 flex-1", children: [_jsx("div", { className: "font-semibold text-text-primary truncate", children: agent.name }), _jsxs("div", { className: "text-xs text-text-tertiary", children: [agent.type, " \u2022 ", agent.reasoningMode] })] })] }), _jsxs("div", { className: "flex items-center gap-4 flex-shrink-0", children: [_jsxs("div", { className: "text-right text-sm", children: [_jsxs("div", { className: "text-text-primary font-semibold", children: [(agent.confidence * 100).toFixed(0), "%"] }), _jsx("div", { className: "text-xs text-text-tertiary", children: "confidence" })] }), _jsx("span", { className: `badge-agent-${agent.state}`, children: agent.state.charAt(0).toUpperCase() + agent.state.slice(1) })] })] }, agent.id))) }))] }), _jsxs("div", { className: "card p-6", children: [_jsx("h2", { className: "text-xl font-semibold text-text-primary mb-4", children: "Detailed View" }), _jsx("div", { className: "text-center py-12 text-text-secondary", children: _jsx("p", { children: "\uD83D\uDCCA Agent details and graphs coming soon" }) })] })] }));
}
//# sourceMappingURL=AgentsView.js.map