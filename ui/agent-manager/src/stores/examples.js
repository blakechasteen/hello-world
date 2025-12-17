import { jsxs as _jsxs, jsx as _jsx, Fragment as _Fragment } from "react/jsx-runtime";
/**
 * Agent Manager Store Usage Examples
 * Real-world examples of how to use the store in React components
 */
import { useCallback, useMemo } from 'react';
import { useAgentManagerStore, useThreadWithDependencies, useSwarmOverview, useActiveThreadDetails, } from './index';
// ============================================================================
// EXAMPLE 1: Thread List Component with Filtering
// ============================================================================
export function ThreadList() {
    // Subscribe to filtered threads and filter action
    const threads = useAgentManagerStore((state) => state.getFilteredThreads());
    const filter = useAgentManagerStore((state) => state.filter);
    const setFilter = useAgentManagerStore((state) => state.setFilter);
    const setActiveThread = useAgentManagerStore((state) => state.setActiveThread);
    return (_jsxs("div", { className: "thread-list", children: [_jsxs("div", { className: "filter-bar", children: [_jsxs("button", { className: filter === 'all' ? 'active' : '', onClick: () => setFilter('all'), children: ["All (", Object.keys(useAgentManagerStore.getState().threads).length, ")"] }), _jsx("button", { className: filter === 'active' ? 'active' : '', onClick: () => setFilter('active'), children: "Active" }), _jsx("button", { className: filter === 'completed' ? 'active' : '', onClick: () => setFilter('completed'), children: "Completed" }), _jsx("button", { className: filter === 'failed' ? 'active' : '', onClick: () => setFilter('failed'), children: "Failed" })] }), _jsx("ul", { className: "thread-items", children: threads.map((thread) => (_jsxs("li", { className: "thread-item", onClick: () => setActiveThread(thread.id), children: [_jsx("div", { className: "thread-name", children: thread.name }), _jsx("div", { className: "thread-status", children: thread.status }), _jsxs("div", { className: "thread-progress", children: [thread.currentStep, " / ", thread.totalSteps] })] }, thread.id))) })] }));
}
// ============================================================================
// EXAMPLE 2: Thread Detail Card with Dependencies
// ============================================================================
export function ThreadDetailCard({ threadId }) {
    const { thread, dependencies, children } = useThreadWithDependencies(threadId);
    if (!thread) {
        return _jsx("div", { className: "error", children: "Thread not found" });
    }
    const progressPercent = (thread.currentStep / thread.totalSteps) * 100;
    return (_jsxs("div", { className: "thread-detail-card", children: [_jsxs("header", { children: [_jsx("h3", { children: thread.name }), _jsx("span", { className: `status status-${thread.status}`, children: thread.status })] }), _jsxs("section", { className: "metrics", children: [_jsxs("div", { className: "metric", children: [_jsx("label", { children: "Progress" }), _jsx("progress", { value: progressPercent, max: 100 }), _jsxs("span", { children: [thread.currentStep, "/", thread.totalSteps, " steps"] })] }), _jsxs("div", { className: "metric", children: [_jsx("label", { children: "Confidence" }), _jsx("div", { className: "confidence-bar", children: _jsx("div", { className: "confidence-fill", style: { width: `${thread.confidence * 100}%` } }) }), _jsxs("span", { children: [(thread.confidence * 100).toFixed(1), "%"] })] }), _jsxs("div", { className: "metric", children: [_jsx("label", { children: "Epistemic Confidence" }), _jsxs("span", { children: [(thread.epistemicConfidence * 100).toFixed(1), "%"] })] }), _jsxs("div", { className: "metric", children: [_jsx("label", { children: "Tokens" }), _jsxs("span", { children: [thread.tokensUsed, " / ", thread.tokenBudget ?? '∞'] })] }), _jsxs("div", { className: "metric", children: [_jsx("label", { children: "Time" }), _jsxs("span", { children: [(thread.elapsedTimeMs / 1000).toFixed(2), "s"] })] })] }), _jsxs("section", { className: "dependencies", children: [_jsx("h4", { children: "Dependencies" }), dependencies.dependsOn.length > 0 ? (_jsx("ul", { children: dependencies.dependsOn.map((dep) => (_jsx("li", { children: dep.name }, dep.id))) })) : (_jsx("p", { className: "empty", children: "No dependencies" }))] }), _jsxs("section", { className: "blocking", children: [_jsx("h4", { children: "Blocking" }), dependencies.blocks.length > 0 ? (_jsx("ul", { children: dependencies.blocks.map((block) => (_jsx("li", { children: block.name }, block.id))) })) : (_jsx("p", { className: "empty", children: "Not blocking anything" }))] }), _jsxs("section", { className: "children", children: [_jsxs("h4", { children: ["Child Threads (", children.length, ")"] }), children.length > 0 ? (_jsx("ul", { children: children.map((child) => (_jsx("li", { children: child.name }, child.id))) })) : (_jsx("p", { className: "empty", children: "No child threads" }))] })] }));
}
// ============================================================================
// EXAMPLE 3: Active Thread Panel with Controls
// ============================================================================
export function ActiveThreadPanel() {
    const { thread, dependencies, children } = useActiveThreadDetails();
    const pauseThread = useAgentManagerStore((state) => state.pauseThread);
    const resumeThread = useAgentManagerStore((state) => state.resumeThread);
    const cancelThread = useAgentManagerStore((state) => state.cancelThread);
    const upvoteThread = useAgentManagerStore((state) => state.upvoteThread);
    const downvoteThread = useAgentManagerStore((state) => state.downvoteThread);
    if (!thread) {
        return (_jsx("div", { className: "empty-state", children: _jsx("p", { children: "Select a thread to view details" }) }));
    }
    const handlePauseResume = useCallback(() => {
        if (thread.status === 'running') {
            pauseThread(thread.id);
        }
        else if (thread.status === 'paused') {
            resumeThread(thread.id);
        }
    }, [thread.id, thread.status, pauseThread, resumeThread]);
    const handleCancel = useCallback(() => {
        if (confirm(`Cancel thread "${thread.name}"?`)) {
            cancelThread(thread.id);
        }
    }, [thread.id, thread.name, cancelThread]);
    return (_jsxs("div", { className: "active-thread-panel", children: [_jsxs("header", { children: [_jsx("h2", { children: thread.name }), _jsx("div", { className: "controls", children: (thread.status === 'running' || thread.status === 'paused') && (_jsxs(_Fragment, { children: [_jsx("button", { onClick: handlePauseResume, children: thread.status === 'running' ? 'Pause' : 'Resume' }), _jsx("button", { onClick: handleCancel, className: "danger", children: "Cancel" })] })) })] }), _jsxs("section", { className: "priority-control", children: [_jsxs("label", { children: ["Priority: ", thread.priority] }), _jsxs("div", { className: "priority-buttons", children: [_jsx("button", { onClick: () => downvoteThread(thread.id), children: "\u2212" }), _jsx("div", { className: "priority-bar", children: _jsx("div", { className: "priority-fill", style: { width: `${(thread.priority / 100) * 100}%` } }) }), _jsx("button", { onClick: () => upvoteThread(thread.id), children: "+" })] })] }), _jsxs("section", { className: "info", children: [_jsxs("div", { className: "info-row", children: [_jsx("span", { className: "label", children: "Type:" }), _jsx("span", { className: "value", children: thread.agentType })] }), _jsxs("div", { className: "info-row", children: [_jsx("span", { className: "label", children: "Mode:" }), _jsx("span", { className: "value", children: thread.reasoningMode })] }), _jsxs("div", { className: "info-row", children: [_jsx("span", { className: "label", children: "Status:" }), _jsx("span", { className: `status status-${thread.status}`, children: thread.status })] }), _jsxs("div", { className: "info-row", children: [_jsx("span", { className: "label", children: "Progress:" }), _jsxs("span", { className: "value", children: [thread.currentStep, " / ", thread.totalSteps] })] }), _jsxs("div", { className: "info-row", children: [_jsx("span", { className: "label", children: "Confidence:" }), _jsxs("span", { className: "value", children: [(thread.confidence * 100).toFixed(1), "%"] })] }), _jsxs("div", { className: "info-row", children: [_jsx("span", { className: "label", children: "Epistemic:" }), _jsxs("span", { className: "value", children: [(thread.epistemicConfidence * 100).toFixed(1), "%"] })] })] }), thread.finalResponse && (_jsxs("section", { className: "response", children: [_jsx("h3", { children: "Response" }), _jsx("p", { children: thread.finalResponse })] }))] }));
}
// ============================================================================
// EXAMPLE 4: Swarm Dashboard
// ============================================================================
export function SwarmDashboard({ swarmId }) {
    const { threads, status } = useSwarmOverview(swarmId);
    const threadsByStatus = useMemo(() => ({
        running: threads.filter((t) => t.status === 'running').length,
        paused: threads.filter((t) => t.status === 'paused').length,
        completed: threads.filter((t) => t.status === 'completed').length,
        failed: threads.filter((t) => t.status === 'failed' || t.status === 'cancelled')
            .length,
    }), [threads]);
    const totalTokensUsed = useMemo(() => threads.reduce((sum, t) => sum + t.tokensUsed, 0), [threads]);
    const totalTimeMs = useMemo(() => Math.max(...threads.map((t) => t.elapsedTimeMs), 0), [threads]);
    return (_jsxs("div", { className: "swarm-dashboard", children: [_jsxs("h1", { children: ["Swarm: ", swarmId] }), _jsxs("section", { className: "summary", children: [_jsxs("div", { className: "stat", children: [_jsx("label", { children: "Total Threads" }), _jsx("div", { className: "value", children: status.total })] }), _jsxs("div", { className: "stat", children: [_jsx("label", { children: "Running" }), _jsx("div", { className: "value", children: status.running })] }), _jsxs("div", { className: "stat", children: [_jsx("label", { children: "Completed" }), _jsx("div", { className: "value", children: status.completed })] }), _jsxs("div", { className: "stat", children: [_jsx("label", { children: "Failed" }), _jsx("div", { className: "value", children: status.failed })] }), _jsxs("div", { className: "stat", children: [_jsx("label", { children: "Avg Confidence" }), _jsxs("div", { className: "value", children: [(status.avgConfidence * 100).toFixed(1), "%"] })] })] }), _jsxs("section", { className: "thread-breakdown", children: [_jsx("h3", { children: "Thread Status" }), _jsxs("div", { className: "status-grid", children: [_jsxs("div", { className: "status-item running", children: [_jsx("span", { className: "count", children: threadsByStatus.running }), _jsx("span", { className: "label", children: "Running" })] }), _jsxs("div", { className: "status-item paused", children: [_jsx("span", { className: "count", children: threadsByStatus.paused }), _jsx("span", { className: "label", children: "Paused" })] }), _jsxs("div", { className: "status-item completed", children: [_jsx("span", { className: "count", children: threadsByStatus.completed }), _jsx("span", { className: "label", children: "Completed" })] }), _jsxs("div", { className: "status-item failed", children: [_jsx("span", { className: "count", children: threadsByStatus.failed }), _jsx("span", { className: "label", children: "Failed" })] })] })] }), _jsxs("section", { className: "resource-usage", children: [_jsx("h3", { children: "Resource Usage" }), _jsxs("div", { className: "resource-row", children: [_jsx("label", { children: "Total Tokens" }), _jsx("span", { children: totalTokensUsed.toLocaleString() })] }), _jsxs("div", { className: "resource-row", children: [_jsx("label", { children: "Total Time" }), _jsxs("span", { children: [(totalTimeMs / 1000).toFixed(2), "s"] })] })] }), _jsxs("section", { className: "thread-list", children: [_jsxs("h3", { children: ["Threads (", threads.length, ")"] }), _jsx("ul", { children: threads.map((thread) => (_jsxs("li", { className: `thread-row status-${thread.status}`, children: [_jsx("span", { className: "name", children: thread.name }), _jsx("span", { className: "status", children: thread.status }), _jsxs("span", { className: "progress", children: [thread.currentStep, "/", thread.totalSteps] }), _jsxs("span", { className: "confidence", children: [(thread.confidence * 100).toFixed(0), "%"] })] }, thread.id))) })] })] }));
}
// ============================================================================
// EXAMPLE 5: Thread Manager with Updates
// ============================================================================
export function ThreadManager() {
    const updateThread = useAgentManagerStore((state) => state.updateThread);
    const removeThread = useAgentManagerStore((state) => state.removeThread);
    const handleUpdateProgress = useCallback((threadId, step, elapsedMs) => {
        updateThread(threadId, {
            currentStep: step,
            elapsedTimeMs: elapsedMs,
        });
    }, [updateThread]);
    const handleComplete = useCallback((threadId, response, confidence) => {
        updateThread(threadId, {
            status: 'completed',
            finalResponse: response,
            confidence,
        });
    }, [updateThread]);
    const handleFail = useCallback((threadId, error) => {
        updateThread(threadId, {
            status: 'failed',
            finalResponse: `Error: ${error}`,
        });
    }, [updateThread]);
    return (_jsxs("div", { className: "thread-manager", children: [_jsx("p", { children: "Thread Manager - Used to update threads via WebSocket/API callbacks" }), _jsx("code", { children: JSON.stringify({
                    onUpdateProgress: 'handleUpdateProgress(threadId, step, time)',
                    onComplete: 'handleComplete(threadId, response, confidence)',
                    onFail: 'handleFail(threadId, error)',
                }, null, 2) })] }));
}
// ============================================================================
// EXAMPLE 6: Connection Status Indicator
// ============================================================================
export function ConnectionIndicator() {
    const isConnected = useAgentManagerStore((state) => state.isConnected);
    const connectionError = useAgentManagerStore((state) => state.connectionError);
    return (_jsxs("div", { className: `connection-indicator ${isConnected ? 'connected' : 'disconnected'}`, children: [_jsx("div", { className: "status-dot" }), _jsx("span", { className: "status-text", children: isConnected ? 'Connected' : `Disconnected: ${connectionError}` })] }));
}
// ============================================================================
// EXAMPLE 7: View Mode Switcher
// ============================================================================
export function ViewModeSwitcher() {
    const viewMode = useAgentManagerStore((state) => state.viewMode);
    const setViewMode = useAgentManagerStore((state) => state.setViewMode);
    return (_jsxs("div", { className: "view-mode-switcher", children: [_jsx("button", { className: viewMode === 'outline' ? 'active' : '', onClick: () => setViewMode('outline'), children: "Outline" }), _jsx("button", { className: viewMode === 'tree' ? 'active' : '', onClick: () => setViewMode('tree'), children: "Tree" }), _jsx("button", { className: viewMode === 'swarm' ? 'active' : '', onClick: () => setViewMode('swarm'), children: "Swarm" })] }));
}
//# sourceMappingURL=examples.js.map