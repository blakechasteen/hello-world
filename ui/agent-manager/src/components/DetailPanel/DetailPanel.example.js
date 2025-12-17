import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * DetailPanel Examples
 * HoloLoom Agent Manager UI Phase 4 - DetailPanel Component Usage Examples
 *
 * This file demonstrates various usage patterns and integration scenarios
 * for the DetailPanel component.
 *
 * Run: STORYBOOK_DOCS_ENABLED=true npm run storybook
 * Or import examples into your test files
 */
import React, { useState } from 'react';
import { DetailPanel } from './DetailPanel';
import { useAgentManagerStore } from '../../stores/agentManagerStore';
/**
 * Example 1: Basic DetailPanel with Sample Data
 * Shows the most basic usage pattern
 */
export function BasicDetailPanelExample() {
    const [selectedThreadId, setSelectedThreadId] = useState('thread-1');
    const addThread = useAgentManagerStore((state) => state.addThread);
    // Initialize sample thread on mount
    React.useEffect(() => {
        const sampleThread = {
            id: 'thread-1',
            name: 'Query: Thompson Sampling Analysis',
            status: 'running',
            priority: 5,
            agentType: 'Researcher',
            reasoningMode: 'RESEARCH',
            currentStep: 3,
            totalSteps: 5,
            elapsedTimeMs: 2500,
            tokensUsed: 1250,
            tokenBudget: 5000,
            confidence: 0.75,
            epistemicConfidence: 0.82,
            childThreadIds: [],
            dependsOn: [],
            blocks: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        };
        addThread(sampleThread);
    }, [addThread]);
    return (_jsxs("div", { className: "flex h-screen bg-slate-950", children: [_jsxs("div", { className: "flex-1 border-r border-slate-700 p-4", children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 mb-4", children: "Threads" }), _jsx("button", { onClick: () => setSelectedThreadId('thread-1'), className: "w-full px-4 py-2 text-left rounded bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors", children: "Thompson Sampling Analysis" })] }), selectedThreadId && (_jsx("div", { className: "w-96", children: _jsx(DetailPanel, { threadId: selectedThreadId, onClose: () => setSelectedThreadId('') }) }))] }));
}
/**
 * Example 2: DetailPanel with Multiple Threads & Switching
 * Shows switching between different threads
 */
export function MultiThreadDetailPanelExample() {
    const [selectedThreadId, setSelectedThreadId] = useState('thread-1');
    const addThread = useAgentManagerStore((state) => state.addThread);
    // Initialize multiple sample threads
    React.useEffect(() => {
        const threads = [
            {
                id: 'thread-1',
                name: 'Query: Thompson Sampling Analysis',
                status: 'completed',
                priority: 5,
                agentType: 'Researcher',
                reasoningMode: 'RESEARCH',
                currentStep: 5,
                totalSteps: 5,
                elapsedTimeMs: 8500,
                tokensUsed: 2850,
                tokenBudget: 5000,
                confidence: 0.92,
                epistemicConfidence: 0.88,
                childThreadIds: ['thread-2', 'thread-3'],
                dependsOn: [],
                blocks: [],
                finalResponse: 'Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation...',
                createdAt: new Date(Date.now() - 15000).toISOString(),
                updatedAt: new Date(Date.now() - 1000).toISOString(),
            },
            {
                id: 'thread-2',
                name: 'Sub-query: Implementation Details',
                status: 'completed',
                priority: 4,
                agentType: 'Developer',
                reasoningMode: 'VERIFY',
                currentStep: 4,
                totalSteps: 4,
                elapsedTimeMs: 3200,
                tokensUsed: 1500,
                confidence: 0.85,
                epistemicConfidence: 0.79,
                childThreadIds: [],
                dependsOn: ['thread-1'],
                blocks: [],
                createdAt: new Date(Date.now() - 10000).toISOString(),
                updatedAt: new Date(Date.now() - 1500).toISOString(),
            },
            {
                id: 'thread-3',
                name: 'Sub-query: Performance Comparison',
                status: 'running',
                priority: 3,
                agentType: 'Analyst',
                reasoningMode: 'PLAN_EXECUTE',
                currentStep: 2,
                totalSteps: 4,
                elapsedTimeMs: 2100,
                tokensUsed: 950,
                confidence: 0.68,
                epistemicConfidence: 0.62,
                childThreadIds: [],
                dependsOn: ['thread-1'],
                blocks: [],
                createdAt: new Date(Date.now() - 5000).toISOString(),
                updatedAt: new Date().toISOString(),
            },
        ];
        threads.forEach((thread) => addThread(thread));
    }, [addThread]);
    return (_jsxs("div", { className: "flex h-screen bg-slate-950", children: [_jsxs("div", { className: "flex-1 border-r border-slate-700 p-4", children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 mb-4", children: "Threads" }), _jsx("div", { className: "space-y-2", children: ['thread-1', 'thread-2', 'thread-3'].map((id) => (_jsx("button", { onClick: () => setSelectedThreadId(id), className: `w-full px-4 py-2 text-left rounded transition-colors ${selectedThreadId === id
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`, children: id }, id))) })] }), selectedThreadId && (_jsx("div", { className: "w-96", children: _jsx(DetailPanel, { threadId: selectedThreadId, onClose: () => setSelectedThreadId('') }) }))] }));
}
/**
 * Example 3: DetailPanel with Thread Dependencies
 * Shows how the component displays and visualizes dependencies
 */
export function DependencyDetailPanelExample() {
    const [selectedThreadId, setSelectedThreadId] = useState('main-query');
    const addThread = useAgentManagerStore((state) => state.addThread);
    // Create a dependency graph
    React.useEffect(() => {
        const threads = [
            {
                id: 'main-query',
                name: 'Main Query: Thompson Sampling vs UCB',
                status: 'running',
                priority: 10,
                agentType: 'Coordinator',
                reasoningMode: 'RESEARCH',
                currentStep: 2,
                totalSteps: 3,
                elapsedTimeMs: 5000,
                tokensUsed: 2000,
                confidence: 0.78,
                epistemicConfidence: 0.75,
                childThreadIds: ['analyze-ts', 'analyze-ucb'],
                dependsOn: [],
                blocks: [],
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString(),
            },
            {
                id: 'analyze-ts',
                name: 'Analyze Thompson Sampling',
                status: 'completed',
                priority: 8,
                agentType: 'Researcher',
                reasoningMode: 'VERIFY',
                currentStep: 4,
                totalSteps: 4,
                elapsedTimeMs: 3500,
                tokensUsed: 1200,
                confidence: 0.89,
                epistemicConfidence: 0.85,
                childThreadIds: [],
                dependsOn: [],
                blocks: ['main-query'],
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString(),
            },
            {
                id: 'analyze-ucb',
                name: 'Analyze UCB Algorithm',
                status: 'running',
                priority: 8,
                agentType: 'Researcher',
                reasoningMode: 'VERIFY',
                currentStep: 3,
                totalSteps: 4,
                elapsedTimeMs: 2800,
                tokensUsed: 1100,
                confidence: 0.76,
                epistemicConfidence: 0.72,
                childThreadIds: [],
                dependsOn: [],
                blocks: ['main-query'],
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString(),
            },
        ];
        threads.forEach((thread) => addThread(thread));
    }, [addThread]);
    return (_jsxs("div", { className: "flex h-screen bg-slate-950", children: [_jsxs("div", { className: "flex-1 border-r border-slate-700 p-4", children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 mb-4", children: "Dependency Graph" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("p", { className: "text-sm text-slate-400 mb-2", children: "Main Query (depends on sub-queries)" }), _jsx("button", { onClick: () => setSelectedThreadId('main-query'), className: `w-full px-4 py-2 text-left rounded transition-colors ${selectedThreadId === 'main-query'
                                            ? 'bg-blue-600 text-white'
                                            : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`, children: "Main Query" })] }), _jsxs("div", { children: [_jsx("p", { className: "text-sm text-slate-400 mb-2", children: "Sub-Queries" }), _jsx("div", { className: "space-y-2", children: ['analyze-ts', 'analyze-ucb'].map((id) => (_jsx("button", { onClick: () => setSelectedThreadId(id), className: `w-full px-4 py-2 text-left rounded transition-colors text-sm ${selectedThreadId === id
                                                ? 'bg-blue-600 text-white'
                                                : 'bg-slate-800 hover:bg-slate-700 text-slate-300'}`, children: id }, id))) })] })] })] }), selectedThreadId && (_jsx("div", { className: "w-96", children: _jsx(DetailPanel, { threadId: selectedThreadId, onClose: () => setSelectedThreadId('') }) }))] }));
}
/**
 * Example 4: DetailPanel with Low Epistemic Confidence
 * Shows warning indicators when system confidence is low
 */
export function LowEpistemicConfidenceExample() {
    const [selectedThreadId, setSelectedThreadId] = useState('uncertain-query');
    const addThread = useAgentManagerStore((state) => state.addThread);
    React.useEffect(() => {
        const thread = {
            id: 'uncertain-query',
            name: 'Query: Speculative Analysis',
            status: 'running',
            priority: 3,
            agentType: 'Analyst',
            reasoningMode: 'RESEARCH',
            currentStep: 2,
            totalSteps: 5,
            elapsedTimeMs: 3200,
            tokensUsed: 1800,
            confidence: 0.42,
            epistemicConfidence: 0.25, // LOW - will show warning
            childThreadIds: [],
            dependsOn: [],
            blocks: [],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        };
        addThread(thread);
    }, [addThread]);
    return (_jsxs("div", { className: "flex h-screen bg-slate-950", children: [_jsxs("div", { className: "flex-1 border-r border-slate-700 p-4", children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100 mb-4", children: "Example: Low Epistemic Confidence" }), _jsx("p", { className: "text-slate-400 text-sm mb-4", children: "Notice the warning indicator in the confidence section when epistemic confidence is below 30%" })] }), selectedThreadId && (_jsx("div", { className: "w-96", children: _jsx(DetailPanel, { threadId: selectedThreadId, onClose: () => setSelectedThreadId('') }) }))] }));
}
/**
 * Example 5: Responsive Layout with DetailPanel
 * Shows how to use DetailPanel in different viewport sizes
 */
export function ResponsiveDetailPanelExample() {
    const [selectedThreadId, setSelectedThreadId] = useState('responsive-thread');
    const addThread = useAgentManagerStore((state) => state.addThread);
    React.useEffect(() => {
        const thread = {
            id: 'responsive-thread',
            name: 'Responsive Layout Example',
            status: 'completed',
            priority: 5,
            agentType: 'Demo',
            reasoningMode: 'DIRECT',
            currentStep: 3,
            totalSteps: 3,
            elapsedTimeMs: 1500,
            tokensUsed: 750,
            tokenBudget: 2000,
            confidence: 0.88,
            epistemicConfidence: 0.85,
            childThreadIds: [],
            dependsOn: [],
            blocks: [],
            finalResponse: 'This example shows how the DetailPanel responds to different container widths.',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        };
        addThread(thread);
    }, [addThread]);
    return (_jsxs("div", { className: "flex h-screen bg-slate-950 gap-4 p-4", children: [_jsx("div", { className: "flex-1 border border-slate-700 rounded", children: _jsxs("div", { className: "p-4", children: [_jsx("h2", { className: "text-lg font-semibold text-slate-100", children: "Regular Width (Flex-1)" }), _jsx("p", { className: "text-slate-400 text-sm", children: "Main content area with flexible width" })] }) }), selectedThreadId && (_jsx("div", { className: "w-96", children: _jsx(DetailPanel, { threadId: selectedThreadId, onClose: () => setSelectedThreadId('') }) }))] }));
}
//# sourceMappingURL=DetailPanel.example.js.map