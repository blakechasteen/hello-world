import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * StepHistory Component - Demo & Usage Examples
 * Shows how to use the StepHistory component with various configurations
 *
 * HoloLoom Agent Manager UI - Phase 4
 */
import { useState } from 'react';
import StepHistory from './StepHistory';
/**
 * Generate mock step data for demos
 */
const createMockSteps = (count = 12) => {
    const stepTypes = [
        'query',
        'retrieval',
        'reasoning',
        'synthesis',
        'verification',
        'research',
        'planning',
        'execution',
    ];
    const statuses = [
        'completed',
        'completed',
        'completed',
        'completed',
        'running',
        'paused',
        'failed',
        'idle',
    ];
    const steps = [];
    for (let i = 0; i < count; i++) {
        const stepType = stepTypes[i % stepTypes.length];
        const status = statuses[Math.floor(Math.random() * statuses.length)];
        steps.push({
            id: `step-${i + 1}`,
            threadId: 'thread-001',
            childrenIds: [],
            depth: 0,
            stepType,
            name: `${stepType.charAt(0).toUpperCase() + stepType.slice(1)} Step ${i + 1}`,
            query: i % 2 === 0
                ? `What is the relationship between ${stepType === 'query' ? 'Thompson Sampling' : 'Bayesian methods'}?`
                : undefined,
            status,
            progressPct: status === 'running' ? Math.random() * 100 : status === 'completed' ? 100 : 0,
            elapsedTimeMs: Math.random() * 5000 + 100,
            tokensUsed: Math.floor(Math.random() * 2000) + 100,
            confidence: Math.random() * 0.5 + 0.5,
            dependsOn: i > 0 && Math.random() > 0.7 ? [`step-${i}`] : [],
            blocks: [],
            mrfEligible: Math.random() > 0.7,
            mctsEligible: Math.random() > 0.8,
            injectionApplied: null,
            injectionStrategy: undefined,
            response: status === 'completed' && Math.random() > 0.5
                ? `Thompson Sampling is a Bayesian approach to the multi-armed bandit problem that balances exploration and exploitation using Beta distributions...`
                : undefined,
            toolSelected: status === 'completed' && Math.random() > 0.6 ? 'answer' : undefined,
        });
    }
    return steps;
};
/**
 * Basic demo - Simple usage
 */
export const BasicDemo = () => {
    const [steps] = useState(() => createMockSteps(8));
    const [currentStepIndex, setCurrentStepIndex] = useState(3);
    return (_jsx("div", { className: "h-screen flex flex-col gap-4 p-4 bg-slate-900", children: _jsxs("div", { className: "flex-1 flex gap-4", children: [_jsx("div", { className: "flex-1", children: _jsx(StepHistory, { steps: steps, currentStepIndex: currentStepIndex, onStepSelect: (stepId) => {
                            const index = steps.findIndex((s) => s.id === stepId);
                            setCurrentStepIndex(index);
                        } }) }), _jsxs("div", { className: "flex-1 bg-slate-800 rounded-lg p-4 overflow-y-auto", children: [_jsx("h2", { className: "text-lg font-bold text-slate-100 mb-4", children: "Step Details" }), steps[currentStepIndex] && (_jsxs("div", { className: "space-y-2 text-sm text-slate-300", children: [_jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "ID:" }), " ", steps[currentStepIndex].id] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Type:" }), " ", steps[currentStepIndex].stepType] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Name:" }), " ", steps[currentStepIndex].name] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Status:" }), ' ', _jsx("span", { className: "font-mono", children: steps[currentStepIndex].status })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Confidence:" }), ' ', _jsx("span", { className: "font-mono", children: steps[currentStepIndex].confidence.toFixed(2) })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Tokens:" }), ' ', _jsx("span", { className: "font-mono", children: steps[currentStepIndex].tokensUsed })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Time:" }), ' ', _jsxs("span", { className: "font-mono", children: [steps[currentStepIndex].elapsedTimeMs.toFixed(0), "ms"] })] })] }))] })] }) }));
};
/**
 * Advanced demo - With callbacks and injections
 */
export const AdvancedDemo = () => {
    const [steps, setSteps] = useState(() => createMockSteps(20));
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [injectionLog, setInjectionLog] = useState([]);
    const handleInjectMRF = (stepId) => {
        setInjectionLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] MRF injected: ${stepId}`]);
        setSteps((prev) => prev.map((s) => (s.id === stepId ? { ...s, injectionApplied: 'mrf' } : s)));
    };
    const handleInjectMCTS = (stepId) => {
        setInjectionLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] MCTS injected: ${stepId}`]);
        setSteps((prev) => prev.map((s) => (s.id === stepId ? { ...s, injectionApplied: 'mcts' } : s)));
    };
    return (_jsxs("div", { className: "h-screen flex gap-4 p-4 bg-slate-900", children: [_jsx("div", { className: "flex-1", children: _jsx(StepHistory, { steps: steps, currentStepIndex: currentStepIndex, onStepSelect: (stepId) => {
                        const index = steps.findIndex((s) => s.id === stepId);
                        setCurrentStepIndex(index);
                    }, onInjectMRF: handleInjectMRF, onInjectMCTS: handleInjectMCTS }) }), _jsxs("div", { className: "w-80 flex flex-col gap-4", children: [_jsxs("div", { className: "bg-slate-800 rounded-lg p-4 flex-1 flex flex-col", children: [_jsx("h3", { className: "text-sm font-bold text-slate-100 mb-3", children: "Injection Events" }), _jsx("div", { className: "flex-1 overflow-y-auto space-y-1", children: injectionLog.length === 0 ? (_jsx("div", { className: "text-xs text-slate-500 italic", children: "No injections yet. Click \"Inject MRF\" or \"Inject MCTS\" on step rows." })) : (injectionLog.map((log, i) => (_jsx("div", { className: "text-xs text-slate-300 font-mono", children: log }, i)))) }), _jsx("button", { onClick: () => setInjectionLog([]), className: "mt-2 px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors", children: "Clear Log" })] }), _jsxs("div", { className: "bg-slate-800 rounded-lg p-4", children: [_jsx("h3", { className: "text-sm font-bold text-slate-100 mb-3", children: "Statistics" }), _jsxs("div", { className: "space-y-2 text-xs text-slate-300", children: [_jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Total Steps:" }), ' ', _jsx("span", { className: "font-mono text-slate-100", children: steps.length })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Completed:" }), ' ', _jsx("span", { className: "font-mono text-emerald-400", children: steps.filter((s) => s.status === 'completed').length })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Running:" }), ' ', _jsx("span", { className: "font-mono text-blue-400", children: steps.filter((s) => s.status === 'running').length })] }), _jsxs("div", { children: [_jsx("span", { className: "text-slate-500", children: "Failed:" }), ' ', _jsx("span", { className: "font-mono text-red-400", children: steps.filter((s) => s.status === 'failed').length })] }), _jsxs("div", { className: "border-t border-slate-700 pt-2 mt-2", children: [_jsx("span", { className: "text-slate-500", children: "Avg Confidence:" }), _jsx("span", { className: "block font-mono text-slate-100", children: (steps.reduce((sum, s) => sum + s.confidence, 0) / steps.length).toFixed(2) })] })] })] })] })] }));
};
/**
 * Performance demo - Large number of steps
 */
export const PerformanceDemo = () => {
    const [steps] = useState(() => createMockSteps(100));
    const [currentStepIndex, setCurrentStepIndex] = useState(50);
    return (_jsxs("div", { className: "h-screen flex flex-col gap-4 p-4 bg-slate-900", children: [_jsx("div", { className: "text-sm text-slate-400", children: "Virtualized list with 100 steps - scroll smoothly without performance impact" }), _jsx("div", { className: "flex-1", children: _jsx(StepHistory, { steps: steps, currentStepIndex: currentStepIndex, onStepSelect: (stepId) => {
                        const index = steps.findIndex((s) => s.id === stepId);
                        setCurrentStepIndex(index);
                    } }) })] }));
};
/**
 * Responsive demo - Shows how component adapts to different screen sizes
 */
export const ResponsiveDemo = () => {
    const [steps] = useState(() => createMockSteps(15));
    const [currentStepIndex, setCurrentStepIndex] = useState(7);
    return (_jsx("div", { className: "min-h-screen bg-slate-900 p-4", children: _jsxs("div", { className: "max-w-7xl mx-auto", children: [_jsx("h1", { className: "text-2xl font-bold text-slate-100 mb-4", children: "Responsive StepHistory Demo" }), _jsx("p", { className: "text-sm text-slate-400 mb-6", children: "Try resizing your browser to see how the component adapts. On mobile, some columns hide; on desktop, all details are visible." }), _jsx("div", { className: "h-[600px]", children: _jsx(StepHistory, { steps: steps, currentStepIndex: currentStepIndex, onStepSelect: (stepId) => {
                            const index = steps.findIndex((s) => s.id === stepId);
                            setCurrentStepIndex(index);
                        } }) }), _jsxs("div", { className: "mt-4 space-y-2 text-xs text-slate-500", children: [_jsx("div", { className: "sm:hidden", children: "\uD83D\uDCF1 Mobile (< 640px) - Minimal columns" }), _jsx("div", { className: "hidden sm:block md:hidden", children: "\uD83D\uDCF1 Tablet Small (640px - 768px) - Added duration" }), _jsx("div", { className: "hidden md:block lg:hidden", children: "\uD83D\uDCF1 Tablet Large (768px - 1024px) - Added status label" }), _jsx("div", { className: "hidden lg:block", children: "\uD83D\uDCBB Desktop (\u2265 1024px) - All columns visible" })] })] }) }));
};
/**
 * Story export for Storybook or similar
 */
export default BasicDemo;
//# sourceMappingURL=StepHistory.demo.js.map