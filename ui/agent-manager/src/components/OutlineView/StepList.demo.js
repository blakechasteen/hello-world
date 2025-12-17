import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
/**
 * StepList Demo Component
 * Shows example usage and test data for StepList and StepRow
 * Phase 3 - Outline View demonstration
 */
import React, { useState } from 'react';
import StepList from './StepList';
/**
 * Sample data generator for demos
 */
const generateSampleSteps = () => [
    {
        id: 'step-1',
        threadId: 'thread-001',
        depth: 0,
        stepType: 'query',
        name: 'Parse User Query',
        query: 'Explain Thompson Sampling and compare with UCB algorithms',
        status: 'completed',
        progressPct: 100,
        elapsedTimeMs: 245,
        tokensUsed: 128,
        confidence: 0.94,
        dependsOn: [],
        blocks: ['step-2', 'step-3'],
        mrfEligible: true,
        mctsEligible: false,
        childrenIds: ['step-2'],
    },
    {
        id: 'step-2',
        threadId: 'thread-001',
        parentId: 'step-1',
        depth: 1,
        stepType: 'research',
        name: 'Research Thompson Sampling',
        query: 'Find information about Thompson Sampling algorithm',
        status: 'completed',
        progressPct: 100,
        elapsedTimeMs: 1840,
        tokensUsed: 2145,
        confidence: 0.87,
        dependsOn: ['step-1'],
        blocks: ['step-4'],
        mrfEligible: true,
        mctsEligible: true,
        childrenIds: ['step-4'],
    },
    {
        id: 'step-3',
        threadId: 'thread-001',
        depth: 1,
        stepType: 'research',
        name: 'Research UCB Algorithm',
        query: 'Find information about Upper Confidence Bound algorithm',
        status: 'completed',
        progressPct: 100,
        elapsedTimeMs: 1620,
        tokensUsed: 1987,
        confidence: 0.89,
        dependsOn: ['step-1'],
        blocks: ['step-4'],
        mrfEligible: true,
        mctsEligible: true,
        childrenIds: [],
    },
    {
        id: 'step-4',
        threadId: 'thread-001',
        parentId: 'step-2',
        depth: 2,
        stepType: 'synthesize',
        name: 'Synthesize Comparison',
        query: 'Compare Thompson Sampling vs UCB on exploration-exploitation tradeoff',
        status: 'running',
        progressPct: 72,
        elapsedTimeMs: 3240,
        tokensUsed: 1654,
        confidence: 0.76,
        dependsOn: ['step-2', 'step-3'],
        blocks: ['step-5'],
        mrfEligible: true,
        mctsEligible: true,
        injectionApplied: 'mrf_verify',
        childrenIds: ['step-5'],
    },
    {
        id: 'step-5',
        threadId: 'thread-001',
        parentId: 'step-4',
        depth: 3,
        stepType: 'verify',
        name: 'Verify Comparison Claims',
        query: 'Verify accuracy and completeness of comparison',
        status: 'pending',
        progressPct: 0,
        elapsedTimeMs: 0,
        tokensUsed: 0,
        confidence: 0.0,
        dependsOn: ['step-4'],
        blocks: ['step-6'],
        mrfEligible: true,
        mctsEligible: false,
        childrenIds: ['step-6'],
    },
    {
        id: 'step-6',
        threadId: 'thread-001',
        parentId: 'step-5',
        depth: 4,
        stepType: 'execute',
        name: 'Generate Final Response',
        query: 'Create comprehensive comparison response',
        status: 'pending',
        progressPct: 0,
        elapsedTimeMs: 0,
        tokensUsed: 0,
        confidence: 0.0,
        dependsOn: ['step-5'],
        blocks: [],
        mrfEligible: false,
        mctsEligible: false,
        childrenIds: [],
    },
    {
        id: 'step-7',
        threadId: 'thread-001',
        depth: 1,
        stepType: 'research',
        name: 'Research Failure - Network Error',
        status: 'failed',
        progressPct: 0,
        elapsedTimeMs: 5000,
        tokensUsed: 0,
        confidence: 0.0,
        dependsOn: ['step-1'],
        blocks: [],
        mrfEligible: false,
        mctsEligible: false,
        childrenIds: [],
    },
    {
        id: 'step-8',
        threadId: 'thread-001',
        depth: 1,
        stepType: 'query',
        name: 'Skipped Alternative Path',
        status: 'skipped',
        progressPct: 0,
        elapsedTimeMs: 0,
        tokensUsed: 0,
        confidence: 0.0,
        dependsOn: ['step-1'],
        blocks: [],
        mrfEligible: false,
        mctsEligible: false,
        childrenIds: [],
    },
];
/**
 * Standalone demo component
 */
export const StepListDemo = () => {
    const [steps, setSteps] = useState(generateSampleSteps());
    const [selectedId, setSelectedId] = useState('step-4');
    const [hoveredId, setHoveredId] = useState(null);
    // Simulate running step progress
    React.useEffect(() => {
        const interval = setInterval(() => {
            setSteps((prevSteps) => prevSteps.map((step) => {
                if (step.status === 'running' && step.progressPct < 100) {
                    return {
                        ...step,
                        progressPct: Math.min(100, step.progressPct + Math.random() * 15),
                        elapsedTimeMs: step.elapsedTimeMs + 100,
                    };
                }
                if (step.status === 'running' && step.progressPct >= 100) {
                    return {
                        ...step,
                        status: 'completed',
                        progressPct: 100,
                    };
                }
                return step;
            }));
        }, 500);
        return () => clearInterval(interval);
    }, []);
    const handleInjectMRF = (stepId) => {
        console.log('MRF Injection requested for step:', stepId);
        setSteps((prev) => prev.map((s) => s.id === stepId
            ? { ...s, injectionApplied: 'mrf_verify', confidence: Math.min(1, s.confidence + 0.05) }
            : s));
    };
    const handleInjectMCTS = (stepId) => {
        console.log('MCTS Injection requested for step:', stepId);
        setSteps((prev) => prev.map((s) => s.id === stepId
            ? { ...s, injectionApplied: 'mcts_research', confidence: Math.min(1, s.confidence + 0.1) }
            : s));
    };
    const rootTask = steps[0];
    return (_jsx("div", { className: "w-full h-screen bg-slate-900 p-8", children: _jsxs("div", { className: "max-w-4xl mx-auto", children: [_jsxs("div", { className: "mb-8", children: [_jsx("h1", { className: "text-3xl font-bold text-white mb-2", children: "StepList Component Demo" }), _jsx("p", { className: "text-slate-400", children: "HoloLoom Agent Manager UI - Phase 3 Outline View" })] }), _jsxs("div", { className: "space-y-4", children: [_jsx(StepList, { steps: steps, threadId: "thread-001", rootTask: rootTask, hoveredStepId: hoveredId, selectedStepId: selectedId, onStepHover: setHoveredId, onStepSelect: setSelectedId, onInjectMRF: handleInjectMRF, onInjectMCTS: handleInjectMCTS, showQueryPreview: true, className: "shadow-xl" }), selectedId && (_jsxs("div", { className: "bg-slate-800 border border-slate-700 rounded p-4", children: [_jsx("h3", { className: "text-lg font-semibold text-white mb-3", children: "Selected Step Details" }), (() => {
                                    const selectedStep = steps.find((s) => s.id === selectedId);
                                    if (!selectedStep)
                                        return null;
                                    return (_jsxs("div", { className: "grid grid-cols-2 gap-4 text-sm text-slate-300", children: [_jsxs("div", { children: [_jsx("p", { className: "text-slate-400", children: "ID" }), _jsx("p", { className: "font-mono", children: selectedStep.id })] }), _jsxs("div", { children: [_jsx("p", { className: "text-slate-400", children: "Type" }), _jsx("p", { children: selectedStep.stepType })] }), _jsxs("div", { children: [_jsx("p", { className: "text-slate-400", children: "Status" }), _jsx("p", { className: "capitalize", children: selectedStep.status })] }), _jsxs("div", { children: [_jsx("p", { className: "text-slate-400", children: "Confidence" }), _jsx("p", { children: selectedStep.confidence.toFixed(2) })] }), _jsxs("div", { children: [_jsx("p", { className: "text-slate-400", children: "Progress" }), _jsxs("p", { children: [selectedStep.progressPct.toFixed(0), "%"] })] }), _jsxs("div", { children: [_jsx("p", { className: "text-slate-400", children: "Tokens Used" }), _jsx("p", { children: selectedStep.tokensUsed.toLocaleString() })] }), _jsxs("div", { className: "col-span-2", children: [_jsx("p", { className: "text-slate-400", children: "Query" }), _jsx("p", { className: "italic", children: selectedStep.query || '(none)' })] }), _jsxs("div", { className: "col-span-2", children: [_jsx("p", { className: "text-slate-400 mb-2", children: "Actions" }), _jsxs("div", { className: "flex gap-2", children: [selectedStep.mrfEligible && !selectedStep.injectionApplied && (_jsx("button", { onClick: () => handleInjectMRF(selectedStep.id), className: "px-3 py-1 bg-emerald-900/40 text-emerald-300 rounded text-sm hover:bg-emerald-800/50 transition-colors", children: "Inject MRF" })), selectedStep.mctsEligible && !selectedStep.injectionApplied && (_jsx("button", { onClick: () => handleInjectMCTS(selectedStep.id), className: "px-3 py-1 bg-cyan-900/40 text-cyan-300 rounded text-sm hover:bg-cyan-800/50 transition-colors", children: "Inject MCTS" })), selectedStep.injectionApplied && (_jsxs("span", { className: "px-3 py-1 bg-purple-900/40 text-purple-300 rounded text-sm", children: ["Applied: ", selectedStep.injectionApplied] }))] })] })] }));
                                })()] })), _jsxs("div", { className: "grid grid-cols-4 gap-4", children: [_jsxs("div", { className: "bg-slate-800 border border-slate-700 rounded p-4", children: [_jsx("p", { className: "text-slate-400 text-sm", children: "Total Steps" }), _jsx("p", { className: "text-2xl font-bold text-white", children: steps.length })] }), _jsxs("div", { className: "bg-slate-800 border border-slate-700 rounded p-4", children: [_jsx("p", { className: "text-slate-400 text-sm", children: "Completed" }), _jsx("p", { className: "text-2xl font-bold text-emerald-400", children: steps.filter((s) => s.status === 'completed').length })] }), _jsxs("div", { className: "bg-slate-800 border border-slate-700 rounded p-4", children: [_jsx("p", { className: "text-slate-400 text-sm", children: "Running" }), _jsx("p", { className: "text-2xl font-bold text-blue-400", children: steps.filter((s) => s.status === 'running').length })] }), _jsxs("div", { className: "bg-slate-800 border border-slate-700 rounded p-4", children: [_jsx("p", { className: "text-slate-400 text-sm", children: "Failed" }), _jsx("p", { className: "text-2xl font-bold text-red-400", children: steps.filter((s) => s.status === 'failed').length })] })] })] })] }) }));
};
export default StepListDemo;
//# sourceMappingURL=StepList.demo.js.map