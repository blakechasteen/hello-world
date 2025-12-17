/**
 * StepList Demo Component
 * Shows example usage and test data for StepList and StepRow
 * Phase 3 - Outline View demonstration
 */

import React, { useState } from 'react';
import StepList from './StepList';
import { TaskNode } from './types';

/**
 * Sample data generator for demos
 */
const generateSampleSteps = (): TaskNode[] => [
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
export const StepListDemo: React.FC = () => {
  const [steps, setSteps] = useState<TaskNode[]>(generateSampleSteps());
  const [selectedId, setSelectedId] = useState<string | null>('step-4');
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  // Simulate running step progress
  React.useEffect(() => {
    const interval = setInterval(() => {
      setSteps((prevSteps) =>
        prevSteps.map((step) => {
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
              status: 'completed' as const,
              progressPct: 100,
            };
          }
          return step;
        })
      );
    }, 500);

    return () => clearInterval(interval);
  }, []);

  const handleInjectMRF = (stepId: string) => {
    console.log('MRF Injection requested for step:', stepId);
    setSteps((prev) =>
      prev.map((s) =>
        s.id === stepId
          ? { ...s, injectionApplied: 'mrf_verify', confidence: Math.min(1, s.confidence + 0.05) }
          : s
      )
    );
  };

  const handleInjectMCTS = (stepId: string) => {
    console.log('MCTS Injection requested for step:', stepId);
    setSteps((prev) =>
      prev.map((s) =>
        s.id === stepId
          ? { ...s, injectionApplied: 'mcts_research', confidence: Math.min(1, s.confidence + 0.1) }
          : s
      )
    );
  };

  const rootTask = steps[0];

  return (
    <div className="w-full h-screen bg-slate-900 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">
            StepList Component Demo
          </h1>
          <p className="text-slate-400">
            HoloLoom Agent Manager UI - Phase 3 Outline View
          </p>
        </div>

        <div className="space-y-4">
          {/* Main StepList */}
          <StepList
            steps={steps}
            threadId="thread-001"
            rootTask={rootTask}
            hoveredStepId={hoveredId}
            selectedStepId={selectedId}
            onStepHover={setHoveredId}
            onStepSelect={setSelectedId}
            onInjectMRF={handleInjectMRF}
            onInjectMCTS={handleInjectMCTS}
            showQueryPreview={true}
            className="shadow-xl"
          />

          {/* Info Panel */}
          {selectedId && (
            <div className="bg-slate-800 border border-slate-700 rounded p-4">
              <h3 className="text-lg font-semibold text-white mb-3">
                Selected Step Details
              </h3>
              {(() => {
                const selectedStep = steps.find((s) => s.id === selectedId);
                if (!selectedStep) return null;

                return (
                  <div className="grid grid-cols-2 gap-4 text-sm text-slate-300">
                    <div>
                      <p className="text-slate-400">ID</p>
                      <p className="font-mono">{selectedStep.id}</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Type</p>
                      <p>{selectedStep.stepType}</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Status</p>
                      <p className="capitalize">{selectedStep.status}</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Confidence</p>
                      <p>{selectedStep.confidence.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Progress</p>
                      <p>{selectedStep.progressPct.toFixed(0)}%</p>
                    </div>
                    <div>
                      <p className="text-slate-400">Tokens Used</p>
                      <p>{selectedStep.tokensUsed.toLocaleString()}</p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-slate-400">Query</p>
                      <p className="italic">{selectedStep.query || '(none)'}</p>
                    </div>
                    <div className="col-span-2">
                      <p className="text-slate-400 mb-2">Actions</p>
                      <div className="flex gap-2">
                        {selectedStep.mrfEligible && !selectedStep.injectionApplied && (
                          <button
                            onClick={() => handleInjectMRF(selectedStep.id)}
                            className="px-3 py-1 bg-emerald-900/40 text-emerald-300 rounded text-sm hover:bg-emerald-800/50 transition-colors"
                          >
                            Inject MRF
                          </button>
                        )}
                        {selectedStep.mctsEligible && !selectedStep.injectionApplied && (
                          <button
                            onClick={() => handleInjectMCTS(selectedStep.id)}
                            className="px-3 py-1 bg-cyan-900/40 text-cyan-300 rounded text-sm hover:bg-cyan-800/50 transition-colors"
                          >
                            Inject MCTS
                          </button>
                        )}
                        {selectedStep.injectionApplied && (
                          <span className="px-3 py-1 bg-purple-900/40 text-purple-300 rounded text-sm">
                            Applied: {selectedStep.injectionApplied}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {/* Statistics */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-slate-800 border border-slate-700 rounded p-4">
              <p className="text-slate-400 text-sm">Total Steps</p>
              <p className="text-2xl font-bold text-white">{steps.length}</p>
            </div>
            <div className="bg-slate-800 border border-slate-700 rounded p-4">
              <p className="text-slate-400 text-sm">Completed</p>
              <p className="text-2xl font-bold text-emerald-400">
                {steps.filter((s) => s.status === 'completed').length}
              </p>
            </div>
            <div className="bg-slate-800 border border-slate-700 rounded p-4">
              <p className="text-slate-400 text-sm">Running</p>
              <p className="text-2xl font-bold text-blue-400">
                {steps.filter((s) => s.status === 'running').length}
              </p>
            </div>
            <div className="bg-slate-800 border border-slate-700 rounded p-4">
              <p className="text-slate-400 text-sm">Failed</p>
              <p className="text-2xl font-bold text-red-400">
                {steps.filter((s) => s.status === 'failed').length}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StepListDemo;
