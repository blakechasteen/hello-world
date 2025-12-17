/**
 * StepHistory Component - Demo & Usage Examples
 * Shows how to use the StepHistory component with various configurations
 *
 * HoloLoom Agent Manager UI - Phase 4
 */

import React, { useState } from 'react';
import StepHistory from './StepHistory';
import { TaskNode, StepStatus, StepType } from './types';

/**
 * Generate mock step data for demos
 */
const createMockSteps = (count: number = 12): TaskNode[] => {
  const stepTypes: StepType[] = [
    'query',
    'retrieval',
    'reasoning',
    'synthesis',
    'verification',
    'research',
    'planning',
    'execution',
  ];

  const statuses: StepStatus[] = [
    'completed',
    'completed',
    'completed',
    'completed',
    'running',
    'paused',
    'failed',
    'idle',
  ];

  const steps: TaskNode[] = [];

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
      query:
        i % 2 === 0
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

      response:
        status === 'completed' && Math.random() > 0.5
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
export const BasicDemo: React.FC = () => {
  const [steps] = useState(() => createMockSteps(8));
  const [currentStepIndex, setCurrentStepIndex] = useState(3);

  return (
    <div className="h-screen flex flex-col gap-4 p-4 bg-slate-900">
      <div className="flex-1 flex gap-4">
        <div className="flex-1">
          <StepHistory
            steps={steps}
            currentStepIndex={currentStepIndex}
            onStepSelect={(stepId) => {
              const index = steps.findIndex((s) => s.id === stepId);
              setCurrentStepIndex(index);
            }}
          />
        </div>

        {/* Details panel */}
        <div className="flex-1 bg-slate-800 rounded-lg p-4 overflow-y-auto">
          <h2 className="text-lg font-bold text-slate-100 mb-4">Step Details</h2>
          {steps[currentStepIndex] && (
            <div className="space-y-2 text-sm text-slate-300">
              <div>
                <span className="text-slate-500">ID:</span> {steps[currentStepIndex].id}
              </div>
              <div>
                <span className="text-slate-500">Type:</span> {steps[currentStepIndex].stepType}
              </div>
              <div>
                <span className="text-slate-500">Name:</span> {steps[currentStepIndex].name}
              </div>
              <div>
                <span className="text-slate-500">Status:</span>{' '}
                <span className="font-mono">{steps[currentStepIndex].status}</span>
              </div>
              <div>
                <span className="text-slate-500">Confidence:</span>{' '}
                <span className="font-mono">{steps[currentStepIndex].confidence.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-slate-500">Tokens:</span>{' '}
                <span className="font-mono">{steps[currentStepIndex].tokensUsed}</span>
              </div>
              <div>
                <span className="text-slate-500">Time:</span>{' '}
                <span className="font-mono">{steps[currentStepIndex].elapsedTimeMs.toFixed(0)}ms</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Advanced demo - With callbacks and injections
 */
export const AdvancedDemo: React.FC = () => {
  const [steps, setSteps] = useState(() => createMockSteps(20));
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [injectionLog, setInjectionLog] = useState<string[]>([]);

  const handleInjectMRF = (stepId: string) => {
    setInjectionLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] MRF injected: ${stepId}`]);
    setSteps((prev) =>
      prev.map((s) => (s.id === stepId ? { ...s, injectionApplied: 'mrf' } : s))
    );
  };

  const handleInjectMCTS = (stepId: string) => {
    setInjectionLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] MCTS injected: ${stepId}`]);
    setSteps((prev) =>
      prev.map((s) => (s.id === stepId ? { ...s, injectionApplied: 'mcts' } : s))
    );
  };

  return (
    <div className="h-screen flex gap-4 p-4 bg-slate-900">
      {/* Main step history */}
      <div className="flex-1">
        <StepHistory
          steps={steps}
          currentStepIndex={currentStepIndex}
          onStepSelect={(stepId) => {
            const index = steps.findIndex((s) => s.id === stepId);
            setCurrentStepIndex(index);
          }}
          onInjectMRF={handleInjectMRF}
          onInjectMCTS={handleInjectMCTS}
        />
      </div>

      {/* Injection log */}
      <div className="w-80 flex flex-col gap-4">
        <div className="bg-slate-800 rounded-lg p-4 flex-1 flex flex-col">
          <h3 className="text-sm font-bold text-slate-100 mb-3">Injection Events</h3>
          <div className="flex-1 overflow-y-auto space-y-1">
            {injectionLog.length === 0 ? (
              <div className="text-xs text-slate-500 italic">No injections yet. Click "Inject MRF" or "Inject MCTS" on step rows.</div>
            ) : (
              injectionLog.map((log, i) => (
                <div key={i} className="text-xs text-slate-300 font-mono">
                  {log}
                </div>
              ))
            )}
          </div>
          <button
            onClick={() => setInjectionLog([])}
            className="mt-2 px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors"
          >
            Clear Log
          </button>
        </div>

        {/* Statistics */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-sm font-bold text-slate-100 mb-3">Statistics</h3>
          <div className="space-y-2 text-xs text-slate-300">
            <div>
              <span className="text-slate-500">Total Steps:</span>{' '}
              <span className="font-mono text-slate-100">{steps.length}</span>
            </div>
            <div>
              <span className="text-slate-500">Completed:</span>{' '}
              <span className="font-mono text-emerald-400">{steps.filter((s) => s.status === 'completed').length}</span>
            </div>
            <div>
              <span className="text-slate-500">Running:</span>{' '}
              <span className="font-mono text-blue-400">{steps.filter((s) => s.status === 'running').length}</span>
            </div>
            <div>
              <span className="text-slate-500">Failed:</span>{' '}
              <span className="font-mono text-red-400">{steps.filter((s) => s.status === 'failed').length}</span>
            </div>
            <div className="border-t border-slate-700 pt-2 mt-2">
              <span className="text-slate-500">Avg Confidence:</span>
              <span className="block font-mono text-slate-100">
                {(steps.reduce((sum, s) => sum + s.confidence, 0) / steps.length).toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Performance demo - Large number of steps
 */
export const PerformanceDemo: React.FC = () => {
  const [steps] = useState(() => createMockSteps(100));
  const [currentStepIndex, setCurrentStepIndex] = useState(50);

  return (
    <div className="h-screen flex flex-col gap-4 p-4 bg-slate-900">
      <div className="text-sm text-slate-400">
        Virtualized list with 100 steps - scroll smoothly without performance impact
      </div>
      <div className="flex-1">
        <StepHistory
          steps={steps}
          currentStepIndex={currentStepIndex}
          onStepSelect={(stepId) => {
            const index = steps.findIndex((s) => s.id === stepId);
            setCurrentStepIndex(index);
          }}
        />
      </div>
    </div>
  );
};

/**
 * Responsive demo - Shows how component adapts to different screen sizes
 */
export const ResponsiveDemo: React.FC = () => {
  const [steps] = useState(() => createMockSteps(15));
  const [currentStepIndex, setCurrentStepIndex] = useState(7);

  return (
    <div className="min-h-screen bg-slate-900 p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold text-slate-100 mb-4">Responsive StepHistory Demo</h1>
        <p className="text-sm text-slate-400 mb-6">
          Try resizing your browser to see how the component adapts. On mobile, some columns hide; on desktop,
          all details are visible.
        </p>

        <div className="h-[600px]">
          <StepHistory
            steps={steps}
            currentStepIndex={currentStepIndex}
            onStepSelect={(stepId) => {
              const index = steps.findIndex((s) => s.id === stepId);
              setCurrentStepIndex(index);
            }}
          />
        </div>

        {/* Responsive breakpoint indicators */}
        <div className="mt-4 space-y-2 text-xs text-slate-500">
          <div className="sm:hidden">📱 Mobile (&lt; 640px) - Minimal columns</div>
          <div className="hidden sm:block md:hidden">
            📱 Tablet Small (640px - 768px) - Added duration
          </div>
          <div className="hidden md:block lg:hidden">
            📱 Tablet Large (768px - 1024px) - Added status label
          </div>
          <div className="hidden lg:block">💻 Desktop (≥ 1024px) - All columns visible</div>
        </div>
      </div>
    </div>
  );
};

/**
 * Story export for Storybook or similar
 */
export default BasicDemo;
