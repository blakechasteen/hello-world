/**
 * Real-Time Weaving Cycle Visualizer
 *
 * Shows live progress through HoloLoom's 9-step weaving cycle:
 * 1. Loom Command → Pattern selection
 * 2. Chrono Trigger → Temporal windows
 * 3. Yarn Graph → Memory thread selection
 * 4. Resonance Shed → Feature extraction (DotPlasma)
 * 5. Warp Space → Continuous manifold tensioning
 * 6. Convergence Engine → Decision collapse
 * 7. Tool Execution → Action
 * 8. Spacetime Fabric → Provenance trace
 * 9. Reflection Buffer → Learning from outcome
 */

import React, { useEffect, useState } from 'react';
import { CheckCircle2, Circle, Loader2, AlertCircle, Clock } from 'lucide-react';
import { WeavingCycle, WeavingStep, StepStatus } from '../types';

interface WeavingVisualizerProps {
  weavingCycle?: WeavingCycle;
  onQuerySubmit?: (query: string) => void;
}

const WEAVING_STEPS = [
  { step: 1, name: 'Loom Command', description: 'Pattern selection (BARE/FAST/FUSED)' },
  { step: 2, name: 'Chrono Trigger', description: 'Temporal window creation' },
  { step: 3, name: 'Yarn Graph', description: 'Memory thread selection' },
  { step: 4, name: 'Resonance Shed', description: 'Feature extraction (DotPlasma)' },
  { step: 5, name: 'Warp Space', description: 'Continuous manifold tensioning' },
  { step: 6, name: 'Convergence Engine', description: 'Decision collapse' },
  { step: 7, name: 'Tool Execution', description: 'Action execution' },
  { step: 8, name: 'Spacetime Fabric', description: 'Provenance trace creation' },
  { step: 9, name: 'Reflection Buffer', description: 'Learning from outcome' },
];

const StepIcon: React.FC<{ status: StepStatus }> = ({ status }) => {
  switch (status) {
    case StepStatus.COMPLETED:
      return <CheckCircle2 className="w-5 h-5 text-green-500" />;
    case StepStatus.IN_PROGRESS:
      return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
    case StepStatus.ERROR:
      return <AlertCircle className="w-5 h-5 text-red-500" />;
    default:
      return <Circle className="w-5 h-5 text-gray-400" />;
  }
};

const StepStatusBadge: React.FC<{ status: StepStatus }> = ({ status }) => {
  const colors = {
    [StepStatus.WAITING]: 'bg-gray-100 text-gray-600',
    [StepStatus.IN_PROGRESS]: 'bg-blue-100 text-blue-700',
    [StepStatus.COMPLETED]: 'bg-green-100 text-green-700',
    [StepStatus.ERROR]: 'bg-red-100 text-red-700',
  };

  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${colors[status]}`}>
      {status}
    </span>
  );
};

export const WeavingVisualizer: React.FC<WeavingVisualizerProps> = ({
  weavingCycle,
  onQuerySubmit,
}) => {
  const [query, setQuery] = useState('');
  const [currentSteps, setCurrentSteps] = useState<WeavingStep[]>([]);

  useEffect(() => {
    if (weavingCycle) {
      setCurrentSteps(weavingCycle.steps);
    }
  }, [weavingCycle]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && onQuerySubmit) {
      onQuerySubmit(query.trim());
    }
  };

  const getStepStatus = (stepNum: number): WeavingStep | undefined => {
    return currentSteps.find(s => s.step === stepNum);
  };

  const getCurrentStep = (): WeavingStep | undefined => {
    return currentSteps.find(s => s.status === StepStatus.IN_PROGRESS);
  };

  const totalLatency = currentSteps.reduce((sum, step) => sum + (step.latency_ms || 0), 0);
  const completedSteps = currentSteps.filter(s => s.status === StepStatus.COMPLETED).length;
  const currentStep = getCurrentStep();

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Weaving Cycle Visualizer
        </h2>
        <p className="text-gray-600">
          Real-time visualization of HoloLoom's 9-step processing pipeline
        </p>
      </div>

      {/* Query Input */}
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={!query.trim()}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Weave
          </button>
        </div>
      </form>

      {/* Current Query Info */}
      {weavingCycle && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="text-sm text-gray-500 mb-1">Current Query</p>
              <p className="font-medium text-gray-900">{weavingCycle.query_text}</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500 mb-1">Progress</p>
              <p className="text-2xl font-bold text-blue-600">
                {completedSteps}/{WEAVING_STEPS.length}
              </p>
            </div>
          </div>

          {currentStep && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <p className="text-sm text-gray-600">
                Current Step: <span className="font-medium text-gray-900">{currentStep.name}</span>
                {currentStep.metadata && (
                  <span className="ml-2 text-gray-500">
                    ({JSON.stringify(currentStep.metadata)})
                  </span>
                )}
              </p>
            </div>
          )}

          <div className="mt-3 pt-3 border-t border-gray-200 flex items-center justify-between text-sm">
            <div className="flex items-center gap-2 text-gray-600">
              <Clock className="w-4 h-4" />
              <span>Total Latency: <span className="font-medium">{totalLatency.toFixed(1)}ms</span></span>
            </div>
            {weavingCycle.confidence !== undefined && (
              <div className="text-gray-600">
                Confidence: <span className="font-medium">{(weavingCycle.confidence * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Steps List */}
      <div className="space-y-2">
        {WEAVING_STEPS.map(({ step, name, description }) => {
          const stepData = getStepStatus(step);
          const status = stepData?.status || StepStatus.WAITING;
          const latency = stepData?.latency_ms;

          return (
            <div
              key={step}
              className={`
                p-4 rounded-lg border-2 transition-all
                ${status === StepStatus.IN_PROGRESS ? 'border-blue-500 bg-blue-50' : ''}
                ${status === StepStatus.COMPLETED ? 'border-green-500 bg-green-50' : ''}
                ${status === StepStatus.ERROR ? 'border-red-500 bg-red-50' : ''}
                ${status === StepStatus.WAITING ? 'border-gray-200 bg-white' : ''}
              `}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 flex-1">
                  <StepIcon status={status} />
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-900">
                        {step}. {name}
                      </span>
                      <StepStatusBadge status={status} />
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{description}</p>
                  </div>
                </div>

                {latency !== undefined && (
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">{latency.toFixed(1)}ms</p>
                  </div>
                )}
              </div>

              {stepData?.error && (
                <div className="mt-3 p-3 bg-red-100 border border-red-300 rounded text-sm text-red-700">
                  <strong>Error:</strong> {stepData.error}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Final Result */}
      {weavingCycle?.response && (
        <div className="mt-6 p-4 bg-indigo-50 border-2 border-indigo-500 rounded-lg">
          <h3 className="font-medium text-indigo-900 mb-2">Response</h3>
          <p className="text-gray-700">{weavingCycle.response}</p>

          {weavingCycle.tool_used && (
            <p className="mt-3 text-sm text-indigo-600">
              Tool Used: <span className="font-medium">{weavingCycle.tool_used}</span>
            </p>
          )}
        </div>
      )}
    </div>
  );
};
