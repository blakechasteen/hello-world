/**
 * ConnectedMetricsPane
 *
 * Connected wrapper for the Awareness/Metrics pane (right side).
 * Shows confidence, reasoning steps, and session metrics.
 *
 * Part of HoloLoom Phase 1 Frontend (November 2025)
 */

import React, { useMemo } from 'react';
import { clsx } from 'clsx';
import { useHoloLoomStore } from '../../store';

// ============================================================================
// Props
// ============================================================================

export interface ConnectedMetricsPaneProps {
  /** Dark mode */
  darkMode?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Sub-Components
// ============================================================================

interface ConfidenceGaugeProps {
  value: number;
  epistemicValue: number;
  history: number[];
  darkMode?: boolean;
}

const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
  value,
  epistemicValue,
  history,
  darkMode,
}) => {
  const percentage = Math.round(value * 100);
  const epistemicPercentage = Math.round(epistemicValue * 100);

  // Color based on confidence level
  const getColor = (v: number) => {
    if (v >= 0.8) return 'text-green-500';
    if (v >= 0.6) return 'text-blue-500';
    if (v >= 0.4) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className={clsx('p-3 rounded-lg', darkMode ? 'bg-gray-800' : 'bg-gray-50')}>
      <h4 className={clsx('text-xs font-medium mb-2', darkMode ? 'text-gray-400' : 'text-gray-600')}>
        Confidence
      </h4>

      {/* Main confidence */}
      <div className="flex items-baseline gap-1 mb-2">
        <span className={clsx('text-3xl font-bold', getColor(value))}>{percentage}</span>
        <span className={clsx('text-sm', darkMode ? 'text-gray-500' : 'text-gray-400')}>%</span>
      </div>

      {/* Progress bar */}
      <div
        className={clsx('h-2 rounded-full overflow-hidden mb-2', darkMode ? 'bg-gray-700' : 'bg-gray-200')}
      >
        <div
          className={clsx('h-full transition-all duration-300', getColor(value).replace('text-', 'bg-'))}
          style={{ width: `${percentage}%` }}
        />
      </div>

      {/* Epistemic confidence */}
      <div className="flex justify-between items-center">
        <span className={clsx('text-xs', darkMode ? 'text-gray-500' : 'text-gray-500')}>
          Epistemic
        </span>
        <span className={clsx('text-xs font-medium', darkMode ? 'text-gray-400' : 'text-gray-600')}>
          {epistemicPercentage}%
        </span>
      </div>

      {/* Sparkline */}
      {history.length > 1 && (
        <div className="mt-2 h-6">
          <svg className="w-full h-full" viewBox={`0 0 ${history.length} 20`} preserveAspectRatio="none">
            <polyline
              fill="none"
              stroke={darkMode ? '#6b7280' : '#9ca3af'}
              strokeWidth="1"
              points={history
                .map((v, i) => `${i},${20 - v * 20}`)
                .join(' ')}
            />
          </svg>
        </div>
      )}
    </div>
  );
};

interface ReasoningStepsProps {
  steps: Array<{
    index: number;
    type: string;
    description: string;
    confidence?: number;
    status: string;
  }>;
  activeStep: number | null;
  darkMode?: boolean;
}

const ReasoningSteps: React.FC<ReasoningStepsProps> = ({ steps, activeStep, darkMode }) => {
  if (steps.length === 0) {
    return (
      <div className={clsx('text-center py-4 text-sm', darkMode ? 'text-gray-500' : 'text-gray-400')}>
        No reasoning steps yet
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {steps.map(step => (
        <div
          key={step.index}
          className={clsx(
            'p-2 rounded-lg transition-colors',
            step.index === activeStep
              ? darkMode
                ? 'bg-blue-900/30 border border-blue-700'
                : 'bg-blue-50 border border-blue-200'
              : darkMode
                ? 'bg-gray-800'
                : 'bg-gray-50',
            step.status === 'complete' && 'opacity-75'
          )}
        >
          <div className="flex items-center justify-between mb-1">
            <span
              className={clsx(
                'text-xs font-medium uppercase tracking-wide',
                step.index === activeStep
                  ? darkMode
                    ? 'text-blue-400'
                    : 'text-blue-600'
                  : darkMode
                    ? 'text-gray-400'
                    : 'text-gray-500'
              )}
            >
              {step.type}
            </span>
            {step.confidence !== undefined && (
              <span
                className={clsx(
                  'text-xs',
                  step.confidence >= 0.7
                    ? 'text-green-500'
                    : step.confidence >= 0.4
                      ? 'text-yellow-500'
                      : 'text-red-500'
                )}
              >
                {Math.round(step.confidence * 100)}%
              </span>
            )}
          </div>
          <p className={clsx('text-sm', darkMode ? 'text-gray-300' : 'text-gray-700')}>
            {step.description}
          </p>
        </div>
      ))}
    </div>
  );
};

interface SessionMetricsProps {
  totalDuration: number | null;
  tokensGenerated: number;
  contextTokens: number;
  cacheHit: boolean;
  stagesCompleted: number;
  totalStages: number;
  darkMode?: boolean;
}

const SessionMetrics: React.FC<SessionMetricsProps> = ({
  totalDuration,
  tokensGenerated,
  contextTokens,
  cacheHit,
  stagesCompleted,
  totalStages,
  darkMode,
}) => {
  const metrics = [
    {
      label: 'Duration',
      value: totalDuration ? `${totalDuration.toFixed(0)}ms` : '—',
    },
    {
      label: 'Tokens',
      value: tokensGenerated.toLocaleString(),
    },
    {
      label: 'Context',
      value: `${contextTokens.toLocaleString()} tok`,
    },
    {
      label: 'Stages',
      value: `${stagesCompleted}/${totalStages}`,
    },
  ];

  return (
    <div className={clsx('p-3 rounded-lg', darkMode ? 'bg-gray-800' : 'bg-gray-50')}>
      <div className="flex justify-between items-center mb-2">
        <h4 className={clsx('text-xs font-medium', darkMode ? 'text-gray-400' : 'text-gray-600')}>
          Session Metrics
        </h4>
        {cacheHit && (
          <span
            className={clsx(
              'text-xs px-1.5 py-0.5 rounded',
              darkMode ? 'bg-green-900/50 text-green-400' : 'bg-green-100 text-green-700'
            )}
          >
            Cache Hit
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2">
        {metrics.map(m => (
          <div key={m.label}>
            <div className={clsx('text-xs', darkMode ? 'text-gray-500' : 'text-gray-400')}>
              {m.label}
            </div>
            <div className={clsx('text-sm font-medium', darkMode ? 'text-gray-200' : 'text-gray-800')}>
              {m.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

export const ConnectedMetricsPane: React.FC<ConnectedMetricsPaneProps> = ({
  darkMode = false,
  className,
}) => {
  // Select state from store
  const confidence = useHoloLoomStore(state => state.confidence);
  const epistemicConfidence = useHoloLoomStore(state => state.epistemicConfidence);
  const confidenceHistory = useHoloLoomStore(state => state.confidenceHistory);
  const reasoningSteps = useHoloLoomStore(state => state.reasoningSteps);
  const activeReasoningStep = useHoloLoomStore(state => state.activeReasoningStep);
  const totalDuration = useHoloLoomStore(state => state.totalDuration);
  const tokensGenerated = useHoloLoomStore(state => state.tokensGenerated);
  const totalContextTokens = useHoloLoomStore(state => state.totalContextTokens);
  const cacheHit = useHoloLoomStore(state => state.cacheHit);
  const stages = useHoloLoomStore(state => state.stages);
  const streamState = useHoloLoomStore(state => state.streamState);
  const error = useHoloLoomStore(state => state.error);

  // Calculate completed stages
  const completedStages = useMemo(() => {
    return stages.filter(s => s.status === 'complete').length;
  }, [stages]);

  return (
    <div className={clsx('p-4 space-y-4', className)}>
      {/* Error Display */}
      {error && (
        <div
          className={clsx(
            'p-3 rounded-lg',
            darkMode ? 'bg-red-900/30 border border-red-700' : 'bg-red-50 border border-red-200'
          )}
        >
          <div className={clsx('text-sm font-medium', darkMode ? 'text-red-400' : 'text-red-700')}>
            Error: {error.code}
          </div>
          <div className={clsx('text-xs mt-1', darkMode ? 'text-red-300' : 'text-red-600')}>
            {error.message}
          </div>
        </div>
      )}

      {/* Confidence Gauge */}
      <ConfidenceGauge
        value={confidence}
        epistemicValue={epistemicConfidence}
        history={confidenceHistory}
        darkMode={darkMode}
      />

      {/* Reasoning Steps */}
      <div>
        <h4
          className={clsx(
            'text-xs font-medium mb-2',
            darkMode ? 'text-gray-400' : 'text-gray-600'
          )}
        >
          Reasoning Steps
        </h4>
        <ReasoningSteps
          steps={reasoningSteps}
          activeStep={activeReasoningStep}
          darkMode={darkMode}
        />
      </div>

      {/* Session Metrics */}
      <SessionMetrics
        totalDuration={totalDuration}
        tokensGenerated={tokensGenerated}
        contextTokens={totalContextTokens}
        cacheHit={cacheHit}
        stagesCompleted={completedStages}
        totalStages={stages.length}
        darkMode={darkMode}
      />

      {/* Stream State Indicator */}
      <div className="flex items-center justify-center gap-2 py-2">
        <div
          className={clsx(
            'w-2 h-2 rounded-full',
            streamState === 'streaming'
              ? 'bg-green-500 animate-pulse'
              : streamState === 'complete'
                ? 'bg-blue-500'
                : streamState === 'error'
                  ? 'bg-red-500'
                  : 'bg-gray-400'
          )}
        />
        <span className={clsx('text-xs capitalize', darkMode ? 'text-gray-400' : 'text-gray-500')}>
          {streamState}
        </span>
      </div>
    </div>
  );
};

ConnectedMetricsPane.displayName = 'ConnectedMetricsPane';

export default ConnectedMetricsPane;
