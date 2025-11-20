/**
 * Stage Timing Chart Component
 *
 * Visualizes the duration of each weaving stage with:
 * - Horizontal bar chart showing timing per stage
 * - Color coding: green (<20%), yellow (20-40%), red (>40% - bottleneck)
 * - Tooltips with exact timing
 * - Bottleneck detection and highlighting
 * - Summary statistics
 *
 * Pure CSS/SVG - no external chart libraries
 */

import React, { useMemo } from 'react';
import { AlertTriangle, Clock, TrendingDown } from 'lucide-react';
import { WeavingStep, StepStatus } from '../types';

interface StageTimingChartProps {
  steps: WeavingStep[];
  totalLatency: number;
}

interface StageData {
  step: number;
  name: string;
  duration: number;
  percentage: number;
  isBottleneck: boolean;
  status: StepStatus;
}

const STAGE_NAMES: Record<number, string> = {
  1: 'Loom Command',
  2: 'Chrono Trigger',
  3: 'Yarn Graph',
  4: 'Resonance Shed',
  5: 'Warp Space',
  6: 'Convergence Engine',
  7: 'Tool Execution',
  8: 'Spacetime Fabric',
  9: 'Reflection Buffer',
};

const BOTTLENECK_THRESHOLD = 0.4; // 40%

const getColorClass = (percentage: number): string => {
  if (percentage >= BOTTLENECK_THRESHOLD) {
    return 'bg-red-500'; // >40% - Bottleneck
  } else if (percentage >= 0.2) {
    return 'bg-yellow-500'; // 20-40% - Warning
  }
  return 'bg-green-500'; // <20% - Good
};

const getColorLabel = (percentage: number): string => {
  if (percentage >= BOTTLENECK_THRESHOLD) {
    return 'Bottleneck';
  } else if (percentage >= 0.2) {
    return 'Warning';
  }
  return 'Optimal';
};

export const StageTimingChart: React.FC<StageTimingChartProps> = ({
  steps,
  totalLatency,
}) => {
  const stageData = useMemo(() => {
    const data: StageData[] = steps
      .filter(step => step.latency_ms !== undefined && step.latency_ms > 0)
      .map(step => {
        const percentage = totalLatency > 0 ? step.latency_ms! / totalLatency : 0;
        return {
          step: step.step,
          name: STAGE_NAMES[step.step] || `Stage ${step.step}`,
          duration: step.latency_ms || 0,
          percentage,
          isBottleneck: percentage >= BOTTLENECK_THRESHOLD,
          status: step.status,
        };
      });

    return data.sort((a, b) => a.step - b.step);
  }, [steps, totalLatency]);

  const bottlenecks = useMemo(
    () => stageData.filter(s => s.isBottleneck),
    [stageData]
  );

  const avgStageDuration = useMemo(() => {
    if (stageData.length === 0) return 0;
    const sum = stageData.reduce((acc, s) => acc + s.duration, 0);
    return sum / stageData.length;
  }, [stageData]);

  const slowestStage = useMemo(() => {
    if (stageData.length === 0) return null;
    return stageData.reduce((max, s) => (s.duration > max.duration ? s : max));
  }, [stageData]);

  if (stageData.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Stage Timing</h3>
        <p className="text-gray-500 text-center py-8">
          No timing data available yet. Stages will appear as they complete.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Stage Timing Breakdown</h3>

      {/* Summary Statistics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
          <p className="text-xs text-blue-600 font-medium mb-1">Total Latency</p>
          <p className="text-lg font-bold text-blue-900">{totalLatency.toFixed(1)}ms</p>
        </div>

        <div className="bg-purple-50 p-3 rounded-lg border border-purple-200">
          <p className="text-xs text-purple-600 font-medium mb-1">Avg per Stage</p>
          <p className="text-lg font-bold text-purple-900">{avgStageDuration.toFixed(1)}ms</p>
        </div>

        {slowestStage && (
          <div className="bg-orange-50 p-3 rounded-lg border border-orange-200">
            <p className="text-xs text-orange-600 font-medium mb-1">Slowest Stage</p>
            <p className="text-sm font-bold text-orange-900 truncate">
              {slowestStage.name.split(' ')[0]}
            </p>
            <p className="text-xs text-orange-700">{slowestStage.duration.toFixed(1)}ms</p>
          </div>
        )}

        <div className="bg-green-50 p-3 rounded-lg border border-green-200">
          <p className="text-xs text-green-600 font-medium mb-1">Stages Completed</p>
          <p className="text-lg font-bold text-green-900">
            {stageData.filter(s => s.status === StepStatus.COMPLETED).length}/{stageData.length}
          </p>
        </div>
      </div>

      {/* Bottleneck Alert */}
      {bottlenecks.length > 0 && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-sm font-semibold text-red-900">
                {bottlenecks.length} Bottleneck{bottlenecks.length !== 1 ? 's' : ''} Detected
              </p>
              <p className="text-xs text-red-700 mt-1">
                {bottlenecks.map(s => s.name).join(', ')} exceeding 40% of total time
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Stage Timing Bars */}
      <div className="space-y-3">
        {stageData.map((stage) => {
          const barColor = getColorClass(stage.percentage);
          const label = getColorLabel(stage.percentage);
          const percentDisplay = (stage.percentage * 100).toFixed(1);

          return (
            <div key={stage.step} className="group">
              {/* Stage Name and Duration */}
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <span className="text-sm font-medium text-gray-700 w-6 text-center">
                    {stage.step}
                  </span>
                  <span className="text-sm font-medium text-gray-900 truncate">
                    {stage.name}
                  </span>
                </div>
                <div className="flex items-center gap-2 ml-2 flex-shrink-0">
                  <span className="text-sm font-semibold text-gray-900 w-12 text-right">
                    {stage.duration.toFixed(1)}ms
                  </span>
                  <span className="text-xs font-medium text-gray-500 w-10 text-right">
                    {percentDisplay}%
                  </span>
                  {stage.isBottleneck && (
                    <span className="inline-block px-2 py-0.5 bg-red-100 text-red-700 text-xs font-semibold rounded">
                      ⚠ {label}
                    </span>
                  )}
                </div>
              </div>

              {/* Progress Bar */}
              <div className="relative w-full h-6 bg-gray-100 rounded-full overflow-hidden border border-gray-200">
                {/* Bar Fill */}
                <div
                  className={`h-full transition-all duration-500 flex items-center justify-end pr-2 ${barColor} relative`}
                  style={{ width: `${Math.max(stage.percentage * 100, 2)}%` }}
                  title={`${stage.name}: ${stage.duration.toFixed(1)}ms (${percentDisplay}%)`}
                >
                  {stage.percentage > 0.15 && (
                    <span className="text-xs font-bold text-white drop-shadow">
                      {percentDisplay}%
                    </span>
                  )}
                </div>

                {/* Invisible tooltip trigger area */}
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                  <div className="h-full flex items-center justify-center">
                    <div className="bg-gray-900 text-white text-xs py-1 px-2 rounded whitespace-nowrap">
                      {stage.duration.toFixed(2)}ms
                    </div>
                  </div>
                </div>
              </div>

              {/* Status Indicator */}
              <div className="mt-1 flex items-center gap-1">
                {stage.status === StepStatus.COMPLETED && (
                  <span className="text-xs text-green-600 font-medium">✓ Completed</span>
                )}
                {stage.status === StepStatus.IN_PROGRESS && (
                  <span className="text-xs text-blue-600 font-medium animate-pulse">⋯ In Progress</span>
                )}
                {stage.status === StepStatus.WAITING && (
                  <span className="text-xs text-gray-400 font-medium">○ Waiting</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Performance Summary */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="flex items-center gap-2 mb-3">
          <TrendingDown className="w-4 h-4 text-gray-600" />
          <p className="text-sm font-semibold text-gray-900">Performance Summary</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
          <div>
            <p className="text-gray-600 mb-1">Fastest Stage</p>
            {stageData.length > 0 && (
              <p className="font-medium text-gray-900">
                {stageData.reduce((min, s) => (s.duration < min.duration ? s : min)).name}
              </p>
            )}
          </div>

          <div>
            <p className="text-gray-600 mb-1">Median Duration</p>
            {stageData.length > 0 && (
              <p className="font-medium text-gray-900">
                {stageData.length > 0
                  ? stageData
                      .map(s => s.duration)
                      .sort((a, b) => a - b)[Math.floor(stageData.length / 2)]
                      .toFixed(1)
                  : '0'}
                ms
              </p>
            )}
          </div>

          <div>
            <p className="text-gray-600 mb-1">Optimization Potential</p>
            {bottlenecks.length > 0 ? (
              <p className="font-medium text-red-600">
                {(
                  (bottlenecks.reduce((sum, s) => sum + s.duration, 0) / totalLatency) *
                  100
                ).toFixed(1)}
                % reduction possible
              </p>
            ) : (
              <p className="font-medium text-green-600">Optimal</p>
            )}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <p className="text-xs font-semibold text-gray-900 mb-3">Color Legend</p>
        <div className="grid grid-cols-3 gap-3 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="text-gray-600">&lt;20% (Optimal)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500" />
            <span className="text-gray-600">20-40% (Caution)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span className="text-gray-600">&gt;40% (Bottleneck)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
