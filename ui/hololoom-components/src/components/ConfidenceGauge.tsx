/**
 * ConfidenceGauge Component
 *
 * Displays current confidence as an arc gauge with optional sparkline history.
 * Follows Tufte principles: maximize data-ink ratio, show the data.
 */

import React, { useMemo, useCallback } from 'react';
import { clsx } from 'clsx';
import type { ConfidenceGaugeProps } from '../types';

// ============================================================================
// Constants
// ============================================================================

const SIZE_CONFIG = {
  sm: { width: 80, height: 48, fontSize: 12, strokeWidth: 6, radius: 32 },
  md: { width: 120, height: 72, fontSize: 16, strokeWidth: 8, radius: 48 },
  lg: { width: 160, height: 96, fontSize: 20, strokeWidth: 10, radius: 64 },
} as const;

const CONFIDENCE_COLORS = {
  low: '#ef4444',      // red-500
  medium: '#f59e0b',   // amber-500
  high: '#22c55e',     // green-500
  excellent: '#10b981', // emerald-500
} as const;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get color based on confidence value
 */
function getConfidenceColor(value: number): string {
  if (value < 0.3) return CONFIDENCE_COLORS.low;
  if (value < 0.6) return CONFIDENCE_COLORS.medium;
  if (value < 0.85) return CONFIDENCE_COLORS.high;
  return CONFIDENCE_COLORS.excellent;
}

/**
 * Get confidence label
 */
function getConfidenceLabel(value: number): string {
  if (value < 0.3) return 'Low';
  if (value < 0.6) return 'Moderate';
  if (value < 0.85) return 'High';
  return 'Excellent';
}

/**
 * Calculate arc path for gauge
 */
function describeArc(
  x: number,
  y: number,
  radius: number,
  startAngle: number,
  endAngle: number
): string {
  const start = polarToCartesian(x, y, radius, endAngle);
  const end = polarToCartesian(x, y, radius, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? '0' : '1';

  return [
    'M', start.x, start.y,
    'A', radius, radius, 0, largeArcFlag, 0, end.x, end.y,
  ].join(' ');
}

function polarToCartesian(
  centerX: number,
  centerY: number,
  radius: number,
  angleInDegrees: number
): { x: number; y: number } {
  const angleInRadians = ((angleInDegrees - 90) * Math.PI) / 180;
  return {
    x: centerX + radius * Math.cos(angleInRadians),
    y: centerY + radius * Math.sin(angleInRadians),
  };
}

// ============================================================================
// Sparkline Sub-component
// ============================================================================

interface SparklineProps {
  data: number[];
  width: number;
  height: number;
  color: string;
}

const Sparkline: React.FC<SparklineProps> = React.memo(({ data, width, height, color }) => {
  if (data.length < 2) return null;

  const points = useMemo(() => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    return data.map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');
  }, [data, width, height]);

  return (
    <svg
      width={width}
      height={height}
      className="sparkline"
      aria-hidden="true"
    >
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.6}
      />
      {/* Endpoint marker */}
      {data.length > 0 && (
        <circle
          cx={width}
          cy={height - ((data[data.length - 1] - Math.min(...data)) / (Math.max(...data) - Math.min(...data) || 1)) * height}
          r={2}
          fill={color}
        />
      )}
    </svg>
  );
});

Sparkline.displayName = 'Sparkline';

// ============================================================================
// Main Component
// ============================================================================

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
  confidence,
  epistemicConfidence,
  history = [],
  size = 'md',
  showValue = true,
  showTrend = true,
  label,
  animated = true,
  className,
}) => {
  const config = SIZE_CONFIG[size];
  const color = getConfidenceColor(confidence);
  const confidenceLabel = getConfidenceLabel(confidence);

  // Calculate arc angles (semicircle: -90 to 90 degrees = 180 degree arc)
  const startAngle = -90;
  const endAngle = 90;
  const valueAngle = startAngle + (confidence * 180);

  // Arc paths
  const backgroundArc = useMemo(
    () => describeArc(config.width / 2, config.height, config.radius, startAngle, endAngle),
    [config]
  );

  const valueArc = useMemo(
    () => describeArc(config.width / 2, config.height, config.radius, startAngle, valueAngle),
    [config, valueAngle]
  );

  // Epistemic confidence indicator (inner arc)
  const epistemicArc = useMemo(() => {
    if (epistemicConfidence === undefined) return null;
    const innerRadius = config.radius - config.strokeWidth - 2;
    const epistemicAngle = startAngle + (epistemicConfidence * 180);
    return describeArc(config.width / 2, config.height, innerRadius, startAngle, epistemicAngle);
  }, [config, epistemicConfidence]);

  // Format percentage
  const percentageValue = Math.round(confidence * 100);

  return (
    <div
      className={clsx(
        'confidence-gauge inline-flex flex-col items-center',
        className
      )}
      role="meter"
      aria-valuenow={percentageValue}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label || `Confidence: ${percentageValue}%`}
    >
      <svg
        width={config.width}
        height={config.height + 8}
        viewBox={`0 0 ${config.width} ${config.height + 8}`}
        className="overflow-visible"
      >
        {/* Background arc */}
        <path
          d={backgroundArc}
          fill="none"
          stroke="currentColor"
          strokeWidth={config.strokeWidth}
          strokeLinecap="round"
          className="text-gray-200 dark:text-gray-700"
        />

        {/* Value arc */}
        <path
          d={valueArc}
          fill="none"
          stroke={color}
          strokeWidth={config.strokeWidth}
          strokeLinecap="round"
          className={clsx(animated && 'transition-all duration-300 ease-out')}
          style={{
            filter: 'drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1))',
          }}
        />

        {/* Epistemic confidence arc (inner) */}
        {epistemicArc && (
          <path
            d={epistemicArc}
            fill="none"
            stroke={color}
            strokeWidth={config.strokeWidth / 2}
            strokeLinecap="round"
            opacity={0.4}
            className={clsx(animated && 'transition-all duration-300 ease-out')}
          />
        )}

        {/* Center value */}
        {showValue && (
          <text
            x={config.width / 2}
            y={config.height - config.radius / 3}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={config.fontSize}
            fontWeight={600}
            fill="currentColor"
            className="text-gray-900 dark:text-gray-100"
          >
            {percentageValue}%
          </text>
        )}

        {/* Label */}
        <text
          x={config.width / 2}
          y={config.height + 6}
          textAnchor="middle"
          dominantBaseline="hanging"
          fontSize={config.fontSize * 0.65}
          fill="currentColor"
          className="text-gray-500 dark:text-gray-400"
        >
          {label || confidenceLabel}
        </text>
      </svg>

      {/* Sparkline trend */}
      {showTrend && history.length > 1 && (
        <div className="mt-1">
          <Sparkline
            data={history}
            width={config.width * 0.8}
            height={16}
            color={color}
          />
        </div>
      )}

      {/* Epistemic confidence indicator */}
      {epistemicConfidence !== undefined && (
        <div
          className="text-xs text-gray-500 dark:text-gray-400 mt-0.5"
          title="Epistemic confidence (meta-uncertainty)"
        >
          ε: {Math.round(epistemicConfidence * 100)}%
        </div>
      )}
    </div>
  );
};

ConfidenceGauge.displayName = 'ConfidenceGauge';

export default ConfidenceGauge;

// ============================================================================
// Connected Component (uses store)
// ============================================================================

import { useHoloLoomStore, selectConfidence, selectEpistemicConfidence, selectConfidenceHistory } from '../store';

export interface ConnectedConfidenceGaugeProps extends Omit<ConfidenceGaugeProps, 'confidence' | 'epistemicConfidence' | 'history'> {}

export const ConnectedConfidenceGauge: React.FC<ConnectedConfidenceGaugeProps> = (props) => {
  const confidence = useHoloLoomStore(selectConfidence);
  const epistemicConfidence = useHoloLoomStore(selectEpistemicConfidence);
  const history = useHoloLoomStore(selectConfidenceHistory);

  return (
    <ConfidenceGauge
      confidence={confidence}
      epistemicConfidence={epistemicConfidence}
      history={history}
      {...props}
    />
  );
};

ConnectedConfidenceGauge.displayName = 'ConnectedConfidenceGauge';
