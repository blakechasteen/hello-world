'use client';

import type { TimeRange } from '../../app/analytics/page';

interface MetricPoint {
  timestamp: number;
  value: number;
}

interface Anomaly {
  index: number;
  type: 'sudden_drop' | 'prolonged_low' | 'high_variance' | 'cache_miss_cluster';
  description: string;
}

interface ConfidenceTrajectoryProps {
  data: MetricPoint[];
  timeRange: TimeRange;
  anomalies: Anomaly[];
}

export function ConfidenceTrajectory({ data, timeRange, anomalies }: ConfidenceTrajectoryProps) {
  if (data.length === 0) return null;

  const width = 500;
  const height = 180;
  const padding = { top: 20, right: 20, bottom: 30, left: 50 };

  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const values = data.map((d) => d.value);
  const minValue = 0;
  const maxValue = 1;

  const xScale = (index: number) =>
    padding.left + (index / (data.length - 1)) * chartWidth;
  const yScale = (value: number) =>
    padding.top + chartHeight - ((value - minValue) / (maxValue - minValue)) * chartHeight;

  // Calculate mean and std bands
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);

  // Create path
  const linePath = data
    .map((point, index) => {
      const x = xScale(index);
      const y = yScale(point.value);
      return index === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
    })
    .join(' ');

  // Create gradient fill area
  const areaPath = `${linePath} L ${xScale(data.length - 1)} ${padding.top + chartHeight} L ${padding.left} ${padding.top + chartHeight} Z`;

  // Y-axis ticks for confidence (0-100%)
  const yTickValues = [0, 0.25, 0.5, 0.75, 1];

  // X-axis labels
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    if (timeRange === '1h' || timeRange === '6h') {
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } else if (timeRange === '24h') {
      return date.toLocaleTimeString('en-US', { hour: '2-digit' });
    } else {
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  };

  const xLabelCount = 5;
  const xLabels = Array.from({ length: xLabelCount }, (_, i) => {
    const index = Math.floor((i / (xLabelCount - 1)) * (data.length - 1));
    return {
      x: xScale(index),
      label: formatTime(data[index].timestamp),
    };
  });

  // Threshold lines
  const thresholdHigh = 0.85;
  const thresholdLow = 0.65;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-44">
      <defs>
        <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgb(16, 185, 129)" stopOpacity="0.2" />
          <stop offset="100%" stopColor="rgb(16, 185, 129)" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Standard deviation band */}
      <rect
        x={padding.left}
        y={yScale(mean + std)}
        width={chartWidth}
        height={yScale(mean - std) - yScale(mean + std)}
        fill="currentColor"
        fillOpacity="0.05"
      />

      {/* Grid lines */}
      {yTickValues.map((tick, index) => (
        <line
          key={index}
          x1={padding.left}
          y1={yScale(tick)}
          x2={width - padding.right}
          y2={yScale(tick)}
          stroke="currentColor"
          strokeOpacity="0.1"
          strokeDasharray="4 4"
        />
      ))}

      {/* Threshold lines */}
      <line
        x1={padding.left}
        y1={yScale(thresholdHigh)}
        x2={width - padding.right}
        y2={yScale(thresholdHigh)}
        stroke="rgb(16, 185, 129)"
        strokeWidth="1"
        strokeDasharray="4 2"
        opacity="0.5"
      />
      <line
        x1={padding.left}
        y1={yScale(thresholdLow)}
        x2={width - padding.right}
        y2={yScale(thresholdLow)}
        stroke="rgb(239, 68, 68)"
        strokeWidth="1"
        strokeDasharray="4 2"
        opacity="0.5"
      />

      {/* Y-axis labels */}
      {yTickValues.map((tick, index) => (
        <text
          key={index}
          x={padding.left - 8}
          y={yScale(tick)}
          textAnchor="end"
          dominantBaseline="middle"
          className="fill-fg-tertiary text-xs"
        >
          {(tick * 100).toFixed(0)}%
        </text>
      ))}

      {/* X-axis labels */}
      {xLabels.map((label, index) => (
        <text
          key={index}
          x={label.x}
          y={height - 8}
          textAnchor="middle"
          className="fill-fg-tertiary text-xs"
        >
          {label.label}
        </text>
      ))}

      {/* Area fill */}
      <path d={areaPath} fill="url(#confidenceGradient)" />

      {/* Mean line */}
      <line
        x1={padding.left}
        y1={yScale(mean)}
        x2={width - padding.right}
        y2={yScale(mean)}
        stroke="rgb(139, 92, 246)"
        strokeWidth="1"
        strokeDasharray="2 2"
        opacity="0.5"
      />

      {/* Main line */}
      <path
        d={linePath}
        fill="none"
        stroke="rgb(16, 185, 129)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Anomaly markers */}
      {anomalies.map((anomaly, index) => {
        if (anomaly.index >= data.length) return null;
        const x = xScale(anomaly.index);
        const y = yScale(data[anomaly.index].value);
        const color = anomaly.type === 'sudden_drop' ? '#EF4444' : '#F59E0B';

        return (
          <g key={index}>
            <circle cx={x} cy={y} r="6" fill={color} fillOpacity="0.2" />
            <circle cx={x} cy={y} r="3" fill={color} />
          </g>
        );
      })}

      {/* Current value dot */}
      <circle
        cx={xScale(data.length - 1)}
        cy={yScale(data[data.length - 1].value)}
        r="4"
        fill="rgb(16, 185, 129)"
        stroke="white"
        strokeWidth="2"
      />
    </svg>
  );
}
