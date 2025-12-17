'use client';

import type { TimeRange } from '../../app/analytics/page';

interface MetricPoint {
  timestamp: number;
  value: number;
}

interface LatencyChartProps {
  data: MetricPoint[];
  timeRange: TimeRange;
}

export function LatencyChart({ data, timeRange }: LatencyChartProps) {
  if (data.length === 0) return null;

  const width = 500;
  const height = 200;
  const padding = { top: 20, right: 20, bottom: 30, left: 50 };

  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const values = data.map((d) => d.value);
  const minValue = Math.min(...values) * 0.9;
  const maxValue = Math.max(...values) * 1.1;

  const xScale = (index: number) =>
    padding.left + (index / (data.length - 1)) * chartWidth;
  const yScale = (value: number) =>
    padding.top + chartHeight - ((value - minValue) / (maxValue - minValue)) * chartHeight;

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

  // Y-axis ticks
  const yTicks = 5;
  const yTickValues = Array.from({ length: yTicks }, (_, i) => {
    return minValue + (i / (yTicks - 1)) * (maxValue - minValue);
  });

  // X-axis labels based on time range
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

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-48">
      <defs>
        <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgb(139, 92, 246)" stopOpacity="0.3" />
          <stop offset="100%" stopColor="rgb(139, 92, 246)" stopOpacity="0" />
        </linearGradient>
      </defs>

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
          {tick.toFixed(0)}
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
      <path d={areaPath} fill="url(#latencyGradient)" />

      {/* Line */}
      <path
        d={linePath}
        fill="none"
        stroke="rgb(139, 92, 246)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Hover dots (showing last point) */}
      <circle
        cx={xScale(data.length - 1)}
        cy={yScale(data[data.length - 1].value)}
        r="4"
        fill="rgb(139, 92, 246)"
        stroke="white"
        strokeWidth="2"
      />
    </svg>
  );
}
