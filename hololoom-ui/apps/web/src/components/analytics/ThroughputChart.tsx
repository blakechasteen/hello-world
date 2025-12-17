'use client';

import type { TimeRange } from '../../app/analytics/page';

interface MetricPoint {
  timestamp: number;
  value: number;
}

interface ThroughputChartProps {
  data: MetricPoint[];
  timeRange: TimeRange;
}

export function ThroughputChart({ data, timeRange }: ThroughputChartProps) {
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

  // Create bar data (sample every N points for bars)
  const barCount = Math.min(24, data.length);
  const barWidth = (chartWidth / barCount) * 0.7;

  const bars = Array.from({ length: barCount }, (_, i) => {
    const dataIndex = Math.floor((i / barCount) * data.length);
    const value = data[dataIndex].value;
    return {
      x: padding.left + (i / barCount) * chartWidth + barWidth * 0.15,
      y: yScale(value),
      height: padding.top + chartHeight - yScale(value),
      value,
    };
  });

  // Y-axis ticks
  const yTicks = 5;
  const yTickValues = Array.from({ length: yTicks }, (_, i) => {
    return minValue + (i / (yTicks - 1)) * (maxValue - minValue);
  });

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
      x: padding.left + (i / (xLabelCount - 1)) * chartWidth,
      label: formatTime(data[index].timestamp),
    };
  });

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-48">
      <defs>
        <linearGradient id="throughputGradient" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgb(16, 185, 129)" stopOpacity="0.8" />
          <stop offset="100%" stopColor="rgb(16, 185, 129)" stopOpacity="0.3" />
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

      {/* Bars */}
      {bars.map((bar, index) => (
        <rect
          key={index}
          x={bar.x}
          y={bar.y}
          width={barWidth}
          height={bar.height}
          fill="url(#throughputGradient)"
          rx="2"
        />
      ))}

      {/* Average line */}
      <line
        x1={padding.left}
        y1={yScale((minValue + maxValue) / 2)}
        x2={width - padding.right}
        y2={yScale((minValue + maxValue) / 2)}
        stroke="rgb(16, 185, 129)"
        strokeWidth="1.5"
        strokeDasharray="6 4"
        opacity="0.6"
      />
    </svg>
  );
}
