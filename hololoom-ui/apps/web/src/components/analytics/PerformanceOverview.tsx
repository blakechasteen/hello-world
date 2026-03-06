'use client';

import { Card } from '@hololoom/design-system';

interface PerformanceMetrics {
  avgLatency: number;
  p95Latency: number;
  p99Latency: number;
  throughput: number;
  cacheHitRate: number;
  errorRate: number;
}

interface PerformanceOverviewProps {
  metrics: PerformanceMetrics;
}

export function PerformanceOverview({ metrics }: PerformanceOverviewProps) {
  const cards: MetricCardProps[] = [
    {
      label: 'Avg Latency',
      value: `${metrics.avgLatency.toFixed(1)}ms`,
      trend: -8.5,
      sparkline: [145, 138, 132, 128, 125],
      status: metrics.avgLatency < 150 ? 'good' : metrics.avgLatency < 250 ? 'warning' : 'critical',
    },
    {
      label: 'Throughput',
      value: `${metrics.throughput.toLocaleString()}`,
      unit: 'queries/min',
      trend: 12.3,
      sparkline: [720, 780, 810, 835, 847],
      status: 'good',
    },
    {
      label: 'Cache Hit Rate',
      value: `${(metrics.cacheHitRate * 100).toFixed(1)}%`,
      trend: 3.2,
      sparkline: [72, 74, 75, 76, 78],
      status: metrics.cacheHitRate > 0.7 ? 'good' : metrics.cacheHitRate > 0.5 ? 'warning' : 'critical',
    },
    {
      label: 'Error Rate',
      value: `${(metrics.errorRate * 100).toFixed(2)}%`,
      trend: -15.8,
      sparkline: [1.8, 1.5, 1.4, 1.3, 1.2],
      status: metrics.errorRate < 0.02 ? 'good' : metrics.errorRate < 0.05 ? 'warning' : 'critical',
    },
  ];

  return (
    <div className="grid grid-cols-4 gap-4">
      {cards.map((card) => (
        <MetricCard key={card.label} {...card} />
      ))}
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  trend: number;
  sparkline: number[];
  status: 'good' | 'warning' | 'critical';
}

function MetricCard({ label, value, unit, trend, sparkline, status }: MetricCardProps) {
  const statusColors = {
    good: 'text-safety-safe',
    warning: 'text-safety-risky',
    critical: 'text-safety-critical',
  };

  const statusDots = {
    good: 'bg-safety-safe',
    warning: 'bg-safety-risky',
    critical: 'bg-safety-critical',
  };

  return (
    <Card className="p-4">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${statusDots[status]}`} />
          <span className="text-sm text-fg-tertiary">{label}</span>
        </div>
        <TrendIndicator value={trend} />
      </div>

      <div className="flex items-baseline gap-1.5 mb-3">
        <span className={`text-2xl font-bold ${statusColors[status]}`}>{value}</span>
        {unit && <span className="text-xs text-fg-tertiary">{unit}</span>}
      </div>

      <Sparkline data={sparkline} status={status} />
    </Card>
  );
}

function TrendIndicator({ value }: { value: number }) {
  const isPositive = value >= 0;
  const color = isPositive ? 'text-safety-safe' : 'text-safety-critical';

  return (
    <div className={`flex items-center gap-0.5 text-xs font-medium ${color}`}>
      {isPositive ? (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="18 15 12 9 6 15" />
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="6 9 12 15 18 9" />
        </svg>
      )}
      {Math.abs(value).toFixed(1)}%
    </div>
  );
}

function Sparkline({ data, status }: { data: number[]; status: 'good' | 'warning' | 'critical' }) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const width = 100;
  const height = 24;
  const padding = 2;

  const points = data.map((value, index) => {
    const x = padding + (index / (data.length - 1)) * (width - padding * 2);
    const y = height - padding - ((value - min) / range) * (height - padding * 2);
    return `${x},${y}`;
  }).join(' ');

  const strokeColors = {
    good: '#10B981',
    warning: '#F59E0B',
    critical: '#EF4444',
  };

  return (
    <svg width={width} height={height} className="w-full">
      <polyline
        points={points}
        fill="none"
        stroke={strokeColors[status]}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* End dot */}
      <circle
        cx={width - padding}
        cy={height - padding - ((data[data.length - 1] - min) / range) * (height - padding * 2)}
        r="2.5"
        fill={strokeColors[status]}
      />
    </svg>
  );
}
