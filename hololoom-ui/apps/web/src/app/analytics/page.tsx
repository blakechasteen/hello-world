'use client';

import { useState, useRef } from 'react';
import { Navigation } from '../../components/Navigation';
import { Button, Badge, Card } from '@hololoom/design-system';
import { useStats } from '@hololoom/api-client';
import {
  PerformanceOverview,
  LatencyChart,
  ThroughputChart,
  CacheEffectiveness,
  LearningMetrics,
  ConfidenceTrajectory,
  UsageBreakdown,
  SystemHealth,
} from '../../components/analytics';

export type TimeRange = '1h' | '6h' | '24h' | '7d' | '30d';

export interface MetricPoint {
  timestamp: number;
  value: number;
}

export interface PerformanceMetrics {
  avgLatency: number;
  p95Latency: number;
  p99Latency: number;
  throughput: number;
  cacheHitRate: number;
  errorRate: number;
}

export interface LearningStats {
  thompsonAlpha: number;
  thompsonBeta: number;
  patternsLearned: number;
  hotPatterns: number;
  coldPatterns: number;
  adaptiveAccuracy: number;
}

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const { data: stats, source: statsSource } = useStats(autoRefresh ? 10000 : 0);

  // Rolling time-series accumulator — appends live stats snapshots
  const timeSeriesRef = useRef<{
    latency: MetricPoint[];
    throughput: MetricPoint[];
    confidence: MetricPoint[];
  }>({
    latency: generateMockTimeSeries(timeRange, 120, 180),
    throughput: generateMockTimeSeries(timeRange, 700, 1000),
    confidence: generateMockTimeSeries(timeRange, 0.7, 0.95),
  });

  // Append live data point when stats arrive
  if (stats && statsSource === 'live') {
    const now = Date.now();
    const ts = timeSeriesRef.current;
    const lastLatency = ts.latency[ts.latency.length - 1];
    if (!lastLatency || now - lastLatency.timestamp > 5000) {
      ts.latency.push({ timestamp: now, value: stats.avgLatencyMs });
      ts.throughput.push({ timestamp: now, value: stats.queriesLastHour });
      ts.confidence.push({ timestamp: now, value: stats.avgConfidence });
      // Keep max 200 points
      if (ts.latency.length > 200) { ts.latency.shift(); ts.throughput.shift(); ts.confidence.shift(); }
    }
  }

  // Live metrics or fallback
  const performanceMetrics: PerformanceMetrics = stats ? {
    avgLatency: stats.avgLatencyMs,
    p95Latency: stats.avgLatencyMs * 1.8,  // estimate until backend exposes percentiles
    p99Latency: stats.avgLatencyMs * 3.0,
    throughput: stats.queriesLastHour,
    cacheHitRate: stats.cacheHitRate,
    errorRate: 0.012,
  } : {
    avgLatency: 125.4,
    p95Latency: 245.8,
    p99Latency: 412.3,
    throughput: 847,
    cacheHitRate: 0.78,
    errorRate: 0.012,
  };

  const learningStats: LearningStats = stats?.learning ? {
    thompsonAlpha: 142.5,
    thompsonBeta: 23.8,
    patternsLearned: stats.learning.patternsLearned,
    hotPatterns: 156,
    coldPatterns: 89,
    adaptiveAccuracy: stats.learning.successRate,
  } : {
    thompsonAlpha: 142.5,
    thompsonBeta: 23.8,
    patternsLearned: 1847,
    hotPatterns: 156,
    coldPatterns: 89,
    adaptiveAccuracy: 0.943,
  };

  const latencyHistory = timeSeriesRef.current.latency;
  const throughputHistory = timeSeriesRef.current.throughput;
  const confidenceHistory = timeSeriesRef.current.confidence;

  return (
    <div className="min-h-screen bg-bg-primary">
      <Navigation />

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold text-fg-primary mb-1">
              System Analytics
            </h1>
            <p className="text-fg-tertiary">
              Real-time performance monitoring and learning system insights
            </p>
          </div>

          <div className="flex items-center gap-4">
            {/* Auto Refresh Toggle */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="w-4 h-4 rounded border-border-primary bg-bg-secondary text-cosmic-nebula focus:ring-cosmic-nebula"
              />
              <span className="text-sm text-fg-secondary">Auto-refresh</span>
            </label>

            {/* Time Range Selector */}
            <div className="flex items-center bg-bg-secondary rounded-lg p-1">
              {(['1h', '6h', '24h', '7d', '30d'] as TimeRange[]).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    timeRange === range
                      ? 'bg-cosmic-nebula text-white'
                      : 'text-fg-tertiary hover:text-fg-primary'
                  }`}
                >
                  {range}
                </button>
              ))}
            </div>

            {/* Export Button */}
            <Button variant="secondary" size="sm">
              <ExportIcon />
              <span className="ml-1.5">Export</span>
            </Button>
          </div>
        </div>

        {/* Performance Overview Cards */}
        <PerformanceOverview metrics={performanceMetrics} />

        {/* Main Charts Grid */}
        <div className="grid grid-cols-2 gap-6 mt-6">
          {/* Latency Chart */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-fg-primary">Response Latency</h3>
              <Badge variant="secondary" size="sm">ms</Badge>
            </div>
            <LatencyChart data={latencyHistory} timeRange={timeRange} />
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-border-primary">
              <MetricStat label="Avg" value={`${performanceMetrics.avgLatency.toFixed(1)}ms`} />
              <MetricStat label="P95" value={`${performanceMetrics.p95Latency.toFixed(1)}ms`} />
              <MetricStat label="P99" value={`${performanceMetrics.p99Latency.toFixed(1)}ms`} />
            </div>
          </Card>

          {/* Throughput Chart */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-fg-primary">Throughput</h3>
              <Badge variant="secondary" size="sm">queries/min</Badge>
            </div>
            <ThroughputChart data={throughputHistory} timeRange={timeRange} />
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-border-primary">
              <MetricStat label="Current" value={`${performanceMetrics.throughput}`} />
              <MetricStat label="Peak" value="1,247" />
              <MetricStat label="Avg" value="891" />
            </div>
          </Card>
        </div>

        {/* Second Row */}
        <div className="grid grid-cols-3 gap-6 mt-6">
          {/* Cache Effectiveness */}
          <CacheEffectiveness
            hitRate={performanceMetrics.cacheHitRate}
            totalQueries={stats?.totalQueries ?? 12847}
            cacheHits={Math.round((stats?.totalQueries ?? 12847) * performanceMetrics.cacheHitRate)}
            avgCachedLatency={12.4}
            avgUncachedLatency={performanceMetrics.avgLatency}
          />

          {/* Learning Metrics */}
          <LearningMetrics stats={learningStats} />

          {/* System Health */}
          <SystemHealth
            cpuUsage={0.42}
            memoryUsage={0.68}
            diskUsage={0.35}
            networkLatency={8.2}
          />
        </div>

        {/* Third Row - Full Width */}
        <div className="grid grid-cols-2 gap-6 mt-6">
          {/* Confidence Trajectory */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-fg-primary">Confidence Trajectory</h3>
              <Badge variant="success" size="sm">Stable</Badge>
            </div>
            <ConfidenceTrajectory
              data={confidenceHistory}
              timeRange={timeRange}
              anomalies={[]}
            />
          </Card>

          {/* Usage Breakdown */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-fg-primary">Usage Breakdown</h3>
              <Badge variant="secondary" size="sm">Last {timeRange}</Badge>
            </div>
            <UsageBreakdown timeRange={timeRange} />
          </Card>
        </div>

        {/* Recent Events */}
        <Card className="mt-6 p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-fg-primary">Recent System Events</h3>
            <Button variant="secondary" size="sm">View All</Button>
          </div>
          <RecentEvents />
        </Card>
      </main>
    </div>
  );
}

function MetricStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <p className="text-xs text-fg-tertiary mb-0.5">{label}</p>
      <p className="text-sm font-semibold text-fg-primary">{value}</p>
    </div>
  );
}

function RecentEvents() {
  const events = [
    { type: 'info', message: 'Pattern deployment completed (42 new patterns)', time: '2 min ago' },
    { type: 'success', message: 'Cache hit rate exceeded 80% threshold', time: '15 min ago' },
    { type: 'warning', message: 'P99 latency spike detected (412ms)', time: '32 min ago' },
    { type: 'info', message: 'Thompson Sampling priors updated', time: '1 hr ago' },
    { type: 'success', message: 'Adaptive learning validation passed (94.3% accuracy)', time: '2 hr ago' },
  ];

  const typeStyles = {
    info: 'bg-cosmic-nebula/10 text-cosmic-nebula',
    success: 'bg-safety-safe/10 text-safety-safe',
    warning: 'bg-safety-risky/10 text-safety-risky',
    error: 'bg-safety-critical/10 text-safety-critical',
  };

  return (
    <div className="space-y-3">
      {events.map((event, index) => (
        <div
          key={index}
          className="flex items-center gap-3 p-3 rounded-lg bg-bg-secondary/50"
        >
          <span
            className={`px-2 py-0.5 rounded text-xs font-medium ${
              typeStyles[event.type as keyof typeof typeStyles]
            }`}
          >
            {event.type.toUpperCase()}
          </span>
          <span className="flex-1 text-sm text-fg-primary">{event.message}</span>
          <span className="text-xs text-fg-tertiary">{event.time}</span>
        </div>
      ))}
    </div>
  );
}

function ExportIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  );
}

// Helper function to generate mock time series data
function generateMockTimeSeries(
  timeRange: TimeRange,
  min: number,
  max: number
): MetricPoint[] {
  const points: MetricPoint[] = [];
  const now = Date.now();

  const intervals: Record<TimeRange, { count: number; step: number }> = {
    '1h': { count: 60, step: 60 * 1000 },
    '6h': { count: 72, step: 5 * 60 * 1000 },
    '24h': { count: 96, step: 15 * 60 * 1000 },
    '7d': { count: 168, step: 60 * 60 * 1000 },
    '30d': { count: 180, step: 4 * 60 * 60 * 1000 },
  };

  const { count, step } = intervals[timeRange];

  for (let i = count; i >= 0; i--) {
    const variance = (max - min) * 0.3;
    const baseValue = (max + min) / 2;
    const value = baseValue + (Math.random() - 0.5) * variance * 2;

    points.push({
      timestamp: now - i * step,
      value: Math.max(min, Math.min(max, value)),
    });
  }

  return points;
}
