'use client';

import { Card } from '@hololoom/design-system';
import { useWeaving } from '@/contexts/WeavingContext';
import { useStats } from '@hololoom/api-client';
import { useRef } from 'react';

const STAGE_NAMES: Record<number, string> = {
  1: 'Loom Command',
  2: 'Chrono Trigger',
  3: 'Yarn Graph',
  4: 'Resonance Shed',
  5: 'Warp Space',
  6: 'Memory Retrieval',
  7: 'Convergence',
  8: 'Tool Execution',
  9: 'Spacetime Fabric',
};

// --- Sparkline (extracted from PerformanceOverview) ---

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const width = 80;
  const height = 20;
  const pad = 2;

  const points = data
    .map((v, i) => {
      const x = pad + (i / (data.length - 1)) * (width - pad * 2);
      const y = height - pad - ((v - min) / range) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg width={width} height={height} className="w-full">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        cx={width - pad}
        cy={
          height -
          pad -
          ((data[data.length - 1] - min) / range) * (height - pad * 2)
        }
        r="2"
        fill={color}
      />
    </svg>
  );
}

// --- Metric Row ---

interface MetricRowProps {
  label: string;
  value: string;
  sparkline: number[];
  color: string;
}

function MetricRow({ label, value, sparkline, color }: MetricRowProps) {
  return (
    <div className="flex items-center justify-between gap-2 py-2">
      <div className="flex flex-col min-w-0">
        <span className="text-xs text-fg-tertiary truncate">{label}</span>
        <span className="text-sm font-semibold text-fg-primary">{value}</span>
      </div>
      <div className="w-20 flex-shrink-0">
        <Sparkline data={sparkline} color={color} />
      </div>
    </div>
  );
}

// --- Stage Indicator ---

function StageIndicator({ stage, stageName }: { stage: number; stageName: string }) {
  if (stage === 0) {
    return (
      <div className="text-xs text-fg-tertiary text-center py-2">Idle</div>
    );
  }

  return (
    <div className="py-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-fg-tertiary">Stage {stage}/9</span>
        <span className="text-xs text-cosmic-nebula font-medium truncate ml-2">
          {stageName}
        </span>
      </div>
      <div className="w-full h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
        <div
          className="h-full bg-cosmic-nebula rounded-full transition-all duration-300"
          style={{ width: `${(stage / 9) * 100}%` }}
        />
      </div>
    </div>
  );
}

// --- Main Component ---

export function CompactMetrics() {
  const { confidence, toolUsed, latencyMs, currentStage, currentStageName } =
    useWeaving();
  const { data: stats, source: statsSource } = useStats(5000);

  // Rolling sparkline accumulator — keeps last 5 data points
  const historyRef = useRef<{
    latency: number[];
    throughput: number[];
    cacheHit: number[];
    confidence: number[];
  }>({
    latency: [145, 138, 132, 128, 125],
    throughput: [720, 780, 810, 835, 847],
    cacheHit: [72, 74, 75, 76, 78],
    confidence: [0.82, 0.85, 0.88, 0.87, 0.89],
  });

  // Accumulate live stats into sparkline history
  if (stats && statsSource === 'live') {
    const h = historyRef.current;
    const push = (arr: number[], val: number) => {
      if (arr[arr.length - 1] !== val) {
        arr.push(val);
        if (arr.length > 5) arr.shift();
      }
    };
    push(h.latency, stats.avgLatencyMs);
    push(h.throughput, stats.queriesLastHour);
    push(h.cacheHit, Math.round(stats.cacheHitRate * 100));
    push(h.confidence, stats.avgConfidence);
  }

  const sparkLatency = historyRef.current.latency;
  const sparkThroughput = historyRef.current.throughput;
  const sparkCacheHit = historyRef.current.cacheHit;
  const sparkConfidence = historyRef.current.confidence;

  // Display values: prefer live stats, fall back to weaving context, then sparkline tail
  const displayLatency = latencyMs ?? stats?.avgLatencyMs ?? sparkLatency[sparkLatency.length - 1];
  const displayConfidence = confidence ?? stats?.avgConfidence ?? sparkConfidence[sparkConfidence.length - 1];
  const displayCacheHit = stats ? Math.round(stats.cacheHitRate * 100) : sparkCacheHit[sparkCacheHit.length - 1];
  const displayThroughput = stats?.queriesLastHour ?? sparkThroughput[sparkThroughput.length - 1];

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-3 py-2 border-b border-border-primary">
        <span className="text-xs font-medium text-fg-tertiary uppercase tracking-wider">
          Awareness
        </span>
      </div>

      {/* Stage Progress */}
      <div className="px-3 border-b border-border-primary">
        <StageIndicator
          stage={currentStage}
          stageName={currentStageName || STAGE_NAMES[currentStage] || ''}
        />
      </div>

      {/* Metrics */}
      <div className="px-3 flex-1 overflow-y-auto">
        <MetricRow
          label="Latency"
          value={displayLatency != null ? `${displayLatency.toFixed(0)}ms` : '—'}
          sparkline={sparkLatency}
          color="#10B981"
        />
        <MetricRow
          label="Throughput"
          value={`${displayThroughput} q/hr`}
          sparkline={sparkThroughput}
          color="#3B82F6"
        />
        <MetricRow
          label="Cache Hit"
          value={`${displayCacheHit}%`}
          sparkline={sparkCacheHit}
          color="#F59E0B"
        />
        <MetricRow
          label="Confidence"
          value={displayConfidence != null ? `${(displayConfidence * 100).toFixed(0)}%` : '—'}
          sparkline={sparkConfidence}
          color="#8B5CF6"
        />
      </div>

      {/* Tool Badge + Data Source */}
      <div className="px-3 py-2 border-t border-border-primary">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-xs text-fg-tertiary">Tool:</span>
            <span className="text-xs font-medium text-fg-secondary">
              {toolUsed ?? 'none'}
            </span>
          </div>
          {statsSource !== 'live' && (
            <span className="text-xs text-cosmic-aurora">{statsSource}</span>
          )}
        </div>
      </div>
    </div>
  );
}
