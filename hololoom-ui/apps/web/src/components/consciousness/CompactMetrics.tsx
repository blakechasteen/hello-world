'use client';

import { Card } from '@hololoom/design-system';
import { useWeaving } from '@/contexts/WeavingContext';

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

  // Mock sparkline data — will be replaced by useStats() in Phase 2
  const mockLatency = [145, 138, 132, 128, latencyMs ?? 125];
  const mockThroughput = [720, 780, 810, 835, 847];
  const mockCacheHit = [72, 74, 75, 76, 78];
  const mockConfidence = [0.82, 0.85, 0.88, 0.87, confidence ?? 0.89];

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
          value={latencyMs != null ? `${latencyMs.toFixed(0)}ms` : '—'}
          sparkline={mockLatency}
          color="#10B981"
        />
        <MetricRow
          label="Throughput"
          value="847 q/min"
          sparkline={mockThroughput}
          color="#3B82F6"
        />
        <MetricRow
          label="Cache Hit"
          value="78%"
          sparkline={mockCacheHit}
          color="#F59E0B"
        />
        <MetricRow
          label="Confidence"
          value={confidence != null ? `${(confidence * 100).toFixed(0)}%` : '—'}
          sparkline={mockConfidence}
          color="#8B5CF6"
        />
      </div>

      {/* Tool Badge */}
      <div className="px-3 py-2 border-t border-border-primary">
        <div className="flex items-center gap-2">
          <span className="text-xs text-fg-tertiary">Tool:</span>
          <span className="text-xs font-medium text-fg-secondary">
            {toolUsed ?? 'none'}
          </span>
        </div>
      </div>
    </div>
  );
}
