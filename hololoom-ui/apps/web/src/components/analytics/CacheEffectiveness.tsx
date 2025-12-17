'use client';

import { Card } from '@hololoom/design-system';

interface CacheEffectivenessProps {
  hitRate: number;
  totalQueries: number;
  cacheHits: number;
  avgCachedLatency: number;
  avgUncachedLatency: number;
}

export function CacheEffectiveness({
  hitRate,
  totalQueries,
  cacheHits,
  avgCachedLatency,
  avgUncachedLatency,
}: CacheEffectivenessProps) {
  const effectiveness = getEffectivenessRating(hitRate);
  const speedup = avgUncachedLatency / avgCachedLatency;
  const timeSaved = (cacheHits * (avgUncachedLatency - avgCachedLatency)) / 1000;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-fg-primary">Cache Effectiveness</h3>
        <EffectivenessBadge rating={effectiveness.rating} />
      </div>

      {/* Gauge */}
      <div className="flex justify-center mb-6">
        <CacheGauge hitRate={hitRate} rating={effectiveness.rating} />
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-4">
        <StatItem
          label="Cache Hits"
          value={cacheHits.toLocaleString()}
          subtext={`of ${totalQueries.toLocaleString()}`}
        />
        <StatItem
          label="Speedup"
          value={`${speedup.toFixed(1)}x`}
          subtext="vs uncached"
        />
        <StatItem
          label="Avg Cached"
          value={`${avgCachedLatency.toFixed(1)}ms`}
          subtext="latency"
        />
        <StatItem
          label="Time Saved"
          value={`${timeSaved.toFixed(1)}s`}
          subtext="total"
        />
      </div>

      {/* Recommendation */}
      <div className="mt-4 p-3 bg-bg-secondary rounded-lg">
        <p className="text-xs text-fg-tertiary">{effectiveness.recommendation}</p>
      </div>
    </Card>
  );
}

function StatItem({ label, value, subtext }: { label: string; value: string; subtext: string }) {
  return (
    <div className="text-center">
      <p className="text-xs text-fg-tertiary mb-0.5">{label}</p>
      <p className="text-lg font-semibold text-fg-primary">{value}</p>
      <p className="text-xs text-fg-tertiary">{subtext}</p>
    </div>
  );
}

function EffectivenessBadge({ rating }: { rating: string }) {
  const colors: Record<string, string> = {
    excellent: 'bg-safety-safe/10 text-safety-safe',
    good: 'bg-emerald-500/10 text-emerald-500',
    fair: 'bg-safety-risky/10 text-safety-risky',
    poor: 'bg-orange-500/10 text-orange-500',
    critical: 'bg-safety-critical/10 text-safety-critical',
  };

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium capitalize ${colors[rating]}`}>
      {rating}
    </span>
  );
}

function CacheGauge({ hitRate, rating }: { hitRate: number; rating: string }) {
  const percentage = hitRate * 100;
  const radius = 60;
  const strokeWidth = 10;
  const circumference = 2 * Math.PI * radius;
  const halfCircumference = circumference / 2;
  const strokeDashoffset = halfCircumference - (percentage / 100) * halfCircumference;

  const colors: Record<string, string> = {
    excellent: '#10B981',
    good: '#34D399',
    fair: '#F59E0B',
    poor: '#F97316',
    critical: '#EF4444',
  };

  return (
    <svg width="160" height="100" viewBox="0 0 160 100">
      {/* Background arc */}
      <path
        d="M 20 90 A 60 60 0 0 1 140 90"
        fill="none"
        stroke="currentColor"
        strokeOpacity="0.1"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
      />

      {/* Value arc */}
      <path
        d="M 20 90 A 60 60 0 0 1 140 90"
        fill="none"
        stroke={colors[rating]}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeDasharray={halfCircumference}
        strokeDashoffset={strokeDashoffset}
        style={{ transition: 'stroke-dashoffset 0.5s ease' }}
      />

      {/* Percentage text */}
      <text
        x="80"
        y="75"
        textAnchor="middle"
        className="fill-fg-primary text-2xl font-bold"
        style={{ fontSize: '24px' }}
      >
        {percentage.toFixed(1)}%
      </text>

      {/* Label */}
      <text
        x="80"
        y="92"
        textAnchor="middle"
        className="fill-fg-tertiary"
        style={{ fontSize: '10px' }}
      >
        Hit Rate
      </text>
    </svg>
  );
}

function getEffectivenessRating(hitRate: number) {
  if (hitRate >= 0.8) {
    return {
      rating: 'excellent',
      recommendation: 'Cache is performing optimally. Consider increasing cache size for even better performance.',
    };
  } else if (hitRate >= 0.6) {
    return {
      rating: 'good',
      recommendation: 'Good cache performance. Review query patterns for potential optimization.',
    };
  } else if (hitRate >= 0.4) {
    return {
      rating: 'fair',
      recommendation: 'Cache could be improved. Consider caching more query types or increasing TTL.',
    };
  } else if (hitRate >= 0.2) {
    return {
      rating: 'poor',
      recommendation: 'Low cache effectiveness. Review caching strategy and query patterns.',
    };
  } else {
    return {
      rating: 'critical',
      recommendation: 'Critical: Cache is underutilized. Check cache configuration and query normalization.',
    };
  }
}
