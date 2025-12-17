'use client';

import { Card, Badge } from '@hololoom/design-system';

interface LearningStats {
  thompsonAlpha: number;
  thompsonBeta: number;
  patternsLearned: number;
  hotPatterns: number;
  coldPatterns: number;
  adaptiveAccuracy: number;
}

interface LearningMetricsProps {
  stats: LearningStats;
}

export function LearningMetrics({ stats }: LearningMetricsProps) {
  const expectedReward = stats.thompsonAlpha / (stats.thompsonAlpha + stats.thompsonBeta);
  const explorationRate = 1 - expectedReward;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-fg-primary">Learning System</h3>
        <Badge variant="success" size="sm">Active</Badge>
      </div>

      {/* Thompson Sampling Visualization */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-fg-tertiary">Thompson Sampling Priors</span>
          <span className="text-xs text-fg-tertiary">
            E[X] = {(expectedReward * 100).toFixed(1)}%
          </span>
        </div>
        <ThompsonBar alpha={stats.thompsonAlpha} beta={stats.thompsonBeta} />
        <div className="flex items-center justify-between mt-1">
          <span className="text-xs text-fg-tertiary">α = {stats.thompsonAlpha.toFixed(1)}</span>
          <span className="text-xs text-fg-tertiary">β = {stats.thompsonBeta.toFixed(1)}</span>
        </div>
      </div>

      {/* Pattern Stats */}
      <div className="space-y-3 mb-4">
        <MetricRow
          label="Patterns Learned"
          value={stats.patternsLearned.toLocaleString()}
          icon={<BrainIcon />}
        />
        <MetricRow
          label="Hot Patterns"
          value={stats.hotPatterns.toString()}
          icon={<FireIcon />}
          highlight="hot"
        />
        <MetricRow
          label="Cold Patterns"
          value={stats.coldPatterns.toString()}
          icon={<SnowflakeIcon />}
          highlight="cold"
        />
        <MetricRow
          label="Adaptive Accuracy"
          value={`${(stats.adaptiveAccuracy * 100).toFixed(1)}%`}
          icon={<TargetIcon />}
        />
      </div>

      {/* Exploration Rate */}
      <div className="p-3 bg-bg-secondary rounded-lg">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-fg-tertiary">Exploration Rate</span>
          <span className="text-xs font-medium text-cosmic-nebula">
            {(explorationRate * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full h-1.5 bg-bg-primary rounded-full overflow-hidden">
          <div
            className="h-full bg-cosmic-nebula rounded-full transition-all duration-500"
            style={{ width: `${explorationRate * 100}%` }}
          />
        </div>
      </div>
    </Card>
  );
}

function ThompsonBar({ alpha, beta }: { alpha: number; beta: number }) {
  const total = alpha + beta;
  const alphaPercent = (alpha / total) * 100;

  return (
    <div className="w-full h-3 bg-bg-secondary rounded-full overflow-hidden flex">
      <div
        className="h-full bg-safety-safe transition-all duration-500"
        style={{ width: `${alphaPercent}%` }}
      />
      <div
        className="h-full bg-safety-risky transition-all duration-500"
        style={{ width: `${100 - alphaPercent}%` }}
      />
    </div>
  );
}

interface MetricRowProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  highlight?: 'hot' | 'cold';
}

function MetricRow({ label, value, icon, highlight }: MetricRowProps) {
  const highlightColors = {
    hot: 'text-orange-500',
    cold: 'text-blue-400',
  };

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className="text-fg-tertiary">{icon}</span>
        <span className="text-sm text-fg-secondary">{label}</span>
      </div>
      <span className={`text-sm font-semibold ${highlight ? highlightColors[highlight] : 'text-fg-primary'}`}>
        {value}
      </span>
    </div>
  );
}

function BrainIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
      <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
      <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
      <path d="M17.599 6.5a3 3 0 0 0 .399-1.375" />
      <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5" />
      <path d="M3.477 10.896a4 4 0 0 1 .585-.396" />
      <path d="M19.938 10.5a4 4 0 0 1 .585.396" />
      <path d="M6 18a4 4 0 0 1-1.967-.516" />
      <path d="M19.967 17.484A4 4 0 0 1 18 18" />
    </svg>
  );
}

function FireIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z" />
    </svg>
  );
}

function SnowflakeIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="2" x2="22" y1="12" y2="12" />
      <line x1="12" x2="12" y1="2" y2="22" />
      <path d="m20 16-4-4 4-4" />
      <path d="m4 8 4 4-4 4" />
      <path d="m16 4-4 4-4-4" />
      <path d="m8 20 4-4 4 4" />
    </svg>
  );
}

function TargetIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="6" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  );
}
