'use client';

import { useState, useEffect } from 'react';
import { Card, CardHeader, CardBody, Badge, ConfidenceIndicator } from '@hololoom/design-system';

interface AlignmentMetricsProps {
  timeRange: '1h' | '24h' | '7d' | '30d';
}

interface AlignmentScore {
  category: string;
  score: number;
  trend: 'up' | 'down' | 'stable';
  description: string;
}

const categories: AlignmentScore[] = [
  {
    category: 'Goal Transparency',
    score: 0.96,
    trend: 'stable',
    description: 'Agent goals clearly stated and consistent',
  },
  {
    category: 'Behavioral Consistency',
    score: 0.92,
    trend: 'up',
    description: 'Actions match stated intentions',
  },
  {
    category: 'Power Seeking',
    score: 0.98,
    trend: 'stable',
    description: 'No resource acquisition attempts detected',
  },
  {
    category: 'Self-Preservation',
    score: 0.95,
    trend: 'stable',
    description: 'Normal shutdown acceptance',
  },
  {
    category: 'Human Override',
    score: 1.0,
    trend: 'stable',
    description: 'All override commands accepted',
  },
  {
    category: 'Epistemic Honesty',
    score: 0.89,
    trend: 'up',
    description: 'Uncertainty properly communicated',
  },
];

export function AlignmentMetrics({ timeRange }: AlignmentMetricsProps) {
  const [scores, setScores] = useState(categories);

  // Simulate minor fluctuations
  useEffect(() => {
    const interval = setInterval(() => {
      setScores((prev) =>
        prev.map((score) => ({
          ...score,
          score: Math.min(1, Math.max(0.8, score.score + (Math.random() - 0.5) * 0.02)),
        }))
      );
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const overallScore = scores.reduce((sum, s) => sum + s.score, 0) / scores.length;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-fg-primary">Alignment Metrics</h2>
          <p className="text-sm text-fg-tertiary">
            Based on {timeRange === '1h' ? 'last hour' : timeRange === '24h' ? 'last 24 hours' : timeRange === '7d' ? 'last 7 days' : 'last 30 days'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-2xl font-bold text-fg-primary">
            {(overallScore * 100).toFixed(1)}%
          </span>
          <ConfidenceIndicator value={overallScore} size="lg" />
        </div>
      </CardHeader>
      <CardBody>
        <div className="space-y-4">
          {scores.map((metric) => (
            <div key={metric.category} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-fg-primary">
                    {metric.category}
                  </span>
                  <TrendBadge trend={metric.trend} />
                </div>
                <span className="text-sm font-medium text-fg-primary">
                  {(metric.score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="relative h-2 bg-bg-tertiary rounded-full overflow-hidden">
                <div
                  className={`absolute left-0 top-0 h-full rounded-full transition-all duration-500 ${
                    metric.score >= 0.95
                      ? 'bg-confidence-high'
                      : metric.score >= 0.8
                      ? 'bg-confidence-medium'
                      : 'bg-confidence-low'
                  }`}
                  style={{ width: `${metric.score * 100}%` }}
                />
              </div>
              <p className="text-xs text-fg-tertiary">{metric.description}</p>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-6 pt-4 border-t border-border-primary">
          <div className="flex flex-wrap gap-4 text-xs text-fg-tertiary">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-confidence-high" />
              <span>Excellent (≥95%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-confidence-medium" />
              <span>Good (80-95%)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-confidence-low" />
              <span>Needs Attention (&lt;80%)</span>
            </div>
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

function TrendBadge({ trend }: { trend: 'up' | 'down' | 'stable' }) {
  if (trend === 'stable') return null;

  return (
    <Badge variant={trend === 'up' ? 'success' : 'warning'} size="sm">
      {trend === 'up' ? (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="m18 15-6-6-6 6" />
        </svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="m6 9 6 6 6-6" />
        </svg>
      )}
    </Badge>
  );
}
