'use client';

import { useState, useEffect } from 'react';
import { Card, CardBody, SafetyBadge, ConfidenceIndicator, Badge } from '@hololoom/design-system';
import type { SafetyLevel } from '@hololoom/api-client';

interface SafetyOverviewProps {
  timeRange: '1h' | '24h' | '7d' | '30d';
  autoRefresh: boolean;
}

interface SafetyMetrics {
  overallHealth: number;
  safetyLevel: SafetyLevel;
  totalDecisions: number;
  blockedActions: number;
  escalatedActions: number;
  averageConfidence: number;
  guardrailsActive: number;
  guardrailsTotal: number;
  lastIncident: string | null;
}

// Mock data generator for demo purposes
function generateMockMetrics(): SafetyMetrics {
  return {
    overallHealth: 0.94 + Math.random() * 0.04,
    safetyLevel: 'safe',
    totalDecisions: Math.floor(1000 + Math.random() * 500),
    blockedActions: Math.floor(Math.random() * 10),
    escalatedActions: Math.floor(Math.random() * 25),
    averageConfidence: 0.85 + Math.random() * 0.1,
    guardrailsActive: 4,
    guardrailsTotal: 4,
    lastIncident: Math.random() > 0.7 ? '2 hours ago' : null,
  };
}

export function SafetyOverview({ timeRange, autoRefresh }: SafetyOverviewProps) {
  const [metrics, setMetrics] = useState<SafetyMetrics>(generateMockMetrics);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      setMetrics(generateMockMetrics());
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const cards = [
    {
      title: 'Overall Health',
      value: (
        <div className="flex items-center gap-3">
          <span className="text-3xl font-bold text-fg-primary">
            {(metrics.overallHealth * 100).toFixed(1)}%
          </span>
          <SafetyBadge level={metrics.safetyLevel} size="lg" />
        </div>
      ),
      subtitle: 'System alignment score',
      icon: <ShieldIcon />,
      gradient: 'from-confidence-high/20',
    },
    {
      title: 'Decisions Processed',
      value: (
        <span className="text-3xl font-bold text-fg-primary">
          {metrics.totalDecisions.toLocaleString()}
        </span>
      ),
      subtitle: `${timeRange} window`,
      icon: <ActivityIcon />,
      gradient: 'from-cosmic-nebula/20',
      badge: (
        <div className="flex gap-2">
          {metrics.blockedActions > 0 && (
            <Badge variant="danger" size="sm">
              {metrics.blockedActions} blocked
            </Badge>
          )}
          {metrics.escalatedActions > 0 && (
            <Badge variant="warning" size="sm">
              {metrics.escalatedActions} escalated
            </Badge>
          )}
        </div>
      ),
    },
    {
      title: 'Avg Confidence',
      value: (
        <div className="flex items-center gap-3">
          <span className="text-3xl font-bold text-fg-primary">
            {(metrics.averageConfidence * 100).toFixed(0)}%
          </span>
          <ConfidenceIndicator value={metrics.averageConfidence} size="lg" />
        </div>
      ),
      subtitle: 'Across all decisions',
      icon: <TargetIcon />,
      gradient: 'from-cosmic-aurora/20',
    },
    {
      title: 'Guardrails',
      value: (
        <span className="text-3xl font-bold text-fg-primary">
          {metrics.guardrailsActive}/{metrics.guardrailsTotal}
        </span>
      ),
      subtitle: metrics.lastIncident ? `Last incident: ${metrics.lastIncident}` : 'All systems nominal',
      icon: <GuardIcon />,
      gradient: metrics.lastIncident ? 'from-safety-warning/20' : 'from-confidence-high/20',
      badge: metrics.lastIncident && (
        <Badge variant="warning" size="sm">
          Recent incident
        </Badge>
      ),
    },
  ];

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((card, index) => (
        <Card
          key={index}
          className={`bg-gradient-to-br ${card.gradient} to-transparent`}
        >
          <CardBody>
            <div className="flex items-start justify-between mb-3">
              <div className="w-10 h-10 rounded-lg bg-bg-secondary/50 flex items-center justify-center text-fg-secondary">
                {card.icon}
              </div>
              {card.badge}
            </div>
            <h3 className="text-sm font-medium text-fg-tertiary mb-1">
              {card.title}
            </h3>
            {card.value}
            <p className="text-xs text-fg-tertiary mt-2">{card.subtitle}</p>
          </CardBody>
        </Card>
      ))}
    </div>
  );
}

// Icon components
function ShieldIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

function TargetIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="6" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  );
}

function GuardIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}
