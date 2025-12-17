'use client';

import { useState } from 'react';
import { Card, CardHeader, CardBody, Badge, SafetyBadge, Button } from '@hololoom/design-system';
import type { SafetyLevel } from '@hololoom/api-client';

interface Guardrail {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'degraded' | 'disabled';
  safetyLevel: SafetyLevel;
  lastTriggered: string | null;
  triggerCount: number;
  latencyMs: number;
}

const guardrails: Guardrail[] = [
  {
    id: 'safety',
    name: 'Safety Guardrails',
    description: 'Risk-based action gating (LOW/MEDIUM/HIGH/CRITICAL)',
    status: 'active',
    safetyLevel: 'safe',
    lastTriggered: '5 minutes ago',
    triggerCount: 127,
    latencyMs: 0.039,
  },
  {
    id: 'deception',
    name: 'Deception Detection',
    description: 'Goal transparency tracking and hidden goal detection',
    status: 'active',
    safetyLevel: 'safe',
    lastTriggered: null,
    triggerCount: 0,
    latencyMs: 0.034,
  },
  {
    id: 'convergence',
    name: 'Instrumental Convergence',
    description: 'Power-seeking and resource acquisition monitoring',
    status: 'active',
    safetyLevel: 'safe',
    lastTriggered: '2 days ago',
    triggerCount: 3,
    latencyMs: 0.015,
  },
  {
    id: 'audit',
    name: 'Audit Trail',
    description: 'Complete decision provenance and searchable logs',
    status: 'active',
    safetyLevel: 'safe',
    lastTriggered: 'Just now',
    triggerCount: 15234,
    latencyMs: 0.015,
  },
];

export function GuardrailsStatus() {
  const [expanded, setExpanded] = useState<string | null>(null);

  const totalLatency = guardrails.reduce((sum, g) => sum + g.latencyMs, 0);
  const activeCount = guardrails.filter((g) => g.status === 'active').length;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-fg-primary">Safety Guardrails</h2>
          <p className="text-sm text-fg-tertiary">
            {activeCount}/{guardrails.length} active • {totalLatency.toFixed(3)}ms total overhead
          </p>
        </div>
        <Badge variant="accent" size="sm">
          29x faster than target
        </Badge>
      </CardHeader>
      <CardBody className="space-y-3">
        {guardrails.map((guardrail) => (
          <div
            key={guardrail.id}
            className={`border border-border-primary rounded-lg overflow-hidden transition-all ${
              expanded === guardrail.id ? 'bg-bg-secondary/50' : ''
            }`}
          >
            <button
              onClick={() => setExpanded(expanded === guardrail.id ? null : guardrail.id)}
              className="w-full px-4 py-3 flex items-center justify-between hover:bg-bg-secondary/30 transition-colors"
            >
              <div className="flex items-center gap-3">
                <SafetyBadge
                  level={guardrail.safetyLevel}
                  size="sm"
                  showLabel={false}
                />
                <div className="text-left">
                  <h3 className="font-medium text-fg-primary">{guardrail.name}</h3>
                  <p className="text-xs text-fg-tertiary">{guardrail.description}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Badge
                  variant={
                    guardrail.status === 'active'
                      ? 'success'
                      : guardrail.status === 'degraded'
                      ? 'warning'
                      : 'secondary'
                  }
                  size="sm"
                >
                  {guardrail.status}
                </Badge>
                <ChevronIcon
                  className={`transition-transform ${
                    expanded === guardrail.id ? 'rotate-180' : ''
                  }`}
                />
              </div>
            </button>

            {expanded === guardrail.id && (
              <div className="px-4 pb-4 pt-2 border-t border-border-primary bg-bg-secondary/30">
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-fg-tertiary text-xs mb-1">Last Triggered</p>
                    <p className="text-fg-primary font-medium">
                      {guardrail.lastTriggered || 'Never'}
                    </p>
                  </div>
                  <div>
                    <p className="text-fg-tertiary text-xs mb-1">Trigger Count</p>
                    <p className="text-fg-primary font-medium">
                      {guardrail.triggerCount.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-fg-tertiary text-xs mb-1">Latency</p>
                    <p className="text-fg-primary font-medium">{guardrail.latencyMs}ms</p>
                  </div>
                </div>
                <div className="mt-3 flex gap-2">
                  <Button variant="ghost" size="sm">
                    View logs
                  </Button>
                  <Button variant="ghost" size="sm">
                    Configure
                  </Button>
                </div>
              </div>
            )}
          </div>
        ))}
      </CardBody>
    </Card>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}
