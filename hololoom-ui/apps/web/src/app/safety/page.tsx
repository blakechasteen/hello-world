'use client';

import { useState, useEffect } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  Badge,
  SafetyBadge,
  ConfidenceIndicator,
  LoadingSpinner,
} from '@hololoom/design-system';
import { useHoloLoom } from '@hololoom/api-client';
import { Navigation } from '@/components/Navigation';
import { SafetyOverview } from '@/components/safety/SafetyOverview';
import { AuditTrailTable } from '@/components/safety/AuditTrailTable';
import { GuardrailsStatus } from '@/components/safety/GuardrailsStatus';
import { AlignmentMetrics } from '@/components/safety/AlignmentMetrics';
import { RiskTimeline } from '@/components/safety/RiskTimeline';

export default function SafetyDashboardPage() {
  const { isConnected } = useHoloLoom();
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [autoRefresh, setAutoRefresh] = useState(true);

  return (
    <div className="page-container">
      <Navigation />

      {/* Page header */}
      <div className="main-content py-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl font-bold text-fg-primary">Safety Dashboard</h1>
              <Badge variant="accent" className="animate-pulse">Live</Badge>
            </div>
            <p className="text-fg-secondary">
              Monitor alignment health, safety guardrails, and audit trails in real-time.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Time range selector */}
            <div className="flex bg-bg-secondary rounded-lg p-1">
              {(['1h', '24h', '7d', '30d'] as const).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                    timeRange === range
                      ? 'bg-cosmic-nebula text-white'
                      : 'text-fg-secondary hover:text-fg-primary'
                  }`}
                >
                  {range}
                </button>
              ))}
            </div>

            {/* Auto-refresh toggle */}
            <Button
              variant={autoRefresh ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              {autoRefresh ? (
                <>
                  <PauseIcon />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <PlayIcon />
                  <span>Resume</span>
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Connection status warning */}
        {!isConnected && (
          <div className="mb-6 flex items-center gap-3 bg-safety-warning/10 border border-safety-warning/30 rounded-lg px-4 py-3">
            <SafetyBadge level="warning" size="sm" showLabel={false} />
            <span className="text-sm text-safety-warning">
              Not connected to HoloLoom backend. Safety metrics may be stale.
            </span>
          </div>
        )}

        {/* Overview cards */}
        <SafetyOverview timeRange={timeRange} autoRefresh={autoRefresh} />

        {/* Main content grid */}
        <div className="grid gap-6 lg:grid-cols-2 mt-6">
          {/* Guardrails status */}
          <GuardrailsStatus />

          {/* Alignment metrics */}
          <AlignmentMetrics timeRange={timeRange} />
        </div>

        {/* Risk timeline */}
        <div className="mt-6">
          <RiskTimeline timeRange={timeRange} />
        </div>

        {/* Audit trail */}
        <div className="mt-6">
          <AuditTrailTable timeRange={timeRange} autoRefresh={autoRefresh} />
        </div>
      </div>
    </div>
  );
}

// Icon components
function PauseIcon() {
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
      className="mr-1"
    >
      <rect x="6" y="4" width="4" height="16" />
      <rect x="14" y="4" width="4" height="16" />
    </svg>
  );
}

function PlayIcon() {
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
      className="mr-1"
    >
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  );
}
