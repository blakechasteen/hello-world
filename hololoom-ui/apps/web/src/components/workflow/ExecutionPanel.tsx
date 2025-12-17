'use client';

import { Button, Badge } from '@hololoom/design-system';

interface ExecutionPanelProps {
  results: Record<string, unknown>;
  onClose: () => void;
}

export function ExecutionPanel({ results, onClose }: ExecutionPanelProps) {
  const isSuccess = results.success as boolean;
  const output = results.output as string | undefined;
  const error = results.error as string | undefined;
  const duration = results.duration as string | undefined;

  return (
    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 w-[480px] bg-bg-primary border border-border-primary rounded-xl shadow-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-primary bg-bg-secondary/50">
        <div className="flex items-center gap-3">
          {isSuccess ? (
            <SuccessIcon className="w-5 h-5 text-safety-safe" />
          ) : (
            <ErrorIcon className="w-5 h-5 text-safety-critical" />
          )}
          <span className="font-medium text-fg-primary">
            {isSuccess ? 'Execution Complete' : 'Execution Failed'}
          </span>
          <Badge variant={isSuccess ? 'success' : 'warning'} size="sm">
            {isSuccess ? 'Success' : 'Error'}
          </Badge>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-fg-tertiary hover:text-fg-primary hover:bg-bg-secondary transition-colors"
        >
          <CloseIcon />
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Duration */}
        {duration && (
          <div className="flex items-center gap-2 text-sm">
            <ClockIcon className="w-4 h-4 text-fg-tertiary" />
            <span className="text-fg-tertiary">Duration:</span>
            <span className="text-fg-primary font-medium">{duration}</span>
          </div>
        )}

        {/* Output or Error */}
        {isSuccess && output && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <OutputIcon className="w-4 h-4 text-fg-tertiary" />
              <span className="text-fg-tertiary">Output:</span>
            </div>
            <div className="p-3 bg-bg-secondary rounded-lg border border-border-primary">
              <p className="text-sm text-fg-primary">{output}</p>
            </div>
          </div>
        )}

        {!isSuccess && error && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <ErrorIcon className="w-4 h-4 text-safety-critical" />
              <span className="text-safety-critical">Error:</span>
            </div>
            <div className="p-3 bg-safety-critical/10 rounded-lg border border-safety-critical/20">
              <p className="text-sm text-safety-critical">{error}</p>
            </div>
          </div>
        )}

        {/* Metrics */}
        {isSuccess && (
          <div className="grid grid-cols-3 gap-3">
            <MetricCard
              label="Nodes Executed"
              value={results.nodesExecuted as number || 0}
              icon={<NodesIcon className="w-4 h-4" />}
            />
            <MetricCard
              label="Connections"
              value={results.connectionsUsed as number || 0}
              icon={<ConnectionIcon className="w-4 h-4" />}
            />
            <MetricCard
              label="Avg Confidence"
              value={`${((results.avgConfidence as number || 0) * 100).toFixed(0)}%`}
              icon={<ConfidenceIcon className="w-4 h-4" />}
            />
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end gap-2 pt-2">
          <Button variant="secondary" size="sm" onClick={onClose}>
            Dismiss
          </Button>
          {isSuccess && (
            <Button variant="default" size="sm">
              <DownloadIcon className="w-4 h-4" />
              <span className="ml-1.5">Export Results</span>
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
}

function MetricCard({ label, value, icon }: MetricCardProps) {
  return (
    <div className="p-3 bg-bg-secondary rounded-lg border border-border-primary">
      <div className="flex items-center gap-2 text-fg-tertiary mb-1">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <p className="text-lg font-semibold text-fg-primary">{value}</p>
    </div>
  );
}

function SuccessIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function ErrorIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" x2="12" y1="8" y2="12" />
      <line x1="12" x2="12.01" y1="16" y2="16" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" x2="6" y1="6" y2="18" />
      <line x1="6" x2="18" y1="6" y2="18" />
    </svg>
  );
}

function ClockIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}

function OutputIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" x2="20" y1="19" y2="19" />
    </svg>
  );
}

function NodesIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect width="8" height="8" x="3" y="3" rx="2" />
      <rect width="8" height="8" x="13" y="13" rx="2" />
      <path d="M8 11v4" />
      <path d="M16 5v8" />
    </svg>
  );
}

function ConnectionIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M15 3h6v6" />
      <path d="M9 21H3v-6" />
      <path d="M21 3 9 15" />
      <path d="M3 21l6-6" />
    </svg>
  );
}

function ConfidenceIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

function DownloadIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  );
}
