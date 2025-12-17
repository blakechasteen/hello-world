'use client';

import { Card, Badge } from '@hololoom/design-system';

interface SystemHealthProps {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkLatency: number;
}

export function SystemHealth({
  cpuUsage,
  memoryUsage,
  diskUsage,
  networkLatency,
}: SystemHealthProps) {
  const overallHealth = calculateOverallHealth(cpuUsage, memoryUsage, diskUsage, networkLatency);

  const resources = [
    {
      label: 'CPU',
      value: cpuUsage,
      icon: <CpuIcon />,
      status: getResourceStatus(cpuUsage),
    },
    {
      label: 'Memory',
      value: memoryUsage,
      icon: <MemoryIcon />,
      status: getResourceStatus(memoryUsage),
    },
    {
      label: 'Disk',
      value: diskUsage,
      icon: <DiskIcon />,
      status: getResourceStatus(diskUsage),
    },
  ];

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-fg-primary">System Health</h3>
        <HealthBadge health={overallHealth} />
      </div>

      {/* Overall Health Ring */}
      <div className="flex justify-center mb-6">
        <HealthRing health={overallHealth} />
      </div>

      {/* Resource Bars */}
      <div className="space-y-4">
        {resources.map((resource) => (
          <ResourceBar
            key={resource.label}
            label={resource.label}
            value={resource.value}
            icon={resource.icon}
            status={resource.status}
          />
        ))}
      </div>

      {/* Network Latency */}
      <div className="mt-4 p-3 bg-bg-secondary rounded-lg flex items-center justify-between">
        <div className="flex items-center gap-2">
          <NetworkIcon className="w-4 h-4 text-fg-tertiary" />
          <span className="text-sm text-fg-secondary">Network Latency</span>
        </div>
        <span className={`text-sm font-semibold ${
          networkLatency < 10 ? 'text-safety-safe' :
          networkLatency < 50 ? 'text-safety-risky' : 'text-safety-critical'
        }`}>
          {networkLatency.toFixed(1)}ms
        </span>
      </div>

      {/* Services Status */}
      <div className="mt-4 pt-4 border-t border-border-primary">
        <h4 className="text-xs font-medium text-fg-tertiary uppercase tracking-wide mb-3">
          Services
        </h4>
        <div className="grid grid-cols-3 gap-2">
          <ServicePill name="Neo4j" status="healthy" />
          <ServicePill name="Qdrant" status="healthy" />
          <ServicePill name="Ollama" status="healthy" />
        </div>
      </div>
    </Card>
  );
}

function HealthBadge({ health }: { health: number }) {
  let variant: 'success' | 'warning' | 'accent' = 'success';
  let label = 'Healthy';

  if (health < 0.5) {
    variant = 'accent';
    label = 'Critical';
  } else if (health < 0.75) {
    variant = 'warning';
    label = 'Degraded';
  }

  return <Badge variant={variant} size="sm">{label}</Badge>;
}

function HealthRing({ health }: { health: number }) {
  const radius = 45;
  const strokeWidth = 8;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - health * circumference;

  let color = '#10B981';
  if (health < 0.5) {
    color = '#EF4444';
  } else if (health < 0.75) {
    color = '#F59E0B';
  }

  return (
    <svg width="120" height="120" viewBox="0 0 120 120">
      {/* Background circle */}
      <circle
        cx="60"
        cy="60"
        r={radius}
        fill="none"
        stroke="currentColor"
        strokeOpacity="0.1"
        strokeWidth={strokeWidth}
      />

      {/* Progress circle */}
      <circle
        cx="60"
        cy="60"
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={strokeDashoffset}
        transform="rotate(-90 60 60)"
        style={{ transition: 'stroke-dashoffset 0.5s ease' }}
      />

      {/* Percentage text */}
      <text
        x="60"
        y="55"
        textAnchor="middle"
        className="fill-fg-primary font-bold"
        style={{ fontSize: '24px' }}
      >
        {(health * 100).toFixed(0)}%
      </text>
      <text
        x="60"
        y="72"
        textAnchor="middle"
        className="fill-fg-tertiary"
        style={{ fontSize: '10px' }}
      >
        Overall
      </text>
    </svg>
  );
}

interface ResourceBarProps {
  label: string;
  value: number;
  icon: React.ReactNode;
  status: 'good' | 'warning' | 'critical';
}

function ResourceBar({ label, value, icon, status }: ResourceBarProps) {
  const colors = {
    good: 'bg-safety-safe',
    warning: 'bg-safety-risky',
    critical: 'bg-safety-critical',
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-fg-tertiary">{icon}</span>
          <span className="text-sm text-fg-secondary">{label}</span>
        </div>
        <span className="text-sm font-medium text-fg-primary">
          {(value * 100).toFixed(0)}%
        </span>
      </div>
      <div className="w-full h-1.5 bg-bg-secondary rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${colors[status]}`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

function ServicePill({ name, status }: { name: string; status: 'healthy' | 'degraded' | 'down' }) {
  const colors = {
    healthy: 'bg-safety-safe/10 text-safety-safe border-safety-safe/20',
    degraded: 'bg-safety-risky/10 text-safety-risky border-safety-risky/20',
    down: 'bg-safety-critical/10 text-safety-critical border-safety-critical/20',
  };

  const dots = {
    healthy: 'bg-safety-safe',
    degraded: 'bg-safety-risky',
    down: 'bg-safety-critical',
  };

  return (
    <div className={`flex items-center gap-1.5 px-2 py-1 rounded border ${colors[status]}`}>
      <div className={`w-1.5 h-1.5 rounded-full ${dots[status]}`} />
      <span className="text-xs font-medium">{name}</span>
    </div>
  );
}

function getResourceStatus(value: number): 'good' | 'warning' | 'critical' {
  if (value < 0.6) return 'good';
  if (value < 0.85) return 'warning';
  return 'critical';
}

function calculateOverallHealth(cpu: number, memory: number, disk: number, network: number): number {
  // Weighted average (lower usage = better health)
  const cpuHealth = 1 - cpu;
  const memoryHealth = 1 - memory;
  const diskHealth = 1 - disk;
  const networkHealth = network < 10 ? 1 : network < 50 ? 0.7 : 0.3;

  return cpuHealth * 0.3 + memoryHealth * 0.3 + diskHealth * 0.2 + networkHealth * 0.2;
}

function CpuIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="16" height="16" x="4" y="4" rx="2" />
      <rect width="6" height="6" x="9" y="9" rx="1" />
      <path d="M15 2v2" />
      <path d="M15 20v2" />
      <path d="M2 15h2" />
      <path d="M2 9h2" />
      <path d="M20 15h2" />
      <path d="M20 9h2" />
      <path d="M9 2v2" />
      <path d="M9 20v2" />
    </svg>
  );
}

function MemoryIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 19v-3" />
      <path d="M10 19v-3" />
      <path d="M14 19v-3" />
      <path d="M18 19v-3" />
      <path d="M8 11V9" />
      <path d="M16 11V9" />
      <path d="M12 11V9" />
      <path d="M2 15h20" />
      <path d="M2 7a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v1.1a2 2 0 0 0 0 3.837V17a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-5.1a2 2 0 0 0 0-3.837Z" />
    </svg>
  );
}

function DiskIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5V19A9 3 0 0 0 21 19V5" />
      <path d="M3 12A9 3 0 0 0 21 12" />
    </svg>
  );
}

function NetworkIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4" />
    </svg>
  );
}
