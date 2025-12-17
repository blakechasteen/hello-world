'use client';

import type { TimeRange } from '../../app/analytics/page';

interface UsageBreakdownProps {
  timeRange: TimeRange;
}

export function UsageBreakdown({ timeRange }: UsageBreakdownProps) {
  // Mock data - in production would come from API
  const usageData = [
    { label: 'Direct Queries', value: 4823, percentage: 0.42, color: '#8B5CF6' },
    { label: 'Verify Mode', value: 2156, percentage: 0.19, color: '#10B981' },
    { label: 'Research Mode', value: 1847, percentage: 0.16, color: '#F59E0B' },
    { label: 'Plan Execute', value: 1523, percentage: 0.13, color: '#EC4899' },
    { label: 'Memory Ops', value: 1098, percentage: 0.10, color: '#6366F1' },
  ];

  const total = usageData.reduce((sum, item) => sum + item.value, 0);

  const agentUsage = [
    { agent: 'hololoom_query', count: 3421, percentage: 0.35 },
    { agent: 'memory_search', count: 2156, percentage: 0.22 },
    { agent: 'synthesizer', count: 1523, percentage: 0.16 },
    { agent: 'recursive_refiner', count: 1234, percentage: 0.13 },
    { agent: 'convergence_engine', count: 987, percentage: 0.10 },
    { agent: 'other', count: 389, percentage: 0.04 },
  ];

  return (
    <div className="space-y-6">
      {/* Query Type Distribution */}
      <div>
        <h4 className="text-sm font-medium text-fg-secondary mb-3">Query Distribution</h4>
        <div className="space-y-2">
          {usageData.map((item) => (
            <UsageBar
              key={item.label}
              label={item.label}
              value={item.value}
              percentage={item.percentage}
              color={item.color}
            />
          ))}
        </div>
        <p className="text-xs text-fg-tertiary mt-2 text-right">
          Total: {total.toLocaleString()} queries
        </p>
      </div>

      {/* Agent Usage */}
      <div>
        <h4 className="text-sm font-medium text-fg-secondary mb-3">Top Agents</h4>
        <div className="grid grid-cols-2 gap-2">
          {agentUsage.slice(0, 6).map((item) => (
            <AgentPill
              key={item.agent}
              agent={item.agent}
              count={item.count}
              percentage={item.percentage}
            />
          ))}
        </div>
      </div>

      {/* Time Distribution Mini Chart */}
      <div>
        <h4 className="text-sm font-medium text-fg-secondary mb-3">Hourly Activity</h4>
        <HourlyActivity />
      </div>
    </div>
  );
}

interface UsageBarProps {
  label: string;
  value: number;
  percentage: number;
  color: string;
}

function UsageBar({ label, value, percentage, color }: UsageBarProps) {
  return (
    <div className="flex items-center gap-3">
      <div className="w-24 text-xs text-fg-tertiary truncate">{label}</div>
      <div className="flex-1 h-2 bg-bg-secondary rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${percentage * 100}%`, backgroundColor: color }}
        />
      </div>
      <div className="w-12 text-xs text-fg-secondary text-right">
        {(percentage * 100).toFixed(0)}%
      </div>
    </div>
  );
}

interface AgentPillProps {
  agent: string;
  count: number;
  percentage: number;
}

function AgentPill({ agent, count, percentage }: AgentPillProps) {
  const displayName = agent.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());

  return (
    <div className="flex items-center justify-between p-2 bg-bg-secondary rounded-lg">
      <span className="text-xs text-fg-secondary truncate max-w-[80px]" title={displayName}>
        {displayName}
      </span>
      <span className="text-xs font-medium text-fg-primary">
        {count >= 1000 ? `${(count / 1000).toFixed(1)}k` : count}
      </span>
    </div>
  );
}

function HourlyActivity() {
  // Generate 24-hour activity data
  const hourlyData = Array.from({ length: 24 }, (_, i) => {
    // Simulate typical usage pattern (higher during work hours)
    const baseActivity = Math.sin((i - 6) * Math.PI / 12) * 0.5 + 0.5;
    const noise = (Math.random() - 0.5) * 0.3;
    return Math.max(0, Math.min(1, baseActivity + noise));
  });

  const maxValue = Math.max(...hourlyData);
  const currentHour = new Date().getHours();

  return (
    <div className="flex items-end justify-between h-16 gap-0.5">
      {hourlyData.map((value, hour) => {
        const height = (value / maxValue) * 100;
        const isCurrentHour = hour === currentHour;

        return (
          <div
            key={hour}
            className="flex-1 group relative"
            title={`${hour}:00 - ${((value / maxValue) * 100).toFixed(0)}%`}
          >
            <div
              className={`w-full rounded-t transition-all duration-200 ${
                isCurrentHour ? 'bg-cosmic-nebula' : 'bg-cosmic-nebula/40 group-hover:bg-cosmic-nebula/60'
              }`}
              style={{ height: `${height}%` }}
            />
            {hour % 6 === 0 && (
              <span className="absolute -bottom-4 left-1/2 -translate-x-1/2 text-[9px] text-fg-tertiary">
                {hour}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
