import { useSystemMetrics, useAgents, useTasks } from '@stores/appStore'

/**
 * Overview View
 *
 * Displays system-wide metrics and quick stats:
 * - Total agents / running / errors
 * - System latency and cache hit rate
 * - Recent activity feed
 * - Quick action buttons
 */
export default function OverviewView() {
  const systemMetrics = useSystemMetrics()
  const agents = useAgents()
  const tasks = useTasks()

  const stats = [
    { label: 'Total Agents', value: agents.length, color: 'holo-primary' },
    {
      label: 'Running',
      value: agents.filter((a) => a.state === 'running').length,
      color: 'holo-accent',
    },
    {
      label: 'Errors',
      value: agents.filter((a) => a.state === 'error').length,
      color: 'holo-danger',
    },
    { label: 'Pending Tasks', value: tasks.filter((t) => t.status === 'pending').length, color: 'holo-warning' },
  ]

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div>
        <h1 className="text-3xl font-bold text-text-primary">Dashboard Overview</h1>
        <p className="text-text-secondary mt-1">
          Real-time monitoring of HoloLoom agent system
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div key={stat.label} className="card p-6">
            <div className="text-text-secondary text-sm font-medium mb-2">
              {stat.label}
            </div>
            <div className={`text-3xl font-bold text-${stat.color}`}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {/* System Metrics */}
      {systemMetrics && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Latency */}
          <div className="card p-6">
            <div className="text-text-secondary text-sm font-medium mb-4">
              System Latency
            </div>
            <div className="text-4xl font-bold text-holo-accent">
              {systemMetrics.systemLatencyMs.toFixed(1)}
              <span className="text-lg ml-2">ms</span>
            </div>
            <div className="text-xs text-text-tertiary mt-2">
              Average response time
            </div>
          </div>

          {/* Cache Hit Rate */}
          <div className="card p-6">
            <div className="text-text-secondary text-sm font-medium mb-4">
              Cache Hit Rate
            </div>
            <div className="text-4xl font-bold text-holo-success">
              {(systemMetrics.cacheHitRate * 100).toFixed(1)}
              <span className="text-lg ml-2">%</span>
            </div>
            <div className="text-xs text-text-tertiary mt-2">
              Query cache efficiency
            </div>
          </div>

          {/* Avg Confidence */}
          <div className="card p-6">
            <div className="text-text-secondary text-sm font-medium mb-4">
              Avg Confidence
            </div>
            <div className="text-4xl font-bold text-holo-secondary">
              {(systemMetrics.avgConfidence * 100).toFixed(1)}
              <span className="text-lg ml-2">%</span>
            </div>
            <div className="text-xs text-text-tertiary mt-2">
              System-wide confidence
            </div>
          </div>
        </div>
      )}

      {/* Placeholder Content */}
      <div className="card p-6">
        <h2 className="text-xl font-semibold text-text-primary mb-4">
          Recent Activity
        </h2>
        <div className="space-y-2 text-center text-text-secondary">
          <p>🚀 Dashboard views in development</p>
          <p className="text-sm">Detailed charts and activity feed coming soon</p>
        </div>
      </div>
    </div>
  )
}
