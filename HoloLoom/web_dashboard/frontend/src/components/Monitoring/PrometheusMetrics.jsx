import React from 'react'
import { Activity, Cpu, Database, Network, TrendingUp, TrendingDown } from 'lucide-react'
import { usePrometheus } from '../../hooks/usePrometheus'

function PrometheusMetrics() {
  const { metrics, loading, error, lastUpdate, refresh } = usePrometheus(10000) // Refresh every 10s

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
        <div className="flex items-center justify-center h-64">
          <div className="spinner" />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <p className="text-red-600">Failed to load metrics: {error}</p>
        <button
          onClick={refresh}
          className="mt-4 px-4 py-2 bg-hololoom-primary text-white rounded-lg hover:bg-hololoom-secondary transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  // Demo metrics if no real data available
  const demoMetrics = {
    api_requests_total: 1523,
    api_latency_avg: 142.5,
    cache_hit_rate: 0.87,
    memory_usage_mb: 256,
    cpu_usage_percent: 42,
    active_connections: 12,
    queries_per_second: 8.3,
    error_rate: 0.02,
  }

  const metricsData = metrics || demoMetrics

  const metricCards = [
    {
      title: 'API Requests',
      value: metricsData.api_requests_total?.toLocaleString() || '0',
      change: '+12.5%',
      trend: 'up',
      icon: Activity,
      color: 'blue',
    },
    {
      title: 'Avg Latency',
      value: `${metricsData.api_latency_avg?.toFixed(1) || 0}ms`,
      change: '-5.2%',
      trend: 'down',
      icon: TrendingUp,
      color: 'green',
    },
    {
      title: 'Cache Hit Rate',
      value: `${((metricsData.cache_hit_rate || 0) * 100).toFixed(1)}%`,
      change: '+3.1%',
      trend: 'up',
      icon: Database,
      color: 'purple',
    },
    {
      title: 'Memory Usage',
      value: `${metricsData.memory_usage_mb || 0}MB`,
      change: '+8.3%',
      trend: 'up',
      icon: Cpu,
      color: 'orange',
    },
    {
      title: 'Active Connections',
      value: metricsData.active_connections || 0,
      change: '+2',
      trend: 'up',
      icon: Network,
      color: 'cyan',
    },
    {
      title: 'Error Rate',
      value: `${((metricsData.error_rate || 0) * 100).toFixed(2)}%`,
      change: '-0.5%',
      trend: 'down',
      icon: TrendingDown,
      color: 'red',
    },
  ]

  const getColorClasses = (color) => {
    const colors = {
      blue: 'bg-blue-100 text-blue-600',
      green: 'bg-green-100 text-green-600',
      purple: 'bg-purple-100 text-purple-600',
      orange: 'bg-orange-100 text-orange-600',
      cyan: 'bg-cyan-100 text-cyan-600',
      red: 'bg-red-100 text-red-600',
    }
    return colors[color] || colors.blue
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Prometheus Metrics</h2>
          <p className="text-sm text-gray-600 mt-1">
            Last updated: {lastUpdate ? lastUpdate.toLocaleTimeString() : 'Never'}
          </p>
        </div>
        <button
          onClick={refresh}
          className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
        >
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {metricCards.map((card, index) => {
          const Icon = card.icon
          return (
            <div
              key={index}
              className="p-4 border border-gray-200 rounded-lg hover:border-hololoom-primary transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <div className={`p-2 rounded-lg ${getColorClasses(card.color)}`}>
                  <Icon className="w-5 h-5" />
                </div>
                <div className={`flex items-center gap-1 text-xs font-medium ${
                  card.trend === 'up' ? 'text-green-600' : 'text-red-600'
                }`}>
                  {card.trend === 'up' ? (
                    <TrendingUp className="w-3 h-3" />
                  ) : (
                    <TrendingDown className="w-3 h-3" />
                  )}
                  {card.change}
                </div>
              </div>
              <p className="text-sm font-medium text-gray-600 mb-1">{card.title}</p>
              <p className="text-2xl font-bold text-gray-900">{card.value}</p>
            </div>
          )
        })}
      </div>

      {/* Additional System Metrics */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h3 className="text-sm font-semibold text-gray-900 mb-4">System Health</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">CPU Usage</span>
            <span className="text-sm font-medium text-gray-900">
              {metricsData.cpu_usage_percent || 0}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-hololoom-primary h-2 rounded-full transition-all"
              style={{ width: `${metricsData.cpu_usage_percent || 0}%` }}
            />
          </div>

          <div className="flex items-center justify-between mt-4">
            <span className="text-sm text-gray-600">Queries/sec</span>
            <span className="text-sm font-medium text-gray-900">
              {metricsData.queries_per_second?.toFixed(1) || 0}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-green-500 h-2 rounded-full transition-all"
              style={{ width: `${Math.min((metricsData.queries_per_second || 0) * 10, 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default PrometheusMetrics
