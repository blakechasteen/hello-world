import React, { useEffect, useState } from 'react'
import { Server, Database, Cpu, HardDrive, Activity, TrendingUp } from 'lucide-react'
import { useHoloLoom } from '../../hooks/useHoloLoom'
import { useWebSocket } from '../../hooks/useWebSocket'

function SystemStatus() {
  const { getStats } = useHoloLoom()
  const { subscribe } = useWebSocket()
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Initial fetch
    fetchStats()

    // Refresh every 10 seconds
    const interval = setInterval(fetchStats, 10000)

    // Subscribe to real-time updates
    const unsubscribe = subscribe('stats_update', (data) => {
      setStats(prev => ({ ...prev, ...data }))
    })

    return () => {
      clearInterval(interval)
      unsubscribe()
    }
  }, [getStats, subscribe])

  const fetchStats = async () => {
    try {
      const data = await getStats()
      setStats(data)
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8">
        <div className="flex items-center justify-center">
          <div className="spinner" />
        </div>
      </div>
    )
  }

  const statusCards = [
    {
      title: 'System Status',
      value: stats?.orchestrator_ready ? 'Online' : 'Offline',
      icon: Server,
      color: stats?.orchestrator_ready ? 'green' : 'red',
      subtext: `Uptime: ${stats?.uptime_formatted || 'N/A'}`
    },
    {
      title: 'Total Queries',
      value: stats?.total_queries || 0,
      icon: Activity,
      color: 'blue',
      subtext: `Success rate: ${stats?.success_rate || 0}%`
    },
    {
      title: 'Avg Latency',
      value: `${stats?.avg_latency_ms || 0}ms`,
      icon: TrendingUp,
      color: 'purple',
      subtext: `P95: ${stats?.p95_latency_ms || 0}ms`
    },
    {
      title: 'Memory Shards',
      value: stats?.memory_shards || 0,
      icon: Database,
      color: 'yellow',
      subtext: 'Knowledge base size'
    },
  ]

  const getColorClasses = (color) => {
    const colors = {
      green: 'bg-green-100 text-green-600',
      red: 'bg-red-100 text-red-600',
      blue: 'bg-blue-100 text-blue-600',
      purple: 'bg-purple-100 text-purple-600',
      yellow: 'bg-yellow-100 text-yellow-600',
    }
    return colors[color] || colors.blue
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {statusCards.map((card, index) => {
        const Icon = card.icon
        return (
          <div
            key={index}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600 mb-2">{card.title}</p>
                <p className="text-3xl font-bold text-gray-900 mb-1">{card.value}</p>
                <p className="text-xs text-gray-500">{card.subtext}</p>
              </div>
              <div className={`p-3 rounded-lg ${getColorClasses(card.color)}`}>
                <Icon className="w-6 h-6" />
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default SystemStatus
