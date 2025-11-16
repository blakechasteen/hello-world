import React, { useEffect, useState } from 'react'
import { AlertTriangle, AlertCircle, Info, CheckCircle, Bell, BellOff } from 'lucide-react'
import { useWebSocket } from '../../hooks/useWebSocket'

function AlertsPanel() {
  const { subscribe } = useWebSocket()
  const [alerts, setAlerts] = useState([])
  const [filter, setFilter] = useState('all') // all, critical, warning, info

  useEffect(() => {
    // Subscribe to alert events
    const unsubscribe = subscribe('alert', (alert) => {
      setAlerts(prev => [alert, ...prev.slice(0, 19)]) // Keep last 20
    })

    // Load demo alerts
    setAlerts(createDemoAlerts())

    return () => unsubscribe()
  }, [subscribe])

  const createDemoAlerts = () => {
    return [
      {
        id: '1',
        severity: 'critical',
        title: 'High Memory Usage',
        message: 'System memory usage exceeded 90%',
        timestamp: new Date(Date.now() - 5 * 60000),
        acknowledged: false,
      },
      {
        id: '2',
        severity: 'warning',
        title: 'Slow Query Detected',
        message: 'Query latency exceeded 500ms threshold',
        timestamp: new Date(Date.now() - 15 * 60000),
        acknowledged: false,
      },
      {
        id: '3',
        severity: 'info',
        title: 'Backup Completed',
        message: 'Daily backup successfully completed',
        timestamp: new Date(Date.now() - 30 * 60000),
        acknowledged: true,
      },
      {
        id: '4',
        severity: 'warning',
        title: 'Cache Miss Rate High',
        message: 'Cache hit rate dropped below 80%',
        timestamp: new Date(Date.now() - 45 * 60000),
        acknowledged: true,
      },
      {
        id: '5',
        severity: 'critical',
        title: 'Database Connection Lost',
        message: 'Neo4j connection timeout - retrying...',
        timestamp: new Date(Date.now() - 60 * 60000),
        acknowledged: true,
      },
    ]
  }

  const filteredAlerts = alerts.filter(alert => {
    if (filter === 'all') return true
    return alert.severity === filter
  })

  const getAlertIcon = (severity) => {
    switch (severity) {
      case 'critical':
        return <AlertCircle className="w-5 h-5 text-red-600" />
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-600" />
      case 'info':
        return <Info className="w-5 h-5 text-blue-600" />
      default:
        return <CheckCircle className="w-5 h-5 text-green-600" />
    }
  }

  const getAlertBgColor = (severity) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-50 border-red-200'
      case 'warning':
        return 'bg-yellow-50 border-yellow-200'
      case 'info':
        return 'bg-blue-50 border-blue-200'
      default:
        return 'bg-gray-50 border-gray-200'
    }
  }

  const acknowledgeAlert = (id) => {
    setAlerts(prev =>
      prev.map(alert =>
        alert.id === id ? { ...alert, acknowledged: true } : alert
      )
    )
  }

  const unacknowledgedCount = alerts.filter(a => !a.acknowledged).length

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-red-100">
            <Bell className="w-5 h-5 text-red-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Alerts</h2>
            <p className="text-sm text-gray-600">
              {unacknowledgedCount} unacknowledged
            </p>
          </div>
        </div>

        {/* Filter Buttons */}
        <div className="flex gap-2">
          {['all', 'critical', 'warning', 'info'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                filter === f
                  ? 'bg-hololoom-primary text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-12">
            <BellOff className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">No alerts</p>
            <p className="text-sm text-gray-500 mt-1">All systems operating normally</p>
          </div>
        ) : (
          filteredAlerts.map(alert => (
            <div
              key={alert.id}
              className={`p-4 border rounded-lg ${getAlertBgColor(alert.severity)} ${
                alert.acknowledged ? 'opacity-60' : ''
              }`}
            >
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-0.5">
                  {getAlertIcon(alert.severity)}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="font-medium text-gray-900 mb-1">{alert.title}</p>
                      <p className="text-sm text-gray-700 mb-2">{alert.message}</p>
                      <p className="text-xs text-gray-500">
                        {alert.timestamp.toLocaleString()}
                      </p>
                    </div>

                    {!alert.acknowledged && (
                      <button
                        onClick={() => acknowledgeAlert(alert.id)}
                        className="px-3 py-1 text-xs font-medium bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors whitespace-nowrap"
                      >
                        Acknowledge
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default AlertsPanel
