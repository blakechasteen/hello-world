import React from 'react'
import { Menu, Search, Bell, Settings, Activity } from 'lucide-react'
import { useHoloLoom } from '../../hooks/useHoloLoom'
import { useWebSocket } from '../../hooks/useWebSocket'

function Header({ onMenuClick }) {
  const { checkHealth } = useHoloLoom()
  const { isConnected, connectionState } = useWebSocket()
  const [healthStatus, setHealthStatus] = React.useState(null)

  React.useEffect(() => {
    // Check health on mount
    checkHealth()
      .then(data => setHealthStatus(data))
      .catch(() => setHealthStatus({ status: 'error' }))

    // Check periodically
    const interval = setInterval(() => {
      checkHealth()
        .then(data => setHealthStatus(data))
        .catch(() => setHealthStatus({ status: 'error' }))
    }, 30000) // Every 30 seconds

    return () => clearInterval(interval)
  }, [checkHealth])

  const getStatusColor = () => {
    if (!healthStatus) return 'bg-gray-400'
    if (healthStatus.status === 'ok') return 'bg-green-500'
    return 'bg-red-500'
  }

  const getWebSocketColor = () => {
    switch (connectionState) {
      case 'CONNECTED':
        return 'bg-green-500'
      case 'CONNECTING':
        return 'bg-yellow-500'
      default:
        return 'bg-red-500'
    }
  }

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left: Menu and Logo */}
        <div className="flex items-center gap-4">
          <button
            onClick={onMenuClick}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <Menu className="w-5 h-5 text-gray-600" />
          </button>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-hololoom-primary to-hololoom-secondary flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">HoloLoom</h1>
              <p className="text-xs text-gray-500">Dashboard v1.0.0</p>
            </div>
          </div>
        </div>

        {/* Center: Search */}
        <div className="flex-1 max-w-2xl mx-8">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search queries, graphs, metrics..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-hololoom-primary focus:border-transparent"
            />
          </div>
        </div>

        {/* Right: Status and Actions */}
        <div className="flex items-center gap-4">
          {/* API Status */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-50">
            <div className={`w-2 h-2 rounded-full ${getStatusColor()} pulse-dot`} />
            <span className="text-xs font-medium text-gray-700">
              {healthStatus?.status === 'ok' ? 'API Online' : 'API Offline'}
            </span>
          </div>

          {/* WebSocket Status */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-50">
            <div className={`w-2 h-2 rounded-full ${getWebSocketColor()} pulse-dot`} />
            <span className="text-xs font-medium text-gray-700">
              WS: {connectionState}
            </span>
          </div>

          {/* Notifications */}
          <button className="p-2 rounded-lg hover:bg-gray-100 transition-colors relative">
            <Bell className="w-5 h-5 text-gray-600" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
          </button>

          {/* Settings */}
          <button className="p-2 rounded-lg hover:bg-gray-100 transition-colors">
            <Settings className="w-5 h-5 text-gray-600" />
          </button>
        </div>
      </div>
    </header>
  )
}

export default Header
