import React from 'react'
import SystemStatus from '../components/Dashboard/SystemStatus'
import QueryInterface from '../components/Dashboard/QueryInterface'
import RecentQueries from '../components/Dashboard/RecentQueries'
import PerformanceMetrics from '../components/Dashboard/PerformanceMetrics'

function DashboardView() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">Real-time system overview and query interface</p>
      </div>

      {/* System Status */}
      <SystemStatus />

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Query Interface (2/3 width) */}
        <div className="lg:col-span-2">
          <QueryInterface />
        </div>

        {/* Performance Metrics (1/3 width) */}
        <div>
          <PerformanceMetrics />
        </div>
      </div>

      {/* Recent Queries */}
      <RecentQueries />
    </div>
  )
}

export default DashboardView
