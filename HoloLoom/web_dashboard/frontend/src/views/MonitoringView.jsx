import React from 'react'
import PrometheusMetrics from '../components/Monitoring/PrometheusMetrics'
import AlertsPanel from '../components/Monitoring/AlertsPanel'
import HistoricalCharts from '../components/Monitoring/HistoricalCharts'

function MonitoringView() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Monitoring</h1>
        <p className="text-gray-600 mt-1">Live metrics, alerts, and performance trends</p>
      </div>

      {/* Alerts Panel */}
      <AlertsPanel />

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Prometheus Metrics */}
        <PrometheusMetrics />

        {/* Historical Charts */}
        <HistoricalCharts />
      </div>
    </div>
  )
}

export default MonitoringView
