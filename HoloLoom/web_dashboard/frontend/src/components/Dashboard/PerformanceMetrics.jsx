import React, { useEffect, useState } from 'react'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { useHoloLoom } from '../../hooks/useHoloLoom'
import { useWebSocket } from '../../hooks/useWebSocket'

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

function PerformanceMetrics() {
  const { getStats } = useHoloLoom()
  const { subscribe } = useWebSocket()
  const [latencyData, setLatencyData] = useState({
    labels: [],
    datasets: [{
      label: 'Latency (ms)',
      data: [],
      borderColor: 'rgb(102, 126, 234)',
      backgroundColor: 'rgba(102, 126, 234, 0.1)',
      fill: true,
      tension: 0.4,
    }]
  })

  useEffect(() => {
    // Initial fetch
    fetchMetrics()

    // Subscribe to real-time updates
    const unsubscribe = subscribe('metrics_update', (data) => {
      if (data.latency_ms !== undefined) {
        updateLatencyChart(data.latency_ms)
      }
    })

    // Refresh every 5 seconds
    const interval = setInterval(fetchMetrics, 5000)

    return () => {
      unsubscribe()
      clearInterval(interval)
    }
  }, [subscribe])

  const fetchMetrics = async () => {
    try {
      const stats = await getStats()
      if (stats.avg_latency_ms !== undefined) {
        updateLatencyChart(stats.avg_latency_ms)
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error)
    }
  }

  const updateLatencyChart = (latency) => {
    setLatencyData(prev => {
      const newLabels = [...prev.labels, new Date().toLocaleTimeString()]
      const newData = [...prev.datasets[0].data, latency]

      // Keep last 20 data points
      const maxPoints = 20
      if (newLabels.length > maxPoints) {
        newLabels.shift()
        newData.shift()
      }

      return {
        labels: newLabels,
        datasets: [{
          ...prev.datasets[0],
          data: newData,
        }]
      }
    })
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        padding: 12,
        cornerRadius: 8,
        titleFont: {
          size: 13,
          weight: 'bold',
        },
        bodyFont: {
          size: 12,
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: false,
        },
        ticks: {
          maxTicksLimit: 6,
          font: {
            size: 10,
          },
        },
      },
      y: {
        display: true,
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          font: {
            size: 10,
          },
          callback: (value) => `${value}ms`,
        },
      },
    },
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Performance</h2>
        <p className="text-sm text-gray-600 mt-1">Real-time latency tracking</p>
      </div>

      {/* Chart */}
      <div style={{ height: '250px' }}>
        <Line data={latencyData} options={chartOptions} />
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-4 mt-6 pt-6 border-t border-gray-200">
        <div>
          <p className="text-xs text-gray-500 mb-1">Current</p>
          <p className="text-2xl font-bold text-gray-900">
            {latencyData.datasets[0].data.length > 0
              ? `${latencyData.datasets[0].data[latencyData.datasets[0].data.length - 1].toFixed(0)}ms`
              : 'N/A'}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1">Average</p>
          <p className="text-2xl font-bold text-gray-900">
            {latencyData.datasets[0].data.length > 0
              ? `${(latencyData.datasets[0].data.reduce((a, b) => a + b, 0) / latencyData.datasets[0].data.length).toFixed(0)}ms`
              : 'N/A'}
          </p>
        </div>
      </div>
    </div>
  )
}

export default PerformanceMetrics
