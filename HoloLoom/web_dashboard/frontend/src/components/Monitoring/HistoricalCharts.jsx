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
import { TrendingUp, BarChart3, Clock } from 'lucide-react'

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

function HistoricalCharts() {
  const [timeRange, setTimeRange] = useState('1h') // 1h, 6h, 24h, 7d
  const [chartData, setChartData] = useState(null)

  useEffect(() => {
    // Generate demo data based on time range
    setChartData(generateDemoData(timeRange))
  }, [timeRange])

  const generateDemoData = (range) => {
    const dataPoints = range === '1h' ? 12 : range === '6h' ? 36 : range === '24h' ? 48 : 168
    const labels = []
    const latencyData = []
    const throughputData = []
    const errorData = []

    for (let i = 0; i < dataPoints; i++) {
      // Generate time labels
      const date = new Date()
      if (range === '1h') {
        date.setMinutes(date.getMinutes() - (dataPoints - i) * 5)
        labels.push(date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
      } else if (range === '6h') {
        date.setMinutes(date.getMinutes() - (dataPoints - i) * 10)
        labels.push(date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }))
      } else if (range === '24h') {
        date.setMinutes(date.getMinutes() - (dataPoints - i) * 30)
        labels.push(date.toLocaleTimeString([], { hour: '2-digit' }))
      } else {
        date.setHours(date.getHours() - (dataPoints - i))
        labels.push(date.toLocaleDateString([], { month: 'short', day: 'numeric' }))
      }

      // Generate synthetic data with some variation
      latencyData.push(120 + Math.random() * 60 + Math.sin(i / 5) * 20)
      throughputData.push(50 + Math.random() * 30 + Math.cos(i / 7) * 15)
      errorData.push(Math.random() * 3)
    }

    return {
      labels,
      datasets: [
        {
          label: 'Latency (ms)',
          data: latencyData,
          borderColor: 'rgb(102, 126, 234)',
          backgroundColor: 'rgba(102, 126, 234, 0.1)',
          yAxisID: 'y',
          fill: true,
          tension: 0.4,
        },
        {
          label: 'Throughput (req/s)',
          data: throughputData,
          borderColor: 'rgb(16, 185, 129)',
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          yAxisID: 'y1',
          fill: true,
          tension: 0.4,
        },
        {
          label: 'Error Rate (%)',
          data: errorData,
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          yAxisID: 'y2',
          fill: true,
          tension: 0.4,
        },
      ],
    }
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 15,
          font: {
            size: 12,
          },
        },
      },
      tooltip: {
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
          maxTicksLimit: 10,
          font: {
            size: 10,
          },
        },
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Latency (ms)',
          font: {
            size: 11,
          },
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Throughput (req/s)',
          font: {
            size: 11,
          },
        },
        grid: {
          drawOnChartArea: false,
        },
      },
      y2: {
        type: 'linear',
        display: false,
      },
    },
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-purple-100">
            <BarChart3 className="w-5 h-5 text-purple-600" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Historical Trends</h2>
            <p className="text-sm text-gray-600">Performance over time</p>
          </div>
        </div>

        {/* Time Range Selector */}
        <div className="flex gap-2">
          {['1h', '6h', '24h', '7d'].map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                timeRange === range
                  ? 'bg-hololoom-primary text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div style={{ height: '350px' }}>
        {chartData && <Line data={chartData} options={chartOptions} />}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-gray-200">
        <div className="text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-blue-600" />
            <p className="text-xs font-medium text-gray-500">Avg Latency</p>
          </div>
          <p className="text-2xl font-bold text-gray-900">142ms</p>
          <p className="text-xs text-green-600 mt-1">-5.2% vs prev</p>
        </div>

        <div className="text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <TrendingUp className="w-4 h-4 text-green-600" />
            <p className="text-xs font-medium text-gray-500">Avg Throughput</p>
          </div>
          <p className="text-2xl font-bold text-gray-900">65.3 req/s</p>
          <p className="text-xs text-green-600 mt-1">+8.1% vs prev</p>
        </div>

        <div className="text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-red-600" />
            <p className="text-xs font-medium text-gray-500">Error Rate</p>
          </div>
          <p className="text-2xl font-bold text-gray-900">1.2%</p>
          <p className="text-xs text-green-600 mt-1">-0.3% vs prev</p>
        </div>
      </div>
    </div>
  )
}

export default HistoricalCharts
