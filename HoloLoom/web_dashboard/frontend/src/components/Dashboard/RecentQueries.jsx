import React, { useEffect, useState } from 'react'
import { Clock, TrendingUp, CheckCircle, AlertCircle } from 'lucide-react'
import { useHoloLoom } from '../../hooks/useHoloLoom'
import { useWebSocket } from '../../hooks/useWebSocket'

function RecentQueries() {
  const { getAuditTrail } = useHoloLoom()
  const { subscribe } = useWebSocket()
  const [queries, setQueries] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Initial fetch
    fetchQueries()

    // Subscribe to new queries via WebSocket
    const unsubscribe = subscribe('query_completed', (data) => {
      setQueries(prev => [data, ...prev.slice(0, 9)]) // Keep last 10
    })

    return () => unsubscribe()
  }, [getAuditTrail, subscribe])

  const fetchQueries = async () => {
    try {
      const data = await getAuditTrail(10)
      setQueries(data.entries || [])
      setLoading(false)
    } catch (error) {
      console.error('Failed to fetch recent queries:', error)
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

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Recent Queries</h2>
          <p className="text-sm text-gray-600 mt-1">Latest system activity</p>
        </div>
        <button className="text-sm text-hololoom-primary hover:text-hololoom-secondary font-medium">
          View All
        </button>
      </div>

      {queries.length === 0 ? (
        <div className="text-center py-12">
          <Clock className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No recent queries</p>
          <p className="text-sm text-gray-500 mt-1">Submit a query to get started</p>
        </div>
      ) : (
        <div className="space-y-3">
          {queries.map((query, index) => (
            <div
              key={query.decision_id || index}
              className="p-4 border border-gray-200 rounded-lg hover:border-hololoom-primary transition-colors"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    {query.outcome === 'success' ? (
                      <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />
                    )}
                    <span className="text-sm font-medium text-gray-900 truncate">
                      {query.decision_type || 'Query'}
                    </span>
                  </div>

                  {query.reason && (
                    <p className="text-sm text-gray-600 mb-2 line-clamp-2">{query.reason}</p>
                  )}

                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {new Date(query.timestamp).toLocaleTimeString()}
                    </span>
                    {query.confidence !== undefined && (
                      <span className="flex items-center gap-1">
                        <TrendingUp className="w-3 h-3" />
                        {(query.confidence * 100).toFixed(0)}% confidence
                      </span>
                    )}
                  </div>
                </div>

                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  query.outcome === 'success'
                    ? 'bg-green-100 text-green-700'
                    : 'bg-red-100 text-red-700'
                }`}>
                  {query.outcome || 'unknown'}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default RecentQueries
