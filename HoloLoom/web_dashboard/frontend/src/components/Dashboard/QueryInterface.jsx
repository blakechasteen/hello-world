import React, { useState } from 'react'
import { Send, Loader, CheckCircle, AlertCircle } from 'lucide-react'
import { useHoloLoom } from '../../hooks/useHoloLoom'

const REASONING_MODES = [
  { value: 'direct', label: 'Direct', description: 'Single-pass answer (~150ms)' },
  { value: 'verify', label: 'Verify', description: 'Answer + verification (~600ms)' },
  { value: 'research', label: 'Research', description: 'Multi-query exploration (~900ms)' },
  { value: 'plan_execute', label: 'Plan & Execute', description: 'Goal decomposition (~750ms)' },
]

function QueryInterface() {
  const { query, loading } = useHoloLoom()
  const [queryText, setQueryText] = useState('')
  const [mode, setMode] = useState('verify')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!queryText.trim()) {
      setError('Please enter a query')
      return
    }

    setError(null)
    setResult(null)

    try {
      const response = await query(queryText, { mode })
      setResult(response)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Query failed')
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Query Interface</h2>
          <p className="text-sm text-gray-600 mt-1">Ask HoloLoom anything</p>
        </div>
      </div>

      {/* Query Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Text Input */}
        <div>
          <textarea
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder="Enter your query here..."
            rows={4}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-hololoom-primary focus:border-transparent resize-none"
            disabled={loading}
          />
        </div>

        {/* Mode Selection */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {REASONING_MODES.map((modeOption) => (
            <button
              key={modeOption.value}
              type="button"
              onClick={() => setMode(modeOption.value)}
              className={`p-3 rounded-lg border-2 text-left transition-all ${
                mode === modeOption.value
                  ? 'border-hololoom-primary bg-hololoom-primary/5'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <p className="font-medium text-sm text-gray-900">{modeOption.label}</p>
              <p className="text-xs text-gray-500 mt-1">{modeOption.description}</p>
            </button>
          ))}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || !queryText.trim()}
          className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-gradient-to-r from-hololoom-primary to-hololoom-secondary text-white font-medium rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader className="w-5 h-5 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Send className="w-5 h-5" />
              Submit Query
            </>
          )}
        </button>
      </form>

      {/* Error Display */}
      {error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-red-900">Error</p>
            <p className="text-sm text-red-700 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Result Display */}
      {result && (
        <div className="mt-6 space-y-4">
          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="font-medium text-green-900">Response</p>
                <p className="text-sm text-green-800 mt-2 whitespace-pre-wrap">{result.response}</p>
              </div>
            </div>
          </div>

          {/* Metadata */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Confidence</p>
              <p className="text-lg font-semibold text-gray-900">{(result.confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Mode</p>
              <p className="text-lg font-semibold text-gray-900">{result.reasoning_mode}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Steps</p>
              <p className="text-lg font-semibold text-gray-900">{result.total_queries}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Duration</p>
              <p className="text-lg font-semibold text-gray-900">{result.total_duration_ms.toFixed(0)}ms</p>
            </div>
          </div>

          {/* Verification (if available) */}
          {result.verification && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="font-medium text-blue-900 mb-2">Verification</p>
              <div className="space-y-2 text-sm">
                <p>
                  <span className="font-medium">Verified:</span>{' '}
                  {result.verification.verified ? 'Yes' : 'No'}
                </p>
                <p>
                  <span className="font-medium">Confidence:</span>{' '}
                  {(result.verification.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default QueryInterface
