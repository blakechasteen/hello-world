import { useState, useEffect, useCallback } from 'react'
import { apiClient } from '../utils/api'

/**
 * Custom hook for Prometheus metrics
 *
 * Fetches and manages Prometheus metrics with auto-refresh
 *
 * @param {number} refreshInterval - Auto-refresh interval in milliseconds (0 = disabled)
 * @returns {Object} Metrics state and methods
 */
export function usePrometheus(refreshInterval = 5000) {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)

  // Fetch metrics
  const fetchMetrics = useCallback(async () => {
    try {
      setError(null)
      const response = await apiClient.getMetrics()
      setMetrics(response.data)
      setLastUpdate(new Date())
      setLoading(false)
    } catch (err) {
      console.error('Failed to fetch Prometheus metrics:', err)
      setError(err.message)
      setLoading(false)
    }
  }, [])

  // Auto-refresh effect
  useEffect(() => {
    // Initial fetch
    fetchMetrics()

    // Set up auto-refresh if enabled
    if (refreshInterval > 0) {
      const intervalId = setInterval(fetchMetrics, refreshInterval)
      return () => clearInterval(intervalId)
    }
  }, [refreshInterval, fetchMetrics])

  // Manual refresh
  const refresh = useCallback(() => {
    setLoading(true)
    fetchMetrics()
  }, [fetchMetrics])

  return {
    metrics,
    loading,
    error,
    lastUpdate,
    refresh,
  }
}

/**
 * Parse Prometheus metrics into structured data
 *
 * @param {string} metricsText - Raw Prometheus metrics text
 * @returns {Object} Parsed metrics by name
 */
export function parsePrometheusMetrics(metricsText) {
  const metrics = {}
  const lines = metricsText.split('\n')

  for (const line of lines) {
    // Skip comments and empty lines
    if (line.startsWith('#') || line.trim() === '') {
      continue
    }

    // Parse metric line (format: metric_name{labels} value timestamp)
    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?([^}]*)\}?\s+([^\s]+)/)
    if (match) {
      const [, name, labelsStr, value] = match
      const labels = {}

      // Parse labels
      if (labelsStr) {
        const labelMatches = labelsStr.matchAll(/([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"/g)
        for (const labelMatch of labelMatches) {
          labels[labelMatch[1]] = labelMatch[2]
        }
      }

      // Store metric
      if (!metrics[name]) {
        metrics[name] = []
      }

      metrics[name].push({
        labels,
        value: parseFloat(value),
      })
    }
  }

  return metrics
}
