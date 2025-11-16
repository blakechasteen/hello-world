import { useState, useCallback } from 'react'
import { apiClient } from '../utils/api'

/**
 * Custom hook for HoloLoom API interactions
 *
 * Provides methods for querying, stats, and memory management
 *
 * @returns {Object} API methods and state
 */
export function useHoloLoom() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Query HoloLoom
  const query = useCallback(async (text, options = {}) => {
    setLoading(true)
    setError(null)

    try {
      const response = await apiClient.query({
        text,
        mode: options.mode || 'verify',
        max_steps: options.maxSteps || 5,
        context: options.context || null,
      })

      setLoading(false)
      return response.data
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
      setLoading(false)
      throw err
    }
  }, [])

  // Get system stats
  const getStats = useCallback(async () => {
    try {
      const response = await apiClient.getStats()
      return response.data
    } catch (err) {
      console.error('Failed to fetch stats:', err)
      throw err
    }
  }, [])

  // Get audit trail
  const getAuditTrail = useCallback(async (limit = 100) => {
    try {
      const response = await apiClient.getAuditTrail(limit)
      return response.data
    } catch (err) {
      console.error('Failed to fetch audit trail:', err)
      throw err
    }
  }, [])

  // Add memory
  const addMemory = useCallback(async (memory) => {
    try {
      const response = await apiClient.addMemory(memory)
      return response.data
    } catch (err) {
      console.error('Failed to add memory:', err)
      throw err
    }
  }, [])

  // Health check
  const checkHealth = useCallback(async () => {
    try {
      const response = await apiClient.health()
      return response.data
    } catch (err) {
      console.error('Health check failed:', err)
      throw err
    }
  }, [])

  return {
    loading,
    error,
    query,
    getStats,
    getAuditTrail,
    addMemory,
    checkHealth,
  }
}
