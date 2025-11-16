import axios from 'axios'

// Base API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('[API] Request error:', error)
    return Promise.reject(error)
  }
)

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      console.error('[API] Response error:', error.response.status, error.response.data)
    } else if (error.request) {
      console.error('[API] No response received:', error.request)
    } else {
      console.error('[API] Error:', error.message)
    }
    return Promise.reject(error)
  }
)

// API methods
export const apiClient = {
  // Health check
  health: () => api.get('/health'),

  // Query endpoints
  query: (data) => api.post('/query', data),

  // Statistics
  getStats: () => api.get('/stats'),

  // Audit trail
  getAuditTrail: (limit = 100) => api.get(`/audit-trail?limit=${limit}`),

  // Memory management
  addMemory: (memory) => api.post('/memories/add', memory),

  // Prometheus metrics (custom endpoint we'll add)
  getMetrics: () => api.get('/metrics'),

  // Graph data
  getGraphData: () => api.get('/graph/data'),
}

export default api
