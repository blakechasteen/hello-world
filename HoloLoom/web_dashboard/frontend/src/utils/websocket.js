/**
 * WebSocket Client for Real-Time Updates
 *
 * Handles WebSocket connection to HoloLoom backend for:
 * - Real-time query updates
 * - Live metrics streaming
 * - System status changes
 * - Alert notifications
 */

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

class WebSocketClient {
  constructor() {
    this.ws = null
    this.listeners = new Map()
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
    this.reconnectDelay = 1000
    this.isConnecting = false
  }

  connect() {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      console.log('[WebSocket] Already connected or connecting')
      return
    }

    this.isConnecting = true
    console.log('[WebSocket] Connecting to:', WS_URL)

    try {
      this.ws = new WebSocket(WS_URL)

      this.ws.onopen = () => {
        console.log('[WebSocket] Connected successfully')
        this.isConnecting = false
        this.reconnectAttempts = 0
        this.emit('connected', null)
      }

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          console.log('[WebSocket] Message received:', data.type)
          this.emit(data.type, data.payload)
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error)
        }
      }

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error)
        this.emit('error', error)
      }

      this.ws.onclose = () => {
        console.log('[WebSocket] Connection closed')
        this.isConnecting = false
        this.emit('disconnected', null)
        this.attemptReconnect()
      }
    } catch (error) {
      console.error('[WebSocket] Failed to create connection:', error)
      this.isConnecting = false
    }
  }

  disconnect() {
    if (this.ws) {
      console.log('[WebSocket] Disconnecting...')
      this.ws.close()
      this.ws = null
    }
    this.reconnectAttempts = this.maxReconnectAttempts // Prevent reconnection
  }

  send(type, payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({ type, payload })
      this.ws.send(message)
      console.log('[WebSocket] Sent:', type)
    } else {
      console.warn('[WebSocket] Cannot send, not connected')
    }
  }

  on(eventType, callback) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, [])
    }
    this.listeners.get(eventType).push(callback)
  }

  off(eventType, callback) {
    if (this.listeners.has(eventType)) {
      const callbacks = this.listeners.get(eventType)
      const index = callbacks.indexOf(callback)
      if (index > -1) {
        callbacks.splice(index, 1)
      }
    }
  }

  emit(eventType, data) {
    if (this.listeners.has(eventType)) {
      this.listeners.get(eventType).forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error(`[WebSocket] Error in ${eventType} listener:`, error)
        }
      })
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('[WebSocket] Max reconnect attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`)

    setTimeout(() => {
      this.connect()
    }, delay)
  }

  getConnectionState() {
    if (!this.ws) return 'DISCONNECTED'

    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'CONNECTING'
      case WebSocket.OPEN:
        return 'CONNECTED'
      case WebSocket.CLOSING:
        return 'CLOSING'
      case WebSocket.CLOSED:
        return 'DISCONNECTED'
      default:
        return 'UNKNOWN'
    }
  }
}

// Singleton instance
const wsClient = new WebSocketClient()

export default wsClient
