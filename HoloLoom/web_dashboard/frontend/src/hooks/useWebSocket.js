import { useEffect, useState, useCallback } from 'react'
import wsClient from '../utils/websocket'

/**
 * Custom hook for WebSocket communication
 *
 * Manages WebSocket connection and provides real-time data updates
 *
 * @param {boolean} autoConnect - Whether to connect automatically on mount
 * @returns {Object} WebSocket state and methods
 */
export function useWebSocket(autoConnect = true) {
  const [isConnected, setIsConnected] = useState(false)
  const [connectionState, setConnectionState] = useState('DISCONNECTED')
  const [lastMessage, setLastMessage] = useState(null)

  // Update connection state
  const updateConnectionState = useCallback(() => {
    const state = wsClient.getConnectionState()
    setConnectionState(state)
    setIsConnected(state === 'CONNECTED')
  }, [])

  // Subscribe to event
  const subscribe = useCallback((eventType, callback) => {
    wsClient.on(eventType, callback)
    return () => wsClient.off(eventType, callback)
  }, [])

  // Send message
  const send = useCallback((type, payload) => {
    wsClient.send(type, payload)
  }, [])

  // Connect/disconnect
  const connect = useCallback(() => {
    wsClient.connect()
  }, [])

  const disconnect = useCallback(() => {
    wsClient.disconnect()
  }, [])

  useEffect(() => {
    // Connection event handlers
    const handleConnected = () => {
      console.log('[useWebSocket] Connected')
      updateConnectionState()
    }

    const handleDisconnected = () => {
      console.log('[useWebSocket] Disconnected')
      updateConnectionState()
    }

    const handleError = (error) => {
      console.error('[useWebSocket] Error:', error)
    }

    // Subscribe to connection events
    wsClient.on('connected', handleConnected)
    wsClient.on('disconnected', handleDisconnected)
    wsClient.on('error', handleError)

    // Auto-connect if enabled
    if (autoConnect) {
      wsClient.connect()
    }

    // Update state initially
    updateConnectionState()

    // Cleanup on unmount
    return () => {
      wsClient.off('connected', handleConnected)
      wsClient.off('disconnected', handleDisconnected)
      wsClient.off('error', handleError)
    }
  }, [autoConnect, updateConnectionState])

  return {
    isConnected,
    connectionState,
    lastMessage,
    subscribe,
    send,
    connect,
    disconnect,
  }
}
