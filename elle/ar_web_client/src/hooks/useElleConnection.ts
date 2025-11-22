/**
 * useElleConnection Hook - WebSocket connection to HoloLoom AR API
 *
 * Manages bidirectional WebSocket communication with backend:
 * - Session initialization
 * - Query sending
 * - Visualization receiving
 * - Automatic reconnection
 *
 * Created: 2025-11-22
 */

import { useState, useEffect, useRef, useCallback } from 'react'

interface ElleConnectionState {
  connected: boolean
  sessionId: string | null
  visualizations: any[]
  error: string | null
}

interface UseElleConnectionReturn extends ElleConnectionState {
  sendQuery: (text: string, context: any) => Promise<void>
  disconnect: () => void
}

export function useElleConnection(url: string): UseElleConnectionReturn {
  const [state, setState] = useState<ElleConnectionState>({
    connected: false,
    sessionId: null,
    visualizations: [],
    error: null,
  })

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const reconnectAttempts = useRef(0)

  // Connect to WebSocket
  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url)

      ws.onopen = () => {
        console.log('WebSocket connected')
        reconnectAttempts.current = 0

        setState((prev) => ({
          ...prev,
          connected: true,
          error: null,
        }))

        // Start session
        ws.send(JSON.stringify({
          type: 'start_session',
          config: {
            mode: 'immersive-ar',
            features: ['local', 'hit-test', 'anchors'],
          },
        }))
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          handleMessage(message)
        } catch (error) {
          console.error('Failed to parse message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setState((prev) => ({
          ...prev,
          error: 'Connection error',
        }))
      }

      ws.onclose = () => {
        console.log('WebSocket closed')
        setState((prev) => ({
          ...prev,
          connected: false,
        }))

        // Attempt reconnection with exponential backoff
        if (reconnectAttempts.current < 5) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000)
          reconnectAttempts.current++

          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`)

          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect()
          }, delay)
        }
      }

      wsRef.current = ws
    } catch (error) {
      console.error('Failed to connect:', error)
      setState((prev) => ({
        ...prev,
        error: 'Failed to connect',
      }))
    }
  }, [url])

  // Handle incoming messages
  const handleMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'session_started':
        setState((prev) => ({
          ...prev,
          sessionId: message.session_id,
        }))
        console.log('Session started:', message.session_id)
        break

      case 'response':
        // Update visualizations from Elle
        setState((prev) => ({
          ...prev,
          visualizations: message.data.visualizations || [],
        }))
        console.log('Received visualizations:', message.data.visualizations)
        break

      case 'error':
        console.error('Server error:', message.error)
        setState((prev) => ({
          ...prev,
          error: message.error,
        }))
        break

      default:
        console.log('Unknown message type:', message.type)
    }
  }, [])

  // Send query to backend
  const sendQuery = useCallback(async (text: string, context: any) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected')
      return
    }

    const message = {
      type: 'ar_event',
      event: {
        eventType: 'voice',
        transcript: text,
        context,
        timestamp: new Date().toISOString(),
      },
    }

    wsRef.current.send(JSON.stringify(message))
  }, [])

  // Disconnect
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: 'end_session' }))
      wsRef.current.close()
      wsRef.current = null
    }

    setState({
      connected: false,
      sessionId: null,
      visualizations: [],
      error: null,
    })
  }, [])

  // Initialize connection on mount
  useEffect(() => {
    connect()

    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    ...state,
    sendQuery,
    disconnect,
  }
}
