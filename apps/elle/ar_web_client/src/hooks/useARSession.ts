/**
 * useARSession Hook - AR session state management
 *
 * Manages AR spatial context and updates.
 * Throttles context updates to avoid overwhelming backend.
 *
 * Created: 2025-11-22
 */

import { useState, useCallback, useRef } from 'react'

interface ARContext {
  userPosition: { x: number; y: number; z: number }
  userRotation: { x: number; y: number; z: number; w: number }
  gazeDirection: { x: number; y: number; z: number }
  visibleObjects: any[]
  sessionId: string
  platform: string
}

interface UseARSessionReturn {
  context: ARContext | null
  updateContext: (context: ARContext) => void
}

const CONTEXT_UPDATE_THROTTLE_MS = 100 // 10 updates per second max

export function useARSession(): UseARSessionReturn {
  const [context, setContext] = useState<ARContext | null>(null)
  const lastUpdateRef = useRef<number>(0)
  const pendingUpdateRef = useRef<ARContext | null>(null)
  const throttleTimeoutRef = useRef<number | null>(null)

  // Throttled context update
  const updateContext = useCallback((newContext: ARContext) => {
    const now = Date.now()
    const timeSinceLastUpdate = now - lastUpdateRef.current

    if (timeSinceLastUpdate >= CONTEXT_UPDATE_THROTTLE_MS) {
      // Immediate update
      setContext(newContext)
      lastUpdateRef.current = now
      pendingUpdateRef.current = null

      // Clear any pending timeout
      if (throttleTimeoutRef.current) {
        clearTimeout(throttleTimeoutRef.current)
        throttleTimeoutRef.current = null
      }
    } else {
      // Queue update for later
      pendingUpdateRef.current = newContext

      // Set timeout if not already set
      if (!throttleTimeoutRef.current) {
        const remaining = CONTEXT_UPDATE_THROTTLE_MS - timeSinceLastUpdate

        throttleTimeoutRef.current = window.setTimeout(() => {
          if (pendingUpdateRef.current) {
            setContext(pendingUpdateRef.current)
            lastUpdateRef.current = Date.now()
            pendingUpdateRef.current = null
          }
          throttleTimeoutRef.current = null
        }, remaining)
      }
    }
  }, [])

  return {
    context,
    updateContext,
  }
}
