/**
 * useAgentWebSocket Hook
 *
 * React hook for managing WebSocket connection and injection feedback.
 * Integrates with step store to handle MRF/MCTS injection events.
 *
 * HoloLoom Agent Manager UI - Phase 5
 * Date: 2025-12-15
 */

import { useEffect, useCallback, useState, useRef } from 'react';
import { wsClient, ConnectionState, WebSocketMessage } from '../lib/websocketClient';
import { useStepStore, InjectionFeedbackMessage } from '../stores/stepStore';

// ============================================================================
// Types
// ============================================================================

export interface UseAgentWebSocketOptions {
  /** Thread ID to subscribe to (optional - subscribes to all if not provided) */
  threadId?: string;

  /** Enable injection feedback handling */
  enableInjectionFeedback?: boolean;

  /** Callback when MRF injection feedback received */
  onMRFInjected?: (message: InjectionFeedbackMessage) => void;

  /** Callback when MCTS injection feedback received */
  onMCTSInjected?: (message: InjectionFeedbackMessage) => void;

  /** Callback when any message received */
  onMessage?: (message: WebSocketMessage) => void;

  /** Callback when connection state changes */
  onStateChange?: (state: ConnectionState) => void;

  /** Auto-connect on mount (default: true) */
  autoConnect?: boolean;
}

export interface UseAgentWebSocketReturn {
  /** Current connection state */
  connectionState: ConnectionState;

  /** Whether currently connected */
  isConnected: boolean;

  /** Connect to WebSocket server */
  connect: () => void;

  /** Disconnect from WebSocket server */
  disconnect: () => void;

  /** Subscribe to a pattern (e.g., 'thread:xyz', '*') */
  subscribe: (pattern: string) => void;

  /** Unsubscribe from a pattern */
  unsubscribe: (pattern: string) => void;

  /** Send a message */
  send: (action: string, data?: any) => void;

  /** Last injection feedback received */
  lastInjectionFeedback: InjectionFeedbackMessage | null;
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * Hook for managing WebSocket connection and injection feedback
 */
export function useAgentWebSocket(
  options: UseAgentWebSocketOptions = {}
): UseAgentWebSocketReturn {
  const {
    threadId,
    enableInjectionFeedback = true,
    onMRFInjected,
    onMCTSInjected,
    onMessage,
    onStateChange,
    autoConnect = true,
  } = options;

  // State
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    wsClient.getState()
  );
  const [lastInjectionFeedback, setLastInjectionFeedback] =
    useState<InjectionFeedbackMessage | null>(null);

  // Refs for callbacks (avoid stale closures)
  const onMRFInjectedRef = useRef(onMRFInjected);
  const onMCTSInjectedRef = useRef(onMCTSInjected);
  const onMessageRef = useRef(onMessage);
  const onStateChangeRef = useRef(onStateChange);

  // Update refs when callbacks change
  useEffect(() => {
    onMRFInjectedRef.current = onMRFInjected;
    onMCTSInjectedRef.current = onMCTSInjected;
    onMessageRef.current = onMessage;
    onStateChangeRef.current = onStateChange;
  }, [onMRFInjected, onMCTSInjected, onMessage, onStateChange]);

  // Step store for injection feedback
  const handleInjectionFeedback = useStepStore(
    (state) => state.handleInjectionFeedback
  );

  // Handle MRF injection feedback
  const handleMRFInjected = useCallback(
    (message: WebSocketMessage) => {
      if (!enableInjectionFeedback) return;

      const feedback: InjectionFeedbackMessage = {
        type: 'MRF_INJECTED',
        thread_id: message.thread_id || '',
        step_id: message.step_id || '',
        success: message.success ?? true,
        injection_type: 'mrf',
        strategy: message.strategy,
        timestamp: message.timestamp,
        message: message.message,
      };

      // Update step store
      handleInjectionFeedback(feedback);

      // Update local state
      setLastInjectionFeedback(feedback);

      // Call user callback
      onMRFInjectedRef.current?.(feedback);

      // Log for debugging
      console.log(
        `[WebSocket] MRF Injected: ${feedback.strategy} on step ${feedback.step_id}`,
        feedback
      );
    },
    [enableInjectionFeedback, handleInjectionFeedback]
  );

  // Handle MCTS injection feedback
  const handleMCTSInjected = useCallback(
    (message: WebSocketMessage) => {
      if (!enableInjectionFeedback) return;

      const feedback: InjectionFeedbackMessage = {
        type: 'MCTS_INJECTED',
        thread_id: message.thread_id || '',
        step_id: message.step_id || '',
        success: message.success ?? true,
        injection_type: 'mcts',
        config: message.config,
        timestamp: message.timestamp,
        message: message.message,
      };

      // Update step store
      handleInjectionFeedback(feedback);

      // Update local state
      setLastInjectionFeedback(feedback);

      // Call user callback
      onMCTSInjectedRef.current?.(feedback);

      // Log for debugging
      console.log(
        `[WebSocket] MCTS Injected: budget=${feedback.config?.budget} on step ${feedback.step_id}`,
        feedback
      );
    },
    [enableInjectionFeedback, handleInjectionFeedback]
  );

  // Handle all messages (for user callback)
  const handleMessage = useCallback((message: WebSocketMessage) => {
    onMessageRef.current?.(message);
  }, []);

  // Handle state changes
  const handleStateChange = useCallback((state: ConnectionState) => {
    setConnectionState(state);
    onStateChangeRef.current?.(state);
  }, []);

  // Setup WebSocket event handlers
  useEffect(() => {
    // Register handlers
    const unsubMRF = wsClient.on('MRF_INJECTED', handleMRFInjected);
    const unsubMCTS = wsClient.on('MCTS_INJECTED', handleMCTSInjected);
    const unsubAll = wsClient.on('*', handleMessage);
    const unsubState = wsClient.onStateChange(handleStateChange);

    // Auto-connect if enabled
    if (autoConnect && wsClient.getState() === 'disconnected') {
      wsClient.connect();
    }

    // Subscribe to thread pattern if provided
    if (threadId) {
      wsClient.subscribe(`thread:${threadId}`);
    } else {
      // Subscribe to all messages
      wsClient.subscribe('*');
    }

    // Cleanup
    return () => {
      unsubMRF();
      unsubMCTS();
      unsubAll();
      unsubState();

      // Unsubscribe from thread pattern
      if (threadId) {
        wsClient.unsubscribe(`thread:${threadId}`);
      }
    };
  }, [
    threadId,
    autoConnect,
    handleMRFInjected,
    handleMCTSInjected,
    handleMessage,
    handleStateChange,
  ]);

  // Public API
  const connect = useCallback(() => {
    wsClient.connect();
  }, []);

  const disconnect = useCallback(() => {
    wsClient.disconnect();
  }, []);

  const subscribe = useCallback((pattern: string) => {
    wsClient.subscribe(pattern);
  }, []);

  const unsubscribe = useCallback((pattern: string) => {
    wsClient.unsubscribe(pattern);
  }, []);

  const send = useCallback((action: string, data?: any) => {
    wsClient.send(action, data);
  }, []);

  return {
    connectionState,
    isConnected: connectionState === 'connected',
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    send,
    lastInjectionFeedback,
  };
}

// ============================================================================
// Convenience Hooks
// ============================================================================

/**
 * Hook specifically for injection feedback handling
 * Simplified interface for components that only care about injections
 */
export function useInjectionFeedback(threadId?: string) {
  const [lastMRF, setLastMRF] = useState<InjectionFeedbackMessage | null>(null);
  const [lastMCTS, setLastMCTS] = useState<InjectionFeedbackMessage | null>(null);

  const { isConnected, lastInjectionFeedback } = useAgentWebSocket({
    threadId,
    enableInjectionFeedback: true,
    onMRFInjected: (feedback) => {
      setLastMRF(feedback);
    },
    onMCTSInjected: (feedback) => {
      setLastMCTS(feedback);
    },
  });

  return {
    isConnected,
    lastMRF,
    lastMCTS,
    lastInjectionFeedback,
  };
}

/**
 * Hook for connection status only
 * Minimal hook for components that just need to show connection state
 */
export function useWebSocketStatus() {
  const [connectionState, setConnectionState] = useState<ConnectionState>(
    wsClient.getState()
  );

  useEffect(() => {
    const unsubscribe = wsClient.onStateChange(setConnectionState);
    return unsubscribe;
  }, []);

  return {
    connectionState,
    isConnected: connectionState === 'connected',
    isReconnecting: connectionState === 'reconnecting',
    hasError: connectionState === 'error',
  };
}

export default useAgentWebSocket;
