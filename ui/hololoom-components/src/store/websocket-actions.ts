/**
 * WebSocket Store Actions
 *
 * Connects WebSocket events to Zustand store with clean action interface.
 * Provides action creators and subscriptions for real-time streaming.
 *
 * Part of HoloLoom Phase 1 Frontend (November 2025)
 */

import { useEffect, useCallback } from 'react';
import { useHoloLoomStore } from '../store';
import { getWebSocketClient, useWebSocket } from '../WebSocketClient';
import type {
  ReasoningMode,
  WeavingStage,
  StageMetrics,
  ContextChunk,
  ReasoningStep,
  ConnectionState,
  StreamState,
} from '../types';

// ============================================================================
// Action Creators
// ============================================================================

/**
 * Action creators for direct store manipulation.
 * Use these when you need programmatic control outside of WebSocket events.
 */
export const wsActions = {
  // Connection Actions
  setConnected: () => useHoloLoomStore.getState().setConnectionState('connected'),
  setDisconnected: () => useHoloLoomStore.getState().setConnectionState('disconnected'),
  setConnecting: () => useHoloLoomStore.getState().setConnectionState('connecting'),
  setReconnecting: () => useHoloLoomStore.getState().setConnectionState('reconnecting'),
  setConnectionError: () => useHoloLoomStore.getState().setConnectionState('error'),

  // Session Actions
  startStream: (sessionId: string, query: string, mode: ReasoningMode) => {
    useHoloLoomStore.getState().startSession(sessionId, query, mode);
  },
  endStream: () => {
    useHoloLoomStore.getState().endSession();
  },

  // Token Actions
  appendToken: (token: string) => {
    useHoloLoomStore.getState().appendToken(token);
  },
  setResponse: (text: string) => {
    useHoloLoomStore.getState().setAccumulatedText(text);
  },

  // Context Actions
  addContext: (chunk: ContextChunk) => {
    useHoloLoomStore.getState().addContextChunk(chunk);
  },
  clearContext: () => {
    useHoloLoomStore.getState().clearContext();
  },

  // Confidence Actions
  updateConfidence: (confidence: number, epistemic?: number) => {
    useHoloLoomStore.getState().updateConfidence(confidence, epistemic);
  },

  // Stage Actions
  setStageActive: (stage: WeavingStage) => {
    useHoloLoomStore.getState().updateStage(stage, 'active');
  },
  setStageComplete: (stage: WeavingStage, metrics?: StageMetrics) => {
    useHoloLoomStore.getState().updateStage(stage, 'complete', metrics);
  },
  setStageError: (stage: WeavingStage) => {
    useHoloLoomStore.getState().updateStage(stage, 'error');
  },

  // Reasoning Actions
  addReasoningStep: (step: ReasoningStep) => {
    useHoloLoomStore.getState().addReasoningStep(step);
  },
  updateReasoningStep: (index: number, updates: Partial<ReasoningStep>) => {
    useHoloLoomStore.getState().updateReasoningStep(index, updates);
  },

  // Error Actions
  setError: (code: string, message: string) => {
    useHoloLoomStore.getState().setError({ code, message });
  },
  clearError: () => {
    useHoloLoomStore.getState().setError(null);
  },

  // Reset
  reset: () => {
    useHoloLoomStore.getState().reset();
  },
};

// ============================================================================
// WebSocket Connection Hook
// ============================================================================

export interface UseWebSocketConnectionOptions {
  /** Auto-connect on mount */
  autoConnect?: boolean;
  /** WebSocket URL override */
  url?: string;
  /** Called when connected */
  onConnect?: () => void;
  /** Called when disconnected */
  onDisconnect?: () => void;
  /** Called on error */
  onError?: (error: Event) => void;
  /** Called on reconnect attempt */
  onReconnect?: (attempt: number) => void;
}

/**
 * Hook for managing WebSocket connection with store integration.
 * Handles connection lifecycle and state synchronization.
 */
export function useWebSocketConnection(options: UseWebSocketConnectionOptions = {}) {
  const { autoConnect = true, url, onConnect, onDisconnect, onError, onReconnect } = options;

  const connectionState = useHoloLoomStore(state => state.connectionState);
  const reconnectAttempts = useHoloLoomStore(state => state.reconnectAttempts);

  const ws = useWebSocket(
    url ? { url } : undefined,
    {
      onOpen: onConnect,
      onClose: (event) => {
        if (!event.wasClean) {
          onDisconnect?.();
        }
      },
      onError,
      onReconnect,
    }
  );

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && connectionState === 'disconnected') {
      ws.connect().catch(console.error);
    }
  }, [autoConnect]);

  return {
    ...ws,
    connectionState,
    reconnectAttempts,
    isConnecting: connectionState === 'connecting',
    isReconnecting: connectionState === 'reconnecting',
    hasError: connectionState === 'error',
  };
}

// ============================================================================
// Stream Management Hook
// ============================================================================

export interface UseStreamOptions {
  /** Called when stream starts */
  onStreamStart?: (query: string, mode: ReasoningMode) => void;
  /** Called on each token */
  onToken?: (token: string, accumulated: string) => void;
  /** Called when stream completes */
  onStreamEnd?: (response: string) => void;
  /** Called on stream error */
  onStreamError?: (error: { code: string; message: string }) => void;
  /** Called on confidence update */
  onConfidenceUpdate?: (confidence: number, epistemic: number) => void;
  /** Called on stage complete */
  onStageComplete?: (stage: WeavingStage, metrics?: StageMetrics) => void;
}

/**
 * Hook for managing streaming sessions with callbacks.
 */
export function useStream(options: UseStreamOptions = {}) {
  const {
    onStreamStart,
    onToken,
    onStreamEnd,
    onStreamError,
    onConfidenceUpdate,
    onStageComplete,
  } = options;

  // Select relevant state
  const streamState = useHoloLoomStore(state => state.streamState);
  const accumulatedText = useHoloLoomStore(state => state.accumulatedText);
  const tokens = useHoloLoomStore(state => state.tokens);
  const confidence = useHoloLoomStore(state => state.confidence);
  const epistemicConfidence = useHoloLoomStore(state => state.epistemicConfidence);
  const activeStage = useHoloLoomStore(state => state.activeStage);
  const stages = useHoloLoomStore(state => state.stages);
  const error = useHoloLoomStore(state => state.error);
  const query = useHoloLoomStore(state => state.query);
  const mode = useHoloLoomStore(state => state.mode);

  // Subscribe to state changes for callbacks
  useEffect(() => {
    if (streamState === 'streaming' && query && mode && onStreamStart) {
      onStreamStart(query, mode);
    }
  }, [streamState === 'streaming']);

  useEffect(() => {
    if (tokens.length > 0 && onToken) {
      const lastToken = tokens[tokens.length - 1];
      onToken(lastToken, accumulatedText);
    }
  }, [tokens.length]);

  useEffect(() => {
    if (streamState === 'complete' && onStreamEnd) {
      onStreamEnd(accumulatedText);
    }
  }, [streamState === 'complete']);

  useEffect(() => {
    if (error && onStreamError) {
      onStreamError(error);
    }
  }, [error]);

  useEffect(() => {
    if (onConfidenceUpdate) {
      onConfidenceUpdate(confidence, epistemicConfidence);
    }
  }, [confidence, epistemicConfidence]);

  // Track completed stages
  const lastCompletedStageRef = useEffect(() => {
    const completedStages = stages.filter(s => s.status === 'complete');
    if (completedStages.length > 0 && onStageComplete) {
      const lastCompleted = completedStages[completedStages.length - 1];
      onStageComplete(lastCompleted.stage, lastCompleted.metrics);
    }
  }, [stages.filter(s => s.status === 'complete').length]);

  // Get WebSocket client for sending
  const ws = useWebSocket();

  // Send query
  const sendQuery = useCallback((queryText: string, reasoningMode: ReasoningMode = 'direct') => {
    wsActions.reset();
    ws.sendQuery(queryText, reasoningMode);
  }, []);

  // Cancel stream
  const cancel = useCallback(() => {
    ws.cancelStream();
    wsActions.endStream();
  }, []);

  return {
    // State
    streamState,
    isStreaming: streamState === 'streaming',
    isComplete: streamState === 'complete',
    hasError: streamState === 'error',

    // Response
    response: accumulatedText,
    tokens,
    tokenCount: tokens.length,

    // Confidence
    confidence,
    epistemicConfidence,

    // Stages
    activeStage,
    stages,
    completedStages: stages.filter(s => s.status === 'complete'),

    // Error
    error,

    // Actions
    sendQuery,
    cancel,
    reset: wsActions.reset,
  };
}

// ============================================================================
// Store Subscriptions
// ============================================================================

/**
 * Subscribe to specific state changes.
 * Returns unsubscribe function.
 */
export function subscribeToState<T>(
  selector: (state: ReturnType<typeof useHoloLoomStore.getState>) => T,
  callback: (value: T, prevValue: T) => void
): () => void {
  return useHoloLoomStore.subscribe(
    selector,
    callback,
    { fireImmediately: false }
  );
}

/**
 * Subscribe to connection state changes.
 */
export function subscribeToConnection(
  callback: (state: ConnectionState, prev: ConnectionState) => void
): () => void {
  return subscribeToState(state => state.connectionState, callback);
}

/**
 * Subscribe to stream state changes.
 */
export function subscribeToStreamState(
  callback: (state: StreamState, prev: StreamState) => void
): () => void {
  return subscribeToState(state => state.streamState, callback);
}

/**
 * Subscribe to new tokens.
 */
export function subscribeToTokens(
  callback: (tokens: string[], prev: string[]) => void
): () => void {
  return subscribeToState(state => state.tokens, callback);
}

/**
 * Subscribe to confidence changes.
 */
export function subscribeToConfidence(
  callback: (value: { confidence: number; epistemic: number }, prev: { confidence: number; epistemic: number }) => void
): () => void {
  return subscribeToState(
    state => ({ confidence: state.confidence, epistemic: state.epistemicConfidence }),
    callback
  );
}

// ============================================================================
// Exports
// ============================================================================

export type {
  ConnectionState,
  StreamState,
  ReasoningMode,
  WeavingStage,
  StageMetrics,
  ContextChunk,
  ReasoningStep,
};
