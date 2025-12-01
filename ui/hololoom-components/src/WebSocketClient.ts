/**
 * HoloLoom WebSocket Client
 *
 * Manages WebSocket connection to HoloLoom streaming API.
 * Features: auto-reconnect, heartbeat, message dispatching.
 */

import type {
  WebSocketConfig,
  WebSocketHandlers,
  WSMessage,
  AnyWSMessage,
  StreamStartMessage,
  ContextChunkMessage,
  TokenMessage,
  ConfidenceUpdateMessage,
  StageCompleteMessage,
  ReasoningStepMessage,
  StreamEndMessage,
  ErrorMessage,
  HeartbeatMessage,
  ReasoningMode,
} from './types';
import { useHoloLoomStore } from './store';

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: Required<WebSocketConfig> = {
  url: 'ws://localhost:8000/ws/stream',
  autoReconnect: true,
  maxReconnectAttempts: 5,
  reconnectDelay: 1000,
  heartbeatInterval: 30000,
  connectionTimeout: 10000,
};

// ============================================================================
// WebSocket Client Class
// ============================================================================

export class HoloLoomWebSocketClient {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private handlers: WebSocketHandlers;
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private connectionTimer: ReturnType<typeof setTimeout> | null = null;
  private lastHeartbeatTime: number = 0;
  private messageQueue: string[] = [];

  constructor(config?: Partial<WebSocketConfig>, handlers?: WebSocketHandlers) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.handlers = handlers || {};
  }

  // ==========================================================================
  // Connection Management
  // ==========================================================================

  /**
   * Connect to WebSocket server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      // Update store connection state
      const store = useHoloLoomStore.getState();
      store.setConnectionState('connecting');

      // Clear any existing timers
      this.clearTimers();

      // Create WebSocket connection
      try {
        this.ws = new WebSocket(this.config.url);
      } catch (error) {
        store.setConnectionState('error');
        reject(error);
        return;
      }

      // Connection timeout
      this.connectionTimer = setTimeout(() => {
        if (this.ws?.readyState !== WebSocket.OPEN) {
          this.ws?.close();
          store.setConnectionState('error');
          reject(new Error('Connection timeout'));
        }
      }, this.config.connectionTimeout);

      // WebSocket event handlers
      this.ws.onopen = () => {
        this.clearConnectionTimer();
        this.reconnectAttempts = 0;
        store.setConnectionState('connected');
        store.resetReconnectAttempts();
        this.startHeartbeat();
        this.flushMessageQueue();
        this.handlers.onOpen?.();
        resolve();
      };

      this.ws.onclose = (event: CloseEvent) => {
        this.clearTimers();
        store.setConnectionState('disconnected');
        this.handlers.onClose?.(event);

        // Auto-reconnect if enabled and not a clean close
        if (this.config.autoReconnect && !event.wasClean) {
          this.attemptReconnect();
        }
      };

      this.ws.onerror = (error: Event) => {
        store.setConnectionState('error');
        this.handlers.onError?.(error);
      };

      this.ws.onmessage = (event: MessageEvent) => {
        this.handleMessage(event.data);
      };
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.clearTimers();
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    useHoloLoomStore.getState().setConnectionState('disconnected');
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // ==========================================================================
  // Message Handling
  // ==========================================================================

  /**
   * Send message to server
   */
  send(data: object): void {
    const message = JSON.stringify(data);

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(message);
    } else {
      // Queue message for later
      this.messageQueue.push(message);
    }
  }

  /**
   * Send a query to start streaming
   */
  sendQuery(query: string, mode: ReasoningMode = 'direct'): void {
    this.send({
      type: 'query',
      query,
      mode,
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Cancel current stream
   */
  cancelStream(): void {
    this.send({
      type: 'cancel',
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Handle incoming message
   */
  private handleMessage(data: string): void {
    let message: AnyWSMessage;

    try {
      message = JSON.parse(data);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      return;
    }

    // Dispatch to store based on message type
    const store = useHoloLoomStore.getState();

    switch (message.type) {
      case 'stream_start':
        this.handleStreamStart(message as StreamStartMessage, store);
        break;

      case 'context_chunk':
        this.handleContextChunk(message as ContextChunkMessage, store);
        break;

      case 'token':
        this.handleToken(message as TokenMessage, store);
        break;

      case 'confidence_update':
        this.handleConfidenceUpdate(message as ConfidenceUpdateMessage, store);
        break;

      case 'stage_complete':
        this.handleStageComplete(message as StageCompleteMessage, store);
        break;

      case 'reasoning_step':
        this.handleReasoningStep(message as ReasoningStepMessage, store);
        break;

      case 'stream_end':
        this.handleStreamEnd(message as StreamEndMessage, store);
        break;

      case 'error':
        this.handleError(message as ErrorMessage, store);
        break;

      case 'heartbeat':
        this.handleHeartbeat(message as HeartbeatMessage, store);
        break;

      default:
        console.warn('Unknown message type:', (message as WSMessage).type);
    }

    // Call general message handler
    this.handlers.onMessage?.(message);
  }

  // ==========================================================================
  // Message Type Handlers
  // ==========================================================================

  private handleStreamStart(
    message: StreamStartMessage,
    store: ReturnType<typeof useHoloLoomStore.getState>
  ): void {
    store.startSession(message.session_id, message.query, message.mode);
  }

  private handleContextChunk(
    message: ContextChunkMessage,
    store: ReturnType<typeof useHoloLoomStore.getState>
  ): void {
    store.addContextChunk({
      id: `chunk-${message.chunk_index}`,
      content: message.nodes.map(n => n.content).join('\n'),
      relevance: Object.values(message.relevance_scores).reduce((a, b) => a + b, 0) /
        Object.values(message.relevance_scores).length,
      hopDistance: message.hop_distance,
      tokenCount: message.token_count,
      source: message.nodes[0]?.source || 'unknown',
    });
  }

  private handleToken(
    message: TokenMessage,
    store: ReturnType<typeof useHoloLoomStore.getState>
  ): void {
    store.appendToken(message.token);
  }

  private handleConfidenceUpdate(
    message: ConfidenceUpdateMessage,
    store: ReturnType<typeof useHoloLoomStore.getState>
  ): void {
    store.updateConfidence(message.confidence, message.epistemic_confidence);
  }

  private handleStageComplete(
    message: StageCompleteMessage,
    store: ReturnType<typeof useHoloLoomStore.getState>
  ): void {
    store.updateStage(message.stage, 'complete', message.metrics);
  }

  private handleReasoningStep(
    message: ReasoningStepMessage,
    store: ReturnType<typeof useHoloLoomStore.getState>
  ): void {
    store.addReasoningStep({
      index: message.step_index,
      type: message.step_type,
      description: message.description,
      confidence: message.confidence,
      substeps: message.substeps,
      status: 'complete',
    });
  }

  private handleStreamEnd(
    message: StreamEndMessage,
    store: ReturnType<typeof useHoloLoomStore.getState>
  ): void {
    store.setMetrics({
      totalDuration: message.total_duration_ms,
      tokensGenerated: message.tokens_generated,
      cacheHit: message.cache_hit,
    });
    store.endSession();
  }

  private handleError(
    message: ErrorMessage,
    store: ReturnType<typeof useHoloLoomStore.getState>
  ): void {
    store.setError({
      code: message.error_code,
      message: message.message,
    });
  }

  private handleHeartbeat(
    message: HeartbeatMessage,
    store: ReturnType<typeof useHoloLoomStore.getState>
  ): void {
    const now = Date.now();
    const latency = now - this.lastHeartbeatTime;
    store.updateHeartbeat(message.server_time);
    this.handlers.onHeartbeat?.(latency);
  }

  // ==========================================================================
  // Heartbeat Management
  // ==========================================================================

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.lastHeartbeatTime = Date.now();
        this.send({
          type: 'heartbeat',
          timestamp: new Date().toISOString(),
        });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // ==========================================================================
  // Reconnection Logic
  // ==========================================================================

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      useHoloLoomStore.getState().setConnectionState('error');
      return;
    }

    this.reconnectAttempts++;
    useHoloLoomStore.getState().incrementReconnectAttempts();
    useHoloLoomStore.getState().setConnectionState('reconnecting');
    this.handlers.onReconnect?.(this.reconnectAttempts);

    // Exponential backoff
    const delay = this.config.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(() => {
        // Will trigger another reconnect attempt via onclose
      });
    }, delay);
  }

  // ==========================================================================
  // Utility Methods
  // ==========================================================================

  private clearTimers(): void {
    this.clearConnectionTimer();
    this.stopHeartbeat();
    this.clearReconnectTimer();
  }

  private clearConnectionTimer(): void {
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
      const message = this.messageQueue.shift();
      if (message) {
        this.ws.send(message);
      }
    }
  }

  /**
   * Get connection latency from last heartbeat
   */
  getLatency(): number {
    return Date.now() - this.lastHeartbeatTime;
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<WebSocketConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Update event handlers
   */
  updateHandlers(handlers: Partial<WebSocketHandlers>): void {
    this.handlers = { ...this.handlers, ...handlers };
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let clientInstance: HoloLoomWebSocketClient | null = null;

/**
 * Get or create the WebSocket client singleton
 */
export function getWebSocketClient(
  config?: Partial<WebSocketConfig>,
  handlers?: WebSocketHandlers
): HoloLoomWebSocketClient {
  if (!clientInstance) {
    clientInstance = new HoloLoomWebSocketClient(config, handlers);
  } else if (config || handlers) {
    if (config) clientInstance.updateConfig(config);
    if (handlers) clientInstance.updateHandlers(handlers);
  }
  return clientInstance;
}

/**
 * Reset the client singleton (for testing)
 */
export function resetWebSocketClient(): void {
  if (clientInstance) {
    clientInstance.disconnect();
    clientInstance = null;
  }
}

// ============================================================================
// React Hook
// ============================================================================

import { useEffect, useRef, useCallback } from 'react';

/**
 * React hook for using WebSocket client
 */
export function useWebSocket(
  config?: Partial<WebSocketConfig>,
  handlers?: WebSocketHandlers
) {
  const clientRef = useRef<HoloLoomWebSocketClient | null>(null);

  // Initialize client
  useEffect(() => {
    clientRef.current = getWebSocketClient(config, handlers);

    return () => {
      // Don't disconnect on unmount - let the singleton persist
    };
  }, []);

  // Update handlers when they change
  useEffect(() => {
    if (clientRef.current && handlers) {
      clientRef.current.updateHandlers(handlers);
    }
  }, [handlers]);

  // Connect function
  const connect = useCallback(() => {
    return clientRef.current?.connect() ?? Promise.reject(new Error('No client'));
  }, []);

  // Disconnect function
  const disconnect = useCallback(() => {
    clientRef.current?.disconnect();
  }, []);

  // Send query function
  const sendQuery = useCallback((query: string, mode?: ReasoningMode) => {
    clientRef.current?.sendQuery(query, mode);
  }, []);

  // Cancel stream function
  const cancelStream = useCallback(() => {
    clientRef.current?.cancelStream();
  }, []);

  // Check connection status
  const isConnected = useCallback(() => {
    return clientRef.current?.isConnected() ?? false;
  }, []);

  return {
    connect,
    disconnect,
    sendQuery,
    cancelStream,
    isConnected,
    client: clientRef.current,
  };
}
