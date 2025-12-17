/**
 * WebSocket Client for Agent Manager UI
 *
 * Provides real-time bidirectional communication with the Agent Manager backend.
 * Features:
 * - Auto-reconnection with exponential backoff
 * - Pattern-based subscription system
 * - Heartbeat/ping mechanism
 * - Message queueing during disconnection
 * - Event-based message routing
 *
 * Date: 2025-12-11
 */

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export interface WebSocketMessage {
  type: string;
  timestamp: string;
  thread_id?: string;
  project_id?: string;
  step_index?: number;
  status?: string;
  progress?: number;
  data?: any;
  [key: string]: any;
}

export interface OutgoingMessage {
  action: string;
  pattern?: string;
  data?: any;
}

export type MessageHandler = (message: WebSocketMessage) => void;
export type StateChangeHandler = (state: ConnectionState) => void;

/**
 * WebSocket Manager for Agent Manager backend communication
 */
export class AgentManagerWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10;
  private reconnectDelay: number = 1000;
  private maxReconnectDelay: number = 30000;
  private state: ConnectionState = 'disconnected';

  // Subscriptions and handlers
  private subscriptions: Set<string> = new Set();
  private messageHandlers: Map<string, Set<MessageHandler>> = new Map();
  private stateHandlers: Set<StateChangeHandler> = new Set();

  // Message queue for offline mode
  private messageQueue: OutgoingMessage[] = [];

  // Heartbeat
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private heartbeatTimeout: NodeJS.Timeout | null = null;
  private lastHeartbeatTime: number = 0;

  // Backoff calculation
  private getReconnectDelay(): number {
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay
    );
    // Add jitter (±20%)
    const jitter = delay * 0.2 * (Math.random() * 2 - 1);
    return Math.max(100, delay + jitter);
  }

  constructor(url: string) {
    this.url = url;
  }

  /**
   * Connect to WebSocket server
   */
  public connect(): void {
    if (this.state === 'connecting' || this.state === 'connected') {
      console.warn('WebSocket already connecting or connected');
      return;
    }

    this.setConnectionState('connecting');

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => this.handleOpen();
      this.ws.onmessage = (event) => this.handleMessage(event);
      this.ws.onerror = (event) => this.handleError(event);
      this.ws.onclose = () => this.handleClose();
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.handleConnectionError(error);
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  public disconnect(): void {
    // Clear heartbeat
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = null;
    }

    // Close WebSocket
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.setConnectionState('disconnected');
  }

  /**
   * Subscribe to messages matching a pattern
   * Examples: 'thread:xyz', 'project:abc', '*' (all messages)
   */
  public subscribe(pattern: string): void {
    if (this.subscriptions.has(pattern)) {
      return; // Already subscribed
    }

    this.subscriptions.add(pattern);

    // Send subscription if connected
    if (this.state === 'connected' && this.ws) {
      this.sendDirect({ action: 'subscribe', pattern });
    }
  }

  /**
   * Unsubscribe from a pattern
   */
  public unsubscribe(pattern: string): void {
    this.subscriptions.delete(pattern);

    // Send unsubscription if connected
    if (this.state === 'connected' && this.ws) {
      this.sendDirect({ action: 'unsubscribe', pattern });
    }
  }

  /**
   * Register a handler for a specific message type
   * Returns unsubscribe function
   */
  public on(eventType: string, handler: MessageHandler): () => void {
    if (!this.messageHandlers.has(eventType)) {
      this.messageHandlers.set(eventType, new Set());
    }

    this.messageHandlers.get(eventType)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(eventType);
      if (handlers) {
        handlers.delete(handler);
      }
    };
  }

  /**
   * Register a handler for connection state changes
   * Returns unsubscribe function
   */
  public onStateChange(handler: StateChangeHandler): () => void {
    this.stateHandlers.add(handler);

    // Return unsubscribe function
    return () => {
      this.stateHandlers.delete(handler);
    };
  }

  /**
   * Send a message (queued if disconnected)
   */
  public send(action: string, data?: any): void {
    const message: OutgoingMessage = { action, data };

    if (this.state === 'connected' && this.ws) {
      this.sendDirect(message);
    } else {
      // Queue for later
      this.messageQueue.push(message);

      // Attempt reconnect if disconnected
      if (this.state === 'disconnected') {
        this.connect();
      }
    }
  }

  /**
   * Send a raw message directly (internal use)
   */
  private sendDirect(message: OutgoingMessage): void {
    if (this.ws && this.state === 'connected') {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        this.messageQueue.push(message);
      }
    }
  }

  /**
   * Get current connection state
   */
  public getState(): ConnectionState {
    return this.state;
  }

  /**
   * Check if connected
   */
  public isConnected(): boolean {
    return this.state === 'connected';
  }

  // ==================== Private Methods ====================

  private setConnectionState(newState: ConnectionState): void {
    if (this.state === newState) return;

    this.state = newState;
    console.log(`[WebSocket] State: ${newState}`);

    // Notify state change handlers
    this.stateHandlers.forEach((handler) => {
      try {
        handler(newState);
      } catch (error) {
        console.error('Error in state change handler:', error);
      }
    });
  }

  private handleOpen = (): void => {
    console.log('[WebSocket] Connected');
    this.reconnectAttempts = 0;
    this.setConnectionState('connected');

    // Resubscribe to all patterns
    this.subscriptions.forEach((pattern) => {
      this.sendDirect({ action: 'subscribe', pattern });
    });

    // Flush message queue
    const queue = this.messageQueue.splice(0);
    queue.forEach((message) => {
      this.sendDirect(message);
    });

    // Start heartbeat
    this.startHeartbeat();
  };

  private handleMessage = (event: MessageEvent): void => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      // Handle pong (heartbeat response)
      if (message.type === 'pong') {
        this.handlePong();
        return;
      }

      // Route to handlers by message type
      const typeHandlers = this.messageHandlers.get(message.type);
      if (typeHandlers) {
        typeHandlers.forEach((handler) => {
          try {
            handler(message);
          } catch (error) {
            console.error(`Error in handler for ${message.type}:`, error);
          }
        });
      }

      // Also route to wildcard handlers
      const wildcardHandlers = this.messageHandlers.get('*');
      if (wildcardHandlers && message.type !== '*') {
        wildcardHandlers.forEach((handler) => {
          try {
            handler(message);
          } catch (error) {
            console.error('Error in wildcard handler:', error);
          }
        });
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error, event.data);
    }
  };

  private handleError = (event: Event): void => {
    console.error('[WebSocket] Error:', event);
    this.handleConnectionError(event);
  };

  private handleClose = (): void => {
    console.log('[WebSocket] Closed');
    this.stopHeartbeat();
    this.ws = null;

    // Attempt reconnect if not intentionally disconnected
    if (this.state !== 'disconnected') {
      this.attemptReconnect();
    }
  };

  private handleConnectionError = (error: any): void => {
    this.stopHeartbeat();
    this.setConnectionState('error');

    // Attempt reconnect
    this.attemptReconnect();
  };

  private attemptReconnect = (): void => {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached');
      this.setConnectionState('error');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.getReconnectDelay();

    console.log(
      `[WebSocket] Attempting reconnect ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${Math.round(delay)}ms`
    );
    this.setConnectionState('reconnecting');

    setTimeout(() => {
      this.connect();
    }, delay);
  };

  private startHeartbeat = (): void => {
    // Send ping every 30 seconds
    this.heartbeatInterval = setInterval(() => {
      if (this.ws && this.state === 'connected') {
        this.sendDirect({ action: 'ping' });
        this.lastHeartbeatTime = Date.now();

        // Set timeout for pong response (5 seconds)
        this.heartbeatTimeout = setTimeout(() => {
          console.warn('[WebSocket] Heartbeat timeout - no pong response');
          if (this.ws) {
            this.ws.close();
          }
        }, 5000);
      }
    }, 30000);
  };

  private stopHeartbeat = (): void => {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = null;
    }
  };

  private handlePong = (): void => {
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = null;
    }
    this.lastHeartbeatTime = Date.now();
  };
}

/**
 * Singleton WebSocket client instance
 */
export const wsClient = new AgentManagerWebSocket(
  `${typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss' : 'ws'}://${
    typeof window !== 'undefined' ? window.location.host : 'localhost:8002'
  }/ws/agent-manager`
);
