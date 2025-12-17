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
/**
 * WebSocket Manager for Agent Manager backend communication
 */
export class AgentManagerWebSocket {
    // Backoff calculation
    getReconnectDelay() {
        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), this.maxReconnectDelay);
        // Add jitter (±20%)
        const jitter = delay * 0.2 * (Math.random() * 2 - 1);
        return Math.max(100, delay + jitter);
    }
    constructor(url) {
        Object.defineProperty(this, "ws", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "url", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "reconnectAttempts", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "maxReconnectAttempts", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 10
        });
        Object.defineProperty(this, "reconnectDelay", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 1000
        });
        Object.defineProperty(this, "maxReconnectDelay", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 30000
        });
        Object.defineProperty(this, "state", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 'disconnected'
        });
        // Subscriptions and handlers
        Object.defineProperty(this, "subscriptions", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: new Set()
        });
        Object.defineProperty(this, "messageHandlers", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: new Map()
        });
        Object.defineProperty(this, "stateHandlers", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: new Set()
        });
        // Message queue for offline mode
        Object.defineProperty(this, "messageQueue", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
        // Heartbeat
        Object.defineProperty(this, "heartbeatInterval", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "heartbeatTimeout", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
        Object.defineProperty(this, "lastHeartbeatTime", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: 0
        });
        Object.defineProperty(this, "handleOpen", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
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
            }
        });
        Object.defineProperty(this, "handleMessage", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (event) => {
                try {
                    const message = JSON.parse(event.data);
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
                            }
                            catch (error) {
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
                            }
                            catch (error) {
                                console.error('Error in wildcard handler:', error);
                            }
                        });
                    }
                }
                catch (error) {
                    console.error('Failed to parse WebSocket message:', error, event.data);
                }
            }
        });
        Object.defineProperty(this, "handleError", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (event) => {
                console.error('[WebSocket] Error:', event);
                this.handleConnectionError(event);
            }
        });
        Object.defineProperty(this, "handleClose", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                console.log('[WebSocket] Closed');
                this.stopHeartbeat();
                this.ws = null;
                // Attempt reconnect if not intentionally disconnected
                if (this.state !== 'disconnected') {
                    this.attemptReconnect();
                }
            }
        });
        Object.defineProperty(this, "handleConnectionError", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: (error) => {
                this.stopHeartbeat();
                this.setConnectionState('error');
                // Attempt reconnect
                this.attemptReconnect();
            }
        });
        Object.defineProperty(this, "attemptReconnect", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                    console.error('[WebSocket] Max reconnection attempts reached');
                    this.setConnectionState('error');
                    return;
                }
                this.reconnectAttempts++;
                const delay = this.getReconnectDelay();
                console.log(`[WebSocket] Attempting reconnect ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${Math.round(delay)}ms`);
                this.setConnectionState('reconnecting');
                setTimeout(() => {
                    this.connect();
                }, delay);
            }
        });
        Object.defineProperty(this, "startHeartbeat", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
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
            }
        });
        Object.defineProperty(this, "stopHeartbeat", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                if (this.heartbeatInterval) {
                    clearInterval(this.heartbeatInterval);
                    this.heartbeatInterval = null;
                }
                if (this.heartbeatTimeout) {
                    clearTimeout(this.heartbeatTimeout);
                    this.heartbeatTimeout = null;
                }
            }
        });
        Object.defineProperty(this, "handlePong", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: () => {
                if (this.heartbeatTimeout) {
                    clearTimeout(this.heartbeatTimeout);
                    this.heartbeatTimeout = null;
                }
                this.lastHeartbeatTime = Date.now();
            }
        });
        this.url = url;
    }
    /**
     * Connect to WebSocket server
     */
    connect() {
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
        }
        catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.handleConnectionError(error);
        }
    }
    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
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
    subscribe(pattern) {
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
    unsubscribe(pattern) {
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
    on(eventType, handler) {
        if (!this.messageHandlers.has(eventType)) {
            this.messageHandlers.set(eventType, new Set());
        }
        this.messageHandlers.get(eventType).add(handler);
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
    onStateChange(handler) {
        this.stateHandlers.add(handler);
        // Return unsubscribe function
        return () => {
            this.stateHandlers.delete(handler);
        };
    }
    /**
     * Send a message (queued if disconnected)
     */
    send(action, data) {
        const message = { action, data };
        if (this.state === 'connected' && this.ws) {
            this.sendDirect(message);
        }
        else {
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
    sendDirect(message) {
        if (this.ws && this.state === 'connected') {
            try {
                this.ws.send(JSON.stringify(message));
            }
            catch (error) {
                console.error('Failed to send WebSocket message:', error);
                this.messageQueue.push(message);
            }
        }
    }
    /**
     * Get current connection state
     */
    getState() {
        return this.state;
    }
    /**
     * Check if connected
     */
    isConnected() {
        return this.state === 'connected';
    }
    // ==================== Private Methods ====================
    setConnectionState(newState) {
        if (this.state === newState)
            return;
        this.state = newState;
        console.log(`[WebSocket] State: ${newState}`);
        // Notify state change handlers
        this.stateHandlers.forEach((handler) => {
            try {
                handler(newState);
            }
            catch (error) {
                console.error('Error in state change handler:', error);
            }
        });
    }
}
/**
 * Singleton WebSocket client instance
 */
export const wsClient = new AgentManagerWebSocket(`${typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss' : 'ws'}://${typeof window !== 'undefined' ? window.location.host : 'localhost:8002'}/ws/agent-manager`);
//# sourceMappingURL=websocketClient.js.map