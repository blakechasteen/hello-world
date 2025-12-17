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
export declare class AgentManagerWebSocket {
    private ws;
    private url;
    private reconnectAttempts;
    private maxReconnectAttempts;
    private reconnectDelay;
    private maxReconnectDelay;
    private state;
    private subscriptions;
    private messageHandlers;
    private stateHandlers;
    private messageQueue;
    private heartbeatInterval;
    private heartbeatTimeout;
    private lastHeartbeatTime;
    private getReconnectDelay;
    constructor(url: string);
    /**
     * Connect to WebSocket server
     */
    connect(): void;
    /**
     * Disconnect from WebSocket server
     */
    disconnect(): void;
    /**
     * Subscribe to messages matching a pattern
     * Examples: 'thread:xyz', 'project:abc', '*' (all messages)
     */
    subscribe(pattern: string): void;
    /**
     * Unsubscribe from a pattern
     */
    unsubscribe(pattern: string): void;
    /**
     * Register a handler for a specific message type
     * Returns unsubscribe function
     */
    on(eventType: string, handler: MessageHandler): () => void;
    /**
     * Register a handler for connection state changes
     * Returns unsubscribe function
     */
    onStateChange(handler: StateChangeHandler): () => void;
    /**
     * Send a message (queued if disconnected)
     */
    send(action: string, data?: any): void;
    /**
     * Send a raw message directly (internal use)
     */
    private sendDirect;
    /**
     * Get current connection state
     */
    getState(): ConnectionState;
    /**
     * Check if connected
     */
    isConnected(): boolean;
    private setConnectionState;
    private handleOpen;
    private handleMessage;
    private handleError;
    private handleClose;
    private handleConnectionError;
    private attemptReconnect;
    private startHeartbeat;
    private stopHeartbeat;
    private handlePong;
}
/**
 * Singleton WebSocket client instance
 */
export declare const wsClient: AgentManagerWebSocket;
//# sourceMappingURL=websocketClient.d.ts.map