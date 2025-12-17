/**
 * Agent Manager Library - Public API
 *
 * Exports all public types and functions for WebSocket communication
 * and React integration.
 *
 * Date: 2025-12-11
 */
export { AgentManagerWebSocket, wsClient } from './websocketClient';
export type { ConnectionState, WebSocketMessage, OutgoingMessage, MessageHandler, StateChangeHandler, } from './websocketClient';
export { useAgentManagerWS, useAgentManagerMessages, useAgentManagerPattern, useAgentManagerAction } from './useAgentManagerWS';
export type { UseAgentManagerWSOptions, UseAgentManagerWSReturn } from './useAgentManagerWS';
//# sourceMappingURL=index.d.ts.map