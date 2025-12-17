/**
 * Agent Manager Library - Public API
 *
 * Exports all public types and functions for WebSocket communication
 * and React integration.
 *
 * Date: 2025-12-11
 */

// WebSocket Client
export { AgentManagerWebSocket, wsClient } from './websocketClient';
export type {
  ConnectionState,
  WebSocketMessage,
  OutgoingMessage,
  MessageHandler,
  StateChangeHandler,
} from './websocketClient';

// React Hooks
export { useAgentManagerWS, useAgentManagerMessages, useAgentManagerPattern, useAgentManagerAction } from './useAgentManagerWS';
export type { UseAgentManagerWSOptions, UseAgentManagerWSReturn } from './useAgentManagerWS';
