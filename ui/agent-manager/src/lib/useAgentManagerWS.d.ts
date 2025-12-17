/**
 * React Hook for Agent Manager WebSocket
 *
 * Provides easy integration of WebSocket client in React components.
 * Handles connection lifecycle and cleanup.
 *
 * Date: 2025-12-11
 */
import { WebSocketMessage, MessageHandler, ConnectionState } from './websocketClient';
export interface UseAgentManagerWSOptions {
    autoConnect?: boolean;
    subscriptions?: string[];
}
export interface UseAgentManagerWSReturn {
    isConnected: boolean;
    state: ConnectionState;
    subscribe: (pattern: string) => void;
    unsubscribe: (pattern: string) => void;
    on: (eventType: string, handler: MessageHandler) => () => void;
    send: (action: string, data?: any) => void;
    connect: () => void;
    disconnect: () => void;
}
/**
 * React hook for Agent Manager WebSocket
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { isConnected, on, subscribe } = useAgentManagerWS({
 *     autoConnect: true,
 *     subscriptions: ['thread:xyz']
 *   });
 *
 *   useEffect(() => {
 *     const unsubscribe = on('step_progress', (msg) => {
 *       console.log('Step progress:', msg);
 *     });
 *     return unsubscribe;
 *   }, [on]);
 *
 *   return <div>Connected: {isConnected ? 'Yes' : 'No'}</div>;
 * }
 * ```
 */
export declare function useAgentManagerWS(options?: UseAgentManagerWSOptions): UseAgentManagerWSReturn;
/**
 * Hook to listen to specific message types
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const messages = useAgentManagerMessages('step_progress');
 *   return <div>{messages.length} progress messages</div>;
 * }
 * ```
 */
export declare function useAgentManagerMessages(eventType: string, maxMessages?: number): WebSocketMessage[];
/**
 * Hook to subscribe to a pattern and listen for messages
 *
 * @example
 * ```tsx
 * function ThreadPanel({ threadId }) {
 *   const messages = useAgentManagerPattern(`thread:${threadId}`);
 *   return <div>{messages.length} messages for thread</div>;
 * }
 * ```
 */
export declare function useAgentManagerPattern(pattern: string, maxMessages?: number): WebSocketMessage[];
/**
 * Hook to send actions and track responses
 *
 * @example
 * ```tsx
 * function ActionButton() {
 *   const { send, isLoading, error } = useAgentManagerAction('execute_step');
 *
 *   return (
 *     <button onClick={() => send({ thread_id: '123', step: 0 })} disabled={isLoading}>
 *       Execute {isLoading && 'Loading...'} {error && 'Error!'}
 *     </button>
 *   );
 * }
 * ```
 */
export declare function useAgentManagerAction(action: string, responseEventType?: string): {
    send: (data?: any) => void;
    isLoading: boolean;
    error: string | null;
    response: WebSocketMessage | null;
};
//# sourceMappingURL=useAgentManagerWS.d.ts.map