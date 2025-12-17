/**
 * React Hook for Agent Manager WebSocket
 *
 * Provides easy integration of WebSocket client in React components.
 * Handles connection lifecycle and cleanup.
 *
 * Date: 2025-12-11
 */

import { useEffect, useRef, useState } from 'react';
import { wsClient, WebSocketMessage, MessageHandler, ConnectionState } from './websocketClient';

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
export function useAgentManagerWS(options: UseAgentManagerWSOptions = {}): UseAgentManagerWSReturn {
  const { autoConnect = true, subscriptions = [] } = options;
  const [state, setState] = useState<ConnectionState>(wsClient.getState());
  const subscriptionsRef = useRef(new Set(subscriptions));

  // Connect/disconnect lifecycle
  useEffect(() => {
    if (autoConnect) {
      wsClient.connect();
    }

    // Listen for state changes
    const unsubscribeState = wsClient.onStateChange((newState) => {
      setState(newState);
    });

    return () => {
      unsubscribeState();
      // Don't disconnect on unmount - connection is shared across components
      // Only disconnect explicitly if needed
    };
  }, [autoConnect]);

  // Handle initial subscriptions
  useEffect(() => {
    subscriptions.forEach((pattern) => {
      if (!subscriptionsRef.current.has(pattern)) {
        wsClient.subscribe(pattern);
        subscriptionsRef.current.add(pattern);
      }
    });
  }, [subscriptions]);

  return {
    isConnected: state === 'connected',
    state,
    subscribe: (pattern: string) => {
      wsClient.subscribe(pattern);
      subscriptionsRef.current.add(pattern);
    },
    unsubscribe: (pattern: string) => {
      wsClient.unsubscribe(pattern);
      subscriptionsRef.current.delete(pattern);
    },
    on: wsClient.on.bind(wsClient),
    send: wsClient.send.bind(wsClient),
    connect: wsClient.connect.bind(wsClient),
    disconnect: wsClient.disconnect.bind(wsClient),
  };
}

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
export function useAgentManagerMessages(eventType: string, maxMessages: number = 100): WebSocketMessage[] {
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);

  useEffect(() => {
    const unsubscribe = wsClient.on(eventType, (message) => {
      setMessages((prev) => [message, ...prev.slice(0, maxMessages - 1)]);
    });

    return unsubscribe;
  }, [eventType, maxMessages]);

  return messages;
}

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
export function useAgentManagerPattern(pattern: string, maxMessages: number = 100): WebSocketMessage[] {
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const { subscribe, on } = useAgentManagerWS({
    autoConnect: true,
    subscriptions: [pattern],
  });

  useEffect(() => {
    const unsubscribe = on('*', (message) => {
      // Check if message matches pattern (simple matching)
      const matches =
        pattern === '*' || // Match all
        (pattern.startsWith('thread:') && message.thread_id === pattern.split(':')[1]) ||
        (pattern.startsWith('project:') && message.project_id === pattern.split(':')[1]) ||
        pattern === message.type; // Match exact type

      if (matches) {
        setMessages((prev) => [message, ...prev.slice(0, maxMessages - 1)]);
      }
    });

    return unsubscribe;
  }, [pattern, maxMessages, on]);

  return messages;
}

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
export function useAgentManagerAction(
  action: string,
  responseEventType?: string
): {
  send: (data?: any) => void;
  isLoading: boolean;
  error: string | null;
  response: WebSocketMessage | null;
} {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<WebSocketMessage | null>(null);
  const { send, on } = useAgentManagerWS({ autoConnect: true });
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!responseEventType) return;

    const unsubscribe = on(responseEventType, (message) => {
      setIsLoading(false);
      setError(null);
      setResponse(message);

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    });

    return unsubscribe;
  }, [responseEventType, on]);

  const sendAction = (data?: any) => {
    setIsLoading(true);
    setError(null);
    setResponse(null);

    send(action, data);

    // Timeout after 30 seconds
    if (responseEventType) {
      timeoutRef.current = setTimeout(() => {
        setIsLoading(false);
        setError('Request timeout');
      }, 30000);
    }
  };

  return {
    send: sendAction,
    isLoading,
    error,
    response,
  };
}
