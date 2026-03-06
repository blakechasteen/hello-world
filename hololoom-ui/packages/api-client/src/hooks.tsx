/**
 * HoloLoom API Client - React Hooks
 *
 * React hooks for easy integration with React applications.
 * Provides context, query hooks, and real-time subscriptions.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  type ReactNode,
} from 'react';
import { HoloLoomClient, createHoloLoomClient } from './client';
import type {
  HoloLoomClientConfig,
  QueryRequest,
  QueryResponse,
  RecallRequest,
  RecallResponse,
  HealthStatus,
  SystemStats,
  ProgressEvent,
  MemoryGraph,
} from './types';

// =============================================================================
// CONTEXT
// =============================================================================

interface HoloLoomContextValue {
  client: HoloLoomClient;
  isConnected: boolean;
  health: HealthStatus | null;
}

const HoloLoomContext = createContext<HoloLoomContextValue | null>(null);

export interface HoloLoomProviderProps {
  config: HoloLoomClientConfig;
  children: ReactNode;
}

/** Provider component for HoloLoom API client */
export function HoloLoomProvider({ config, children }: HoloLoomProviderProps) {
  const [client] = useState(() => createHoloLoomClient(config));
  const [isConnected, setIsConnected] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    // Initial health check
    client.health()
      .then((status) => {
        setHealth(status);
        setIsConnected(status.status !== 'unhealthy');
      })
      .catch(() => {
        setIsConnected(false);
      });

    // Connect WebSocket
    if (config.enableWebSocket !== false) {
      client.connectWebSocket()
        .then(() => setIsConnected(true))
        .catch(() => setIsConnected(false));
    }

    // Periodic health checks
    const interval = setInterval(() => {
      client.health()
        .then((status) => {
          setHealth(status);
          setIsConnected(status.status !== 'unhealthy');
        })
        .catch(() => {
          setIsConnected(false);
        });
    }, 30000);

    return () => {
      clearInterval(interval);
      client.destroy();
    };
  }, [client, config.enableWebSocket]);

  return (
    <HoloLoomContext.Provider value={{ client, isConnected, health }}>
      {children}
    </HoloLoomContext.Provider>
  );
}

/** Hook to access HoloLoom client */
export function useHoloLoom(): HoloLoomContextValue {
  const context = useContext(HoloLoomContext);
  if (!context) {
    throw new Error('useHoloLoom must be used within a HoloLoomProvider');
  }
  return context;
}

// =============================================================================
// QUERY HOOKS
// =============================================================================

interface QueryState {
  data: QueryResponse | null;
  isLoading: boolean;
  error: Error | null;
  progress: ProgressEvent | null;
}

interface UseQueryOptions {
  /** Enable streaming progress updates */
  streaming?: boolean;
  /** Called on each progress update */
  onProgress?: (event: ProgressEvent) => void;
  /** Called on success */
  onSuccess?: (data: QueryResponse) => void;
  /** Called on error */
  onError?: (error: Error) => void;
}

/** Hook for executing HoloLoom queries */
export function useQuery(options: UseQueryOptions = {}) {
  const { client } = useHoloLoom();
  const [state, setState] = useState<QueryState>({
    data: null,
    isLoading: false,
    error: null,
    progress: null,
  });

  const execute = useCallback(
    async (request: QueryRequest) => {
      setState((prev) => ({
        ...prev,
        isLoading: true,
        error: null,
        progress: null,
      }));

      try {
        let data: QueryResponse;

        if (options.streaming) {
          data = await client.queryWithProgress(request, (event) => {
            setState((prev) => ({ ...prev, progress: event }));
            options.onProgress?.(event);
          });
        } else {
          data = await client.query(request);
        }

        setState({
          data,
          isLoading: false,
          error: null,
          progress: null,
        });
        options.onSuccess?.(data);
        return data;
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: err,
        }));
        options.onError?.(err);
        throw err;
      }
    },
    [client, options]
  );

  const reset = useCallback(() => {
    setState({
      data: null,
      isLoading: false,
      error: null,
      progress: null,
    });
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
}

// =============================================================================
// MEMORY HOOKS
// =============================================================================

interface RecallState {
  data: RecallResponse | null;
  isLoading: boolean;
  error: Error | null;
}

/** Hook for recalling memories */
export function useRecall() {
  const { client } = useHoloLoom();
  const [state, setState] = useState<RecallState>({
    data: null,
    isLoading: false,
    error: null,
  });

  const recall = useCallback(
    async (request: RecallRequest) => {
      setState({ data: null, isLoading: true, error: null });

      try {
        const data = await client.recall(request);
        setState({ data, isLoading: false, error: null });
        return data;
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        setState({ data: null, isLoading: false, error: err });
        throw err;
      }
    },
    [client]
  );

  return { ...state, recall };
}

interface MemoryGraphState {
  data: MemoryGraph | null;
  isLoading: boolean;
  error: Error | null;
}

/** Hook for fetching memory graph */
export function useMemoryGraph(options?: { limit?: number; includeInactive?: boolean }) {
  const { client } = useHoloLoom();
  const [state, setState] = useState<MemoryGraphState>({
    data: null,
    isLoading: true,
    error: null,
  });

  const fetch = useCallback(async () => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    try {
      const data = await client.getMemoryGraph(options);
      setState({ data, isLoading: false, error: null });
      return data;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      setState({ data: null, isLoading: false, error: err });
      throw err;
    }
  }, [client, options]);

  useEffect(() => {
    fetch();
  }, [fetch]);

  return { ...state, refetch: fetch };
}

// =============================================================================
// STATS HOOKS
// =============================================================================

interface StatsState {
  data: SystemStats | null;
  isLoading: boolean;
  error: Error | null;
}

/** Hook for fetching system stats */
export function useStats(refreshInterval = 10000) {
  const { client } = useHoloLoom();
  const [state, setState] = useState<StatsState>({
    data: null,
    isLoading: true,
    error: null,
  });

  const fetch = useCallback(async () => {
    try {
      const data = await client.stats();
      setState({ data, isLoading: false, error: null });
      return data;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      setState((prev) => ({ ...prev, isLoading: false, error: err }));
      throw err;
    }
  }, [client]);

  useEffect(() => {
    fetch();

    const interval = setInterval(fetch, refreshInterval);
    return () => clearInterval(interval);
  }, [fetch, refreshInterval]);

  return { ...state, refetch: fetch };
}

// =============================================================================
// REAL-TIME HOOKS
// =============================================================================

/** Hook for subscribing to real-time progress updates */
export function useProgressSubscription(
  jobId: string | null,
  onProgress: (event: ProgressEvent) => void
) {
  const { client } = useHoloLoom();
  const callbackRef = useRef(onProgress);
  callbackRef.current = onProgress;

  useEffect(() => {
    if (!jobId) return;

    const callback = (event: ProgressEvent) => {
      callbackRef.current(event);
    };

    client.subscribeToJob(jobId, callback);

    return () => {
      client.unsubscribeFromJob(jobId, callback);
    };
  }, [client, jobId]);
}

// =============================================================================
// CONVERSATION HOOK
// =============================================================================

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    confidence?: number;
    toolUsed?: string;
    cached?: boolean;
    latencyMs?: number;
    reasoningMode?: string;
    safetyLevel?: string;
    sourcesCount?: number;
  };
}

interface ConversationState {
  messages: Message[];
  isLoading: boolean;
  error: Error | null;
}

/** Hook for managing a conversation with HoloLoom */
export function useConversation() {
  const { client } = useHoloLoom();
  const [state, setState] = useState<ConversationState>({
    messages: [],
    isLoading: false,
    error: null,
  });

  const sendMessage = useCallback(
    async (content: string, options?: Partial<QueryRequest>) => {
      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: 'user',
        content,
        timestamp: new Date(),
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
        isLoading: true,
        error: null,
      }));

      try {
        const response = await client.query({
          text: content,
          mode: options?.mode || 'direct',
          ...options,
        });

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.response,
          timestamp: new Date(),
          metadata: {
            confidence: response.confidence,
            toolUsed: response.toolUsed,
            cached: response.cached,
            latencyMs: response.latencyMs,
          },
        };

        setState((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
          isLoading: false,
        }));

        return response;
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: err,
        }));
        throw err;
      }
    },
    [client]
  );

  const clearConversation = useCallback(() => {
    setState({
      messages: [],
      isLoading: false,
      error: null,
    });
  }, []);

  return {
    ...state,
    sendMessage,
    clearConversation,
  };
}
