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
import {
  HoloLoomClient,
  HoloLoomConnectionError,
  createHoloLoomClient,
} from './client';
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
  DataSourceTier,
  DataSourceResult,
  DataSourceOptions,
  PromptlyChatRequest,
  PromptlyChatResponse,
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

/** Hook for fetching memory graph with graceful degradation */
export function useMemoryGraph(
  options?: { limit?: number; includeInactive?: boolean }
): DataSourceResult<MemoryGraph> {
  const { client } = useHoloLoom();

  return useDataSource(
    () => client.getMemoryGraph(options),
    {
      cacheKey: 'hololoom:memory-graph',
    }
  );
}

// =============================================================================
// STATS HOOKS (rebuilt on DataSource protocol)
// =============================================================================

/** Hook for fetching system stats with graceful degradation */
export function useStats(refreshInterval = 10000): DataSourceResult<SystemStats> {
  const { client } = useHoloLoom();

  return useDataSource(
    () => client.stats(),
    {
      cacheKey: 'hololoom:stats',
      refreshInterval,
    }
  );
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
// DATA SOURCE PROTOCOL
// =============================================================================

// Safe sessionStorage access — works in Node.js (VS Code extension) and SSR
// Uses unknown casts because this package has no DOM lib (runs in Node too)
interface StorageLike { getItem(key: string): string | null; setItem(key: string, value: string): void; }
function getSessionStorage(): StorageLike | null {
  try {
    const g = globalThis as unknown as Record<string, unknown>;
    return g.sessionStorage ? (g.sessionStorage as StorageLike) : null;
  } catch { return null; }
}
const storage = {
  get(key: string): string | null {
    return getSessionStorage()?.getItem(key) ?? null;
  },
  set(key: string, value: string): void {
    try { getSessionStorage()?.setItem(key, value); } catch { /* non-critical */ }
  },
};

/**
 * Universal data-fetching primitive with graceful degradation cascade.
 *
 * Mirrors the backend's HYBRID → INMEMORY fallback pattern:
 *   LIVE → STALE (sessionStorage cache) → MOCK (fallback generator) → EMPTY
 *
 * Each tier is per-fetch — one endpoint being down doesn't degrade the whole UI.
 */
export function useDataSource<T>(
  fetcher: () => Promise<T>,
  options: DataSourceOptions<T> = {}
): DataSourceResult<T> {
  const { cacheKey, mock, refreshInterval } = options;
  const [data, setData] = useState<T | null>(null);
  const [source, setSource] = useState<DataSourceTier>('empty');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const mockRef = useRef(mock);
  mockRef.current = mock;

  const doFetch = useCallback(async (): Promise<T | null> => {
    setIsLoading(true);

    try {
      const result = await fetcherRef.current();
      setData(result);
      setSource('live');
      setError(null);

      // Cache for stale tier
      if (cacheKey) {
        storage.set(cacheKey, JSON.stringify(result));
      }

      return result;
    } catch (err) {
      const fetchError = err instanceof Error ? err : new Error(String(err));
      setError(fetchError);

      // Stale tier: try sessionStorage cache
      if (cacheKey) {
        try {
          const cached = storage.get(cacheKey);
          if (cached) {
            const parsed = JSON.parse(cached) as T;
            setData(parsed);
            setSource('stale');
            setIsLoading(false);
            return parsed;
          }
        } catch {
          // Parse failure — fall through to mock
        }
      }

      // Mock tier: generate fallback data
      if (mockRef.current) {
        const mockData = mockRef.current();
        setData(mockData);
        setSource('mock');
        setIsLoading(false);
        return mockData;
      }

      // Empty tier: nothing available
      setData(null);
      setSource('empty');
      setIsLoading(false);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [cacheKey]);

  // Initial fetch
  useEffect(() => {
    doFetch();
  }, [doFetch]);

  // Auto-refresh
  useEffect(() => {
    if (!refreshInterval || refreshInterval <= 0) return;

    const interval = setInterval(doFetch, refreshInterval);
    return () => clearInterval(interval);
  }, [doFetch, refreshInterval]);

  return { data, source, isLoading, error, refetch: doFetch };
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

// =============================================================================
// PROMPTLY CHAT HOOK
// =============================================================================

interface PromptlyMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    confidence?: number;
    model?: string;
    refinement_id?: string;
    jenny_id?: string;
    routing?: PromptlyChatResponse['routing'];
    refinement_info?: PromptlyChatResponse['refinement_info'];
    memory_context?: PromptlyChatResponse['memory_context'];
    refined?: boolean;
  };
}

interface PromptlyChatState {
  messages: PromptlyMessage[];
  isLoading: boolean;
  error: Error | null;
  roomId: string;
}

/**
 * Hook for conversational chat via /promptly/chat.
 * Manages room_id (session-persisted), conversation history, and refinement polling.
 */
export function usePromptlyChat() {
  const { client } = useHoloLoom();

  // Persistent room ID per browser session
  const [roomId] = useState(() => {
    const stored = storage.get('hololoom:room-id');
    if (stored) return stored;
    const id = typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : `room-${Date.now()}`;
    storage.set('hololoom:room-id', id);
    return id;
  });

  const [state, setState] = useState<PromptlyChatState>({
    messages: [],
    isLoading: false,
    error: null,
    roomId,
  });

  const sendMessage = useCallback(
    async (text: string, mode?: PromptlyChatRequest['mode']) => {
      const userMessage: PromptlyMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: text,
        timestamp: new Date(),
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
        isLoading: true,
        error: null,
      }));

      try {
        const response = await client.promptlyChat({ text, room_id: roomId, mode });

        const assistantMessage: PromptlyMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.response,
          timestamp: new Date(),
          metadata: {
            confidence: response.routing?.confidence,
            model: response.routing?.model,
            refinement_id: response.refinement_id,
            jenny_id: response.jenny_id,
            routing: response.routing,
            refinement_info: response.refinement_info,
            memory_context: response.memory_context,
          },
        };

        setState((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
          isLoading: false,
        }));

        // If refinement was triggered, poll for the refined response
        if (response.refinement_id) {
          pollRefinement(response.refinement_id, assistantMessage.id);
        }

        return response;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        setState((prev) => ({ ...prev, isLoading: false, error }));
        throw error;
      }
    },
    [client, roomId]
  );

  const pollRefinement = useCallback(
    async (refinementId: string, messageId: string) => {
      const maxAttempts = 30; // 30 * 2s = 60s max
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise((r) => setTimeout(r, 2000));

        try {
          const refined = await client.promptlyRefinement(refinementId);
          if (refined.response) {
            // Update the message in-place with refined content
            setState((prev) => ({
              ...prev,
              messages: prev.messages.map((m) =>
                m.id === messageId
                  ? {
                      ...m,
                      content: refined.response,
                      metadata: {
                        ...m.metadata,
                        refined: true,
                        confidence: refined.routing?.confidence ?? m.metadata?.confidence,
                      },
                    }
                  : m
              ),
            }));
            return;
          }
        } catch {
          // Refinement endpoint unavailable — graceful degradation, original response stands
          return;
        }
      }
    },
    [client]
  );

  const clearConversation = useCallback(() => {
    setState((prev) => ({
      ...prev,
      messages: [],
      isLoading: false,
      error: null,
    }));
  }, []);

  return {
    ...state,
    sendMessage,
    clearConversation,
  };
}
