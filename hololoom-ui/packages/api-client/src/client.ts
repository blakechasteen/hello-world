/**
 * HoloLoom API Client
 *
 * Main client class for interacting with the HoloLoom backend.
 * Supports all API endpoints, WebSocket streaming, and automatic retries.
 */

import type {
  HoloLoomClientConfig,
  QueryRequest,
  QueryResponse,
  ExperienceRequest,
  ExperienceResponse,
  RecallRequest,
  RecallResponse,
  MemoryGraph,
  HealthStatus,
  SystemStats,
  AuditEntry,
  WorkflowExecuteRequest,
  WorkflowStatus,
  ApiError,
  RateLimitError,
  ProgressEvent,
  WSMessage,
  PromptlyChatRequest,
  PromptlyChatResponse,
  JennySpecDTO,
  StreamUpdateDTO,
} from './types';

// =============================================================================
// ERROR CLASSES
// =============================================================================

export class HoloLoomApiError extends Error {
  code: string;
  details?: Record<string, unknown>;
  traceId?: string;

  constructor(error: ApiError) {
    super(error.message);
    this.name = 'HoloLoomApiError';
    this.code = error.code;
    this.details = error.details;
    this.traceId = error.traceId;
  }
}

export class HoloLoomRateLimitError extends HoloLoomApiError {
  retryAfter: number;
  rateLimit: { limit: number; remaining: number; reset: number };

  constructor(error: RateLimitError) {
    super(error);
    this.name = 'HoloLoomRateLimitError';
    this.retryAfter = error.retryAfter;
    this.rateLimit = error.rateLimit;
  }
}

export class HoloLoomConnectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'HoloLoomConnectionError';
  }
}

// =============================================================================
// DEFAULT CONFIG
// =============================================================================

const DEFAULT_CONFIG: Partial<HoloLoomClientConfig> = {
  timeout: 120000, // 2min — LLM inference can take 10-30s on first load
  enableWebSocket: true,
  retry: {
    maxAttempts: 3,
    baseDelay: 1000,
    maxDelay: 10000,
  },
};

// =============================================================================
// HOLOLOOM CLIENT
// =============================================================================

export class HoloLoomClient {
  private config: Required<HoloLoomClientConfig>;
  private ws: WebSocket | null = null;
  private wsListeners: Map<string, Set<(event: ProgressEvent) => void>> = new Map();
  private wsReconnectAttempts = 0;
  private wsReconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private wsPingInterval: ReturnType<typeof setInterval> | null = null;

  constructor(config: HoloLoomClientConfig) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
      wsUrl: config.wsUrl || config.baseUrl.replace(/^http/, 'ws') + '/ws/progress',
      retry: { ...DEFAULT_CONFIG.retry!, ...config.retry },
    } as Required<HoloLoomClientConfig>;
  }

  // ===========================================================================
  // HTTP HELPERS
  // ===========================================================================

  private async fetch<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.config.headers,
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    let lastError: Error | null = null;
    let attempt = 0;

    while (attempt < this.config.retry.maxAttempts) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

        const response = await fetch(url, {
          ...options,
          headers,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorBody = await response.json().catch(() => ({})) as Record<string, unknown>;

          if (response.status === 429) {
            const retryAfter = parseInt(response.headers.get('Retry-After') || '60', 10);
            throw new HoloLoomRateLimitError({
              code: 'RATE_LIMITED',
              message: 'Rate limit exceeded',
              retryAfter,
              rateLimit: {
                limit: parseInt(response.headers.get('X-RateLimit-Limit') || '0', 10),
                remaining: parseInt(response.headers.get('X-RateLimit-Remaining') || '0', 10),
                reset: parseInt(response.headers.get('X-RateLimit-Reset') || '0', 10),
              },
            });
          }

          throw new HoloLoomApiError({
            code: String(errorBody.code || `HTTP_${response.status}`),
            message: String(errorBody.message || response.statusText),
            details: errorBody.details as Record<string, unknown> | undefined,
            traceId: errorBody.traceId as string | undefined,
          });
        }

        return response.json() as Promise<T>;
      } catch (error) {
        lastError = error as Error;

        if (error instanceof HoloLoomRateLimitError) {
          // Wait for rate limit to reset
          await this.delay(error.retryAfter * 1000);
          attempt++;
          continue;
        }

        if (error instanceof HoloLoomApiError) {
          throw error; // Don't retry API errors
        }

        if (error instanceof Error && error.name === 'AbortError') {
          throw new HoloLoomConnectionError('Request timed out');
        }

        // Retry on network errors
        attempt++;
        if (attempt < this.config.retry.maxAttempts) {
          const delay = Math.min(
            this.config.retry.baseDelay * Math.pow(2, attempt - 1),
            this.config.retry.maxDelay
          );
          await this.delay(delay);
        }
      }
    }

    throw lastError || new HoloLoomConnectionError('Request failed');
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // ===========================================================================
  // HEALTH & STATS
  // ===========================================================================

  /** Check API health status — adapts backend shape to HealthStatus */
  async health(): Promise<HealthStatus> {
    const raw = await this.fetch<Record<string, unknown>>('/health');
    // Backend returns {status: "ok", service, version, timestamp}
    // Client expects {status: "healthy"|"degraded"|"unhealthy", timestamp, components[]}
    const status = raw.status === 'ok' ? 'healthy'
      : raw.status === 'degraded' ? 'degraded'
      : typeof raw.status === 'string' ? (raw.status as HealthStatus['status'])
      : 'unhealthy';
    return {
      status,
      timestamp: String(raw.timestamp ?? new Date().toISOString()),
      components: [],
    };
  }

  /** Get system statistics — adapts flat backend dict to SystemStats */
  async stats(): Promise<SystemStats> {
    const raw = await this.fetch<Record<string, unknown>>('/stats');
    // Backend returns flat: {total_queries, avg_latency_ms, p95_latency_ms, success_rate, ...}
    // Client expects nested: {totalQueries, avgLatencyMs, cacheHitRate, memory: {...}, learning: {...}}
    return {
      totalQueries: Number(raw.total_queries ?? 0),
      queriesLastHour: Number(raw.total_queries ?? 0), // best approximation
      avgLatencyMs: Number(raw.avg_latency_ms ?? 0),
      cacheHitRate: Number(raw.cache_hit_rate ?? 0),
      avgConfidence: Number(raw.avg_confidence ?? Number(raw.success_rate ?? 100) / 100),
      memory: {
        totalNodes: Number(raw.memory_shards ?? 0),
        activeNodes: Number(raw.memory_shards ?? 0),
        totalEdges: 0,
        avgActivation: 0,
        coherence: 0,
      },
      learning: {
        patternsLearned: 0,
        successRate: Number(raw.success_rate ?? 100) / 100,
        lastUpdateTime: new Date().toISOString(),
      },
    };
  }

  // ===========================================================================
  // QUERY
  // ===========================================================================

  /** Execute a query with optional reasoning mode */
  async query(request: QueryRequest): Promise<QueryResponse> {
    return this.fetch<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify({
        text: request.text,
        context: request.context,
        mode: request.mode || 'direct',
        max_steps: request.maxSteps,
        enable_refinement: request.enableRefinement,
      }),
    });
  }

  /** Execute a query and stream progress via WebSocket */
  async queryWithProgress(
    request: QueryRequest,
    onProgress: (event: ProgressEvent) => void
  ): Promise<QueryResponse> {
    // Ensure WebSocket is connected
    await this.connectWebSocket();

    // Submit query and get job ID
    const submitResponse = await this.fetch<{ jobId: string }>('/query/submit', {
      method: 'POST',
      body: JSON.stringify({
        text: request.text,
        context: request.context,
        mode: request.mode || 'direct',
        max_steps: request.maxSteps,
        enable_refinement: request.enableRefinement,
      }),
    });

    // Subscribe to progress updates
    const jobId = submitResponse.jobId;
    this.subscribeToJob(jobId, onProgress);

    // Wait for completion
    return new Promise((resolve, reject) => {
      const checkStatus = async () => {
        try {
          const status = await this.fetch<{ status: string; result?: QueryResponse; error?: string }>(
            `/query/status/${jobId}`
          );

          if (status.status === 'completed' && status.result) {
            this.unsubscribeFromJob(jobId, onProgress);
            resolve(status.result);
          } else if (status.status === 'failed') {
            this.unsubscribeFromJob(jobId, onProgress);
            reject(new HoloLoomApiError({
              code: 'QUERY_FAILED',
              message: status.error || 'Query execution failed',
            }));
          } else {
            // Still running, check again
            setTimeout(checkStatus, 500);
          }
        } catch (error) {
          this.unsubscribeFromJob(jobId, onProgress);
          reject(error);
        }
      };

      checkStatus();
    });
  }

  // ===========================================================================
  // MEMORY
  // ===========================================================================

  /** Store new experience in memory */
  async experience(request: ExperienceRequest): Promise<ExperienceResponse> {
    return this.fetch<ExperienceResponse>('/memory/experience', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /** Recall memories matching query */
  async recall(request: RecallRequest): Promise<RecallResponse> {
    return this.fetch<RecallResponse>('/memory/recall', {
      method: 'POST',
      body: JSON.stringify({
        query: request.query,
        limit: request.limit || 10,
        strategy: request.strategy || 'balanced',
        tags: request.tags,
        include_graph: request.includeGraph,
      }),
    });
  }

  /** Get memory graph structure — tries /api/graph/data, adapts response */
  async getMemoryGraph(options?: {
    limit?: number;
    includeInactive?: boolean;
  }): Promise<MemoryGraph> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.includeInactive) params.set('include_inactive', 'true');

    try {
      return await this.fetch<MemoryGraph>(`/api/graph/data?${params}`);
    } catch {
      // Endpoint may not be available (memory_manager not initialized)
      // Return empty graph — MemoryGraph component will fall back to mock
      throw new HoloLoomConnectionError('Memory graph not available');
    }
  }

  /** Navigate memory in a direction */
  async navigateMemory(
    fromId: string,
    direction: 'forward' | 'backward' | 'sideways' | 'deep',
    steps = 3
  ): Promise<RecallResponse> {
    return this.fetch<RecallResponse>('/memory/navigate', {
      method: 'POST',
      body: JSON.stringify({
        from_id: fromId,
        direction,
        steps,
      }),
    });
  }

  /** Discover patterns in memory */
  async discoverPatterns(options?: {
    patternTypes?: ('loop' | 'cluster' | 'resonance' | 'thread')[];
    minStrength?: number;
  }): Promise<{
    patterns: {
      type: string;
      description: string;
      memories: string[];
      strength: number;
    }[];
  }> {
    return this.fetch('/memory/patterns', {
      method: 'POST',
      body: JSON.stringify({
        pattern_types: options?.patternTypes || ['loop', 'cluster', 'thread'],
        min_strength: options?.minStrength || 0.3,
      }),
    });
  }

  // ===========================================================================
  // AUDIT TRAIL
  // ===========================================================================

  /** Get audit trail entries */
  async getAuditTrail(options?: {
    limit?: number;
    startTime?: string;
    endTime?: string;
    safetyLevel?: string;
  }): Promise<{ entries: AuditEntry[] }> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.startTime) params.set('start_time', options.startTime);
    if (options?.endTime) params.set('end_time', options.endTime);
    if (options?.safetyLevel) params.set('safety_level', options.safetyLevel);

    return this.fetch<{ entries: AuditEntry[] }>(`/audit-trail?${params}`);
  }

  // ===========================================================================
  // WORKFLOW
  // ===========================================================================

  /** Execute a workflow */
  async executeWorkflow(request: WorkflowExecuteRequest): Promise<WorkflowStatus> {
    return this.fetch<WorkflowStatus>('/api/workflow/execute', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /** Get workflow execution status */
  async getWorkflowStatus(executionId: string): Promise<WorkflowStatus> {
    return this.fetch<WorkflowStatus>(`/api/workflow/status/${executionId}`);
  }

  /** Cancel workflow execution */
  async cancelWorkflow(executionId: string): Promise<{ success: boolean }> {
    return this.fetch<{ success: boolean }>(`/api/workflow/cancel/${executionId}`, {
      method: 'POST',
    });
  }

  // ===========================================================================
  // PROMPTLY CHAT
  // ===========================================================================

  /** Send a chat message via /promptly/chat — adapts backend response */
  async promptlyChat(request: PromptlyChatRequest): Promise<PromptlyChatResponse> {
    const raw = await this.fetch<Record<string, unknown>>('/promptly/chat', {
      method: 'POST',
      body: JSON.stringify(request),
    });
    const rawRouting = raw.routing as Record<string, unknown> | undefined;
    const rawRefinement = raw.refinement_info as Record<string, unknown> | undefined;
    const rawMemory = raw.memory_context as Record<string, unknown> | undefined;

    return {
      response: String(raw.response ?? ''),
      refinement_id: raw.refinement_id as string | undefined,
      jenny_id: raw.jenny_id as string | undefined,
      routing: rawRouting ? {
        intent: String(rawRouting.intent ?? ''),
        confidence: Number(rawRouting.intent_confidence ?? rawRouting.confidence ?? 0),
        model: String(rawRouting.model_id ?? rawRouting.model ?? ''),
        fallback: Boolean(rawRouting.fallback),
      } : undefined,
      refinement_info: rawRefinement ? {
        triggered: Boolean(rawRefinement.triggered),
        complexity_score: Number(rawRefinement.complexity_score ?? 0),
        max_passes: Number(rawRefinement.max_passes ?? 0),
      } : undefined,
      memory_context: rawMemory ? {
        // Backend sends hits as array, client expects count
        hits: Array.isArray(rawMemory.hits) ? rawMemory.hits.length : Number(rawMemory.hits ?? 0),
        latency_ms: Number(rawMemory.recall_ms ?? rawMemory.latency_ms ?? 0),
        backend: String(rawMemory.backend ?? ''),
      } : undefined,
    };
  }

  /** Poll for a refinement result */
  async promptlyRefinement(refinementId: string): Promise<PromptlyChatResponse> {
    return this.fetch<PromptlyChatResponse>(`/promptly/chat/refinement/${refinementId}`);
  }

  // ===========================================================================
  // JENNY VISUALIZATION
  // ===========================================================================

  /** Ask Jenny to generate visualization panels */
  async jennyAsk(
    query: string,
    options?: { context?: Record<string, unknown> }
  ): Promise<{ panels: JennySpecDTO[] }> {
    return this.fetch<{ panels: JennySpecDTO[] }>('/jenny/ask', {
      method: 'POST',
      body: JSON.stringify({ query, ...options }),
    });
  }

  /** List active Jenny panels */
  async jennyPanels(lifecycle?: string): Promise<{ panels: JennySpecDTO[] }> {
    const params = new URLSearchParams();
    if (lifecycle) params.set('lifecycle', lifecycle);
    return this.fetch<{ panels: JennySpecDTO[] }>(`/jenny/panels?${params}`);
  }

  /** Get a specific Jenny panel */
  async jennyPanel(panelId: string): Promise<JennySpecDTO> {
    return this.fetch<JennySpecDTO>(`/jenny/panels/${panelId}`);
  }

  /** Execute an action on a Jenny panel */
  async jennyAct(
    panelId: string,
    action: string,
    context?: Record<string, unknown>
  ): Promise<{ success: boolean; result?: unknown }> {
    return this.fetch<{ success: boolean; result?: unknown }>(
      `/jenny/panels/${panelId}/act`,
      {
        method: 'POST',
        body: JSON.stringify({ action, context }),
      }
    );
  }

  /** Get Jenny runtime statistics */
  async jennyStats(): Promise<Record<string, unknown>> {
    return this.fetch<Record<string, unknown>>('/jenny/stats');
  }

  // ===========================================================================
  // WEBSOCKET
  // ===========================================================================

  /** Connect to WebSocket for real-time updates */
  async connectWebSocket(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    if (!this.config.enableWebSocket) {
      return; // WebSocket disabled
    }

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.config.wsUrl);

        this.ws.onopen = () => {
          this.wsReconnectAttempts = 0;

          // Keepalive ping every 30s to prevent idle disconnects
          this.wsPingInterval = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
              this.ws.send(JSON.stringify({ type: 'ping' }));
            }
          }, 30000);

          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data) as WSMessage;
            this.handleWSMessage(message);
          } catch {
            console.error('Failed to parse WebSocket message');
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
          this.handleWSClose();
        };
      } catch (error) {
        reject(new HoloLoomConnectionError('Failed to connect to WebSocket'));
      }
    });
  }

  /** Disconnect WebSocket */
  disconnectWebSocket(): void {
    if (this.wsPingInterval) {
      clearInterval(this.wsPingInterval);
      this.wsPingInterval = null;
    }

    if (this.wsReconnectTimeout) {
      clearTimeout(this.wsReconnectTimeout);
      this.wsReconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.wsListeners.clear();
  }

  /** Subscribe to job progress updates */
  subscribeToJob(jobId: string, callback: (event: ProgressEvent) => void): void {
    const pattern = `job:${jobId}`;

    if (!this.wsListeners.has(pattern)) {
      this.wsListeners.set(pattern, new Set());

      // Send subscription message
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'subscribe', pattern }));
      }
    }

    this.wsListeners.get(pattern)!.add(callback);
  }

  /** Unsubscribe from job progress updates */
  unsubscribeFromJob(jobId: string, callback: (event: ProgressEvent) => void): void {
    const pattern = `job:${jobId}`;
    const listeners = this.wsListeners.get(pattern);

    if (listeners) {
      listeners.delete(callback);

      if (listeners.size === 0) {
        this.wsListeners.delete(pattern);

        // Send unsubscription message
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: 'unsubscribe', pattern }));
        }
      }
    }
  }

  private handleWSMessage(message: WSMessage): void {
    if (message.type === 'progress') {
      const event = message.payload as ProgressEvent;
      const pattern = `job:${event.jobId}`;
      const listeners = this.wsListeners.get(pattern);

      if (listeners) {
        listeners.forEach((callback) => callback(event));
      }
    }
  }

  private handleWSClose(): void {
    // Attempt reconnection with exponential backoff
    if (this.wsReconnectAttempts < 5) {
      const delay = Math.min(1000 * Math.pow(2, this.wsReconnectAttempts), 30000);
      this.wsReconnectAttempts++;

      this.wsReconnectTimeout = setTimeout(() => {
        this.connectWebSocket().catch(() => {
          // Reconnection failed, will retry
        });
      }, delay);
    }
  }

  // ===========================================================================
  // CLEANUP
  // ===========================================================================

  /** Clean up resources */
  destroy(): void {
    this.disconnectWebSocket();
  }
}

// =============================================================================
// FACTORY FUNCTION
// =============================================================================

/** Create a HoloLoom API client */
export function createHoloLoomClient(config: HoloLoomClientConfig): HoloLoomClient {
  return new HoloLoomClient(config);
}
