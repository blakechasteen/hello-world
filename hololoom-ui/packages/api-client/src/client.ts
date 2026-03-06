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
  timeout: 30000,
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

  /** Check API health status */
  async health(): Promise<HealthStatus> {
    return this.fetch<HealthStatus>('/health');
  }

  /** Get system statistics */
  async stats(): Promise<SystemStats> {
    return this.fetch<SystemStats>('/stats');
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

  /** Get memory graph structure */
  async getMemoryGraph(options?: {
    limit?: number;
    includeInactive?: boolean;
  }): Promise<MemoryGraph> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.includeInactive) params.set('include_inactive', 'true');

    return this.fetch<MemoryGraph>(`/memory/graph?${params}`);
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
