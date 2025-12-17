/**
 * HoloLoom API Client for VS Code Extension
 *
 * Wraps the HoloLoom API for use in the extension context.
 */

export type ReasoningMode = 'direct' | 'verify' | 'research' | 'plan_execute';

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version?: string;
  uptime_seconds?: number;
}

export interface QueryRequest {
  text: string;
  mode?: ReasoningMode;
  context?: {
    file?: string;
    language?: string;
    selection?: string;
    lineNumber?: number;
  };
  max_steps?: number;
}

export interface Source {
  id: string;
  content: string;
  relevance: number;
  metadata?: Record<string, unknown>;
}

export interface QueryResponse {
  response: string;
  confidence: number;
  reasoning_mode: ReasoningMode;
  sources: Source[];
  steps_taken?: Array<{
    type: string;
    query: string;
    result: string;
  }>;
  verification?: {
    verified: boolean;
    issues?: string[];
  };
  metadata?: Record<string, unknown>;
}

export interface MemoryItem {
  id: string;
  content: string;
  type: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface MemoryStoreRequest {
  content: string;
  type?: string;
  metadata?: Record<string, unknown>;
}

export interface MemorySearchRequest {
  query: string;
  limit?: number;
  filters?: Record<string, unknown>;
}

export interface StatsResponse {
  memory: {
    totalNodes: number;
    totalEdges: number;
    activeNodes: number;
  };
  cacheHitRate: number;
  avgLatencyMs: number;
  queriesProcessed: number;
}

export class HoloLoomClient {
  private baseUrl: string;
  private timeout: number;
  private abortController: AbortController | null = null;

  constructor(baseUrl: string, timeout: number = 30000) {
    this.baseUrl = baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.timeout = timeout;
  }

  private async fetch<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    this.abortController = new AbortController();
    const timeoutId = setTimeout(() => {
      this.abortController?.abort();
    }, this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        signal: this.abortController.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      return response.json();
    } finally {
      clearTimeout(timeoutId);
      this.abortController = null;
    }
  }

  /**
   * Cancel any in-flight requests
   */
  cancelRequest(): void {
    this.abortController?.abort();
  }

  /**
   * Update the base URL
   */
  setBaseUrl(url: string): void {
    this.baseUrl = url.replace(/\/$/, '');
  }

  /**
   * Check server health
   */
  async checkHealth(): Promise<HealthResponse> {
    return this.fetch<HealthResponse>('/health');
  }

  /**
   * Send a query to HoloLoom
   */
  async query(request: QueryRequest): Promise<QueryResponse> {
    return this.fetch<QueryResponse>('/query', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get system statistics
   */
  async getStats(): Promise<StatsResponse> {
    return this.fetch<StatsResponse>('/stats');
  }

  /**
   * Store content in memory
   */
  async storeMemory(request: MemoryStoreRequest): Promise<MemoryItem> {
    return this.fetch<MemoryItem>('/memory/store', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Search memories
   */
  async searchMemory(request: MemorySearchRequest): Promise<MemoryItem[]> {
    return this.fetch<MemoryItem[]>('/memory/search', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get recent memories
   */
  async getRecentMemories(limit: number = 20): Promise<MemoryItem[]> {
    return this.fetch<MemoryItem[]>(`/memory/recent?limit=${limit}`);
  }

  /**
   * Delete a memory by ID
   */
  async deleteMemory(id: string): Promise<void> {
    await this.fetch<void>(`/memory/${id}`, {
      method: 'DELETE',
    });
  }

  /**
   * Test connection to server
   */
  async testConnection(): Promise<boolean> {
    try {
      const health = await this.checkHealth();
      return health.status === 'healthy' || health.status === 'degraded';
    } catch {
      return false;
    }
  }
}
