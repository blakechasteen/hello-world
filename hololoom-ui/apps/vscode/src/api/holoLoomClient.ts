/**
 * HoloLoom API Client for VS Code Extension
 *
 * Thin adapter over @hololoom/api-client's HoloLoomClient.
 * Re-exports the shared types so extension commands don't need to import
 * from two packages. The shared client handles retry, rate-limiting, and
 * WebSocket — this wrapper adds VS Code-specific conveniences.
 */

import {
  HoloLoomClient as SharedClient,
  createHoloLoomClient,
  HoloLoomConnectionError,
} from '@hololoom/api-client';
import type {
  QueryRequest,
  QueryResponse,
  SystemStats,
  HealthStatus,
  RecallRequest,
  RecallResponse,
  ExperienceRequest,
  ExperienceResponse,
  PromptlyChatRequest,
  PromptlyChatResponse,
  ReasoningMode,
} from '@hololoom/api-client';

// Re-export types so extension commands import from one place
export type {
  QueryRequest,
  QueryResponse,
  SystemStats,
  HealthStatus,
  RecallRequest,
  RecallResponse,
  ExperienceRequest,
  ExperienceResponse,
  PromptlyChatRequest,
  PromptlyChatResponse,
  ReasoningMode,
};

// Legacy type aliases for existing extension code
export type HealthResponse = HealthStatus;
export type StatsResponse = SystemStats;

export interface Source {
  id: string;
  content: string;
  relevance: number;
  metadata?: Record<string, unknown>;
}

export interface Verification {
  verified: boolean;
  issues?: string[];
}

export interface MemoryItem {
  id: string;
  content: string;
  type: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export type MemoryStoreRequest = ExperienceRequest;
export type MemorySearchRequest = RecallRequest;

export class HoloLoomClient {
  private client: SharedClient;

  constructor(baseUrl: string, timeout: number = 30000) {
    this.client = createHoloLoomClient({
      baseUrl: baseUrl.replace(/\/$/, ''),
      timeout,
      enableWebSocket: false, // VS Code extension host — no WebSocket needed
    });
  }

  /** Update the base URL (recreates the underlying client) */
  setBaseUrl(url: string): void {
    this.client = createHoloLoomClient({
      baseUrl: url.replace(/\/$/, ''),
      enableWebSocket: false,
    });
  }

  /** Check server health */
  async checkHealth(): Promise<HealthStatus> {
    return this.client.health();
  }

  /** Send a query */
  async query(request: QueryRequest): Promise<QueryResponse> {
    return this.client.query(request);
  }

  /** Chat via Promptly endpoint */
  async chat(request: PromptlyChatRequest): Promise<PromptlyChatResponse> {
    return this.client.promptlyChat(request);
  }

  /** Get system stats — returns the shared SystemStats type */
  async getStats(): Promise<SystemStats & { queriesProcessed?: number }> {
    const stats = await this.client.stats();
    return { ...stats, queriesProcessed: stats.totalQueries };
  }

  /** Store content in memory — accepts legacy {type} field as {contentType} */
  async storeMemory(request: MemoryStoreRequest & { type?: string }): Promise<ExperienceResponse> {
    const { type, ...rest } = request;
    return this.client.experience({ ...rest, contentType: type as ExperienceRequest['contentType'] });
  }

  /** Search memories — returns MemoryItem[] for backward compatibility */
  async searchMemory(request: RecallRequest): Promise<MemoryItem[]> {
    const response = await this.client.recall(request);
    return response.memories.map((m) => ({
      id: m.id,
      content: m.text,
      type: m.state,
      timestamp: m.timestamp ?? new Date().toISOString(),
      metadata: { relevance: m.relevance },
    }));
  }

  /** Get recent memories */
  async getRecentMemories(limit: number = 20): Promise<MemoryItem[]> {
    return this.searchMemory({ query: '', limit });
  }

  /** Delete a memory by ID — no-op until backend supports DELETE */
  async deleteMemory(_id: string): Promise<void> {
    // Not yet supported by the shared client
  }

  /** Test connection to server */
  async testConnection(): Promise<boolean> {
    try {
      const health = await this.client.health();
      return health.status !== 'unhealthy';
    } catch {
      return false;
    }
  }

  /** Cancel not needed — shared client uses AbortController internally */
  cancelRequest(): void {
    // No-op: the shared client manages its own abort controllers per-request
  }

  /** Clean up resources */
  destroy(): void {
    this.client.destroy();
  }
}
