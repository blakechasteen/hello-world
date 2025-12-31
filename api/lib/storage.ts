/**
 * In-memory storage engine for HoloLoom Lite
 * Uses serverless-friendly storage patterns
 */

export interface Memory {
  id: string;
  content: string;
  embedding?: number[];
  metadata: {
    timestamp: number;
    source?: string;
    confidence?: number;
    [key: string]: any;
  };
  relevance?: number;
}

export interface StorageConfig {
  maxMemories: number;
  cacheTTL: number; // seconds
}

/**
 * Hybrid in-memory storage with optional KV backing
 * Designed for Vercel serverless environment
 */
export class MemoryStore {
  private memories: Map<string, Memory> = new Map();
  private config: StorageConfig;
  private kvClient?: any;
  private lastCompaction: number = Date.now();
  private compactionInterval: number = 60000; // 1 minute

  constructor(config: StorageConfig, kvClient?: any) {
    this.config = config;
    this.kvClient = kvClient;
  }

  /**
   * Store a new memory or update existing
   */
  async store(memory: Memory): Promise<{ id: string; cached: boolean }> {
    // Auto-compact if needed
    if (Date.now() - this.lastCompaction > this.compactionInterval) {
      await this.compact();
    }

    // Check capacity
    if (this.memories.size >= this.config.maxMemories) {
      // Remove oldest memory (LRU)
      const oldest = Array.from(this.memories.values()).sort(
        (a, b) => a.metadata.timestamp - b.metadata.timestamp
      )[0];
      if (oldest) {
        this.memories.delete(oldest.id);
      }
    }

    // Store in memory
    this.memories.set(memory.id, {
      ...memory,
      metadata: {
        ...memory.metadata,
        timestamp: memory.metadata.timestamp || Date.now()
      }
    });

    // Try to persist to KV if available
    let cached = false;
    if (this.kvClient) {
      try {
        await this.kvClient.set(
          `memory:${memory.id}`,
          JSON.stringify(memory),
          { ex: this.config.cacheTTL }
        );
        cached = true;
      } catch (err) {
        console.warn('KV storage failed, using memory only:', err);
      }
    }

    return { id: memory.id, cached };
  }

  /**
   * Retrieve memories by query (simple similarity)
   */
  async recall(query: string, limit: number = 10): Promise<Memory[]> {
    const queryLower = query.toLowerCase();
    const matches = Array.from(this.memories.values())
      .map(mem => ({
        ...mem,
        relevance: this.calculateRelevance(queryLower, mem.content)
      }))
      .filter(mem => mem.relevance > 0.1)
      .sort((a, b) => (b.relevance ?? 0) - (a.relevance ?? 0))
      .slice(0, limit);

    return matches;
  }

  /**
   * Simple string similarity scoring
   * Uses word overlap + substring matching
   */
  private calculateRelevance(query: string, content: string): number {
    const contentLower = content.toLowerCase();

    // Exact substring match: highest score
    if (contentLower.includes(query)) return 0.95;

    // Word overlap
    const queryWords = query.split(/\s+/);
    const contentWords = contentLower.split(/\s+/);

    let matches = 0;
    for (const word of queryWords) {
      if (contentWords.some(w => w.includes(word) || word.includes(w))) {
        matches++;
      }
    }

    // Normalized word overlap score
    const wordScore = queryWords.length > 0 ? matches / queryWords.length : 0;

    // Check for semantic similarity keywords
    const semanticBoost = this.getSemanticBoost(query, contentLower);

    return Math.min(1.0, wordScore * 0.7 + semanticBoost * 0.3);
  }

  /**
   * Simple semantic keyword matching
   */
  private getSemanticBoost(query: string, content: string): number {
    const semanticPairs: [string[], string][] = [
      [['explain', 'what', 'is'], 'definition'],
      [['how', 'do', 'does'], 'process'],
      [['why', 'reason', 'cause'], 'reason'],
      [['compare', 'versus', 'vs'], 'comparison'],
      [['example', 'instance', 'like'], 'example']
    ];

    for (const [queryKeywords, contentKeyword] of semanticPairs) {
      const hasQueryKeyword = queryKeywords.some(kw => query.includes(kw));
      const hasContentKeyword = content.includes(contentKeyword);
      if (hasQueryKeyword && hasContentKeyword) {
        return 0.3; // Small boost for semantic alignment
      }
    }

    return 0;
  }

  /**
   * Compact storage: remove expired entries
   */
  private async compact(): Promise<void> {
    const now = Date.now();
    const toDelete: string[] = [];

    for (const [id, mem] of this.memories.entries()) {
      const age = (now - mem.metadata.timestamp) / 1000;
      if (age > this.config.cacheTTL) {
        toDelete.push(id);
      }
    }

    for (const id of toDelete) {
      this.memories.delete(id);
      if (this.kvClient) {
        try {
          await this.kvClient.del(`memory:${id}`);
        } catch (err) {
          console.warn('KV delete failed:', err);
        }
      }
    }

    this.lastCompaction = now;
  }

  /**
   * Get storage statistics
   */
  getStats(): {
    total_memories: number;
    memory_usage_mb: number;
    avg_age_seconds: number;
    oldest_memory_seconds: number;
  } {
    const memories = Array.from(this.memories.values());
    const now = Date.now();

    if (memories.length === 0) {
      return {
        total_memories: 0,
        memory_usage_mb: 0,
        avg_age_seconds: 0,
        oldest_memory_seconds: 0
      };
    }

    // Estimate memory usage
    const memoryUsage = memories.reduce((acc, mem) => {
      return acc + JSON.stringify(mem).length;
    }, 0);

    const ages = memories.map(m => (now - m.metadata.timestamp) / 1000);
    const avgAge = ages.reduce((a, b) => a + b, 0) / ages.length;
    const oldestAge = Math.max(...ages);

    return {
      total_memories: memories.length,
      memory_usage_mb: Math.round(memoryUsage / 1024 / 1024 * 100) / 100,
      avg_age_seconds: Math.round(avgAge),
      oldest_memory_seconds: Math.round(oldestAge)
    };
  }

  /**
   * Clear all memories
   */
  async clear(): Promise<void> {
    this.memories.clear();
    // TODO: Clear KV if available
  }
}

// Global store instance (per serverless function container)
let globalStore: MemoryStore | null = null;

export function getStore(config?: StorageConfig, kvClient?: any): MemoryStore {
  if (!globalStore) {
    const defaultConfig: StorageConfig = {
      maxMemories: parseInt(process.env.HOLOLOOM_MAX_MEMORIES || '1000'),
      cacheTTL: parseInt(process.env.HOLOLOOM_CACHE_TTL || '3600')
    };
    globalStore = new MemoryStore(config || defaultConfig, kvClient);
  }
  return globalStore;
}
