/**
 * HoloLoom TypeScript SDK - Main Client
 * Thin wrapper around HoloLoom API with Lite API methods
 */

import type {
  ClientConfig,
  RequestOptions,
  RecallOptions,
  ExperienceInput,
  ExperienceBatchInput,
  ExperienceResult,
  RecallResult,
  ReflectInput,
  ReflectResult,
  ReasoningOptions,
  ReasonResult,
  QueryOptions,
  Spacetime,
  LiteResult,
} from './types/index.js';

import { HoloLoomClientError } from './types/index.js';

/**
 * HoloLoomClient - Main SDK class
 * Provides Lite API methods: experience, recall, reflect, reason, query
 */
export class HoloLoomClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;
  private retries: number;
  private retryDelay: number;
  private headers: Record<string, string>;
  private userAgent: string;

  constructor(config: ClientConfig = {}) {
    // Validate environment
    this.baseUrl = config.baseUrl || this.detectBaseUrl();
    this.apiKey = config.apiKey || process.env.HOLOLOOM_API_KEY;
    this.timeout = config.timeout || 30000;
    this.retries = config.retries || 3;
    this.retryDelay = config.retryDelay || 100;
    this.userAgent = config.userAgent || `@hololoom/sdk/1.0.0`;

    // Initialize headers
    this.headers = {
      'Content-Type': 'application/json',
      'User-Agent': this.userAgent,
      ...config.headers,
    };

    if (this.apiKey) {
      this.headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
  }

  /**
   * Auto-detect base URL based on environment
   */
  private detectBaseUrl(): string {
    if (typeof window !== 'undefined') {
      // Browser environment
      return window.location.origin === 'chrome-extension://'
        ? 'http://localhost:8000'
        : `${window.location.protocol}//${window.location.host}`;
    }
    // Node.js environment
    return process.env.HOLOLOOM_API_URL || 'http://localhost:8000';
  }

  /**
   * Make HTTP request with retry logic
   */
  private async request<T>(
    method: string,
    path: string,
    body?: any,
    options: RequestOptions = {},
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const timeout = options.timeout || this.timeout;
    const retries = options.retries ?? this.retries;
    const headers = { ...this.headers, ...options.headers };

    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const response = await fetch(url, {
          method,
          headers,
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorData = await this.safeParseJson(response);
          throw new HoloLoomClientError(
            errorData?.message || `HTTP ${response.status}`,
            `HTTP_${response.status}`,
            response.status,
            errorData,
          );
        }

        return await response.json() as T;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // Don't retry on client errors (4xx)
        if (error instanceof HoloLoomClientError && error.statusCode && error.statusCode < 500) {
          throw error;
        }

        // Exponential backoff for retries
        if (attempt < retries) {
          const delay = this.retryDelay * Math.pow(2, attempt);
          await this.sleep(delay);
        }
      }
    }

    throw lastError || new HoloLoomClientError('Request failed', 'REQUEST_FAILED');
  }

  /**
   * Safe JSON parsing
   */
  private async safeParseJson(response: Response): Promise<any> {
    try {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        return await response.json();
      }
      return { message: await response.text() };
    } catch {
      return { message: 'Unknown error' };
    }
  }

  /**
   * Sleep utility for retries
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // ========================================================================
  // Lite API Methods
  // ========================================================================

  /**
   * Experience - Store a new memory
   *
   * @param input - Experience content and optional metadata
   * @returns Stored memory ID and metadata
   *
   * @example
   * const result = await client.experience({
   *   content: "Thompson Sampling balances exploration and exploitation",
   *   metadata: { source: "research", confidence: 0.95 }
   * });
   */
  async experience(input: ExperienceInput, options?: RequestOptions): Promise<LiteResult<ExperienceResult>> {
    return this.request('POST', '/lite/experience', input, options);
  }

  /**
   * Experience (batch) - Store multiple memories
   *
   * @param input - Array of experience items
   * @returns Array of stored memory IDs
   */
  async experienceBatch(input: ExperienceBatchInput, options?: RequestOptions): Promise<LiteResult<ExperienceResult[]>> {
    return this.request('POST', '/lite/experience/batch', input, options);
  }

  /**
   * Recall - Retrieve relevant memories
   *
   * @param query - Search query or memory ID
   * @param recallOptions - Strategy, limit, filters
   * @returns Matching memories with relevance scores
   *
   * @example
   * const memories = await client.recall(
   *   "What did I learn about Thompson Sampling?",
   *   { strategy: 'CONNECTED', limit: 10 }
   * );
   */
  async recall(
    query: string,
    recallOptions?: RecallOptions,
    options?: RequestOptions,
  ): Promise<LiteResult<RecallResult>> {
    const body = {
      query,
      strategy: recallOptions?.strategy || 'BALANCED',
      limit: recallOptions?.limit || 10,
      minRelevance: recallOptions?.minRelevance,
      ...recallOptions,
    };

    return this.request('POST', '/lite/recall', body, options);
  }

  /**
   * Reflect - Learn from feedback
   *
   * @param input - Memory IDs and feedback
   * @returns Update statistics
   *
   * @example
   * const result = await client.reflect({
   *   memoryIds: ["mem_123", "mem_456"],
   *   feedback: { helpful: true, accurate: true, quality: 0.95 },
   *   tags: ["important", "verified"]
   * });
   */
  async reflect(input: ReflectInput, options?: RequestOptions): Promise<LiteResult<ReflectResult>> {
    return this.request('POST', '/lite/reflect', input, options);
  }

  /**
   * Reason - Multi-step reasoning with optional verification
   *
   * @param query - Reasoning query
   * @param reasoningOptions - Mode, max steps, verification threshold
   * @returns Reasoned response with confidence and steps
   *
   * @example
   * const result = await client.reason(
   *   "Compare Thompson Sampling vs UCB",
   *   { mode: 'VERIFY', maxSteps: 5 }
   * );
   */
  async reason(
    query: string,
    reasoningOptions?: ReasoningOptions,
    options?: RequestOptions,
  ): Promise<LiteResult<ReasonResult>> {
    const body = {
      query,
      mode: reasoningOptions?.mode || 'DIRECT',
      maxSteps: reasoningOptions?.maxSteps || 5,
      verificationThreshold: reasoningOptions?.verificationThreshold || 0.7,
      ...reasoningOptions,
    };

    return this.request('POST', '/lite/reason', body, options);
  }

  /**
   * Query - Full orchestrator weaving cycle
   *
   * @param query - User query
   * @param queryOptions - Reasoning mode, complexity, context
   * @returns Full Spacetime result with response, sources, confidence
   *
   * @example
   * const spacetime = await client.query(
   *   "Explain Thompson Sampling",
   *   { mode: 'RESEARCH', complexityLevel: 'COMPLEX' }
   * );
   * console.log(spacetime.data.response);
   * console.log(`Confidence: ${spacetime.data.confidence}`);
   * console.log(`Sources: ${spacetime.data.sources.length}`);
   */
  async query(
    query: string,
    queryOptions?: QueryOptions,
    options?: RequestOptions,
  ): Promise<LiteResult<Spacetime>> {
    const body = {
      query,
      mode: queryOptions?.mode || 'DIRECT',
      maxSteps: queryOptions?.maxSteps || 5,
      complexityLevel: queryOptions?.complexityLevel,
      context: queryOptions?.context,
      format: queryOptions?.format || 'text',
      includeProvenance: queryOptions?.includeProvenance ?? true,
      ...queryOptions,
    };

    return this.request('POST', '/lite/query', body, options);
  }

  // ========================================================================
  // Utility Methods
  // ========================================================================

  /**
   * Check API health and connectivity
   */
  async health(options?: RequestOptions): Promise<LiteResult<{ status: string; version: string }>> {
    return this.request('GET', '/health', undefined, options);
  }

  /**
   * Get current system metrics
   */
  async metrics(options?: RequestOptions): Promise<LiteResult<Record<string, any>>> {
    return this.request('GET', '/metrics', undefined, options);
  }

  /**
   * Configure client settings at runtime
   */
  configure(config: Partial<ClientConfig>): void {
    if (config.baseUrl) this.baseUrl = config.baseUrl;
    if (config.apiKey) {
      this.apiKey = config.apiKey;
      this.headers['Authorization'] = `Bearer ${config.apiKey}`;
    }
    if (config.timeout) this.timeout = config.timeout;
    if (config.retries) this.retries = config.retries;
    if (config.headers) {
      this.headers = { ...this.headers, ...config.headers };
    }
  }

  /**
   * Get current configuration (redacted)
   */
  getConfig(): Omit<ClientConfig, 'apiKey'> {
    return {
      baseUrl: this.baseUrl,
      timeout: this.timeout,
      retries: this.retries,
    };
  }
}

export default HoloLoomClient;
