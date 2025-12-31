/**
 * HoloLoom TypeScript SDK - Type Definitions
 * Mirrors Python Lite API response types
 */

// ============================================================================
// Memory Types
// ============================================================================

export interface Memory {
  id: string;
  content: string;
  timestamp: string;
  embedding: number[] | null;
  metadata: Record<string, any>;
  relevance?: number;
  confidence?: number;
  sourceSystem?: string;
  activation?: number;
  heat?: number;
}

export type RecallStrategy = 'RECENT' | 'SIMILAR' | 'CONNECTED' | 'RESONANT' | 'BALANCED';

export interface RecallOptions {
  strategy?: RecallStrategy;
  limit?: number;
  minRelevance?: number;
  includeMetadata?: boolean;
  contextMemoryIds?: string[];
}

// ============================================================================
// Experience Types (Input)
// ============================================================================

export interface ExperienceInput {
  content: string;
  metadata?: Record<string, any>;
  sourceId?: string;
  timestamp?: string;
}

export interface ExperienceBatchInput {
  items: ExperienceInput[];
}

// ============================================================================
// Lite API Response Types
// ============================================================================

export interface LiteResult<T = any> {
  success: boolean;
  data: T;
  error?: string;
  metadata?: {
    latencyMs: number;
    timestamp: string;
    version: string;
  };
}

export interface ExperienceResult {
  memoryId: string;
  timestamp: string;
  embedded: boolean;
  graphUpdated: boolean;
}

export interface RecallResult {
  memories: Memory[];
  totalCount: number;
  strategy: RecallStrategy;
}

export interface ReflectInput {
  memoryIds?: string[];
  feedback?: {
    helpful?: boolean;
    accurate?: boolean;
    relevant?: boolean;
    quality?: number; // 0.0-1.0
  };
  tags?: string[];
}

export interface ReflectResult {
  processed: number;
  updated: number;
  feedback: Record<string, any>;
}

// ============================================================================
// Reasoning Types
// ============================================================================

export type ReasoningMode = 'DIRECT' | 'VERIFY' | 'RESEARCH' | 'PLAN_EXECUTE';

export interface ReasoningOptions {
  mode?: ReasoningMode;
  maxSteps?: number;
  verificationThreshold?: number;
  enableAudit?: boolean;
}

export interface ReasoningStep {
  index: number;
  type: string;
  query: string;
  result: string;
  confidence: number;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface ReasonResult {
  response: string;
  confidence: number;
  mode: ReasoningMode;
  stepsTaken: ReasoningStep[];
  reasoning?: string;
  verification?: {
    verified: boolean;
    issues: string[];
    corrections?: string[];
  };
}

// ============================================================================
// Query Types (Full Orchestrator)
// ============================================================================

export type ComplexityLevel = 'TRIVIAL' | 'SIMPLE' | 'MODERATE' | 'COMPLEX' | 'RESEARCH';

export interface QueryOptions {
  mode?: ReasoningMode;
  maxSteps?: number;
  complexityLevel?: ComplexityLevel;
  context?: string;
  format?: 'text' | 'json' | 'markdown';
  includeProvenance?: boolean;
}

export interface ToolCall {
  name: string;
  confidence: number;
  parameters?: Record<string, any>;
}

export interface Spacetime {
  response: string;
  confidence: number;
  sources: Memory[];
  toolUsed: string;
  reasoning: string;
  trace?: {
    stages: Array<{
      name: string;
      durationMs: number;
      status: 'success' | 'warning' | 'error';
    }>;
    totalMs: number;
  };
  metadata?: {
    cacheHit?: boolean;
    complexity?: ComplexityLevel;
    threadCount?: number;
  };
}

// ============================================================================
// Error Types
// ============================================================================

export interface HoloLoomError {
  code: string;
  message: string;
  details?: Record<string, any>;
  statusCode?: number;
  timestamp?: string;
}

export interface APIError extends HoloLoomError {
  path?: string;
  method?: string;
}

export class HoloLoomClientError extends Error implements HoloLoomError {
  code: string;
  details?: Record<string, any>;
  statusCode?: number;
  timestamp: string;

  constructor(message: string, code: string, statusCode?: number, details?: Record<string, any>) {
    super(message);
    this.name = 'HoloLoomClientError';
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
    this.timestamp = new Date().toISOString();
  }
}

// ============================================================================
// Configuration Types
// ============================================================================

export interface ClientConfig {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  userAgent?: string;
  headers?: Record<string, string>;
  proxy?: string;
}

export interface RequestOptions {
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
}

// ============================================================================
// Event & Streaming Types (Optional)
// ============================================================================

export type EventType = 'experience' | 'recall' | 'reflect' | 'reason' | 'query' | 'error';

export interface HoloLoomEvent {
  type: EventType;
  timestamp: string;
  data: any;
  metadata?: Record<string, any>;
}

export interface StreamChunk {
  type: 'text' | 'metadata' | 'complete';
  data: string | Record<string, any>;
  timestamp: string;
}

// ============================================================================
// Metrics Types
// ============================================================================

export interface MetricsSnapshot {
  totalQueries: number;
  totalMemories: number;
  avgLatencyMs: number;
  cacheHitRate: number;
  activeConnections: number;
  timestamp: string;
}

export interface PerformanceMetrics {
  latencyMs: number;
  cacheHit: boolean;
  tokensUsed?: number;
  confidence?: number;
}
