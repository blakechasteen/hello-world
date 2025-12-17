/**
 * HoloLoom API Client - Type Definitions
 *
 * Complete TypeScript types for all HoloLoom API endpoints.
 */

// =============================================================================
// CORE TYPES
// =============================================================================

/** Reasoning modes for agentic queries */
export type ReasoningMode = 'direct' | 'verify' | 'research' | 'plan_execute';

/** Safety levels for alignment monitoring */
export type SafetyLevel = 'safe' | 'caution' | 'warning' | 'danger' | 'critical';

/** Thread/weaving status */
export type ThreadStatus = 'pending' | 'weaving' | 'complete' | 'error' | 'cached';

/** Memory node states */
export type MemoryState = 'active' | 'recent' | 'dormant' | 'hot';

/** Confidence levels */
export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'unknown';

// =============================================================================
// QUERY TYPES
// =============================================================================

/** Context for code-related queries */
export interface CodeContext {
  /** Programming language ID */
  languageId?: string;
  /** File name */
  fileName?: string;
  /** Selected code snippet */
  selection?: string;
  /** Full file content */
  fileContent?: string;
  /** Cursor position */
  cursorPosition?: { line: number; column: number };
}

/** Query request payload */
export interface QueryRequest {
  /** Query text */
  text: string;
  /** Optional code context */
  context?: CodeContext;
  /** Reasoning mode */
  mode?: ReasoningMode;
  /** Maximum reasoning steps (for research/plan_execute) */
  maxSteps?: number;
  /** Enable refinement for low-confidence results */
  enableRefinement?: boolean;
  /** Custom timeout in milliseconds */
  timeout?: number;
}

/** Verification result for 'verify' mode */
export interface VerificationResult {
  /** Was the answer verified */
  verified: boolean;
  /** Verification confidence */
  confidence: number;
  /** Verification details */
  details?: string;
  /** Contradictions found */
  contradictions?: string[];
}

/** Reasoning step in multi-query modes */
export interface ReasoningStep {
  /** Step index */
  index: number;
  /** Step type */
  type: 'query' | 'verification' | 'synthesis' | 'planning' | 'execution';
  /** Step query/action */
  query: string;
  /** Step result */
  result?: string;
  /** Step confidence */
  confidence?: number;
  /** Epistemic confidence (confidence in our confidence) */
  epistemicConfidence?: number;
  /** Duration in ms */
  durationMs?: number;
}

/** Query response */
export interface QueryResponse {
  /** Generated response text */
  response: string;
  /** Overall confidence (0-1) */
  confidence: number;
  /** Epistemic confidence */
  epistemicConfidence?: number;
  /** Tool used for generation */
  toolUsed?: string;
  /** Reasoning mode used */
  mode: ReasoningMode;
  /** Reasoning steps (for multi-query modes) */
  steps?: ReasoningStep[];
  /** Verification result (for 'verify' mode) */
  verification?: VerificationResult;
  /** Whether result came from cache */
  cached: boolean;
  /** Processing latency in ms */
  latencyMs: number;
  /** Token count */
  tokens?: number;
  /** Safety assessment */
  safety?: SafetyAssessment;
  /** Source memories used */
  sources?: MemorySource[];
  /** Trace ID for debugging */
  traceId?: string;
}

// =============================================================================
// MEMORY TYPES
// =============================================================================

/** Memory source reference */
export interface MemorySource {
  /** Memory ID */
  id: string;
  /** Source text snippet */
  text: string;
  /** Relevance score */
  relevance: number;
  /** Memory state */
  state: MemoryState;
  /** Activation level */
  activation?: number;
  /** Heat score */
  heat?: number;
  /** Timestamp */
  timestamp?: string;
}

/** Memory node in knowledge graph */
export interface MemoryNode {
  /** Node ID */
  id: string;
  /** Node content */
  content: string;
  /** Node type (entity, fact, episode, etc.) */
  type: string;
  /** Node state */
  state: MemoryState;
  /** Activation level (0-1) */
  activation: number;
  /** Heat score (0-1) */
  heat: number;
  /** Connected nodes */
  connections?: string[];
  /** Node metadata */
  metadata?: Record<string, unknown>;
  /** Creation timestamp */
  createdAt: string;
  /** Last accessed timestamp */
  accessedAt: string;
}

/** Memory edge in knowledge graph */
export interface MemoryEdge {
  /** Source node ID */
  source: string;
  /** Target node ID */
  target: string;
  /** Edge type */
  type: 'IS_A' | 'USES' | 'MENTIONS' | 'LEADS_TO' | 'PART_OF' | 'IN_TIME' | 'OCCURRED_AT';
  /** Edge weight */
  weight: number;
}

/** Memory graph structure */
export interface MemoryGraph {
  /** Nodes in the graph */
  nodes: MemoryNode[];
  /** Edges in the graph */
  edges: MemoryEdge[];
  /** Graph statistics */
  stats: {
    totalNodes: number;
    totalEdges: number;
    activeNodes: number;
    avgActivation: number;
    coherence: number;
  };
}

/** Experience (store memory) request */
export interface ExperienceRequest {
  /** Content to store */
  content: string;
  /** Content type hint */
  contentType?: 'text' | 'code' | 'document' | 'conversation';
  /** Custom metadata */
  metadata?: Record<string, unknown>;
  /** Tags for organization */
  tags?: string[];
}

/** Experience response */
export interface ExperienceResponse {
  /** Created memory ID */
  id: string;
  /** Processing result */
  success: boolean;
  /** Extracted entities */
  entities?: string[];
  /** Extracted motifs */
  motifs?: string[];
}

/** Recall (search memory) request */
export interface RecallRequest {
  /** Search query */
  query: string;
  /** Maximum results */
  limit?: number;
  /** Recall strategy */
  strategy?: 'recent' | 'similar' | 'connected' | 'resonant' | 'balanced';
  /** Filter by tags */
  tags?: string[];
  /** Include graph context */
  includeGraph?: boolean;
}

/** Recall response */
export interface RecallResponse {
  /** Retrieved memories */
  memories: MemorySource[];
  /** Graph subgraph (if requested) */
  graph?: MemoryGraph;
  /** Search latency in ms */
  latencyMs: number;
}

// =============================================================================
// SAFETY / ALIGNMENT TYPES
// =============================================================================

/** Safety assessment */
export interface SafetyAssessment {
  /** Overall safety level */
  level: SafetyLevel;
  /** Safety score (0-1, higher = safer) */
  score: number;
  /** Risk factors detected */
  riskFactors?: string[];
  /** Was human approval required */
  humanApprovalRequired?: boolean;
  /** Audit trail ID */
  auditId?: string;
}

/** Audit trail entry */
export interface AuditEntry {
  /** Entry ID */
  id: string;
  /** Timestamp */
  timestamp: string;
  /** Action taken */
  action: string;
  /** Query that triggered action */
  query: string;
  /** Safety assessment */
  safety: SafetyAssessment;
  /** Outcome */
  outcome: 'allowed' | 'blocked' | 'escalated';
  /** Decision reasoning */
  reasoning?: string;
}

// =============================================================================
// WORKFLOW TYPES
// =============================================================================

/** Workflow node definition */
export interface WorkflowNode {
  /** Node ID */
  id: string;
  /** Node type */
  type: string;
  /** Node position */
  position: { x: number; y: number };
  /** Node configuration */
  config: Record<string, unknown>;
}

/** Workflow connection */
export interface WorkflowConnection {
  /** Source node ID */
  source: string;
  /** Target node ID */
  target: string;
  /** Source port */
  sourcePort?: string;
  /** Target port */
  targetPort?: string;
}

/** Workflow definition */
export interface WorkflowDefinition {
  /** Workflow version */
  version: string;
  /** Workflow name */
  name: string;
  /** Workflow description */
  description?: string;
  /** Nodes in workflow */
  nodes: WorkflowNode[];
  /** Connections between nodes */
  connections: WorkflowConnection[];
}

/** Workflow execution request */
export interface WorkflowExecuteRequest {
  /** Workflow to execute */
  workflow: WorkflowDefinition;
  /** Input data */
  inputData: Record<string, unknown>;
}

/** Workflow execution status */
export interface WorkflowStatus {
  /** Execution ID */
  executionId: string;
  /** Current status */
  status: 'pending' | 'running' | 'completed' | 'failed';
  /** Current step */
  currentStep?: string;
  /** Progress (0-100) */
  progress: number;
  /** Step results */
  stepResults?: Record<string, unknown>;
  /** Error if failed */
  error?: string;
}

// =============================================================================
// STATS / METRICS TYPES
// =============================================================================

/** System health status */
export interface HealthStatus {
  /** Overall status */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Timestamp */
  timestamp: string;
  /** Component statuses */
  components: {
    name: string;
    status: 'healthy' | 'degraded' | 'unhealthy';
    latencyMs?: number;
    message?: string;
  }[];
}

/** System statistics */
export interface SystemStats {
  /** Total queries processed */
  totalQueries: number;
  /** Queries in last hour */
  queriesLastHour: number;
  /** Average latency in ms */
  avgLatencyMs: number;
  /** Cache hit rate (0-1) */
  cacheHitRate: number;
  /** Average confidence */
  avgConfidence: number;
  /** Memory stats */
  memory: {
    totalNodes: number;
    activeNodes: number;
    totalEdges: number;
    avgActivation: number;
    coherence: number;
  };
  /** Learning stats */
  learning: {
    patternsLearned: number;
    successRate: number;
    lastUpdateTime: string;
  };
}

// =============================================================================
// WEBSOCKET TYPES
// =============================================================================

/** WebSocket message types */
export type WSMessageType =
  | 'subscribe'
  | 'unsubscribe'
  | 'progress'
  | 'complete'
  | 'error'
  | 'heartbeat';

/** WebSocket message */
export interface WSMessage {
  /** Message type */
  type: WSMessageType;
  /** Message payload */
  payload: unknown;
  /** Timestamp */
  timestamp: string;
}

/** Progress event for streaming updates */
export interface ProgressEvent {
  /** Job/Query ID */
  jobId: string;
  /** Current stage (1-9 for weaving cycle) */
  stage: number;
  /** Stage name */
  stageName: string;
  /** Progress within stage (0-100) */
  progress: number;
  /** Stage status */
  status: 'pending' | 'running' | 'complete' | 'error';
  /** Stage result (if complete) */
  result?: unknown;
  /** Total elapsed time in ms */
  elapsedMs: number;
}

// =============================================================================
// CLIENT CONFIG
// =============================================================================

/** API client configuration */
export interface HoloLoomClientConfig {
  /** Base URL of the HoloLoom API */
  baseUrl: string;
  /** API key for authentication (if required) */
  apiKey?: string;
  /** Default timeout in milliseconds */
  timeout?: number;
  /** Enable WebSocket for real-time updates */
  enableWebSocket?: boolean;
  /** WebSocket URL (defaults to ws://baseUrl) */
  wsUrl?: string;
  /** Custom headers */
  headers?: Record<string, string>;
  /** Retry configuration */
  retry?: {
    /** Maximum retry attempts */
    maxAttempts: number;
    /** Base delay between retries in ms */
    baseDelay: number;
    /** Maximum delay between retries in ms */
    maxDelay: number;
  };
}

// =============================================================================
// ERROR TYPES
// =============================================================================

/** API error response */
export interface ApiError {
  /** Error code */
  code: string;
  /** Error message */
  message: string;
  /** Additional details */
  details?: Record<string, unknown>;
  /** Trace ID for debugging */
  traceId?: string;
}

/** Rate limit error */
export interface RateLimitError extends ApiError {
  /** Retry after (seconds) */
  retryAfter: number;
  /** Rate limit info */
  rateLimit: {
    limit: number;
    remaining: number;
    reset: number;
  };
}
