/**
 * HoloLoom Component Library Type Definitions
 *
 * Core types for WebSocket streaming, state management, and components.
 * Aligned with the 9-stage HoloLoom weaving cycle.
 */

// ============================================================================
// WebSocket Message Types
// ============================================================================

export type MessageType =
  | 'stream_start'
  | 'context_chunk'
  | 'token'
  | 'confidence_update'
  | 'stage_complete'
  | 'reasoning_step'
  | 'stream_end'
  | 'error'
  | 'heartbeat';

/**
 * Base WebSocket message structure
 */
export interface WSMessage {
  type: MessageType;
  timestamp: string;
  session_id: string;
  sequence: number;
}

/**
 * Stream initialization message
 */
export interface StreamStartMessage extends WSMessage {
  type: 'stream_start';
  query: string;
  mode: ReasoningMode;
  expected_stages: string[];
}

/**
 * Context chunk from memory retrieval
 */
export interface ContextChunkMessage extends WSMessage {
  type: 'context_chunk';
  chunk_index: number;
  nodes: ContextNode[];
  hop_distance: number;
  relevance_scores: Record<string, number>;
  token_count: number;
  cumulative_tokens: number;
  is_final: boolean;
}

/**
 * Single token from LLM generation
 */
export interface TokenMessage extends WSMessage {
  type: 'token';
  token: string;
  cumulative_text: string;
  token_index: number;
  is_final: boolean;
  metadata?: {
    context_tokens: number;
    generation_tokens: number;
  };
}

/**
 * Confidence update during processing
 */
export interface ConfidenceUpdateMessage extends WSMessage {
  type: 'confidence_update';
  confidence: number;
  epistemic_confidence: number;
  source: string;
  stage: string;
}

/**
 * Stage completion notification (9-stage cycle)
 */
export interface StageCompleteMessage extends WSMessage {
  type: 'stage_complete';
  stage: WeavingStage;
  duration_ms: number;
  metrics: StageMetrics;
}

/**
 * Reasoning step for agentic modes
 */
export interface ReasoningStepMessage extends WSMessage {
  type: 'reasoning_step';
  step_index: number;
  step_type: 'query' | 'verify' | 'plan' | 'execute' | 'synthesize';
  description: string;
  confidence: number;
  substeps?: string[];
}

/**
 * Stream completion
 */
export interface StreamEndMessage extends WSMessage {
  type: 'stream_end';
  total_duration_ms: number;
  final_confidence: number;
  tokens_generated: number;
  context_tokens_used: number;
  cache_hit: boolean;
  reasoning_steps?: number;
}

/**
 * Error message
 */
export interface ErrorMessage extends WSMessage {
  type: 'error';
  error_code: string;
  message: string;
  recoverable: boolean;
  retry_after_ms?: number;
}

/**
 * Heartbeat for connection health
 */
export interface HeartbeatMessage extends WSMessage {
  type: 'heartbeat';
  server_time: string;
  latency_ms?: number;
}

// ============================================================================
// Domain Types
// ============================================================================

/**
 * HoloLoom reasoning modes
 */
export type ReasoningMode = 'direct' | 'verify' | 'research' | 'plan_execute';

/**
 * 9-stage weaving cycle stages
 */
export type WeavingStage =
  | 'loom_command'
  | 'chrono_trigger'
  | 'yarn_graph'
  | 'resonance_shed'
  | 'dot_plasma'
  | 'warp_space'
  | 'convergence'
  | 'spacetime'
  | 'reflection';

/**
 * Stage metrics from weaving cycle
 */
export interface StageMetrics {
  duration_ms: number;
  memory_used?: number;
  items_processed?: number;
  cache_hit?: boolean;
}

/**
 * Context node from memory retrieval
 */
export interface ContextNode {
  id: string;
  content: string;
  relevance: number;
  source: string;
  hop_distance: number;
  edge_type?: string;
}

// ============================================================================
// Component Props Types
// ============================================================================

/**
 * Confidence gauge component props
 */
export interface ConfidenceGaugeProps {
  /** Current confidence value 0-1 */
  confidence: number;
  /** Epistemic confidence (meta-uncertainty) 0-1 */
  epistemicConfidence?: number;
  /** Historical confidence values for sparkline */
  history?: number[];
  /** Component size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Show numerical value */
  showValue?: boolean;
  /** Show sparkline trend */
  showTrend?: boolean;
  /** Custom label */
  label?: string;
  /** Animate value changes */
  animated?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Streaming response component props
 */
export interface StreamingResponseProps {
  /** Accumulated response text */
  text: string;
  /** Whether stream is active */
  isStreaming: boolean;
  /** Render as markdown */
  markdown?: boolean;
  /** Show cursor while streaming */
  showCursor?: boolean;
  /** Cursor character */
  cursorChar?: string;
  /** Token-by-token animation */
  tokenAnimation?: boolean;
  /** Callback when token rendered */
  onTokenRender?: (token: string, index: number) => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Stage timeline component props
 */
export interface StageTimelineProps {
  /** Completed stages with metrics */
  stages: StageCompletion[];
  /** Currently active stage */
  activeStage?: WeavingStage;
  /** Show timing information */
  showTiming?: boolean;
  /** Compact mode */
  compact?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Stage completion data
 */
export interface StageCompletion {
  stage: WeavingStage;
  status: 'pending' | 'active' | 'complete' | 'error';
  duration_ms?: number;
  metrics?: StageMetrics;
}

/**
 * Context panel component props
 */
export interface ContextPanelProps {
  /** Retrieved context chunks */
  chunks: ContextChunk[];
  /** Currently highlighted chunk */
  highlightedChunk?: number;
  /** Show relevance scores */
  showRelevance?: boolean;
  /** Collapsible mode */
  collapsible?: boolean;
  /** Max visible chunks before scroll */
  maxVisible?: number;
  /** Callback on chunk click */
  onChunkClick?: (chunk: ContextChunk) => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Context chunk display data
 */
export interface ContextChunk {
  id: string;
  content: string;
  relevance: number;
  hopDistance: number;
  tokenCount: number;
  source: string;
}

/**
 * Reasoning steps component props
 */
export interface ReasoningStepsProps {
  /** Completed reasoning steps */
  steps: ReasoningStep[];
  /** Currently active step */
  activeStep?: number;
  /** Show sub-steps */
  showSubsteps?: boolean;
  /** Expandable steps */
  expandable?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Single reasoning step data
 */
export interface ReasoningStep {
  index: number;
  type: 'query' | 'verify' | 'plan' | 'execute' | 'synthesize';
  description: string;
  confidence: number;
  substeps?: string[];
  status: 'pending' | 'active' | 'complete' | 'error';
}

// ============================================================================
// Store Types
// ============================================================================

/**
 * Connection state for WebSocket
 */
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

/**
 * Stream state
 */
export type StreamState = 'idle' | 'streaming' | 'complete' | 'error';

/**
 * HoloLoom streaming store state
 */
export interface HoloLoomState {
  // Connection
  connectionState: ConnectionState;
  lastHeartbeat: string | null;
  reconnectAttempts: number;

  // Session
  sessionId: string | null;
  query: string | null;
  mode: ReasoningMode | null;

  // Streaming
  streamState: StreamState;
  tokens: string[];
  accumulatedText: string;

  // Context
  contextChunks: ContextChunk[];
  totalContextTokens: number;

  // Confidence
  confidence: number;
  epistemicConfidence: number;
  confidenceHistory: number[];

  // Stages
  stages: StageCompletion[];
  activeStage: WeavingStage | null;

  // Reasoning
  reasoningSteps: ReasoningStep[];
  activeReasoningStep: number | null;

  // Metrics
  startTime: number | null;
  endTime: number | null;
  totalDuration: number | null;
  tokensGenerated: number;
  cacheHit: boolean;

  // Errors
  error: { code: string; message: string } | null;
}

/**
 * Store actions
 */
export interface HoloLoomActions {
  // Connection
  setConnectionState: (state: ConnectionState) => void;
  updateHeartbeat: (timestamp: string) => void;
  incrementReconnectAttempts: () => void;
  resetReconnectAttempts: () => void;

  // Session
  startSession: (sessionId: string, query: string, mode: ReasoningMode) => void;
  endSession: () => void;

  // Streaming
  setStreamState: (state: StreamState) => void;
  appendToken: (token: string) => void;
  setAccumulatedText: (text: string) => void;

  // Context
  addContextChunk: (chunk: ContextChunk) => void;
  clearContext: () => void;

  // Confidence
  updateConfidence: (confidence: number, epistemic?: number) => void;

  // Stages
  updateStage: (stage: WeavingStage, status: 'active' | 'complete' | 'error', metrics?: StageMetrics) => void;

  // Reasoning
  addReasoningStep: (step: ReasoningStep) => void;
  updateReasoningStep: (index: number, updates: Partial<ReasoningStep>) => void;

  // Metrics
  setMetrics: (metrics: Partial<Pick<HoloLoomState, 'totalDuration' | 'tokensGenerated' | 'cacheHit'>>) => void;

  // Errors
  setError: (error: { code: string; message: string } | null) => void;

  // Reset
  reset: () => void;
}

/**
 * Complete store type
 */
export type HoloLoomStore = HoloLoomState & HoloLoomActions;

// ============================================================================
// WebSocket Client Types
// ============================================================================

/**
 * WebSocket client configuration
 */
export interface WebSocketConfig {
  /** WebSocket URL */
  url: string;
  /** Auto-reconnect on disconnect */
  autoReconnect?: boolean;
  /** Max reconnect attempts */
  maxReconnectAttempts?: number;
  /** Base reconnect delay (ms) */
  reconnectDelay?: number;
  /** Heartbeat interval (ms) */
  heartbeatInterval?: number;
  /** Connection timeout (ms) */
  connectionTimeout?: number;
}

/**
 * WebSocket event handlers
 */
export interface WebSocketHandlers {
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (error: Event) => void;
  onMessage?: (message: WSMessage) => void;
  onReconnect?: (attempt: number) => void;
  onHeartbeat?: (latency: number) => void;
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Extract message type from discriminated union
 */
export type MessageOfType<T extends MessageType> =
  T extends 'stream_start' ? StreamStartMessage :
  T extends 'context_chunk' ? ContextChunkMessage :
  T extends 'token' ? TokenMessage :
  T extends 'confidence_update' ? ConfidenceUpdateMessage :
  T extends 'stage_complete' ? StageCompleteMessage :
  T extends 'reasoning_step' ? ReasoningStepMessage :
  T extends 'stream_end' ? StreamEndMessage :
  T extends 'error' ? ErrorMessage :
  T extends 'heartbeat' ? HeartbeatMessage :
  never;

/**
 * All possible WebSocket messages
 */
export type AnyWSMessage =
  | StreamStartMessage
  | ContextChunkMessage
  | TokenMessage
  | ConfidenceUpdateMessage
  | StageCompleteMessage
  | ReasoningStepMessage
  | StreamEndMessage
  | ErrorMessage
  | HeartbeatMessage;
