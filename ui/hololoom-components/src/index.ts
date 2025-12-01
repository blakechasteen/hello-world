/**
 * HoloLoom Component Library
 *
 * React components for building streaming AI interfaces.
 * Designed for the HoloLoom agentic reasoning system.
 *
 * @packageDocumentation
 */

// ============================================================================
// Types
// ============================================================================

export type {
  // WebSocket Messages
  MessageType,
  WSMessage,
  StreamStartMessage,
  ContextChunkMessage,
  TokenMessage,
  ConfidenceUpdateMessage,
  StageCompleteMessage,
  ReasoningStepMessage,
  StreamEndMessage,
  ErrorMessage,
  HeartbeatMessage,
  AnyWSMessage,

  // Domain Types
  ReasoningMode,
  WeavingStage,
  StageMetrics,
  ContextNode,

  // Component Props
  ConfidenceGaugeProps,
  StreamingResponseProps,
  StageTimelineProps,
  ContextPanelProps,
  ReasoningStepsProps,

  // Data Types
  StageCompletion,
  ContextChunk,
  ReasoningStep,

  // Store Types
  ConnectionState,
  StreamState,
  HoloLoomState,
  HoloLoomActions,
  HoloLoomStore,

  // WebSocket Types
  WebSocketConfig,
  WebSocketHandlers,
} from './types';

// ============================================================================
// Store
// ============================================================================

export {
  useHoloLoomStore,
  // Selectors
  selectConnectionState,
  selectIsConnected,
  selectStreamState,
  selectIsStreaming,
  selectIsComplete,
  selectAccumulatedText,
  selectTokens,
  selectTokenCount,
  selectContextChunks,
  selectContextTokenCount,
  selectConfidence,
  selectEpistemicConfidence,
  selectConfidenceHistory,
  selectStages,
  selectActiveStage,
  selectCompletedStages,
  selectReasoningSteps,
  selectActiveReasoningStep,
  selectMetrics,
  selectError,
  selectHasError,
  selectProgress,
  selectAverageStageDuration,
  selectCurrentStageDisplay,
} from './store';

// ============================================================================
// WebSocket Client
// ============================================================================

export {
  HoloLoomWebSocketClient,
  getWebSocketClient,
  resetWebSocketClient,
  useWebSocket,
} from './WebSocketClient';

// ============================================================================
// Components
// ============================================================================

export {
  ConfidenceGauge,
  ConnectedConfidenceGauge,
} from './components/ConfidenceGauge';

export {
  StreamingResponse,
  ConnectedStreamingResponse,
} from './components/StreamingResponse';

// ============================================================================
// Utilities
// ============================================================================

/**
 * Format duration in milliseconds to human-readable string
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

/**
 * Format token count with K/M suffixes
 */
export function formatTokenCount(count: number): string {
  if (count < 1000) return String(count);
  if (count < 1000000) return `${(count / 1000).toFixed(1)}K`;
  return `${(count / 1000000).toFixed(1)}M`;
}

/**
 * Format confidence as percentage
 */
export function formatConfidence(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * Get stage display name
 */
export function getStageDisplayName(stage: string): string {
  const names: Record<string, string> = {
    loom_command: 'Pattern Selection',
    chrono_trigger: 'Temporal Window',
    yarn_graph: 'Memory Retrieval',
    resonance_shed: 'Feature Extraction',
    dot_plasma: 'Feature Fusion',
    warp_space: 'Tensor Operations',
    convergence: 'Decision Making',
    spacetime: 'Response Weaving',
    reflection: 'Learning',
  };
  return names[stage] || stage;
}

// ============================================================================
// Version
// ============================================================================

export const VERSION = '0.1.0';
