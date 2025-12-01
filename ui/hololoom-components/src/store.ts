/**
 * HoloLoom Zustand Store
 *
 * Centralized state management for streaming AI interface.
 * Uses immer for immutable updates and devtools for debugging.
 */

import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import type {
  HoloLoomState,
  HoloLoomActions,
  HoloLoomStore,
  ConnectionState,
  StreamState,
  ReasoningMode,
  WeavingStage,
  StageMetrics,
  ContextChunk,
  ReasoningStep,
  StageCompletion,
} from './types';

// ============================================================================
// Initial State
// ============================================================================

const WEAVING_STAGES: WeavingStage[] = [
  'loom_command',
  'chrono_trigger',
  'yarn_graph',
  'resonance_shed',
  'dot_plasma',
  'warp_space',
  'convergence',
  'spacetime',
  'reflection',
];

const initialStages: StageCompletion[] = WEAVING_STAGES.map(stage => ({
  stage,
  status: 'pending' as const,
}));

const initialState: HoloLoomState = {
  // Connection
  connectionState: 'disconnected',
  lastHeartbeat: null,
  reconnectAttempts: 0,

  // Session
  sessionId: null,
  query: null,
  mode: null,

  // Streaming
  streamState: 'idle',
  tokens: [],
  accumulatedText: '',

  // Context
  contextChunks: [],
  totalContextTokens: 0,

  // Confidence
  confidence: 0,
  epistemicConfidence: 0,
  confidenceHistory: [],

  // Stages
  stages: initialStages,
  activeStage: null,

  // Reasoning
  reasoningSteps: [],
  activeReasoningStep: null,

  // Metrics
  startTime: null,
  endTime: null,
  totalDuration: null,
  tokensGenerated: 0,
  cacheHit: false,

  // Errors
  error: null,
};

// ============================================================================
// Store Creation
// ============================================================================

export const useHoloLoomStore = create<HoloLoomStore>()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        ...initialState,

        // ====================================================================
        // Connection Actions
        // ====================================================================

        setConnectionState: (state: ConnectionState) =>
          set(
            draft => {
              draft.connectionState = state;
              if (state === 'connected') {
                draft.reconnectAttempts = 0;
              }
            },
            false,
            'setConnectionState'
          ),

        updateHeartbeat: (timestamp: string) =>
          set(
            draft => {
              draft.lastHeartbeat = timestamp;
            },
            false,
            'updateHeartbeat'
          ),

        incrementReconnectAttempts: () =>
          set(
            draft => {
              draft.reconnectAttempts += 1;
            },
            false,
            'incrementReconnectAttempts'
          ),

        resetReconnectAttempts: () =>
          set(
            draft => {
              draft.reconnectAttempts = 0;
            },
            false,
            'resetReconnectAttempts'
          ),

        // ====================================================================
        // Session Actions
        // ====================================================================

        startSession: (sessionId: string, query: string, mode: ReasoningMode) =>
          set(
            draft => {
              draft.sessionId = sessionId;
              draft.query = query;
              draft.mode = mode;
              draft.streamState = 'streaming';
              draft.startTime = Date.now();
              draft.tokens = [];
              draft.accumulatedText = '';
              draft.contextChunks = [];
              draft.totalContextTokens = 0;
              draft.confidence = 0;
              draft.epistemicConfidence = 0;
              draft.confidenceHistory = [];
              draft.stages = initialStages.map(s => ({ ...s, status: 'pending' as const }));
              draft.activeStage = null;
              draft.reasoningSteps = [];
              draft.activeReasoningStep = null;
              draft.error = null;
            },
            false,
            'startSession'
          ),

        endSession: () =>
          set(
            draft => {
              draft.streamState = 'complete';
              draft.endTime = Date.now();
              if (draft.startTime) {
                draft.totalDuration = draft.endTime - draft.startTime;
              }
              draft.activeStage = null;
              draft.activeReasoningStep = null;
            },
            false,
            'endSession'
          ),

        // ====================================================================
        // Streaming Actions
        // ====================================================================

        setStreamState: (state: StreamState) =>
          set(
            draft => {
              draft.streamState = state;
            },
            false,
            'setStreamState'
          ),

        appendToken: (token: string) =>
          set(
            draft => {
              draft.tokens.push(token);
              draft.accumulatedText += token;
              draft.tokensGenerated = draft.tokens.length;
            },
            false,
            'appendToken'
          ),

        setAccumulatedText: (text: string) =>
          set(
            draft => {
              draft.accumulatedText = text;
            },
            false,
            'setAccumulatedText'
          ),

        // ====================================================================
        // Context Actions
        // ====================================================================

        addContextChunk: (chunk: ContextChunk) =>
          set(
            draft => {
              draft.contextChunks.push(chunk);
              draft.totalContextTokens += chunk.tokenCount;
            },
            false,
            'addContextChunk'
          ),

        clearContext: () =>
          set(
            draft => {
              draft.contextChunks = [];
              draft.totalContextTokens = 0;
            },
            false,
            'clearContext'
          ),

        // ====================================================================
        // Confidence Actions
        // ====================================================================

        updateConfidence: (confidence: number, epistemic?: number) =>
          set(
            draft => {
              draft.confidence = confidence;
              if (epistemic !== undefined) {
                draft.epistemicConfidence = epistemic;
              }
              // Keep last 50 confidence values for sparkline
              draft.confidenceHistory.push(confidence);
              if (draft.confidenceHistory.length > 50) {
                draft.confidenceHistory.shift();
              }
            },
            false,
            'updateConfidence'
          ),

        // ====================================================================
        // Stage Actions
        // ====================================================================

        updateStage: (
          stage: WeavingStage,
          status: 'active' | 'complete' | 'error',
          metrics?: StageMetrics
        ) =>
          set(
            draft => {
              const stageIndex = draft.stages.findIndex(s => s.stage === stage);
              if (stageIndex !== -1) {
                draft.stages[stageIndex].status = status;
                if (metrics) {
                  draft.stages[stageIndex].duration_ms = metrics.duration_ms;
                  draft.stages[stageIndex].metrics = metrics;
                }
              }

              if (status === 'active') {
                draft.activeStage = stage;
              } else if (status === 'complete' && draft.activeStage === stage) {
                // Move to next stage
                const nextIndex = stageIndex + 1;
                if (nextIndex < draft.stages.length) {
                  draft.activeStage = draft.stages[nextIndex].stage;
                } else {
                  draft.activeStage = null;
                }
              }
            },
            false,
            'updateStage'
          ),

        // ====================================================================
        // Reasoning Actions
        // ====================================================================

        addReasoningStep: (step: ReasoningStep) =>
          set(
            draft => {
              draft.reasoningSteps.push(step);
              if (step.status === 'active') {
                draft.activeReasoningStep = step.index;
              }
            },
            false,
            'addReasoningStep'
          ),

        updateReasoningStep: (index: number, updates: Partial<ReasoningStep>) =>
          set(
            draft => {
              const stepIndex = draft.reasoningSteps.findIndex(s => s.index === index);
              if (stepIndex !== -1) {
                Object.assign(draft.reasoningSteps[stepIndex], updates);
                if (updates.status === 'active') {
                  draft.activeReasoningStep = index;
                } else if (
                  updates.status === 'complete' &&
                  draft.activeReasoningStep === index
                ) {
                  draft.activeReasoningStep = null;
                }
              }
            },
            false,
            'updateReasoningStep'
          ),

        // ====================================================================
        // Metrics Actions
        // ====================================================================

        setMetrics: metrics =>
          set(
            draft => {
              if (metrics.totalDuration !== undefined) {
                draft.totalDuration = metrics.totalDuration;
              }
              if (metrics.tokensGenerated !== undefined) {
                draft.tokensGenerated = metrics.tokensGenerated;
              }
              if (metrics.cacheHit !== undefined) {
                draft.cacheHit = metrics.cacheHit;
              }
            },
            false,
            'setMetrics'
          ),

        // ====================================================================
        // Error Actions
        // ====================================================================

        setError: error =>
          set(
            draft => {
              draft.error = error;
              if (error) {
                draft.streamState = 'error';
              }
            },
            false,
            'setError'
          ),

        // ====================================================================
        // Reset Action
        // ====================================================================

        reset: () =>
          set(
            () => ({
              ...initialState,
              stages: initialStages.map(s => ({ ...s, status: 'pending' as const })),
            }),
            false,
            'reset'
          ),
      }))
    ),
    { name: 'HoloLoomStore' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

/**
 * Select connection status
 */
export const selectConnectionState = (state: HoloLoomStore) => state.connectionState;
export const selectIsConnected = (state: HoloLoomStore) => state.connectionState === 'connected';

/**
 * Select streaming status
 */
export const selectStreamState = (state: HoloLoomStore) => state.streamState;
export const selectIsStreaming = (state: HoloLoomStore) => state.streamState === 'streaming';
export const selectIsComplete = (state: HoloLoomStore) => state.streamState === 'complete';

/**
 * Select response data
 */
export const selectAccumulatedText = (state: HoloLoomStore) => state.accumulatedText;
export const selectTokens = (state: HoloLoomStore) => state.tokens;
export const selectTokenCount = (state: HoloLoomStore) => state.tokens.length;

/**
 * Select context data
 */
export const selectContextChunks = (state: HoloLoomStore) => state.contextChunks;
export const selectContextTokenCount = (state: HoloLoomStore) => state.totalContextTokens;

/**
 * Select confidence data
 */
export const selectConfidence = (state: HoloLoomStore) => state.confidence;
export const selectEpistemicConfidence = (state: HoloLoomStore) => state.epistemicConfidence;
export const selectConfidenceHistory = (state: HoloLoomStore) => state.confidenceHistory;

/**
 * Select stage data
 */
export const selectStages = (state: HoloLoomStore) => state.stages;
export const selectActiveStage = (state: HoloLoomStore) => state.activeStage;
export const selectCompletedStages = (state: HoloLoomStore) =>
  state.stages.filter(s => s.status === 'complete');

/**
 * Select reasoning data
 */
export const selectReasoningSteps = (state: HoloLoomStore) => state.reasoningSteps;
export const selectActiveReasoningStep = (state: HoloLoomStore) => state.activeReasoningStep;

/**
 * Select metrics
 */
export const selectMetrics = (state: HoloLoomStore) => ({
  totalDuration: state.totalDuration,
  tokensGenerated: state.tokensGenerated,
  contextTokens: state.totalContextTokens,
  cacheHit: state.cacheHit,
  stageCount: state.stages.filter(s => s.status === 'complete').length,
});

/**
 * Select error state
 */
export const selectError = (state: HoloLoomStore) => state.error;
export const selectHasError = (state: HoloLoomStore) => state.error !== null;

// ============================================================================
// Derived Selectors
// ============================================================================

/**
 * Calculate overall progress (0-100)
 */
export const selectProgress = (state: HoloLoomStore): number => {
  const completedStages = state.stages.filter(s => s.status === 'complete').length;
  return Math.round((completedStages / state.stages.length) * 100);
};

/**
 * Calculate average stage duration
 */
export const selectAverageStageDuration = (state: HoloLoomStore): number | null => {
  const completedWithDuration = state.stages.filter(
    s => s.status === 'complete' && s.duration_ms !== undefined
  );
  if (completedWithDuration.length === 0) return null;

  const total = completedWithDuration.reduce((sum, s) => sum + (s.duration_ms || 0), 0);
  return total / completedWithDuration.length;
};

/**
 * Get current stage name for display
 */
export const selectCurrentStageDisplay = (state: HoloLoomStore): string => {
  if (!state.activeStage) {
    if (state.streamState === 'complete') return 'Complete';
    if (state.streamState === 'error') return 'Error';
    return 'Idle';
  }

  const stageNames: Record<WeavingStage, string> = {
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

  return stageNames[state.activeStage];
};
