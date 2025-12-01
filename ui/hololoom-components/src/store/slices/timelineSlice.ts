/**
 * Timeline Slice for HoloLoom Store
 *
 * Manages state for the Streaming Consciousness Timeline component.
 * Tracks stage timing, token rate, and animation state.
 *
 * @module store/slices/timelineSlice
 */

import type { StateCreator } from 'zustand';
import type { WeavingStage } from '../../types';

// ============================================================================
// Types
// ============================================================================

/**
 * 9 weaving stages in order
 */
export const WEAVING_STAGES: WeavingStage[] = [
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

/**
 * Human-readable stage names
 */
export const STAGE_DISPLAY_NAMES: Record<WeavingStage, string> = {
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

/**
 * Stage status
 */
export type StageStatus = 'pending' | 'active' | 'complete' | 'error';

/**
 * Stage timing data
 */
export interface StageTimingData {
  stage: WeavingStage;
  status: StageStatus;
  startTime: number | null;
  endTime: number | null;
  duration: number | null;
  /** Rolling average duration from previous runs */
  avgDuration: number | null;
  /** Metrics from backend */
  metrics?: {
    itemsProcessed?: number;
    memoryUsed?: number;
    cacheHit?: boolean;
  };
}

/**
 * Token rate history entry
 */
export interface TokenRateEntry {
  timestamp: number;
  tokensPerSecond: number;
}

/**
 * Timeline state
 */
export interface TimelineState {
  /** Timing data for each stage */
  stageTiming: Record<WeavingStage, StageTimingData>;
  /** Currently active stage (null if none) */
  activeStage: WeavingStage | null;
  /** Current token generation rate (tokens/second) */
  tokenRate: number;
  /** Token rate history (ring buffer, last 50 entries) */
  tokenRateHistory: TokenRateEntry[];
  /** Total tokens generated this session */
  totalTokens: number;
  /** Session start time */
  sessionStartTime: number | null;
  /** Whether timeline is expanded (showing details) */
  isExpanded: boolean;
  /** Historical average durations per stage (learned over time) */
  historicalAverages: Record<WeavingStage, number[]>;
}

/**
 * Timeline actions
 */
export interface TimelineActions {
  /** Mark stage as started */
  stageStart: (stage: WeavingStage) => void;
  /** Mark stage as completed with optional metrics */
  stageComplete: (stage: WeavingStage, durationMs: number, metrics?: StageTimingData['metrics']) => void;
  /** Mark stage as error */
  stageError: (stage: WeavingStage, error?: string) => void;
  /** Update token rate */
  updateTokenRate: (tokensPerSecond: number) => void;
  /** Increment token count */
  incrementTokens: (count?: number) => void;
  /** Start new session */
  startTimelineSession: () => void;
  /** Reset timeline to initial state */
  resetTimeline: () => void;
  /** Toggle expanded state */
  toggleExpanded: () => void;
  /** Set expanded state explicitly */
  setExpanded: (expanded: boolean) => void;
}

/**
 * Combined timeline slice
 */
export type TimelineSlice = TimelineState & TimelineActions;

// ============================================================================
// Initial State
// ============================================================================

function createInitialStageTiming(): Record<WeavingStage, StageTimingData> {
  const timing: Partial<Record<WeavingStage, StageTimingData>> = {};

  for (const stage of WEAVING_STAGES) {
    timing[stage] = {
      stage,
      status: 'pending',
      startTime: null,
      endTime: null,
      duration: null,
      avgDuration: null,
    };
  }

  return timing as Record<WeavingStage, StageTimingData>;
}

function createInitialHistoricalAverages(): Record<WeavingStage, number[]> {
  const averages: Partial<Record<WeavingStage, number[]>> = {};

  for (const stage of WEAVING_STAGES) {
    averages[stage] = [];
  }

  return averages as Record<WeavingStage, number[]>;
}

const initialTimelineState: TimelineState = {
  stageTiming: createInitialStageTiming(),
  activeStage: null,
  tokenRate: 0,
  tokenRateHistory: [],
  totalTokens: 0,
  sessionStartTime: null,
  isExpanded: false,
  historicalAverages: createInitialHistoricalAverages(),
};

// ============================================================================
// Slice Creator
// ============================================================================

const MAX_TOKEN_RATE_HISTORY = 50;
const MAX_HISTORICAL_AVERAGES = 10;

/**
 * Create the timeline slice for Zustand store.
 *
 * Usage with existing store:
 * ```typescript
 * import { create } from 'zustand';
 * import { immer } from 'zustand/middleware/immer';
 * import { createTimelineSlice } from './slices/timelineSlice';
 *
 * const useStore = create<HoloLoomStore & TimelineSlice>()(
 *   immer((...a) => ({
 *     ...existingSlices(...a),
 *     ...createTimelineSlice(...a),
 *   }))
 * );
 * ```
 */
export const createTimelineSlice: StateCreator<
  TimelineSlice,
  [['zustand/immer', never]],
  [],
  TimelineSlice
> = (set, get) => ({
  ...initialTimelineState,

  stageStart: (stage: WeavingStage) =>
    set((state) => {
      const now = Date.now();

      // Reset previous active stage if any
      if (state.activeStage && state.activeStage !== stage) {
        state.stageTiming[state.activeStage].status = 'pending';
      }

      state.stageTiming[stage] = {
        ...state.stageTiming[stage],
        status: 'active',
        startTime: now,
        endTime: null,
        duration: null,
      };
      state.activeStage = stage;
    }),

  stageComplete: (stage: WeavingStage, durationMs: number, metrics?) =>
    set((state) => {
      const now = Date.now();

      state.stageTiming[stage] = {
        ...state.stageTiming[stage],
        status: 'complete',
        endTime: now,
        duration: durationMs,
        metrics,
      };

      // Update historical averages
      state.historicalAverages[stage].push(durationMs);
      if (state.historicalAverages[stage].length > MAX_HISTORICAL_AVERAGES) {
        state.historicalAverages[stage].shift();
      }

      // Calculate average duration
      const history = state.historicalAverages[stage];
      state.stageTiming[stage].avgDuration =
        history.reduce((a, b) => a + b, 0) / history.length;

      // Move to next stage
      const currentIndex = WEAVING_STAGES.indexOf(stage);
      if (currentIndex < WEAVING_STAGES.length - 1) {
        state.activeStage = WEAVING_STAGES[currentIndex + 1];
      } else {
        state.activeStage = null;
      }
    }),

  stageError: (stage: WeavingStage) =>
    set((state) => {
      state.stageTiming[stage].status = 'error';
      state.stageTiming[stage].endTime = Date.now();
      state.activeStage = null;
    }),

  updateTokenRate: (tokensPerSecond: number) =>
    set((state) => {
      state.tokenRate = tokensPerSecond;

      // Add to history
      state.tokenRateHistory.push({
        timestamp: Date.now(),
        tokensPerSecond,
      });

      // Keep history bounded
      if (state.tokenRateHistory.length > MAX_TOKEN_RATE_HISTORY) {
        state.tokenRateHistory.shift();
      }
    }),

  incrementTokens: (count = 1) =>
    set((state) => {
      state.totalTokens += count;
    }),

  startTimelineSession: () =>
    set((state) => {
      // Reset all stage timing
      for (const stage of WEAVING_STAGES) {
        state.stageTiming[stage] = {
          stage,
          status: 'pending',
          startTime: null,
          endTime: null,
          duration: null,
          avgDuration: state.stageTiming[stage].avgDuration, // Preserve averages
        };
      }

      state.activeStage = null;
      state.tokenRate = 0;
      state.tokenRateHistory = [];
      state.totalTokens = 0;
      state.sessionStartTime = Date.now();
    }),

  resetTimeline: () =>
    set(() => ({
      ...initialTimelineState,
      historicalAverages: createInitialHistoricalAverages(),
    })),

  toggleExpanded: () =>
    set((state) => {
      state.isExpanded = !state.isExpanded;
    }),

  setExpanded: (expanded: boolean) =>
    set((state) => {
      state.isExpanded = expanded;
    }),
});

// ============================================================================
// Selectors
// ============================================================================

export const selectStageTiming = (state: TimelineSlice) => state.stageTiming;
export const selectActiveStage = (state: TimelineSlice) => state.activeStage;
export const selectTokenRate = (state: TimelineSlice) => state.tokenRate;
export const selectTokenRateHistory = (state: TimelineSlice) => state.tokenRateHistory;
export const selectTotalTokens = (state: TimelineSlice) => state.totalTokens;
export const selectIsTimelineExpanded = (state: TimelineSlice) => state.isExpanded;

/**
 * Get ordered array of stage timing data
 */
export const selectOrderedStages = (state: TimelineSlice): StageTimingData[] =>
  WEAVING_STAGES.map(stage => state.stageTiming[stage]);

/**
 * Get completed stages count
 */
export const selectCompletedStagesCount = (state: TimelineSlice): number =>
  WEAVING_STAGES.filter(stage => state.stageTiming[stage].status === 'complete').length;

/**
 * Get progress percentage (0-100)
 */
export const selectProgress = (state: TimelineSlice): number => {
  const completed = selectCompletedStagesCount(state);
  return Math.round((completed / WEAVING_STAGES.length) * 100);
};

/**
 * Get total elapsed time
 */
export const selectTotalElapsedTime = (state: TimelineSlice): number => {
  const completedStages = WEAVING_STAGES.filter(
    stage => state.stageTiming[stage].duration !== null
  );

  return completedStages.reduce(
    (total, stage) => total + (state.stageTiming[stage].duration || 0),
    0
  );
};

/**
 * Get estimated remaining time based on historical averages
 */
export const selectEstimatedRemainingTime = (state: TimelineSlice): number | null => {
  const pendingStages = WEAVING_STAGES.filter(
    stage => state.stageTiming[stage].status === 'pending' || stage === state.activeStage
  );

  let hasEstimate = false;
  let estimated = 0;

  for (const stage of pendingStages) {
    const avgDuration = state.stageTiming[stage].avgDuration;
    if (avgDuration !== null) {
      hasEstimate = true;
      estimated += avgDuration;
    }
  }

  return hasEstimate ? estimated : null;
};

/**
 * Get average token rate over history
 */
export const selectAverageTokenRate = (state: TimelineSlice): number => {
  if (state.tokenRateHistory.length === 0) return 0;

  const sum = state.tokenRateHistory.reduce((acc, entry) => acc + entry.tokensPerSecond, 0);
  return sum / state.tokenRateHistory.length;
};

export default createTimelineSlice;
