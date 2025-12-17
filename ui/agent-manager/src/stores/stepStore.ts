/**
 * Step Store for Agent Manager
 *
 * Manages step data per thread with MRF/MCTS injection support.
 * Integrates with injection API and WebSocket for real-time updates.
 *
 * HoloLoom Agent Manager UI - Phase 5
 * Date: 2025-12-15
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { TaskNode, StepStatus } from '../components/DetailPanel/types';
import { api, MCTSConfig, InjectionResponse, InjectionHistoryEntry } from '../lib/api';

// ============================================================================
// Types
// ============================================================================

/**
 * MRF strategy options
 */
export type MRFStrategy = 'auto' | 'verify' | 'elegance' | 'critique' | 'refine' | 'hofstadter';

/**
 * Injection request state
 */
export interface InjectionRequest {
  threadId: string;
  stepId: string;
  type: 'mrf' | 'mcts';
  strategy?: MRFStrategy;
  config?: MCTSConfig;
  status: 'pending' | 'in_progress' | 'success' | 'error';
  error?: string;
  timestamp: string;
}

/**
 * Step Store State
 */
export interface StepStoreState {
  // Data: Steps organized by thread ID
  stepsByThread: Record<string, TaskNode[]>;

  // Injection history per thread
  injectionHistory: Record<string, InjectionHistoryEntry[]>;

  // Active injection requests (for loading states)
  activeInjections: Record<string, InjectionRequest>;

  // Actions: Step Management
  setStepsForThread: (threadId: string, steps: TaskNode[]) => void;
  updateStep: (threadId: string, stepId: string, updates: Partial<TaskNode>) => void;
  addStep: (threadId: string, step: TaskNode) => void;
  clearStepsForThread: (threadId: string) => void;

  // Actions: Injection
  injectMRF: (threadId: string, stepId: string, strategy: MRFStrategy) => Promise<InjectionResponse>;
  injectMCTS: (threadId: string, stepId: string, config: MCTSConfig) => Promise<InjectionResponse>;
  loadInjectionHistory: (threadId: string) => Promise<void>;

  // Actions: Handle WebSocket injection feedback
  handleInjectionFeedback: (message: InjectionFeedbackMessage) => void;

  // Selectors
  getStepsForThread: (threadId: string) => TaskNode[];
  getStepById: (threadId: string, stepId: string) => TaskNode | undefined;
  getInjectionHistoryForThread: (threadId: string) => InjectionHistoryEntry[];
  isInjectionInProgress: (threadId: string, stepId: string) => boolean;
}

/**
 * WebSocket injection feedback message format
 */
export interface InjectionFeedbackMessage {
  type: 'MRF_INJECTED' | 'MCTS_INJECTED';
  thread_id: string;
  step_id: string;
  success: boolean;
  injection_type: 'mrf' | 'mcts';
  strategy?: string;
  config?: MCTSConfig;
  message?: string;
  timestamp: string;
}

// ============================================================================
// Store Implementation
// ============================================================================

export const useStepStore = create<StepStoreState>()(
  immer((set, get) => ({
    // Initial State
    stepsByThread: {},
    injectionHistory: {},
    activeInjections: {},

    // ========================================================================
    // Step Management Actions
    // ========================================================================

    setStepsForThread: (threadId: string, steps: TaskNode[]) =>
      set((state) => {
        state.stepsByThread[threadId] = steps;
      }),

    updateStep: (threadId: string, stepId: string, updates: Partial<TaskNode>) =>
      set((state) => {
        const steps = state.stepsByThread[threadId];
        if (steps) {
          const stepIndex = steps.findIndex((s) => s.id === stepId);
          if (stepIndex !== -1) {
            state.stepsByThread[threadId][stepIndex] = {
              ...steps[stepIndex],
              ...updates,
            };
          }
        }
      }),

    addStep: (threadId: string, step: TaskNode) =>
      set((state) => {
        if (!state.stepsByThread[threadId]) {
          state.stepsByThread[threadId] = [];
        }
        state.stepsByThread[threadId].push(step);
      }),

    clearStepsForThread: (threadId: string) =>
      set((state) => {
        delete state.stepsByThread[threadId];
        delete state.injectionHistory[threadId];
      }),

    // ========================================================================
    // Injection Actions
    // ========================================================================

    injectMRF: async (threadId: string, stepId: string, strategy: MRFStrategy) => {
      const requestKey = `${threadId}:${stepId}`;

      // Set injection in progress
      set((state) => {
        state.activeInjections[requestKey] = {
          threadId,
          stepId,
          type: 'mrf',
          strategy,
          status: 'in_progress',
          timestamp: new Date().toISOString(),
        };
      });

      try {
        // Call API
        const response = await api.injectMRF(threadId, stepId, strategy);

        // Update step with injection result
        set((state) => {
          const steps = state.stepsByThread[threadId];
          if (steps) {
            const stepIndex = steps.findIndex((s) => s.id === stepId);
            if (stepIndex !== -1) {
              state.stepsByThread[threadId][stepIndex].injectionApplied = 'mrf';
              state.stepsByThread[threadId][stepIndex].injectionStrategy = strategy;
            }
          }

          // Update active injection status
          state.activeInjections[requestKey] = {
            ...state.activeInjections[requestKey],
            status: 'success',
          };

          // Add to injection history
          if (!state.injectionHistory[threadId]) {
            state.injectionHistory[threadId] = [];
          }
          state.injectionHistory[threadId].unshift({
            id: `${response.thread_id}-${response.step_id}-${Date.now()}`,
            threadId: response.thread_id,
            stepId: response.step_id,
            injectionType: 'mrf',
            strategy: response.strategy,
            timestamp: response.timestamp,
            success: response.success,
          });
        });

        // Clear active injection after a delay
        setTimeout(() => {
          set((state) => {
            delete state.activeInjections[requestKey];
          });
        }, 2000);

        return response;
      } catch (error) {
        // Update active injection with error
        set((state) => {
          state.activeInjections[requestKey] = {
            ...state.activeInjections[requestKey],
            status: 'error',
            error: error instanceof Error ? error.message : 'Unknown error',
          };
        });

        // Clear active injection after a delay
        setTimeout(() => {
          set((state) => {
            delete state.activeInjections[requestKey];
          });
        }, 5000);

        throw error;
      }
    },

    injectMCTS: async (threadId: string, stepId: string, config: MCTSConfig) => {
      const requestKey = `${threadId}:${stepId}`;

      // Set injection in progress
      set((state) => {
        state.activeInjections[requestKey] = {
          threadId,
          stepId,
          type: 'mcts',
          config,
          status: 'in_progress',
          timestamp: new Date().toISOString(),
        };
      });

      try {
        // Call API
        const response = await api.injectMCTS(threadId, stepId, config);

        // Update step with injection result
        set((state) => {
          const steps = state.stepsByThread[threadId];
          if (steps) {
            const stepIndex = steps.findIndex((s) => s.id === stepId);
            if (stepIndex !== -1) {
              state.stepsByThread[threadId][stepIndex].injectionApplied = 'mcts';
            }
          }

          // Update active injection status
          state.activeInjections[requestKey] = {
            ...state.activeInjections[requestKey],
            status: 'success',
          };

          // Add to injection history
          if (!state.injectionHistory[threadId]) {
            state.injectionHistory[threadId] = [];
          }
          state.injectionHistory[threadId].unshift({
            id: `${response.thread_id}-${response.step_id}-${Date.now()}`,
            threadId: response.thread_id,
            stepId: response.step_id,
            injectionType: 'mcts',
            config: response.config,
            timestamp: response.timestamp,
            success: response.success,
          });
        });

        // Clear active injection after a delay
        setTimeout(() => {
          set((state) => {
            delete state.activeInjections[requestKey];
          });
        }, 2000);

        return response;
      } catch (error) {
        // Update active injection with error
        set((state) => {
          state.activeInjections[requestKey] = {
            ...state.activeInjections[requestKey],
            status: 'error',
            error: error instanceof Error ? error.message : 'Unknown error',
          };
        });

        // Clear active injection after a delay
        setTimeout(() => {
          set((state) => {
            delete state.activeInjections[requestKey];
          });
        }, 5000);

        throw error;
      }
    },

    loadInjectionHistory: async (threadId: string) => {
      try {
        const history = await api.getInjectionHistory(threadId);
        set((state) => {
          state.injectionHistory[threadId] = history;
        });
      } catch (error) {
        // Silently fail - history is optional
        console.warn('Failed to load injection history:', error);
      }
    },

    // ========================================================================
    // WebSocket Feedback Handler
    // ========================================================================

    handleInjectionFeedback: (message: InjectionFeedbackMessage) =>
      set((state) => {
        const { thread_id, step_id, success, injection_type, strategy, config, timestamp } = message;

        // Update step
        const steps = state.stepsByThread[thread_id];
        if (steps) {
          const stepIndex = steps.findIndex((s) => s.id === step_id);
          if (stepIndex !== -1 && success) {
            state.stepsByThread[thread_id][stepIndex].injectionApplied = injection_type;
            if (strategy) {
              state.stepsByThread[thread_id][stepIndex].injectionStrategy = strategy;
            }
          }
        }

        // Add to injection history
        if (!state.injectionHistory[thread_id]) {
          state.injectionHistory[thread_id] = [];
        }
        state.injectionHistory[thread_id].unshift({
          id: `${thread_id}-${step_id}-${Date.now()}`,
          threadId: thread_id,
          stepId: step_id,
          injectionType: injection_type,
          strategy,
          config,
          timestamp,
          success,
        });

        // Clear any active injection for this step
        const requestKey = `${thread_id}:${step_id}`;
        delete state.activeInjections[requestKey];
      }),

    // ========================================================================
    // Selectors
    // ========================================================================

    getStepsForThread: (threadId: string) => {
      const state = get();
      return state.stepsByThread[threadId] || [];
    },

    getStepById: (threadId: string, stepId: string) => {
      const state = get();
      const steps = state.stepsByThread[threadId];
      return steps?.find((s) => s.id === stepId);
    },

    getInjectionHistoryForThread: (threadId: string) => {
      const state = get();
      return state.injectionHistory[threadId] || [];
    },

    isInjectionInProgress: (threadId: string, stepId: string) => {
      const state = get();
      const requestKey = `${threadId}:${stepId}`;
      const injection = state.activeInjections[requestKey];
      return injection?.status === 'in_progress';
    },
  }))
);

// ============================================================================
// Convenience Hooks
// ============================================================================

/**
 * Hook to get steps and injection callbacks for a thread
 */
export const useThreadSteps = (threadId: string) => {
  const steps = useStepStore((state) => state.getStepsForThread(threadId));
  const injectMRF = useStepStore((state) => state.injectMRF);
  const injectMCTS = useStepStore((state) => state.injectMCTS);
  const isInjectionInProgress = useStepStore((state) => state.isInjectionInProgress);

  return {
    steps,
    injectMRF: (stepId: string, strategy: MRFStrategy) => injectMRF(threadId, stepId, strategy),
    injectMCTS: (stepId: string, config: MCTSConfig) => injectMCTS(threadId, stepId, config),
    isInjecting: (stepId: string) => isInjectionInProgress(threadId, stepId),
  };
};

/**
 * Hook to get injection history for a thread
 */
export const useInjectionHistory = (threadId: string) => {
  const history = useStepStore((state) => state.getInjectionHistoryForThread(threadId));
  const loadHistory = useStepStore((state) => state.loadInjectionHistory);

  return {
    history,
    reload: () => loadHistory(threadId),
  };
};

export default useStepStore;
