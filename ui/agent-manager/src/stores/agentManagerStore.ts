import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

/**
 * Agent Thread Data Structure
 * Represents a single agent reasoning thread in the swarm
 */
export interface AgentThread {
  id: string;
  name: string;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  priority: number;
  agentType: string;
  reasoningMode: 'DIRECT' | 'VERIFY' | 'RESEARCH' | 'PLAN_EXECUTE';
  currentStep: number;
  totalSteps: number;
  elapsedTimeMs: number;
  tokensUsed: number;
  tokenBudget?: number;
  confidence: number;
  epistemicConfidence: number;
  swarmId?: string;
  parentThreadId?: string;
  childThreadIds: string[];
  dependsOn: string[];
  blocks: string[];
  finalResponse?: string;
  createdAt: string;
  updatedAt: string;
}

/**
 * Agent Manager State
 * Manages the overall state of the agent manager UI
 */
export interface AgentManagerState {
  // Data
  threads: Record<string, AgentThread>;
  activeThreadId: string | null;

  // UI Configuration
  filter: 'all' | 'active' | 'completed' | 'failed';
  viewMode: 'outline' | 'tree' | 'swarm';

  // Connection Status
  isConnected: boolean;
  connectionError: string | null;

  // Actions
  addThread: (thread: AgentThread) => void;
  updateThread: (id: string, updates: Partial<AgentThread>) => void;
  removeThread: (id: string) => void;
  setActiveThread: (id: string | null) => void;
  setFilter: (filter: 'all' | 'active' | 'completed' | 'failed') => void;
  setViewMode: (mode: 'outline' | 'tree' | 'swarm') => void;
  setConnectionStatus: (connected: boolean, error?: string | null) => void;
  pauseThread: (id: string) => void;
  resumeThread: (id: string) => void;
  cancelThread: (id: string) => void;
  upvoteThread: (id: string) => void;
  downvoteThread: (id: string) => void;

  // Selectors (computed)
  getFilteredThreads: () => AgentThread[];
  getActiveThread: () => AgentThread | undefined;
  getThreadById: (id: string) => AgentThread | undefined;
  getChildThreads: (parentId: string) => AgentThread[];
  getThreadsBySwarm: (swarmId: string) => AgentThread[];
  getThreadDependencies: (id: string) => { dependsOn: AgentThread[]; blocks: AgentThread[] };
  getSwarmStatus: (swarmId: string) => {
    total: number;
    running: number;
    completed: number;
    failed: number;
    avgConfidence: number;
  };
}

/**
 * Zustand store for Agent Manager
 * Uses Immer middleware for immutable updates
 */
export const useAgentManagerStore = create<AgentManagerState>()(
  immer((set, get) => ({
    // Initial State
    threads: {},
    activeThreadId: null,
    filter: 'all',
    viewMode: 'outline',
    isConnected: false,
    connectionError: null,

    // Actions: Thread Management
    addThread: (thread: AgentThread) =>
      set((state) => {
        state.threads[thread.id] = thread;
      }),

    updateThread: (id: string, updates: Partial<AgentThread>) =>
      set((state) => {
        if (state.threads[id]) {
          state.threads[id] = {
            ...state.threads[id],
            ...updates,
            updatedAt: new Date().toISOString(),
          };
        }
      }),

    removeThread: (id: string) =>
      set((state) => {
        delete state.threads[id];
        if (state.activeThreadId === id) {
          state.activeThreadId = null;
        }
      }),

    setActiveThread: (id: string | null) =>
      set((state) => {
        state.activeThreadId = id;
      }),

    setFilter: (filter: 'all' | 'active' | 'completed' | 'failed') =>
      set((state) => {
        state.filter = filter;
      }),

    setViewMode: (mode: 'outline' | 'tree' | 'swarm') =>
      set((state) => {
        state.viewMode = mode;
      }),

    setConnectionStatus: (connected: boolean, error?: string | null) =>
      set((state) => {
        state.isConnected = connected;
        state.connectionError = error ?? null;
      }),

    // Actions: Thread State Transitions
    pauseThread: (id: string) =>
      set((state) => {
        if (state.threads[id] && state.threads[id].status === 'running') {
          state.threads[id].status = 'paused';
          state.threads[id].updatedAt = new Date().toISOString();
        }
      }),

    resumeThread: (id: string) =>
      set((state) => {
        if (state.threads[id] && state.threads[id].status === 'paused') {
          state.threads[id].status = 'running';
          state.threads[id].updatedAt = new Date().toISOString();
        }
      }),

    cancelThread: (id: string) =>
      set((state) => {
        if (state.threads[id]) {
          const isRunning = state.threads[id].status === 'running' || state.threads[id].status === 'paused';
          if (isRunning) {
            state.threads[id].status = 'cancelled';
            state.threads[id].updatedAt = new Date().toISOString();
          }
        }
      }),

    // Actions: Priority Management
    upvoteThread: (id: string) =>
      set((state) => {
        if (state.threads[id]) {
          state.threads[id].priority = Math.min(100, state.threads[id].priority + 1);
          state.threads[id].updatedAt = new Date().toISOString();
        }
      }),

    downvoteThread: (id: string) =>
      set((state) => {
        if (state.threads[id]) {
          state.threads[id].priority = Math.max(0, state.threads[id].priority - 1);
          state.threads[id].updatedAt = new Date().toISOString();
        }
      }),

    // Selectors: Filtering & Lookup
    getFilteredThreads: () => {
      const state = get();
      const threadsArray = Object.values(state.threads);

      switch (state.filter) {
        case 'active':
          return threadsArray.filter(
            (t) => t.status === 'running' || t.status === 'paused'
          );
        case 'completed':
          return threadsArray.filter((t) => t.status === 'completed');
        case 'failed':
          return threadsArray.filter((t) => t.status === 'failed' || t.status === 'cancelled');
        case 'all':
        default:
          return threadsArray;
      }
    },

    getActiveThread: () => {
      const state = get();
      return state.activeThreadId ? state.threads[state.activeThreadId] : undefined;
    },

    getThreadById: (id: string) => {
      const state = get();
      return state.threads[id];
    },

    // Selectors: Hierarchy Navigation
    getChildThreads: (parentId: string) => {
      const state = get();
      return Object.values(state.threads).filter((t) => t.parentThreadId === parentId);
    },

    getThreadsBySwarm: (swarmId: string) => {
      const state = get();
      return Object.values(state.threads).filter((t) => t.swarmId === swarmId);
    },

    getThreadDependencies: (id: string) => {
      const state = get();
      const thread = state.threads[id];

      if (!thread) {
        return { dependsOn: [], blocks: [] };
      }

      const dependsOn = thread.dependsOn
        .map((depId) => state.threads[depId])
        .filter((t): t is AgentThread => Boolean(t));

      const blocks = thread.blocks
        .map((blockId) => state.threads[blockId])
        .filter((t): t is AgentThread => Boolean(t));

      return { dependsOn, blocks };
    },

    // Selectors: Swarm Analytics
    getSwarmStatus: (swarmId: string) => {
      const state = get();
      const swarmThreads = Object.values(state.threads).filter(
        (t) => t.swarmId === swarmId
      );

      const total = swarmThreads.length;
      const running = swarmThreads.filter((t) => t.status === 'running').length;
      const completed = swarmThreads.filter((t) => t.status === 'completed').length;
      const failed = swarmThreads.filter(
        (t) => t.status === 'failed' || t.status === 'cancelled'
      ).length;
      const avgConfidence =
        total > 0
          ? swarmThreads.reduce((sum, t) => sum + t.confidence, 0) / total
          : 0;

      return { total, running, completed, failed, avgConfidence };
    },
  }))
);

/**
 * Composite selector for thread with full dependency info
 * Useful for rendering thread details with context
 */
export const useThreadWithDependencies = (threadId: string) => {
  const thread = useAgentManagerStore((state) => state.getThreadById(threadId));
  const dependencies = useAgentManagerStore((state) =>
    state.getThreadDependencies(threadId)
  );
  const children = useAgentManagerStore((state) =>
    state.getChildThreads(threadId)
  );

  return { thread, dependencies, children };
};

/**
 * Composite selector for swarm overview
 * Combines swarm status with all member threads
 */
export const useSwarmOverview = (swarmId: string) => {
  const threads = useAgentManagerStore((state) =>
    state.getThreadsBySwarm(swarmId)
  );
  const status = useAgentManagerStore((state) =>
    state.getSwarmStatus(swarmId)
  );

  return { threads, status };
};

/**
 * Hook for active thread with dependencies
 * Automatically updates when active thread changes
 */
export const useActiveThreadDetails = () => {
  const activeThreadId = useAgentManagerStore((state) => state.activeThreadId);

  return useThreadWithDependencies(activeThreadId || '');
};

export default useAgentManagerStore;
