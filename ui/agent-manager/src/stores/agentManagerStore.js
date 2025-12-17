import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
/**
 * Zustand store for Agent Manager
 * Uses Immer middleware for immutable updates
 */
export const useAgentManagerStore = create()(immer((set, get) => ({
    // Initial State
    threads: {},
    activeThreadId: null,
    filter: 'all',
    viewMode: 'outline',
    isConnected: false,
    connectionError: null,
    // Actions: Thread Management
    addThread: (thread) => set((state) => {
        state.threads[thread.id] = thread;
    }),
    updateThread: (id, updates) => set((state) => {
        if (state.threads[id]) {
            state.threads[id] = {
                ...state.threads[id],
                ...updates,
                updatedAt: new Date().toISOString(),
            };
        }
    }),
    removeThread: (id) => set((state) => {
        delete state.threads[id];
        if (state.activeThreadId === id) {
            state.activeThreadId = null;
        }
    }),
    setActiveThread: (id) => set((state) => {
        state.activeThreadId = id;
    }),
    setFilter: (filter) => set((state) => {
        state.filter = filter;
    }),
    setViewMode: (mode) => set((state) => {
        state.viewMode = mode;
    }),
    setConnectionStatus: (connected, error) => set((state) => {
        state.isConnected = connected;
        state.connectionError = error ?? null;
    }),
    // Actions: Thread State Transitions
    pauseThread: (id) => set((state) => {
        if (state.threads[id] && state.threads[id].status === 'running') {
            state.threads[id].status = 'paused';
            state.threads[id].updatedAt = new Date().toISOString();
        }
    }),
    resumeThread: (id) => set((state) => {
        if (state.threads[id] && state.threads[id].status === 'paused') {
            state.threads[id].status = 'running';
            state.threads[id].updatedAt = new Date().toISOString();
        }
    }),
    cancelThread: (id) => set((state) => {
        if (state.threads[id]) {
            const isRunning = state.threads[id].status === 'running' || state.threads[id].status === 'paused';
            if (isRunning) {
                state.threads[id].status = 'cancelled';
                state.threads[id].updatedAt = new Date().toISOString();
            }
        }
    }),
    // Actions: Priority Management
    upvoteThread: (id) => set((state) => {
        if (state.threads[id]) {
            state.threads[id].priority = Math.min(100, state.threads[id].priority + 1);
            state.threads[id].updatedAt = new Date().toISOString();
        }
    }),
    downvoteThread: (id) => set((state) => {
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
                return threadsArray.filter((t) => t.status === 'running' || t.status === 'paused');
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
    getThreadById: (id) => {
        const state = get();
        return state.threads[id];
    },
    // Selectors: Hierarchy Navigation
    getChildThreads: (parentId) => {
        const state = get();
        return Object.values(state.threads).filter((t) => t.parentThreadId === parentId);
    },
    getThreadsBySwarm: (swarmId) => {
        const state = get();
        return Object.values(state.threads).filter((t) => t.swarmId === swarmId);
    },
    getThreadDependencies: (id) => {
        const state = get();
        const thread = state.threads[id];
        if (!thread) {
            return { dependsOn: [], blocks: [] };
        }
        const dependsOn = thread.dependsOn
            .map((depId) => state.threads[depId])
            .filter((t) => Boolean(t));
        const blocks = thread.blocks
            .map((blockId) => state.threads[blockId])
            .filter((t) => Boolean(t));
        return { dependsOn, blocks };
    },
    // Selectors: Swarm Analytics
    getSwarmStatus: (swarmId) => {
        const state = get();
        const swarmThreads = Object.values(state.threads).filter((t) => t.swarmId === swarmId);
        const total = swarmThreads.length;
        const running = swarmThreads.filter((t) => t.status === 'running').length;
        const completed = swarmThreads.filter((t) => t.status === 'completed').length;
        const failed = swarmThreads.filter((t) => t.status === 'failed' || t.status === 'cancelled').length;
        const avgConfidence = total > 0
            ? swarmThreads.reduce((sum, t) => sum + t.confidence, 0) / total
            : 0;
        return { total, running, completed, failed, avgConfidence };
    },
})));
/**
 * Composite selector for thread with full dependency info
 * Useful for rendering thread details with context
 */
export const useThreadWithDependencies = (threadId) => {
    const thread = useAgentManagerStore((state) => state.getThreadById(threadId));
    const dependencies = useAgentManagerStore((state) => state.getThreadDependencies(threadId));
    const children = useAgentManagerStore((state) => state.getChildThreads(threadId));
    return { thread, dependencies, children };
};
/**
 * Composite selector for swarm overview
 * Combines swarm status with all member threads
 */
export const useSwarmOverview = (swarmId) => {
    const threads = useAgentManagerStore((state) => state.getThreadsBySwarm(swarmId));
    const status = useAgentManagerStore((state) => state.getSwarmStatus(swarmId));
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
//# sourceMappingURL=agentManagerStore.js.map