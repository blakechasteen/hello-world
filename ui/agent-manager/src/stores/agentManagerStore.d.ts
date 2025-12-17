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
    threads: Record<string, AgentThread>;
    activeThreadId: string | null;
    filter: 'all' | 'active' | 'completed' | 'failed';
    viewMode: 'outline' | 'tree' | 'swarm';
    isConnected: boolean;
    connectionError: string | null;
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
    getFilteredThreads: () => AgentThread[];
    getActiveThread: () => AgentThread | undefined;
    getThreadById: (id: string) => AgentThread | undefined;
    getChildThreads: (parentId: string) => AgentThread[];
    getThreadsBySwarm: (swarmId: string) => AgentThread[];
    getThreadDependencies: (id: string) => {
        dependsOn: AgentThread[];
        blocks: AgentThread[];
    };
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
export declare const useAgentManagerStore: import("zustand").UseBoundStore<Omit<import("zustand").StoreApi<AgentManagerState>, "setState"> & {
    setState(nextStateOrUpdater: AgentManagerState | Partial<AgentManagerState> | ((state: Draft<T>) => void), shouldReplace?: boolean | undefined): void;
}>;
/**
 * Composite selector for thread with full dependency info
 * Useful for rendering thread details with context
 */
export declare const useThreadWithDependencies: (threadId: string) => {
    thread: AgentThread | undefined;
    dependencies: {
        dependsOn: AgentThread[];
        blocks: AgentThread[];
    };
    children: AgentThread[];
};
/**
 * Composite selector for swarm overview
 * Combines swarm status with all member threads
 */
export declare const useSwarmOverview: (swarmId: string) => {
    threads: AgentThread[];
    status: {
        total: number;
        running: number;
        completed: number;
        failed: number;
        avgConfidence: number;
    };
};
/**
 * Hook for active thread with dependencies
 * Automatically updates when active thread changes
 */
export declare const useActiveThreadDetails: () => {
    thread: AgentThread | undefined;
    dependencies: {
        dependsOn: AgentThread[];
        blocks: AgentThread[];
    };
    children: AgentThread[];
};
export default useAgentManagerStore;
//# sourceMappingURL=agentManagerStore.d.ts.map