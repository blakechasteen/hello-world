import type { Agent, Task, Message, SystemMetrics, LogEntry, DashboardView } from '../types';
/**
 * Application Global Store
 *
 * Manages:
 * - Agent state and metrics
 * - Task queue and execution
 * - System-wide metrics
 * - Logging and notifications
 * - UI state (selected agent, view mode, filters)
 *
 * Uses Zustand with middleware for:
 * - Selector subscriptions (useShallow for partial updates)
 * - Immutability patterns
 * - DevTools integration (in development)
 */
interface AppStoreState {
    agents: Agent[];
    tasks: Task[];
    messages: Message[];
    logs: LogEntry[];
    systemMetrics: SystemMetrics | null;
    isConnected: boolean;
    currentView: DashboardView;
    selectedAgentId: string | null;
    sidebarOpen: boolean;
    searchQuery: string;
    filterState: string | null;
    autoRefresh: boolean;
    refreshInterval: number;
    setAgents: (agents: Agent[]) => void;
    updateAgent: (id: string, updates: Partial<Agent>) => void;
    addTask: (task: Task) => void;
    updateTask: (id: string, updates: Partial<Task>) => void;
    removeTask: (id: string) => void;
    addMessage: (message: Message) => void;
    addLog: (source: string, message: string, level: LogEntry['level']) => void;
    clearLogs: () => void;
    setSystemMetrics: (metrics: SystemMetrics) => void;
    setConnected: (connected: boolean) => void;
    setCurrentView: (view: DashboardView) => void;
    setSelectedAgent: (id: string | null) => void;
    setSidebarOpen: (open: boolean) => void;
    setSearchQuery: (query: string) => void;
    setFilterState: (state: string | null) => void;
    setAutoRefresh: (enabled: boolean) => void;
    setRefreshInterval: (interval: number) => void;
    reset: () => void;
}
export declare const useAppStore: import("zustand").UseBoundStore<Omit<import("zustand").StoreApi<AppStoreState>, "subscribe"> & {
    subscribe: {
        (listener: (selectedState: AppStoreState, previousSelectedState: AppStoreState) => void): () => void;
        <U>(selector: (state: AppStoreState) => U, listener: (selectedState: U, previousSelectedState: U) => void, options?: {
            equalityFn?: ((a: U, b: U) => boolean) | undefined;
            fireImmediately?: boolean;
        } | undefined): () => void;
    };
}>;
export declare const useAgents: () => Agent[];
export declare const useTasks: () => Task[];
export declare const useMessages: () => Message[];
export declare const useLogs: () => LogEntry[];
export declare const useSystemMetrics: () => SystemMetrics | null;
export declare const useIsConnected: () => boolean;
export declare const useCurrentView: () => DashboardView;
export declare const useSelectedAgent: () => string | null;
export declare const useSidebarOpen: () => boolean;
export declare const useAutoRefresh: () => boolean;
export declare const useRefreshInterval: () => number;
export declare const useSelectedAgentData: () => Agent | null | undefined;
export declare const useRunningAgents: () => Agent[];
export declare const useAgentsByType: (type: string) => Agent[];
export declare const usePendingTasks: () => Task[];
export declare const useRecentLogs: (limit?: number) => LogEntry[];
export declare const useErrorLogs: () => LogEntry[];
export {};
//# sourceMappingURL=appStore.d.ts.map