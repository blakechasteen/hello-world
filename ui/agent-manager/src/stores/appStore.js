import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
const initialState = {
    agents: [],
    tasks: [],
    messages: [],
    logs: [],
    systemMetrics: null,
    isConnected: false,
    currentView: 'overview',
    selectedAgentId: null,
    sidebarOpen: true,
    searchQuery: '',
    filterState: null,
    autoRefresh: true,
    refreshInterval: 5000,
};
export const useAppStore = create()(subscribeWithSelector((set) => ({
    ...initialState,
    setAgents: (agents) => set({ agents }),
    updateAgent: (id, updates) => set((state) => ({
        agents: state.agents.map((agent) => agent.id === id ? { ...agent, ...updates } : agent),
    })),
    addTask: (task) => set((state) => ({
        tasks: [task, ...state.tasks],
    })),
    updateTask: (id, updates) => set((state) => ({
        tasks: state.tasks.map((task) => task.id === id ? { ...task, ...updates } : task),
    })),
    removeTask: (id) => set((state) => ({
        tasks: state.tasks.filter((task) => task.id !== id),
    })),
    addMessage: (message) => set((state) => ({
        messages: [message, ...state.messages].slice(0, 1000), // Keep last 1000
    })),
    addLog: (source, message, level) => set((state) => {
        const logEntry = {
            id: `${Date.now()}-${Math.random()}`,
            timestamp: Date.now(),
            source,
            message,
            level,
        };
        return {
            logs: [logEntry, ...state.logs].slice(0, 500), // Keep last 500 logs
        };
    }),
    clearLogs: () => set({ logs: [] }),
    setSystemMetrics: (metrics) => set({ systemMetrics: metrics }),
    setConnected: (connected) => set({ isConnected: connected }),
    setCurrentView: (view) => set({ currentView: view }),
    setSelectedAgent: (id) => set({ selectedAgentId: id }),
    setSidebarOpen: (open) => set({ sidebarOpen: open }),
    setSearchQuery: (query) => set({ searchQuery: query }),
    setFilterState: (state) => set({ filterState: state }),
    setAutoRefresh: (enabled) => set({ autoRefresh: enabled }),
    setRefreshInterval: (interval) => set({ refreshInterval: interval }),
    reset: () => set(initialState),
})));
// Selector hooks for performance optimization
export const useAgents = () => useAppStore((state) => state.agents);
export const useTasks = () => useAppStore((state) => state.tasks);
export const useMessages = () => useAppStore((state) => state.messages);
export const useLogs = () => useAppStore((state) => state.logs);
export const useSystemMetrics = () => useAppStore((state) => state.systemMetrics);
export const useIsConnected = () => useAppStore((state) => state.isConnected);
export const useCurrentView = () => useAppStore((state) => state.currentView);
export const useSelectedAgent = () => useAppStore((state) => state.selectedAgentId);
export const useSidebarOpen = () => useAppStore((state) => state.sidebarOpen);
export const useAutoRefresh = () => useAppStore((state) => state.autoRefresh);
export const useRefreshInterval = () => useAppStore((state) => state.refreshInterval);
// Derived selectors
export const useSelectedAgentData = () => useAppStore((state) => {
    const agentId = state.selectedAgentId;
    return agentId ? state.agents.find((a) => a.id === agentId) : null;
});
export const useRunningAgents = () => useAppStore((state) => state.agents.filter((agent) => agent.state === 'running'));
export const useAgentsByType = (type) => useAppStore((state) => state.agents.filter((agent) => agent.type === type));
export const usePendingTasks = () => useAppStore((state) => state.tasks.filter((task) => task.status === 'pending'));
export const useRecentLogs = (limit = 20) => useAppStore((state) => state.logs.slice(0, limit));
export const useErrorLogs = () => useAppStore((state) => state.logs.filter((log) => log.level === 'error'));
//# sourceMappingURL=appStore.js.map