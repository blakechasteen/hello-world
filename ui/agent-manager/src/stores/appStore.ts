import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'
import type {
  Agent,
  Task,
  Message,
  SystemMetrics,
  LogEntry,
  DashboardView,
} from '../types'

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
  // Core Data
  agents: Agent[]
  tasks: Task[]
  messages: Message[]
  logs: LogEntry[]

  // System State
  systemMetrics: SystemMetrics | null
  isConnected: boolean
  currentView: DashboardView

  // UI State
  selectedAgentId: string | null
  sidebarOpen: boolean
  searchQuery: string
  filterState: string | null
  autoRefresh: boolean
  refreshInterval: number

  // Actions
  setAgents: (agents: Agent[]) => void
  updateAgent: (id: string, updates: Partial<Agent>) => void
  addTask: (task: Task) => void
  updateTask: (id: string, updates: Partial<Task>) => void
  removeTask: (id: string) => void
  addMessage: (message: Message) => void
  addLog: (source: string, message: string, level: LogEntry['level']) => void
  clearLogs: () => void
  setSystemMetrics: (metrics: SystemMetrics) => void
  setConnected: (connected: boolean) => void
  setCurrentView: (view: DashboardView) => void
  setSelectedAgent: (id: string | null) => void
  setSidebarOpen: (open: boolean) => void
  setSearchQuery: (query: string) => void
  setFilterState: (state: string | null) => void
  setAutoRefresh: (enabled: boolean) => void
  setRefreshInterval: (interval: number) => void
  reset: () => void
}

const initialState = {
  agents: [],
  tasks: [],
  messages: [],
  logs: [],
  systemMetrics: null,
  isConnected: false,
  currentView: 'overview' as DashboardView,
  selectedAgentId: null,
  sidebarOpen: true,
  searchQuery: '',
  filterState: null,
  autoRefresh: true,
  refreshInterval: 5000,
}

export const useAppStore = create<AppStoreState>()(
  subscribeWithSelector((set) => ({
    ...initialState,

    setAgents: (agents) => set({ agents }),

    updateAgent: (id, updates) =>
      set((state) => ({
        agents: state.agents.map((agent) =>
          agent.id === id ? { ...agent, ...updates } : agent
        ),
      })),

    addTask: (task) =>
      set((state) => ({
        tasks: [task, ...state.tasks],
      })),

    updateTask: (id, updates) =>
      set((state) => ({
        tasks: state.tasks.map((task) =>
          task.id === id ? { ...task, ...updates } : task
        ),
      })),

    removeTask: (id) =>
      set((state) => ({
        tasks: state.tasks.filter((task) => task.id !== id),
      })),

    addMessage: (message) =>
      set((state) => ({
        messages: [message, ...state.messages].slice(0, 1000), // Keep last 1000
      })),

    addLog: (source, message, level) =>
      set((state) => {
        const logEntry: LogEntry = {
          id: `${Date.now()}-${Math.random()}`,
          timestamp: Date.now(),
          source,
          message,
          level,
        }
        return {
          logs: [logEntry, ...state.logs].slice(0, 500), // Keep last 500 logs
        }
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
  }))
)

// Selector hooks for performance optimization
export const useAgents = () => useAppStore((state) => state.agents)
export const useTasks = () => useAppStore((state) => state.tasks)
export const useMessages = () => useAppStore((state) => state.messages)
export const useLogs = () => useAppStore((state) => state.logs)
export const useSystemMetrics = () => useAppStore((state) => state.systemMetrics)
export const useIsConnected = () => useAppStore((state) => state.isConnected)
export const useCurrentView = () => useAppStore((state) => state.currentView)
export const useSelectedAgent = () =>
  useAppStore((state) => state.selectedAgentId)
export const useSidebarOpen = () => useAppStore((state) => state.sidebarOpen)
export const useAutoRefresh = () => useAppStore((state) => state.autoRefresh)
export const useRefreshInterval = () =>
  useAppStore((state) => state.refreshInterval)

// Derived selectors
export const useSelectedAgentData = () =>
  useAppStore((state) => {
    const agentId = state.selectedAgentId
    return agentId ? state.agents.find((a) => a.id === agentId) : null
  })

export const useRunningAgents = () =>
  useAppStore((state) =>
    state.agents.filter((agent) => agent.state === 'running')
  )

export const useAgentsByType = (type: string) =>
  useAppStore((state) => state.agents.filter((agent) => agent.type === type))

export const usePendingTasks = () =>
  useAppStore((state) => state.tasks.filter((task) => task.status === 'pending'))

export const useRecentLogs = (limit: number = 20) =>
  useAppStore((state) => state.logs.slice(0, limit))

export const useErrorLogs = () =>
  useAppStore((state) => state.logs.filter((log) => log.level === 'error'))
