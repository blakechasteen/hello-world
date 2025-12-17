/**
 * Core Types for Agent Manager UI
 *
 * These types correspond to HoloLoom's agentic reasoning system:
 * - Agent: Individual reasoning unit in the swarm
 * - Task: Work unit assigned to agents
 * - Message: Inter-agent communication
 * - Metrics: Performance tracking for agents and system
 */

/** Agent State Enum */
export enum AgentState {
  IDLE = 'idle',
  READY = 'ready',
  RUNNING = 'running',
  SUCCESS = 'success',
  WARNING = 'warning',
  ERROR = 'error',
  PAUSED = 'paused',
}

/** Agent Type */
export enum AgentType {
  QUERY = 'query',
  PROCESS = 'process',
  MEMORY = 'memory',
  DECISION = 'decision',
  OUTPUT = 'output',
  CONTROL = 'control',
}

/** Reasoning Mode for Agents */
export enum ReasoningMode {
  DIRECT = 'direct',
  VERIFY = 'verify',
  RESEARCH = 'research',
  PLAN_EXECUTE = 'plan_execute',
}

/** Agent Configuration */
export interface Agent {
  id: string
  name: string
  type: AgentType
  state: AgentState
  reasoningMode: ReasoningMode
  confidence: number // 0.0-1.0
  metrics: AgentMetrics
  lastUpdated: number // Unix timestamp
  description?: string
  icon?: string
}

/** Agent Metrics */
export interface AgentMetrics {
  queriesProcessed: number
  successCount: number
  errorCount: number
  avgLatencyMs: number
  totalTimeMs: number
  cacheHitRate: number // 0.0-1.0
  confidenceScore: number // 0.0-1.0
}

/** Task Assigned to Agents */
export interface Task {
  id: string
  agentId: string
  query: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  startTime: number
  endTime?: number
  result?: Record<string, any>
  error?: string
}

/** Inter-Agent Message */
export interface Message {
  id: string
  from: string
  to: string
  content: string
  timestamp: number
  type: 'query' | 'response' | 'error' | 'info'
}

/** System-wide Metrics */
export interface SystemMetrics {
  agentsTotal: number
  agentsRunning: number
  agentsError: number
  avgConfidence: number
  totalQueriesProcessed: number
  systemLatencyMs: number
  cacheHitRate: number
  uptime: number // seconds
}

/** Application Log Entry */
export interface LogEntry {
  id: string
  timestamp: number
  source: string
  message: string
  level: 'info' | 'warning' | 'error' | 'success'
  metadata?: Record<string, any>
}

/** WebSocket Message Types from Backend */
export interface WebSocketMessage {
  type: 'agent_update' | 'task_complete' | 'error' | 'metrics' | 'message'
  payload: Record<string, any>
  timestamp: number
}

/** Application State */
export interface AppState {
  agents: Agent[]
  tasks: Task[]
  messages: Message[]
  systemMetrics: SystemMetrics
  logs: LogEntry[]
  isConnected: boolean
  selectedAgentId?: string
  lastSyncTime: number
}

/** Dashboard View Type */
export enum DashboardView {
  OVERVIEW = 'overview',
  AGENTS = 'agents',
  TASKS = 'tasks',
  METRICS = 'metrics',
  LOGS = 'logs',
  SETTINGS = 'settings',
}

/** Notification Type */
export interface Notification {
  id: string
  title: string
  message: string
  type: 'info' | 'success' | 'warning' | 'error'
  duration?: number // ms, undefined = persistent
  timestamp: number
}

/** API Response Wrapper */
export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
  timestamp: number
}

/** Pagination Info */
export interface Pagination {
  page: number
  pageSize: number
  total: number
  hasMore: boolean
}

/** Filter Options for Logs/Tasks */
export interface FilterOptions {
  agentId?: string
  state?: AgentState
  type?: AgentType
  startTime?: number
  endTime?: number
  searchQuery?: string
  limit?: number
  offset?: number
}

/** Agent Capability */
export interface AgentCapability {
  name: string
  description: string
  icon: string
  inputs: string[]
  outputs: string[]
}

/** Agent Details (Extended) */
export interface AgentDetails extends Agent {
  capabilities: AgentCapability[]
  dependencies: string[] // Agent IDs this depends on
  parameters: Record<string, any>
  history: {
    lastQuery?: string
    lastResult?: Record<string, any>
    lastError?: string
  }
}
