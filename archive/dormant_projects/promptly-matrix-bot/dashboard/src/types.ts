/**
 * Type definitions for Promptly Dashboard
 */

// Weaving Step Status
export enum StepStatus {
  WAITING = 'waiting',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  ERROR = 'error',
}

// HoloLoom 9-Step Weaving Cycle
export interface WeavingStep {
  step: number;
  name: string;
  status: StepStatus;
  latency_ms?: number;
  metadata?: Record<string, any>;
  error?: string;
}

export interface WeavingCycle {
  query_id: string;
  query_text: string;
  timestamp: string;
  steps: WeavingStep[];
  total_latency_ms: number;
  confidence: number;
  tool_used?: string;
  response?: string;
}

// Knowledge Graph
export interface GraphNode {
  id: string;
  label: string;
  type: 'entity' | 'motif' | 'concept';
  connections: number;
  metadata?: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: 'IS_A' | 'USES' | 'MENTIONS' | 'LEADS_TO' | 'PART_OF';
  weight: number;
  metadata?: Record<string, any>;
}

export interface KnowledgeGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// Audit Trail
export enum EventType {
  COMMAND = 'command',
  DECISION = 'decision',
  APPROVAL = 'approval',
  WORKFLOW = 'workflow',
  ACCESS = 'access',
  ERROR = 'error',
  CONFIG_CHANGE = 'config',
  SYSTEM = 'system',
}

export enum Outcome {
  SUCCESS = 'success',
  FAILURE = 'failure',
  PENDING = 'pending',
  CANCELLED = 'cancelled',
}

export interface AuditEvent {
  event_id: string;
  timestamp: string;
  event_type: EventType;
  user: string;
  room?: string;
  action: string;
  context: Record<string, any>;
  outcome: Outcome;
  metadata?: Record<string, any>;
  error_message?: string;
}

// Team Context
export enum ContextScope {
  TEAM = 'team',
  ROOM = 'room',
  USER = 'user',
}

export enum Permission {
  READ = 'read',
  WRITE = 'write',
  ADMIN = 'admin',
}

export interface SharedPrompt {
  name: string;
  prompt: string;
  scope: ContextScope;
  created_by: string;
  created_at: string;
  version: number;
  tags: string[];
  permissions: Record<string, Permission>;
  usage_count?: number;
  success_rate?: number;
}

// Workflow
export interface WorkflowNode {
  id: string;
  type: 'optimize' | 'verify' | 'refine' | 'approval' | 'deploy' | 'conditional' | 'loop';
  label: string;
  config: Record<string, any>;
  position: { x: number; y: number };
}

export interface WorkflowConnection {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

export interface Workflow {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  created_by: string;
  created_at: string;
  scope: ContextScope;
}

// WebSocket Events
export interface WSMessage {
  type: 'weaving_update' | 'audit_event' | 'graph_update' | 'error';
  data: any;
  timestamp: string;
}

// API Responses
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Dashboard Stats
export interface DashboardStats {
  total_queries: number;
  avg_latency_ms: number;
  avg_confidence: number;
  cache_hit_rate: number;
  active_users: number;
  memory_shards: number;
  most_used_tools: Record<string, number>;
  recent_queries: WeavingCycle[];
}
