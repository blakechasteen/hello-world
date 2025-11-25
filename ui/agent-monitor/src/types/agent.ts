export interface AgentSession {
  agent_id: string;
  project: string;
  query: string;
  mode: string;
  status: 'running' | 'completed' | 'failed' | 'waiting' | 'verify' | 'research';
  feed_line1: string;
  feed_line2: string;
  current_step: number;
  total_steps: number;
  files: string[];
  start_time: string;
  total_duration_ms?: number;
  tree?: TreeNode;
}

export interface TreeNode {
  node_id: string;
  step_type: string;
  query?: string;
  finding?: string;
  confidence?: number;
  epistemic_confidence?: number;
  children?: TreeNode[];
}

export interface Metrics {
  total_agents_started: number;
  total_agents_completed: number;
  total_agents_failed: number;
  active_agents: number;
  avg_latency_ms: number;
  success_rate: number;
  projects: string[];
  ws_connections: number;
}

export interface WebSocketMessage {
  type: string;
  agent_id?: string;
  project?: string;
  query?: string;
  mode?: string;
  status?: string;
  line1?: string;
  line2?: string;
  step?: number;
  total_steps?: number;
  total_duration_ms?: number;
  error?: string;
  timestamp?: string;
  files?: string[];
}
