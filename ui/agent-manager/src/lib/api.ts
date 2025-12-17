/**
 * API Client for Agent Manager
 *
 * Provides typed REST API calls to the Agent Manager backend.
 * Handles injection endpoints for MRF strategy and MCTS configuration.
 *
 * Date: 2025-12-15
 */

// ============================================================================
// Types
// ============================================================================

export interface MCTSConfig {
  budget: number;
  exploration: number;
}

export interface InjectMRFRequest {
  strategy: string;
}

export interface InjectMCTSRequest {
  budget: number;
  exploration_c: number;
}

export interface InjectionResponse {
  success: boolean;
  message: string;
  thread_id: string;
  step_id: string;
  injection_type: 'mrf' | 'mcts';
  strategy?: string;
  config?: MCTSConfig;
  timestamp: string;
}

export interface InjectionHistoryEntry {
  id: string;
  threadId: string;
  stepId: string;
  injectionType: 'mrf' | 'mcts';
  strategy?: string;
  config?: MCTSConfig;
  timestamp: string;
  success: boolean;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public details?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL =
  typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.host}`
    : 'http://localhost:8002';

// ============================================================================
// Helper Functions
// ============================================================================

async function fetchJson<T>(
  url: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    let details;
    try {
      details = await response.json();
    } catch {
      details = await response.text();
    }
    throw new ApiError(
      `API request failed: ${response.statusText}`,
      response.status,
      details
    );
  }

  return response.json();
}

// ============================================================================
// Injection API
// ============================================================================

/**
 * Inject MRF refinement strategy at a specific step
 *
 * @param threadId - The thread ID
 * @param stepId - The step ID to inject at
 * @param strategy - MRF strategy: 'auto', 'verify', 'elegance', 'critique', 'refine', 'hofstadter'
 * @returns Injection result
 */
export async function injectMRF(
  threadId: string,
  stepId: string,
  strategy: string
): Promise<InjectionResponse> {
  const url = `${API_BASE_URL}/api/threads/${threadId}/steps/${stepId}/inject/mrf`;
  const body: InjectMRFRequest = { strategy };

  return fetchJson<InjectionResponse>(url, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/**
 * Inject MCTS configuration at a specific step
 *
 * @param threadId - The thread ID
 * @param stepId - The step ID to inject at
 * @param config - MCTS configuration with budget and exploration coefficient
 * @returns Injection result
 */
export async function injectMCTS(
  threadId: string,
  stepId: string,
  config: MCTSConfig
): Promise<InjectionResponse> {
  const url = `${API_BASE_URL}/api/threads/${threadId}/steps/${stepId}/inject/mcts`;
  const body: InjectMCTSRequest = {
    budget: config.budget,
    exploration_c: config.exploration,
  };

  return fetchJson<InjectionResponse>(url, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

/**
 * Get injection history for a thread
 *
 * @param threadId - The thread ID
 * @returns List of injection history entries
 */
export async function getInjectionHistory(
  threadId: string
): Promise<InjectionHistoryEntry[]> {
  const url = `${API_BASE_URL}/api/threads/${threadId}/injections`;

  try {
    return fetchJson<InjectionHistoryEntry[]>(url, { method: 'GET' });
  } catch (error) {
    // Return empty array if endpoint doesn't exist yet
    if (error instanceof ApiError && error.status === 404) {
      return [];
    }
    throw error;
  }
}

// ============================================================================
// Thread Control API
// ============================================================================

/**
 * Pause a running thread
 */
export async function pauseThread(threadId: string): Promise<void> {
  const url = `${API_BASE_URL}/api/threads/${threadId}/pause`;
  await fetchJson(url, { method: 'POST' });
}

/**
 * Resume a paused thread
 */
export async function resumeThread(threadId: string): Promise<void> {
  const url = `${API_BASE_URL}/api/threads/${threadId}/resume`;
  await fetchJson(url, { method: 'POST' });
}

/**
 * Cancel a thread
 */
export async function cancelThread(threadId: string): Promise<void> {
  const url = `${API_BASE_URL}/api/threads/${threadId}/cancel`;
  await fetchJson(url, { method: 'POST' });
}

/**
 * Update thread priority
 */
export async function updateThreadPriority(
  threadId: string,
  priority: number
): Promise<void> {
  const url = `${API_BASE_URL}/api/threads/${threadId}/priority`;
  await fetchJson(url, {
    method: 'PATCH',
    body: JSON.stringify({ priority }),
  });
}

// ============================================================================
// Export convenience object
// ============================================================================

export const api = {
  injectMRF,
  injectMCTS,
  getInjectionHistory,
  pauseThread,
  resumeThread,
  cancelThread,
  updateThreadPriority,
};

export default api;
