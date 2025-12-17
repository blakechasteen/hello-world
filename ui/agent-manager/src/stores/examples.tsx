/**
 * Agent Manager Store Usage Examples
 * Real-world examples of how to use the store in React components
 */

import { useCallback, useMemo } from 'react';
import {
  useAgentManagerStore,
  useThreadWithDependencies,
  useSwarmOverview,
  useActiveThreadDetails,
} from './index';
import type { AgentThread } from './index';

// ============================================================================
// EXAMPLE 1: Thread List Component with Filtering
// ============================================================================

export function ThreadList() {
  // Subscribe to filtered threads and filter action
  const threads = useAgentManagerStore((state) => state.getFilteredThreads());
  const filter = useAgentManagerStore((state) => state.filter);
  const setFilter = useAgentManagerStore((state) => state.setFilter);
  const setActiveThread = useAgentManagerStore((state) => state.setActiveThread);

  return (
    <div className="thread-list">
      <div className="filter-bar">
        <button
          className={filter === 'all' ? 'active' : ''}
          onClick={() => setFilter('all')}
        >
          All ({Object.keys(useAgentManagerStore.getState().threads).length})
        </button>
        <button
          className={filter === 'active' ? 'active' : ''}
          onClick={() => setFilter('active')}
        >
          Active
        </button>
        <button
          className={filter === 'completed' ? 'active' : ''}
          onClick={() => setFilter('completed')}
        >
          Completed
        </button>
        <button
          className={filter === 'failed' ? 'active' : ''}
          onClick={() => setFilter('failed')}
        >
          Failed
        </button>
      </div>

      <ul className="thread-items">
        {threads.map((thread) => (
          <li
            key={thread.id}
            className="thread-item"
            onClick={() => setActiveThread(thread.id)}
          >
            <div className="thread-name">{thread.name}</div>
            <div className="thread-status">{thread.status}</div>
            <div className="thread-progress">
              {thread.currentStep} / {thread.totalSteps}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ============================================================================
// EXAMPLE 2: Thread Detail Card with Dependencies
// ============================================================================

export function ThreadDetailCard({ threadId }: { threadId: string }) {
  const { thread, dependencies, children } = useThreadWithDependencies(threadId);

  if (!thread) {
    return <div className="error">Thread not found</div>;
  }

  const progressPercent = (thread.currentStep / thread.totalSteps) * 100;

  return (
    <div className="thread-detail-card">
      <header>
        <h3>{thread.name}</h3>
        <span className={`status status-${thread.status}`}>{thread.status}</span>
      </header>

      <section className="metrics">
        <div className="metric">
          <label>Progress</label>
          <progress value={progressPercent} max={100} />
          <span>{thread.currentStep}/{thread.totalSteps} steps</span>
        </div>

        <div className="metric">
          <label>Confidence</label>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: `${thread.confidence * 100}%` }}
            />
          </div>
          <span>{(thread.confidence * 100).toFixed(1)}%</span>
        </div>

        <div className="metric">
          <label>Epistemic Confidence</label>
          <span>{(thread.epistemicConfidence * 100).toFixed(1)}%</span>
        </div>

        <div className="metric">
          <label>Tokens</label>
          <span>{thread.tokensUsed} / {thread.tokenBudget ?? '∞'}</span>
        </div>

        <div className="metric">
          <label>Time</label>
          <span>{(thread.elapsedTimeMs / 1000).toFixed(2)}s</span>
        </div>
      </section>

      <section className="dependencies">
        <h4>Dependencies</h4>
        {dependencies.dependsOn.length > 0 ? (
          <ul>
            {dependencies.dependsOn.map((dep) => (
              <li key={dep.id}>{dep.name}</li>
            ))}
          </ul>
        ) : (
          <p className="empty">No dependencies</p>
        )}
      </section>

      <section className="blocking">
        <h4>Blocking</h4>
        {dependencies.blocks.length > 0 ? (
          <ul>
            {dependencies.blocks.map((block) => (
              <li key={block.id}>{block.name}</li>
            ))}
          </ul>
        ) : (
          <p className="empty">Not blocking anything</p>
        )}
      </section>

      <section className="children">
        <h4>Child Threads ({children.length})</h4>
        {children.length > 0 ? (
          <ul>
            {children.map((child) => (
              <li key={child.id}>{child.name}</li>
            ))}
          </ul>
        ) : (
          <p className="empty">No child threads</p>
        )}
      </section>
    </div>
  );
}

// ============================================================================
// EXAMPLE 3: Active Thread Panel with Controls
// ============================================================================

export function ActiveThreadPanel() {
  const { thread, dependencies, children } = useActiveThreadDetails();
  const pauseThread = useAgentManagerStore((state) => state.pauseThread);
  const resumeThread = useAgentManagerStore((state) => state.resumeThread);
  const cancelThread = useAgentManagerStore((state) => state.cancelThread);
  const upvoteThread = useAgentManagerStore((state) => state.upvoteThread);
  const downvoteThread = useAgentManagerStore((state) => state.downvoteThread);

  if (!thread) {
    return (
      <div className="empty-state">
        <p>Select a thread to view details</p>
      </div>
    );
  }

  const handlePauseResume = useCallback(() => {
    if (thread.status === 'running') {
      pauseThread(thread.id);
    } else if (thread.status === 'paused') {
      resumeThread(thread.id);
    }
  }, [thread.id, thread.status, pauseThread, resumeThread]);

  const handleCancel = useCallback(() => {
    if (confirm(`Cancel thread "${thread.name}"?`)) {
      cancelThread(thread.id);
    }
  }, [thread.id, thread.name, cancelThread]);

  return (
    <div className="active-thread-panel">
      <header>
        <h2>{thread.name}</h2>
        <div className="controls">
          {(thread.status === 'running' || thread.status === 'paused') && (
            <>
              <button onClick={handlePauseResume}>
                {thread.status === 'running' ? 'Pause' : 'Resume'}
              </button>
              <button onClick={handleCancel} className="danger">
                Cancel
              </button>
            </>
          )}
        </div>
      </header>

      <section className="priority-control">
        <label>Priority: {thread.priority}</label>
        <div className="priority-buttons">
          <button onClick={() => downvoteThread(thread.id)}>−</button>
          <div className="priority-bar">
            <div
              className="priority-fill"
              style={{ width: `${(thread.priority / 100) * 100}%` }}
            />
          </div>
          <button onClick={() => upvoteThread(thread.id)}>+</button>
        </div>
      </section>

      <section className="info">
        <div className="info-row">
          <span className="label">Type:</span>
          <span className="value">{thread.agentType}</span>
        </div>
        <div className="info-row">
          <span className="label">Mode:</span>
          <span className="value">{thread.reasoningMode}</span>
        </div>
        <div className="info-row">
          <span className="label">Status:</span>
          <span className={`status status-${thread.status}`}>{thread.status}</span>
        </div>
        <div className="info-row">
          <span className="label">Progress:</span>
          <span className="value">
            {thread.currentStep} / {thread.totalSteps}
          </span>
        </div>
        <div className="info-row">
          <span className="label">Confidence:</span>
          <span className="value">{(thread.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="info-row">
          <span className="label">Epistemic:</span>
          <span className="value">
            {(thread.epistemicConfidence * 100).toFixed(1)}%
          </span>
        </div>
      </section>

      {thread.finalResponse && (
        <section className="response">
          <h3>Response</h3>
          <p>{thread.finalResponse}</p>
        </section>
      )}
    </div>
  );
}

// ============================================================================
// EXAMPLE 4: Swarm Dashboard
// ============================================================================

export function SwarmDashboard({ swarmId }: { swarmId: string }) {
  const { threads, status } = useSwarmOverview(swarmId);

  const threadsByStatus = useMemo(
    () => ({
      running: threads.filter((t) => t.status === 'running').length,
      paused: threads.filter((t) => t.status === 'paused').length,
      completed: threads.filter((t) => t.status === 'completed').length,
      failed: threads.filter((t) => t.status === 'failed' || t.status === 'cancelled')
        .length,
    }),
    [threads]
  );

  const totalTokensUsed = useMemo(
    () => threads.reduce((sum, t) => sum + t.tokensUsed, 0),
    [threads]
  );

  const totalTimeMs = useMemo(
    () => Math.max(...threads.map((t) => t.elapsedTimeMs), 0),
    [threads]
  );

  return (
    <div className="swarm-dashboard">
      <h1>Swarm: {swarmId}</h1>

      <section className="summary">
        <div className="stat">
          <label>Total Threads</label>
          <div className="value">{status.total}</div>
        </div>
        <div className="stat">
          <label>Running</label>
          <div className="value">{status.running}</div>
        </div>
        <div className="stat">
          <label>Completed</label>
          <div className="value">{status.completed}</div>
        </div>
        <div className="stat">
          <label>Failed</label>
          <div className="value">{status.failed}</div>
        </div>
        <div className="stat">
          <label>Avg Confidence</label>
          <div className="value">{(status.avgConfidence * 100).toFixed(1)}%</div>
        </div>
      </section>

      <section className="thread-breakdown">
        <h3>Thread Status</h3>
        <div className="status-grid">
          <div className="status-item running">
            <span className="count">{threadsByStatus.running}</span>
            <span className="label">Running</span>
          </div>
          <div className="status-item paused">
            <span className="count">{threadsByStatus.paused}</span>
            <span className="label">Paused</span>
          </div>
          <div className="status-item completed">
            <span className="count">{threadsByStatus.completed}</span>
            <span className="label">Completed</span>
          </div>
          <div className="status-item failed">
            <span className="count">{threadsByStatus.failed}</span>
            <span className="label">Failed</span>
          </div>
        </div>
      </section>

      <section className="resource-usage">
        <h3>Resource Usage</h3>
        <div className="resource-row">
          <label>Total Tokens</label>
          <span>{totalTokensUsed.toLocaleString()}</span>
        </div>
        <div className="resource-row">
          <label>Total Time</label>
          <span>{(totalTimeMs / 1000).toFixed(2)}s</span>
        </div>
      </section>

      <section className="thread-list">
        <h3>Threads ({threads.length})</h3>
        <ul>
          {threads.map((thread) => (
            <li key={thread.id} className={`thread-row status-${thread.status}`}>
              <span className="name">{thread.name}</span>
              <span className="status">{thread.status}</span>
              <span className="progress">
                {thread.currentStep}/{thread.totalSteps}
              </span>
              <span className="confidence">
                {(thread.confidence * 100).toFixed(0)}%
              </span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

// ============================================================================
// EXAMPLE 5: Thread Manager with Updates
// ============================================================================

export function ThreadManager() {
  const updateThread = useAgentManagerStore((state) => state.updateThread);
  const removeThread = useAgentManagerStore((state) => state.removeThread);

  const handleUpdateProgress = useCallback(
    (threadId: string, step: number, elapsedMs: number) => {
      updateThread(threadId, {
        currentStep: step,
        elapsedTimeMs: elapsedMs,
      });
    },
    [updateThread]
  );

  const handleComplete = useCallback(
    (threadId: string, response: string, confidence: number) => {
      updateThread(threadId, {
        status: 'completed',
        finalResponse: response,
        confidence,
      });
    },
    [updateThread]
  );

  const handleFail = useCallback(
    (threadId: string, error: string) => {
      updateThread(threadId, {
        status: 'failed',
        finalResponse: `Error: ${error}`,
      });
    },
    [updateThread]
  );

  return (
    <div className="thread-manager">
      <p>Thread Manager - Used to update threads via WebSocket/API callbacks</p>
      <code>
        {JSON.stringify(
          {
            onUpdateProgress: 'handleUpdateProgress(threadId, step, time)',
            onComplete: 'handleComplete(threadId, response, confidence)',
            onFail: 'handleFail(threadId, error)',
          },
          null,
          2
        )}
      </code>
    </div>
  );
}

// ============================================================================
// EXAMPLE 6: Connection Status Indicator
// ============================================================================

export function ConnectionIndicator() {
  const isConnected = useAgentManagerStore((state) => state.isConnected);
  const connectionError = useAgentManagerStore((state) => state.connectionError);

  return (
    <div className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
      <div className="status-dot" />
      <span className="status-text">
        {isConnected ? 'Connected' : `Disconnected: ${connectionError}`}
      </span>
    </div>
  );
}

// ============================================================================
// EXAMPLE 7: View Mode Switcher
// ============================================================================

export function ViewModeSwitcher() {
  const viewMode = useAgentManagerStore((state) => state.viewMode);
  const setViewMode = useAgentManagerStore((state) => state.setViewMode);

  return (
    <div className="view-mode-switcher">
      <button
        className={viewMode === 'outline' ? 'active' : ''}
        onClick={() => setViewMode('outline')}
      >
        Outline
      </button>
      <button
        className={viewMode === 'tree' ? 'active' : ''}
        onClick={() => setViewMode('tree')}
      >
        Tree
      </button>
      <button
        className={viewMode === 'swarm' ? 'active' : ''}
        onClick={() => setViewMode('swarm')}
      >
        Swarm
      </button>
    </div>
  );
}
