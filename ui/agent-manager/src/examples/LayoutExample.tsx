/**
 * LayoutExample.tsx
 *
 * Complete example showing how to use all Phase 1 layout components together.
 * This file demonstrates:
 * - Component composition
 * - Zustand store integration
 * - Status badge variations
 * - Responsive behavior
 *
 * To use: Import Layout in your App.tsx and render it
 */

import React from 'react';
import { Layout } from '@components';
import { useAgentManagerStore, AgentThread } from '@stores/agentManagerStore';

/**
 * Example: Full Application with Layout
 *
 * This is the simplest way to use all Phase 1 components:
 */
export function LayoutExample() {
  return <Layout />;
}

/**
 * Example: Custom integration with Zustand hooks
 *
 * Shows how to build additional components that work with the store
 */
export function CustomThreadListExample() {
  // Select only threads, not entire store (for performance)
  const threads = useAgentManagerStore((state) => state.getFilteredThreads());
  const activeThreadId = useAgentManagerStore((state) => state.activeThreadId);
  const setActiveThread = useAgentManagerStore((state) => state.setActiveThread);

  if (threads.length === 0) {
    return (
      <div className="p-8 text-center text-slate-400">
        <div className="text-6xl mb-4">≡</div>
        <p>No threads created yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 p-4">
      {threads.map((thread) => (
        <ThreadItemExample
          key={thread.id}
          thread={thread}
          isActive={thread.id === activeThreadId}
          onSelect={() => setActiveThread(thread.id)}
        />
      ))}
    </div>
  );
}

/**
 * Example: Individual thread item component
 */
function ThreadItemExample({
  thread,
  isActive,
  onSelect,
}: {
  thread: AgentThread;
  isActive: boolean;
  onSelect: () => void;
}) {
  // Import status badge
  const { StatusBadge } = require('@components');

  return (
    <button
      onClick={onSelect}
      className={`
        w-full text-left p-3 rounded-md transition-colors
        ${isActive ? 'bg-emerald-600 bg-opacity-20 border border-emerald-600' : 'bg-slate-800 hover:bg-slate-700 border border-slate-700'}
      `}
    >
      <div className="flex items-center justify-between gap-3">
        {/* Left: Status badge + Name */}
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <StatusBadge status={thread.status} size="sm" showLabel={false} />
          <div className="flex-1 min-w-0">
            <div className="font-medium text-slate-100 truncate">
              {thread.name}
            </div>
            <div className="text-xs text-slate-500">
              {thread.reasoningMode} • P{thread.priority}
            </div>
          </div>
        </div>

        {/* Right: Metrics */}
        <div className="text-right text-xs text-slate-400">
          <div>{Math.round(thread.elapsedTimeMs)}ms</div>
          <div>
            {thread.tokensUsed} / {thread.tokenBudget ?? '∞'}
          </div>
        </div>
      </div>
    </button>
  );
}

/**
 * Example: Dashboard overview with status grid
 */
export function DashboardOverviewExample() {
  const { StatusGrid } = require('@components');
  const threads = useAgentManagerStore((state) => Object.values(state.threads));

  // Count threads by status
  const statusCounts = {
    idle: threads.filter((t: AgentThread) => t.status === 'idle').length,
    running: threads.filter((t: AgentThread) => t.status === 'running').length,
    paused: threads.filter((t: AgentThread) => t.status === 'paused').length,
    completed: threads.filter((t: AgentThread) => t.status === 'completed').length,
    failed: threads.filter((t: AgentThread) => t.status === 'failed' || t.status === 'cancelled').length,
  };

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-slate-100 mb-2">
          Agent Swarm Dashboard
        </h1>
        <p className="text-slate-400">
          {threads.length} total threads across all swarms
        </p>
      </div>

      {/* Status Overview Grid */}
      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">Overview</h2>
        <StatusGrid
          idle={statusCounts.idle}
          running={statusCounts.running}
          paused={statusCounts.paused}
          completed={statusCounts.completed}
          failed={statusCounts.failed}
        />
      </div>

      {/* Thread List */}
      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">Recent Threads</h2>
        <CustomThreadListExample />
      </div>
    </div>
  );
}

/**
 * Example: Status badge variations
 *
 * Shows all available sizes and styles
 */
export function StatusBadgeVariationsExample() {
  const { StatusBadge, StatusIndicator } = require('@components');

  return (
    <div className="p-8 space-y-8">
      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          Status Badge Sizes
        </h2>
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <StatusBadge status="running" size="sm" />
            <span className="text-slate-500">size: sm</span>
          </div>
          <div className="flex items-center gap-4">
            <StatusBadge status="running" size="md" />
            <span className="text-slate-500">size: md (default)</span>
          </div>
          <div className="flex items-center gap-4">
            <StatusBadge status="running" size="lg" />
            <span className="text-slate-500">size: lg</span>
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          All Status Types
        </h2>
        <div className="grid grid-cols-2 gap-4">
          {(['idle', 'running', 'paused', 'completed', 'failed', 'cancelled'] as const).map(
            (status) => (
              <div key={status} className="flex items-center gap-3">
                <StatusBadge status={status} size="md" />
                <span className="text-slate-500 text-sm">{status}</span>
              </div>
            )
          )}
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          Compact Indicators
        </h2>
        <div className="flex gap-4">
          {(['idle', 'running', 'paused', 'completed', 'failed', 'cancelled'] as const).map(
            (status) => (
              <div key={status} className="flex flex-col items-center gap-2">
                <StatusIndicator status={status} size="md" />
                <span className="text-slate-500 text-xs">{status}</span>
              </div>
            )
          )}
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          Icon Only (showLabel=false)
        </h2>
        <div className="flex gap-3">
          {(['idle', 'running', 'paused', 'completed', 'failed', 'cancelled'] as const).map(
            (status) => (
              <StatusBadge
                key={status}
                status={status}
                size="md"
                showLabel={false}
              />
            )
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Example: Real-time connection status
 *
 * Shows how connection status affects the header indicator
 */
export function ConnectionStatusExample() {
  const { isConnected, connectionError, setConnectionStatus } =
    useAgentManagerStore();

  const toggleConnection = () => {
    if (isConnected) {
      setConnectionStatus(false, 'Simulated disconnect');
    } else {
      setConnectionStatus(true);
    }
  };

  return (
    <div className="p-8 space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          Connection Status
        </h2>
        <div className="space-y-2">
          <p className="text-slate-400">
            Current: {isConnected ? 'Connected' : 'Offline'}
          </p>
          {connectionError && (
            <p className="text-red-400">Error: {connectionError}</p>
          )}
          <button
            onClick={toggleConnection}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-md font-medium transition-colors"
          >
            {isConnected ? 'Disconnect' : 'Reconnect'}
          </button>
        </div>
      </div>

      <div>
        <p className="text-slate-500 text-sm">
          Note: The header shows a green pulsing dot when connected, red dot when offline
        </p>
      </div>
    </div>
  );
}

/**
 * Example: Filter and view mode switching
 *
 * Shows how to interact with filter and viewMode store
 */
export function FilterAndViewExample() {
  const { filter, setFilter, viewMode, setViewMode } = useAgentManagerStore();

  return (
    <div className="p-8 space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          Filter Control
        </h2>
        <div className="flex gap-2">
          {(['all', 'active', 'completed', 'failed'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`
                px-3 py-1.5 rounded-md text-sm font-medium transition-colors
                ${
                  filter === f
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }
              `}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
        <p className="text-slate-500 text-sm mt-2">
          Current filter: <span className="font-mono">{filter}</span>
        </p>
      </div>

      <div>
        <h2 className="text-lg font-semibold text-slate-200 mb-4">
          View Mode Control
        </h2>
        <div className="flex gap-2">
          {(['outline', 'tree', 'swarm'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={`
                px-3 py-1.5 rounded-md text-sm font-medium transition-colors
                ${
                  viewMode === mode
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }
              `}
            >
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>
        <p className="text-slate-500 text-sm mt-2">
          Current view: <span className="font-mono">{viewMode}</span>
        </p>
      </div>
    </div>
  );
}

/**
 * Main example component that shows all examples
 *
 * Use this to visualize and test all Phase 1 components
 */
export function AllExamplesShowcase() {
  const [activeExample, setActiveExample] = React.useState<
    'layout' | 'dashboard' | 'status' | 'connection' | 'filter'
  >('layout');

  const examples = [
    { id: 'layout', label: 'Full Layout', component: LayoutExample },
    { id: 'dashboard', label: 'Dashboard', component: DashboardOverviewExample },
    { id: 'status', label: 'Status Badges', component: StatusBadgeVariationsExample },
    { id: 'connection', label: 'Connection', component: ConnectionStatusExample },
    { id: 'filter', label: 'Filter & View', component: FilterAndViewExample },
  ] as const;

  const ActiveComponent = examples.find(
    (e) => e.id === activeExample
  )?.component || (() => <div>Not found</div>);

  return (
    <div className="min-h-screen bg-slate-950">
      <div className="flex gap-6">
        {/* Navigation */}
        <div className="w-48 bg-slate-900 border-r border-slate-800 p-4 sticky top-0 h-screen overflow-y-auto">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide mb-4">
            Examples
          </h2>
          <div className="space-y-1">
            {examples.map((example) => (
              <button
                key={example.id}
                onClick={() => setActiveExample(example.id)}
                className={`
                  w-full text-left px-3 py-2 rounded-md text-sm font-medium transition-colors
                  ${
                    activeExample === example.id
                      ? 'bg-emerald-600 text-white'
                      : 'text-slate-300 hover:bg-slate-800'
                  }
                `}
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto">
          <ActiveComponent />
        </div>
      </div>
    </div>
  );
}

export default LayoutExample;
