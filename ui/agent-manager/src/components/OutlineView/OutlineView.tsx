import React, { useMemo } from 'react';
import { useAgentManagerStore, AgentThread } from '../../stores/agentManagerStore';
import ThreadCard from './ThreadCard';

/**
 * OutlineView Component
 * Displays all agent threads in outline mode, sorted by priority (highest first)
 *
 * Features:
 * - Lists ThreadCards in priority order
 * - Virtualized scrolling for performance (using CSS overflow)
 * - Empty state when no threads
 * - Filter indicator showing current filter
 * - Real-time updates via Zustand store
 */

export const OutlineView: React.FC = () => {
  // Get filtered threads and active thread ID
  const { threads, activeThreadId, filter } = useAgentManagerStore((state) => ({
    threads: state.getFilteredThreads(),
    activeThreadId: state.activeThreadId,
    filter: state.filter,
  }));

  // Sort threads by priority (highest first), then by creation time (newest first)
  const sortedThreads = useMemo(() => {
    return [...threads].sort((a, b) => {
      // Primary sort: priority (descending)
      if (a.priority !== b.priority) {
        return b.priority - a.priority;
      }
      // Secondary sort: creation time (newest first)
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });
  }, [threads]);

  // Get thread count by status for header
  const threadStats = useMemo(() => {
    return {
      total: sortedThreads.length,
      running: sortedThreads.filter((t) => t.status === 'running').length,
      paused: sortedThreads.filter((t) => t.status === 'paused').length,
      completed: sortedThreads.filter((t) => t.status === 'completed').length,
      failed: sortedThreads.filter((t) => t.status === 'failed' || t.status === 'cancelled')
        .length,
    };
  }, [sortedThreads]);

  // Get filter label
  const getFilterLabel = (): string => {
    switch (filter) {
      case 'active':
        return 'Active Threads';
      case 'completed':
        return 'Completed Threads';
      case 'failed':
        return 'Failed Threads';
      case 'all':
      default:
        return 'All Threads';
    }
  };

  // Empty state
  if (sortedThreads.length === 0) {
    return (
      <div className="min-h-full bg-slate-950 p-8">
        <div className="space-y-2 mb-8">
          <h2 className="text-2xl font-bold text-slate-100">Thread Outline</h2>
          <p className="text-slate-400">
            Hierarchical view of agent threads and their dependencies
          </p>
        </div>

        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="text-6xl mb-4 text-slate-700">≡</div>
          <h3 className="text-lg font-semibold text-slate-300 mb-2">
            No Threads Yet
          </h3>
          <p className="text-slate-500 max-w-md">
            Create a new thread from the sidebar to see it appear in the outline view.
            Threads will be sorted by priority and displayed here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-full bg-slate-950">
      <div className="p-8 space-y-6">
        {/* Header */}
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-bold text-slate-100">Thread Outline</h2>
            <p className="text-slate-400">
              Hierarchical view of agent threads and their dependencies
            </p>
          </div>

          {/* Filter and stats bar */}
          <div className="flex items-center justify-between bg-slate-900 border border-slate-800 rounded-lg px-4 py-3">
            <div className="flex items-center gap-4">
              <div>
                <h3 className="text-sm font-semibold text-slate-300">
                  {getFilterLabel()}
                </h3>
                <p className="text-xs text-slate-500 mt-1">
                  Sorted by priority (highest first)
                </p>
              </div>
            </div>

            {/* Thread statistics */}
            <div className="flex items-center gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-slate-500"></div>
                <span className="text-slate-400">
                  {threadStats.total} total
                </span>
              </div>
              {threadStats.running > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
                  <span className="text-slate-400">
                    {threadStats.running} running
                  </span>
                </div>
              )}
              {threadStats.paused > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-amber-500"></div>
                  <span className="text-slate-400">
                    {threadStats.paused} paused
                  </span>
                </div>
              )}
              {threadStats.completed > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                  <span className="text-slate-400">
                    {threadStats.completed} completed
                  </span>
                </div>
              )}
              {threadStats.failed > 0 && (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-red-500"></div>
                  <span className="text-slate-400">
                    {threadStats.failed} failed
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Thread list with scrolling */}
        <div className="space-y-3 max-h-[calc(100vh-400px)] overflow-y-auto pr-2">
          {/* Add scroll styling */}
          <style>{`
            .outline-view-scroll::-webkit-scrollbar {
              width: 8px;
            }
            .outline-view-scroll::-webkit-scrollbar-track {
              background: transparent;
            }
            .outline-view-scroll::-webkit-scrollbar-thumb {
              background: #475569;
              border-radius: 4px;
            }
            .outline-view-scroll::-webkit-scrollbar-thumb:hover {
              background: #64748b;
            }
          `}</style>
          <div className="outline-view-scroll space-y-3 max-h-[calc(100vh-400px)] overflow-y-auto pr-2">
            {sortedThreads.map((thread) => (
              <ThreadCard
                key={thread.id}
                thread={thread}
                isActive={activeThreadId === thread.id}
                onSelect={() => {
                  // Selection handled by store within ThreadCard
                }}
              />
            ))}
          </div>
        </div>

        {/* Footer info */}
        <div className="text-xs text-slate-500 border-t border-slate-800 pt-4">
          <p>
            💡 Tip: Click on a thread name to edit it. Click the expand button to see
            detailed steps and controls. Use the +/- buttons to adjust priority.
          </p>
        </div>
      </div>
    </div>
  );
};

export default OutlineView;
