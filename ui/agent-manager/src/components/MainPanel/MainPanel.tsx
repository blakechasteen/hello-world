import React, { useCallback } from 'react';
import { useAgentManagerStore } from '../../stores/agentManagerStore';
import { OutlineView } from '../OutlineView';
import { DetailPanel } from '../DetailPanel';

/**
 * MainPanel Component
 * Renders different views based on viewMode:
 * - outline: Agent thread outline view (Phase 3 - Implemented)
 * - tree: Hierarchical thread tree view (Phase 6 - Coming soon)
 * - swarm: Swarm visualization (Phase 7 - Coming soon)
 *
 * Also includes a slide-in DetailPanel for expanded thread view (Phase 4)
 */
export const MainPanel: React.FC = () => {
  const { viewMode, threads, activeThreadId, setActiveThread } = useAgentManagerStore((state) => ({
    viewMode: state.viewMode,
    threads: state.getFilteredThreads(),
    activeThreadId: state.activeThreadId,
    setActiveThread: state.setActiveThread,
  }));

  // Handle detail panel close - clears active thread
  const handleDetailClose = useCallback(() => {
    setActiveThread(null);
  }, [setActiveThread]);

  const threadCount = threads.length;

  const renderView = () => {
    switch (viewMode) {
      case 'outline':
        return <OutlineView />;

      case 'tree':
        return (
          <div className="p-8 space-y-6">
            <div className="space-y-2">
              <h2 className="text-2xl font-bold text-slate-100">
                Thread Tree
              </h2>
              <p className="text-slate-400">
                Dependency tree showing parent-child relationships
              </p>
            </div>

            {threadCount === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="text-6xl mb-4">⊢</div>
                <h3 className="text-lg font-semibold text-slate-300 mb-2">
                  No Threads Yet
                </h3>
                <p className="text-slate-500 max-w-md">
                  Create a new thread from the sidebar to see it appear in the
                  tree view
                </p>
              </div>
            ) : (
              <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
                <div className="text-slate-400">
                  {threadCount} thread{threadCount !== 1 ? 's' : ''} loaded
                </div>
                <p className="text-slate-500 text-sm mt-2">
                  Tree rendering coming in Phase 3
                </p>
              </div>
            )}
          </div>
        );

      case 'swarm':
        return (
          <div className="p-8 space-y-6">
            <div className="space-y-2">
              <h2 className="text-2xl font-bold text-slate-100">
                Swarm Visualization
              </h2>
              <p className="text-slate-400">
                Force-directed graph of agent interactions and data flow
              </p>
            </div>

            {threadCount === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="text-6xl mb-4">◆</div>
                <h3 className="text-lg font-semibold text-slate-300 mb-2">
                  No Threads Yet
                </h3>
                <p className="text-slate-500 max-w-md">
                  Create a new thread from the sidebar to see it appear in the
                  swarm visualization
                </p>
              </div>
            ) : (
              <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
                <div className="text-slate-400">
                  {threadCount} thread{threadCount !== 1 ? 's' : ''} loaded
                </div>
                <p className="text-slate-500 text-sm mt-2">
                  Swarm visualization coming in Phase 3
                </p>
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="min-h-full bg-slate-950 flex">
      {/* Main content area - shifts left when detail panel is open */}
      <div className={`flex-1 transition-all duration-300 ease-out ${activeThreadId ? 'mr-[500px]' : ''}`}>
        {renderView()}
      </div>

      {/* Detail Panel - Slide-in from right edge */}
      <div
        className={`
          fixed top-0 right-0 h-full w-[500px] z-50
          transform transition-transform duration-300 ease-out
          ${activeThreadId ? 'translate-x-0' : 'translate-x-full'}
        `}
      >
        {activeThreadId && (
          <DetailPanel
            threadId={activeThreadId}
            onClose={handleDetailClose}
          />
        )}
      </div>

      {/* Backdrop for mobile/tablet - click to close */}
      {activeThreadId && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={handleDetailClose}
          aria-hidden="true"
        />
      )}
    </div>
  );
};

export default MainPanel;
