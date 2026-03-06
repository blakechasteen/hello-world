'use client';

import { useCallback, useEffect } from 'react';
import { useWeaving, useWeavingDispatch } from '@/contexts/WeavingContext';
import { MemoryGraph } from '@/components/memory/MemoryGraph';
import { ChatInterface } from '@/components/ChatInterface';
import { CompactMetrics } from './CompactMetrics';

function PaneHeader({
  label,
  paneKey,
  visible,
}: {
  label: string;
  paneKey: 'memory' | 'chat' | 'metrics';
  visible: boolean;
}) {
  const dispatch = useWeavingDispatch();

  return (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-border-primary bg-bg-secondary/50">
      <span className="text-xs font-medium text-fg-tertiary uppercase tracking-wider">
        {label}
      </span>
      <button
        onClick={() => dispatch({ type: 'TOGGLE_PANE', pane: paneKey })}
        className="text-fg-tertiary hover:text-fg-secondary text-xs px-1"
        title={visible ? `Collapse ${label}` : `Expand ${label}`}
      >
        {visible ? '−' : '+'}
      </button>
    </div>
  );
}

export function CognitiveShell() {
  const state = useWeaving();
  const dispatch = useWeavingDispatch();
  const { panes, selectedNodeId, highlightedNodeIds, pendingReference } = state;

  // Keyboard shortcuts: Cmd/Ctrl + 1/2/3 to toggle panes
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey)) return;
      switch (e.key) {
        case '1':
          e.preventDefault();
          dispatch({ type: 'TOGGLE_PANE', pane: 'memory' });
          break;
        case '2':
          e.preventDefault();
          dispatch({ type: 'TOGGLE_PANE', pane: 'chat' });
          break;
        case '3':
          e.preventDefault();
          dispatch({ type: 'TOGGLE_PANE', pane: 'metrics' });
          break;
      }
    },
    [dispatch],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Count visible panes for grid columns
  const visiblePanes = [panes.memory, panes.chat, panes.metrics].filter(Boolean).length;

  // Build grid template based on which panes are visible
  const gridCols = [
    panes.memory ? 'minmax(250px,1fr)' : null,
    panes.chat ? 'minmax(400px,2fr)' : null,
    panes.metrics ? '300px' : null,
  ]
    .filter(Boolean)
    .join(' ');

  const handleNodeSelect = useCallback(
    (nodeId: string) => {
      dispatch({ type: 'SELECT_NODE', nodeId });
      // Also set as pending reference for the chat input
      dispatch({ type: 'REFERENCE_NODE', nodeId, content: nodeId });
    },
    [dispatch],
  );

  const handleClearReference = useCallback(() => {
    dispatch({ type: 'CLEAR_REFERENCE' });
  }, [dispatch]);

  return (
    <div className="flex flex-col h-screen">
      {/* Collapsed pane indicators */}
      {visiblePanes < 3 && (
        <div className="flex items-center gap-2 px-4 py-1 bg-bg-secondary/30 border-b border-border-primary">
          {!panes.memory && (
            <button
              onClick={() => dispatch({ type: 'TOGGLE_PANE', pane: 'memory' })}
              className="text-xs text-fg-tertiary hover:text-fg-secondary px-2 py-0.5 rounded bg-bg-tertiary"
            >
              + Memory Palace
            </button>
          )}
          {!panes.chat && (
            <button
              onClick={() => dispatch({ type: 'TOGGLE_PANE', pane: 'chat' })}
              className="text-xs text-fg-tertiary hover:text-fg-secondary px-2 py-0.5 rounded bg-bg-tertiary"
            >
              + Active Thinking
            </button>
          )}
          {!panes.metrics && (
            <button
              onClick={() =>
                dispatch({ type: 'TOGGLE_PANE', pane: 'metrics' })
              }
              className="text-xs text-fg-tertiary hover:text-fg-secondary px-2 py-0.5 rounded bg-bg-tertiary"
            >
              + Awareness
            </button>
          )}
          <span className="text-xs text-fg-tertiary ml-auto">
            Ctrl+1/2/3 to toggle panes
          </span>
        </div>
      )}

      {/* Three-pane grid */}
      <div
        className="flex-1 grid gap-0 overflow-hidden"
        style={{ gridTemplateColumns: gridCols }}
      >
        {/* Left: Memory Palace */}
        {panes.memory && (
          <div className="flex flex-col border-r border-border-primary overflow-hidden">
            <PaneHeader label="Memory Palace" paneKey="memory" visible />
            <div className="flex-1 relative">
              <MemoryGraph
                searchQuery=""
                onNodeSelect={handleNodeSelect}
                selectedNodeId={selectedNodeId}
                highlightedNodeIds={highlightedNodeIds}
              />
            </div>
          </div>
        )}

        {/* Center: Active Thinking */}
        {panes.chat && (
          <div className="flex flex-col overflow-hidden">
            <PaneHeader label="Active Thinking" paneKey="chat" visible />
            <div className="flex-1 overflow-hidden">
              <ChatInterface
                className="h-full"
                pendingReference={pendingReference}
                onClearReference={handleClearReference}
              />
            </div>
          </div>
        )}

        {/* Right: Awareness */}
        {panes.metrics && (
          <div className="flex flex-col border-l border-border-primary overflow-hidden">
            <CompactMetrics />
          </div>
        )}
      </div>
    </div>
  );
}
