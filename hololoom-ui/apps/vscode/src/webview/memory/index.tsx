/**
 * Memory Browser Webview Entry Point
 *
 * This is a placeholder for the React-based memory browser webview.
 * Currently, the memory UI is implemented inline in memoryViewProvider.ts
 * for simplicity. This file can be expanded for more complex UI needs.
 */

import React from 'react';
import { createRoot } from 'react-dom/client';

// VS Code API
declare const acquireVsCodeApi: () => {
  postMessage: (message: unknown) => void;
  getState: () => unknown;
  setState: (state: unknown) => void;
};

const vscode = acquireVsCodeApi();

interface MemoryItem {
  id: string;
  content: string;
  type: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

function MemoryApp() {
  const [memories, setMemories] = React.useState<MemoryItem[]>([]);
  const [searchQuery, setSearchQuery] = React.useState('');
  const [isLoading, setIsLoading] = React.useState(true);

  React.useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const data = event.data;
      switch (data.type) {
        case 'memories':
          setMemories(data.memories);
          break;
        case 'loading':
          setIsLoading(data.loading);
          break;
      }
    };

    window.addEventListener('message', handleMessage);
    vscode.postMessage({ type: 'loadRecent' });

    return () => window.removeEventListener('message', handleMessage);
  }, []);

  const handleSearch = React.useCallback((query: string) => {
    setSearchQuery(query);
    vscode.postMessage({ type: 'search', query });
  }, []);

  const handleDelete = (id: string) => {
    vscode.postMessage({ type: 'delete', id });
  };

  const handleInsert = (content: string) => {
    vscode.postMessage({ type: 'insert', content });
  };

  const handleCopy = async (content: string) => {
    await navigator.clipboard.writeText(content);
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="memory-app">
      <div className="search-area">
        <input
          type="text"
          className="search-input"
          value={searchQuery}
          onChange={(e) => handleSearch(e.target.value)}
          placeholder="Search memories..."
        />
      </div>

      <div className="memories">
        {isLoading ? (
          <div className="loading">
            <div className="loading-spinner" />
            <span>Loading...</span>
          </div>
        ) : memories.length === 0 ? (
          <div className="empty-state">
            <h4>No memories found</h4>
            <p>Start adding memories or try a different search.</p>
            <button
              className="refresh-btn"
              onClick={() => vscode.postMessage({ type: 'loadRecent' })}
            >
              Refresh
            </button>
          </div>
        ) : (
          memories.map((mem) => (
            <div key={mem.id} className="memory-item">
              <div className="memory-header">
                <span className="memory-type">{mem.type || 'memory'}</span>
                <span className="memory-time">{formatTime(mem.timestamp)}</span>
              </div>
              <div className="memory-content">{mem.content}</div>
              <div className="memory-actions">
                <button className="action-btn" onClick={() => handleInsert(mem.content)}>
                  Insert
                </button>
                <button className="action-btn" onClick={() => handleCopy(mem.content)}>
                  Copy
                </button>
                <button className="action-btn delete" onClick={() => handleDelete(mem.id)}>
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// Mount the app
const container = document.getElementById('app');
if (container) {
  const root = createRoot(container);
  root.render(<MemoryApp />);
}
