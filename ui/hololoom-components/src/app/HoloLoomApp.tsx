/**
 * HoloLoomApp - Main Application Component
 *
 * Wires together the ThreePaneLayout with all connected components
 * and manages WebSocket connection lifecycle.
 *
 * Layout:
 * - Left: Memory Palace (Knowledge Graph)
 * - Center: Active Thinking (Stage Timeline)
 * - Right: Awareness (Metrics Pane)
 *
 * Part of HoloLoom Phase 1 Frontend (November 2025)
 */

import React, { useState, useCallback, useEffect } from 'react';
import { clsx } from 'clsx';
import { ThreePaneLayout } from './ThreePaneLayout';
import { ConnectedStageTimeline } from '../components/connected/ConnectedStageTimeline';
import { ConnectedKnowledgeGraph } from '../components/connected/ConnectedKnowledgeGraph';
import { ConnectedMetricsPane } from '../components/connected/ConnectedMetricsPane';
import { useWebSocketConnection, useStream } from '../store/websocket-actions';
import { useHoloLoomStore } from '../store';
import type { ReasoningMode } from '../types';

// ============================================================================
// Props
// ============================================================================

export interface HoloLoomAppProps {
  /** WebSocket URL (defaults to ws://localhost:8000/ws) */
  wsUrl?: string;
  /** Initial dark mode state */
  initialDarkMode?: boolean;
  /** Called when query is submitted */
  onQuerySubmit?: (query: string, mode: ReasoningMode) => void;
  /** Called when stream completes */
  onStreamComplete?: (response: string) => void;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Query Input Component
// ============================================================================

interface QueryInputProps {
  onSubmit: (query: string, mode: ReasoningMode) => void;
  disabled?: boolean;
  darkMode?: boolean;
}

const QueryInput: React.FC<QueryInputProps> = ({ onSubmit, disabled, darkMode }) => {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<ReasoningMode>('direct');

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (query.trim() && !disabled) {
        onSubmit(query.trim(), mode);
        setQuery('');
      }
    },
    [query, mode, disabled, onSubmit]
  );

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask HoloLoom a question..."
        disabled={disabled}
        className={clsx(
          'flex-1 px-4 py-2 rounded-lg border',
          'focus:outline-none focus:ring-2 focus:ring-blue-500',
          darkMode
            ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-500'
            : 'bg-white border-gray-300 text-gray-900 placeholder-gray-400',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      />
      <select
        value={mode}
        onChange={(e) => setMode(e.target.value as ReasoningMode)}
        disabled={disabled}
        className={clsx(
          'px-3 py-2 rounded-lg border',
          'focus:outline-none focus:ring-2 focus:ring-blue-500',
          darkMode
            ? 'bg-gray-800 border-gray-700 text-white'
            : 'bg-white border-gray-300 text-gray-900',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <option value="direct">Direct</option>
        <option value="verify">Verify</option>
        <option value="research">Research</option>
        <option value="plan_execute">Plan & Execute</option>
      </select>
      <button
        type="submit"
        disabled={disabled || !query.trim()}
        className={clsx(
          'px-6 py-2 rounded-lg font-medium',
          'focus:outline-none focus:ring-2 focus:ring-blue-500',
          'transition-colors duration-150',
          disabled || !query.trim()
            ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
            : 'bg-blue-600 text-white hover:bg-blue-700'
        )}
      >
        Send
      </button>
    </form>
  );
};

// ============================================================================
// Response Display Component
// ============================================================================

interface ResponseDisplayProps {
  darkMode?: boolean;
}

const ResponseDisplay: React.FC<ResponseDisplayProps> = ({ darkMode }) => {
  const accumulatedText = useHoloLoomStore((state) => state.accumulatedText);
  const streamState = useHoloLoomStore((state) => state.streamState);
  const tokensGenerated = useHoloLoomStore((state) => state.tokensGenerated);

  if (!accumulatedText && streamState === 'idle') {
    return (
      <div
        className={clsx(
          'flex-1 flex items-center justify-center',
          darkMode ? 'text-gray-500' : 'text-gray-400'
        )}
      >
        <p className="text-center">
          <span className="block text-lg mb-2">Welcome to HoloLoom</span>
          <span className="text-sm">Ask a question to see the weaving process in real-time</span>
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-auto p-4">
      {/* Response text */}
      <div
        className={clsx(
          'prose max-w-none',
          darkMode ? 'prose-invert' : '',
          'whitespace-pre-wrap'
        )}
      >
        {accumulatedText}
        {streamState === 'streaming' && (
          <span className="inline-block w-2 h-4 ml-1 bg-blue-500 animate-pulse" />
        )}
      </div>

      {/* Token count */}
      {tokensGenerated > 0 && (
        <div
          className={clsx(
            'mt-4 text-xs',
            darkMode ? 'text-gray-500' : 'text-gray-400'
          )}
        >
          {tokensGenerated.toLocaleString()} tokens generated
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Connection Status Component
// ============================================================================

interface ConnectionStatusProps {
  darkMode?: boolean;
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ darkMode }) => {
  const connectionState = useHoloLoomStore((state) => state.connectionState);
  const reconnectAttempts = useHoloLoomStore((state) => state.reconnectAttempts);

  const statusConfig = {
    disconnected: { color: 'bg-gray-400', text: 'Disconnected' },
    connecting: { color: 'bg-yellow-400 animate-pulse', text: 'Connecting...' },
    connected: { color: 'bg-green-500', text: 'Connected' },
    reconnecting: { color: 'bg-yellow-400 animate-pulse', text: `Reconnecting (${reconnectAttempts})...` },
    error: { color: 'bg-red-500', text: 'Connection Error' },
  };

  const config = statusConfig[connectionState];

  return (
    <div className="flex items-center gap-2">
      <div className={clsx('w-2 h-2 rounded-full', config.color)} />
      <span className={clsx('text-xs', darkMode ? 'text-gray-400' : 'text-gray-500')}>
        {config.text}
      </span>
    </div>
  );
};

// ============================================================================
// Header Component
// ============================================================================

interface HeaderProps {
  darkMode: boolean;
  onToggleDarkMode: () => void;
}

const Header: React.FC<HeaderProps> = ({ darkMode, onToggleDarkMode }) => {
  return (
    <div
      className={clsx(
        'flex items-center justify-between px-4 py-2',
        darkMode ? 'bg-gray-900' : 'bg-white'
      )}
    >
      <div className="flex items-center gap-4">
        <h1
          className={clsx(
            'text-lg font-semibold',
            darkMode ? 'text-white' : 'text-gray-900'
          )}
        >
          HoloLoom
        </h1>
        <ConnectionStatus darkMode={darkMode} />
      </div>
      <button
        onClick={onToggleDarkMode}
        className={clsx(
          'p-2 rounded-lg transition-colors',
          darkMode
            ? 'text-gray-400 hover:text-white hover:bg-gray-800'
            : 'text-gray-500 hover:text-gray-900 hover:bg-gray-100'
        )}
        title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {darkMode ? (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
            />
          </svg>
        ) : (
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
            />
          </svg>
        )}
      </button>
    </div>
  );
};

// ============================================================================
// Main Application Component
// ============================================================================

export const HoloLoomApp: React.FC<HoloLoomAppProps> = ({
  wsUrl,
  initialDarkMode = false,
  onQuerySubmit,
  onStreamComplete,
  className,
}) => {
  // Dark mode state
  const [darkMode, setDarkMode] = useState(initialDarkMode);

  // Apply dark mode class to document
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  // Load dark mode preference from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('hololoom-dark-mode');
    if (stored !== null) {
      setDarkMode(stored === 'true');
    }
  }, []);

  // Toggle dark mode
  const toggleDarkMode = useCallback(() => {
    setDarkMode((prev) => {
      const newValue = !prev;
      localStorage.setItem('hololoom-dark-mode', String(newValue));
      return newValue;
    });
  }, []);

  // WebSocket connection
  const ws = useWebSocketConnection({
    autoConnect: true,
    url: wsUrl,
  });

  // Stream management
  const stream = useStream({
    onStreamEnd: onStreamComplete,
  });

  // Handle query submission
  const handleQuerySubmit = useCallback(
    (query: string, mode: ReasoningMode) => {
      stream.sendQuery(query, mode);
      onQuerySubmit?.(query, mode);
    },
    [stream, onQuerySubmit]
  );

  // Center pane content: Timeline + Response
  const centerPane = (
    <div className="flex flex-col h-full">
      {/* Timeline at top */}
      <div className="flex-shrink-0 border-b border-gray-200 dark:border-gray-700">
        <ConnectedStageTimeline animated darkMode={darkMode} className="p-4" />
      </div>

      {/* Response display */}
      <ResponseDisplay darkMode={darkMode} />

      {/* Query input at bottom */}
      <div className="flex-shrink-0 border-t border-gray-200 dark:border-gray-700">
        <QueryInput
          onSubmit={handleQuerySubmit}
          disabled={stream.isStreaming}
          darkMode={darkMode}
        />
      </div>
    </div>
  );

  return (
    <ThreePaneLayout
      className={className}
      header={<Header darkMode={darkMode} onToggleDarkMode={toggleDarkMode} />}
      leftPane={
        <ConnectedKnowledgeGraph
          darkMode={darkMode}
          className="h-full"
        />
      }
      centerPane={centerPane}
      rightPane={
        <ConnectedMetricsPane darkMode={darkMode} />
      }
      paneConfig={{
        left: { title: 'Memory Palace' },
        center: { title: 'Active Thinking' },
        right: { title: 'Awareness' },
      }}
    />
  );
};

HoloLoomApp.displayName = 'HoloLoomApp';

export default HoloLoomApp;
