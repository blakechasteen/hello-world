/**
 * Advanced WebSocket Usage Examples
 *
 * Complex patterns for building sophisticated Agent Manager UI components.
 *
 * Date: 2025-12-11
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  useAgentManagerWS,
  useAgentManagerMessages,
  useAgentManagerPattern,
  useAgentManagerAction
} from '@/lib/agent-manager';
import type { WebSocketMessage } from '@/lib/agent-manager';

/**
 * Real-time thread executor with progress tracking
 */
export function ThreadExecutor({ threadId }: { threadId: string }) {
  const [executing, setExecuting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);

  // Use action hook for execute command
  const { send: executeStep, isLoading: executing_ } = useAgentManagerAction(
    'execute_thread',
    'thread_complete'
  );

  // Use pattern hook to get all thread events
  const threadEvents = useAgentManagerPattern(`thread:${threadId}`, 100);

  // Process incoming events
  useEffect(() => {
    threadEvents.forEach((event) => {
      switch (event.type) {
        case 'thread_started':
          setExecuting(true);
          setLogs((prev) => [...prev, '▶️ Thread started']);
          break;

        case 'step_progress':
          setCurrentStep(event.step_index || 0);
          setProgress(event.progress || 0);
          break;

        case 'step_complete':
          setLogs((prev) => [...prev, `✅ Step ${event.step_index} complete`]);
          break;

        case 'thread_complete':
          setExecuting(false);
          setProgress(100);
          setLogs((prev) => [...prev, '✓ Thread complete']);
          break;

        case 'error':
          setLogs((prev) => [...prev, `❌ Error: ${event.data?.message || 'Unknown'}`]);
          break;
      }
    });
  }, [threadEvents]);

  const handleStart = useCallback(() => {
    executeStep({ thread_id: threadId });
    setProgress(0);
    setCurrentStep(0);
    setLogs([]);
  }, [executeStep, threadId]);

  return (
    <div className="executor">
      <h2>Thread: {threadId}</h2>

      <button onClick={handleStart} disabled={executing}>
        {executing ? 'Executing...' : 'Start'}
      </button>

      <div className="progress-section">
        <p>Step {currentStep}</p>
        <progress value={progress} max={100} />
        <p>{progress}%</p>
      </div>

      <div className="logs">
        <h4>Logs</h4>
        <div style={{ height: '200px', overflow: 'auto', fontFamily: 'monospace', fontSize: '12px' }}>
          {logs.map((log, idx) => (
            <p key={idx}>{log}</p>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * Live project dashboard with multiple thread tracking
 */
export function ProjectDashboard({ projectId }: { projectId: string }) {
  const [threads, setThreads] = useState<Map<string, any>>(new Map());
  const [stats, setStats] = useState({ total: 0, completed: 0, running: 0, error: 0 });

  // Subscribe to project events
  const { subscribe } = useAgentManagerWS({ autoConnect: true });

  // Get all project messages
  const projectMessages = useAgentManagerPattern(`project:${projectId}`, 500);

  useEffect(() => {
    subscribe(`project:${projectId}`);
  }, [projectId, subscribe]);

  // Process messages and update threads
  useEffect(() => {
    const threadsMap = new Map(threads);
    let completed = 0,
      running = 0,
      errored = 0;

    projectMessages.forEach((msg) => {
      if (!msg.thread_id) return;

      const thread = threadsMap.get(msg.thread_id) || { id: msg.thread_id, status: 'pending', progress: 0 };

      switch (msg.type) {
        case 'thread_started':
          thread.status = 'running';
          running++;
          break;
        case 'step_progress':
          thread.progress = msg.progress || 0;
          running++;
          break;
        case 'thread_complete':
          thread.status = 'complete';
          thread.progress = 100;
          completed++;
          break;
        case 'error':
          thread.status = 'error';
          errored++;
          break;
      }

      threadsMap.set(msg.thread_id, thread);
    });

    setThreads(threadsMap);
    setStats({
      total: threadsMap.size,
      completed,
      running,
      error: errored
    });
  }, [projectMessages]);

  return (
    <div className="dashboard">
      <h2>Project: {projectId}</h2>

      <div className="stats">
        <div>Total: {stats.total}</div>
        <div style={{ color: 'green' }}>Completed: {stats.completed}</div>
        <div style={{ color: 'blue' }}>Running: {stats.running}</div>
        <div style={{ color: 'red' }}>Errors: {stats.error}</div>
      </div>

      <div className="threads">
        {Array.from(threads.values()).map((thread) => (
          <div key={thread.id} className={`thread-card ${thread.status}`}>
            <h4>{thread.id}</h4>
            <p>Status: {thread.status}</p>
            <progress value={thread.progress} max={100} />
            <p>{thread.progress}%</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Message inspector for debugging
 */
export function MessageInspector() {
  const [messages, setMessages] = useState<WebSocketMessage[]>([]);
  const [filter, setFilter] = useState('');
  const [paused, setPaused] = useState(false);
  const messagesRef = useRef<WebSocketMessage[]>([]);

  const { on } = useAgentManagerWS({ autoConnect: true });

  useEffect(() => {
    const unsubscribe = on('*', (message) => {
      if (!paused) {
        messagesRef.current = [message, ...messagesRef.current.slice(0, 99)];
        setMessages([...messagesRef.current]);
      }
    });

    return unsubscribe;
  }, [on, paused]);

  const filteredMessages = messages.filter(
    (msg) =>
      !filter ||
      msg.type.includes(filter) ||
      JSON.stringify(msg).toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="inspector">
      <h3>Message Inspector</h3>

      <div className="controls">
        <button onClick={() => setPaused(!paused)}>{paused ? 'Resume' : 'Pause'}</button>
        <button onClick={() => setMessages([])}>Clear</button>
        <input
          type="text"
          placeholder="Filter messages..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <span>Total: {messages.length}</span>
      </div>

      <div className="messages">
        {filteredMessages.map((msg, idx) => (
          <div key={idx} className="message" style={{ padding: '8px', marginBottom: '8px', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
            <strong>{msg.type}</strong>
            <p style={{ fontSize: '12px', color: '#666' }}>{msg.timestamp}</p>
            <pre style={{ fontSize: '11px', overflow: 'auto', maxHeight: '200px' }}>
              {JSON.stringify(msg, null, 2)}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Real-time notification system
 */
export function NotificationCenter() {
  const [notifications, setNotifications] = useState<
    Array<{ id: string; type: 'info' | 'success' | 'warning' | 'error'; message: string; timestamp: Date }>
  >([]);

  const { on } = useAgentManagerWS({ autoConnect: true });

  useEffect(() => {
    const handlers: Array<[string, (msg: WebSocketMessage) => void]> = [
      [
        'thread_created',
        (msg) => addNotification(`Thread created: ${msg.thread_id}`, 'info')
      ],
      [
        'thread_complete',
        (msg) => addNotification(`Thread completed: ${msg.thread_id}`, 'success')
      ],
      [
        'error',
        (msg) => addNotification(`Error: ${msg.data?.message || 'Unknown error'}`, 'error')
      ]
    ];

    const unsubscribes = handlers.map(([type, handler]) => on(type, handler));

    return () => unsubscribes.forEach((fn) => fn());
  }, [on]);

  const addNotification = (message: string, type: 'info' | 'success' | 'warning' | 'error') => {
    const id = `notif-${Date.now()}`;
    setNotifications((prev) => [{ id, type, message, timestamp: new Date() }, ...prev.slice(0, 9)]);

    // Auto-remove after 5 seconds
    setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
    }, 5000);
  };

  return (
    <div className="notification-center" style={{ position: 'fixed', top: '20px', right: '20px', zIndex: 1000 }}>
      {notifications.map((notif) => (
        <div
          key={notif.id}
          className={`notification ${notif.type}`}
          style={{
            padding: '12px 16px',
            marginBottom: '8px',
            borderRadius: '4px',
            backgroundColor: {
              info: '#e3f2fd',
              success: '#e8f5e9',
              warning: '#fff3e0',
              error: '#ffebee'
            }[notif.type],
            color: {
              info: '#01579b',
              success: '#1b5e20',
              warning: '#e65100',
              error: '#b71c1c'
            }[notif.type],
            fontSize: '14px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          }}
        >
          {notif.message}
        </div>
      ))}
    </div>
  );
}

/**
 * Reconnection monitor
 */
export function ReconnectionMonitor() {
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastReconnectTime, setLastReconnectTime] = useState<Date | null>(null);
  const { state, onStateChange } = useAgentManagerWS({ autoConnect: true });

  useEffect(() => {
    const unsubscribe = onStateChange?.((newState) => {
      if (newState === 'reconnecting') {
        setReconnectAttempts((prev) => prev + 1);
        setLastReconnectTime(new Date());
      }
    });

    return unsubscribe;
  }, [onStateChange]);

  return (
    <div
      style={{
        padding: '10px',
        backgroundColor: state === 'error' ? '#ffcccc' : '#ccccff',
        borderRadius: '4px',
        fontSize: '12px'
      }}
    >
      <p>State: <strong>{state}</strong></p>
      {reconnectAttempts > 0 && (
        <>
          <p>Reconnect Attempts: {reconnectAttempts}</p>
          {lastReconnectTime && (
            <p>Last Attempt: {lastReconnectTime.toLocaleTimeString()}</p>
          )}
        </>
      )}
    </div>
  );
}
