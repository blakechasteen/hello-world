/**
 * Basic WebSocket Usage Examples
 *
 * Simple examples of using the Agent Manager WebSocket client
 * in React components.
 *
 * Date: 2025-12-11
 */

import { useEffect, useState } from 'react';
import { useAgentManagerWS } from '@/lib/agent-manager';

/**
 * Simple connection status indicator
 */
export function ConnectionStatus() {
  const { isConnected, state } = useAgentManagerWS({
    autoConnect: true
  });

  return (
    <div className="connection-status">
      <span className={`status-dot ${state}`} />
      <span className="status-text">
        {isConnected ? '🟢 Connected' : `⚠️ ${state}`}
      </span>
    </div>
  );
}

/**
 * Listen to a specific message type
 */
export function StepProgressListener() {
  const [currentStep, setCurrentStep] = useState<number | null>(null);
  const [progress, setProgress] = useState(0);
  const { on } = useAgentManagerWS({ autoConnect: true });

  useEffect(() => {
    const unsubscribe = on('step_progress', (message) => {
      setCurrentStep(message.step_index);
      setProgress(message.progress || 0);
    });

    return unsubscribe;
  }, [on]);

  if (currentStep === null) {
    return <p>Waiting for step updates...</p>;
  }

  return (
    <div>
      <h3>Step {currentStep}</h3>
      <progress value={progress} max={100} />
      <p>{progress}%</p>
    </div>
  );
}

/**
 * Subscribe to thread updates
 */
export function ThreadSubscriber({ threadId }: { threadId: string }) {
  const [updates, setUpdates] = useState<string[]>([]);
  const { on } = useAgentManagerWS({
    autoConnect: true,
    subscriptions: [`thread:${threadId}`]
  });

  useEffect(() => {
    const unsubscribe = on('*', (message) => {
      if (message.thread_id === threadId) {
        setUpdates((prev) => [
          `[${new Date().toLocaleTimeString()}] ${message.type}`,
          ...prev.slice(0, 9)
        ]);
      }
    });

    return unsubscribe;
  }, [threadId, on]);

  return (
    <div>
      <h3>Thread {threadId} Updates</h3>
      <ul>
        {updates.map((update, idx) => (
          <li key={idx}>{update}</li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Send an action and wait for response
 */
export function ActionSender() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const { send, on } = useAgentManagerWS({ autoConnect: true });

  const handleExecute = () => {
    setLoading(true);
    setResult(null);

    // Send action
    send('execute_step', {
      thread_id: 'thread-123',
      step: 0
    });
  };

  // Listen for completion
  useEffect(() => {
    const unsubscribe = on('step_complete', (message) => {
      setLoading(false);
      setResult(`✅ Completed with status: ${message.status}`);
    });

    return unsubscribe;
  }, [on]);

  return (
    <div>
      <button onClick={handleExecute} disabled={loading}>
        {loading ? 'Executing...' : 'Execute Step'}
      </button>
      {result && <p>{result}</p>}
    </div>
  );
}

/**
 * Multi-pattern subscription
 */
export function MultiSubscriber() {
  const [events, setEvents] = useState<string[]>([]);
  const { subscribe, on } = useAgentManagerWS({ autoConnect: true });

  useEffect(() => {
    // Subscribe to multiple patterns
    subscribe('thread:abc123');
    subscribe('project:proj-456');
    subscribe('step_progress');
  }, [subscribe]);

  useEffect(() => {
    const unsubscribe = on('*', (message) => {
      setEvents((prev) => [
        `${message.type} - ${JSON.stringify(message).slice(0, 50)}...`,
        ...prev.slice(0, 19)
      ]);
    });

    return unsubscribe;
  }, [on]);

  return (
    <div>
      <h3>All Events ({events.length})</h3>
      <div style={{ maxHeight: '300px', overflow: 'auto' }}>
        {events.map((event, idx) => (
          <p key={idx} style={{ fontFamily: 'monospace', fontSize: '12px' }}>
            {event}
          </p>
        ))}
      </div>
    </div>
  );
}

/**
 * Error handling example
 */
export function ErrorHandler() {
  const [errors, setErrors] = useState<string[]>([]);
  const { state } = useAgentManagerWS({ autoConnect: true });

  useEffect(() => {
    if (state === 'error') {
      setErrors((prev) => [
        `${new Date().toLocaleTimeString()} - Connection error`,
        ...prev.slice(0, 4)
      ]);
    }
  }, [state]);

  return (
    <div>
      {errors.length > 0 && (
        <div style={{ padding: '10px', backgroundColor: '#ffcccc', borderRadius: '4px' }}>
          <h4>Recent Errors</h4>
          <ul>
            {errors.map((error, idx) => (
              <li key={idx}>{error}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
