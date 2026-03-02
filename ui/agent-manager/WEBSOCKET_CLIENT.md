# Agent Manager WebSocket Client

**Date**: 2025-12-11
**Status**: ✅ Production Ready
**Location**: `ui/agent-manager/src/lib/`
**Total Code**: ~580 lines (client + hooks)

Provides real-time bidirectional communication between the Agent Manager UI and backend server with automatic reconnection, subscription management, and React integration.

## Overview

The WebSocket client enables:
- ✅ **Auto-reconnection** with exponential backoff (1s → 30s max)
- ✅ **Subscription system** with pattern-based routing (e.g., `thread:xyz`, `project:abc`, `*`)
- ✅ **Message queueing** during disconnection
- ✅ **Heartbeat/ping** mechanism (30s interval)
- ✅ **Event-based routing** with message type handlers
- ✅ **React integration** via custom hooks
- ✅ **TypeScript support** with full type safety

## Quick Start

### Standalone Usage

```typescript
import { wsClient } from '@/lib/agent-manager';

// Connect
wsClient.connect();

// Subscribe to thread updates
wsClient.subscribe('thread:abc123');

// Listen to messages
const unsubscribe = wsClient.on('step_progress', (message) => {
  console.log('Step progress:', message);
});

// Send action
wsClient.send('execute_step', { thread_id: 'abc123', step: 0 });

// Cleanup
unsubscribe();
wsClient.disconnect();
```

### React Hook Usage (Recommended)

```tsx
import { useAgentManagerWS, useAgentManagerMessages } from '@/lib/agent-manager';

function AgentPanel() {
  const { isConnected, subscribe, on } = useAgentManagerWS({
    autoConnect: true,
    subscriptions: ['thread:xyz']
  });

  // Listen to specific message type
  const messages = useAgentManagerMessages('step_progress');

  return (
    <div>
      <p>Status: {isConnected ? '🟢 Connected' : '🔴 Disconnected'}</p>
      <p>Messages: {messages.length}</p>
    </div>
  );
}
```

## Architecture

### Connection Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│  Connect Request                                         │
└─────────┬───────────────────────────────────────────────┘
          │
          ↓
┌─────────────────────────────────────────────────────────┐
│  connecting → connected (send pending subscriptions)    │
└─────────┬───────────────────────────────────────────────┘
          │
          ├─→ Success: heartbeat starts, messages flush
          │
          └─→ Error/Close:
                  ├─ Stop heartbeat
                  ├─ Queue outgoing messages
                  └─ Exponential backoff reconnect
```

### Message Flow

```
User Component
    ↓
[wsClient.on/send]
    ↓
Message Handler Registry
    ├─ Type-specific handlers
    ├─ Wildcard (*) handlers
    └─ State change handlers
    ↓
Event Dispatching
    ├─ Parse incoming JSON
    ├─ Route by message.type
    └─ Error handling
    ↓
Subscriptions
    └─ Pattern-based filtering
```

## API Reference

### AgentManagerWebSocket

```typescript
class AgentManagerWebSocket {
  // Lifecycle
  connect(): void
  disconnect(): void

  // Subscriptions
  subscribe(pattern: string): void      // e.g., 'thread:xyz', 'project:abc', '*'
  unsubscribe(pattern: string): void

  // Messaging
  send(action: string, data?: any): void
  on(eventType: string, handler: MessageHandler): () => void

  // State
  getState(): ConnectionState
  isConnected(): boolean
  onStateChange(handler: StateChangeHandler): () => void
}
```

### Subscription Patterns

```typescript
// Subscribe to specific thread
wsClient.subscribe('thread:abc123');

// Subscribe to specific project
wsClient.subscribe('project:project-xyz');

// Subscribe to message type (route by type field)
wsClient.subscribe('step_progress');  // Receives all step_progress messages

// Subscribe to all messages
wsClient.subscribe('*');
```

### Message Types

**Incoming Messages** (from backend):

```typescript
interface WebSocketMessage {
  type: string;                    // e.g., 'thread_created', 'step_progress'
  timestamp: string;               // ISO 8601 timestamp
  thread_id?: string;
  project_id?: string;
  step_index?: number;
  status?: string;                 // e.g., 'running', 'completed', 'error'
  progress?: number;               // 0-100
  data?: any;                       // Message-specific payload
  [key: string]: any;              // Additional fields
}
```

**Outgoing Messages** (to backend):

```typescript
// Subscribe
{ action: 'subscribe', pattern: 'thread:xyz' }

// Unsubscribe
{ action: 'unsubscribe', pattern: 'thread:xyz' }

// Custom action
{ action: 'execute_step', data: { thread_id: '...', step: 0 } }

// Heartbeat
{ action: 'ping' }
```

## React Hooks

### useAgentManagerWS

Main hook for WebSocket integration.

```typescript
const {
  isConnected,           // boolean
  state,                 // 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error'
  subscribe,             // (pattern: string) => void
  unsubscribe,           // (pattern: string) => void
  on,                    // (type: string, handler: MessageHandler) => () => void
  send,                  // (action: string, data?: any) => void
  connect,               // () => void
  disconnect             // () => void
} = useAgentManagerWS({
  autoConnect: true,                    // Default: connect on mount
  subscriptions: ['thread:xyz']          // Default: []
});
```

**Example:**

```tsx
function ThreadDetails({ threadId }: { threadId: string }) {
  const { isConnected, on, subscribe } = useAgentManagerWS({
    autoConnect: true,
    subscriptions: [`thread:${threadId}`]
  });

  useEffect(() => {
    const unsubscribe = on('step_progress', (msg) => {
      console.log('Step:', msg.step_index, 'Progress:', msg.progress);
    });

    return unsubscribe;
  }, [on]);

  return (
    <div>
      <h2>{threadId}</h2>
      {isConnected ? <p>🟢 Live Updates</p> : <p>⏳ Reconnecting...</p>}
    </div>
  );
}
```

### useAgentManagerMessages

Listen to specific message types and collect history.

```typescript
const messages: WebSocketMessage[] = useAgentManagerMessages(
  'step_progress',  // Message type
  100               // Max messages to keep (default: 100)
);

// messages[0] is most recent
```

**Example:**

```tsx
function StepLog() {
  const messages = useAgentManagerMessages('step_progress', 50);

  return (
    <div>
      {messages.map((msg, idx) => (
        <div key={idx}>
          Step {msg.step_index}: {msg.progress}%
        </div>
      ))}
    </div>
  );
}
```

### useAgentManagerPattern

Subscribe to pattern and collect matching messages.

```typescript
const messages: WebSocketMessage[] = useAgentManagerPattern(
  'thread:abc123',  // Pattern (thread:id, project:id, or message type)
  100               // Max messages (default: 100)
);
```

**Example:**

```tsx
function ProjectTimeline({ projectId }: { projectId: string }) {
  const messages = useAgentManagerPattern(`project:${projectId}`, 200);

  return (
    <div>
      <h3>Project Activity ({messages.length} events)</h3>
      <Timeline messages={messages} />
    </div>
  );
}
```

### useAgentManagerAction

Send actions and track responses.

```typescript
const {
  send,          // (data?: any) => void
  isLoading,     // boolean (waiting for response)
  error,         // string | null
  response       // WebSocketMessage | null
} = useAgentManagerAction(
  'execute_step',           // Action to send
  'step_complete'           // Optional: response event type (waits 30s)
);
```

**Example:**

```tsx
function StepExecutor({ threadId }: { threadId: string }) {
  const { send, isLoading, error, response } = useAgentManagerAction(
    'execute_step',
    'step_complete'
  );

  const handleExecute = () => {
    send({ thread_id: threadId, step: 0 });
  };

  return (
    <div>
      <button onClick={handleExecute} disabled={isLoading}>
        {isLoading ? 'Executing...' : 'Execute Step'}
      </button>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {response && <p>Completed: {response.status}</p>}
    </div>
  );
}
```

## Connection Management

### Auto-Reconnection

The client automatically reconnects with exponential backoff:

- Attempt 1: 1s delay
- Attempt 2: 2s delay
- Attempt 3: 4s delay
- ...
- Max: 30s delay
- Jitter: ±20% to prevent thundering herd
- Max attempts: 10 (then gives up)

### Heartbeat Mechanism

- **Interval**: 30 seconds
- **Timeout**: 5 seconds for pong response
- **Failure**: If no pong, close connection and reconnect

### Message Queuing

Messages sent while disconnected are queued and sent on reconnect:

```typescript
// Not connected yet
wsClient.send('execute_step', { /* ... */ });  // Queued

// Later, on reconnect, all queued messages are sent automatically
```

### Graceful Disconnection

```typescript
wsClient.disconnect();  // Stops heartbeat, closes connection
// State: 'disconnected'
// No reconnect attempts
```

## Connection States

```typescript
type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';
```

**State Transitions:**

```
disconnected
    ↓
connecting → connected
    ↓         ↓
  error ← close/error
    ↓
reconnecting → connected
    (or)
    error (max attempts reached)
```

**Monitoring State:**

```typescript
// Direct check
if (wsClient.isConnected()) {
  // Safe to send messages
}

// Listen to changes
const unsubscribe = wsClient.onStateChange((state) => {
  console.log('Connection state:', state);
  // 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error'
});
```

## Error Handling

```typescript
const { state, on } = useAgentManagerWS();

useEffect(() => {
  // Listen to connection errors
  const unsubscribe = wsClient.onStateChange((state) => {
    if (state === 'error') {
      showErrorNotification('Connection error - will retry automatically');
    }
  });

  return unsubscribe;
}, []);

// Handle message errors
useEffect(() => {
  const unsubscribe = on('error', (msg) => {
    console.error('Server error:', msg.data?.message);
  });

  return unsubscribe;
}, [on]);
```

## Performance Considerations

### Message Handler Overhead

- Type routing: O(1) - hash map lookup
- Handler execution: O(n) - n handlers per type
- Total per message: <1ms typical

### Memory Usage

- Message queue: ~1KB per message
- Subscriptions: ~100 bytes per pattern
- Handlers: ~500 bytes per handler

### Network

- Heartbeat: ~50 bytes every 30s
- Message size: ~100-5000 bytes typical
- Recommended max message size: 1MB

## Testing

### Unit Tests

```typescript
// Test connection lifecycle
test('connects and reconnects', async () => {
  const ws = new AgentManagerWebSocket('ws://localhost:8002');
  ws.connect();
  expect(ws.getState()).toBe('connecting');
  // ... wait for connection ...
  expect(ws.getState()).toBe('connected');
});

// Test message routing
test('routes messages by type', async () => {
  const handler = jest.fn();
  ws.on('step_progress', handler);

  // Simulate message
  ws['handleMessage'](new MessageEvent('message', {
    data: JSON.stringify({
      type: 'step_progress',
      step_index: 0,
      progress: 50
    })
  }));

  expect(handler).toHaveBeenCalled();
});
```

### Integration Tests

```tsx
// Test React hook
render(
  <TestWrapper>
    <AgentPanel />
  </TestWrapper>
);

// Wait for connection
await waitFor(() => {
  expect(screen.getByText('Connected')).toBeInTheDocument();
});

// Send message
fireEvent.click(screen.getByRole('button', { name: /execute/i }));

// Verify queue
expect(wsClient['messageQueue'].length).toBeGreaterThan(0);
```

## Debugging

### Enable Verbose Logging

```typescript
// Patch console
const originalLog = console.log;
const originalError = console.error;

wsClient.onStateChange((state) => {
  originalLog(`[WS] ${state}`);
});
```

### Monitor State

```typescript
setInterval(() => {
  console.log({
    state: wsClient.getState(),
    isConnected: wsClient.isConnected(),
    timestamp: new Date().toISOString()
  });
}, 5000);
```

### Check Message Queue

```typescript
// Access queue (private, for debugging)
console.log('Queued messages:', (wsClient as any).messageQueue);
```

## Best Practices

1. **Always cleanup handlers**:
   ```typescript
   const unsubscribe = wsClient.on('event', handler);
   // Later:
   unsubscribe();
   ```

2. **Use React hooks for components**:
   ```tsx
   // ✅ Good
   const { on } = useAgentManagerWS();

   // ❌ Avoid
   wsClient.on('event', handler);  // Manual cleanup needed
   ```

3. **Handle connection state**:
   ```tsx
   if (!isConnected) {
     return <Spinner label="Connecting..." />;
   }
   ```

4. **Queue important messages**:
   ```typescript
   // ✅ Automatically queued if disconnected
   wsClient.send('save_state', { /* ... */ });
   ```

5. **Use pattern subscriptions**:
   ```typescript
   // ✅ Efficient
   wsClient.subscribe(`thread:${threadId}`);

   // ❌ Less efficient
   wsClient.on('*', (msg) => {
     if (msg.thread_id === threadId) { /* ... */ }
   });
   ```

## Migration Guide

### From Direct WebSocket

```typescript
// Before
const ws = new WebSocket('ws://localhost:8002');
ws.onmessage = (event) => { /* ... */ };

// After
import { useAgentManagerWS } from '@/lib/agent-manager';
const { on } = useAgentManagerWS();
on('message_type', (msg) => { /* ... */ });
```

### From Poll-Based Updates

```tsx
// Before
useEffect(() => {
  const interval = setInterval(async () => {
    const data = await fetch(`/api/thread/${threadId}`);
    setThread(data);
  }, 1000);
}, []);

// After
const messages = useAgentManagerPattern(`thread:${threadId}`);
useEffect(() => {
  if (messages.length > 0) {
    setThread(messages[0].data);
  }
}, [messages]);
```

## Troubleshooting

### Connection Won't Establish

1. Check server is running: `ws://localhost:8002/ws/agent-manager`
2. Check browser console for errors
3. Verify protocol (http → ws, https → wss)
4. Check CORS headers if cross-origin

### Messages Not Received

1. Verify subscription: `wsClient.subscribe('thread:xyz')`
2. Check message type matches handler: `on('step_progress', ...)`
3. Monitor incoming messages: `on('*', (msg) => console.log(msg))`
4. Check network tab in DevTools

### Frequent Disconnections

1. Check server logs for errors
2. Increase heartbeat timeout in code
3. Check network stability
4. Monitor memory usage (message queue buildup)

### High Memory Usage

1. Reduce message buffer size: `useAgentManagerMessages(type, 10)` (not 100)
2. Unsubscribe from unused patterns
3. Clear message handlers on unmount
4. Monitor handler count

## Files

- `websocketClient.ts` (380 lines) - Core WebSocket client
- `useAgentManagerWS.ts` (200 lines) - React hooks
- `index.ts` (30 lines) - Public API exports

**Total**: ~610 lines of production code

## See Also

- [Agent Manager API Documentation](../API.md)
- [Backend WebSocket Handler](../../../hololoom/chatops/handlers/websocket_progress.py)
- [React Integration Examples](./examples/)
