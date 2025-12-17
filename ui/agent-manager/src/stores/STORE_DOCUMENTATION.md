# Agent Manager Zustand Store Documentation

## Overview

The Agent Manager store (`agentManagerStore.ts`) provides comprehensive state management for the Agent Swarm UI using Zustand with Immer middleware for immutable updates.

## Store Location

- **Main Store**: `ui/agent-manager/src/stores/agentManagerStore.ts` (289 lines)
- **Exports**: `ui/agent-manager/src/stores/index.ts`

## Data Structures

### AgentThread Interface

Represents a single agent reasoning thread in the swarm:

```typescript
interface AgentThread {
  id: string;                                    // Unique thread identifier
  name: string;                                  // User-friendly thread name
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  priority: number;                              // 0-100, for upvoting/downvoting
  agentType: string;                             // Type of agent (e.g., "research", "analysis")
  reasoningMode: 'DIRECT' | 'VERIFY' | 'RESEARCH' | 'PLAN_EXECUTE';
  currentStep: number;                           // Current progress (0-based)
  totalSteps: number;                            // Total steps in reasoning
  elapsedTimeMs: number;                         // Elapsed execution time
  tokensUsed: number;                            // Tokens consumed so far
  tokenBudget?: number;                          // Optional token budget limit
  confidence: number;                            // 0.0-1.0 confidence in current state
  epistemicConfidence: number;                   // 0.0-1.0 epistemic uncertainty awareness
  swarmId?: string;                              // Parent swarm (if part of swarm)
  parentThreadId?: string;                       // Parent thread (if hierarchical)
  childThreadIds: string[];                      // List of child thread IDs
  dependsOn: string[];                           // Thread IDs this depends on
  blocks: string[];                              // Thread IDs this blocks
  finalResponse?: string;                        // Final response (when completed)
  createdAt: string;                             // ISO timestamp
  updatedAt: string;                             // ISO timestamp
}
```

### AgentManagerState Interface

Manages overall UI state:

```typescript
interface AgentManagerState {
  // State
  threads: Record<string, AgentThread>;          // All threads by ID
  activeThreadId: string | null;                 // Currently selected thread
  filter: 'all' | 'active' | 'completed' | 'failed';
  viewMode: 'outline' | 'tree' | 'swarm';
  isConnected: boolean;                          // WebSocket/API connection status
  connectionError: string | null;                // Connection error message

  // Actions (mutation methods)
  addThread: (thread: AgentThread) => void;
  updateThread: (id: string, updates: Partial<AgentThread>) => void;
  removeThread: (id: string) => void;
  setActiveThread: (id: string | null) => void;
  setFilter: (filter: ...) => void;
  setViewMode: (mode: ...) => void;
  setConnectionStatus: (connected: boolean, error?: string) => void;
  pauseThread: (id: string) => void;
  resumeThread: (id: string) => void;
  cancelThread: (id: string) => void;
  upvoteThread: (id: string) => void;             // Increment priority (max 100)
  downvoteThread: (id: string) => void;           // Decrement priority (min 0)

  // Selectors (computed/query methods)
  getFilteredThreads: () => AgentThread[];
  getActiveThread: () => AgentThread | undefined;
  getThreadById: (id: string) => AgentThread | undefined;
  getChildThreads: (parentId: string) => AgentThread[];
  getThreadsBySwarm: (swarmId: string) => AgentThread[];
  getThreadDependencies: (id: string) => { dependsOn: AgentThread[]; blocks: AgentThread[] };
  getSwarmStatus: (swarmId: string) => { total, running, completed, failed, avgConfidence };
}
```

## Core Features

### 1. Thread Management

#### Add Thread
```typescript
const addThread = useAgentManagerStore((state) => state.addThread);
addThread({
  id: 'thread-1',
  name: 'Research Agent',
  status: 'idle',
  priority: 50,
  agentType: 'research',
  reasoningMode: 'RESEARCH',
  currentStep: 0,
  totalSteps: 5,
  elapsedTimeMs: 0,
  tokensUsed: 0,
  confidence: 0.0,
  epistemicConfidence: 0.5,
  childThreadIds: [],
  dependsOn: [],
  blocks: [],
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
});
```

#### Update Thread
```typescript
const updateThread = useAgentManagerStore((state) => state.updateThread);
updateThread('thread-1', {
  currentStep: 2,
  elapsedTimeMs: 150,
  tokensUsed: 250,
  confidence: 0.85,
});
// Note: updatedAt is automatically set to current time
```

#### Remove Thread
```typescript
const removeThread = useAgentManagerStore((state) => state.removeThread);
removeThread('thread-1');
// Note: If thread is active, activeThreadId is cleared
```

### 2. Thread State Transitions

#### Pause/Resume
```typescript
const pauseThread = useAgentManagerStore((state) => state.pauseThread);
const resumeThread = useAgentManagerStore((state) => state.resumeThread);

pauseThread('thread-1');  // running -> paused
resumeThread('thread-1'); // paused -> running
```

#### Cancel
```typescript
const cancelThread = useAgentManagerStore((state) => state.cancelThread);
cancelThread('thread-1'); // running/paused -> cancelled
```

### 3. Priority Management (Voting)

```typescript
const upvoteThread = useAgentManagerStore((state) => state.upvoteThread);
const downvoteThread = useAgentManagerStore((state) => state.downvoteThread);

upvoteThread('thread-1');     // priority = min(100, priority + 1)
downvoteThread('thread-1');   // priority = max(0, priority - 1)
```

### 4. UI Control

#### Filter Threads
```typescript
const setFilter = useAgentManagerStore((state) => state.setFilter);
setFilter('active');          // Show running/paused threads only
setFilter('completed');       // Show completed threads
setFilter('failed');          // Show failed/cancelled threads
setFilter('all');             // Show all threads
```

#### Change View Mode
```typescript
const setViewMode = useAgentManagerStore((state) => state.setViewMode);
setViewMode('tree');          // Hierarchical tree view
setViewMode('swarm');         // Swarm dependency graph
setViewMode('outline');       // Linear outline
```

#### Set Active Thread
```typescript
const setActiveThread = useAgentManagerStore((state) => state.setActiveThread);
setActiveThread('thread-1');  // Select for details view
setActiveThread(null);        // Deselect
```

### 5. Connection Management

```typescript
const setConnectionStatus = useAgentManagerStore((state) => state.setConnectionStatus);
setConnectionStatus(true);                    // Connected
setConnectionStatus(false, 'Connection lost'); // Disconnected with error
```

## Selectors (Query Methods)

### Basic Selectors

```typescript
const store = useAgentManagerStore();

// Get filtered threads based on current filter setting
const threads = store.getFilteredThreads();

// Get currently active thread
const activeThread = store.getActiveThread();

// Get thread by ID
const thread = store.getThreadById('thread-1');
```

### Hierarchy Navigation

```typescript
// Get child threads of a parent
const children = store.getChildThreads('parent-thread-1');

// Get all threads in a swarm
const swarmThreads = store.getThreadsBySwarm('swarm-1');

// Get thread dependencies
const { dependsOn, blocks } = store.getThreadDependencies('thread-1');
```

### Analytics

```typescript
// Get swarm status summary
const status = store.getSwarmStatus('swarm-1');
// Returns: { total: 5, running: 2, completed: 2, failed: 1, avgConfidence: 0.82 }
```

## Composite Hooks

The store provides specialized hooks for common use cases:

### useThreadWithDependencies
```typescript
import { useThreadWithDependencies } from '@/stores';

function ThreadCard({ threadId }: { threadId: string }) {
  const { thread, dependencies, children } = useThreadWithDependencies(threadId);

  if (!thread) return <div>Thread not found</div>;

  return (
    <div>
      <h3>{thread.name}</h3>
      <p>Depends on: {dependencies.dependsOn.map(t => t.name).join(', ')}</p>
      <p>Blocks: {dependencies.blocks.map(t => t.name).join(', ')}</p>
      <p>Children: {children.length}</p>
    </div>
  );
}
```

### useSwarmOverview
```typescript
import { useSwarmOverview } from '@/stores';

function SwarmDashboard({ swarmId }: { swarmId: string }) {
  const { threads, status } = useSwarmOverview(swarmId);

  return (
    <div>
      <h2>{status.total} threads</h2>
      <p>Running: {status.running} | Completed: {status.completed} | Failed: {status.failed}</p>
      <p>Avg Confidence: {(status.avgConfidence * 100).toFixed(1)}%</p>
      <ul>
        {threads.map(t => (
          <li key={t.id}>{t.name} - {t.status}</li>
        ))}
      </ul>
    </div>
  );
}
```

### useActiveThreadDetails
```typescript
import { useActiveThreadDetails } from '@/stores';

function DetailPanel() {
  const { thread, dependencies, children } = useActiveThreadDetails();

  if (!thread) return <div>No thread selected</div>;

  return (
    <div>
      <h3>{thread.name}</h3>
      <p>Status: {thread.status}</p>
      <p>Progress: {thread.currentStep} / {thread.totalSteps}</p>
      {/* ... render dependencies and children ... */}
    </div>
  );
}
```

## Usage in React Components

### Basic Usage
```typescript
import { useAgentManagerStore } from '@/stores';

function ThreadList() {
  // Subscribe to filtered threads
  const threads = useAgentManagerStore((state) => state.getFilteredThreads());
  const setFilter = useAgentManagerStore((state) => state.setFilter);

  return (
    <div>
      <button onClick={() => setFilter('active')}>Active</button>
      <button onClick={() => setFilter('completed')}>Completed</button>
      <ul>
        {threads.map(t => (
          <li key={t.id}>{t.name} - {t.status}</li>
        ))}
      </ul>
    </div>
  );
}
```

### Selective Subscription
```typescript
function ThreadCard({ threadId }: { threadId: string }) {
  // Only re-render when this specific thread changes
  const thread = useAgentManagerStore(
    (state) => state.getThreadById(threadId),
    (prev, next) => prev?.id === next?.id && prev?.updatedAt === next?.updatedAt
  );

  if (!thread) return null;

  return <div>{thread.name} - {thread.status}</div>;
}
```

### Batch Updates
```typescript
function updateProgress(threadId: string, step: number, time: number) {
  useAgentManagerStore((state) =>
    state.updateThread(threadId, {
      currentStep: step,
      elapsedTimeMs: time,
    })
  );
}
```

## Immer Middleware Benefits

The store uses Zustand's Immer middleware for automatic immutable updates:

1. **Draft Mutations**: You can mutate state directly in action callbacks
2. **Automatic Cloning**: Immer automatically creates new references where needed
3. **Deep Updates**: Complex nested updates work naturally
4. **Immutability Guarantee**: Final state is always immutable

Example (handled automatically by Immer):
```typescript
// Instead of this (without Immer):
state.threads[id] = { ...state.threads[id], status: 'paused' };

// You can do this (with Immer):
state.threads[id].status = 'paused';
// Immer ensures proper immutability behind the scenes
```

## Performance Considerations

### Selective Subscriptions

Always subscribe to only the data you need:

```typescript
// ✅ Good: Only subscribe to filtered threads
const threads = useAgentManagerStore((state) => state.getFilteredThreads());

// ❌ Avoid: Subscribing to entire state
const store = useAgentManagerStore();
const threads = store.getFilteredThreads();
```

### Memoization

Use `useCallback` for callbacks passed to threads:

```typescript
const onThreadSelect = useCallback(
  (id: string) => {
    useAgentManagerStore((state) => state.setActiveThread(id));
  },
  []
);
```

## Testing

The store can be tested independently:

```typescript
import { renderHook, act } from '@testing-library/react';
import { useAgentManagerStore } from '@/stores';

test('adding a thread', () => {
  const { result } = renderHook(() => useAgentManagerStore());

  act(() => {
    result.current.addThread({
      id: 'test-1',
      name: 'Test Thread',
      status: 'idle',
      // ... other required fields
    });
  });

  expect(result.current.threads['test-1']).toBeDefined();
});
```

## Future Extensions

The store is designed to support:

1. **Persistent Storage**: Can add zustand-persist middleware
2. **Undo/Redo**: Can add zustand-immer undo-middleware
3. **WebSocket Sync**: Easily integrate real-time updates
4. **Local Storage**: Save/restore state across sessions
5. **DevTools Integration**: Zustand has built-in Redux DevTools support

## Files

- `/ui/agent-manager/src/stores/agentManagerStore.ts` - Main store implementation (289 lines)
- `/ui/agent-manager/src/stores/index.ts` - Central export point (8 lines)
- `/ui/agent-manager/src/stores/STORE_DOCUMENTATION.md` - This file
