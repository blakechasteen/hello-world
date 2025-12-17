# Agent Manager Store - Integration Guide

**Date**: December 11, 2025
**Version**: 1.0.0
**Status**: Production Ready

## Overview

This guide explains how to integrate the new `useAgentManagerStore` (thread-specific) with the existing `useAppStore` (global app state).

## Store Architecture

The Agent Manager UI has **two complementary stores**:

### 1. Global App Store (`appStore.ts`)
**Purpose**: Application-wide state
- Agent list and selection
- Tasks queue and execution
- System metrics and logs
- UI configuration (sidebar, view mode, auto-refresh)
- Connection status

**When to use**: App-level features, navigation, global settings

### 2. Agent Manager Store (`agentManagerStore.ts`)
**Purpose**: Detailed thread management
- Individual thread state and progress
- Thread hierarchy (parent/child relationships)
- Thread dependencies (depends-on, blocks)
- Fine-grained filtering and views
- Swarm-specific analytics

**When to use**: Thread details, progress tracking, dependency visualization

## Integration Patterns

### Pattern 1: Thread Creation Flow

**Global Store**: Initial agent/task creation
**Agent Manager Store**: Detailed thread tracking

```typescript
// Global store: User selects an agent and creates a task
function TaskCreator() {
  const selectedAgent = useSelectedAgentData();
  const addTask = useAppStore((state) => state.addTask);

  const handleCreateTask = (taskSpec: TaskSpec) => {
    // Create task in global store
    const task = createTask(taskSpec);
    addTask(task);

    // Create corresponding thread(s) in agent manager store
    const threadSpec = convertTaskToThread(task, selectedAgent);
    useAgentManagerStore((state) => state.addThread(threadSpec));
  };

  return <div>{/* ... */}</div>;
}
```

### Pattern 2: Progress Tracking

**Global Store**: High-level task status
**Agent Manager Store**: Detailed thread progress

```typescript
// WebSocket handler: Update progress
ws.on('message', (event) => {
  if (event.type === 'progress') {
    const { taskId, threadId, step, time, confidence } = event;

    // Update global task status
    useAppStore((state) => state.updateTask(taskId, {
      progress: step,
      status: step === totalSteps ? 'completed' : 'running',
    }));

    // Update detailed thread progress
    useAgentManagerStore((state) => state.updateThread(threadId, {
      currentStep: step,
      elapsedTimeMs: time,
      confidence,
    }));
  }
});
```

### Pattern 3: UI Navigation

**Global Store**: View mode and selection
**Agent Manager Store**: Thread-specific details

```typescript
function AgentManagerUI() {
  const selectedAgentId = useSelectedAgent();
  const currentView = useCurrentView();

  // Agent Manager store for thread details
  const threads = useAgentManagerStore((state) => state.getFilteredThreads());
  const setViewMode = useAgentManagerStore((state) => state.setViewMode);
  const setActiveThread = useAgentManagerStore((state) => state.setActiveThread);

  // Switch to detail view when an agent is selected
  useEffect(() => {
    if (selectedAgentId && currentView !== 'agent_manager') {
      // Update global view
      useAppStore((state) => state.setCurrentView('agent_manager'));

      // Filter threads for this agent
      // (could add agentId field to threads for filtering)
    }
  }, [selectedAgentId]);

  return (
    <div>
      <ThreadList threads={threads} />
      <ThreadDetailPanel />
    </div>
  );
}
```

### Pattern 4: Error Handling

**Global Store**: Log errors
**Agent Manager Store**: Update thread status

```typescript
ws.on('error', (event) => {
  const { threadId, error } = event;

  // Log error in global store
  useAppStore((state) =>
    state.addLog('ThreadManager', error, 'error')
  );

  // Update thread status in agent manager store
  useAgentManagerStore((state) => state.updateThread(threadId, {
    status: 'failed',
    finalResponse: `Error: ${error}`,
  }));
});
```

## Component Organization

### Recommended Component Hierarchy

```
App
├── GlobalUILayout
│   ├── Sidebar (uses appStore)
│   ├── TopBar (uses appStore)
│   │   └── ConnectionIndicator (uses agentManagerStore)
│   └── ViewContainer
│       ├── DashboardView (uses appStore)
│       └── AgentManagerView
│           ├── ThreadList (uses agentManagerStore)
│           ├── ThreadDetailPanel (uses agentManagerStore)
│           └── ControlPanel (uses both stores)
```

### Example Component Integration

**Component**: `AgentManagerView.tsx` (uses both stores)

```typescript
import { useAppStore } from '@/stores/appStore';
import { useAgentManagerStore, useActiveThreadDetails } from '@/stores';

export function AgentManagerView() {
  // Global state
  const selectedAgentId = useAppStore((state) => state.selectedAgentId);

  // Agent manager state
  const threads = useAgentManagerStore((state) =>
    state.getThreadsBySwarm(selectedAgentId || '')
  );
  const viewMode = useAgentManagerStore((state) => state.viewMode);
  const { thread: activeThread } = useActiveThreadDetails();

  return (
    <div className="agent-manager">
      <div className="threads-list">
        <ThreadListView threads={threads} viewMode={viewMode} />
      </div>

      {activeThread && (
        <div className="thread-details">
          <ThreadDetailView thread={activeThread} />
        </div>
      )}
    </div>
  );
}
```

## Data Flow Examples

### Example 1: Create and Monitor a Thread

```
User Action (Component)
↓
Global Store: addTask()
↓
WebSocket: Send task_created event
↓
Server: Start processing
↓
WebSocket: Receive progress events
↓
Agent Manager Store: updateThread() [periodic updates]
↓
UI Components: Re-render with new progress
↓
Server: Complete/fail
↓
Agent Manager Store: updateThread() [final status]
↓
UI: Show final response
```

### Example 2: Manage Thread Priorities

```
User Upvotes Thread (Component)
↓
Agent Manager Store: upvoteThread() [priority++]
↓
Component: Re-render with new priority
↓
User Can Optionally: Send priority to server
↓
Server: Adjust scheduling (if supported)
```

### Example 3: Switch Views

```
User Clicks "View Mode: Tree" Button
↓
Agent Manager Store: setViewMode('tree')
↓
Component: Gets new viewMode from selector
↓
ThreadListView: Re-render in tree layout
```

## WebSocket Integration

### Setup: Connect global store with WebSocket

```typescript
// hooks/useWebSocket.ts
import { useEffect } from 'react';
import { useAppStore } from '@/stores/appStore';
import { useAgentManagerStore } from '@/stores';

export function useWebSocket(url: string) {
  const setConnected = useAppStore((state) => state.setConnected);
  const addLog = useAppStore((state) => state.addLog);
  const updateThread = useAgentManagerStore((state) => state.updateThread);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      addLog('WebSocket', 'Connected', 'info');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'thread_update':
          updateThread(data.threadId, data.updates);
          break;

        case 'thread_completed':
          updateThread(data.threadId, {
            status: 'completed',
            finalResponse: data.response,
            confidence: data.confidence,
          });
          break;

        case 'thread_failed':
          updateThread(data.threadId, {
            status: 'failed',
            finalResponse: `Error: ${data.error}`,
          });
          addLog('Thread', data.error, 'error');
          break;
      }
    };

    ws.onerror = () => {
      setConnected(false);
      addLog('WebSocket', 'Connection error', 'error');
    };

    return () => ws.close();
  }, [url, setConnected, addLog, updateThread]);
}
```

### Setup: Use WebSocket hook in App

```typescript
function App() {
  useWebSocket(process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws');

  return (
    <div className="app">
      {/* App content */}
    </div>
  );
}
```

## Performance Optimization

### 1. Use Selective Subscriptions

```typescript
// ❌ Bad: Subscribes to entire store
const store = useAgentManagerStore();
const threads = store.getFilteredThreads();

// ✅ Good: Only subscribes to filtered threads
const threads = useAgentManagerStore((state) => state.getFilteredThreads());
```

### 2. Memoize Component Props

```typescript
// ✅ Good: Memoize thread list to prevent unnecessary re-renders
import { useMemo } from 'react';

function ThreadListContainer() {
  const threads = useAgentManagerStore((state) => state.getFilteredThreads());

  const memoizedThreads = useMemo(() => threads, [threads]);

  return <ThreadList threads={memoizedThreads} />;
}
```

### 3. Use Composite Hooks

```typescript
// ✅ Good: Use pre-built composite hooks
const { thread, dependencies } = useActiveThreadDetails();

// Instead of multiple separate hooks
const thread = useAgentManagerStore((state) => state.getActiveThread());
const dependencies = useAgentManagerStore((state) =>
  state.getThreadDependencies(thread?.id || '')
);
```

## Testing

### Unit Test Example

```typescript
import { renderHook, act } from '@testing-library/react';
import { useAgentManagerStore } from '@/stores';

describe('useAgentManagerStore', () => {
  test('addThread adds thread to store', () => {
    const { result } = renderHook(() => useAgentManagerStore());

    act(() => {
      result.current.addThread({
        id: 'test-1',
        name: 'Test Thread',
        status: 'idle',
        // ... other fields
      });
    });

    expect(result.current.threads['test-1']).toBeDefined();
  });

  test('updateThread updates existing thread', () => {
    const { result } = renderHook(() => useAgentManagerStore());

    // Setup
    act(() => {
      result.current.addThread({
        id: 'test-1',
        name: 'Test',
        status: 'idle',
        currentStep: 0,
        // ... other fields
      });
    });

    // Update
    act(() => {
      result.current.updateThread('test-1', {
        status: 'running',
        currentStep: 1,
      });
    });

    expect(result.current.threads['test-1'].status).toBe('running');
    expect(result.current.threads['test-1'].currentStep).toBe(1);
  });
});
```

## TypeScript Best Practices

### 1. Type Thread Operations

```typescript
function updateProgress(
  threadId: string,
  step: number,
  time: number
): void {
  const updateThread = useAgentManagerStore((state) => state.updateThread);

  updateThread(threadId, {
    currentStep: step,
    elapsedTimeMs: time,
  });
}
```

### 2. Type Derived Selectors

```typescript
const selectActiveThreadWithContext = (
  state: typeof useAgentManagerStore.getState()
): ThreadWithContext | null => {
  const thread = state.getActiveThread();
  if (!thread) return null;

  const deps = state.getThreadDependencies(thread.id);
  const children = state.getChildThreads(thread.id);

  return { thread, dependencies: deps, children };
};

const activeThreadWithContext = useAgentManagerStore(selectActiveThreadWithContext);
```

### 3. Type WebSocket Event Handlers

```typescript
interface ThreadProgressEvent {
  type: 'thread_progress';
  threadId: string;
  updates: Partial<AgentThread>;
}

function handleThreadEvent(event: ThreadProgressEvent): void {
  useAgentManagerStore((state) =>
    state.updateThread(event.threadId, event.updates)
  );
}
```

## Common Integration Tasks

### Task 1: Display Thread Progress

```typescript
function ThreadProgressBar({ threadId }: { threadId: string }) {
  const thread = useAgentManagerStore((state) =>
    state.getThreadById(threadId)
  );

  if (!thread) return null;

  const progress = (thread.currentStep / thread.totalSteps) * 100;

  return (
    <div className="progress-bar">
      <div className="progress-fill" style={{ width: `${progress}%` }} />
      <span className="progress-text">
        {thread.currentStep} / {thread.totalSteps}
      </span>
    </div>
  );
}
```

### Task 2: Handle Thread Completion

```typescript
function handleThreadComplete(threadId: string, response: string) {
  useAgentManagerStore((state) =>
    state.updateThread(threadId, {
      status: 'completed',
      finalResponse: response,
    })
  );

  // Optionally update global task
  const thread = useAgentManagerStore.getState().getThreadById(threadId);
  if (thread?.parentThreadId) {
    useAppStore((state) =>
      state.updateTask(thread.parentThreadId, { status: 'completed' })
    );
  }
}
```

### Task 3: Pause/Resume Thread

```typescript
function ThreadControls({ threadId }: { threadId: string }) {
  const thread = useAgentManagerStore((state) =>
    state.getThreadById(threadId)
  );
  const pauseThread = useAgentManagerStore((state) => state.pauseThread);
  const resumeThread = useAgentManagerStore((state) => state.resumeThread);

  if (!thread) return null;

  const canControl =
    thread.status === 'running' || thread.status === 'paused';

  if (!canControl) return null;

  return (
    <button
      onClick={() => {
        if (thread.status === 'running') {
          pauseThread(threadId);
        } else {
          resumeThread(threadId);
        }
      }}
    >
      {thread.status === 'running' ? 'Pause' : 'Resume'}
    </button>
  );
}
```

## Migration Checklist

If migrating from an existing thread management system:

- [ ] Update imports to use new store
- [ ] Replace old thread state with store subscriptions
- [ ] Update WebSocket handlers to use store actions
- [ ] Replace manual state updates with store actions
- [ ] Test all thread operations
- [ ] Verify filtering/view modes work correctly
- [ ] Test thread creation and deletion
- [ ] Verify parent/child relationships display correctly
- [ ] Test dependency visualization
- [ ] Load test with 100+ threads

## Troubleshooting

### Issue: Components not updating when thread changes

**Solution**: Use selective subscriptions
```typescript
// Instead of entire thread
const thread = useAgentManagerStore((state) => state.getThreadById(id));

// Subscribe only to the properties you need
const threadName = useAgentManagerStore((state) =>
  state.getThreadById(id)?.name
);
```

### Issue: Performance degradation with many threads

**Solution**: Implement virtualization
```typescript
import { FixedSizeList as List } from 'react-window';

const threads = useAgentManagerStore((state) => state.getFilteredThreads());

return (
  <List
    height={600}
    itemCount={threads.length}
    itemSize={50}
  >
    {({ index, style }) => (
      <div style={style}>
        <ThreadRow thread={threads[index]} />
      </div>
    )}
  </List>
);
```

### Issue: Memory leak with WebSocket updates

**Solution**: Cleanup subscriptions
```typescript
useEffect(() => {
  const unsubscribe = useAgentManagerStore.subscribe(
    (state) => state.threads,
    (threads) => {
      // Handle threads update
    }
  );

  return () => unsubscribe();
}, []);
```

## Resources

- **Main Documentation**: `STORE_DOCUMENTATION.md`
- **Quick Reference**: `QUICK_REFERENCE.md`
- **Code Examples**: `examples.tsx`
- **Global Store**: `appStore.ts`

---

**Version**: 1.0.0
**Last Updated**: 2025-12-11
**Status**: Complete and verified ✅
