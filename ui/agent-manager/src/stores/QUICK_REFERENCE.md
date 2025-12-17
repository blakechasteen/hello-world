# Agent Manager Store - Quick Reference

## Import

```typescript
import { useAgentManagerStore, useThreadWithDependencies, useSwarmOverview, useActiveThreadDetails } from '@/stores';
import type { AgentThread, AgentManagerState } from '@/stores';
```

## Common Patterns

### Add a New Thread
```typescript
const addThread = useAgentManagerStore((state) => state.addThread);

addThread({
  id: 'thread-001',
  name: 'Research Query',
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

### Update Thread Progress
```typescript
const updateThread = useAgentManagerStore((state) => state.updateThread);

updateThread('thread-001', {
  currentStep: 2,
  elapsedTimeMs: 1500,
  tokensUsed: 450,
  confidence: 0.87,
});
// updatedAt is set automatically
```

### Handle Thread State Changes
```typescript
const pauseThread = useAgentManagerStore((state) => state.pauseThread);
const resumeThread = useAgentManagerStore((state) => state.resumeThread);
const cancelThread = useAgentManagerStore((state) => state.cancelThread);

pauseThread('thread-001');   // running -> paused
resumeThread('thread-001');  // paused -> running
cancelThread('thread-001');  // any -> cancelled
```

### Manage Thread Priority (Voting)
```typescript
const upvoteThread = useAgentManagerStore((state) => state.upvoteThread);
const downvoteThread = useAgentManagerStore((state) => state.downvoteThread);

upvoteThread('thread-001');    // priority += 1 (max 100)
downvoteThread('thread-001');  // priority -= 1 (min 0)
```

### Filter & View Threads
```typescript
const setFilter = useAgentManagerStore((state) => state.setFilter);
const setViewMode = useAgentManagerStore((state) => state.setViewMode);
const setActiveThread = useAgentManagerStore((state) => state.setActiveThread);

setFilter('active');        // Show running/paused
setFilter('completed');     // Show completed
setFilter('failed');        // Show failed/cancelled
setFilter('all');           // Show all

setViewMode('tree');        // Hierarchical view
setViewMode('swarm');       // Dependency graph
setViewMode('outline');     // Linear list

setActiveThread('thread-001'); // Select thread
setActiveThread(null);      // Deselect
```

### Query Threads
```typescript
const getFilteredThreads = useAgentManagerStore((state) => state.getFilteredThreads);
const getThreadById = useAgentManagerStore((state) => state.getThreadById);
const getActiveThread = useAgentManagerStore((state) => state.getActiveThread);
const getChildThreads = useAgentManagerStore((state) => state.getChildThreads);
const getThreadsBySwarm = useAgentManagerStore((state) => state.getThreadsBySwarm);
const getThreadDependencies = useAgentManagerStore((state) => state.getThreadDependencies);
const getSwarmStatus = useAgentManagerStore((state) => state.getSwarmStatus);

const threads = getFilteredThreads();
const thread = getThreadById('thread-001');
const active = getActiveThread();
const children = getChildThreads('parent-thread-001');
const swarmThreads = getThreadsBySwarm('swarm-001');
const { dependsOn, blocks } = getThreadDependencies('thread-001');
const status = getSwarmStatus('swarm-001');
```

### Connection Management
```typescript
const setConnectionStatus = useAgentManagerStore((state) => state.setConnectionStatus);

setConnectionStatus(true);                    // Connected
setConnectionStatus(false, 'Connection lost'); // Disconnected
```

### Use Composite Hooks
```typescript
// For a specific thread with dependencies
const { thread, dependencies, children } = useThreadWithDependencies('thread-001');

// For a swarm overview
const { threads, status } = useSwarmOverview('swarm-001');

// For active thread with all context
const { thread, dependencies, children } = useActiveThreadDetails();
```

## Subscription Examples

### Subscribe to Filtered Threads
```typescript
function ThreadList() {
  const threads = useAgentManagerStore((state) => state.getFilteredThreads());

  return <ul>{threads.map(t => <li key={t.id}>{t.name}</li>)}</ul>;
}
```

### Subscribe to Specific Thread
```typescript
function ThreadDetail({ threadId }: { threadId: string }) {
  const thread = useAgentManagerStore((state) => state.getThreadById(threadId));

  if (!thread) return <div>Not found</div>;
  return <div>{thread.name} - {thread.status}</div>;
}
```

### Subscribe to Active Thread
```typescript
function ActiveThreadPanel() {
  const thread = useAgentManagerStore((state) => state.getActiveThread());

  if (!thread) return <div>Select a thread</div>;
  return <div>{thread.name}</div>;
}
```

### Subscribe to UI State
```typescript
function FilterBar() {
  const [filter, setFilter] = useAgentManagerStore((state) => [
    state.filter,
    state.setFilter,
  ]);

  return (
    <div>
      <p>Current filter: {filter}</p>
      <button onClick={() => setFilter('active')}>Active</button>
      <button onClick={() => setFilter('all')}>All</button>
    </div>
  );
}
```

## State Structure

```typescript
{
  // Thread data (stored by ID for O(1) lookup)
  threads: {
    'thread-001': {
      id: 'thread-001',
      name: 'Research Agent',
      status: 'running',
      priority: 50,
      agentType: 'research',
      reasoningMode: 'RESEARCH',
      currentStep: 2,
      totalSteps: 5,
      elapsedTimeMs: 2500,
      tokensUsed: 450,
      tokenBudget: 10000,
      confidence: 0.85,
      epistemicConfidence: 0.7,
      swarmId: 'swarm-001',
      parentThreadId: undefined,
      childThreadIds: ['thread-002', 'thread-003'],
      dependsOn: [],
      blocks: ['thread-004'],
      finalResponse: undefined,
      createdAt: '2024-...',
      updatedAt: '2024-...',
    },
    // ... more threads
  },

  // UI state
  activeThreadId: 'thread-001',
  filter: 'active',
  viewMode: 'tree',

  // Connection state
  isConnected: true,
  connectionError: null,
}
```

## Reasoning Modes

```typescript
type ReasoningMode =
  | 'DIRECT'        // Single-pass answer (~150ms)
  | 'VERIFY'        // Answer + verification (~600ms)
  | 'RESEARCH'      // Multi-query exploration (~900ms)
  | 'PLAN_EXECUTE'  // Goal decomposition (~750ms)
```

## Thread Status Transitions

```
idle ──┬──> running ──┬──> completed ✓
       │              │
       │              └──> paused <──┐
       │                     │       │
       │                     └───────┘
       │
       └──────────────────> cancelled ✗

       failed ✗
```

## View Modes

```typescript
'outline' // Linear list view - simple thread list
'tree'    // Hierarchical tree - parent/child relationships
'swarm'   // Dependency graph - complex dependencies visualized
```

## Filter Types

```typescript
'all'       // Show all threads
'active'    // Show running or paused threads
'completed' // Show completed threads
'failed'    // Show failed or cancelled threads
```

## Performance Tips

1. **Selective Subscriptions**: Only subscribe to the exact data you need
   ```typescript
   // Good
   const threads = useAgentManagerStore((state) => state.getFilteredThreads());

   // Avoid
   const store = useAgentManagerStore();
   ```

2. **Use Memoization**: Memoize callbacks and complex selectors
   ```typescript
   const selectThread = useCallback((id: string) =>
     useAgentManagerStore((state) => state.getThreadById(id)),
   []);
   ```

3. **Avoid Inline Selectors**: Define selectors outside component
   ```typescript
   // Good
   const selectThread = (state) => state.getThreadById('thread-001');
   const thread = useAgentManagerStore(selectThread);

   // Avoid
   const thread = useAgentManagerStore((state) => state.getThreadById('thread-001'));
   ```

## TypeScript Typing

```typescript
import type { AgentThread, AgentManagerState } from '@/stores';

// Type a component that uses the store
function MyComponent({ thread }: { thread: AgentThread }) {
  return <div>{thread.name}</div>;
}

// Type action parameters
function updateThreadStatus(id: string, status: AgentThread['status']) {
  useAgentManagerStore((state) => state.updateThread(id, { status }));
}
```

## Common Errors & Solutions

### "Cannot read property 'name' of undefined"
**Problem**: Thread not found (might be recently removed)
**Solution**: Always check if thread exists before accessing properties
```typescript
const thread = useAgentManagerStore((state) => state.getThreadById(id));
if (!thread) return <div>Thread not found</div>;
```

### "Action not updating state"
**Problem**: Forgot that Immer requires mutations within the action
**Solution**: Remember that updateThread already handles immutability
```typescript
// Correct
updateThread('id', { status: 'paused' });

// Not recommended (unnecessary wrapping)
updateThread('id', { ...thread, status: 'paused' });
```

### Component not re-rendering
**Problem**: Selector returns reference that hasn't changed
**Solution**: Use shallow equality or custom comparison
```typescript
// Ensure selector returns different reference when needed
const thread = useAgentManagerStore(
  (state) => state.getThreadById(id),
  (prev, next) => prev?.updatedAt === next?.updatedAt // Custom comparison
);
```
