# Component Integration Guide

Quick reference for using Phase 1 layout components in the Agent Manager UI.

## Quick Start

### 1. Main App Layout

```tsx
// src/App.tsx
import { Layout } from '@components';

function App() {
  return <Layout />;
}

export default App;
```

That's it! The Layout component handles:
- Sidebar toggle
- Header with navigation
- Main content panel
- All Zustand store integration

### 2. Using Status Badges

Show thread status inline:

```tsx
import { StatusBadge, StatusIndicator, StatusGrid } from '@components';

// Full badge with label and icon
<StatusBadge status="running" size="md" />
// Output: ▶ Running (blue, pulsing)

// Icon only (for tight spaces)
<StatusBadge status="running" showLabel={false} />
// Output: ▶ (just the icon)

// Compact dot indicator
<StatusIndicator status="running" />
// Output: Small pulsing blue dot

// Overview grid (for dashboards)
<StatusGrid
  running={5}
  completed={12}
  failed={2}
/>
// Output: 3 colored boxes showing counts
```

### 3. Accessing Zustand State

In any component, read/write store state:

```tsx
import { useAgentManagerStore } from '@stores';

function MyComponent() {
  // Read from store
  const { filter, viewMode, isConnected } = useAgentManagerStore();

  // Write to store
  const { setFilter, setViewMode, setConnectionStatus } = useAgentManagerStore();

  return (
    <div>
      <button onClick={() => setFilter('active')}>
        Active only
      </button>
    </div>
  );
}
```

## Component APIs

### Layout

```tsx
<Layout />
```

**Props:** None (uses Zustand internally)

**Renders:**
- Header (sticky top)
- Sidebar (toggleable left)
- MainPanel (flexible right)

---

### Header

```tsx
<Header onToggleSidebar={() => console.log('toggle')} />
```

**Props:**
- `onToggleSidebar?: () => void` - Called when hamburger clicked

**Features:**
- "+ New Thread" button (primary action)
- Filter dropdown (All/Active/Completed/Failed)
- View toggle (Outline/Tree/Swarm)
- Connection indicator (green dot = connected)

**Store Integration:**
- Reads: `filter`, `viewMode`, `isConnected`
- Writes: `setFilter()`, `setViewMode()`

---

### Sidebar

```tsx
<Sidebar />
```

**Props:** None

**Features:**
- Agent Type selector (Weaving/RAG/Agentic/Custom)
- Reasoning Mode selector (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE)
- Token Budget input (optional number)
- Priority slider (1-10)
- Create Thread button

**Current Behavior:**
- Logs thread config to console on "Create Thread"
- TODO: WebSocket integration in Phase 2

---

### MainPanel

```tsx
<MainPanel />
```

**Props:** None

**Renders Based on:**
- `viewMode` from Zustand (outline/tree/swarm)
- Currently shows placeholders for all 3 modes
- Will be replaced in Phase 3 with actual implementations

---

### StatusBadge

```tsx
<StatusBadge
  status="running"
  size="md"
  showLabel={true}
  className="custom-class"
/>
```

**Props:**
- `status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'`
- `size?: 'sm' | 'md' | 'lg'` - default: `'md'`
- `showLabel?: boolean` - Show text label - default: `true`
- `className?: string` - Extra Tailwind classes

**Status Colors:**
| Status | Color | Icon | Animation |
|--------|-------|------|-----------|
| idle | gray | ○ | None |
| running | blue | ▶ | Pulse |
| paused | amber | ⏸ | None |
| completed | green | ✓ | None |
| failed | red | ✕ | None |
| cancelled | gray | × | None |

---

### StatusIndicator

```tsx
<StatusIndicator
  status="running"
  size="md"
  className="custom-class"
/>
```

**Props:**
- `status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'`
- `size?: 'sm' | 'md' | 'lg'` - default: `'md'`
- `className?: string`

**Output:** Colored dot (2px to 3px depending on size)

---

### StatusGrid

```tsx
<StatusGrid
  idle={2}
  running={5}
  paused={1}
  completed={12}
  failed={3}
  cancelled={1}
  className="custom-class"
/>
```

**Props:**
- All counts optional (only renders non-zero counts)
- `className?: string`

**Output:** 3-column grid of colored boxes with counts

## Common Patterns

### Pattern 1: Thread List with Status

```tsx
function ThreadList() {
  const threads = useAgentManagerStore((state) => state.getFilteredThreads());

  return (
    <div className="space-y-2">
      {threads.map((thread) => (
        <div
          key={thread.id}
          className="flex items-center gap-3 p-3 bg-slate-800 rounded-md"
        >
          <StatusBadge
            status={thread.status}
            size="sm"
            showLabel={false}
          />
          <div className="flex-1">
            <div className="font-medium text-slate-100">{thread.name}</div>
            <div className="text-xs text-slate-500">
              Priority: {thread.priority}
            </div>
          </div>
          <div className="text-xs text-slate-400">
            {thread.tokensUsed} / {thread.tokenBudget ?? '∞'} tokens
          </div>
        </div>
      ))}
    </div>
  );
}
```

### Pattern 2: Dashboard Overview

```tsx
function Dashboard() {
  const threads = useAgentManagerStore((state) => Object.values(state.threads));

  const statusCounts = {
    idle: threads.filter((t) => t.status === 'idle').length,
    running: threads.filter((t) => t.status === 'running').length,
    paused: threads.filter((t) => t.status === 'paused').length,
    completed: threads.filter((t) => t.status === 'completed').length,
    failed: threads.filter((t) => t.status === 'failed' || t.status === 'cancelled').length,
  };

  return (
    <div className="p-6 space-y-4">
      <h1 className="text-2xl font-bold">Agent Swarm Overview</h1>
      <StatusGrid
        idle={statusCounts.idle}
        running={statusCounts.running}
        paused={statusCounts.paused}
        completed={statusCounts.completed}
        failed={statusCounts.failed}
        className="mb-6"
      />
    </div>
  );
}
```

### Pattern 3: Real-time Connection Status

```tsx
function ConnectionMonitor() {
  const { isConnected, connectionError } = useAgentManagerStore();

  return isConnected ? (
    <div className="text-green-500 flex items-center gap-2">
      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
      Connected
    </div>
  ) : (
    <div className="text-red-500 flex items-center gap-2">
      <div className="w-2 h-2 bg-red-500 rounded-full" />
      Offline: {connectionError || 'Unknown error'}
    </div>
  );
}
```

## Styling Customization

### Using Tailwind's @apply

```tsx
// tailwind.config.ts
theme: {
  extend: {
    components: {
      // Custom button style
      '.btn-primary': {
        '@apply px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-md font-semibold transition-colors': {},
      },
    },
  },
}

// Usage in component
<button className="btn-primary">Click me</button>
```

### Dark Mode Forced

All components assume dark mode. To force dark mode globally:

```tsx
// index.tsx or main.tsx
document.documentElement.classList.add('dark');
```

Or in Tailwind config:
```ts
export default {
  darkMode: 'class', // or 'media'
  theme: { /* ... */ }
}
```

## WebSocket Integration (Phase 2)

Once WebSocket is implemented, components will automatically update via Zustand:

```tsx
// In your WebSocket handler
import { useAgentManagerStore } from '@stores';

const ws = new WebSocket('ws://localhost:8002/ws/agents');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'thread_created') {
    const { addThread } = useAgentManagerStore();
    addThread(data.thread);
  }

  if (data.type === 'thread_updated') {
    const { updateThread } = useAgentManagerStore();
    updateThread(data.thread.id, data.thread);
  }

  if (data.type === 'connection_status') {
    const { setConnectionStatus } = useAgentManagerStore();
    setConnectionStatus(data.connected, data.error);
  }
};
```

## Performance Tips

1. **Memoize heavy components:**
   ```tsx
   const ThreadList = React.memo(({ threads }) => {
     return threads.map(t => <ThreadItem key={t.id} thread={t} />);
   });
   ```

2. **Use selectors to avoid re-renders:**
   ```tsx
   // Good: Only re-renders when viewMode changes
   const viewMode = useAgentManagerStore((state) => state.viewMode);

   // Bad: Re-renders on any store change
   const store = useAgentManagerStore();
   const viewMode = store.viewMode;
   ```

3. **Batch store updates:**
   ```tsx
   const { updateThread, updateConnection } = useAgentManagerStore();

   // Instead of separate calls:
   updateThread(id, { status: 'completed' });
   updateConnection(true);

   // Consider batch updates in the store
   ```

## Troubleshooting

### Components not appearing

1. Check Tailwind CSS is properly configured
2. Verify `@components` alias in `vite.config.ts`
3. Ensure `PYTHONPATH` set when running Vite: `PYTHONPATH=. npm run dev`

### Store not updating

1. Verify `useAgentManagerStore` hook is imported correctly
2. Check that Zustand state is being mutated (use Immer middleware)
3. In React DevTools, check Zustand tab for state changes

### Styling issues

1. Dark theme might be overridden by global CSS
2. Ensure Tailwind's `@layer` directives are correct
3. Check specificity of custom CSS classes

## Next Steps

See [COMPONENTS_PHASE_1_COMPLETE.md](../COMPONENTS_PHASE_1_COMPLETE.md) for full documentation.

Phase 2 will add:
- Thread list rendering
- WebSocket connection
- Real-time updates
- Thread detail panel

---

**Last Updated:** December 2025
**Status:** Phase 1 Complete ✅
