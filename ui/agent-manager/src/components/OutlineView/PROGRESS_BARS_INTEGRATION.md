# ProgressBars Integration Guide

Complete guide for integrating ProgressBars components into HoloLoom Agent Manager UI Phase 3.

## Quick Start

### 1. Import the Components

```typescript
import { ProgressBar, ProgressBars } from '@components/OutlineView';
```

### 2. Basic Usage

```tsx
<ProgressBars
  currentStep={3}
  totalSteps={7}
  elapsedTimeMs={4500}
  timeBudgetMs={10500}
  tokensUsed={750}
  tokenBudget={1750}
/>
```

### 3. With Thread Data

```tsx
const thread: AgentThread = {
  id: 'thread-1',
  currentStep: 3,
  totalSteps: 7,
  elapsedTimeMs: 4500,
  tokensUsed: 750,
  tokenBudget: 1750,
};

<ProgressBars
  currentStep={thread.currentStep}
  totalSteps={thread.totalSteps}
  elapsedTimeMs={thread.elapsedTimeMs}
  timeBudgetMs={10500}
  tokensUsed={thread.tokensUsed}
  tokenBudget={thread.tokenBudget}
/>
```

## Integration Scenarios

### Scenario 1: Thread-Level Progress (ThreadCard)

Show overall progress for an agent thread.

```tsx
import { ProgressBars } from '@components/OutlineView';

interface ThreadCardProps {
  thread: AgentThread;
  isActive?: boolean;
  onSelect?: (threadId: string) => void;
}

export const ThreadCard: React.FC<ThreadCardProps> = ({
  thread,
  isActive,
  onSelect,
}) => {
  return (
    <div
      onClick={() => onSelect?.(thread.id)}
      className={`
        p-3 border rounded cursor-pointer
        ${isActive ? 'border-blue-500 bg-slate-700/50' : 'border-slate-700'}
      `}
    >
      <h3 className="text-sm font-semibold text-slate-100 mb-2">
        {thread.name}
      </h3>

      {/* Thread progress */}
      <ProgressBars
        currentStep={thread.currentStep}
        totalSteps={thread.totalSteps}
        elapsedTimeMs={thread.elapsedTimeMs}
        timeBudgetMs={thread.timeBudgetMs}
        tokensUsed={thread.tokensUsed}
        tokenBudget={thread.tokenBudget}
        variant="stacked"
        size="sm"
      />

      {/* Status info */}
      <div className="flex justify-between mt-2 text-xs text-slate-400">
        <span>{thread.currentStep}/{thread.totalSteps} steps</span>
        <span>{thread.status}</span>
      </div>
    </div>
  );
};
```

### Scenario 2: Expanded Thread Details (ThreadDetails)

Show detailed progress in an expanded panel with formatting options.

```tsx
import { ProgressBars } from '@components/OutlineView';

interface ThreadDetailsProps {
  thread: AgentThread;
  showValues?: boolean;
  showPercentages?: boolean;
}

export const ThreadDetails: React.FC<ThreadDetailsProps> = ({
  thread,
  showValues = true,
  showPercentages = false,
}) => {
  return (
    <div className="space-y-4 p-4">
      <h2 className="text-lg font-semibold text-slate-100">
        {thread.name}
      </h2>

      {/* Detailed progress view */}
      <ProgressBars
        currentStep={thread.currentStep}
        totalSteps={thread.totalSteps}
        elapsedTimeMs={thread.elapsedTimeMs}
        timeBudgetMs={thread.timeBudgetMs}
        tokensUsed={thread.tokensUsed}
        tokenBudget={thread.tokenBudget}
        variant="detailed"
        size="md"
        showValues={showValues}
        showPercentages={showPercentages}
      />

      {/* Additional thread info */}
      <div className="grid grid-cols-2 gap-2 text-xs text-slate-400">
        <div>
          <span className="text-slate-500">Status:</span> {thread.status}
        </div>
        <div>
          <span className="text-slate-500">Priority:</span> {thread.priority}
        </div>
        <div>
          <span className="text-slate-500">Model:</span> {thread.model}
        </div>
        <div>
          <span className="text-slate-500">Confidence:</span>{' '}
          {thread.confidence.toFixed(2)}
        </div>
      </div>
    </div>
  );
};
```

### Scenario 3: Inline Progress in List

Show compact progress in a list view for multiple threads.

```tsx
import { ProgressBars } from '@components/OutlineView';

interface ThreadListProps {
  threads: AgentThread[];
  selectedThreadId?: string;
  onSelectThread?: (threadId: string) => void;
}

export const ThreadList: React.FC<ThreadListProps> = ({
  threads,
  selectedThreadId,
  onSelectThread,
}) => {
  return (
    <div className="space-y-1">
      {threads.map(thread => (
        <div
          key={thread.id}
          onClick={() => onSelectThread?.(thread.id)}
          className={`
            p-2 rounded cursor-pointer transition-colors
            ${
              selectedThreadId === thread.id
                ? 'bg-slate-600'
                : 'bg-slate-700/50 hover:bg-slate-600/50'
            }
          `}
        >
          {/* Thread name and status */}
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-medium text-slate-100">
              {thread.name}
            </span>
            <span className="text-xs text-slate-400">
              {thread.currentStep}/{thread.totalSteps}
            </span>
          </div>

          {/* Inline progress bars */}
          <ProgressBars
            currentStep={thread.currentStep}
            totalSteps={thread.totalSteps}
            elapsedTimeMs={thread.elapsedTimeMs}
            timeBudgetMs={thread.timeBudgetMs}
            tokensUsed={thread.tokensUsed}
            tokenBudget={thread.tokenBudget}
            variant="inline"
            size="sm"
          />
        </div>
      ))}
    </div>
  );
};
```

### Scenario 4: Monitoring Dashboard

Show progress for all active threads in a dashboard layout.

```tsx
import { ProgressBars } from '@components/OutlineView';

interface MonitoringDashboardProps {
  threads: AgentThread[];
  refreshInterval?: number; // ms
}

export const MonitoringDashboard: React.FC<MonitoringDashboardProps> = ({
  threads,
  refreshInterval = 500,
}) => {
  // Separate active and completed threads
  const activeThreads = threads.filter(t => t.status !== 'completed');
  const completedThreads = threads.filter(t => t.status === 'completed');

  return (
    <div className="space-y-6 p-4">
      {/* Active Threads Section */}
      {activeThreads.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-slate-100 mb-3">
            Active Threads ({activeThreads.length})
          </h2>
          <div className="space-y-3">
            {activeThreads.map(thread => (
              <div
                key={thread.id}
                className="p-3 bg-slate-700/30 border border-slate-600 rounded"
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-slate-100">
                    {thread.name}
                  </h3>
                  <span className="text-xs px-2 py-1 rounded bg-blue-900/40 text-blue-300">
                    {thread.status}
                  </span>
                </div>

                <ProgressBars
                  currentStep={thread.currentStep}
                  totalSteps={thread.totalSteps}
                  elapsedTimeMs={thread.elapsedTimeMs}
                  timeBudgetMs={thread.timeBudgetMs}
                  tokensUsed={thread.tokensUsed}
                  tokenBudget={thread.tokenBudget}
                  variant="stacked"
                  size="md"
                />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Completed Threads Section */}
      {completedThreads.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-slate-100 mb-3">
            Completed Threads ({completedThreads.length})
          </h2>
          <div className="space-y-2 opacity-75">
            {completedThreads.map(thread => (
              <div
                key={thread.id}
                className="p-2 bg-slate-700/20 border border-slate-700/50 rounded text-sm text-slate-400"
              >
                <div className="flex items-center justify-between">
                  <span>{thread.name}</span>
                  <span>✓ {thread.currentStep} steps</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
};
```

### Scenario 5: Custom Progress Display

Create a custom progress display with additional metrics.

```tsx
import { ProgressBar } from '@components/OutlineView';

interface CustomProgressDisplayProps {
  thread: AgentThread;
  showHistorical?: boolean;
  historicalData?: number[];
}

export const CustomProgressDisplay: React.FC<CustomProgressDisplayProps> = ({
  thread,
  showHistorical = false,
  historicalData = [],
}) => {
  const averageProgress = historicalData.length
    ? (historicalData.reduce((a, b) => a + b, 0) / historicalData.length).toFixed(1)
    : null;

  return (
    <div className="space-y-3">
      {/* Current progress */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-slate-400">
          <span>Current Progress</span>
          {averageProgress && <span>Avg: {averageProgress}%</span>}
        </div>
        <ProgressBar
          value={thread.currentStep}
          max={thread.totalSteps}
          color="blue"
          size="lg"
          showLabel={true}
          label={`${thread.currentStep}/${thread.totalSteps}`}
        />
      </div>

      {/* Historical trend (if available) */}
      {showHistorical && historicalData.length > 0 && (
        <div className="space-y-1">
          <div className="text-xs text-slate-400">Trend</div>
          <div className="flex gap-0.5 h-6 items-end">
            {historicalData.map((value, i) => (
              <div
                key={i}
                className="flex-1 bg-gradient-to-t from-blue-500 to-cyan-500 rounded-t"
                style={{
                  height: `${value}%`,
                  opacity: i === historicalData.length - 1 ? 1 : 0.6,
                }}
                title={`${value}%`}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
```

## Type Integration

### AgentThread Type Definition

Ensure your `AgentThread` type includes these fields:

```typescript
interface AgentThread {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  priority: number;
  model: string;
  confidence: number;

  // Progress tracking fields (for ProgressBars)
  currentStep: number;
  totalSteps: number;
  elapsedTimeMs: number;
  timeBudgetMs?: number;      // Optional
  tokensUsed: number;
  tokenBudget?: number;        // Optional

  // Additional fields...
}
```

### Using with TypeScript

```typescript
import type { ProgressBarsProps } from '@components/OutlineView';

// Type-safe progress props from thread
const getProgressProps = (thread: AgentThread): ProgressBarsProps => ({
  currentStep: thread.currentStep,
  totalSteps: thread.totalSteps,
  elapsedTimeMs: thread.elapsedTimeMs,
  timeBudgetMs: thread.timeBudgetMs,
  tokensUsed: thread.tokensUsed,
  tokenBudget: thread.tokenBudget,
});

<ProgressBars {...getProgressProps(thread)} />
```

## State Management Integration

### With Zustand

```typescript
import { create } from 'zustand';

interface ThreadStore {
  threads: AgentThread[];
  selectedThreadId: string | null;
  updateThread: (id: string, updates: Partial<AgentThread>) => void;
  selectThread: (id: string) => void;
}

export const useThreadStore = create<ThreadStore>((set) => ({
  threads: [],
  selectedThreadId: null,

  updateThread: (id, updates) =>
    set((state) => ({
      threads: state.threads.map((t) =>
        t.id === id ? { ...t, ...updates } : t
      ),
    })),

  selectThread: (id) => set({ selectedThreadId: id }),
}));

// Usage in component
export const ThreadProgress: React.FC<{ threadId: string }> = ({
  threadId,
}) => {
  const thread = useThreadStore((state) =>
    state.threads.find((t) => t.id === threadId)
  );

  if (!thread) return null;

  return (
    <ProgressBars
      currentStep={thread.currentStep}
      totalSteps={thread.totalSteps}
      elapsedTimeMs={thread.elapsedTimeMs}
      timeBudgetMs={thread.timeBudgetMs}
      tokensUsed={thread.tokensUsed}
      tokenBudget={thread.tokenBudget}
    />
  );
};
```

## Real-Time Updates

### WebSocket Integration

```typescript
import { useEffect } from 'react';

interface ThreadProgressProps {
  threadId: string;
  wsUrl: string;
}

export const ThreadProgress: React.FC<ThreadProgressProps> = ({
  threadId,
  wsUrl,
}) => {
  const [thread, setThread] = useState<AgentThread | null>(null);
  const updateThread = useThreadStore((state) => state.updateThread);

  useEffect(() => {
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'thread_update' && data.threadId === threadId) {
        // Update both store and local state for real-time display
        updateThread(threadId, data.thread);
        setThread(data.thread);
      }
    };

    return () => ws.close();
  }, [threadId, wsUrl, updateThread]);

  if (!thread) return <div>Loading...</div>;

  return (
    <ProgressBars
      currentStep={thread.currentStep}
      totalSteps={thread.totalSteps}
      elapsedTimeMs={thread.elapsedTimeMs}
      timeBudgetMs={thread.timeBudgetMs}
      tokensUsed={thread.tokensUsed}
      tokenBudget={thread.tokenBudget}
      variant="detailed"
      size="md"
    />
  );
};
```

## CSS Customization

The component uses Tailwind CSS classes that can be customized via the `className` prop:

```tsx
<ProgressBars
  {...props}
  className="p-3 bg-slate-900/50 rounded-lg border border-slate-700/50 shadow-lg"
/>
```

Or override globally in your Tailwind config:

```typescript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        'progress-blue': 'rgb(59 130 246)',
        'progress-amber': 'rgb(217 119 6)',
        'progress-purple': 'rgb(147 51 234)',
      },
    },
  },
};
```

## Accessibility Checklist

- ✅ ARIA labels and roles properly set
- ✅ Color not the only indicator of status
- ✅ Keyboard navigation support
- ✅ Screen reader friendly
- ✅ High contrast sufficient
- ✅ Responsive text sizing
- ✅ No flashing content (optional shimmer only)

## Performance Considerations

1. **Memoization**: Components use `useMemo` for expensive calculations
2. **CSS Transitions**: Hardware-accelerated animations
3. **No Re-renders**: Only update when props change
4. **Bundle Size**: ~8kb minified total

## Testing

### Unit Testing Example

```typescript
import { render, screen } from '@testing-library/react';
import { ProgressBars } from '@components/OutlineView';

describe('ProgressBars', () => {
  it('should render all three bars when all budgets provided', () => {
    render(
      <ProgressBars
        currentStep={3}
        totalSteps={7}
        elapsedTimeMs={4500}
        timeBudgetMs={10500}
        tokensUsed={750}
        tokenBudget={1750}
      />
    );

    const bars = screen.getAllByRole('progressbar');
    expect(bars).toHaveLength(3);
  });

  it('should handle over-budget correctly', () => {
    const { container } = render(
      <ProgressBars
        currentStep={3}
        totalSteps={7}
        elapsedTimeMs={12000}  // Over budget
        timeBudgetMs={10000}
        tokensUsed={750}
        tokenBudget={1750}
      />
    );

    // Check for red gradient in time bar
    const timeBar = container.querySelector('[aria-valuenow="100"]');
    expect(timeBar).toBeInTheDocument();
  });
});
```

## Troubleshooting

### Issue: Bars not showing

**Solution**: Ensure `totalSteps > 0` and `tokenBudget > 0`:

```typescript
// ❌ Wrong
<ProgressBars
  currentStep={3}
  totalSteps={0}  // Invalid
  ...
/>

// ✅ Correct
<ProgressBars
  currentStep={3}
  totalSteps={7}  // Valid
  ...
/>
```

### Issue: Labels not appearing

**Solution**: Use appropriate variant or enable options:

```typescript
// ❌ Wrong - stacked variant with showValues=true still won't show
<ProgressBars
  {...props}
  variant="stacked"
  showValues={true}
/>

// ✅ Correct - use detailed variant for full labels
<ProgressBars
  {...props}
  variant="detailed"
/>

// ✅ Or use stacked with showPercentages
<ProgressBars
  {...props}
  variant="stacked"
  showPercentages={true}
/>
```

### Issue: Animations not working

**Solution**: Check Tailwind CSS configuration includes animation support:

```typescript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      animation: {
        pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
};
```

## Next Steps

1. **Integrate with existing components** - Add to ThreadCard, ThreadDetails
2. **Connect real-time updates** - Use WebSocket or polling
3. **Add historical tracking** - Store progress history for trends
4. **Implement threshold alerts** - Warn when approaching budgets
5. **Customize theming** - Adjust colors and sizing for your design system
