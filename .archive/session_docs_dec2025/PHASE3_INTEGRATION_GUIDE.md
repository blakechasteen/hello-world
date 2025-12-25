# Phase 3 Controls Integration Guide

**Quick Reference for Using the New Control Components**

---

## Quick Start

### 1. Import the Components
```typescript
import {
  PriorityControls,
  ThreadControls,
  InjectMenu,
} from '@/components/OutlineView';
```

### 2. Add to Your Thread Row Component
```typescript
// In ThreadRow.tsx or similar
export const ThreadRow: React.FC<ThreadRowProps> = ({ thread }) => {
  return (
    <div className="flex items-center gap-2 p-2">
      {/* Status indicator */}
      <StatusIndicator status={thread.status} />

      {/* Thread info */}
      <div className="flex-1">
        <div className="font-medium">{thread.name}</div>
        <div className="text-sm text-slate-400">{thread.agentType}</div>
      </div>

      {/* NEW: Priority controls */}
      <PriorityControls
        threadId={thread.id}
        priority={thread.priority}
        size="md"
        orientation="vertical"
      />

      {/* NEW: Thread lifecycle controls */}
      <ThreadControls
        threadId={thread.id}
        status={thread.status}
        size="md"
        showCancelConfirm={true}
      />

      {/* NEW: Injection menu */}
      <InjectMenu
        threadId={thread.id}
        stepId={thread.currentStep.toString()}
        mrfEligible={thread.status === 'running' || thread.status === 'paused'}
        mctsEligible={thread.status === 'running' || thread.status === 'paused'}
        onInjectMRF={(strategy) => {
          console.log('Injecting MRF:', strategy);
          // Handle MRF injection
        }}
        onInjectMCTS={({ budget, exploration }) => {
          console.log('Injecting MCTS:', budget, exploration);
          // Handle MCTS injection
        }}
      />
    </div>
  );
};
```

---

## Component Reference

### PriorityControls

**Minimal Setup**:
```typescript
<PriorityControls threadId="123" priority={50} />
```

**Full Options**:
```typescript
<PriorityControls
  threadId="123"
  priority={50}
  size="md"              // 'sm' or 'md'
  orientation="vertical" // 'vertical' or 'horizontal'
  className="my-custom-class"
  onChange={(newPriority) => {
    // Handle priority change
  }}
/>
```

**Use Cases**:
- Thread queue management
- User attention prioritization
- Task ordering by importance

---

### ThreadControls

**Minimal Setup**:
```typescript
<ThreadControls threadId="123" status="running" />
```

**Full Options**:
```typescript
<ThreadControls
  threadId="123"
  status="running"
  size="md"
  showCancelConfirm={true}
  className="my-custom-class"
  onAction={(action) => {
    if (action === 'pause') {
      // Handle pause
    } else if (action === 'resume') {
      // Handle resume
    } else if (action === 'cancel') {
      // Handle cancel
    }
  }}
/>
```

**Status States**:
- `running` → Shows Pause + Cancel
- `paused` → Shows Resume + Cancel
- `idle` → Shows nothing
- `completed`/`failed`/`cancelled` → Shows nothing (or Retry placeholder)

---

### InjectMenu

**Minimal Setup**:
```typescript
<InjectMenu
  threadId="123"
  stepId="step-001"
  mrfEligible={true}
  mctsEligible={false}
/>
```

**Full Options**:
```typescript
<InjectMenu
  threadId="123"
  stepId="step-001"
  mrfEligible={true}
  mctsEligible={true}
  injected="mrf"
  appliedStrategy="verify"
  size="md"
  className="my-custom-class"
  onInjectMRF={(strategy) => {
    // Handle MRF: 'auto' | 'verify' | 'elegance' | 'critique' | 'refine' | 'hofstadter'
  }}
  onInjectMCTS={({ budget, exploration }) => {
    // budget: 50 | 100 | 200 | 500
    // exploration: 0.5 | 1.0 | 1.4 | 2.0
  }}
/>
```

**Eligibility Rules**:
- MRF eligible when: step exists and strategy not yet applied
- MCTS eligible when: planning is needed and MCTS not yet applied
- Show disabled button when neither eligible

---

## Integration Patterns

### Pattern 1: Thread List with Controls
```typescript
const ThreadList: React.FC = () => {
  const threads = useAgentManagerStore((s) => Object.values(s.threads));

  return (
    <div className="space-y-2">
      {threads.map((thread) => (
        <div key={thread.id} className="flex items-center gap-2 p-2 bg-slate-700 rounded">
          <StatusIndicator status={thread.status} />
          <span className="flex-1">{thread.name}</span>

          <PriorityControls
            threadId={thread.id}
            priority={thread.priority}
            size="sm"
            orientation="horizontal"
          />

          <ThreadControls
            threadId={thread.id}
            status={thread.status}
            size="sm"
          />
        </div>
      ))}
    </div>
  );
};
```

### Pattern 2: Detailed Thread View with Injection
```typescript
const ThreadDetail: React.FC<{ threadId: string }> = ({ threadId }) => {
  const thread = useAgentManagerStore((s) => s.getThreadById(threadId));

  if (!thread) return null;

  return (
    <div className="p-4 bg-slate-800 rounded-lg">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-bold">{thread.name}</h2>

        <div className="flex gap-2">
          <PriorityControls
            threadId={thread.id}
            priority={thread.priority}
            size="md"
            orientation="horizontal"
          />

          <ThreadControls
            threadId={thread.id}
            status={thread.status}
            size="md"
          />
        </div>
      </div>

      {/* Steps list */}
      <div className="space-y-2">
        {Array.from({ length: thread.totalSteps }).map((_, i) => (
          <div key={i} className="flex items-center gap-2 p-2 bg-slate-700 rounded">
            <span>Step {i + 1}</span>

            <InjectMenu
              threadId={thread.id}
              stepId={`step-${i}`}
              mrfEligible={thread.status === 'running'}
              mctsEligible={thread.status === 'running'}
              onInjectMRF={(strategy) => {
                console.log(`MRF ${strategy} injected at step ${i}`);
              }}
              onInjectMCTS={({ budget, exploration }) => {
                console.log(`MCTS ${budget}/${exploration} injected at step ${i}`);
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
};
```

### Pattern 3: Inline Controls in Step Row
```typescript
const StepRow: React.FC<{ step: TaskNode }> = ({ step }) => {
  return (
    <div className="flex items-center gap-2 p-2 hover:bg-slate-700">
      <StatusIcon status={step.status} />
      <span className="flex-1">{step.name}</span>
      <ConfidenceDisplay confidence={step.confidence} />

      {/* MRF/MCTS injection point */}
      <InjectMenu
        threadId={step.threadId}
        stepId={step.id}
        mrfEligible={true}
        mctsEligible={true}
        size="sm"
      />
    </div>
  );
};
```

---

## Styling Customization

### Adjust Sizes
```typescript
// Compact controls (small)
<PriorityControls size="sm" />
<ThreadControls size="sm" />
<InjectMenu size="sm" />

// Standard size (medium) - default
<PriorityControls size="md" />
<ThreadControls size="md" />
<InjectMenu size="md" />
```

### Custom Classes
```typescript
<PriorityControls className="ml-2" />
<ThreadControls className="gap-2" />
<InjectMenu className="ml-auto" />
```

### Dark Mode (already applied)
All components use dark theme colors by default:
- Background: `bg-slate-700`, `bg-slate-800`
- Text: `text-slate-200`, `text-white`
- Hover: Lighter shades
- Active states: Scale animations

---

## Keyboard Navigation

### PriorityControls
- `Tab`: Focus upvote button
- `Enter`: Upvote
- `Tab`: Focus priority display
- `Tab`: Focus downvote button
- `Enter`: Downvote
- `ArrowUp`: Upvote (from any control)
- `ArrowDown`: Downvote (from any control)

### ThreadControls
- `Tab`: Focus pause/resume button
- `Enter`: Pause/Resume
- `Tab`: Focus cancel button
- `Enter`: Open cancel confirmation
- `Tab`: Navigate confirmation buttons
- `Enter`: Confirm action
- `Escape`: Dismiss confirmation

### InjectMenu
- `Tab`: Focus menu trigger
- `Enter`: Open/close dropdown
- `Tab`: Navigate menu items (within dropdown)
- `Enter`: Select strategy
- `Escape`: Close dropdown
- Click outside: Close dropdown

---

## Common Issues & Solutions

### Issue: Priority isn't updating
**Solution**: Check that store action is being called
```typescript
const { upvoteThread, downvoteThread } = useAgentManagerStore();
// These are called automatically in PriorityControls
```

### Issue: Cancel button appears for idle threads
**Solution**: Check thread status prop
```typescript
// Only pass 'running' or 'paused' to ThreadControls
// Don't show component for 'idle' status
{thread.status !== 'idle' && (
  <ThreadControls threadId={thread.id} status={thread.status} />
)}
```

### Issue: InjectMenu disabled when it shouldn't be
**Solution**: Verify eligibility props
```typescript
// At least one must be true
mrfEligible={someCondition}
mctsEligible={someOtherCondition}
// If both false, button is disabled
```

### Issue: Tooltips not showing
**Solution**: Parent overflow hidden
```typescript
// Ensure parent doesn't clip with overflow: hidden
<div className="overflow-visible">
  <ThreadControls ... />
</div>
```

---

## Performance Optimization

### Memoize if needed
```typescript
import { memo } from 'react';

const ThreadRowMemo = memo(ThreadRow, (prev, next) => {
  return (
    prev.thread.id === next.thread.id &&
    prev.thread.priority === next.thread.priority &&
    prev.thread.status === next.thread.status
  );
});
```

### Avoid prop drilling
```typescript
// ❌ Don't pass thread all the way down
<ThreadRow thread={thread} />

// ✅ Extract what you need at each level
<ThreadRow threadId={thread.id} />
```

---

## Accessibility Checklist

- [x] All buttons have `aria-label`
- [x] Disabled state indicated via `aria-disabled`
- [x] Keyboard navigation works
- [x] Focus ring visible (2px blue)
- [x] Color not sole differentiator (icons + colors)
- [x] Confirmation dialog has `aria-modal`
- [x] Status updates use `aria-live`
- [x] Menu structure uses ARIA roles

---

## Testing Examples

### Jest Tests for PriorityControls
```typescript
describe('PriorityControls', () => {
  it('should upvote when button clicked', () => {
    const { getByLabelText } = render(
      <PriorityControls threadId="123" priority={50} />
    );
    fireEvent.click(getByLabelText('Increase priority'));
    expect(upvoteThread).toHaveBeenCalledWith('123');
  });

  it('should disable upvote at max priority', () => {
    const { getByLabelText } = render(
      <PriorityControls threadId="123" priority={100} />
    );
    expect(getByLabelText('Increase priority')).toBeDisabled();
  });

  it('should support keyboard navigation', () => {
    const { getByRole } = render(
      <PriorityControls threadId="123" priority={50} />
    );
    const container = getByRole('group');
    fireEvent.keyDown(container, { key: 'ArrowUp' });
    // Should call upvoteThread
  });
});
```

---

## Next Steps

1. **Integrate into OutlineView**
   - Add to ThreadRow component
   - Test with real thread data

2. **Add to StepList** (if needed)
   - Show inject menu per step
   - Manage step-level injection

3. **Connect to Backend**
   - Implement MRF/MCTS handlers
   - Send injection requests to API
   - Update UI with results

4. **Add Analytics**
   - Track priority changes
   - Monitor injection usage
   - Measure execution time improvements

---

## Support & Questions

For issues or questions:
1. Check this guide
2. Review component source code comments
3. Check `PHASE3_CONTROLS_SUMMARY.md` for detailed specifications
4. Verify store integration in `agentManagerStore.ts`

---

**Document Version**: 1.0
**Last Updated**: 2025-12-11
**Status**: Ready for Production Integration
