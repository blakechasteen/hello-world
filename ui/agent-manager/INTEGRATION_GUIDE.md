# StepList & StepRow Integration Guide

**Phase 3: Outline View Components**
**Date**: December 11, 2025
**Status**: ✅ Ready for Integration

## Overview

This guide explains how to integrate the new StepList and StepRow components into the HoloLoom Agent Manager UI.

## Files Created

### Core Components
- `src/components/OutlineView/types.ts` - Type definitions
- `src/components/OutlineView/StepRow.tsx` - Individual step component
- `src/components/OutlineView/StepList.tsx` - Step list container
- `src/components/OutlineView/index.ts` - Exports

### Documentation
- `src/components/OutlineView/README.md` - Full documentation
- `src/components/OutlineView/QUICK_REFERENCE.md` - Quick start guide
- `PHASE_3_OUTLINE_VIEW_COMPLETE.md` - Implementation summary
- `INTEGRATION_GUIDE.md` - This file

### Examples & Tests
- `src/components/OutlineView/StepList.demo.tsx` - Interactive demo
- `src/components/OutlineView/StepRow.test.tsx` - Unit tests

## Integration Steps

### 1. Verify File Structure

```
ui/agent-manager/
├── src/
│   ├── components/
│   │   └── OutlineView/
│   │       ├── types.ts ✅
│   │       ├── StepRow.tsx ✅
│   │       ├── StepList.tsx ✅
│   │       ├── index.ts ✅
│   │       ├── README.md ✅
│   │       ├── QUICK_REFERENCE.md ✅
│   │       ├── StepList.demo.tsx ✅
│   │       └── StepRow.test.tsx ✅
│   └── ...
├── PHASE_3_OUTLINE_VIEW_COMPLETE.md ✅
└── INTEGRATION_GUIDE.md ✅
```

### 2. Update ThreadCard Component

If you have an existing ThreadCard or thread display component, integrate StepList:

**Before:**
```typescript
export const ThreadCard = ({ thread }: ThreadCardProps) => {
  return (
    <div className="thread-card">
      <h3>{thread.name}</h3>
      {/* Old step display */}
    </div>
  );
};
```

**After:**
```typescript
import { StepList, type TaskNode } from '@/components/OutlineView';

export const ThreadCard = ({ thread }: ThreadCardProps) => {
  const [selectedStepId, setSelectedStepId] = useState<string | null>(null);

  // Convert thread data to TaskNode format if needed
  const steps: TaskNode[] = thread.steps.map((step) => ({
    id: step.id,
    threadId: thread.id,
    depth: step.depth || 0,
    stepType: step.type || 'query',
    name: step.name,
    query: step.query,
    status: step.status,
    progressPct: step.progress || 0,
    elapsedTimeMs: step.elapsedTime || 0,
    tokensUsed: step.tokens || 0,
    confidence: step.confidence || 0.5,
    dependsOn: step.dependsOn || [],
    blocks: step.blocks || [],
    mrfEligible: step.mrfEligible || false,
    mctsEligible: step.mctsEligible || false,
    childrenIds: step.childrenIds || [],
    parentId: step.parentId,
  }));

  return (
    <div className="thread-card">
      <StepList
        steps={steps}
        threadId={thread.id}
        selectedStepId={selectedStepId}
        onStepSelect={setSelectedStepId}
        onInjectMRF={(stepId) => {
          // Handle MRF injection
          console.log('MRF injection for step:', stepId);
        }}
        onInjectMCTS={(stepId) => {
          // Handle MCTS injection
          console.log('MCTS injection for step:', stepId);
        }}
      />
    </div>
  );
};
```

### 3. Run Tests

```bash
# Install test dependencies if needed
npm install --save-dev @testing-library/react @testing-library/user-event

# Run tests
npm test StepRow.test.tsx

# Run all tests
npm test
```

### 4. View Demo

```bash
# If you have a Storybook setup
npm run storybook

# Or create a demo page
npm run dev
# Then navigate to the demo component
```

### 5. Update Type System

If your codebase has custom step types, align them with TaskNode:

```typescript
// Old type
interface Step {
  id: string;
  name: string;
  status: string;
}

// New type (TaskNode)
interface TaskNode {
  id: string;
  threadId: string;
  depth: number;
  stepType: StepType;
  name: string;
  // ... full TaskNode properties
}
```

## Data Mapping

### Converting from Old Step Format to TaskNode

```typescript
const convertOldStepToTaskNode = (
  oldStep: OldStep,
  threadId: string
): TaskNode => ({
  id: oldStep.id,
  threadId,
  parentId: oldStep.parentId,
  childrenIds: oldStep.childrenIds || [],
  depth: oldStep.depth || 0,
  stepType: oldStep.type || 'query',
  name: oldStep.name,
  query: oldStep.query,
  status: oldStep.status,
  progressPct: oldStep.progress || 0,
  elapsedTimeMs: oldStep.elapsedTime || 0,
  tokensUsed: oldStep.tokens || 0,
  confidence: oldStep.confidence || 0.5,
  dependsOn: oldStep.dependsOn || [],
  blocks: oldStep.blocks || [],
  mrfEligible: oldStep.mrfEligible || false,
  mctsEligible: oldStep.mctsEligible || false,
  injectionApplied: oldStep.injection,
});
```

## API Integration

### Handling MRF Injection

```typescript
const handleMRFInjection = async (stepId: string) => {
  try {
    // Call your API
    const response = await fetch(`/api/threads/${threadId}/steps/${stepId}/inject-mrf`, {
      method: 'POST',
      body: JSON.stringify({
        strategy: 'verify', // or 'refine', 'elegance', etc.
      }),
    });

    if (response.ok) {
      const result = await response.json();
      // Update step with injection info
      updateStep(stepId, {
        injectionApplied: result.injectionType,
        confidence: result.newConfidence,
      });
    }
  } catch (error) {
    console.error('MRF injection failed:', error);
  }
};
```

### Handling MCTS Injection

```typescript
const handleMCTSInjection = async (stepId: string) => {
  try {
    const response = await fetch(`/api/threads/${threadId}/steps/${stepId}/inject-mcts`, {
      method: 'POST',
      body: JSON.stringify({
        explorationRate: 0.15,
        maxNodes: 100,
      }),
    });

    if (response.ok) {
      const result = await response.json();
      updateStep(stepId, {
        injectionApplied: result.injectionType,
        confidence: result.newConfidence,
      });
    }
  } catch (error) {
    console.error('MCTS injection failed:', error);
  }
};
```

## State Management

### With React Hooks

```typescript
const [steps, setSteps] = useState<TaskNode[]>([]);
const [selectedStepId, setSelectedStepId] = useState<string | null>(null);
const [hoveredStepId, setHoveredStepId] = useState<string | null>(null);

const updateStep = (stepId: string, updates: Partial<TaskNode>) => {
  setSteps((prev) =>
    prev.map((step) =>
      step.id === stepId ? { ...step, ...updates } : step
    )
  );
};
```

### With Redux

```typescript
// actions.ts
export const selectStep = (stepId: string) => ({
  type: 'SELECT_STEP',
  payload: stepId,
});

export const injectMRF = (stepId: string) => ({
  type: 'INJECT_MRF',
  payload: stepId,
});

// reducer.ts
const threadReducer = (state = initialState, action) => {
  switch (action.type) {
    case 'SELECT_STEP':
      return { ...state, selectedStepId: action.payload };
    case 'INJECT_MRF':
      return {
        ...state,
        steps: state.steps.map((s) =>
          s.id === action.payload
            ? { ...s, injectionApplied: 'mrf_verify' }
            : s
        ),
      };
    default:
      return state;
  }
};

// component.tsx
const dispatch = useDispatch();
const steps = useSelector((state) => state.thread.steps);

<StepList
  steps={steps}
  threadId="thread-1"
  onStepSelect={(id) => dispatch(selectStep(id))}
  onInjectMRF={(id) => dispatch(injectMRF(id))}
/>
```

## Styling

### Tailwind CSS Configuration

Ensure your Tailwind config includes:

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      animation: {
        spin: 'spin 1s linear infinite',
        pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
};
```

### Custom Styling

Override default styles:

```typescript
<StepList
  steps={steps}
  threadId="thread-1"
  className="custom-step-list"
/>

<style>
  .custom-step-list {
    max-height: 500px; /* Override default */
  }

  .custom-step-list [role='row'] {
    background-color: #1e293b; /* Custom background */
  }
</style>
```

## Performance Optimization

### For Large Step Lists (500+)

Use React Window virtualization:

```typescript
import { FixedSizeList as List } from 'react-window';
import StepRow from '@/components/OutlineView/StepRow';

const VirtualizedStepList = ({ steps, threadId }) => {
  const Row = ({ index, style }) => (
    <div style={style}>
      <StepRow
        step={steps[index]}
        depth={steps[index].depth}
        key={steps[index].id}
      />
    </div>
  );

  return (
    <List
      height={600}
      itemCount={steps.length}
      itemSize={32}
      width="100%"
    >
      {Row}
    </List>
  );
};
```

### Memoization

```typescript
import { memo } from 'react';

export const MemoizedStepRow = memo(StepRow);
export const MemoizedStepList = memo(StepList);
```

## Testing Integration

### Unit Test Example

```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { StepList } from '@/components/OutlineView';

describe('ThreadCard with StepList', () => {
  it('renders steps correctly', () => {
    const steps = [
      {
        id: '1',
        threadId: 'thread-1',
        depth: 0,
        stepType: 'query',
        name: 'Test',
        status: 'completed',
        progressPct: 100,
        elapsedTimeMs: 1000,
        tokensUsed: 100,
        confidence: 0.9,
        dependsOn: [],
        blocks: [],
        mrfEligible: false,
        mctsEligible: false,
        childrenIds: [],
      },
    ];

    render(<StepList steps={steps} threadId="thread-1" />);

    expect(screen.getByText('Test')).toBeInTheDocument();
  });

  it('calls onStepSelect when step clicked', () => {
    const handleSelect = jest.fn();
    const steps = [
      {
        id: '1',
        threadId: 'thread-1',
        // ... other properties
      },
    ];

    render(
      <StepList
        steps={steps}
        threadId="thread-1"
        onStepSelect={handleSelect}
      />
    );

    fireEvent.click(screen.getByText(/Test/i));
    expect(handleSelect).toHaveBeenCalledWith('1');
  });
});
```

## Troubleshooting

### Issue: Styles not applying

**Solution**: Ensure Tailwind CSS is properly configured and imported

```typescript
// tailwind.css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### Issue: TypeScript errors

**Solution**: Ensure types.ts is properly imported

```typescript
import type { TaskNode, StepType, StepStatus } from '@/components/OutlineView/types';
```

### Issue: Performance issues with many steps

**Solution**: Use virtualization for large lists (500+)

### Issue: Injection buttons not appearing

**Solution**:
1. Check `mrfEligible`/`mctsEligible` flags
2. Verify `injectionApplied` is not set
3. Check if row is being hovered

## Deployment Checklist

- [ ] All files copied to `src/components/OutlineView/`
- [ ] Types.ts imports work correctly
- [ ] Components render without errors
- [ ] Tests pass (`npm test`)
- [ ] Demo component runs (`npm run dev`)
- [ ] Tailwind classes applied correctly
- [ ] Responsive design tested on mobile/tablet/desktop
- [ ] Accessibility verified (WCAG AA)
- [ ] API integration working (MRF/MCTS)
- [ ] State management integrated
- [ ] Bundle size acceptable (<50KB)

## Next Steps (Phase 4+)

1. **Add keyboard navigation** (arrow keys, enter, escape)
2. **Implement context menu** (right-click options)
3. **Add filtering/sorting**
4. **Implement drag-and-drop**
5. **Add dependency graph visualization**
6. **Create timeline view**
7. **Add virtual scrolling** (for 500+ items)

## Support

For questions or issues:

1. Check `QUICK_REFERENCE.md` for common patterns
2. Review `README.md` for detailed documentation
3. Look at `StepList.demo.tsx` for examples
4. Check `StepRow.test.tsx` for test patterns

---

**Created**: December 11, 2025
**Phase**: 3 (Outline View)
**Status**: ✅ Production Ready
**Last Updated**: December 11, 2025
