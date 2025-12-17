# OutlineView Components

HoloLoom Agent Manager UI Phase 3 - Outline View components for displaying and interacting with task execution steps.

## Overview

The OutlineView provides a hierarchical visualization of task execution within threads. It displays:
- **Individual steps** with status, progress, and confidence
- **Hierarchical structure** with parent-child relationships
- **Execution flow** with visual connectors
- **Action buttons** for MRF/MCTS injection
- **Progress tracking** with completion statistics

## Components

### StepList

Container component that displays a list of task steps within a thread.

**Props:**
```typescript
interface StepListProps {
  steps: TaskNode[];                    // Array of task nodes to display
  threadId: string;                     // ID of the parent thread
  rootTask?: TaskNode;                  // Root task node (optional)
  hoveredStepId?: string | null;        // Currently hovered step ID
  selectedStepId?: string | null;       // Currently selected step ID
  onStepHover?: (stepId: string | null) => void;
  onStepSelect?: (stepId: string) => void;
  onInjectMRF?: (stepId: string) => void;
  onInjectMCTS?: (stepId: string) => void;
  showQueryPreview?: boolean;           // Show query on hover (default: true)
  className?: string;                   // Custom container class
}
```

**Features:**
- Sorted step display with parent-child relationship preservation
- Visual progress bar showing overall completion
- Statistics bar with completion counts
- Connector lines between sequential steps
- Scroll container with max-height constraint
- Empty state handling
- Responsive design with screen-size-aware stats

**Usage:**
```typescript
<StepList
  steps={taskNodes}
  threadId="thread-001"
  rootTask={rootTaskNode}
  selectedStepId={selected}
  onStepSelect={handleSelect}
  onInjectMRF={handleMRFInjection}
  onInjectMCTS={handleMCTSInjection}
/>
```

### StepRow

Individual step row component showing a single task with full interactive controls.

**Props:**
```typescript
interface StepRowProps {
  step: TaskNode;                       // The task node to display
  depth: number;                        // Indentation depth (12px per level)
  isHovered?: boolean;                  // Current hover state
  isSelected?: boolean;                 // Current selection state
  onHover?: (stepId: string | null) => void;
  onClick?: (stepId: string) => void;
  onInjectMRF?: (stepId: string) => void;
  onInjectMCTS?: (stepId: string) => void;
  showQueryPreview?: boolean;           // Show query tooltip (default: true)
}
```

**Layout:**
```
[Status Icon] [Name] [Progress] [Confidence] [Tokens] [Time] [Actions]
```

**Status Icons:**
- `○` (pending): Circle outline in slate-400
- `◐` (running): Spinning circle in blue-400 with animation
- `✓` (completed): Checkmark in emerald-500
- `✗` (failed): X symbol in red-500
- `—` (skipped): Dash in slate-500

**Features:**
- Color-coded confidence indicator (green >0.8, amber >0.5, red <0.5)
- Running step animation (subtle pulse)
- Completed step dimming (opacity-75)
- Query text preview on hover
- Injection badge showing applied injections
- Token usage display (large screens)
- Elapsed time display (small screens up)
- Dependency/blocking indicators (extra-large screens)
- Responsive button visibility (shown on hover)

**Usage:**
```typescript
<StepRow
  step={taskNode}
  depth={1}
  isSelected={isSelected}
  onHover={setHovered}
  onClick={handleSelect}
  onInjectMRF={handleMRF}
  onInjectMCTS={handleMCTS}
/>
```

## Data Types

### TaskNode

```typescript
interface TaskNode {
  id: string;                           // Unique identifier
  threadId: string;                     // Parent thread ID
  parentId?: string;                    // Parent task ID (for hierarchy)
  childrenIds: string[];                // Child task IDs
  depth: number;                        // Nesting depth (0 = top-level)
  stepType: StepType;                   // 'query' | 'research' | 'verify' | 'synthesize' | 'execute'
  name: string;                         // Display name
  query?: string;                       // Associated query text
  status: StepStatus;                   // 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  progressPct: number;                  // 0-100 progress percentage
  elapsedTimeMs: number;                // Elapsed time in milliseconds
  tokensUsed: number;                   // Number of tokens consumed
  confidence: number;                   // Confidence score 0.0-1.0
  dependsOn: string[];                  // List of task IDs this depends on
  blocks: string[];                     // List of task IDs this blocks
  mrfEligible: boolean;                 // Can be enhanced with MRF injection
  mctsEligible: boolean;                // Can be enhanced with MCTS injection
  injectionApplied?: string;            // Name of applied injection ('mrf_verify', etc.)
}
```

## Design System

### Colors

**Status Colors:**
- Pending: `text-slate-400`
- Running: `text-blue-400` with spinning animation
- Completed: `text-emerald-500`
- Failed: `text-red-500`
- Skipped: `text-slate-500`

**Confidence Colors:**
- High (>0.8): `text-emerald-400` (green)
- Medium (>0.5): `text-amber-400` (amber)
- Low (<0.5): `text-red-400` (red)

**Injection Colors:**
- MRF: `bg-emerald-900/30` text `text-emerald-300`
- MCTS: `bg-cyan-900/30` text `text-cyan-300`

**Background Opacity:**
- Default: `bg-slate-800/50`
- Hovered: `bg-slate-700/50`
- Selected: `bg-slate-700`

### Sizing

- Row height: 32px (h-8)
- Icon width: 20px
- Progress bar: 48px width, 6px height
- Status icon: 20x20px
- Indentation: 12px per depth level

### Animations

- Running spinner: `animate-spin` (infinite rotation)
- Running pulse: `animate-pulse` (opacity fade)
- Transitions: `transition-all duration-300`

## Responsive Behavior

### Breakpoints

**Small screens** (<640px - `sm`):
- Hide elapsed time
- Stack layout more tightly
- Show action buttons only on hover

**Medium screens** (640px-1024px - `md`):
- Show elapsed time
- Normal spacing

**Large screens** (1024px-1280px - `lg`):
- Show token usage
- Show dependency indicators

**Extra-large screens** (1280px+ - `xl`):
- Show all details
- Full dependency/blocking indicators
- Full injection button labels

## Example Usage

### Basic Step List

```typescript
import { StepList, TaskNode } from '@/components/OutlineView';

const MyComponent = () => {
  const [selected, setSelected] = React.useState<string | null>(null);

  const steps: TaskNode[] = [
    {
      id: 'step-1',
      threadId: 'thread-1',
      depth: 0,
      stepType: 'query',
      name: 'Initial Query',
      query: 'What is Thompson Sampling?',
      status: 'completed',
      progressPct: 100,
      elapsedTimeMs: 1500,
      tokensUsed: 145,
      confidence: 0.92,
      dependsOn: [],
      blocks: ['step-2'],
      mrfEligible: false,
      mctsEligible: false,
      childrenIds: ['step-2'],
    },
    {
      id: 'step-2',
      threadId: 'thread-1',
      parentId: 'step-1',
      depth: 1,
      stepType: 'research',
      name: 'Background Research',
      status: 'running',
      progressPct: 65,
      elapsedTimeMs: 2300,
      tokensUsed: 420,
      confidence: 0.75,
      dependsOn: ['step-1'],
      blocks: ['step-3'],
      mrfEligible: true,
      mctsEligible: false,
      childrenIds: ['step-3'],
    },
  ];

  return (
    <StepList
      steps={steps}
      threadId="thread-1"
      selectedStepId={selected}
      onStepSelect={setSelected}
      onInjectMRF={(id) => console.log('MRF injection:', id)}
    />
  );
};
```

### With State Management

```typescript
import { StepList } from '@/components/OutlineView';

const ThreadExecution = ({ threadId }: { threadId: string }) => {
  const { steps, updateStep } = useThreadState(threadId);
  const [hoveredId, setHoveredId] = React.useState<string | null>(null);
  const [selectedId, setSelectedId] = React.useState<string | null>(null);

  const handleMRFInjection = async (stepId: string) => {
    const result = await api.injectMRF(threadId, stepId);
    updateStep(stepId, { injectionApplied: 'mrf_verify' });
  };

  return (
    <StepList
      steps={steps}
      threadId={threadId}
      hoveredStepId={hoveredId}
      selectedStepId={selectedId}
      onStepHover={setHoveredId}
      onStepSelect={setSelectedId}
      onInjectMRF={handleMRFInjection}
      showQueryPreview={true}
    />
  );
};
```

## Keyboard Navigation (Future)

Planned keyboard support:
- `Up/Down arrows`: Navigate between steps
- `Enter`: Select step
- `M`: Inject MRF
- `C`: Inject MCTS
- `Escape`: Deselect

## Accessibility

- Semantic HTML (`div` with appropriate ARIA roles where needed)
- Title attributes on hover states
- Sufficient color contrast (WCAG AA)
- Non-color indicators (icons, text labels)
- Focus states on interactive elements

## Performance

- Memoized step index mapping
- Sorted array computed once per render
- Callback stabilization with `useCallback`
- Efficient conditional rendering
- Virtual scrolling ready (use `react-window` for large lists)

## Testing

Key areas to test:
1. Step rendering with all status types
2. Hierarchy and indentation
3. Connector line positioning
4. Progress bar calculations
5. Confidence color coding
6. Injection button visibility
7. Hover and click interactions
8. Empty state rendering
9. Responsive visibility changes

## Future Enhancements

1. **Virtual scrolling** for large step lists (1000+ items)
2. **Drag-and-drop** reordering of steps
3. **Filtering** by status, type, or injection eligibility
4. **Grouping** by parent task
5. **Timeline view** showing step timing
6. **Dependency graph** visualization
7. **Detailed step inspector** panel
8. **Undo/redo** for injection actions
9. **Step templates** for common patterns
10. **Performance profiling** integration
