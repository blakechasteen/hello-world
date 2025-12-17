# Outline View Components - Quick Reference

**Phase 3 Implementation** | **December 11, 2025**

## TL;DR

Two main components for displaying task execution steps:
- **StepList**: Container showing all steps with progress
- **StepRow**: Individual step with status, progress, and actions

## Quick Start

```typescript
import { StepList, type TaskNode } from '@/components/OutlineView';

// Create task nodes
const steps: TaskNode[] = [
  {
    id: 'step-1',
    threadId: 'thread-1',
    depth: 0,
    stepType: 'query',
    name: 'Parse Query',
    status: 'completed',
    progressPct: 100,
    elapsedTimeMs: 1500,
    tokensUsed: 256,
    confidence: 0.92,
    dependsOn: [],
    blocks: [],
    mrfEligible: true,
    mctsEligible: false,
    childrenIds: [],
  },
  // ... more steps
];

// Render
<StepList
  steps={steps}
  threadId="thread-1"
  onStepSelect={(id) => console.log('Selected:', id)}
  onInjectMRF={(id) => console.log('MRF:', id)}
/>
```

## Component Props

### StepList
```typescript
<StepList
  steps={steps}                           // Required: TaskNode[]
  threadId="thread-1"                     // Required: string
  rootTask={rootTask}                     // Optional: TaskNode
  selectedStepId={selected}               // Optional: string | null
  onStepSelect={handleSelect}             // Optional: (id: string) => void
  onInjectMRF={handleMRF}                 // Optional: (id: string) => void
  onInjectMCTS={handleMCTS}               // Optional: (id: string) => void
  showQueryPreview                        // Optional: boolean (default: true)
  className="custom-class"                // Optional: string
/>
```

### StepRow
```typescript
<StepRow
  step={taskNode}                         // Required: TaskNode
  depth={0}                               // Required: number (0-based)
  isSelected={false}                      // Optional: boolean
  onHover={(id) => {}}                    // Optional: (id: string | null) => void
  onClick={(id) => {}}                    // Optional: (id: string) => void
  onInjectMRF={(id) => {}}                // Optional: (id: string) => void
  onInjectMCTS={(id) => {}}               // Optional: (id: string) => void
  showQueryPreview                        // Optional: boolean (default: true)
/>
```

## TaskNode Type

```typescript
interface TaskNode {
  id: string;                     // Unique ID
  threadId: string;               // Parent thread
  parentId?: string;              // Parent task (for hierarchy)
  childrenIds: string[];          // Child tasks
  depth: number;                  // Nesting level
  stepType: 'query' | 'research' | 'verify' | 'synthesize' | 'execute';
  name: string;                   // Display name
  query?: string;                 // Query text
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  progressPct: number;            // 0-100
  elapsedTimeMs: number;          // Time elapsed
  tokensUsed: number;             // Tokens consumed
  confidence: number;             // 0.0-1.0
  dependsOn: string[];            // Task IDs this depends on
  blocks: string[];               // Task IDs this blocks
  mrfEligible: boolean;           // Can use MRF injection
  mctsEligible: boolean;          // Can use MCTS injection
  injectionApplied?: string;      // Applied injection name
}
```

## Status Icons

| Status | Icon | Color |
|--------|------|-------|
| pending | ○ | slate-400 |
| running | ◐ (spinning) | blue-400 |
| completed | ✓ | emerald-500 |
| failed | ✗ | red-500 |
| skipped | — | slate-500 |

## Step Types (with Emojis)

| Type | Emoji | Description |
|------|-------|-------------|
| query | 🔍 | Initial query parsing |
| research | 📚 | Research/exploration |
| verify | ✓ | Verification/validation |
| synthesize | 🔀 | Synthesis/combination |
| execute | ⚙️ | Execution/generation |

## Confidence Colors

```
High   (>0.8)   : 🟢 emerald-400
Medium (0.5-0.8): 🟡 amber-400
Low    (<0.5)   : 🔴 red-400
```

## Layout

### StepRow Layout
```
[Icon] [Name] [Progress] [Confidence] [Time] [Tokens] [Actions]
```

### Responsive Visibility
- **Mobile** (<640px): Minimal info, actions on hover
- **Small+** (640px): Elapsed time shown
- **Medium+** (1024px): Token usage shown
- **Large+** (1280px): Dependency indicators shown

## Common Patterns

### Basic List
```typescript
<StepList steps={steps} threadId="t1" />
```

### With Callbacks
```typescript
<StepList
  steps={steps}
  threadId="t1"
  onStepSelect={handleSelect}
  onInjectMRF={async (id) => {
    const result = await api.injectMRF(id);
    // Update state...
  }}
/>
```

### Standalone Row
```typescript
<StepRow
  step={steps[0]}
  depth={0}
  isSelected={selectedId === steps[0].id}
  onClick={handleSelect}
/>
```

### Controlled Component
```typescript
const [selected, setSelected] = useState<string | null>(null);
const [hovered, setHovered] = useState<string | null>(null);

<StepList
  steps={steps}
  threadId="t1"
  selectedStepId={selected}
  hoveredStepId={hovered}
  onStepSelect={setSelected}
  onStepHover={setHovered}
/>
```

## Styling

### Colors (Dark Theme)
```
Primary bg:   slate-800/50
Hovered:      slate-700/50
Selected:     slate-700
Header:       slate-800/30
Footer:       slate-800/20
Border:       slate-700
Text:         slate-100
```

### Sizing
```
Row height:        32px
Icon size:         20x20px
Indentation:       12px per level
Max list height:   384px
```

### Animations
- Running: `animate-spin` + `animate-pulse`
- Transitions: `transition-all duration-300`

## Features

### ✅ Implemented
- [x] Status icons with animations
- [x] Confidence color coding
- [x] Progress bars
- [x] Query preview tooltips
- [x] MRF/MCTS injection buttons
- [x] Injection badges
- [x] Hierarchy indentation
- [x] Connector lines
- [x] Progress tracking
- [x] Empty states
- [x] Responsive design
- [x] Dark theme
- [x] Accessibility

### 🔄 Coming Soon (Phase 4+)
- Keyboard navigation
- Context menus
- Drag-and-drop
- Filtering/sorting
- Virtual scrolling
- Dependency graphs
- Timeline view

## Troubleshooting

### Buttons not showing?
- They appear on hover by default
- Check `mrfEligible` and `mctsEligible` flags
- Verify `injectionApplied` is not set

### Query preview not showing?
- Set `showQueryPreview={true}`
- Provide `step.query` text
- Hover over step name area

### Progress bar not visible?
- Only shows for `status: 'running'`
- Check `progressPct` > 0

### Styles not applying?
- Import Tailwind CSS
- Use dark background (slate-900+)
- Check viewport for responsive classes

## File Imports

```typescript
// Components
import { StepList, StepRow } from '@/components/OutlineView';

// Types
import type { TaskNode, StepType, StepStatus } from '@/components/OutlineView';

// With aliases
import { StepList as Steps, StepRow as Step } from '@/components/OutlineView';
```

## Performance Tips

1. **Memoize callbacks**:
   ```typescript
   const handleSelect = useCallback((id) => {
     // ...
   }, [dependencies]);
   ```

2. **Use stable task IDs** (not indices)

3. **For 500+ steps**: Consider `react-window` virtualization

4. **Avoid inline functions**:
   ```typescript
   // Bad
   onInjectMRF={(id) => updateStep(id)}

   // Good
   const handleInjectMRF = useCallback((id) => updateStep(id), [])
   ```

## Testing

```typescript
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { StepRow } from '@/components/OutlineView';

it('shows completed checkmark', () => {
  render(
    <StepRow
      step={{ ...mockStep, status: 'completed' }}
      depth={0}
    />
  );
  expect(screen.getByText(/✓/)).toBeInTheDocument();
});
```

## Examples

See `StepList.demo.tsx` for interactive examples:
- Basic step list
- Running step simulation
- Injection handling
- Stats calculation

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile (iOS, Android)

## Questions?

Check:
1. **README.md** - Full documentation
2. **StepList.demo.tsx** - Working examples
3. **types.ts** - Type definitions
4. **StepRow.test.tsx** - Test cases

---

**Last Updated**: December 11, 2025
**Phase**: 3 (Outline View)
**Status**: ✅ Production Ready
