# HoloLoom Agent Manager UI - Phase 3: Outline View & Thread Card Components

**Date Created**: December 11, 2025
**Status**: ✅ Production Ready
**Components**: OutlineView.tsx, ThreadCard.tsx
**Total Lines**: ~950 lines of production TypeScript/React code

## Overview

Phase 3 introduces the core outline view components for the HoloLoom Agent Manager UI, enabling visualization and management of agent execution threads in a clean, hierarchical interface.

## Components

### 1. OutlineView Component

**File**: `OutlineView.tsx` (~250 lines)

Main container component that displays all agent threads sorted by priority.

#### Features

- **Priority-Based Sorting**: Threads listed highest priority first, with secondary sort by creation time (newest first)
- **Empty State**: Helpful placeholder when no threads exist
- **Thread Statistics**: Shows running, paused, completed, and failed counts
- **Filter Indicator**: Displays current filter (all, active, completed, failed)
- **Virtualized Scrolling**: CSS-based scrolling with custom scroll styling
- **Responsive Design**: Full dark theme with slate-950 background

#### Usage

```tsx
import { OutlineView } from './components/OutlineView';

// In MainPanel or parent component
<OutlineView />
```

#### Props

None - uses Zustand store directly for state management.

#### Key Features

- Automatic thread sorting (priority descending, then by creation time)
- Real-time updates via Zustand selectors
- Thread statistics tracking
- Scrollable list with custom styling
- Filter status display

### 2. ThreadCard Component

**File**: `ThreadCard.tsx` (~350 lines)

Individual card component for each agent thread with collapsible detail view.

#### Features

**Header Area**:
- Expand/collapse button (chevron icon)
- Status badge with color-coding and animation
- Editable thread name (click to edit, inline)
- Priority controls (+/- buttons, center value display)

**Summary Line**:
- Current step / total steps
- Elapsed time (formatted as ms or s)
- Token usage (formatted with k suffix)
- Token budget percentage (if budget specified)
- Confidence indicator with color-coded dot
- Epistemic confidence indicator with color-coded dot

**Dependencies Section** (if present):
- "Waiting on: X, Y" list if thread has dependencies
- "Blocks: A, B" list if thread blocks others

**Expanded View** (when expanded):
- Step list (0-indexed, showing progress)
- Current step highlighted in blue with pulse animation
- Completed steps in green
- Pending steps in slate gray
- Pause/Resume button (context-aware)
- Cancel button (for running/paused threads)
- MRF Injection button (for metaprompting refinement)
- MCTS Injection button (for Monte Carlo tree search)
- Child threads count indicator

#### Status-Based Styling

| Status | Border Color | Animation | Use Case |
|--------|-------------|-----------|----------|
| idle | gray (slate-700) | None | Thread not started |
| running | blue (blue-500) | Pulse | Thread executing |
| paused | amber (amber-500) | None | Paused by user |
| completed | green (emerald-500) | None | Successfully finished |
| failed | red (red-500) | None | Error encountered |
| cancelled | gray (slate-600) | None | User cancelled |

#### Confidence Indicators

Two independent confidence metrics:

1. **Confidence** (Primary): 0-100%, shows how certain the system is about its result
   - Green (>70%): High confidence
   - Amber (40-70%): Moderate confidence
   - Red (<40%): Low confidence

2. **Epistemic Confidence** (Secondary): 0-100%, shows how confident the system is in its confidence
   - Cyan (>70%): High epistemic confidence
   - Lime (40-70%): Moderate
   - Rose (<40%): Low epistemic confidence

#### Usage

```tsx
import { ThreadCard } from './components/OutlineView';
import { AgentThread } from '../../stores/agentManagerStore';

const thread: AgentThread = {...};

<ThreadCard
  thread={thread}
  isActive={false}
  onSelect={(threadId) => console.log('Selected:', threadId)}
/>
```

#### Props

```typescript
interface ThreadCardProps {
  thread: AgentThread;           // The thread to display
  isActive?: boolean;             // Is this thread currently active?
  onSelect?: (threadId: string) => void;  // Callback when thread is selected
}
```

#### Interaction Patterns

**Editing Thread Name**:
1. Click on thread name (not edit mode)
2. Input becomes editable with blue border
3. Press Enter to save or Escape to cancel
4. Check/X buttons for confirmation/cancellation

**Priority Management**:
- Click "+" to increase priority (capped at 100)
- Click "-" to decrease priority (minimum 0)
- Display shows current priority (0-100)

**Thread Control**:
- **Running**: Shows pause button
- **Paused**: Shows resume button
- **Either**: Shows cancel button to abort

**Advanced Options**:
- MRF button triggers metaprompting refinement injection
- MCTS button triggers Monte Carlo tree search injection

#### Dark Theme Styling

- Background: slate-950 (main), slate-900 (cards when not active), slate-850 (hover state)
- Text: slate-100 (primary), slate-400 (secondary), slate-500 (tertiary)
- Borders: slate-700 (default), status-color-based (left border)
- Active state: ring-2 ring-blue-500

#### Accessibility Features

- Status indicators have `title` attributes with full status name
- Edit mode keyboard shortcuts (Enter to save, Escape to cancel)
- Color + icons for status indication (not color-dependent alone)
- Proper contrast ratios for WCAG compliance

## Integration

### Store Integration

Both components use the Zustand store (`useAgentManagerStore`):

```typescript
// OutlineView uses:
- getFilteredThreads() - Get threads matching current filter
- activeThreadId - Currently selected thread

// ThreadCard uses:
- updateThread() - Update thread name and other properties
- setActiveThread() - Set active thread on click
- upvoteThread() / downvoteThread() - Adjust priority
- pauseThread() / resumeThread() - Control execution
- cancelThread() - Cancel running thread
- getThreadDependencies() - Get dependency info
- getChildThreads() - Get spawned child threads
```

### MainPanel Integration

OutlineView is now integrated into MainPanel.tsx for the outline view mode:

```typescript
// In MainPanel.tsx
case 'outline':
  return <OutlineView />;
```

## Data Flow

```
User Interaction
       ↓
ThreadCard Handler
       ↓
Zustand Store Action
       ↓
Store State Update (immer)
       ↓
Component Re-render via selector
```

## Performance Considerations

1. **Sorting**: `useMemo` caches sorted thread list, recalculates only when threads change
2. **Statistics**: `useMemo` caches thread stats, updated only when thread list changes
3. **Scrolling**: CSS-based overflow (no virtualization needed for typical thread counts)
4. **Re-renders**: Zustand selectors ensure components only re-render when relevant state changes

## Future Enhancements

1. **Drag and Drop**: Reorder threads by priority via drag
2. **Filtering**: Advanced filters (by agent type, confidence, etc.)
3. **Search**: Filter threads by name/ID
4. **Export**: Export thread state for debugging
5. **Step Details**: Click on step to see detailed logs
6. **Batch Operations**: Select multiple threads for bulk actions
7. **Custom Themes**: Theme switcher (light/dark mode options)

## Testing

Components are designed for easy testing:

- **Unit Tests**: Pure functions for sorting, statistics, formatting
- **Integration Tests**: Mock Zustand store, verify UI updates
- **E2E Tests**: Full workflow with real store

## Responsive Design

Both components are responsive:

- On mobile: Thread cards stack vertically
- Expanded details: Auto-scroll for long lists
- Touch-friendly: Larger tap targets (44px minimum)

## Dependencies

- **React**: 18+ (hooks: useState, useMemo)
- **Zustand**: Store and selectors
- **lucide-react**: Icons (ChevronDown, ChevronRight, Check, X, Zap, Target)
- **StatusBadge**: Custom component from common/

## Browser Support

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: Full support (tested on 15+)
- IE11: Not supported (uses ES6+ features)

## Changelog

### v1.0.0 (December 11, 2025)

- ✅ Initial OutlineView component with thread listing
- ✅ ThreadCard component with collapsible details
- ✅ Inline thread name editing
- ✅ Priority controls (up/down voting)
- ✅ Confidence and epistemic confidence indicators
- ✅ Thread dependency visualization
- ✅ MRF and MCTS injection buttons
- ✅ Status-based styling and animations
- ✅ Integration with MainPanel

## Notes for Future Developers

1. **Store Actions**: Thread state updates use Zustand's immer middleware for immutable updates
2. **Accessibility**: All interactive elements have aria labels or title attributes
3. **Dark Theme**: Uses consistent Tailwind slate palette (slate-50 to slate-950)
4. **Scroll Styling**: Custom webkit scrollbar styling in OutlineView (browsers supporting ::-webkit-scrollbar)
5. **Lucide Icons**: All icons from lucide-react package (18px default size)

## File Structure

```
OutlineView/
├── OutlineView.tsx          # Main container component
├── ThreadCard.tsx           # Individual thread card
├── StepRow.tsx              # Step row in expanded view (existing)
├── StepList.tsx             # Step list wrapper (existing)
├── ThreadControls.tsx       # Control buttons (existing)
├── ProgressBar.tsx          # Progress indicator (existing)
├── PriorityControls.tsx     # Priority up/down (existing)
├── types.ts                 # Type definitions
├── index.ts                 # Exports
└── PHASE3_README.md         # This file
```

## Quick Reference

### Creating a ThreadCard

```tsx
const thread: AgentThread = {
  id: 'thread-123',
  name: 'Research Query',
  status: 'running',
  priority: 50,
  agentType: 'research_agent',
  reasoningMode: 'RESEARCH',
  currentStep: 3,
  totalSteps: 5,
  elapsedTimeMs: 5432,
  tokensUsed: 2150,
  tokenBudget: 8000,
  confidence: 0.87,
  epistemicConfidence: 0.82,
  childThreadIds: [],
  dependsOn: [],
  blocks: [],
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
};

<ThreadCard thread={thread} isActive={true} />
```

### Using OutlineView

```tsx
// In MainPanel or parent component
import { OutlineView } from './components/OutlineView';

<OutlineView />
```

The component automatically:
- Fetches filtered threads from store
- Sorts by priority
- Shows statistics
- Handles all interactions
