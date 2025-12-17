# Phase 1 Components - Quick Reference Card

Print-friendly quick reference for using Agent Manager UI components.

## Component Imports

```tsx
import {
  Layout,           // Main container
  Header,           // Navigation bar
  Sidebar,          // Configuration panel
  MainPanel,        // Content area
  StatusBadge,      // Full status badge
  StatusIndicator,  // Compact dot
  StatusGrid,       // Status overview grid
} from '@components';
```

## Usage Patterns

### Minimal App
```tsx
<Layout />
// That's it! Includes Header, Sidebar, MainPanel
```

### Custom Layout
```tsx
<div className="flex flex-col h-screen bg-slate-950">
  <Header onToggleSidebar={() => {...}} />
  <div className="flex flex-1">
    <Sidebar />
    <MainPanel />
  </div>
</div>
```

### Status Indicators
```tsx
// Full badge with label
<StatusBadge status="running" size="md" />
// → ▶ Running

// Icon only
<StatusBadge status="running" showLabel={false} />
// → ▶

// Compact dot
<StatusIndicator status="running" />
// → 💠 (pulsing)

// Status grid
<StatusGrid running={5} completed={12} failed={2} />
// → 3 colored boxes with counts
```

## Status Types & Colors

| Type | Icon | Color | Animation |
|------|------|-------|-----------|
| `idle` | ○ | Gray | None |
| `running` | ▶ | Blue | Pulse |
| `paused` | ⏸ | Amber | None |
| `completed` | ✓ | Green | None |
| `failed` | ✕ | Red | None |
| `cancelled` | × | Gray | None |

## Component Props

### Header
```tsx
<Header onToggleSidebar?={() => void} />
```
- Reads/writes: `filter`, `viewMode`, `isConnected`

### Sidebar
```tsx
<Sidebar />
```
- No props
- Internal state: Agent type, reasoning mode, budget, priority

### MainPanel
```tsx
<MainPanel />
```
- No props
- Renders based on: `viewMode` (outline|tree|swarm)

### StatusBadge
```tsx
<StatusBadge
  status="running"      // Required
  size="md"             // sm | md | lg (default: md)
  showLabel={true}      // default: true
  className=""          // Extra classes
/>
```

### StatusIndicator
```tsx
<StatusIndicator
  status="running"      // Required
  size="md"             // sm | md | lg (default: md)
  className=""
/>
```

### StatusGrid
```tsx
<StatusGrid
  idle={0}              // Optional
  running={5}           // Optional
  paused={0}            // Optional
  completed={12}        // Optional
  failed={2}            // Optional
  cancelled={0}         // Optional
  className=""          // Optional
/>
```

## Zustand Store Access

```tsx
// Read state
const { filter, viewMode, isConnected } = useAgentManagerStore();

// Write state
const { setFilter, setViewMode, setConnectionStatus } = useAgentManagerStore();

// Get threads
const threads = useAgentManagerStore((state) => state.getFilteredThreads());

// Example
if (isConnected) {
  setFilter('active');
  setViewMode('tree');
}
```

## Common Tasks

### Show thread status
```tsx
<StatusBadge status={thread.status} size="md" />
```

### Switch views
```tsx
<button onClick={() => setViewMode('tree')}>Tree View</button>
```

### Filter threads
```tsx
<button onClick={() => setFilter('active')}>Show Active</button>
```

### Get filtered list
```tsx
const threads = useAgentManagerStore((state) => state.getFilteredThreads());
threads.forEach(t => console.log(t.name));
```

### Update connection status
```tsx
const { setConnectionStatus } = useAgentManagerStore();
ws.onopen = () => setConnectionStatus(true);
ws.onerror = () => setConnectionStatus(false, "Error");
```

## Dark Theme Colors

**Use in custom components:**

| Color | Tailwind | Hex |
|-------|----------|-----|
| Background | `bg-slate-950` | #030712 |
| Panel | `bg-slate-900` | #0f172a |
| Border | `border-slate-800` | #1e293b |
| Text (primary) | `text-slate-100` | #f1f5f9 |
| Text (secondary) | `text-slate-400` | #94a3b8 |
| Text (tertiary) | `text-slate-500` | #64748b |
| Accent (primary) | `bg-emerald-600` | #16a34a |
| Accent (info) | `bg-blue-600` | #2563eb |
| Accent (warning) | `bg-amber-600` | #d97706 |
| Accent (danger) | `bg-red-600` | #dc2626 |

## Sizes Guide

### StatusBadge/Indicator Sizes
- **sm**: `px-2 py-1 text-xs` (compact, inline)
- **md**: `px-2.5 py-1.5 text-sm` (default, standard)
- **lg**: `px-3 py-2 text-base` (prominent, headers)

### Spacing
- **Gap**: `gap-1` to `gap-6` (4px to 24px)
- **Padding**: `px-2` to `px-6`, `py-1` to `py-3`
- **Margin**: `m-2` to `m-8`

## Responsive Classes

```tsx
// Show on mobile, hide on desktop
<div className="md:hidden">Mobile</div>

// Hide on mobile, show on desktop
<div className="hidden md:block">Desktop</div>

// Responsive padding
<div className="p-2 md:p-4 lg:p-6">Content</div>

// Responsive grid
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
```

## Animations

**Built-in animations:**
- `animate-pulse` - Gentle pulse (used for running status)
- `animate-bounce` - Gentle bounce (used for running icon)
- `transition-colors` - Smooth color transitions (200ms default)
- `transition-all` - Smooth all property transitions

**Custom animations:**
```tsx
<div className="transition-all duration-300 ease-in-out">
  Content with smooth transitions
</div>
```

## TypeScript Types

```tsx
type FilterType = 'all' | 'active' | 'completed' | 'failed';
type ViewMode = 'outline' | 'tree' | 'swarm';
type ThreadStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
type BadgeSize = 'sm' | 'md' | 'lg';
```

## Testing Tips

### Test dark theme
```tsx
<div className="bg-slate-950 text-slate-100">
  <Layout />
</div>
```

### Test connection status
```tsx
const { setConnectionStatus } = useAgentManagerStore();
setConnectionStatus(true);   // Connected
setConnectionStatus(false);  // Disconnected
```

### Test filter
```tsx
const { setFilter } = useAgentManagerStore();
['all', 'active', 'completed', 'failed'].forEach(f => {
  setFilter(f);
  // Component re-renders with new filter
});
```

### Test view mode
```tsx
const { setViewMode } = useAgentManagerStore();
['outline', 'tree', 'swarm'].forEach(v => {
  setViewMode(v);
  // Component re-renders with new view
});
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Components not styled | Ensure Tailwind CSS is imported in main.tsx |
| Store not updating | Verify Zustand import: `import { useAgentManagerStore } from '@stores'` |
| Path alias not working | Check `vite.config.ts` has `@components` alias |
| Types not recognized | Run `npm run type-check` to find TS errors |
| Dark theme looks light | Check HTML has `dark` class or Tailwind darkMode config |

## Performance Tips

1. **Use selectors** to read only needed state:
   ```tsx
   const viewMode = useAgentManagerStore((s) => s.viewMode); // Good
   const all = useAgentManagerStore(); // Bad - re-renders often
   ```

2. **Memoize components** that receive filtered data:
   ```tsx
   const ThreadList = React.memo(({ threads }) => (...));
   ```

3. **Use CSS classes** instead of inline styles:
   ```tsx
   className="bg-slate-800" // Fast
   style={{ backgroundColor: 'slate' }} // Slower
   ```

4. **Avoid creating objects in render**:
   ```tsx
   // Good
   const config = { filter: 'active' };
   <Component config={config} />

   // Bad
   <Component config={{ filter: 'active' }} />
   ```

## Migration Notes

### From Phase 0 (if any)
- All components rewritten from scratch
- Zustand store replaces any context API
- Tailwind CSS required

### Preparation for Phase 2
- Sidebar "Create Thread" button currently logs to console
- Phase 2 will add WebSocket integration
- No changes needed to Phase 1 components

### Preparation for Phase 3
- MainPanel views are placeholders
- Phase 3 will implement Outline, Tree, Swarm views
- Component APIs remain the same

## Additional Resources

- **Full Docs**: See `COMPONENTS_PHASE_1_COMPLETE.md`
- **Integration Guide**: See `INTEGRATION_GUIDE.md`
- **Examples**: See `src/examples/LayoutExample.tsx`
- **Zustand Store**: See `src/stores/agentManagerStore.ts`

## Quick Commands

```bash
# Install dependencies
npm install

# Type check
npm run type-check

# Build
npm run build

# Dev server
npm run dev

# Format code
npm run format

# Lint
npm run lint
```

---

**Print this page or bookmark for quick reference while developing.**

**Last Updated:** December 2025 | **Status:** ✅ Phase 1 Complete
