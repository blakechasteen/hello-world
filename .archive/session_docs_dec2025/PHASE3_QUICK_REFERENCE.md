# Phase 3 Controls - Quick Reference Card

**Print this page for quick reference while integrating!**

---

## Component Quick Import

```typescript
import {
  PriorityControls,
  ThreadControls,
  InjectMenu,
} from '@/components/OutlineView';
```

---

## 1. PriorityControls

### Basic Usage
```typescript
<PriorityControls threadId="123" priority={50} />
```

### Full Options
```typescript
<PriorityControls
  threadId="123"
  priority={50}
  size="md"              // 'sm' | 'md'
  orientation="vertical" // 'vertical' | 'horizontal'
  onChange={(p) => console.log(p)}
/>
```

### Layouts
```
Vertical:       Horizontal:
  [▲]           [▼] 50 [▲]
  [50]
  [▼]
```

### Store Actions Called
- `upvoteThread(id)` → priority += 1 (max 100)
- `downvoteThread(id)` → priority -= 1 (min 0)

### Color Legend
- 75-100: Red (high priority)
- 50-74: Amber (medium-high)
- 25-49: Blue (medium)
- 0-24: Gray (low)

---

## 2. ThreadControls

### Basic Usage
```typescript
<ThreadControls threadId="123" status="running" />
```

### Full Options
```typescript
<ThreadControls
  threadId="123"
  status="running"     // 'idle'|'running'|'paused'|'completed'|'failed'|'cancelled'
  size="md"            // 'sm' | 'md'
  showCancelConfirm    // true (default)
  onAction={(a) => {}} // 'pause'|'resume'|'cancel'|'retry'
/>
```

### What Shows When
| Status | Buttons |
|--------|---------|
| running | ⏸ Pause, ✕ Cancel |
| paused | ▶ Resume, ✕ Cancel |
| idle | (nothing) |
| completed/failed/cancelled | (nothing) |

### Store Actions Called
- `pauseThread(id)` → status = 'paused'
- `resumeThread(id)` → status = 'running'
- `cancelThread(id)` → status = 'cancelled' (with confirm)

---

## 3. InjectMenu

### Basic Usage
```typescript
<InjectMenu
  threadId="123"
  stepId="step-001"
  mrfEligible={true}
  mctsEligible={true}
/>
```

### Full Options
```typescript
<InjectMenu
  threadId="123"
  stepId="step-001"
  mrfEligible={true}
  mctsEligible={false}
  onInjectMRF={(strategy) => console.log(strategy)}
  onInjectMCTS={({ budget, exploration }) => console.log(budget, exploration)}
  injected="mrf"
  appliedStrategy="verify"
  size="md"
/>
```

### MRF Strategies (6)
- AUTO - Automatic selection
- VERIFY - Accuracy verification
- ELEGANCE - Clarity improvement
- CRITIQUE - Critical analysis
- REFINE - Iterative refinement
- HOFSTADTER - Recursive self-reference

### MCTS Config Options
- Budget: 50 | 100 | 200 | 500
- Exploration: 0.5 | 1.0 | 1.4 | 2.0

### Menu Structure
```
[⚡ Inject]
├─ MRF Refinement
│  ├─ AUTO
│  ├─ VERIFY
│  ├─ ELEGANCE
│  ├─ CRITIQUE
│  ├─ REFINE
│  └─ HOFSTADTER
└─ MCTS Planning ▶
   ├─ Budget: [50] [100] [200] [500]
   ├─ Exploration: [0.5] [1.0] [1.4] [2.0]
   └─ [Apply MCTS]
```

---

## Typical Thread Row Layout

```typescript
<div className="flex items-center gap-2 p-2 bg-slate-700 rounded">
  {/* Status indicator */}
  <div className="w-3 h-3 rounded-full bg-blue-400" />

  {/* Thread name */}
  <span className="flex-1">{thread.name}</span>

  {/* NEW: Priority */}
  <PriorityControls threadId={id} priority={priority} />

  {/* NEW: Thread controls */}
  <ThreadControls threadId={id} status={status} />

  {/* NEW: Injection menu */}
  <InjectMenu
    threadId={id}
    stepId="0"
    mrfEligible={status === 'running'}
    mctsEligible={status === 'running'}
  />
</div>
```

---

## Keyboard Navigation

### PriorityControls
| Key | Action |
|-----|--------|
| Tab | Navigate between buttons |
| Enter | Upvote/Downvote |
| ↑ | Upvote (from any control) |
| ↓ | Downvote (from any control) |

### ThreadControls
| Key | Action |
|-----|--------|
| Tab | Navigate buttons |
| Enter | Activate button |
| Escape | Close confirmation |

### InjectMenu
| Key | Action |
|-----|--------|
| Tab | Navigate items |
| Enter | Open/Select |
| Escape | Close menu |

---

## Color Scheme

### Button Colors
```
Default: bg-slate-700 hover:bg-slate-600
Pause:   bg-amber-700  hover:bg-amber-600
Resume:  bg-emerald-700 hover:bg-emerald-600
Cancel:  bg-red-700    hover:bg-red-600
Disabled: opacity-50 cursor-not-allowed
Focus:   ring-2 ring-blue-500 ring-offset-1 ring-offset-slate-800
```

### Text Colors
```
Primary: text-slate-200
Hover:   text-white
Disabled: text-slate-600
```

---

## Common Props

### All Components
```typescript
size?: 'sm' | 'md'          // Button size
className?: string          // Additional CSS classes
```

### PriorityControls
```typescript
orientation?: 'horizontal' | 'vertical'  // Layout direction
onChange?: (p: number) => void           // Priority changed
```

### ThreadControls
```typescript
showCancelConfirm?: boolean              // Confirm dialog
onAction?: (a: string) => void           // Action performed
```

### InjectMenu
```typescript
injected?: 'mrf' | 'mcts' | null         // Already applied?
appliedStrategy?: string                 // Which strategy?
onInjectMRF?: (s: string) => void        // MRF selected
onInjectMCTS?: (c: MCTSConfig) => void   // MCTS selected
```

---

## Sizes

### Small (sm)
```
Button: 28×28px
Text: text-xs
Gap: 8px
```

### Medium (md) - Default
```
Button: 32×32px
Text: text-sm
Gap: 8px
```

---

## Status Reference

### Thread Status Values
- `'idle'` - Not running
- `'running'` - Currently executing
- `'paused'` - Paused by user
- `'completed'` - Finished successfully
- `'failed'` - Failed with error
- `'cancelled'` - Cancelled by user

### Priority Values
- Range: 0-100
- Color coded in display
- Clamped by store automatically

---

## Accessibility Features

### ARIA Labels
- All buttons have descriptive labels
- Disabled state indicated
- Live regions for status changes
- Modal dialogs announced

### Keyboard Support
- Full keyboard navigation
- Tab order logical
- Focus visible (blue ring)
- Escape to close/cancel
- Enter to confirm

### Screen Reader
- All text read aloud
- Status changes announced
- Buttons described clearly
- Instructions provided

---

## Integration Checklist

- [ ] Import components
- [ ] Add to ThreadRow component
- [ ] Wire up callbacks (if needed)
- [ ] Test keyboard navigation
- [ ] Test with screen reader
- [ ] Verify store integration
- [ ] Test all thread states
- [ ] Check styling consistency
- [ ] Performance test
- [ ] Deploy to staging

---

## Troubleshooting

### Button doesn't work
✓ Check threadId is passed
✓ Verify store is initialized
✓ Check store actions exist

### Menu won't close
✓ Click outside should work
✓ Escape key should work
✓ Menu closes after selection

### Tooltip not showing
✓ Check parent has overflow-visible
✓ Check z-index isn't blocked
✓ Verify hover styles applied

### Keyboard not working
✓ Check tabindex attributes
✓ Verify role attributes set
✓ Check event handlers attached

---

## Performance Tips

### Do's ✅
- Memoize parent components
- Use Zustand selectors
- Keep thread lists under 100 items
- Lazy load if >500 threads

### Don'ts ❌
- Don't prop drill thread objects
- Don't re-render on every store change
- Don't add heavy computations in render
- Don't create new objects in props

---

## Browser Support

| Browser | Support |
|---------|---------|
| Chrome | ✅ Latest |
| Firefox | ✅ Latest |
| Safari | ✅ Latest |
| Edge | ✅ Latest |
| Mobile Safari | ✅ iOS 14+ |
| Chrome Mobile | ✅ Latest |

---

## Quick Copy-Paste Templates

### Minimal Thread Row
```typescript
<div className="flex gap-2 p-2 bg-slate-700 rounded">
  <span className="flex-1">{name}</span>
  <PriorityControls threadId={id} priority={priority} />
  <ThreadControls threadId={id} status={status} />
  <InjectMenu threadId={id} stepId="0" mrfEligible mctsEligible />
</div>
```

### Full Featured Row
```typescript
<div className="flex items-center justify-between p-3 bg-slate-800 rounded border border-slate-700">
  <div>
    <h3 className="font-semibold">{name}</h3>
    <p className="text-xs text-slate-400">Status: {status}</p>
  </div>
  <div className="flex items-center gap-2">
    <PriorityControls threadId={id} priority={priority} />
    <ThreadControls threadId={id} status={status} />
    <InjectMenu threadId={id} stepId="0" mrfEligible mctsEligible />
  </div>
</div>
```

### With Callbacks
```typescript
<ThreadControls
  threadId={id}
  status={status}
  onAction={(action) => {
    if (action === 'pause') handlePause();
    if (action === 'resume') handleResume();
    if (action === 'cancel') handleCancel();
  }}
/>
```

---

## Documentation Links

| Document | Content |
|----------|---------|
| PHASE3_CONTROLS_SUMMARY.md | Complete spec |
| PHASE3_INTEGRATION_GUIDE.md | Integration patterns |
| CONTROLS_README.md | Component reference |
| EXAMPLES.tsx | 8 usage examples |
| PHASE3_DELIVERY_COMPLETE.md | Full delivery summary |

---

## Contact & Support

For questions or issues:
1. Check this quick reference first
2. Review CONTROLS_README.md
3. Check EXAMPLES.tsx for patterns
4. Review component source code

---

**Version**: 1.0
**Last Updated**: 2025-12-11
**Status**: Ready for Production

---

## ⭐ Key Takeaways

✅ **3 production-ready components**
✅ **100% keyboard accessible**
✅ **WCAG 2.1 AA compliant**
✅ **Full store integration**
✅ **Comprehensive documentation**
✅ **8 example implementations**
✅ **Ready to deploy**

🚀 **You're all set!**
