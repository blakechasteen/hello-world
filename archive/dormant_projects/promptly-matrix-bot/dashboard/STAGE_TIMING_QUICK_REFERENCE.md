# Stage Timing Chart - Quick Reference

## At a Glance

**Component**: `StageTimingChart.tsx`
**Props**: `steps: WeavingStep[]`, `totalLatency: number`
**Location**: Renders below 9-step progress list in WeavingVisualizer

## Visual Guide

```
┌─────────────────────────────────────────────┐
│ Stage Timing Breakdown                      │
├─────────────────────────────────────────────┤
│ Total Latency    | 156.2ms                  │
│ Avg per Stage    | 17.4ms                   │
│ Slowest Stage    | Resonance Shed (45.3ms) │
│ Stages Completed | 9/9                      │
├─────────────────────────────────────────────┤
│                                             │
│ 1. Loom Command  ████░░░░░░░░░░  12.1ms 8%│
│                                             │
│ 4. Resonance     ██████████████░░ 45.3ms   │
│    Shed          29% 🚨 Bottleneck         │
│                                             │
│ 6. Convergence   █████░░░░░░░░░░  18.5ms   │
│    Engine        12%                        │
│                                             │
├─────────────────────────────────────────────┤
│ Fastest: Loom Command (8.1ms)               │
│ Median: 14.5ms                              │
│ Optimization Potential: 29% savings         │
└─────────────────────────────────────────────┘
```

## Color Meanings

| Color | Range | Meaning |
|-------|-------|---------|
| 🟢 | <20% | Optimal - no changes needed |
| 🟡 | 20-40% | Monitor - watch for growth |
| 🔴 | >40% | Bottleneck - optimize this |

## How to Read It

### 1. Find Red Bars
These stages consume >40% of total time
→ These are your bottlenecks

### 2. Check Alert Banner
Lists all bottleneck stages by name
→ Start optimizing from top to bottom

### 3. View Stats
- **Slowest Stage**: Where most time goes
- **Optimization Potential**: How much you could save
- **Average Duration**: Typical stage speed

## Performance Tips

| Problem | Solution |
|---------|----------|
| Retrieval bottleneck (Yarn Graph red) | Reduce memory shards, lower query budget |
| Feature extraction red (Resonance Shed) | Use BARE mode, simpler embeddings |
| Decision bottleneck (Convergence red) | Simplify policy network, fewer tools |
| Multiple red bars | System overloaded, reduce complexity mode |
| All green bars | Great! Production-ready |

## WebSocket Data Format

```json
{
  "step": 4,
  "name": "Resonance Shed",
  "status": "completed",
  "latency_ms": 45.3,
  "metadata": { ... }
}
```

## Integration Example

**In WeavingVisualizer.tsx**:
```tsx
import { StageTimingChart } from './StageTimingChart';

// Inside render:
{currentSteps.length > 0 && (
  <div className="mb-6">
    <StageTimingChart steps={currentSteps} totalLatency={totalLatency} />
  </div>
)}
```

## Testing Checklist

- [ ] Dashboard connects to server (header shows "Connected")
- [ ] Submit a query via WeavingVisualizer
- [ ] Watch stages complete in real-time
- [ ] Chart appears after first stage completes
- [ ] Bars fill as stages progress
- [ ] Colors match performance (green/yellow/red)
- [ ] Bottleneck alert shows if needed
- [ ] Statistics update in real-time
- [ ] No console errors

## Troubleshooting

| Issue | Check |
|-------|-------|
| Chart doesn't appear | Is WebSocket connected? Do stages have latency_ms? |
| Percentages seem wrong | Check rounding in display (1 decimal) |
| All red bars | Stages may be slow, not a bug |
| Tooltip not showing | Try hovering on desktop (mobile uses text) |
| Colors inverted | Check `getColorClass()` function in code |

## Key Metrics to Watch

1. **Total Latency**: Should decrease with optimization
2. **Median Duration**: Better than mean for outliers
3. **Optimization Potential**: How much room to improve
4. **Bottleneck Count**: Should be 0 for optimal

## File Locations

| File | Purpose |
|------|---------|
| `src/components/StageTimingChart.tsx` | Component code |
| `src/components/WeavingVisualizer.tsx` | Parent component |
| `src/types.ts` | Type definitions |
| `STAGE_TIMING_CHART_GUIDE.md` | Full documentation |
| `STAGE_TIMING_IMPLEMENTATION_NOTES.md` | Technical details |

## Common Patterns

### Check for bottlenecks programmatically
```tsx
const bottlenecks = stageData.filter(s => s.percentage >= 0.4);
if (bottlenecks.length > 0) {
  console.log('Found bottlenecks:', bottlenecks);
}
```

### Calculate potential savings
```tsx
const saveable = bottlenecks.reduce((sum, s) => sum + s.duration, 0);
const percent = (saveable / totalLatency) * 100;
console.log(`Could save ${percent.toFixed(1)}%`);
```

### Find slowest stage
```tsx
const slowest = stageData.reduce((max, s) =>
  s.duration > max.duration ? s : max
);
console.log(`Slowest: ${slowest.name} (${slowest.duration}ms)`);
```

## Performance Targets

| Metric | Target | Your System |
|--------|--------|-------------|
| Total Latency | <150ms | Check chart |
| Avg per Stage | <20ms | Statistics show |
| Max Stage | <60ms | Slowest stage shows |
| Bottlenecks | 0 | Alert shows count |

## The 9 Stages at a Glance

```
1. Loom Command        → Choose execution pattern (BARE/FAST/FUSED)
2. Chrono Trigger      → Set up time windows
3. Yarn Graph          → Select memories to use
4. Resonance Shed      → Extract features (embeddings, motifs)
5. Warp Space          → Prepare tensor manifold
6. Convergence Engine  → Choose tool/action
7. Tool Execution      → Run the selected action
8. Spacetime Fabric    → Prepare output + trace
9. Reflection Buffer   → Learn from outcome
```

Which one is slowest?
- Stages 1-3: Memory system slow
- Stage 4: Feature extraction slow
- Stages 5-6: Decision making slow
- Stages 7-9: Output/learning slow

## Monthly Optimization Checklist

```
Week 1: Monitor
- [ ] Run 10+ queries
- [ ] Note which stages are slowest
- [ ] Check for consistent bottlenecks

Week 2: Analyze
- [ ] Group bottlenecks by stage
- [ ] Calculate total optimization potential
- [ ] Document slowest queries

Week 3: Optimize
- [ ] Test optimization strategies
- [ ] Measure latency improvement
- [ ] Document what worked

Week 4: Deploy
- [ ] Roll out optimizations
- [ ] Monitor for regressions
- [ ] Update documentation
```

## Resources

- **Full Guide**: `STAGE_TIMING_CHART_GUIDE.md`
- **Implementation Details**: `STAGE_TIMING_IMPLEMENTATION_NOTES.md`
- **Component Code**: `src/components/StageTimingChart.tsx`
- **Parent Component**: `src/components/WeavingVisualizer.tsx`

---

**Version**: 1.0
**Updated**: November 2025
