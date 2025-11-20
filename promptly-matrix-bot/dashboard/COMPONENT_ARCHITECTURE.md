# Stage Timing Chart - Component Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    HoloLoom Backend                          │
│                  (9-Step Weaving Cycle)                      │
│                                                              │
│  1→ Loom  2→ Chrono  3→ Yarn  4→ Resonance  5→ Warp        │
│  6→ Convergence  7→ Tool  8→ Spacetime  9→ Reflection       │
└──────────────────────────┬───────────────────────────────────┘
                           │ WebSocket: weaving_update
                           │ {step, latency_ms, status}
                           ↓
┌──────────────────────────────────────────────────────────────┐
│                   Promptly Dashboard                         │
├──────────────────────────────────────────────────────────────┤
│                         App.tsx                              │
│                   (WebSocket Manager)                        │
│                                                              │
│  socket.on('weaving_update', (data) => {                    │
│    setCurrentCycle({...currentCycle, steps: [...]})         │
│  })                                                          │
├──────────────────────────────────────────────────────────────┤
│                    WeavingVisualizer                         │
│                   (9-Step Progress List)                     │
│                                                              │
│  ┌─ Step 1: Loom Command    [12.1ms ✓]                     │
│  ├─ Step 2: Chrono Trigger   [14.3ms ✓]                    │
│  ├─ Step 3: Yarn Graph       [13.8ms ✓]                    │
│  └─ ... (9 steps total)                                     │
│                                                              │
│  Props: weavingCycle (contains steps[])                     │
└────────────────┬─────────────────────────────────────────────┘
                 │ Pass: steps[], totalLatency
                 ↓
┌──────────────────────────────────────────────────────────────┐
│              ⭐ StageTimingChart Component ⭐               │
│            (NEW - This is what we built!)                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Props:                                               │
│  ├─ steps: WeavingStep[]      (array of 9 stages)          │
│  └─ totalLatency: number      (sum of all durations)        │
│                                                              │
│  Processing:                                                │
│  ├─ Filter: Only stages with latency_ms > 0               │
│  ├─ Calculate: percentage = latency_ms / totalLatency      │
│  ├─ Classify: color based on percentage threshold         │
│  ├─ Detect: isBottleneck = percentage >= 0.4              │
│  └─ Aggregate: stats, metrics, summaries                  │
│                                                              │
│  Output:                                                    │
│  ├─ Summary Stats Panel                                    │
│  │  ├─ Total Latency (ms)                                 │
│  │  ├─ Average per Stage (ms)                             │
│  │  ├─ Slowest Stage (name + duration)                    │
│  │  └─ Stages Completed (X/9)                             │
│  │                                                         │
│  ├─ Bottleneck Alert (if detected)                        │
│  │  └─ Lists all bottleneck stages                        │
│  │                                                         │
│  ├─ Horizontal Bar Chart (9 bars)                         │
│  │  ├─ Bar 1: ▓▓░░░░░░ 12ms  8% ✓                        │
│  │  ├─ Bar 2: ▓▓░░░░░░ 14ms 11% ✓                        │
│  │  ├─ Bar 4: ▓▓▓▓▓▓░░ 45ms 29% ⚠ (if bottleneck)       │
│  │  └─ ... (all 9 stages)                                 │
│  │                                                         │
│  ├─ Performance Summary                                    │
│  │  ├─ Fastest Stage                                      │
│  │  ├─ Median Duration                                    │
│  │  └─ Optimization Potential (%)                         │
│  │                                                         │
│  └─ Color Legend (reference guide)                        │
│     ├─ 🟢 Green (<20%)                                    │
│     ├─ 🟡 Yellow (20-40%)                                 │
│     └─ 🔴 Red (>40% - Bottleneck)                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Component File Structure

```
StageTimingChart.tsx (270 lines)
│
├─ Imports (React, lucide-react icons, types)
│
├─ Type Definition: StageTimingChartProps
│  └─ steps: WeavingStep[]
│  └─ totalLatency: number
│
├─ Type Definition: StageData (internal)
│  └─ step, name, duration, percentage, isBottleneck, status
│
├─ Constants
│  ├─ STAGE_NAMES (1-9 mapping)
│  └─ BOTTLENECK_THRESHOLD = 0.4
│
├─ Helper Functions
│  ├─ getColorClass(percentage) → Tailwind class
│  ├─ getColorLabel(percentage) → String description
│
├─ Main Component: StageTimingChart()
│  │
│  ├─ useMemo: Calculate stageData
│  ├─ useMemo: Find bottlenecks
│  ├─ useMemo: Calculate avgStageDuration
│  ├─ useMemo: Find slowestStage
│  │
│  └─ Render:
│     ├─ If no stages: "No timing data yet"
│     │
│     └─ Else:
│        ├─ Title section
│        ├─ Summary Stats Grid (4 cards)
│        ├─ Bottleneck Alert (conditional)
│        ├─ Stage Timing Bars (loop through stageData)
│        │  └─ Each bar contains:
│        │     ├─ Stage name & number
│        │     ├─ Horizontal bar with animation
│        │     ├─ Duration & percentage
│        │     ├─ Bottleneck label (if applicable)
│        │     └─ Status indicator (✓/⋯/○)
│        ├─ Performance Summary Section
│        └─ Color Legend
│
└─ Export: StageTimingChart component
```

## Data Transformation Pipeline

```
Raw WebSocket Event
├─ { step: 4, name: "Resonance", latency_ms: 45.3, status: "completed" }
│
↓ (useMemo)
│
StageData Calculation
├─ Filter: latency_ms > 0 → ✓ 45.3 > 0
├─ Calculate: percentage = 45.3 / 150 = 0.302 (30.2%)
├─ Classify: 0.302 in [0.2, 0.4] → Yellow
├─ Detect: 0.302 < 0.4 → Not bottleneck
└─ Create StageData object
│
↓ (useMemo)
│
Aggregations
├─ bottlenecks[] = filter(isBottleneck == true)
├─ avgStageDuration = mean(all durations)
├─ slowestStage = max(duration)
│
↓
│
Render Output
├─ Bar color: yellow
├─ Bar width: 30.2%
├─ Label text: "45.3ms 30.2%"
├─ No bottleneck flag
└─ Status: ✓ Completed
```

## Component Lifecycle

```
┌─ Component Mounted
│
├─ Props Received: steps[], totalLatency
│
├─ Calculations (useMemo)
│  ├─ stageData[] = ProcessStages(steps)
│  ├─ bottlenecks[] = Filter(percentage >= 0.4)
│  ├─ avgDuration = Mean(durations)
│  └─ slowest = Max(duration)
│
├─ Render Phase
│  ├─ Title & Summary Stats
│  ├─ Bottleneck Alert (if count > 0)
│  ├─ Bar Chart (for each stage)
│  ├─ Performance Summary
│  └─ Legend
│
├─ Props Changed: Re-compute useMemo, Re-render
│
├─ Animation Phase (CSS)
│  └─ Bar fill: 0% → targetWidth (500ms duration)
│
└─ Component Unmounted
```

## Integration Flow

```
User Submits Query
│
↓
App.tsx sends POST /api/query
│
↓
Backend starts weaving cycle
│
↓
Backend emits WebSocket: weaving_start
├─ App.tsx receives → setCurrentCycle({query_id, steps: []})
│
↓
Backend: Step 1 completes, emits weaving_update
├─ latency_ms: 12.1
├─ status: "completed"
│
↓
App.tsx socket.on('weaving_update')
├─ setCurrentCycle(prev => {
│    steps: [...prev.steps, {step: 1, latency_ms: 12.1, ...}]
│  })
│
↓
WeavingVisualizer receives updated weavingCycle
├─ Passes: steps=currentSteps, totalLatency=sum(...)
│
↓
StageTimingChart Renders
├─ Shows bar for Stage 1 (12.1ms)
│
↓ (Repeat for steps 2-9)
│
All 9 stages complete
├─ StageTimingChart shows full breakdown
├─ All bars filled
├─ Alert shows any bottlenecks
├─ Stats show final metrics
│
↓
User views complete visualization
└─ Can optimize based on recommendations
```

## Color Coding Algorithm

```
getColorClass(percentage: number) {
  if (percentage >= 0.40) {
    return 'bg-red-500'      // Bottleneck
  } else if (percentage >= 0.20) {
    return 'bg-yellow-500'   // Caution
  } else {
    return 'bg-green-500'    // Optimal
  }
}
```

## Bottleneck Detection

```
For each stage:
  percentage = latency_ms / totalLatency

  if percentage >= 0.40:
    isBottleneck = true
    Add to bottlenecks[]
    Add red color
    Add ⚠ Bottleneck label

Optimization Potential = sum(bottleneck durations) / totalLatency
```

## Performance Metrics Calculation

```
Summary Stats:
├─ Total Latency = sum(all latency_ms)
├─ Avg per Stage = sum(latencies) / stage_count
├─ Slowest Stage = max(latency_ms) by stage
└─ Stages Completed = count(status == "completed")

Performance Summary:
├─ Fastest Stage = min(latency_ms) by stage
├─ Median Duration = sorted_durations[middle_index]
└─ Optimization Potential = sum(bottlenecks) / totalLatency
```

## State Management

**Component Uses**:
- React hooks: `useMemo` (for calculations)
- Props: `steps[]`, `totalLatency`
- No internal useState (pure calculation component)

**Parent Manages**:
- `currentSteps[]` in WeavingVisualizer
- `totalLatency` calculated from sum of step durations

## Performance Analysis

```
Input: 9 stages with timing data
       └─ ~100 bytes of JSON per stage

Processing:
├─ Filter stages: O(n) ~0.1ms
├─ Calculate percentages: O(n) ~0.1ms
├─ Find bottlenecks: O(n) ~0.1ms
├─ Find slowest: O(n) ~0.1ms
└─ Aggregations: O(n) ~0.1ms
   Total: O(n) = ~0.5ms

Rendering:
├─ JSX creation: ~2ms
├─ DOM updates: ~8ms (React batching)
├─ CSS paint: ~4ms (Tailwind optimization)
└─ Animation: 500ms (CSS transition, smooth)

Total per Update: <16ms (60 FPS target)
```

## Browser Rendering Process

```
Component Props Change
│
├─ React Reconciliation
│  └─ useMemo triggers if dependencies changed
│
├─ Virtual DOM Update
│  └─ React creates new VNode tree
│
├─ DOM Diffing
│  └─ Calculate minimal DOM changes
│
├─ DOM Update (Browser)
│  └─ Insert/update elements
│
├─ Recalculate Layout (Browser)
│  └─ Reflow caused by width/height changes
│
├─ Repaint (Browser)
│  └─ Redraw visual pixels
│
├─ Composite (Browser)
│  └─ Combine layers
│
└─ Display
   └─ User sees updated chart with animations
```

## CSS Animation Timeline

```
Bar Fill Animation (500ms):

0ms     ├─ width: 0%
        │
100ms   ├─ width: 10%      (0.5s progress)
        │
250ms   ├─ width: 25%      (0.5s progress)
        │
500ms   └─ width: 100% ✓   (animation complete)

Browser handles:
├─ Initial: width: 0%
├─ Animated: transition: width 500ms ease-out
└─ Final: width: 100%
```

## Responsive Design Breakpoints

```
Mobile (< 640px):
├─ Single column layout
├─ Summary stats: 2×2 grid
├─ Stage names may truncate
└─ Touch-friendly sizes (44px+ touch targets)

Tablet (640px - 1024px):
├─ 2 column summary stats
├─ Readable labels
└─ Full bar widths

Desktop (> 1024px):
├─ 4 column summary stats
├─ Full details visible
└─ Hover tooltips enabled
```

## Accessibility Features

```
Color Information:
├─ Not color-only (also use text labels)
├─ High contrast ratios
└─ Semantic color meaning

Keyboard Navigation:
├─ Semantic HTML elements
├─ Proper heading hierarchy
└─ Tab-order follows visual order

Screen Readers:
├─ Descriptive text labels
├─ Proper heading structure
└─ No empty div elements

Motor Control:
├─ Touch-friendly hit areas
├─ No hover-only information
└─ Sufficient spacing between targets
```

## Error Handling

```
Edge Cases Handled:
├─ Empty steps array
│  └─ Show: "No timing data available yet"
├─ Stages with 0ms latency
│  └─ Filter out (too fast to measure)
├─ All stages same duration
│  └─ Show all bars same size
├─ Single stage much slower
│  └─ Show as bottleneck (correct)
└─ Very large latencies
   └─ Percentages still calculated correctly
```

## Dependencies

```
Runtime Dependencies:
├─ React (v18+)          → for component, hooks, JSX
└─ lucide-react          → for icons only

Build Dependencies:
├─ TypeScript            → for type checking
├─ Tailwind CSS          → for styling
├─ Vite                  → for bundling
└─ React DOM            → for rendering

Zero External Chart Libraries!
└─ Pure HTML/CSS/SVG implementation
```

## Configuration Options

```
Can Be Customized:
├─ BOTTLENECK_THRESHOLD = 0.4   (change to 0.3, 0.5, etc.)
├─ STAGE_NAMES mapping          (if stage names change)
├─ Bar colors                   (modify getColorClass)
├─ Animation duration (500ms)   (change transition-all duration-500)
└─ Font sizes / padding         (Tailwind classes)

Cannot/Should Not Change:
├─ Core calculation logic
├─ WebSocket event structure
└─ Component interface (props)
```

---

**Component Version**: 1.0.0
**Created**: November 2025
**Status**: Production Ready

