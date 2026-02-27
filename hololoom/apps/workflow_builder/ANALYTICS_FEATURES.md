# Workflow Analytics Dashboard - Feature Details

## Visual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  📊 Workflow Analytics    🔄 Refresh  ⏱️ Last 24 Hours  ← Back  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│     47       │    1128ms    │      89%     │      72%     │     152      │     128      │
│  Total Wf    │  Avg Latency │  Success    │  Cache Hit   │   Nodes      │   Edges      │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

┌────────────────────────────────────────┐  ┌────────────────────────────────────────┐
│  ⏱️ Execution Timeline                  │  │  📈 Confidence Trajectory               │
├────────────────────────────────────────┤  ├────────────────────────────────────────┤
│                                        │  │                                        │
│ HoloLoom Query ████░░░ 450ms ⚠️        │  │  1.0 ╱╲    ╱╲ ╱╲                       │
│ Memory Search ██░░░░░  120ms          │  │   0.8╱  ╲╱  ╱  ╲╱                      │
│ Synthesizer   ███░░░░░ 280ms          │  │   0.6                                   │
│ Guardrails    █░░░░░░░  85ms          │  │   0.4                                   │
│ Response Gen  ███░░░░░ 320ms          │  │   0.2                                   │
│                                        │  │   0.0└─────────────────────────────    │
│ Legend: Width=Latency | # on right=calls│  │        Avg: 0.86  Min: 0.72  Max: 0.95│
└────────────────────────────────────────┘  └────────────────────────────────────────┘

┌────────────────────────────────────────┐  ┌────────────────────────────────────────┐
│  🔧 Node Performance Summary             │  │  💾 Cache Effectiveness                │
├────────────────────────────────────────┤  ├────────────────────────────────────────┤
│                                        │  │                                        │
│ ✓ HoloLoom Query    450ms  28 calls    │  │        72%                             │
│ ✓ Memory Search     120ms  45 calls    │  │        Good                            │
│ ✓ Synthesizer       280ms  22 calls    │  │    Cache Hits: 34                      │
│ ✓ Guardrails         85ms  35 calls    │  │    Total Queries: 47                   │
│ ⚠️ Response Gen      320ms  28 calls    │  │    Time Saved: 4.1s                    │
│ ✓ Conf. Scorer       95ms  38 calls    │  │    Avg Speedup: 15-30x                 │
│ ✓ Context Packer    210ms  15 calls    │  │                                        │
│                                        │  │ Cache is working well. Continue        │
└────────────────────────────────────────┘  │ monitoring for optimal results.        │
                                             └────────────────────────────────────────┘

┌────────────────────────────────────────┐  ┌────────────────────────────────────────┐
│  📋 Recent Workflow Executions           │  │  ⚠️ Anomaly Detection                  │
├────────────────────────────────────────┤  ├────────────────────────────────────────┤
│                                        │  │                                        │
│ ✓ Research Pipeline         1250ms    │  │ ⬇️ Sudden Confidence Drop               │
│   5 nodes • 4 edges • ID: wf-001      │  │    Query: "What are the tradeoffs?"    │
│                                        │  │    Dropped from 0.91 to 0.72           │
│ ✓ Lead Scoring Workflow      890ms    │  │                                        │
│   4 nodes • 3 edges • ID: wf-002      │  │ 📉 Prolonged Low Confidence             │
│                                        │  │    3 consecutive queries below 0.75    │
│ ✗ Content Analysis           2100ms   │  │                                        │
│   6 nodes • 5 edges • ID: wf-003      │  │ 💾 Cache Miss Cluster                   │
│                                        │  │    4 misses in 2-minute window         │
│ ✓ Safety Verification        450ms    │  │                                        │
│   3 nodes • 2 edges • ID: wf-004      │  │                                        │
│                                        │  │                                        │
│ ✓ BDR Email Generator        1800ms   │  │                                        │
│   7 nodes • 6 edges • ID: wf-005      │  │                                        │
└────────────────────────────────────────┘  └────────────────────────────────────────┘

Last updated: 10:30:45 | Auto-refresh enabled (30s) | Documentation
```

## Panel Breakdown

### 1. Summary Metrics (Top Bar)

**Purpose**: At-a-glance system health overview

**Components**:
- **Total Workflows**: Cumulative count (47)
- **Average Latency**: Mean execution time (1128ms)
- **Success Rate**: Percentage of completed workflows (89%)
- **Cache Hit Rate**: Percentage of cache hits (72%)
- **Agent Nodes**: Total workflow nodes (152)
- **Connections**: Total edges in all workflows (128)

**Interactivity**: Click any metric card to expand (future enhancement)

**Color Coding**:
- Blue value = Primary metric
- Gray label = Context

**Responsive**:
- Desktop: 6 columns
- Tablet: 3 columns
- Mobile: 2 columns

---

### 2. Execution Timeline (Gantt Chart)

**Purpose**: Identify performance bottlenecks at node level

**Visualization Style**: Horizontal bar chart (Gantt-style)

**Components per Node**:
```
[Node Name] [████████░ Bar] [Duration] [Call Count]
```

**Examples**:
```
HoloLoom Query ████████░░░ 450ms (28)   ← Name | Bar | Duration | Calls
Memory Search  ██░░░░░░░░░ 120ms (45)   ← Short bar = fast
Synthesizer    ███░░░░░░░░ 280ms (22)   ← Long bar = slow
```

**Color Coding**:
- 🟦 Blue gradient: Normal performance
- 🟥 Red gradient: Bottleneck detected
- Width: Normalized to max latency (proportional display)

**Bottleneck Detection**:
- Threshold: >40% of average latency
- Marked with: ⚠️ warning icon
- Color: Red gradient instead of blue

**Metrics Shown**:
- Latency in milliseconds
- Call count (number of invocations)
- Status indicator (✓ or ⚠️)

**Hover Interaction**: Displays tooltip with:
- Full node name
- Average latency
- Call count
- Variance metric

**Legend**:
- Width = Average latency
- Number = Call count
- ⚠️ = Performance bottleneck

---

### 3. Confidence Trajectory Chart

**Purpose**: Track confidence trends and detect anomalies

**Chart Type**: SVG line chart with data points

**Visual Elements**:
1. **Grid Lines**: Subtle dashed lines at 0.2, 0.4, 0.6, 0.8 confidence levels
2. **Confidence Line**: Blue #3b82f6 curve connecting all data points
3. **Area Fill**: Blue-to-transparent gradient under the curve
4. **Data Points**: Color-coded circles at each measurement

**Data Point Colors**:
- 🟢 Green (#10b981): Confidence ≥0.8 (good)
- 🟠 Orange (#f59e0b): Confidence 0.7-0.8 (fair)
- 🔴 Red (#ef4444): Confidence <0.7 (poor)

**Statistics Cards** (bottom):
```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Average  │  │   Min    │  │   Max    │  │  Trend   │
│  0.86    │  │  0.72    │  │  0.95    │  │  📈 Up   │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

**Trend Analysis**:
- 📈 Up: Last value > first value
- 📉 Down: Last value < first value
- Automatic calculation from data

**Axes**:
- X-axis: Query index (0 to N)
- Y-axis: Confidence 0.0 to 1.0
- Labels: Numeric values at corners

**Responsive**: SVG scales to fit container (100% width)

---

### 4. Node Performance Summary

**Purpose**: Detailed per-agent performance metrics

**Table Structure**:
```
┌────────────────────┬──────────┬───────┬──────────┐
│ Node Name          │ Latency  │ Calls │ Variance │
├────────────────────┼──────────┼───────┼──────────┤
│ ✓ HoloLoom Query   │ 450ms    │  28   │  0.15    │
│ ✓ Memory Search    │ 120ms    │  45   │  0.08    │
│ ✓ Synthesizer      │ 280ms    │  22   │  0.12    │
│ ✓ Guardrails       │  85ms    │  35   │  0.05    │
│ ⚠️ Response Gen     │ 320ms    │  28   │  0.18    │
│ ✓ Conf. Scorer     │  95ms    │  38   │  0.06    │
│ ✓ Context Packer   │ 210ms    │  15   │  0.22    │
└────────────────────┴──────────┴───────┴──────────┘
```

**Summary Statistics** (above table):
```
Avg Latency: 214ms | Total Calls: 211 | Bottlenecks: 1
```

**Status Indicators**:
- 🟢 Green circle: Healthy (<300ms)
- 🟠 Orange circle: Warning (300-500ms)
- 🔴 Red circle: Critical (>500ms or bottleneck)

**Columns**:
1. **Node**: Name + status indicator
2. **Latency**: Duration in milliseconds (color-coded)
3. **Calls**: Number of executions
4. **Variance**: Standard deviation of latencies

**Sorting**: Static (not sortable in Wave 1.5)

**Accessibility**: All values readable, icons decorative

---

### 5. Cache Effectiveness Gauge

**Purpose**: Monitor cache hit rate and effectiveness

**Visual**: Radial gauge (circular progress)

**Gauge Components**:
```
        ╭─ Outer ring: Colored arc (0-360°)
        │       Angle = hit_rate * 3.6°
        │
        ├─ Inner ring: Rating label
        │       Text = "Excellent" | "Good" | "Fair" | "Poor" | "Critical"
        │
        └─ Center value: 72%
```

**Color Mapping**:
- 🟢 Green (#10b981): 80%+ (Excellent)
- 🔵 Blue (#3b82f6): 60-80% (Good)
- 🟠 Orange (#f59e0b): 40-60% (Fair)
- 🟠 Orange (#f97316): 20-40% (Poor)
- 🔴 Red (#ef4444): <20% (Critical)

**Metrics Display**:
```
┌──────────────┬──────────────┐
│ Cache Hits   │ Total Queries │
│     34       │      47       │
├──────────────┼──────────────┤
│ Time Saved   │ Avg Speedup  │
│   4.1s       │   15-30x     │
└──────────────┴──────────────┘
```

**Recommendation** (dynamic):
- **Excellent**: No action needed
- **Good**: Cache working well
- **Fair**: Monitor cache policy
- **Poor**: Review TTL settings
- **Critical**: Investigate configuration

**Time Saved Calculation**: `cache_hits * estimated_ms_per_cache (120ms)`

**Speedup Estimation**: `uncached_latency / cached_latency (~15-30x typical)`

---

### 6. Recent Workflow Executions

**Purpose**: Recent activity log with key metrics

**List Items** (newest first):
```
┌──────────────────────────────────────────────────────┐
│ Research Pipeline            ✓ completed   1250ms 92%│
│ 5 nodes • 4 edges • ID: wf-001                       │
├──────────────────────────────────────────────────────┤
│ Lead Scoring Workflow        ✓ completed    890ms 88%│
│ 4 nodes • 3 edges • ID: wf-002                       │
├──────────────────────────────────────────────────────┤
│ Content Analysis             ✗ failed      2100ms 45%│
│ 6 nodes • 5 edges • ID: wf-003                       │
└──────────────────────────────────────────────────────┘
```

**Fields per Execution**:
1. **Workflow Name**: Descriptive title
2. **Metadata**: Node count, edge count, ID
3. **Status Badge**: ✓ (completed) | ✗ (failed) | ⊙ (running)
4. **Latency**: Execution time or "running..."
5. **Confidence**: Percentage score

**Status Colors**:
- 🟢 Green: Completed successfully
- 🔴 Red: Failed execution
- 🔵 Blue: Currently running

**Display Limit**: Last 8 executions

**Sorting**: Chronological (newest first)

**Interactive**: Click to expand (future feature)

---

### 7. Anomaly Detection Panel

**Purpose**: Automated issue detection and alerting

**Anomaly Types Detected**:

#### Type 1: Sudden Confidence Drop ⬇️
```
⬇️ Sudden Confidence Drop
   Confidence: 0.91 → 0.72
   Query: "What are the tradeoffs?"
   Severity: HIGH
```
- Condition: Drop >0.2 in single step
- Severity: HIGH (red)

#### Type 2: Prolonged Low Confidence 📉
```
📉 Prolonged Low Confidence
   3 consecutive queries below 0.75
   Severity: MEDIUM
```
- Condition: <0.75 for 3+ queries
- Severity: MEDIUM (orange)

#### Type 3: High Variance 📊
```
📊 High Variance
   Variance: 0.18 in rolling window
   Severity: MEDIUM
```
- Condition: Std dev >0.15
- Severity: MEDIUM (orange)

#### Type 4: Cache Miss Cluster 💾
```
💾 Cache Miss Cluster
   4 misses in 2-minute window
   Severity: MEDIUM
```
- Condition: 3+ misses in window
- Severity: MEDIUM (orange)

**Empty State**:
```
✨ No anomalies detected - System operating normally
```

**Visual Design**:
- Colored left border (red = high, orange = medium)
- Type emoji + label
- Details text
- Monospace values where applicable

**Auto-Detection**: Continuous during operation

---

## Interaction Flows

### Flow 1: Checking System Health (30 seconds)

```
User opens dashboard
    ↓
Sees summary metrics at top
    ↓
Scans Cache Gauge (is cache healthy?)
    ↓
Checks Confidence Trajectory (any drops?)
    ↓
Reviews Node Performance (any bottlenecks?)
    ↓
Reads Anomaly Panel (anything to fix?)
    ↓
Decision: System OK or needs attention
```

### Flow 2: Debugging Performance Issue (2 minutes)

```
User notices slow workflow execution
    ↓
Opens Workflow Analytics dashboard
    ↓
Checks Execution Timeline
    ↓
Identifies bottleneck node (red bar)
    ↓
Looks at Node Performance table
    ↓
Sees latency, variance, call count
    ↓
Cross-references with Confidence Trajectory
    ↓
Checks Recent Executions for patterns
    ↓
Makes optimization decision
```

### Flow 3: Investigating Anomaly (5 minutes)

```
Dashboard shows anomaly alert
    ↓
User clicks anomaly card
    ↓
Reads anomaly type and details
    ↓
Checks Confidence Trajectory for context
    ↓
Reviews Recent Executions timing
    ↓
Checks Node Performance for correlation
    ↓
Investigates root cause
    ↓
Takes corrective action
```

## Color Palette Reference

### Primary Colors
- **Blue** #3b82f6 - Primary action, positive info
- **Red** #ef4444 - Critical, errors, high severity
- **Green** #10b981 - Success, healthy status
- **Orange** #f59e0b - Warning, caution, medium severity

### Backgrounds
- **Dark Navy** #1a1a2e - Main background
- **Dark Blue** #16213e - Gradient end
- **Dark Gray** #0f0f1e - Panel backgrounds

### Text
- **Light Gray** #e0e0e0 - Primary text
- **Medium Gray** #999 - Secondary text
- **Dark Gray** #666 - Tertiary text

### Borders
- **Light** rgba(255,255,255,0.1) - Primary borders
- **Medium** rgba(255,255,255,0.15) - Hover state
- **Dark** #333 - Strong separation

## Typography

### Font Stack
```css
-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
```

### Sizes
- **Title**: 24px (600 weight)
- **Panel Header**: 14px (600 weight)
- **Body**: 13px (400 weight)
- **Small**: 12px (400 weight)
- **Tiny**: 11px (400 weight)

### Spacing
- **Padding**: 20px (panels)
- **Gap**: 15-20px (between items)
- **Margin**: 12px (list items)
- **Line-height**: 1.4-1.6

## Responsive Breakpoints

### Desktop (1024px+)
- Grid: 2 columns
- All panels visible
- Full feature set

### Tablet (768-1024px)
- Grid: 1 column
- Stacked layout
- Optimized touch targets

### Mobile (<768px)
- Single column
- Smaller panels
- Simplified tables

---

**Created**: December 9, 2025
**Version**: 1.0
**Status**: Production Ready
