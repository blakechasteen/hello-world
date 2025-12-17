# Debugging Tools

Comprehensive debugging features for developing and troubleshooting workflows.

## Overview

The Workflow Builder includes powerful debugging tools:
- Breakpoint system for pause-and-inspect
- Step-by-step execution
- Variable inspector for data flow
- Execution timeline visualization
- Performance profiling

## Breakpoints

### Setting Breakpoints

**Click Method**:
1. Click the left edge of any node
2. Red dot appears indicating breakpoint
3. Click again to remove

**Context Menu**:
1. Right-click a node
2. Select **"Toggle Breakpoint"**

**Keyboard**:
1. Select a node
2. Press `Ctrl+B`

### Breakpoint Types

| Type | Icon | Behavior |
|------|------|----------|
| **Standard** | 🔴 | Pause before node executes |
| **Conditional** | 🟡 | Pause only if condition is true |
| **Logpoint** | 🔵 | Log data without pausing |

### Conditional Breakpoints

1. Right-click breakpoint → **"Edit Condition"**
2. Enter JavaScript expression:

```javascript
// Pause if confidence is low
input.confidence < 0.5

// Pause if specific query type
input.query.includes("Thompson")

// Pause on error
input.error !== undefined
```

### Logpoints

Log data without stopping execution:

1. Right-click node → **"Add Logpoint"**
2. Enter log expression:

```javascript
// Log input
console.log("Query:", input.query)

// Log with formatting
console.log("Results:", JSON.stringify(input.results, null, 2))
```

## Step Execution

### Debug Controls

Located in the toolbar during debug mode:

| Button | Shortcut | Action |
|--------|----------|--------|
| ▶️ **Run** | F5 | Continue to next breakpoint |
| ⏸️ **Pause** | F6 | Pause current execution |
| ⏹️ **Stop** | Shift+F5 | Stop execution |
| ⏭️ **Step Over** | F10 | Execute current node, pause at next |
| ⏬ **Step Into** | F11 | Step into composite nodes |
| ⏫ **Step Out** | Shift+F11 | Step out of composite node |

### Execution Flow

```
[Start] → [Node A] → [🔴 Breakpoint]
                           ↓
                      [Paused]
                           ↓
                    Press F10 (Step Over)
                           ↓
                      [Node B] → [Continue...]
```

### Paused State

When paused at a breakpoint:
- Current node has yellow highlight
- Execution path shown with dotted line
- All upstream data available in inspector
- Can modify node configuration before continuing

## Variable Inspector

### Opening the Inspector

1. Press `F12` to open Debug Panel
2. Or: **View** → **Debug Panel**

### Panel Sections

```
┌─────────────────────────────────────────┐
│ Debug Inspector                      ×  │
├─────────────────────────────────────────┤
│ ▼ Input Data                            │
│   query: "What is Thompson Sampling?"   │
│   context: {limit: 10, mode: "fast"}    │
│                                         │
│ ▼ Output Data                           │
│   response: "Thompson Sampling is..."   │
│   confidence: 0.92                      │
│   sources: [{id: "src-1", ...}]         │
│                                         │
│ ▼ Node State                            │
│   status: "completed"                   │
│   duration_ms: 145                      │
│   cache_hit: true                       │
│                                         │
│ ▼ Trace                                 │
│   step: 3 of 5                          │
│   path: [query-1, process-1, output-1]  │
└─────────────────────────────────────────┘
```

### Inspecting Values

**Expand Objects**:
- Click ▶ to expand nested objects
- Click ▼ to collapse

**Copy Values**:
- Right-click value → **"Copy Value"**
- Copies JSON to clipboard

**Watch Expressions**:
1. Click **"+ Add Watch"**
2. Enter expression: `input.results.length`
3. Value updates in real-time

### Data Flow View

Click **"Data Flow"** tab to see:

```
[Query Node]
    ↓ {query: "...", limit: 10}
[Process Node]
    ↓ {results: [...], confidence: 0.92}
[Output Node]
    ↓ {response: "...", formatted: true}
```

## Execution Timeline

### Opening Timeline

1. After execution, click **"Timeline"** in Debug Panel
2. Or: Right-click canvas → **"Show Timeline"**

### Timeline Visualization

```
Time (ms)  0    50   100   150   200   250
           |    |    |     |     |     |
Query      ████████░░░░░░░░░░░░░░░░░░░░░
Process    ░░░░░░░░████████████░░░░░░░░░
Output     ░░░░░░░░░░░░░░░░░░░░████████░

█ = Executing  ░ = Waiting
```

### Timeline Features

| Feature | Description |
|---------|-------------|
| **Zoom** | Scroll to zoom in/out |
| **Pan** | Drag to scroll timeline |
| **Hover** | Shows node details |
| **Click** | Jumps to that execution point |
| **Markers** | Show breakpoints, errors, warnings |

## Performance Profiling

### Enabling Profiler

1. Press `P` to open Performance Panel
2. Or: **View** → **Performance Panel**

### Metrics Displayed

```
┌─────────────────────────────────────────┐
│ Performance                          ×  │
├─────────────────────────────────────────┤
│ FPS: 60          Frame Time: 16.2ms     │
│ Nodes Rendered: 45/120                  │
│ Connections: 67/180                     │
│                                         │
│ ▼ Node Execution Times                  │
│   Query-1:     145ms ████████████       │
│   Process-1:    89ms ███████            │
│   Output-1:     23ms ██                 │
│                                         │
│ ▼ Memory Usage                          │
│   Heap: 45MB / 100MB                    │
│   Canvas: 12MB                          │
│   WebSocket: 0.5MB                      │
└─────────────────────────────────────────┘
```

### Identifying Bottlenecks

Nodes with performance issues show:
- 🟡 Yellow border: >500ms execution
- 🔴 Red border: >2000ms execution
- ⚠️ Warning icon: Memory pressure

### Performance Tips

Based on profiler data:

| Issue | Solution |
|-------|----------|
| High frame time | Reduce visible nodes (zoom out or collapse groups) |
| Memory pressure | Clear unused workflows, reduce history |
| Slow nodes | Check node configuration, reduce complexity |
| WebSocket lag | Check network, reduce collaboration participants |

## Error Handling

### Error Indicators

| State | Visual | Meaning |
|-------|--------|---------|
| **Warning** | 🟡 Yellow border | Non-fatal issue |
| **Error** | 🔴 Red border + X | Execution failed |
| **Timeout** | ⏱️ Clock icon | Exceeded time limit |

### Error Details

Click error icon to see:

```
┌─────────────────────────────────────────┐
│ Error: Query-1                          │
├─────────────────────────────────────────┤
│ Type: TimeoutError                      │
│ Message: Query exceeded 30s timeout     │
│                                         │
│ Stack Trace:                            │
│   at QueryNode.execute (line 145)       │
│   at Executor.run (line 89)             │
│                                         │
│ Input that caused error:                │
│ { query: "...", complexity: "fused" }   │
│                                         │
│ [Retry] [Skip] [Edit Config]            │
└─────────────────────────────────────────┘
```

### Error Recovery

Options when error occurs:
- **Retry**: Re-execute with same input
- **Skip**: Continue to next node (if optional)
- **Edit Config**: Modify node before retrying
- **Abort**: Stop entire workflow

## Console Output

### Opening Console

1. Press `` Ctrl+` `` to open console
2. Or: **View** → **Console**

### Console Features

```
┌─────────────────────────────────────────┐
│ Console                              ×  │
├─────────────────────────────────────────┤
│ [Filter: All ▼] [Clear]                 │
│                                         │
│ INFO  Query-1 started                   │
│ DEBUG Query: "What is Thompson..."      │
│ INFO  Query-1 completed (145ms)         │
│ WARN  Process-1: Low confidence (0.65)  │
│ ERROR Output-1: Missing required field  │
│                                         │
│ > _                                     │
└─────────────────────────────────────────┘
```

### Log Levels

| Level | Color | Description |
|-------|-------|-------------|
| DEBUG | Gray | Detailed execution info |
| INFO | Blue | Normal execution flow |
| WARN | Yellow | Non-fatal issues |
| ERROR | Red | Execution failures |

### Console Commands

Type in console input:

```javascript
// Inspect node
inspect('query-1')

// Get execution history
history()

// Export logs
exportLogs('json')

// Clear and rerun
clear() && run()
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| F5 | Run/Continue |
| F6 | Pause |
| Shift+F5 | Stop |
| F10 | Step Over |
| F11 | Step Into |
| Shift+F11 | Step Out |
| Ctrl+B | Toggle Breakpoint |
| F12 | Toggle Debug Panel |
| P | Toggle Performance Panel |
| Ctrl+` | Toggle Console |

---

← [Real-Time Collaboration](collaboration.md) | [Voice Commands](voice-commands.md) →
