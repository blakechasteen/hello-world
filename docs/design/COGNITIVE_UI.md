# Cognitive UI Implementation

The plan to compose existing `hololoom-ui/` components into a unified consciousness interface. Not a rewrite — a composition. Each phase is independently shippable.

---

## What Exists

The frontend monorepo (`hololoom-ui/`) is ~30-40% of the way to a cognitive UI:

| Component | File | What it does |
|-----------|------|-------------|
| MemoryGraph | `apps/web/src/components/memory/MemoryGraph.tsx` | Canvas force-directed graph, zoom/pan, node selection, edge types, activation glow |
| ChatInterface | `apps/web/src/components/ChatInterface.tsx` | Messages, 4 reasoning modes, confidence badges, source panel |
| PerformanceOverview | `apps/web/src/components/analytics/PerformanceOverview.tsx` | 4 metric cards with sparklines and trend indicators |
| WeavingIndicator | `packages/design-system/` | 9-stage progress visualization |
| API client | `packages/api-client/src/client.ts` | WebSocket with pattern subscriptions, auto-reconnect, HTTP polling fallback |
| Hooks | `packages/api-client/src/hooks.ts` | useConversation, useQuery, useMemoryGraph, useStats, useProgressSubscription |
| Types | `packages/api-client/src/types.ts` | ProgressEvent, MemoryGraph, QueryResponse, SystemStats |

**The gap**: these components live on separate tab-based pages (`/chat`, `/memory`, `/analytics`) and use mock data. They need to be composed side-by-side and wired to the live backend.

---

## Phase 1: Unified Shell

New `/consciousness` route. Three-pane layout with shared state. Existing pages unchanged.

### Layout

```
grid-template-columns: 1fr 2fr 300px

+------------------+---------------------+------------------+
|  Memory Palace   |  Active Thinking    |  Awareness       |
|  (MemoryGraph)   |  (ChatInterface)    |  (CompactMetrics)|
|                  |                     |                  |
|  Force-directed  |  Messages + input   |  Latency gauge   |
|  graph with      |  Reasoning mode     |  Cache hit rate   |
|  node selection  |  selector           |  Confidence trend |
|                  |  Stage indicator    |  Tool used        |
+------------------+---------------------+------------------+
```

### State bus: WeavingContext

Components dispatch typed actions to a shared reducer. Panes never import each other.

```typescript
type WeavingAction =
  | { type: 'QUERY_START'; jobId: string; query: string }
  | { type: 'STAGE_UPDATE'; stage: number; stageName: string; progress: number }
  | { type: 'STAGE_COMPLETE'; stage: number; durationMs: number }
  | { type: 'QUERY_COMPLETE'; confidence: number; toolUsed: string; memoryIds: string[] }
  | { type: 'SELECT_NODE'; nodeId: string | null }
  | { type: 'HIGHLIGHT_NODES'; nodeIds: string[] }
  | { type: 'REFERENCE_NODE'; nodeId: string; content: string }
  | { type: 'TOGGLE_PANE'; pane: 'memory' | 'chat' | 'metrics' }
```

### New files

| File | Purpose |
|------|---------|
| `contexts/WeavingContext.tsx` | Reducer + provider (the state bus) |
| `app/consciousness/page.tsx` | Route: wraps CognitiveShell in WeavingContext.Provider |
| `components/consciousness/CognitiveShell.tsx` | CSS Grid layout, pane collapse/expand, keyboard shortcuts (Cmd+1/2/3) |
| `components/consciousness/CompactMetrics.tsx` | Condensed metrics panel (4 gauges + confidence + tool badge) |

### Modified files

| File | Change |
|------|--------|
| `Navigation.tsx` | Add Consciousness nav item |
| `MemoryGraph.tsx` | Accept optional `highlightedNodeIds: Set<string>` prop |
| `ChatInterface.tsx` | Accept optional `pendingReference: {nodeId, content}` prop |

---

## Phase 2: Live Wiring

Connect WebSocket progress to all three panes simultaneously.

**When a query runs:**
1. Chat pane shows 9-stage WeavingIndicator stepping through stages
2. Graph pane highlights activated memory nodes as Memory Retrieval completes
3. Metrics pane updates latency, confidence, tool used in real-time
4. StageTimeline bar above all panes shows global progress

### New files

| File | Purpose |
|------|---------|
| `hooks/useWeavingProgress.ts` | Bridges `useProgressSubscription` → WeavingContext dispatch |
| `hooks/useMemoryHighlight.ts` | Reads `activatedMemoryIds` from context → passes to MemoryGraph |
| `components/consciousness/StageTimeline.tsx` | Horizontal 9-stage progress bar (persistent above panes) |

### Backend (if `/ws/progress` doesn't exist)

The API client assumes `/ws/progress` exists. The backend currently only has `/ws/safety`. If needed:

- Create `websocket_progress.py` following the existing `safety_websocket.py` pattern (~100 lines)
- Wire `stage_tracking_callback` from `unified_server.py` to broadcast `JOB_PROGRESS` messages
- Message format: `{type, jobId, stage, stageName, progress, status, elapsedMs}`
- Pattern subscriptions: `job:{id}`, heartbeat every 30s, 100-message buffer

### Data flow

```
User types query in chat pane
  → ChatInterface dispatches QUERY_START to WeavingContext
  → useWeavingProgress subscribes to job:{id} via WebSocket
  → Backend orchestrator runs 9 stages, emits stage_callback per stage
  → WebSocket pushes JOB_PROGRESS messages
  → useWeavingProgress dispatches STAGE_UPDATE / STAGE_COMPLETE
  → StageTimeline reads from context, animates progress
  → CompactMetrics reads from context, updates gauges
  → On QUERY_COMPLETE, useMemoryHighlight passes activatedMemoryIds to graph
  → MemoryGraph renders activation glow on those nodes
```

---

## Phase 3: Cross-Pane Interaction

Bidirectional data flow between panes.

### Interactions

| Trigger | Action | Mechanism |
|---------|--------|-----------|
| Click memory node | `@node` reference appears in chat input | `REFERENCE_NODE` dispatch |
| Hover reasoning step | Highlights related nodes in graph | `HIGHLIGHT_NODES` dispatch |
| Stage completes | Confidence gauge animates to new value | Read from context |
| Query activates subgraph | Graph auto-pans to center activated nodes | `PAN_TO_NODE` dispatch |
| Click stage in timeline | Expands stage details in chat area | `SELECT_STAGE` dispatch |

### New files

| File | Purpose |
|------|---------|
| `components/consciousness/NodeReference.tsx` | `@[node-label]` pill in chat input, click to remove |
| `components/consciousness/ConfidenceTrajectoryMini.tsx` | Per-query sparkline showing confidence across 9 stages |
| `components/consciousness/ReasoningTimeline.tsx` | Rich stage-by-stage view (replaces simple WeavingIndicator during loading) |

### Modified files

| File | Change |
|------|--------|
| `MemoryGraph.tsx` | Accept `panToNodeId` prop, animate transform to center on node |
| `ChatInterface.tsx` | Render NodeReference pills, dispatch HIGHLIGHT_NODES on reasoning step hover |
| `CompactMetrics.tsx` | Add ConfidenceTrajectoryMini, animate gauge transitions |

---

## Phase 4: Enriched Events

Backend emits richer stage details. Frontend renders ms-precision reasoning timeline.

### Backend changes

Extend `stage_callback` signature (backward-compatible):

```python
# Before
stage_callback(stage_id: int, stage_name: str, duration_ms: float)

# After
stage_callback(stage_id: int, stage_name: str, duration_ms: float, details: dict | None = None)
```

Per-stage details:

| Stage | Details |
|-------|---------|
| 1 Loom Command | `{pattern_card: "FUSED"\|"FAST"\|"BARE"}` |
| 3 Yarn Graph | `{motifs_detected: [...], threads_activated: int}` |
| 6 Memory Retrieval | `{memories: [{id, preview, relevance}, ...]}` |
| 7 Convergence | `{candidates: [{tool, confidence, alpha, beta}, ...], selected: str}` |
| 8 Tool Execution | `{tool: str, output_preview: str}` |

### Frontend additions

| File | Purpose |
|------|---------|
| `components/consciousness/BanditVisualization.tsx` | Thompson Sampling prior bar chart (shows candidate tools with confidence intervals) |
| `components/consciousness/MotifBadge.tsx` | Colored badge for detected motifs (temporal_loop, entity_cluster, etc.) |

Extend `api-client/types.ts` with `StageDetails` discriminated union type. `ReasoningTimeline` renders enriched details when available, falls back to basic stage info when not.

---

## Architecture Constraints

These mirror HoloLoom's core philosophy:

- **No cross-imports between panes** — all communication flows through WeavingContext
- **Existing pages untouched** — /chat, /memory, /analytics remain exactly as they are
- **Composable** — each consciousness component works standalone (StageTimeline, BanditVisualization, etc.)
- **Graceful degradation** — no WebSocket → polling fallback. No enriched events → basic stage info. Pane collapsed → stops subscribing
- **Protocol-based** — adding a new pane means: read context, dispatch actions. No wiring changes needed

---

## Component Summary

| Phase | New Components | Modified |
|-------|---------------|----------|
| 1 | WeavingContext, CognitiveShell, CompactMetrics | Navigation, MemoryGraph, ChatInterface |
| 2 | useWeavingProgress, useMemoryHighlight, StageTimeline | CognitiveShell, ChatInterface, MemoryGraph, CompactMetrics |
| 3 | NodeReference, ConfidenceTrajectoryMini, ReasoningTimeline | MemoryGraph, ChatInterface, CompactMetrics |
| 4 | BanditVisualization, MotifBadge | weaving_orchestrator.py, api-client/types.ts, ReasoningTimeline |
