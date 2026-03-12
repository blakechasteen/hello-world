# Jenny Roadmap

**Updated**: 2026-03-11
**Status**: Foundation complete. Conversation Stages 1-3 done. Spatial WebSocket broadcasting. No consumer yet.

---

## Current State

### Foundation (MVP, Dec 2025)

Jenny is HoloLoom's generative UI runtime. "Disposable pixels, durable decisions."

**Panel system**: 13 types (TEXT, CODE, TABLE, GRAPH, CONFIDENCE, TIMELINE, MEMORY_MAP, METRIC, REASONING, SOURCES, ACTIONS, WHY, COMPARISON). Full lifecycle: NASCENT -> STABLE -> DISSOLVING -> ARCHIVED.

**Intelligence**: Thompson Sampling panel type learner (Beta priors, persistence to JSON). MRF-enhanced compilation (ELEGANCE, VERIFY strategies). LLM compiler with fallback chain (LLM -> MRF -> heuristics).

**Runtime**: JennyRuntime orchestrator, StreamingManager, SpecLedger provenance, 6 actions (PIN, DISMISS, WHY, EXPAND, COPY, EXPORT). 5 renderers: HTML, Terminal, JSON, React (props), AR (overlay spec).

**Tests**: 257 passing (239 unit + 18 integration).

### Conversation Awareness (Stages 1-3, Mar 2026)

| Stage | What | Key Files |
|-------|------|-----------|
| 1 | Static HTML panels -> Matrix per message | `matrix_renderer.py` |
| 2 | ConversationGraph -> evolving TABLE/COMPARISON/GRAPH panels | `conversation_graph.py`, `conversation_analyzer.py`, `conversation_strategy.py` |
| 3 | ConversationGraph -> positioned 3D overlays via WebSocket | `spatial_dispatcher.py`, `spatial_websocket.py` |

**Architecture after Stage 3**:

```
User message -> Ollama -> response
                |
        ConversationGraph.add_turn()
                |
        +-------+-------+
        |               |
   Stage 2:         Stage 3:
   Trajectory ->    Trajectory ->
   panel type ->    layout algo ->
   Matrix HTML      spatial scene ->
                    /ws/spatial/{room_id}
```

**Key types**: `Trajectory` enum (EXPLORING, COMPARING, DEEP_DIVE, DECIDING, WRAPPING_UP). `ConversationGraph` with 4 output methods: `to_table_spec()`, `to_graph_spec()`, `to_comparison_spec()`, `to_spatial_overlay()`.

**Feature flags**: `PROMPTLY_JENNY_CONVERSATION=true` (Stage 2), `PROMPTLY_JENNY_SPATIAL=true` (Stage 3).

---

## Built But Not Wired

Things that exist in the codebase but aren't connected to the live conversation pipeline:

| Capability | Where | Gap |
|-----------|-------|-----|
| Thompson Sampling for conversation panels | `jenny_mrf.py` | Strategy uses static `_TRAJECTORY_PANEL` dict, not learned selection |
| LLM-based panel compilation | `jenny_llm_compiler.py` | Exists but not called from promptly_chat |
| 4 unused layout algorithms | `knowledge_overlay.py` (CIRCULAR, CLUSTER, SPATIAL, MEMORY_PALACE) | Dispatcher maps to 4 of 8 |
| 7 overlay styles beyond CARD | `knowledge_overlay.py` (SPHERE, HOLOGRAM, GLOW, CONSTELLATION...) | All overlays default to CARD |
| 5 visibility modes | `knowledge_overlay.py` (PROXIMITY, GAZE, GESTURE, CONTEXT, QUERY) | All overlays always visible |
| 6 edge styles beyond LINE | `knowledge_overlay.py` (ARROW, BEAM, PARTICLES, DASHED, GRADIENT, PULSING) | All edges use default |
| Memory Palace rooms | `spatial/memory_palace.py` | Not connected to conversation sessions |
| Streaming panel updates | `jenny_streaming.py` (StreamingManager, DataSourceProtocol) | In-memory callbacks, no WS bridge |
| React components | `hololoom-ui/` (MemoryGraph, ChatInterface, PerformanceOverview) | Mock data, no WebSocket connection |
| Panel accessibility | `jenny_accessibility.py` | Exists but not integrated into renderers |
| WebSocket API client | `packages/api-client/` (useProgressSubscription, pattern subscriptions) | Built for `/ws/progress` which doesn't exist yet |

---

## Future Stages

Each stage independently valuable. Each has a clear "you can stop here" boundary.

### Stage 4: First Spatial Consumer

The spatial WebSocket broadcasts to nobody. Stage 4 gives it a consumer.

**Option A: Debug Inspector** (~100 LOC)
- Standalone HTML page with vanilla JS
- Connects to `/ws/spatial/{room_id}`, renders as 2D force graph (canvas)
- No build step, just `<script>` tag
- Serves from `GET /spatial/{room_id}/inspector`

**Option B: React Three.js Component** (~300 LOC)
- `SpatialScene.tsx` in `hololoom-ui/`
- `@react-three/fiber` + `@react-three/drei`
- Floating cards in 3D, edges as lines, orbit controls, click-to-select
- Connects via existing `useProgressSubscription` hook

**Recommendation**: A first (validates pipeline), then B (real UI).

### Stage 5: Learned Conversation Panels

Currently `_TRAJECTORY_PANEL` is a static dict. Make it adaptive.

- Wire `PanelTypeLearner` into `ConversationVisualizationStrategy`
- User signals: pin = success, dismiss = failure (already tracked by SpecLedger)
- Trajectory becomes a "query type" for the bandit: `Trajectory.EXPLORING` -> which panel type works best?
- Fallback to static dict if learner has < 10 observations per trajectory
- New Trajectory members: `TEACHING` (one-sided explanation), `BRAINSTORMING` (rapid-fire ideas)
- Feature flag: `PROMPTLY_JENNY_LEARNED=true`

### Stage 6: Rich Spatial Vocabulary

The spatial module has 8 overlay styles, 7 edge styles, 6 visibility modes. Stage 3 uses one of each.

**Overlay style mapping** (node metadata -> style):
- High importance (>0.8) -> GLOW (attention-drawing)
- Entities (@mentions, URLs) -> HOLOGRAM (data-like)
- Clusters of co-occurring topics -> CONSTELLATION
- Default -> CARD

**Edge style mapping** (relationship -> style):
- `co_occurs` -> LINE (neutral)
- `compared_with` -> GRADIENT (tension)
- `leads_to` -> ARROW (directionality)
- Recent (last 2 turns) -> PULSING

**Visibility modes** (trajectory-dependent):
- `exploring` -> all ALWAYS (overview)
- `deep_dive` -> center ALWAYS, periphery PROXIMITY (focus)
- `wrapping_up` -> all CONTEXT (fade out)

~50 lines of mapping logic in `SpatialSceneDispatcher._rebuild_scene()`.

### Stage 7: Memory Palace Sessions

Map conversation sessions to spatial rooms.

- Each room_id gets a `MemoryPalaceRoom`
- Trajectory determines room theme: `exploring` = open atrium, `deep_dive` = study, `comparing` = courtroom, `deciding` = war room
- Topics become placed objects, not floating cards
- Session boundaries (15-min gap) create new rooms connected by doorways
- Walking between rooms = browsing conversation history

**Depends on**: Stage 6 (rich vocabulary makes spatial immersion worth it).

### Stage 8: Cross-Pane Interaction

The v2.0 bridge. Jenny's conversation awareness meets the Cognitive UI vision.

- Click a spatial node -> reference it in chat ("Tell me more about {topic}")
- Reasoning step in chat -> highlight related nodes in spatial scene
- Trajectory change -> animate layout transition (FORCE_DIRECTED morphs to RADIAL)
- Panel lifecycle events -> spatial overlay lifecycle (NASCENT = fade-in)

**Requires**: Stage 4 (spatial consumer) + WeavingContext state bus from `docs/design/COGNITIVE_UI.md`. This bridges Jenny and the consciousness shell.

**Key constraint**: v2.0 Cognitive Interface requires v1.0.0 stable (frozen API). Cross-pane interaction is the first concrete deliverable of that track.

---

## Intelligence Track

Parallel to the spatial stages. Improve conversation understanding quality.

| Feature | What | Effort |
|---------|------|--------|
| LLM topic extraction | Replace regex with Ollama call. Keep regex as <100ms fallback. | 1 day |
| Sentiment trajectory | Detect emotional arc (curious -> frustrated -> satisfied). New metadata on ConversationNode. | 1 day |
| Cross-room patterns | Topics that recur across rooms -> "you always ask about X". Lightweight cross-room index. | 2 days |
| Comparison depth | When COMPARING, extract pro/con lists per topic. Richer COMPARISON panels. | 1 day |
| Decision detection | When DECIDING and user states a choice, mark the decision. New `DECIDED` trajectory? | 1 day |

---

## Metrics

| Metric | Current | Target | How |
|--------|---------|--------|-----|
| Topic extraction quality | Regex (4 patterns) | LLM-upgraded | A/B test on 100 conversations |
| Trajectory accuracy | Heuristic (weight distribution) | Learned + heuristic | User feedback |
| Panel usefulness | Unknown | 30%+ pin rate | SpecLedger analytics |
| Spatial scene latency | ~5ms rebuild | <10ms | Monitor as node count grows |
| WebSocket delivery | Full state per update | Delta patches if >100 overlays | Not needed until graph exceeds 50-node cap |

---

## Sources

- [Conversation graph](conversation_graph.py) -- Trajectory enum, 4 output methods
- [Spatial dispatcher](spatial_dispatcher.py) -- ConversationGraph -> KnowledgeOverlayManager bridge
- [Spatial WebSocket](../apps/server/spatial_websocket.py) -- Room-scoped scene broadcasting
- [Knowledge overlay manager](../spatial/knowledge_overlay.py) -- 8 layouts, 8 styles, 6 visibility modes
- [Cognitive UI design](../docs/design/COGNITIVE_UI.md) -- Three-pane consciousness shell (v2.0)
- [Main roadmap](../docs/ROADMAP.md) -- v1.0 -> v2.0 milestones
