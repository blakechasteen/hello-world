# HoloLoom Roadmap

## Current State

**Version**: `1.0.0-alpha.1`

**Done**: lowercase `hololoom/` package, 13 core modules under `hololoom/core/`, apps extracted to `apps/`, `sys.meta_path` import redirection, `pyproject.toml` extras, 9-stage weaving cycle, Thompson Sampling, Matryoshka embeddings, spring physics (9.6x speedup), reflection buffer, 17 focused guides.

**Alpha gaps**: persistent backends exist but aren't on the default orchestrator path. Test suite has 20 stale collection errors. API surface works but isn't frozen.

---

## Milestones

Version gates with exit criteria. Each gate must close before the next opens.

### v1.0.0-beta.1 — Wired

- [ ] Wire Neo4j + Qdrant into orchestrator via backend factory
- [ ] Default: InMemory (zero config), auto-upgrade when persistent backends are available
- [ ] Docker Compose tested end-to-end
- [ ] Fix or remove 20 stale test collection errors
- [ ] Unit + integration suites green
- [ ] Update classifier: `"3 - Alpha"` → `"4 - Beta"`

### v1.0.0 — Stable

- [ ] Lock public API: `HoloLoom`, `Memory`, `experience()`, `recall()`, `reflect()`
- [ ] CI green: ruff, black, mypy, pytest
- [ ] API reference generated from docstrings
- [ ] All guides verified against actual code paths
- [ ] Update classifier: `"4 - Beta"` → `"5 - Production/Stable"`
- [ ] Semver enforced: no breaking changes in 1.x

### v2.0.0 — Cognitive Interface

The paradigm shift: the interface becomes the cognitive process. This is v2.0 because it changes what HoloLoom *is* — from a library you call into a system you think with.

- [ ] Three-pane consciousness shell (Memory Palace + Active Thinking + Awareness)
- [ ] WebSocket event stream from orchestrator
- [ ] Streaming reasoning timeline
- [ ] New public API surface for UI integration
- [ ] Requires: v1.0.0 stable

---

## Tracks

Independent feature streams. Each track ships as a minor release within its milestone. Tracks don't depend on each other unless noted — like optional modules, they're flat peers.

### Ingestion

Expand SpinningWheel from 4 adapters to 9+.

| Adapter | Input | Status |
|---------|-------|--------|
| TextSpinner | Plain text | Done |
| WebsiteSpinner | URLs, recursive crawling | Done |
| YouTubeSpinner | Video transcripts | Done |
| AudioSpinner | Audio files | Done |
| DocSpinner | PDF, DOCX, Markdown | Planned |
| ImageSpinner | Vision captioning, OCR | Planned |
| SlackSpinner | Channel history, threads | Planned |
| NotionSpinner | Databases, pages | Planned |
| GitHubSpinner | Repos, issues, PRs | Planned |

Each adapter follows the Spinner protocol. ~1-2 days each. No ordering constraints.

### Retrieval

HYPERSPACE mode — recursive gated multipass with Matryoshka importance gating.

```
Depth 0: threshold 0.6   broad exploration
Depth 1: threshold 0.75  focused
Depth 2: threshold 0.85  very focused
```

Graph traversal follows entity relationships. Score fusion across depths with deduplication. Natural funnel from broad to precise.

**Unlocks**: complex multi-hop reasoning, deep context exploration.

### Physics

Multi-physics optimization engine. Foundation complete (spring physics, 9.6x speedup).

| Phase | System | Target | LOC |
|-------|--------|--------|-----|
| 0 | Spring dynamics | Graph retrieval via Hooke's Law | 1,454 (done) |
| 1 | Gradient flow | Queries flow downhill to optimal targets | ~1,200 |
| 2 | Fluid dynamics | Context propagation via pressure gradients | ~1,100 |
| 3 | Thermodynamics | Exploration/exploitation via F = E - TS | ~700 |
| 4 | Wave mechanics | Pattern detection via interference | ~900 |
| 5 | Statistical mechanics | Emergent behavior via Boltzmann distribution | ~900 |
| 6 | Unified engine | All systems integrated in single timestep | ~1,500 |

Phases 1-2 target speed (projected 28.8x combined). Phases 3-5 target intelligence. Phase 6 composes them.

Ships as `pip install hololoom[physics]`. Each phase is independently useful. The unified engine (Phase 6) is the capstone.

### Autonomy

Self-directed reasoning and failure recovery.

- **Goal hierarchy** — complex goals decompose into subtasks tracked in Spacetime
- **Episodic-semantic consolidation** — successful episodes commit to permanent knowledge via reflection buffer
- **Self-critique** — pre-execution validation with confidence thresholds, rollback on low confidence
- **Failure recovery** — Top-K fallback strategies, adaptive re-ranking after tool failures
- **Context budgeting** — token budget management, priority-based pruning, dynamic mode switching (FUSED → FAST → BARE under pressure)

**Unlocks**: long-running autonomous sessions, robust multi-step tasks.

### Observability

Explainability and monitoring for trust.

- Decision explanations ("why this tool?")
- Regret bounds and convergence proofs for Thompson Sampling
- Feature importance and counterfactual analysis
- Real-time monitoring dashboard
- Anomaly detection on system health metrics

**Unlocks**: debuggability, user trust, production confidence.

### Interface

Cognitive UI — the v2.0 track. Compose existing components into a unified consciousness shell.

**Jenny Conversation Stages** (complete, Mar 2026):

| Stage | What | Status |
|-------|------|--------|
| 1 | Static HTML panels → Matrix per message | Done |
| 2 | ConversationGraph → evolving TABLE/COMPARISON/GRAPH panels with trajectory detection | Done |
| 3 | ConversationGraph → positioned 3D overlays via WebSocket (`/ws/spatial/{room_id}`) | Done |

See [Jenny Roadmap](../hololoom/visualization/JENNY_ROADMAP.md) for Stages 4-8 and the intelligence track.

**React frontend** (`hololoom-ui/` — Next.js 14 + React 18 + TypeScript + Tailwind monorepo):

| Component | Location | Status |
|-----------|----------|--------|
| MemoryGraph | `components/memory/MemoryGraph.tsx` | Canvas force-directed, zoom/pan/select |
| ChatInterface | `components/ChatInterface.tsx` | 4 reasoning modes, confidence badges |
| PerformanceOverview | `components/analytics/PerformanceOverview.tsx` | Sparklines, trend indicators |
| API client + WebSocket | `packages/api-client/` | Pattern subscriptions, auto-reconnect |
| Design system | `packages/design-system/` | 12 components, 3 themes |
| WeavingIndicator | Design system | 9-stage visual progress |

**Not built** (4 phases, see `docs/design/COGNITIVE_UI.md`):
1. Unified three-pane shell (Memory Palace + Active Thinking + Awareness)
2. Live WebSocket wiring (pages currently use mock data)
3. Cross-pane interaction (node click → chat, reasoning step → graph highlight)
4. Enriched backend events (stage details, bandit candidates, motif detection)

Design principles: no modals, no spinners, keyboard-first. Panes communicate through typed reducer context, never import each other.

**Requires**: v1.0.0 stable (needs frozen API to build on).

---

## Architecture Constraints

These hold across all tracks and versions:

- **Core stays small** — ~13% of codebase. Test: "does the weaving cycle break without this?" Yes = core.
- **Flat peers** — optional modules don't nest under a wrapper. Categories live in `pyproject.toml` extras and docs.
- **No cross-imports** — each optional module depends only on core.
- **Graceful degradation** — HYBRID → INMEMORY. Optional deps warn, never crash.
- **Protocol-based** — swap backends, policies, retrievers without touching orchestrator.
- **Semver** — no breaking changes within a major version. New tracks extend, never replace.

---

## Adding a Track

This roadmap is extensible. To propose a new track:

1. Define what it does in one sentence
2. Show what exists today (status table or bullet list)
3. State what it unlocks (why it matters)
4. Note dependencies on other tracks (if any — prefer none)
5. Identify the `pyproject.toml` extra it ships under

If it passes the flat peer test (no cross-imports, independently installable), it belongs here.

---

## Sources

Synthesized from:
- [BUILD_PLAN.md](../BUILD_PLAN.md) — restructuring history (complete)
- [MODULE_TAXONOMY.md](../MODULE_TAXONOMY.md) — layer 0-4 classification
- [Physics integration plan](../infra/scripts/dev/generate_physics_roadmap.py) — 6-phase physics design
- [Feature roadmap](archive/stale-architecture/FEATURE_ROADMAP.md) — phases 5-9 (archived)
- [UX vision](roadmaps/REVOLUTIONARY_UX_ROADMAP.md) — cognitive interface design
