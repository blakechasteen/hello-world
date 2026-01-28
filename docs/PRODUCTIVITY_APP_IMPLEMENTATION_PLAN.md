# HoloLoom Productivity App: Implementation Plan

**Date**: 2026-01-28
**Goal**: Build a team productivity and learning app on top of HoloLoom's existing infrastructure
**Codename**: HoloTeam

---

## Executive Summary

HoloLoom already contains ~80,000 lines of production-ready backend code across collaboration, memory, learning, planning, visualization, and API systems. This plan builds a thin integration layer and user-facing application on top of that foundation, rather than rebuilding from scratch.

**Core value proposition**: Teams learn together, retain knowledge collectively, and make better decisions over time — powered by HoloLoom's adaptive memory and Thompson Sampling learning.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    HoloTeam Frontend                     │
│         (Web App — React + TypeScript)                   │
│                                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Workspace│ │ Knowledge│ │  Task    │ │  Insights  │  │
│  │  View    │ │  Base    │ │  Board   │ │  Dashboard │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP + WebSocket
┌────────────────────────┴────────────────────────────────┐
│                  HoloTeam API Layer                       │
│           (FastAPI — new thin integration)                │
│                                                           │
│  Extends HoloLoom/server/agentic_api.py                  │
│  Adds: /teams, /workspaces, /tasks, /insights            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│              Existing HoloLoom Backend                    │
│                                                           │
│  Collaboration │ Memory │ Learning │ Planning │ Viz      │
│  (sessions,    │ (graph,│ (Thompson│ (HTN,    │ (Jenny,  │
│   presence,    │  RAG,  │  Sampling│  POMDP,  │  Tufte,  │
│   sync, RBAC)  │  cache)│  priors) │  agents) │  React)  │
└─────────────────────────────────────────────────────────┘
```

---

## Phases

### Phase 1: Foundation (Core API + Auth)

**Goal**: Team creation, user management, and authenticated API access.

**What exists**:
- `UserManager` — user CRUD, login/logout, session tokens (`collaboration/user_manager.py`)
- `Team` — team creation, member roles (`collaboration/user_manager.py`)
- `AccessController` — RBAC with 5 permission levels (`collaboration/access_control.py`)
- `FastAPI server` — health, query, stats endpoints (`server/agentic_api.py`)

**What to build**:

| Task | Description | Effort |
|------|-------------|--------|
| `holoteam/api/auth.py` | JWT auth middleware wrapping `UserManager.login()` | Small |
| `holoteam/api/teams.py` | REST endpoints for teams: CRUD, invite, roles | Small |
| `holoteam/api/users.py` | REST endpoints for user profile, preferences | Small |
| `holoteam/app.py` | Main FastAPI app mounting existing agentic_api + new routes | Small |

**Integration code** (illustrative):

```python
# holoteam/api/teams.py
from HoloLoom.collaboration import UserManager, UserRole

user_manager = UserManager(storage_path=".holoteam/users.json")

@router.post("/teams")
async def create_team(name: str, user=Depends(get_current_user)):
    team = user_manager.create_team(name, user.user_id)
    return team.to_dict()

@router.post("/teams/{team_id}/members")
async def invite_member(team_id: str, username: str, role: str, user=Depends(get_current_user)):
    target = user_manager.get_user_by_username(username)
    user_manager.add_team_member(team_id, target.user_id, UserRole[role.upper()])
    return {"status": "invited"}
```

**Deliverable**: Authenticated API with team/user management. Teams can be created, members invited with roles.

---

### Phase 2: Shared Knowledge Base

**Goal**: Teams can collectively build, query, and share a knowledge base.

**What exists**:
- `UnifiedMemory` — store, recall, navigate, discover patterns (`memory/unified.py`)
- `SimpleRAG` — zero-config query with 4 reasoning modes (`rag/simple_rag.py`)
- `KnowledgeSharing` — export to JSON/CSV/Markdown/RDF, share with scopes (`collaboration/knowledge_sharing.py`)
- `MemoryConductor` — intelligent multi-system routing (`memory_symphony/`)
- `SpinningWheel` — 47 input adapters for any format (`spinningWheel/`)

**What to build**:

| Task | Description | Effort |
|------|-------------|--------|
| `holoteam/api/knowledge.py` | Endpoints: ingest, query, browse, export, share | Medium |
| `holoteam/services/team_memory.py` | Per-team memory isolation using team-scoped backends | Medium |
| `holoteam/services/ingestion.py` | File upload → SpinningWheel adapter routing | Small |

**Integration code** (illustrative):

```python
# holoteam/services/team_memory.py
from HoloLoom.memory.unified import UnifiedMemory, RecallStrategy
from HoloLoom.rag import SimpleRAG

class TeamKnowledgeBase:
    """One knowledge base per team, backed by UnifiedMemory + RAG."""

    def __init__(self, team_id: str):
        self.team_id = team_id
        self.memory = UnifiedMemory(
            user_id=team_id,
            enable_conductor=True
        )
        self.rag = SimpleRAG()

    async def ingest(self, content: str, user_id: str):
        """Team member adds knowledge."""
        mem = await self.memory.store(content, context={
            "team_id": self.team_id,
            "contributed_by": user_id,
            "timestamp": datetime.now().isoformat()
        })
        await self.rag.ingest(content)
        return mem

    async def query(self, question: str, mode: str = "direct"):
        """Ask the team knowledge base."""
        return await self.rag.query(question, mode=mode)

    async def browse(self, strategy: str = "balanced", limit: int = 20):
        """Browse recent knowledge."""
        return await self.memory.recall("*", strategy=RecallStrategy[strategy.upper()], limit=limit)
```

**API endpoints**:
- `POST /teams/{id}/knowledge` — Ingest content (text, file upload)
- `GET /teams/{id}/knowledge/query?q=...&mode=verify` — Query with reasoning mode
- `GET /teams/{id}/knowledge/browse` — Browse knowledge graph
- `POST /teams/{id}/knowledge/export` — Export to JSON/CSV/Markdown
- `POST /teams/{id}/knowledge/share` — Create shareable link

**Deliverable**: Teams have a shared knowledge base. Members ingest documents, ask questions with RAG, browse the knowledge graph, and export/share knowledge.

---

### Phase 3: Real-Time Collaboration

**Goal**: Team members work together in real-time sessions with presence and sync.

**What exists**:
- `SessionManager` — 5 session types, event system (`collaboration/session.py`)
- `PresenceManager` — cursor tracking, typing indicators (`collaboration/presence.py`)
- `StateSynchronizer` — CRDT conflict resolution (`collaboration/sync.py`)
- `VoiceManager` — WebRTC voice/video rooms (`collaboration/voice.py`)
- `AttributionManager` — 14 contribution types with scoring (`collaboration/attribution.py`)

**What to build**:

| Task | Description | Effort |
|------|-------------|--------|
| `holoteam/api/sessions.py` | REST + WebSocket endpoints for sessions | Medium |
| `holoteam/api/ws.py` | WebSocket hub: presence, sync, notifications | Medium |
| `holoteam/services/attribution.py` | Wire attribution tracking to team activity | Small |

**Integration code** (illustrative):

```python
# holoteam/api/sessions.py
from HoloLoom.collaboration import create_session_manager, SessionType

session_manager = create_session_manager()

@router.post("/teams/{team_id}/sessions")
async def create_session(team_id: str, name: str, session_type: str, user=Depends(get_current_user)):
    session = await session_manager.create_session(
        name=name,
        owner_id=user.user_id,
        owner_name=user.display_name,
        session_type=SessionType[session_type.upper()],
        tags=[team_id]
    )
    return session.to_dict()

@router.websocket("/teams/{team_id}/sessions/{session_id}/ws")
async def session_ws(websocket: WebSocket, team_id: str, session_id: str):
    await websocket.accept()
    session = await session_manager.get_session(session_id)

    # Forward session events to WebSocket
    session.on("participant_joined", lambda e, d: websocket.send_json({"event": e, "data": d}))
    session.on("state_changed", lambda e, d: websocket.send_json({"event": e, "data": d}))

    while True:
        data = await websocket.receive_json()
        if data["type"] == "presence_update":
            presence_manager.update(session_id, user.user_id, data["cursor"])
        elif data["type"] == "state_change":
            synchronizer.apply_operation(session_id, Operation.from_dict(data["op"]))
```

**Session types** (from existing enum):
- `KNOWLEDGE_BASE` — collaborative knowledge curation
- `WHITEBOARD` — brainstorming / ideation
- `RESEARCH` — multi-query exploration
- `REVIEW` — knowledge review and QA
- `PRESENTATION` — read-only knowledge sharing

**Deliverable**: Teams can open live sessions, see each other's cursors, sync state in real-time, and have contributions attributed.

---

### Phase 4: Task Planning & Tracking

**Goal**: Teams decompose goals into tasks, plan execution, and track progress.

**What exists**:
- `Tapestry` / `LoomKeeper` — goal decomposition, thread lifecycle, verification (`tapestry/`)
- `HierarchicalPlanner` — HTN planning with causal reasoning (`planning/planner.py`)
- `ActionItemTracker` — persistent task tracking (`recursive/`)
- `ChainPatterns` — 17 pre-built workflow patterns (`chaining/`)

**What to build**:

| Task | Description | Effort |
|------|-------------|--------|
| `holoteam/api/tasks.py` | REST endpoints for goals, tasks, status updates | Medium |
| `holoteam/services/team_planner.py` | Wire Tapestry + Planner for team goal decomposition | Medium |
| `holoteam/services/workflows.py` | Pre-built team workflow templates from ChainPatterns | Small |

**Integration code** (illustrative):

```python
# holoteam/services/team_planner.py
from HoloLoom.tapestry import LoomKeeper
from HoloLoom.tapestry.protocol import Tapestry, ThreadStatus

class TeamPlanner:
    def __init__(self, team_id: str):
        self.keeper = LoomKeeper(path=f".holoteam/teams/{team_id}/tapestry.json")

    async def create_goal(self, goal: str, tasks: list[str], dependencies: dict = None):
        """Decompose a team goal into tracked tasks."""
        tapestry = await self.keeper.start(goal=goal, threads=tasks)
        return tapestry

    async def get_progress(self) -> dict:
        """Get current goal progress."""
        result = await self.keeper.resume()
        if not result:
            return {"status": "no_active_goal"}
        tapestry, next_thread = result
        summary = tapestry.get_status_summary()
        return {
            "goal": tapestry.goal,
            "total": len(tapestry.threads),
            "completed": summary.get("woven", 0),
            "in_progress": summary.get("weaving", 0),
            "blocked": summary.get("tangled", 0),
            "next_task": next_thread.description if next_thread else None
        }

    async def complete_task(self, thread_id: str, commit_hash: str = None):
        """Mark a task as done."""
        result = await self.keeper.resume()
        tapestry, _ = result
        tapestry = tapestry.update_thread(thread_id, ThreadStatus.WOVEN, commit_hash=commit_hash)
        await self.keeper.backend.save(tapestry)
```

**API endpoints**:
- `POST /teams/{id}/goals` — Create goal with task decomposition
- `GET /teams/{id}/goals/current` — Current goal progress
- `PATCH /teams/{id}/tasks/{task_id}` — Update task status
- `GET /teams/{id}/tasks` — List all tasks with filters
- `POST /teams/{id}/goals/plan` — Auto-plan using HTN planner

**Deliverable**: Teams set goals, break them into tasks with dependencies, track progress, and get auto-planning suggestions.

---

### Phase 5: Insights & Learning Dashboard

**Goal**: Teams see how their knowledge grows, what's working, and get adaptive recommendations.

**What exists**:
- `JennyRuntime` — multi-target panel rendering with Thompson Sampling learning (`visualization/jenny_runtime.py`)
- `FullLearningEngine` — 7 learning systems with statistics (`recursive/`)
- Tufte visualizations — confidence trajectories, waterfall charts, knowledge graphs, cache gauges (`visualization/`)
- `PerformanceReporter` — daily/weekly reports with Prometheus metrics (`routing/learning/`)
- `AttributionManager` — who contributed what with quality scores (`collaboration/attribution.py`)

**What to build**:

| Task | Description | Effort |
|------|-------------|--------|
| `holoteam/api/insights.py` | Endpoints for team analytics and dashboards | Medium |
| `holoteam/services/team_analytics.py` | Aggregate per-team learning metrics | Medium |
| `holoteam/services/recommendations.py` | Thompson Sampling recommendations for team | Small |

**Dashboard panels** (rendered by Jenny → React props):

| Panel | Data Source | Shows |
|-------|------------|-------|
| **Knowledge Growth** | `UnifiedMemory.health_check()` | Nodes/edges over time |
| **Query Patterns** | `detect_temporal_patterns()` | What the team asks about most |
| **Confidence Trajectory** | `render_confidence_trajectory()` | Answer quality over time |
| **Top Contributors** | `AttributionManager` | Who added/reviewed most knowledge |
| **Knowledge Graph** | `render_knowledge_graph_from_kg()` | Visual map of team knowledge |
| **Learning Progress** | `get_learning_statistics()` | Thompson Sampling adaptation curve |
| **Hot Topics** | `HotPatternTracker` | Most accessed knowledge recently |
| **Task Velocity** | `Tapestry.get_status_summary()` | Tasks completed per week |

**Integration code** (illustrative):

```python
# holoteam/services/team_analytics.py
from HoloLoom.visualization.confidence_trajectory import render_confidence_trajectory
from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_kg

class TeamAnalytics:
    def __init__(self, team_kb: TeamKnowledgeBase):
        self.kb = team_kb

    async def get_knowledge_stats(self):
        health = self.kb.memory.health_check()
        patterns = self.kb.memory.detect_temporal_patterns(min_occurrences=2, time_window_days=7)
        return {
            "total_memories": health.get("components", {}).get("graph", {}).get("nodes", 0),
            "total_connections": health.get("components", {}).get("graph", {}).get("edges", 0),
            "recurring_topics": [p["description"] for p in patterns if p["pattern_type"] == "recurring_topic"],
            "activity_bursts": [p for p in patterns if p["pattern_type"] == "temporal_cluster"],
        }

    async def render_knowledge_map(self):
        """Render team knowledge graph as React-compatible props."""
        graph = self.kb.memory._backend.graph  # Access underlying KG
        return render_knowledge_graph_from_kg(graph, title="Team Knowledge Map")
```

**Deliverable**: Team dashboard showing knowledge growth, query patterns, contributor leaderboard, confidence trends, and adaptive recommendations.

---

### Phase 6: Frontend (Web App)

**Goal**: React/TypeScript web application consuming the API.

**What exists**:
- Jenny `ReactRenderer` outputs TypeScript-friendly JSON props
- `workflow_builder.html` — drag-and-drop workflow builder (vanilla JS)
- VS Code `squad/` extension — TypeScript HoloLoom bridge

**What to build**:

| Component | Description | Effort |
|-----------|-------------|--------|
| `ui/src/pages/Login.tsx` | Auth page | Small |
| `ui/src/pages/TeamDashboard.tsx` | Main workspace: KB + sessions + tasks | Large |
| `ui/src/pages/KnowledgeBase.tsx` | Query, browse, ingest knowledge | Large |
| `ui/src/pages/TaskBoard.tsx` | Kanban-style task board | Medium |
| `ui/src/pages/InsightsDashboard.tsx` | Analytics panels (Jenny React props) | Medium |
| `ui/src/components/SessionBar.tsx` | Presence indicators, active session | Medium |
| `ui/src/components/QueryInput.tsx` | RAG query with mode selector | Small |
| `ui/src/components/KnowledgeGraph.tsx` | Interactive graph visualization | Medium |
| `ui/src/hooks/useWebSocket.ts` | WebSocket for presence/sync/notifications | Medium |
| `ui/src/services/api.ts` | API client wrapping all endpoints | Small |

**Tech stack**: React 18, TypeScript, Tailwind CSS, React Query, D3.js (for graphs)

**Deliverable**: Functional web app where teams manage knowledge, collaborate, track tasks, and view insights.

---

## Integration Dependency Map

```
Phase 1 (Auth)
  └── Phase 2 (Knowledge Base)
        ├── Phase 3 (Collaboration)
        │     └── Phase 6 (Frontend — sessions/presence)
        ├── Phase 4 (Tasks)
        │     └── Phase 6 (Frontend — task board)
        └── Phase 5 (Insights)
              └── Phase 6 (Frontend — dashboards)
```

Phases 3, 4, and 5 can run in parallel after Phase 2. Phase 6 builds incrementally on each.

---

## File Structure

```
holoteam/
├── app.py                          # FastAPI main, mounts all routers
├── config.py                       # HoloTeam-specific configuration
├── api/
│   ├── auth.py                     # JWT auth middleware
│   ├── users.py                    # User profile endpoints
│   ├── teams.py                    # Team CRUD + invite
│   ├── knowledge.py                # KB ingest/query/export/share
│   ├── sessions.py                 # Collaboration sessions
│   ├── tasks.py                    # Goals, tasks, progress
│   ├── insights.py                 # Analytics dashboards
│   └── ws.py                       # WebSocket hub
├── services/
│   ├── team_memory.py              # Per-team TeamKnowledgeBase
│   ├── team_planner.py             # Tapestry-backed goal tracking
│   ├── team_analytics.py           # Aggregated team metrics
│   ├── ingestion.py                # File → SpinningWheel routing
│   ├── attribution.py              # Contribution tracking
│   ├── recommendations.py          # Thompson Sampling suggestions
│   └── workflows.py                # Pre-built chain templates
└── ui/                             # React frontend (Phase 6)
    ├── src/
    │   ├── pages/
    │   ├── components/
    │   ├── hooks/
    │   └── services/
    └── package.json
```

---

## What We Reuse vs Build

| Layer | Reuse (existing) | Build (new) |
|-------|-------------------|-------------|
| **User/Team mgmt** | `UserManager`, `Team`, `UserRole` | JWT middleware, REST endpoints |
| **Knowledge base** | `UnifiedMemory`, `SimpleRAG`, `SpinningWheel`, `KnowledgeSharing` | Per-team isolation service, file upload |
| **Collaboration** | `SessionManager`, `PresenceManager`, `StateSynchronizer`, `VoiceManager` | WebSocket hub, session REST API |
| **Task planning** | `Tapestry`, `LoomKeeper`, `HierarchicalPlanner`, `ChainPatterns` | Team planner service, task REST API |
| **Insights** | `JennyRuntime`, Tufte visualizations, `FullLearningEngine`, `AttributionManager` | Team analytics aggregation |
| **API server** | `agentic_api.py`, rate limiter, health checks | New route mounts, auth |
| **Frontend** | Jenny `ReactRenderer` props, `workflow_builder.html` patterns | Full React app |

**Ratio**: ~80% reuse, ~20% new integration code.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Memory isolation between teams | Use team_id prefix in all storage paths; separate UnifiedMemory instances per team |
| Real-time sync at scale | Existing CRDT sync handles conflicts; add Redis pub/sub for multi-server |
| Auth security | JWT with refresh tokens; existing RBAC enforced at API layer |
| Performance with many teams | Memory Conductor's AUTO strategy adapts; query cache provides 100x speedup |
| Data loss | HoloLoom's "archive instead of delete" philosophy; Neo4j + Qdrant persistence |
| Graceful degradation | All HoloLoom modules fall back cleanly (e.g., HYBRID → INMEMORY) |

---

## Success Metrics

| Metric | Target | Measured By |
|--------|--------|-------------|
| Knowledge base query latency | <200ms (cold), <2ms (cached) | `RAGResult.metadata.latency_ms` |
| Team knowledge growth | >10 nodes/week per active team | `UnifiedMemory.health_check()` |
| Query quality over time | Confidence trend >0.8 | `render_confidence_trajectory()` |
| Collaboration adoption | >2 sessions/week per team | `SessionManager` event logs |
| Task completion rate | >70% tasks completed per goal | `Tapestry.get_status_summary()` |
| Learning adaptation | Thompson priors converge within 50 queries | `get_learning_statistics()` |
