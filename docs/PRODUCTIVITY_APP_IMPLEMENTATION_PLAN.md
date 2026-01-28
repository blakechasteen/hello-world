# HoloTeam: A Productivity Platform for Teams That Learn

**Date**: 2026-01-28
**Codename**: HoloTeam
**Foundation**: HoloLoom (~80,000 lines of production backend)

---

## Product Vision

Most productivity tools treat knowledge as static inventory — files in folders, messages in channels, tasks on boards. Teams outgrow these tools because the tools never learn. A question answered six months ago sits buried in a thread; a decision made last quarter lives only in someone's memory.

HoloTeam is different. It is a workspace where **the room remembers**. Every question asked, every document ingested, every decision made becomes part of a living knowledge graph that grows smarter with use. The system learns what your team cares about, surfaces connections you missed, and adapts its behavior to how your team actually works — not how a product manager imagined you would.

**Three convictions shape the product:**

1. **Knowledge compounds.** A team's accumulated understanding is its most valuable asset. HoloTeam treats knowledge as a first-class entity with relationships, provenance, and memory — not as flat text in a search index.

2. **Learning is observable.** Thompson Sampling, the same algorithm that powers HoloLoom's decision engine, drives HoloTeam's recommendations. The system doesn't just store what you know — it tracks what works, what doesn't, and adjusts. Teams can see their learning curve.

3. **Collaboration is the unit of work.** Not the individual. Every feature is designed around shared context: shared memory, shared sessions, shared goals. Attribution ensures individual contributions are recognized, but the workspace belongs to the team.

---

## Design Principles

**Quiet intelligence.** The system should feel like a well-organized library with an unusually attentive librarian — not like a chatbot demanding attention. Recommendations appear when relevant. Connections surface when you're looking. The AI assists; it does not perform.

**Progressive disclosure.** A new team should be productive in five minutes: create a workspace, ingest a few documents, ask a question. Advanced features — reasoning modes, workflow chains, Thompson Sampling dashboards — reveal themselves as the team matures.

**Earned trust.** Every answer shows its sources. Every recommendation explains its reasoning. Confidence scores are visible, not hidden. When the system is uncertain, it says so. This transparency is not a feature; it is a commitment.

**Graceful degradation.** If Neo4j is down, the knowledge graph falls back to in-memory. If the LLM is unavailable, RAG returns sources without generation. If a team member loses connection mid-session, their state is preserved. Nothing crashes. Nothing is lost.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        HoloTeam Frontend                         │
│              React 18 · TypeScript · Tailwind CSS                │
│                                                                   │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Workspace  │  │ Knowledge  │  │  Goals   │  │  Insights   │  │
│  │   Home     │  │   Base     │  │ & Tasks  │  │  Dashboard  │  │
│  └────────────┘  └────────────┘  └──────────┘  └─────────────┘  │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────────┐   │
│  │  Session   │  │  Query     │  │   Knowledge Graph        │   │
│  │  Sidebar   │  │  Console   │  │   (D3 force-directed)    │   │
│  └────────────┘  └────────────┘  └──────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST + WebSocket
┌───────────────────────────┴─────────────────────────────────────┐
│                       HoloTeam API Layer                         │
│                    FastAPI · Python 3.11+                         │
│                                                                   │
│  Auth (JWT)  ·  Teams  ·  Knowledge  ·  Sessions  ·  Tasks      │
│  Insights  ·  WebSocket Hub  ·  File Upload                      │
│                                                                   │
│  Extends: HoloLoom/server/agentic_api.py                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────────┐
│                    HoloLoom Backend (existing)                    │
│                                                                   │
│  Memory          │ Learning        │ Collaboration               │
│  ─────────────── │ ─────────────── │ ───────────────────────     │
│  UnifiedMemory   │ Thompson        │ SessionManager              │
│  SimpleRAG       │  Sampling       │ PresenceManager             │
│  MemoryConductor │ FullLearning    │ StateSynchronizer           │
│  KnowledgeGraph  │  Engine         │ AttributionManager          │
│  QueryCache      │ HotPatterns     │ KnowledgeSharing            │
│                  │                 │ VoiceManager                │
│  Planning        │ Visualization   │ Tapestry                    │
│  ─────────────── │ ─────────────── │ ───────────────────────     │
│  HTN Planner     │ JennyRuntime    │ LoomKeeper                  │
│  Multi-Agent     │ ReactRenderer   │ ThreadStatus lifecycle      │
│  ChainPatterns   │ Tufte panels    │ FabricSignal verify         │
│  POMDP           │ D3 knowledge    │ JSON persistence            │
│                  │  graph          │                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phases

### Phase 1: Foundation — Teams and Identity

**The experience.** A team lead signs up. She creates a workspace called "Platform Engineering." She invites three colleagues by email. Each receives a link, creates an account, and lands in a clean workspace with a welcome message and an empty knowledge base. The whole process takes under two minutes.

**What HoloLoom provides:**

| Existing Module | What It Gives Us |
|----------------|-----------------|
| `collaboration/user_manager.py` | `UserManager` — user CRUD, login/logout, session tokens |
| `collaboration/user_manager.py` | `Team` — team creation, member management |
| `collaboration/access_control.py` | `AccessController` — RBAC with 5 levels: `OWNER · ADMIN · EDITOR · COMMENTER · VIEWER` |
| `server/agentic_api.py` | FastAPI server with health, query, stats endpoints |

**What we build:**

```
holoteam/
├── app.py                    # FastAPI main — mounts agentic_api + HoloTeam routes
├── config.py                 # Environment-based config (dev/staging/prod)
├── api/
│   ├── auth.py               # JWT middleware wrapping UserManager.login()
│   ├── users.py              # Profile, preferences, avatar
│   └── teams.py              # CRUD, invite flow, role management
└── services/
    └── onboarding.py         # Welcome sequence, sample data seeding
```

**Service contract — `TeamService`:**

```python
from HoloLoom.collaboration.user_manager import UserManager, UserRole, Team
from HoloLoom.collaboration.access_control import AccessController

class TeamService:
    """Thin wrapper providing team lifecycle operations."""

    def __init__(self, storage_root: str = ".holoteam"):
        self.users = UserManager(storage_path=f"{storage_root}/users.json")
        self.access = AccessController()

    async def create_team(self, name: str, owner_id: str) -> Team:
        return self.users.create_team(name, owner_id)

    async def invite(self, team_id: str, username: str, role: UserRole = UserRole.EDITOR):
        user = self.users.get_user_by_username(username)
        self.users.add_team_member(team_id, user.user_id, role)
        # Role options: OWNER, ADMIN, EDITOR, COMMENTER, VIEWER

    async def check_permission(self, user_id: str, team_id: str, action: str) -> bool:
        return self.access.check_permission(user_id, team_id, action)
```

**API surface:**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/auth/login` | Authenticate, receive JWT |
| `POST` | `/auth/register` | Create account |
| `POST` | `/teams` | Create workspace |
| `GET` | `/teams` | List user's workspaces |
| `POST` | `/teams/{id}/invite` | Invite member by email/username |
| `PATCH` | `/teams/{id}/members/{uid}` | Change role |
| `DELETE` | `/teams/{id}/members/{uid}` | Remove member |

**Deliverable:** Authenticated API with team lifecycle. Teams created, members invited with roles, permissions enforced.

---

### Phase 2: Shared Knowledge Base — The Team's Memory

**The experience.** The team lead drags a PDF of their architecture doc into the workspace. It's ingested in seconds. A teammate pastes a Confluence URL — also ingested. Someone types a quick note about a design decision. All of it flows into the same knowledge graph, connected by entities and relationships.

Later, a new hire joins. Instead of reading a hundred Slack threads, she opens the workspace and asks: "What's our caching strategy?" The answer comes back with sources, confidence score, and links to the original documents. She bookmarks it. The system notes her interest and surfaces related knowledge over the next few days.

**What HoloLoom provides:**

| Existing Module | What It Gives Us |
|----------------|-----------------|
| `memory/unified.py` | `UnifiedMemory` — store, recall (5 strategies), navigate (4 directions), discover patterns, time travel |
| `rag/simple_rag.py` | `SimpleRAG` — zero-config RAG with 4 reasoning modes: `DIRECT · VERIFY · RESEARCH · PLAN_EXECUTE` |
| `memory_symphony/` | `MemoryConductor` — intelligent routing across 7 memory systems: `FAST · BALANCED · DEEP · RESEARCH · AUTO` |
| `spinningWheel/` | 47 input adapters — PDF, DOCX, URL, audio, video, code, CSV, JSON, Slack, email, and more |
| `collaboration/knowledge_sharing.py` | `KnowledgeSharing` — export (`JSON · JSON_LD · MARKDOWN · CSV · RDF`), share (`USER · TEAM · ORGANIZATION · PUBLIC`) |
| `memory/query_cache.py` | `QueryCache` — 100x speedup for repeated queries (<1ms cached) |

**Recall strategies** (from `UnifiedMemory`):

| Strategy | Conductor Mapping | Best For |
|----------|------------------|---------|
| `RECENT` | → `FAST` | "What did we just discuss?" |
| `SIMILAR` | → `FAST` | "Find things like this" |
| `CONNECTED` | → `BALANCED` | "What's related to X?" |
| `RESONANT` | → `DEEP` | "What patterns connect these?" |
| `BALANCED` | → `AUTO` | General queries (default) |

**Service contract — `TeamKnowledgeBase`:**

```python
from HoloLoom.memory.unified import UnifiedMemory, RecallStrategy, NavigationDirection
from HoloLoom.rag.simple_rag import SimpleRAG, RAGResult
from HoloLoom.collaboration.knowledge_sharing import (
    KnowledgeSharing, ExportFormat, ShareScope, SharedKnowledge
)

class TeamKnowledgeBase:
    """Per-team knowledge base backed by UnifiedMemory + RAG."""

    def __init__(self, team_id: str, storage_root: str = ".holoteam"):
        self.team_id = team_id
        self.memory = UnifiedMemory(
            user_id=team_id,
            enable_conductor=True,   # Multi-system routing via MemoryConductor
            enable_consolidation=True # Background dedup + summarization every 60 min
        )
        self.rag = SimpleRAG()       # Zero-config, uses Config.fast() by default
        self.sharing = KnowledgeSharing()

    async def ingest(self, content: str, contributor_id: str, source: str = None) -> str:
        """Ingest content into the team's knowledge graph."""
        mem = await self.memory.store(content, context={
            "team_id": self.team_id,
            "contributed_by": contributor_id,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        await self.rag.ingest(content)
        return mem.id

    async def query(self, question: str, mode: str = "direct") -> RAGResult:
        """Ask a question. Returns RAGResult with response, sources, confidence, reasoning_mode."""
        # mode: "direct" (<150ms), "verify" (~600ms), "research" (~900ms), "plan_execute" (~750ms)
        return await self.rag.query(question, mode=mode)

    async def recall(self, query: str, strategy: str = "balanced", limit: int = 20):
        """Browse memories by strategy."""
        return await self.memory.recall(
            query, strategy=RecallStrategy[strategy.upper()], limit=limit
        )

    async def navigate(self, from_id: str, direction: str, steps: int = 3):
        """Spatial navigation: FORWARD, BACKWARD, SIDEWAYS, DEEP."""
        return self.memory.navigate(
            from_id, NavigationDirection[direction.upper()], steps=steps
        )

    async def discover_patterns(self, types: list[str] = None, min_strength: float = 0.3):
        """Find emergent patterns: LOOP, CLUSTER, RESONANCE, THREAD."""
        return self.memory.discover_patterns(
            pattern_types=types or ["cluster", "thread", "resonance"],
            min_strength=min_strength
        )

    async def export(self, format: str = "markdown") -> bytes:
        """Export knowledge base."""
        return await self.sharing.export(
            self.memory, format=ExportFormat[format.upper()]
        )

    async def share(self, scope: str = "team") -> SharedKnowledge:
        """Create shareable snapshot."""
        return await self.sharing.share(
            self.memory, scope=ShareScope[scope.upper()]
        )
```

**API surface:**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/teams/{id}/knowledge` | Ingest text content |
| `POST` | `/teams/{id}/knowledge/upload` | Ingest file (routed to SpinningWheel) |
| `GET` | `/teams/{id}/knowledge/query` | `?q=...&mode=verify` — RAG query |
| `GET` | `/teams/{id}/knowledge/browse` | `?strategy=connected&limit=20` — recall |
| `GET` | `/teams/{id}/knowledge/navigate` | `?from=node_id&direction=sideways&steps=3` |
| `GET` | `/teams/{id}/knowledge/patterns` | Discover clusters, threads, loops |
| `GET` | `/teams/{id}/knowledge/timeline` | `?start=...&end=...` — time-travel |
| `POST` | `/teams/{id}/knowledge/export` | Export as Markdown, JSON, CSV, RDF |
| `POST` | `/teams/{id}/knowledge/share` | Create shareable link (team/org/public scope) |

**Deliverable:** Fully functional team knowledge base. Ingest from any format, query with four reasoning modes, browse by five recall strategies, navigate the graph spatially, discover emergent patterns, export, and share.

---

### Phase 3: Real-Time Collaboration — Working Together

**The experience.** Two engineers open a Research session to explore a caching strategy. They see each other's cursors. One asks a question; the RAG answer appears for both. The other highlights a key insight and pins it. Their contributions are tracked — who asked what, who curated which answers.

A third colleague joins from mobile. She sees the session history, adds a comment, and drops off. The CRDT sync ensures no edits conflict. When the session ends, all insights are automatically committed to the team's knowledge base.

**What HoloLoom provides:**

| Existing Module | What It Gives Us |
|----------------|-----------------|
| `collaboration/session.py` | `SessionManager` — 5 session types, event system, participant management |
| `collaboration/presence.py` | `PresenceManager` — cursor tracking, typing indicators, online status |
| `collaboration/sync.py` | `StateSynchronizer` — CRDT conflict resolution with 4 strategies |
| `collaboration/voice.py` | `VoiceManager` — WebRTC voice/video rooms |
| `collaboration/attribution.py` | `AttributionManager` — 14 contribution types with quality scoring |
| `collaboration/ux_learning.py` | `UXLearner` — 8 learnable features with Thompson Sampling |

**Session types** (from `SessionType` enum):

| Type | Purpose | Typical Duration |
|------|---------|-----------------|
| `KNOWLEDGE_BASE` | Curate and organize knowledge | Ongoing |
| `WHITEBOARD` | Brainstorm and ideate | 30-90 min |
| `RESEARCH` | Multi-query exploration | 1-4 hours |
| `REVIEW` | Review and validate knowledge | 30-60 min |
| `PRESENTATION` | Share findings (read-only for viewers) | 15-45 min |

**Participant roles** (from `ParticipantRole` enum):
`OWNER · ADMIN · EDITOR · COMMENTER · VIEWER`

**Contribution tracking** (from `ContributionType` enum — 14 types):
`CREATE · KNOWLEDGE_ADD · QUERY · ANNOTATION · EDIT · DELETE · REVIEW · APPROVE · REJECT · MERGE · LINK · TAG · COMMENT · SHARE`

**UX features that learn** (from `UXLearner`, Thompson Sampling with `BetaPrior(alpha, beta)`):

| Feature | What Adapts |
|---------|-------------|
| `NOTIFICATION_LEVEL` | How often to notify (quiet → frequent) |
| `PRESENCE_STYLE` | How to show others (minimal → detailed) |
| `COLLABORATION_MODE` | Solo vs. pair vs. group default |
| `LAYOUT_PREFERENCE` | Panel arrangement |
| `AUTO_SAVE_FREQUENCY` | How often to save state |
| `SEARCH_DEPTH` | Default recall strategy depth |
| `ANNOTATION_STYLE` | Inline vs. sidebar annotations |
| `VOICE_ACTIVATION` | Push-to-talk vs. voice-activated |

**Service contract — `TeamSessionService`:**

```python
from HoloLoom.collaboration.session import SessionManager, SessionType, ParticipantRole
from HoloLoom.collaboration.presence import PresenceManager
from HoloLoom.collaboration.sync import StateSynchronizer
from HoloLoom.collaboration.attribution import AttributionManager, ContributionType

class TeamSessionService:
    """Manages real-time collaboration sessions for a team."""

    def __init__(self, team_id: str):
        self.team_id = team_id
        self.sessions = SessionManager()
        self.presence = PresenceManager()
        self.sync = StateSynchronizer()
        self.attribution = AttributionManager()

    async def create_session(
        self, name: str, owner_id: str, owner_name: str,
        session_type: SessionType = SessionType.KNOWLEDGE_BASE
    ):
        session = await self.sessions.create_session(
            name=name, owner_id=owner_id, owner_name=owner_name,
            session_type=session_type, tags=[self.team_id]
        )
        return session

    async def join(self, session_id: str, user_id: str, user_name: str):
        await self.sessions.add_participant(session_id, user_id, user_name)
        self.presence.register(session_id, user_id)

    async def record_contribution(
        self, session_id: str, user_id: str,
        contribution_type: ContributionType, content_id: str
    ):
        """Track who contributed what. Feeds into team analytics."""
        self.attribution.record(
            session_id=session_id, user_id=user_id,
            contribution_type=contribution_type,
            content_id=content_id
        )

    async def update_presence(self, session_id: str, user_id: str, cursor: dict):
        """Update cursor position for real-time presence."""
        self.presence.update(session_id, user_id, cursor)

    async def get_presence(self, session_id: str) -> list:
        """Get all active participants with positions."""
        return self.presence.get_active(session_id)
```

**WebSocket protocol:**

```
Client → Server:
  { "type": "presence_update", "cursor": { "x": 120, "y": 340 } }
  { "type": "state_change", "op": { "path": "/notes/3", "value": "...", "version": 7 } }
  { "type": "query", "text": "What's our caching strategy?" }

Server → Client:
  { "event": "participant_joined", "data": { "user_id": "...", "name": "..." } }
  { "event": "presence_broadcast", "data": [{ "user_id": "...", "cursor": {...} }] }
  { "event": "state_synced", "data": { "op": {...}, "version": 8 } }
  { "event": "query_result", "data": { "response": "...", "confidence": 0.91 } }
```

**API surface:**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/teams/{id}/sessions` | Create session (type, name) |
| `GET` | `/teams/{id}/sessions` | List active sessions |
| `GET` | `/teams/{id}/sessions/{sid}` | Session details + participants |
| `POST` | `/teams/{id}/sessions/{sid}/join` | Join session |
| `WS` | `/teams/{id}/sessions/{sid}/ws` | Real-time presence + sync |
| `GET` | `/teams/{id}/contributions` | Attribution summary |

**Deliverable:** Real-time collaboration with presence, CRDT sync, five session types, contribution tracking, and adaptive UX that learns from team preferences.

---

### Phase 4: Goals and Task Planning — The Team's Thread

**The experience.** The tech lead types a goal: "Migrate authentication to OAuth2." The system suggests a task breakdown — it's seen similar goals before and uses the HTN planner to decompose the work. The team reviews, adjusts, and commits the plan. Each task becomes a thread in the Tapestry.

As work progresses, team members update threads. One is blocked — its status changes to TANGLED. The system detects this and suggests a reordering. When the last thread is woven, the goal is marked complete and a summary is committed to the knowledge base.

**What HoloLoom provides:**

| Existing Module | What It Gives Us |
|----------------|-----------------|
| `tapestry/protocol.py` | `Tapestry` — goal container with thread lifecycle |
| `tapestry/keeper.py` | `LoomKeeper` — start, resume, weave threads, verify with FabricSignal |
| `planning/planner.py` | `HierarchicalPlanner` — HTN planning with causal reasoning |
| `planning/multi_agent.py` | Multi-agent coordination with 4 negotiation protocols |
| `chaining/` | 17 pre-built `ChainPatterns` + evaluation (LLMJudge) |

**Thread lifecycle** (from `ThreadStatus` enum):

```
UNWOVEN → WEAVING → WOVEN
                  → TANGLED (blocked)
                  → UNRAVELED (abandoned)
```

**Tapestry data model** (from `tapestry/protocol.py`):

```python
@dataclass(frozen=True)
class Thread:
    id: str                    # Unique thread identifier
    description: str           # What needs to be done
    status: ThreadStatus       # UNWOVEN | WEAVING | WOVEN | TANGLED | UNRAVELED
    depends_on: tuple[str, ...] = ()   # Thread IDs this depends on
    assigned_to: str | None = None     # User ID
    commit_hash: str | None = None     # Git reference when complete
    notes: str = ""

class Tapestry:
    goal: str                          # The team's objective
    threads: tuple[Thread, ...]        # Immutable thread collection

    def create(goal, threads) -> Tapestry
    def next_unwoven() -> Thread | None         # Next unblocked task
    def update_thread(id, status, **kw) -> Tapestry  # Returns new immutable Tapestry
    def get_status_summary() -> dict            # {"unwoven": 3, "weaving": 1, "woven": 5, ...}
    def is_complete() -> bool
```

**Planning capabilities** (from `planning/planner.py`):

```python
@dataclass
class Action:
    name: str
    action_type: ActionType     # PHYSICAL | MENTAL | SOCIAL | INFORMATIONAL | META
    preconditions: list[str]
    effects: list[str]
    resources: list[str]
    duration_estimate: float    # hours

@dataclass
class Plan:
    goal: Goal
    actions: list[Action]       # Ordered by dependency
    estimated_duration: float
    confidence: float           # 0.0-1.0
```

**Multi-agent negotiation** (from `planning/multi_agent.py`):

| Protocol | Use Case |
|----------|---------|
| `CONTRACT_NET` | Task allocation across team members |
| `MONOTONIC_CONCESSION` | Resolving priority conflicts |
| `MEDIATION` | Third-party conflict resolution |
| `AUCTION` | Resource allocation |

**Chain patterns** (17 pre-built, from `chaining/`):

Useful for team workflows: `SequentialChain`, `ParallelChain`, `ConditionalChain`, `MapReduceChain`, `RouterChain`, `RetryChain`, `FallbackChain`, `TransformChain`, `ValidationChain`, `AggregationChain`, `DebateChain`, `RefinementChain`, `VerificationChain`, `SummarizationChain`, `ExtractionChain`, `ClassificationChain`, `ReActChain`.

**Service contract — `TeamPlanner`:**

```python
from HoloLoom.tapestry.protocol import Tapestry, ThreadStatus, Thread
from HoloLoom.tapestry.keeper import LoomKeeper
from HoloLoom.planning.planner import HierarchicalPlanner, Goal, Plan

class TeamPlanner:
    """Goal decomposition and task tracking backed by Tapestry + HTN Planner."""

    def __init__(self, team_id: str, storage_root: str = ".holoteam"):
        self.team_id = team_id
        self.keeper = LoomKeeper(path=f"{storage_root}/teams/{team_id}/tapestry.json")
        self.planner = HierarchicalPlanner()

    async def create_goal(self, goal: str, tasks: list[str], dependencies: dict = None) -> Tapestry:
        """Create a goal with manually specified tasks."""
        return await self.keeper.start(goal=goal, threads=tasks)

    async def auto_plan(self, goal_text: str) -> Plan:
        """Use HTN planner to decompose a goal into actions."""
        goal = Goal(description=goal_text)
        return await self.planner.plan(goal)

    async def get_progress(self) -> dict:
        result = await self.keeper.resume()
        if not result:
            return {"status": "no_active_goal"}
        tapestry, next_thread = result
        summary = tapestry.get_status_summary()
        return {
            "goal": tapestry.goal,
            "threads": [
                {"id": t.id, "description": t.description,
                 "status": t.status.value, "assigned_to": t.assigned_to}
                for t in tapestry.threads
            ],
            "summary": summary,
            "next_task": next_thread.description if next_thread else None,
            "is_complete": tapestry.is_complete()
        }

    async def update_thread(self, thread_id: str, status: str, **kwargs) -> Tapestry:
        """Update a thread's status. Returns new immutable Tapestry."""
        result = await self.keeper.resume()
        tapestry, _ = result
        return tapestry.update_thread(
            thread_id, ThreadStatus[status.upper()], **kwargs
        )
```

**API surface:**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/teams/{id}/goals` | Create goal with tasks |
| `POST` | `/teams/{id}/goals/auto-plan` | HTN-generated task breakdown |
| `GET` | `/teams/{id}/goals/current` | Progress summary |
| `GET` | `/teams/{id}/goals/history` | Past goals |
| `PATCH` | `/teams/{id}/threads/{tid}` | Update status, assignment, notes |
| `GET` | `/teams/{id}/threads` | List with filters (status, assignee) |

**Deliverable:** Goal creation with manual or auto-planned task decomposition, thread lifecycle tracking, dependency awareness, and history.

---

### Phase 5: Insights Dashboard — The Team's Mirror

**The experience.** The team lead opens the Insights tab. She sees six panels, each telling a different story:

- A **knowledge graph** shows the team's understanding as a web of connected concepts — dense clusters around their core expertise, sparse frontier areas where they're still learning.
- A **confidence trajectory** traces answer quality over the past month — it's climbing, with a dip last Tuesday when someone asked about an unfamiliar topic.
- A **contribution map** shows who's been building what — not as a leaderboard, but as a pattern: one engineer curates, another asks deep questions, a third connects ideas.
- A **hot topics** panel highlights what the team accessed most this week.
- A **learning curve** shows Thompson Sampling priors converging — the system is getting better at routing queries.
- A **task velocity** sparkline shows threads woven per week.

The panels are rendered by Jenny Runtime, which learns from the team's interactions: panels they pin stay prominent, panels they dismiss fade. Over time, the dashboard becomes the team's unique mirror.

**What HoloLoom provides:**

| Existing Module | What It Gives Us |
|----------------|-----------------|
| `visualization/jenny_runtime.py` | `JennyRuntime` — multi-target panel rendering (HTML, React, AR) |
| `visualization/jenny_spec.py` | `JennySpec` — 13 panel types with lifecycle management |
| `visualization/jenny_panel_learner.py` | `PanelTypeLearner` — Thompson Sampling for panel selection |
| `visualization/confidence_trajectory.py` | Confidence over time with anomaly detection |
| `visualization/knowledge_graph.py` | Force-directed graph with semantic edge colors |
| `visualization/cache_gauge.py` | Cache effectiveness radial gauge |
| `visualization/stage_waterfall.py` | Pipeline timing waterfall |
| `visualization/small_multiples.py` | Side-by-side query comparison |
| `recursive/hot_pattern_feedback.py` | `HotPatternTracker` — most accessed knowledge |
| `recursive/` | `FullLearningEngine.get_learning_statistics()` |
| `collaboration/attribution.py` | `AttributionManager` — who contributed what |

**Jenny panel types** (from `PanelTypeJenny` enum — 13 types):
`TEXT · CODE · GRAPH · IMAGE · TABLE · COMPOSITE · METRIC · TIMELINE · CONFIDENCE · COMPARISON · REASONING · RECOMMENDATION · SPATIAL`

**Panel sizes**: `SMALL · MEDIUM · LARGE · FULL`
**Binding modes**: `STATIC · REACTIVE · STREAMING`
**Lifecycle**: `NASCENT → STABLE → DISSOLVING → ARCHIVED → SYSTEM`

**Dashboard panels for HoloTeam:**

| Panel | Jenny Type | Data Source | Binding |
|-------|-----------|-------------|---------|
| Knowledge Graph | `GRAPH` | `render_knowledge_graph_from_kg()` | `REACTIVE` |
| Confidence Trajectory | `CONFIDENCE` | `render_confidence_trajectory()` | `STREAMING` |
| Contributor Map | `TABLE` | `AttributionManager.get_summary()` | `REACTIVE` |
| Hot Topics | `RECOMMENDATION` | `HotPatternTracker.get_hot_patterns()` | `REACTIVE` |
| Learning Curve | `METRIC` | `FullLearningEngine.get_learning_statistics()` | `REACTIVE` |
| Task Velocity | `TIMELINE` | `Tapestry.get_status_summary()` history | `REACTIVE` |
| Query Patterns | `COMPARISON` | `detect_temporal_patterns()` | `STATIC` |
| Cache Performance | `METRIC` | `render_cache_gauge()` | `STATIC` |

**Service contract — `TeamAnalytics`:**

```python
from HoloLoom.visualization.jenny_runtime import JennyRuntime, create_runtime
from HoloLoom.visualization.jenny_spec import (
    JennySpec, PanelTypeJenny, PanelSizeJenny, BindingMode
)
from HoloLoom.visualization.confidence_trajectory import render_confidence_trajectory
from HoloLoom.visualization.knowledge_graph import render_knowledge_graph_from_kg

class TeamAnalytics:
    """Aggregates team metrics into Jenny-rendered dashboard panels."""

    def __init__(self, team_kb: TeamKnowledgeBase, attribution: AttributionManager):
        self.kb = team_kb
        self.attribution = attribution
        self.runtime = create_runtime(enable_learning=True)

    async def get_dashboard_specs(self) -> list[JennySpec]:
        """Generate JennySpec list for React rendering."""
        health = self.kb.memory.health_check()
        patterns = self.kb.memory.detect_temporal_patterns(
            min_occurrences=2, time_window_days=7
        )

        return [
            JennySpec(
                spacetime_id=f"{self.kb.team_id}-kg",
                panel_type=PanelTypeJenny.GRAPH,
                title="Team Knowledge Map",
                content={"graph_data": health.get("components", {}).get("graph", {})},
                size=PanelSizeJenny.LARGE,
                priority=1,
                binding_mode=BindingMode.REACTIVE
            ),
            JennySpec(
                spacetime_id=f"{self.kb.team_id}-confidence",
                panel_type=PanelTypeJenny.CONFIDENCE,
                title="Answer Quality Trend",
                content={"trajectory": "streaming"},
                size=PanelSizeJenny.MEDIUM,
                priority=2,
                binding_mode=BindingMode.STREAMING
            ),
            # ... additional panels
        ]

    async def render_for_react(self) -> dict:
        """Render dashboard as React-compatible props."""
        specs = await self.get_dashboard_specs()
        return await self.runtime.render(specs, target="react")

    async def get_knowledge_stats(self) -> dict:
        health = self.kb.memory.health_check()
        return {
            "total_memories": health.get("components", {}).get("graph", {}).get("nodes", 0),
            "total_connections": health.get("components", {}).get("graph", {}).get("edges", 0),
            "recurring_topics": [
                p["description"] for p in
                self.kb.memory.detect_temporal_patterns(min_occurrences=2, time_window_days=7)
                if p["pattern_type"] == "recurring_topic"
            ]
        }

    async def get_contributors(self) -> list:
        """Contribution summary per team member."""
        return self.attribution.get_summary(group_by="user_id")
```

**API surface:**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/teams/{id}/insights/dashboard` | React props for full dashboard |
| `GET` | `/teams/{id}/insights/knowledge-stats` | Knowledge graph metrics |
| `GET` | `/teams/{id}/insights/contributors` | Attribution summary |
| `GET` | `/teams/{id}/insights/hot-topics` | Most accessed knowledge |
| `GET` | `/teams/{id}/insights/learning` | Thompson Sampling convergence |
| `POST` | `/teams/{id}/insights/panels/{pid}/pin` | Pin panel (Thompson learning signal) |
| `POST` | `/teams/{id}/insights/panels/{pid}/dismiss` | Dismiss panel (Thompson learning signal) |

**Deliverable:** Adaptive insights dashboard with eight panel types, Thompson Sampling panel selection that learns from team preferences, and comprehensive team analytics.

---

### Phase 6: Frontend — The Interface

**The experience.** A team member opens HoloTeam in their browser. The workspace feels immediate: a clean sidebar shows the knowledge base, active sessions, and current goals. The query console sits at the bottom — always available, like a command palette.

There's no tutorial needed. The first screen shows three things: your team's knowledge graph (the big picture), the current goal progress (the focused work), and recent activity (what's happening now). Everything else is one click away.

**Tech stack:**
- React 18 with TypeScript
- Tailwind CSS for utility-first styling
- React Query for server state
- D3.js for knowledge graph and visualizations
- WebSocket via native API (no Socket.IO — keep it simple)

**Component architecture:**

```
ui/src/
├── app/
│   ├── App.tsx                    # Root layout + routing
│   └── AuthProvider.tsx           # JWT context
│
├── pages/
│   ├── Login.tsx                  # Auth flow
│   ├── Workspace.tsx              # Main workspace (default view)
│   ├── KnowledgeBase.tsx          # Browse, query, ingest
│   ├── Session.tsx                # Live collaboration view
│   ├── Goals.tsx                  # Task board + planning
│   └── Insights.tsx               # Dashboard (Jenny React props)
│
├── components/
│   ├── layout/
│   │   ├── Sidebar.tsx            # Navigation: KB, sessions, goals, insights
│   │   ├── Header.tsx             # Team name, members online, search
│   │   └── QueryConsole.tsx       # Persistent query bar (bottom)
│   │
│   ├── knowledge/
│   │   ├── QueryResult.tsx        # RAG result with sources + confidence
│   │   ├── SourceCard.tsx         # Individual source with relevance
│   │   ├── IngestDropzone.tsx     # Drag-and-drop file upload
│   │   ├── KnowledgeGraph.tsx     # D3 force-directed graph
│   │   └── PatternList.tsx        # Clusters, threads, loops
│   │
│   ├── collaboration/
│   │   ├── SessionBar.tsx         # Active session indicator
│   │   ├── PresenceCursors.tsx    # Other users' cursors
│   │   ├── ContributionBadge.tsx  # Inline attribution marker
│   │   └── SessionTypeSelector.tsx
│   │
│   ├── tasks/
│   │   ├── ThreadCard.tsx         # Single task with status chip
│   │   ├── GoalProgress.tsx       # Progress bar + summary
│   │   ├── ThreadBoard.tsx        # Kanban-style columns by status
│   │   └── PlanSuggestion.tsx     # Auto-plan from HTN planner
│   │
│   └── insights/
│       ├── JennyPanel.tsx         # Generic panel renderer (from React props)
│       ├── ConfidenceChart.tsx    # SVG confidence trajectory
│       ├── CacheGauge.tsx         # Radial gauge
│       └── ContributorTable.tsx   # Attribution leaderboard
│
├── hooks/
│   ├── useWebSocket.ts           # Session presence + sync
│   ├── useTeamKB.ts              # Knowledge base queries (React Query)
│   ├── useGoals.ts               # Task tracking (React Query)
│   └── useInsights.ts            # Dashboard data (React Query)
│
└── services/
    └── api.ts                     # Typed API client for all endpoints
```

**Key interaction patterns:**

1. **Query Console** — always visible at the bottom. Type a question, select reasoning mode (direct/verify/research), see results inline. Pin results to the knowledge base.

2. **Knowledge Graph** — D3 force-directed layout. Click a node to see its content. Shift-click to navigate (FORWARD/BACKWARD/SIDEWAYS/DEEP). Clusters highlighted in matching colors.

3. **Session View** — split screen: shared workspace on left, query console on right. Presence cursors float over the workspace. Contributions are attributed inline.

4. **Task Board** — four columns: Unwoven, Weaving, Woven, Tangled. Drag to change status. Click to expand details, add notes, assign.

5. **Insights Dashboard** — grid of Jenny panels. Pin to keep, dismiss to remove. Panels rearrange based on importance. Dashboard layout persists per team.

**Deliverable:** Complete web application with five pages, responsive layout, real-time collaboration, and adaptive dashboard.

---

## Integration Dependency Map

```
Phase 1: Foundation (Auth, Teams)
  └── Phase 2: Knowledge Base (Memory, RAG, Ingestion)
        ├── Phase 3: Collaboration (Sessions, Presence, Sync)
        ├── Phase 4: Goals & Tasks (Tapestry, Planner)
        └── Phase 5: Insights (Jenny, Analytics, Learning)
              └── Phase 6: Frontend (React App)
                    ├── consumes Phase 2 API (KB)
                    ├── consumes Phase 3 WebSocket (sessions)
                    ├── consumes Phase 4 API (tasks)
                    └── consumes Phase 5 API (dashboard)
```

Phases 3, 4, and 5 are independent of each other and can be developed in parallel after Phase 2. Phase 6 builds incrementally as each backend phase completes.

---

## File Structure

```
holoteam/
├── app.py                              # FastAPI main — mounts all routers
├── config.py                           # Environment config (dev/staging/prod)
│
├── api/                                # HTTP + WebSocket endpoints
│   ├── auth.py                         # JWT authentication
│   ├── users.py                        # User profile, preferences
│   ├── teams.py                        # Team CRUD, invite, roles
│   ├── knowledge.py                    # Ingest, query, browse, export, share
│   ├── sessions.py                     # Collaboration sessions
│   ├── tasks.py                        # Goals, threads, progress
│   ├── insights.py                     # Dashboard, analytics
│   └── ws.py                           # WebSocket hub (presence, sync, notifications)
│
├── services/                           # Business logic layer
│   ├── team_memory.py                  # TeamKnowledgeBase — per-team UnifiedMemory + RAG
│   ├── team_sessions.py                # TeamSessionService — sessions, presence, sync
│   ├── team_planner.py                 # TeamPlanner — Tapestry + HTN Planner
│   ├── team_analytics.py              # TeamAnalytics — Jenny dashboard + metrics
│   ├── ingestion.py                    # File upload → SpinningWheel routing
│   ├── attribution.py                  # Contribution tracking wrapper
│   ├── recommendations.py             # Thompson Sampling suggestions
│   ├── workflows.py                    # ChainPattern templates for teams
│   └── onboarding.py                   # Welcome flow, sample data
│
└── ui/                                 # React frontend (Phase 6)
    ├── src/
    │   ├── app/                        # Root layout, auth provider
    │   ├── pages/                      # 5 pages (Workspace, KB, Session, Goals, Insights)
    │   ├── components/                 # Organized by domain
    │   ├── hooks/                      # WebSocket, React Query wrappers
    │   └── services/                   # Typed API client
    ├── package.json
    └── tailwind.config.js
```

---

## What We Reuse vs. Build

| Layer | Reuse (HoloLoom) | Build (HoloTeam) |
|-------|-------------------|-------------------|
| **Identity** | `UserManager`, `Team`, `UserRole`, `AccessController` | JWT middleware, REST endpoints, invite flow |
| **Knowledge** | `UnifiedMemory`, `SimpleRAG`, `MemoryConductor`, `SpinningWheel` (47 adapters), `KnowledgeSharing`, `QueryCache` | Per-team isolation service, file upload routing |
| **Collaboration** | `SessionManager` (5 types), `PresenceManager`, `StateSynchronizer` (CRDT), `VoiceManager`, `UXLearner` (8 features) | WebSocket hub, session REST API |
| **Tasks** | `Tapestry`, `LoomKeeper`, `HierarchicalPlanner`, `ChainPatterns` (17), multi-agent negotiation (4 protocols) | Team planner service, task REST API |
| **Insights** | `JennyRuntime` + `ReactRenderer`, Tufte visualizations (7 types), `FullLearningEngine`, `AttributionManager` (14 types), `HotPatternTracker` | Team analytics aggregation, dashboard API |
| **Infrastructure** | FastAPI server, rate limiter, circuit breakers, health checks, Prometheus metrics | New route mounts, auth layer |
| **Frontend** | Jenny React props output, workflow builder patterns | Full React application |

**Ratio:** ~80% reuse of existing HoloLoom backend, ~20% new integration and frontend code.

---

## Data Model Summary

**Team-scoped isolation:** Each team gets its own `UnifiedMemory` instance, `Tapestry` file, and attribution history. The `team_id` prefix ensures complete data separation.

**Key entities and their source:**

| Entity | Source | Storage |
|--------|--------|---------|
| `User` | `collaboration/user_manager.py` | JSON file |
| `Team` | `collaboration/user_manager.py` | JSON file |
| `Memory` | `memory/unified.py` | Knowledge graph (Neo4j or in-memory) |
| `RAGResult` | `rag/simple_rag.py` | Transient (cached via QueryCache) |
| `Session` | `collaboration/session.py` | In-memory + event log |
| `Thread` | `tapestry/protocol.py` | JSON file (immutable snapshots) |
| `Tapestry` | `tapestry/protocol.py` | JSON file |
| `Contribution` | `collaboration/attribution.py` | In-memory + periodic flush |
| `JennySpec` | `visualization/jenny_spec.py` | Transient (rendered on request) |
| `SharedKnowledge` | `collaboration/knowledge_sharing.py` | Export file |

---

## Deployment

**Development:**
```bash
# Start HoloLoom backends (optional — falls back to in-memory)
docker-compose up -d  # Neo4j + Qdrant

# Start HoloTeam API
PYTHONPATH=. uvicorn holoteam.app:app --reload --port 8000

# Start frontend
cd holoteam/ui && npm run dev
```

**Production:**
```bash
# Docker Compose (all services)
docker-compose -f docker-compose.prod.yml up -d

# Includes:
# - Neo4j (graph database)
# - Qdrant (vector database)
# - HoloTeam API (4 uvicorn workers)
# - React frontend (nginx static serving)
# - Redis (session + pub/sub for multi-server)
```

**Kubernetes:**
Extend existing `k8s/` manifests. HoloTeam API deployment follows the same pattern as `hololoom-api-deployment.yaml` — 3 replicas, HPA at 70% CPU, health probes, ConfigMap for environment.

---

## Risk Mitigation

| Risk | Mitigation | HoloLoom Feature |
|------|------------|-----------------|
| Memory isolation between teams | `team_id` prefix on all storage; separate UnifiedMemory instances | Per-user memory scoping |
| Real-time sync conflicts | CRDT resolution with 4 strategies; operational transform fallback | `StateSynchronizer` |
| Auth security | JWT with refresh tokens; RBAC at API layer; rate limiting | `AccessController`, production hardening |
| Performance at scale | MemoryConductor AUTO routing; QueryCache 100x speedup; circuit breakers | `MemoryConductor`, `QueryCache`, circuit breakers |
| Data loss | Archive instead of delete; Neo4j + Qdrant persistence; JSON checkpoints | Graceful degradation philosophy |
| Degraded backends | Automatic fallback: HYBRID → INMEMORY; LLM unavailable → sources only | Graceful degradation throughout |
| Dashboard relevance | Thompson Sampling learns panel preferences; pin/dismiss signals | `PanelTypeLearner` |
| Knowledge quality | RAG verify mode cross-checks answers; confidence scores visible | `SimpleRAG` verify mode |

---

## Success Metrics

| Metric | Target | How We Measure |
|--------|--------|---------------|
| Knowledge query latency | <200ms cold, <2ms cached | `RAGResult.metadata.latency_ms` |
| Team knowledge growth | >10 nodes/week per active team | `UnifiedMemory.health_check()` graph stats |
| Answer quality trend | Confidence >0.8 sustained | `render_confidence_trajectory()` |
| Collaboration adoption | >2 sessions/week per team | `SessionManager` event count |
| Task completion rate | >70% threads woven per goal | `Tapestry.get_status_summary()` |
| Learning adaptation | Thompson priors converge in <50 queries | `FullLearningEngine.get_learning_statistics()` |
| Panel relevance | >60% pin rate on dashboard | `PanelTypeLearner` success rate |
| New hire onboarding | First useful query in <5 minutes | Time-to-first-query metric |
