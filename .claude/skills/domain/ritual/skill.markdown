# Skill: Ritual Orchestrator

## Metadata
- **Name**: `ritual`
- **Version**: `1.0.0`
- **Category**: `domain`
- **Tags**: `workflow, ritual, coding, memory, automation, hololoom`
- **Author**: HoloLoom Team
- **Created**: December 30, 2025

## Description

Orchestrates the 5-phase coding ritual (AWAKEN → PLAN → IMPLEMENT → REVIEW → REFLECT).
Semi-automatic workflow with guided decision points at phase transitions.

The ritual system leverages HoloLoom's memory infrastructure to:
- **Remember** context across sessions
- **Learn** from patterns in your codebase
- **Refine** prompts and plans before implementation
- **Review** code quality automatically
- **Capture** decisions for future recall

## Commands

### `/ritual start [feature_name]`
Begin a new ritual for a feature or task.

**Arguments:**
- `feature_name` (required): Name of the feature being worked on

**What happens:**
1. Creates ritual context with unique ID
2. Emits `ritual.started` event
3. Runs `/ritual-awaken` phase
4. Presents decision point: "Proceed to PLAN?" / "Explore more?"

**Example:**
```
/ritual start "Add Thompson Sampling to policy engine"
```

---

### `/ritual plan`
Transition to the PLAN phase.

**Prerequisites:** Must have an active ritual (after `/ritual start`)

**What happens:**
1. Prepares context handoff from AWAKEN (MI-filtered)
2. Emits `ritual.phase.started` event
3. Runs `/ritual-plan` phase
4. Presents decision point: "Approve plan?" / "Refine?" / "Research more?"

**Example:**
```
/ritual plan
```

---

### `/ritual review`
Transition to the REVIEW phase.

**Prerequisites:** Must have an active ritual

**What happens:**
1. Prepares context handoff from previous phase
2. Emits `ritual.phase.started` event
3. Runs `/ritual-review` phase
4. Presents decision point: "Approve?" / "Address issues?"

**Example:**
```
/ritual review
```

---

### `/ritual end`
Complete the ritual with REFLECT phase.

**Prerequisites:** Must have an active ritual

**What happens:**
1. Runs `/ritual-reflect` phase
2. Stores session summary in HoloLoom memory
3. Emits `ritual.completed` event
4. Closes ritual context

**Example:**
```
/ritual end
```

---

### `/ritual status`
Show current ritual status.

**What it shows:**
- Current phase (AWAKEN/PLAN/IMPLEMENT/REVIEW/REFLECT)
- Feature being worked on
- Decisions made so far
- Memories stored during ritual
- Time elapsed

**Example:**
```
/ritual status
```

---

### `/ritual jump [phase]`
Jump to a specific phase (skip intermediate phases).

**Arguments:**
- `phase`: Target phase (awaken, plan, implement, review, reflect)

**Example:**
```
/ritual jump review
```

---

### `/ritual pause`
Pause the current ritual (can be resumed later).

**What happens:**
1. Saves ritual state to memory
2. Emits `ritual.paused` event
3. Context preserved for later resumption

**Example:**
```
/ritual pause
```

---

### `/ritual resume`
Resume a paused ritual.

**What happens:**
1. Restores ritual context from memory
2. Emits `ritual.resumed` event
3. Continues from last phase

**Example:**
```
/ritual resume
```

---

### `/ritual cancel`
Cancel the current ritual without completing.

**What happens:**
1. Emits `ritual.cancelled` event
2. Clears ritual context
3. Optionally stores partial progress

**Example:**
```
/ritual cancel
```

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    /ritual (Orchestrator)                       │
│  Commands: start | plan | review | end | status | jump          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  AWAKEN  │ → │   PLAN   │ → │IMPLEMENT │ → │  REVIEW  │ →  │
│  │          │   │          │   │(manual)  │   │          │    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘    │
│       │              │              │              │           │
│       ▼              ▼              ▼              ▼           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ DECISION │   │ DECISION │   │   (no    │   │ DECISION │    │
│  │  POINT   │   │  POINT   │   │decision) │   │  POINT   │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│                                                      │         │
│                                               ┌──────▼─────┐   │
│                                               │  REFLECT   │   │
│                                               │            │   │
│                                               └────┬───────┘   │
│                                                    │           │
│                                               DECISION POINT   │
└─────────────────────────────────────────────────────────────────┘
```

## Decision Points

Each phase transition offers guided decision points:

| Phase | Decision Point | Options | Default |
|-------|---------------|---------|---------|
| AWAKEN | After context restore | "Proceed to PLAN" / "Explore more" | Proceed |
| PLAN | After plan creation | "Approve" / "Refine" / "Research more" | Approve |
| REVIEW | After quality check | "Approve" / "Address issues" | Approve |
| REFLECT | After summary | "Confirm" / "Add more" | Confirm |

## Integration with HoloLoom

The ritual orchestrator uses these HoloLoom MCP tools:

- **`holoLoom-memory:recall_memories`** - Restore context, find patterns
- **`holoLoom-memory:store_memory`** - Persist decisions, learnings
- **`holoLoom-memory:chat`** - Conversational interface with importance scoring
- **`holoLoom-memory:process_text`** - Ingest requirements, specs

## Standalone Phase Usage

Each phase can also run independently:

```
/ritual-awaken                    # Just restore context
/ritual-plan                      # Just create a plan
/ritual-review                    # Just review code
/ritual-reflect                   # Just end session
```

## Agent Invocation

The ritual system supports autonomous agent invocation via `RitualAgentCoordinator`:

```python
from .coordinator import create_ritual_coordinator

coordinator = create_ritual_coordinator(
    registry=registry,
    router=router,
    message_bus=bus,
    handoff=handoff
)

# Delegate full ritual to agent system
result = await coordinator.delegate_ritual(
    feature_name="Add Thompson Sampling",
    context_type="new_feature",
    auto_decisions=True  # Let Thompson Sampling decide
)
```

## Events

The orchestrator emits these events via EventBus:

| Event | When | Payload |
|-------|------|---------|
| `ritual.started` | `/ritual start` | ritual_id, feature_name |
| `ritual.phase.started` | Phase begins | phase, ritual_id |
| `ritual.phase.completed` | Phase ends | phase, success, confidence |
| `ritual.decision.required` | Decision needed | options, phase |
| `ritual.decision.made` | User decides | decision, phase |
| `ritual.completed` | `/ritual end` | summary, decisions |
| `ritual.cancelled` | `/ritual cancel` | partial_progress |
| `ritual.paused` | `/ritual pause` | state |
| `ritual.resumed` | `/ritual resume` | state |

All events share a `correlation_id` linking them to the same ritual workflow.

## Context Handoff

Between phases, context is filtered using MI-aware handoff:

- **AGGRESSIVE** (30% kept): Minimal high-MI context only
- **BALANCED** (50% kept): Standard handoffs (default)
- **CONSERVATIVE** (70% kept): Most context preserved
- **FULL** (100% kept): No filtering (research mode)

## Thompson Sampling Learning

The orchestrator learns optimal phase patterns over time:

- Success: `α ← α + confidence` (strengthen prior)
- Failure: `β ← β + (1 - confidence)` (weaken prior)
- Selection uses Beta(α, β) sampling for exploration/exploitation

## Examples

### Full Ritual Flow

```
> /ritual start "Add user authentication"

🌅 Starting ritual for: Add user authentication
   Ritual ID: ritual-20251230-143022

📚 Running AWAKEN phase...
   Recalling recent context...
   Found 5 relevant memories

🤔 Decision Point:
   [1] Proceed to PLAN (recommended)
   [2] Explore more context first

> 1

📋 Running PLAN phase...
   Analyzing requirements...
   Creating implementation plan...

🤔 Decision Point:
   [1] Approve plan
   [2] Refine plan
   [3] Research more

> 1

⚡ IMPLEMENT phase (manual coding)
   ... (you write code) ...

> /ritual review

🔍 Running REVIEW phase...
   Checking code quality...
   No issues found

🤔 Decision Point:
   [1] Approve (recommended)
   [2] Address issues

> 1

> /ritual end

🌙 Running REFLECT phase...
   Generating session summary...
   Storing to HoloLoom memory...

✅ Ritual Complete!
   Duration: 45 minutes
   Decisions: 4
   Memories stored: 3
```

### Quick Context Restore

```
> /ritual start "Continue yesterday's work"

🌅 Starting ritual for: Continue yesterday's work

📚 Running AWAKEN phase...
   Found context from yesterday:
   - Working on: Thompson Sampling integration
   - Last action: Implemented beta prior updates
   - Open items: Need to add logging

Ready to continue!
```

## Configuration

The orchestrator respects these configuration options:

```python
# Default handoff strategy
handoff_strategy = HandoffStrategy.BALANCED

# Token budget for context handoff
token_budget = 2000

# Auto-accept suggested decisions (agent mode)
auto_decisions = False

# Thompson Sampling exploration rate
exploration_rate = 0.1
```

## Dependencies

- `ritual_events.py` - Event types and RitualContext
- `phases/base.py` - AbstractRitualPhase protocol
- `phases/awaken.py` - AWAKEN phase skill
- `phases/plan.py` - PLAN phase skill
- `phases/review.py` - REVIEW phase skill
- `phases/reflect.py` - REFLECT phase skill
- `agent_registration.py` - RitualAgentRegistry
- `thompson_router.py` - RitualPhaseRouter
- `message_bus.py` - RitualMessageBus
- `context_handoff.py` - RitualContextHandoff
- `coordinator.py` - RitualAgentCoordinator

## See Also

- `/loom` - HoloLoom memory retrieval
- `/spinning-wheel` - Data ingestion
- `/prompt` - MRF prompt refinement
