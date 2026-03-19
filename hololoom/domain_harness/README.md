# Domain Harness: Complete Documentation

> **The agent is not the model. The agent is the transformation of one memory state into another.**

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [File Schemas](#file-schemas)
6. [The Ritual (Worker Protocol)](#the-ritual)
7. [Strands Layer](#strands-layer)
8. [Claude Code Integration](#claude-code-integration)
9. [CLI Reference](#cli-reference)
10. [API Reference](#api-reference)

---

## Overview

The Domain Harness implements the **Initializer/Worker pattern** for AI agents with persistent domain memory. It solves the fundamental problem of stateless LLMs: they forget everything between sessions.

### The Problem

```
Session 1: "Build feature A" → LLM implements A
Session 2: "Continue" → LLM: "What were we doing?"
```

### The Solution

```
Session 1: "Build feature A" → Worker reads memory → implements A → writes memory → exits
Session 2: "Continue" → Worker reads memory → sees A complete → implements B → writes memory → exits
```

The worker is stateless. The memory is persistent. Progress accumulates across sessions.

### Key Insight: Strands = Gas

Like gas in Ethereum, **tension** is the cost of incomplete work:

| Ethereum | Domain Harness |
|----------|----------------|
| Gas | Tension |
| High gas = expensive operation | High tension = needs attention |
| Low gas = cheap/complete | Low tension = stable |
| Failed transaction | Regression (tension spike) |
| Gas → 0 | Project complete |

---

## Core Concepts

### Domain Memory

Persistent storage that outlives any single agent session:

```
domain_memory/
├── features.json      # What needs to be built
├── state.json         # Project constraints
├── progress.log       # Append-only history
├── strands/           # Symbolic tension state
│   └── loom_state.json
└── tests/             # Test files (truth source)
    └── test_feature_FXXX.py
```

### Features

Atomic units of work with:
- **ID**: Unique identifier (F001, F002, ...)
- **Status**: pending → failing → in_progress → passing
- **Dependencies**: Other features that must pass first
- **Priority**: Higher = work on first
- **Acceptance Criteria**: What "done" means

### The Worker

A stateless executor that:
1. **Boots**: Reads domain memory
2. **Acts**: Implements ONE feature
3. **Exits**: Records progress, vanishes

The worker has no memory between sessions. Everything must be in files.

### Strands (Symbolic Layer)

Maps practical progress to symbolic weave motions:
- **Tension**: How much attention a feature needs (0-10)
- **Weave Motion**: What happened (warp_tug, weft_wander, etc.)
- **Loom State**: Aggregate view of all strands

---

## Architecture

### Two Backends

**1. Flat Files (Zero Dependencies)**

```python
from HoloLoom.domain_harness import FlatFileBackend

backend = FlatFileBackend(Path("./domain_memory"))
features = backend.get_actionable_features()
backend.update_feature_status("F001", "passing")
```

**2. Neo4j Graph (HoloLoom Integration)**

```python
from HoloLoom.domain_harness import Neo4jDomainProtocol

protocol = Neo4jDomainProtocol(driver, "project-uuid")
context = protocol.boot()
result = protocol.act(context, impl_fn, test_fn)
protocol.exit(result, worker_id="worker-001")
```

### Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     USER / CLAUDE CODE                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      DOMAIN HARNESS                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Initializer │  │   Worker    │  │      Protocol       │  │
│  │  (prompt →  │  │ (stateless  │  │  (boot/act/exit)    │  │
│  │  features)  │  │  executor)  │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      STRANDS LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Tension    │  │   Weave     │  │     Visualizer      │  │
│  │   Model     │  │  Decoder    │  │   (ASCII/Rich)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                       STORAGE                               │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │    Flat Files       │ OR │   Neo4j + Qdrant            │ │
│  │  (features.json)    │    │   (HoloLoom integration)    │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1: Bootstrap Script

```bash
# Drop bootstrap_harness.py in any project
python bootstrap_harness.py "Build a REST API with user auth and blog posts"

# Creates:
#   domain_memory/features.json
#   domain_memory/state.json
#   domain_memory/progress.log
#   domain_memory/tests/
#   CLAUDE.md
```

### Option 2: Manual Setup

```bash
mkdir -p domain_memory/tests domain_memory/strands src

# Create features.json with your features (status: "failing")
# Create state.json with constraints
# Create test files (assertions fail by default)
# Copy CLAUDE.md to project root
```

### Option 3: CLI

```bash
cd HoloLoom/domain_harness
python run_worker.py init "Build a REST API with user auth"
python run_worker.py run
python run_worker.py status
```

---

## File Schemas

### features.json

```json
{
  "metadata": {
    "version": 1,
    "schema": "features-1.0",
    "project": "my-api",
    "created": "2025-12-09T00:00:00Z"
  },
  "features": [
    {
      "id": "F001",
      "title": "User Authentication",
      "description": "JWT-based auth with login/logout",
      "status": "failing",
      "priority": 10,
      "depends_on": [],
      "acceptance_criteria": [
        "POST /login returns JWT",
        "Token validates on protected routes",
        "POST /logout invalidates token"
      ],
      "test_file": "tests/test_feature_F001.py",
      "notes": []
    },
    {
      "id": "F002",
      "title": "User Registration",
      "description": "Create new user accounts",
      "status": "failing",
      "priority": 8,
      "depends_on": ["F001"],
      "acceptance_criteria": [
        "POST /register creates user",
        "Validates email format",
        "Hashes password"
      ],
      "test_file": "tests/test_feature_F002.py",
      "notes": []
    }
  ]
}
```

**Status Values:**
- `pending` - Not started
- `failing` - Tests exist but fail
- `in_progress` - Currently being worked on
- `passing` - All tests pass
- `blocked` - Dependencies not met
- `skipped` - Intentionally skipped

### state.json

```json
{
  "project": {
    "name": "my-api",
    "version": "0.1.0",
    "language": "python"
  },
  "constraints": {
    "must_pass_all_tests": true,
    "atomic_work_unit": true,
    "max_features_per_session": 1,
    "custom": []
  },
  "rules_of_engagement": {
    "allowed_paths": ["domain_memory/", "src/", "tests/"],
    "test_command": "pytest -v",
    "forbidden": ["Delete features", "Skip tests", "Modify constraints"]
  },
  "environment": {
    "python_version": "3.11",
    "dependencies": ["pytest", "fastapi"]
  }
}
```

### progress.log

```
# progress.log
# Project: my-api
# Created: 2025-12-09T00:00:00Z

[INIT] Domain memory initialized. 5 features defined.

[2025-12-09T10:30:00Z] START F001: Beginning user authentication
[2025-12-09T10:45:00Z] PASS F001: JWT auth implemented, all tests green
[2025-12-09T11:00:00Z] START F002: Beginning user registration
[2025-12-09T11:20:00Z] FAIL F002: Email validation failing
[2025-12-09T11:35:00Z] PASS F002: Fixed regex, registration complete
```

### loom_state.json (Strands)

```json
{
  "strands": {
    "F001": {
      "feature_id": "F001",
      "tension": 0.5,
      "weft_position": 1.0,
      "is_regression": false,
      "consecutive_failures": 0,
      "last_updated": "2025-12-09T10:45:00Z"
    },
    "F002": {
      "feature_id": "F002",
      "tension": 6.0,
      "weft_position": 0.3,
      "is_regression": false,
      "consecutive_failures": 1,
      "last_updated": "2025-12-09T11:20:00Z"
    }
  },
  "total_tension": 6.5,
  "average_tension": 3.25,
  "is_stable": true,
  "is_critical": false,
  "is_complete": false
}
```

### Test File Template

```python
# tests/test_feature_F001.py
"""
Feature: F001 - User Authentication
Status: failing
"""
import pytest


class TestF001UserAuthentication:
    
    def test_login_returns_jwt_token(self):
        """POST /login should return a valid JWT token."""
        assert False, "Not yet implemented"
    
    def test_token_validates_on_protected_route(self):
        """Protected routes should validate JWT tokens."""
        assert False, "Not yet implemented"
```

---

## The Ritual

Every Claude Code session follows this exact pattern:

### 1. BOOT (Read State)

```bash
# Read in order:
cat domain_memory/features.json
cat domain_memory/progress.log
cat domain_memory/state.json
```

**Select ONE feature:**
1. Status must be `failing` or `pending`
2. All dependencies must be `passing`
3. Pick highest priority (or highest tension)

### 2. ACTION (Transform State)

```bash
# 1. Mark in progress
# Edit features.json: status → "in_progress"

# 2. Implement
# Write code in src/

# 3. Update tests
# Make tests actually test the implementation

# 4. Run tests
pytest tests/test_feature_F001.py -v

# 5. Update status
# If pass: status → "passing"
# If fail: status → "failing"
```

### 3. EXIT (Record + Vanish)

```bash
# 1. Append to progress.log
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PASS F001: Brief summary" >> domain_memory/progress.log

# 2. STOP
# Do NOT continue to next feature
```

### Decision Algorithm

```python
def select_feature(features):
    """Select ONE feature to work on."""
    passing = {f["id"] for f in features if f["status"] == "passing"}
    
    actionable = [
        f for f in features
        if f["status"] in ("failing", "pending")
        and all(dep in passing for dep in f["depends_on"])
    ]
    
    if not actionable:
        return None  # All done or all blocked
    
    # Sort by priority (higher first), then tension (higher first)
    actionable.sort(key=lambda f: (-f["priority"], -f.get("tension", 5)))
    
    return actionable[0]
```

---

## Strands Layer

### Tension Scale

| Range | Meaning | Action |
|-------|---------|--------|
| 0-2 | Stable | Passing, low maintenance |
| 3-5 | Active | In progress, normal work |
| 6-7 | Attention | Failing, needs focus |
| 8-9 | Priority | Blocked, stale, urgent |
| 10 | Critical | Regression, consecutive failures |

### Tension Modifiers

| Condition | Modifier |
|-----------|----------|
| Base: passing | 0.5 |
| Base: failing | 6.0 |
| Base: blocked | 8.0 |
| Dependency blocked | +2.0 |
| Regression (was passing) | +3.0 |
| Stale (24+ hours) | +1.5 |
| Recent progress | -1.0 |
| Per consecutive failure | +0.5 |

### Weave Motions

| Motion | Symbol | When | Tension Δ |
|--------|--------|------|-----------|
| `warp_tug` | ⟨⟩ | Tests pass (first time) | -0.5 |
| `weft_advance` | → | Tests stay green | -0.3 |
| `relaxation` | ∿ | Feature complete | -2.0 |
| `weft_wander` | ~ | Tests fail | +0.6 |
| `tension_spike` | ⚡ | Regression | +2.0 |
| `snag` | ✕ | Blocked | +1.0 |

### Visualization

```bash
# One-shot ASCII
python strands/visualizer.py

# Daemon mode (updates every 3s)
python strands/visualizer.py --daemon

# Rich terminal (requires: pip install rich)
python strands/visualizer.py --mode rich

# Matplotlib plot
python strands/visualizer.py --mode plot --save loom.png
```

Output:
```
╔══════════════════════════════════════════════════════════════╗
║                    LOOM TENSION STATE                        ║
╠══════════════════════════════════════════════════════════════╣
║ Total:   6.5  Avg:  3.2  Status: STABLE                      ║
╠──────────────────────────────────────────────────────────────╣
║ ✓ F001     ✓ [░░·····························] 0.5           ║
║ ✗ F002     ! [▓▓▓▓▓▓▓▓▓▓▓▓················] 6.0             ║
║ ○ F003     ○ [▒▒▒▒▒▒▒▒····················] 5.0             ║
╠──────────────────────────────────────────────────────────────╣
║ Legend: ░ Low  ▒ Medium  ▓ High  █ Critical                  ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Claude Code Integration

### CLAUDE.md (System Prompt)

Place this in your project root. Claude Code reads it automatically.

```markdown
# CLAUDE.md - Domain Harness Worker

You are a stateless worker. No memory between sessions. Everything must be in files.

## The Ritual

### BOOT
Read: features.json, progress.log, state.json
Select ONE feature: status=failing, dependencies satisfied, highest priority.

### ACTION
1. Set status to `in_progress`
2. Implement in `src/`
3. Update tests
4. Run: `pytest tests/test_feature_FXXX.py -v`
5. Set status to `passing` or `failing`

### EXIT
1. Append to progress.log: `[TIMESTAMP] PASS/FAIL FXXX: Summary`
2. **STOP.** Do not continue.

## Rules
- ONE feature per session
- Tests are truth
- Dependencies must pass first
- Record everything
```

### Example Session

```
Human: Continue working on the project.

Claude: Let me boot up and check the domain memory.

[Reads features.json]
Features:
- F001: User Auth [passing] ✓
- F002: Registration [failing] tension=6.0
- F003: Blog Posts [failing] depends=[F001] ✓
- F004: Comments [failing] depends=[F003] ✗

Actionable: F002 (no deps), F003 (deps satisfied)
Selected: F002 (higher tension)

[Implements F002, runs tests]

Result: 3 passed ✓

[Updates features.json: F002 → passing]
[Appends to progress.log]

Done. F002 complete. Stopping here - next session will handle F003.
```

---

## CLI Reference

```bash
# Initialize from prompt
python run_worker.py init "Build a REST API with user auth"

# Run one worker cycle
python run_worker.py run

# Run until complete (or max cycles)
python run_worker.py loop --max-cycles 50

# Check status
python run_worker.py status

# View progress log
python run_worker.py log

# Reset all features to failing
python run_worker.py reset --force

# Strands visualization
python strands/visualizer.py --daemon --mode rich
```

---

## API Reference

### Schema Classes

```python
from HoloLoom.domain_harness import (
    Feature,           # Dataclass for feature
    FeatureStatus,     # Enum: pending, failing, passing, etc.
    ProgressEntry,     # Dataclass for progress log entry
    ProgressAction,    # Enum: started, completed, failed, etc.
    DomainState,       # Project-level state
    ActionResult,      # Result from worker action
)
```

### Backend Classes

```python
from HoloLoom.domain_harness import (
    FlatFileBackend,      # Flat file storage
    FlatFileGenerator,    # Generate initial files
    Neo4jDomainProtocol,  # Neo4j storage (HoloLoom)
)

# Flat file usage
backend = FlatFileBackend(Path("./domain_memory"))
features = backend.load_features()
actionable = backend.get_actionable_features()
backend.update_feature_status("F001", "passing")
backend.append_progress("F001", "completed", "Implemented auth")
```

### Strands Classes

```python
from HoloLoom.domain_harness.strands import (
    StrandState,       # Per-feature symbolic state
    LoomState,         # Aggregate loom state
    TensionUpdater,    # Update tensions
    WeaveDecoder,      # Decode actions to motions
    WeaveEvent,        # Decoded event
    StrandsUpdater,    # High-level integration
    GitAutoCommit,     # Auto-commit helper
)

# Usage
from HoloLoom.domain_harness.strands import integrate_strands_with_worker

event = integrate_strands_with_worker(
    domain_memory_dir=Path("./domain_memory"),
    feature=feature,
    action_result=result,
    previous_status=FeatureStatus.FAILING
)
print(f"Weave: {event.symbol} {event.motion.value}")
print(f"Tension: {event.old_tension} → {event.new_tension}")
```

---

## File Index

```
HoloLoom/domain_harness/
├── __init__.py              # Public API exports
├── schema.py                # Feature, ProgressEntry, DomainState
├── protocol.py              # DomainProtocol, Boot/Action/Exit
├── initializer.py           # Prompt → features parser
├── worker.py                # DomainWorker, AgentDomainWorker
├── templates.py             # FlatFileGenerator, FlatFileBackend
├── run_worker.py            # CLI orchestrator
├── demo.py                  # In-memory demo
│
├── prompts/
│   ├── initializer_agent.md # LLM prompt for initializer
│   └── worker_agent.md      # LLM prompt for worker
│
├── strands/
│   ├── __init__.py          # Strands API
│   ├── tension_model.py     # TensionModel, LoomState
│   ├── weave_decoder.py     # WeaveDecoder, WeaveEvent
│   ├── update_strands.py    # StrandsUpdater
│   ├── visualizer.py        # ASCII/Rich/Plot visualization
│   └── git_autocommit.py    # GitAutoCommit
│
├── HANDOFF.md               # Full reference doc
├── CLAUDE.md                # Compact worker instructions
├── bootstrap_harness.py     # Quick-start bootstrap
├── README.md                # This file
└── context.json             # Session state
```

---

## The Loom Metaphor

```
Warp (vertical)  = Structure (features, dependencies)
Weft (horizontal) = Progress (implementation)
Tension          = Attention cost (0-10)
Shuttle          = Worker (one pass per session)
Weave            = Complete system

The pattern emerges from many passes, not one.
```

---

## Credits

Architecture derived from:
- Initializer/Worker pattern (ChatGPT handoff)
- HoloLoom semantic memory system
- Strands symbolic layer (tension as gas)

Built for Claude Code integration with stateless workers and persistent domain memory.
