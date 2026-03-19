# Claude Code Handoff: Domain Harness with Strands

## The Core Insight

**The agent is not the model. The agent is the transformation of one memory state into another.**

You (Claude Code) are a stateless worker. You have no memory between invocations. Your context window is your entire existence. When you finish, you vanish.

But the *work persists* in files. The domain memory outlives you.

---

## Strands = Gas

Think of **Strands** like gas in Ethereum:

| Concept | Ethereum | Domain Harness |
|---------|----------|----------------|
| Resource | Gas | Tension |
| High = | Expensive operation | Needs attention |
| Low = | Cheap/complete | Stable, done |
| Zero = | Free | Feature complete |
| Spike = | Failed tx | Regression |

**Tension is the cost of incomplete work.** Every failing test, every blocked dependency, every regression *increases tension*. Your job is to reduce tension by making tests pass.

---

## The Ritual

Every Claude Code session follows this exact pattern:

### 1. BOOT (Read State)

```
Read these files IN ORDER:
1. domain_memory/features.json     → What needs to be built
2. domain_memory/progress.log      → What's been done
3. domain_memory/state.json        → Constraints and rules
4. domain_memory/strands/loom_state.json → Current tension (optional)
```

From this, determine:
- Which features are FAILING
- Which have satisfied dependencies
- Which has highest priority (or highest tension)
- Select **ONE** feature to work on

### 2. ACTION (Transform State)

```
1. Mark feature as in_progress (update features.json)
2. Implement the feature (write code)
3. Run tests: pytest tests/test_feature_FXXX.py -v
4. If tests pass → status = "passing"
5. If tests fail → status = "failing"
6. Update features.json with new status
```

**RULES:**
- ONE feature per session
- Tests are the source of truth
- No changes outside domain_memory/ and src/
- No inventing new features

### 3. EXIT (Record + Vanish)

```
1. Append to progress.log:
   [TIMESTAMP] PASS/FAIL FXXX: Brief summary

2. Update strands (optional):
   - If PASS: tension decreases
   - If FAIL: tension increases
   - If REGRESSION: tension spikes

3. Commit (if git available):
   git add -A && git commit -m "PASS F001: Implemented auth"

4. STOP. Do not continue to next feature.
```

---

## File Structure

```
project/
├── domain_memory/
│   ├── features.json          # Feature definitions + status
│   ├── state.json             # Project constraints
│   ├── progress.log           # Append-only history
│   ├── strands/
│   │   ├── loom_state.json    # Tension per feature
│   │   └── weave_history.json # Symbolic event log
│   └── tests/
│       ├── test_feature_F001.py
│       ├── test_feature_F002.py
│       └── conftest.py
│
└── src/                       # Your implementation code
    └── ...
```

---

## features.json Schema

```json
{
  "metadata": {
    "version": 1,
    "schema": "features-1.0",
    "created": "2025-12-09T00:00:00Z"
  },
  "features": [
    {
      "id": "F001",
      "title": "User Authentication",
      "description": "JWT-based auth with login/logout endpoints",
      "status": "failing",
      "priority": 10,
      "depends_on": [],
      "acceptance_criteria": [
        "POST /login returns JWT token",
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

**Status values:** `pending`, `failing`, `in_progress`, `passing`, `blocked`, `skipped`

---

## state.json Schema

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
    "dependencies": ["pytest", "fastapi", "pydantic"]
  }
}
```

---

## progress.log Format

```
# progress.log
# Project: my-api
# Created: 2025-12-09T00:00:00Z

[INIT] Domain memory initialized. 5 features defined.

[2025-12-09T10:30:00Z] START F001: Beginning user authentication
[2025-12-09T10:45:00Z] PASS F001: JWT auth implemented, all tests green
[2025-12-09T11:00:00Z] START F002: Beginning user registration  
[2025-12-09T11:20:00Z] FAIL F002: Email validation failing, regex issue
[2025-12-09T11:35:00Z] PASS F002: Fixed regex, registration complete
```

---

## Strands: loom_state.json

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

**Tension scale:**
- 0-2: Stable (passing, low maintenance)
- 3-5: Active (in progress)
- 6-7: Needs attention (failing)
- 8-9: High priority (blocked, stale)
- 10: Critical (regression, consecutive failures)

---

## Test File Template

Each feature gets a test file that **fails by default**:

```python
# tests/test_feature_F001.py
"""
Feature: F001 - User Authentication
Status: failing

Acceptance Criteria:
- POST /login returns JWT token
- Token validates on protected routes
- POST /logout invalidates token
"""
import pytest


class TestF001UserAuthentication:
    """Tests for user authentication feature."""
    
    def test_login_returns_jwt_token(self):
        """POST /login should return a valid JWT token."""
        # TODO: Implement
        assert False, "Not yet implemented"
    
    def test_token_validates_on_protected_route(self):
        """Protected routes should validate JWT tokens."""
        # TODO: Implement
        assert False, "Not yet implemented"
    
    def test_logout_invalidates_token(self):
        """POST /logout should invalidate the token."""
        # TODO: Implement
        assert False, "Not yet implemented"
```

---

## Decision Algorithm

When you boot, use this to select your target:

```python
def select_feature(features):
    """Select ONE feature to work on."""
    
    # 1. Filter to actionable features
    actionable = []
    passing_ids = {f["id"] for f in features if f["status"] == "passing"}
    
    for f in features:
        if f["status"] in ("passing", "skipped", "in_progress"):
            continue
        
        # Check dependencies
        deps_satisfied = all(d in passing_ids for d in f["depends_on"])
        if not deps_satisfied:
            continue
            
        actionable.append(f)
    
    if not actionable:
        return None  # All done or all blocked
    
    # 2. Sort by priority (higher first), then by tension (higher first)
    actionable.sort(key=lambda f: (-f["priority"], -f.get("tension", 5)))
    
    # 3. Return the top one
    return actionable[0]
```

---

## Weave Motions (Symbolic Events)

After each action, record what happened symbolically:

| Action | Motion | Symbol | Tension Δ |
|--------|--------|--------|-----------|
| Tests pass (first time) | `warp_tug` | ⟨⟩ | -0.5 |
| Tests pass (staying green) | `weft_advance` | → | -0.3 |
| Feature complete | `relaxation` | ∿ | -2.0 |
| Tests fail | `weft_wander` | ~ | +0.6 |
| Was passing, now failing | `tension_spike` | ⚡ | +2.0 |
| Blocked by dependency | `snag` | ✕ | +1.0 |

---

## Commands Reference

### Initialize (as Initializer Agent)

```bash
# You receive a prompt like:
# "Build a REST API with user auth, blog posts, and comments"

# Your job:
1. Parse into features with dependencies
2. Create domain_memory/features.json (all status: "failing")
3. Create domain_memory/state.json
4. Create domain_memory/progress.log
5. Create test files (all assertions fail)
6. DO NOT write implementation code
```

### Run (as Worker Agent)

```bash
# Boot
cat domain_memory/features.json
cat domain_memory/progress.log
cat domain_memory/state.json

# Select target (use decision algorithm)
# Say: "Selected F002: User Registration (tension: 6.0)"

# Implement
# Write code in src/
# Update tests to actually test the implementation

# Test
pytest tests/test_feature_F002.py -v

# Update state
# Edit features.json: F002 status → "passing" or "failing"
# Append to progress.log

# Exit
# STOP HERE. Do not continue to next feature.
```

### Check Status

```bash
# Quick status check
cat domain_memory/features.json | jq '.features[] | {id, status, priority}'

# Tension overview
cat domain_memory/strands/loom_state.json | jq '{total_tension, average_tension, is_stable}'
```

---

## Example Session

```
BOOT:
Reading domain_memory/features.json...
- F001: User Auth [passing] ✓
- F002: Registration [failing] tension=6.0
- F003: Blog Posts [failing] tension=5.0, depends=[F001] ✓
- F004: Comments [failing] tension=5.0, depends=[F003] ✗

Actionable: F002 (no deps), F003 (deps satisfied)
Selecting: F002 (higher tension: 6.0 vs 5.0)

ACTION:
Implementing F002: User Registration
- Created src/auth/registration.py
- Updated tests/test_feature_F002.py
- Running: pytest tests/test_feature_F002.py -v

Result: 3 passed ✓

EXIT:
Updated features.json: F002 status → "passing"
Appended to progress.log:
  [2025-12-09T12:00:00Z] PASS F002: Registration implemented with email validation

Weave motion: warp_tug ⟨⟩
Tension: 6.0 → 0.5

STOPPING. Next worker will handle F003.
```

---

## Critical Rules

1. **ONE feature per session.** Never continue to the next.

2. **Tests are truth.** If tests fail, the feature is not done.

3. **Dependencies matter.** Never work on a feature whose dependencies aren't passing.

4. **Tension guides priority.** High tension = needs attention first.

5. **You are stateless.** Everything must be in files. If it's not written down, it doesn't exist.

6. **Record everything.** Progress log is append-only history.

7. **Atomic commits.** Each session = one feature = one commit.

---

## File Locations (HoloLoom)

If using the full HoloLoom implementation:

```
HoloLoom/domain_harness/
├── __init__.py           # API exports
├── schema.py             # Feature, ProgressEntry dataclasses
├── protocol.py           # Boot/Action/Exit lifecycle
├── initializer.py        # Prompt → features parser
├── worker.py             # Worker implementation
├── templates.py          # File generators + FlatFileBackend
├── run_worker.py         # CLI orchestrator
├── prompts/
│   ├── initializer_agent.md
│   └── worker_agent.md
└── strands/
    ├── tension_model.py  # TensionModel, LoomState
    ├── weave_decoder.py  # WeaveMotion, WeaveEvent
    ├── update_strands.py # StrandsUpdater
    ├── visualizer.py     # ASCII/Rich visualization
    └── git_autocommit.py # Safe auto-commits
```

---

## Quick Start

```bash
# 1. Initialize a new project
cd my-project
mkdir -p domain_memory/tests domain_memory/strands src

# 2. Create features.json with your features (status: "failing")
# 3. Create state.json with constraints
# 4. Create test files (assertions fail)
# 5. Initialize progress.log

# 6. Run worker loop
# Each Claude Code session: Boot → Select → Implement → Test → Exit

# 7. Repeat until all features passing (total_tension → 0)
```

---

## The Loom Metaphor

```
Warp (vertical) = Structure (features, dependencies)
Weft (horizontal) = Progress (implementation)
Tension = Attention cost
Shuttle = Worker (one pass per session)
Weave = Complete, working system

High tension → Thread needs attention
Low tension → Thread at rest
Regression → "What was woven has unraveled"
Completion → "The thread finds its place"
```

Every worker pass is a motion of the shuttle through the loom. The pattern emerges from many passes, not one.

---

## Handoff Complete

You now have everything needed to:
1. Initialize domain memory from a prompt
2. Run as a stateless worker
3. Track tension as attention cost
4. Record symbolic weave events
5. Make atomic progress toward completion

**The loom awaits. Begin the ritual.**
