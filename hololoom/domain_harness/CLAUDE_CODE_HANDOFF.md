# Claude Code Handoff: Domain Harness

> Drop this entire file into a new Claude Code project as context.

---

## What This Is

A **persistent domain memory system** for stateless AI workers. You (Claude Code) have no memory between sessions. This system gives you memory through files.

**Core insight:** The agent is not the model. The agent is the transformation of one memory state into another.

**Strands = Gas:** Tension is the cost of incomplete work. High tension = needs attention. Your job is to drive tension to zero.

---

## The Files

```
domain_memory/
├── features.json      # What to build (source of truth)
├── state.json         # Project constraints
├── progress.log       # Append-only history
└── tests/             # Test files (truth = tests pass)
    └── test_feature_FXXX.py
```

---

## The Ritual

### BOOT (Every Session)

```bash
cat domain_memory/features.json   # What needs building
cat domain_memory/progress.log    # What's been done
```

**Select ONE feature:**
1. Status = `failing` or `pending`
2. All `depends_on` features = `passing`
3. Highest priority wins

### ACTION

```bash
# 1. Mark in progress
#    Edit features.json: status → "in_progress"

# 2. Implement
#    Write code in src/

# 3. Test
pytest tests/test_feature_FXXX.py -v

# 4. Update status
#    If pass: status → "passing"
#    If fail: status → "failing"
```

### EXIT

```bash
# Append to progress.log:
[2025-12-09T12:00:00Z] PASS F001: Brief description

# STOP. Do not continue to next feature.
```

---

## features.json Schema

```json
{
  "features": [
    {
      "id": "F001",
      "title": "Feature Name",
      "description": "What it does",
      "status": "failing",
      "priority": 10,
      "depends_on": [],
      "acceptance_criteria": ["criterion 1"]
    }
  ]
}
```

**Status:** `pending` | `failing` | `in_progress` | `passing` | `blocked`

---

## Test Files

Tests fail by default. Your job is to make them pass.

```python
# tests/test_feature_F001.py
def test_feature_works():
    assert False, "Not yet implemented"
```

---

## Decision Algorithm

```python
def select_feature(features):
    passing = {f["id"] for f in features if f["status"] == "passing"}
    actionable = [
        f for f in features
        if f["status"] in ("failing", "pending")
        and all(d in passing for d in f["depends_on"])
    ]
    return max(actionable, key=lambda f: f["priority"], default=None)
```

---

## Rules

1. **ONE feature per session.** Never continue to next.
2. **Tests are truth.** Failing tests = not done.
3. **Dependencies first.** Never work on blocked features.
4. **Record everything.** If not in files, it doesn't exist.
5. **You are stateless.** Context window = entire existence.

---

## Progress Log Format

```
[TIMESTAMP] PASS/FAIL FXXX: Brief summary
```

Examples:
```
[2025-12-09T10:30:00Z] PASS F001: Implemented JWT authentication
[2025-12-09T11:00:00Z] FAIL F002: Email validation edge case
[2025-12-09T11:15:00Z] PASS F002: Fixed regex, all tests pass
```

---

## Tension (Strands)

Tension = attention cost. Like gas in Ethereum.

| Tension | Meaning |
|---------|---------|
| 0-2 | Stable (passing) |
| 3-5 | Active (in progress) |
| 6-7 | Needs attention (failing) |
| 8-9 | Urgent (blocked, stale) |
| 10 | Critical (regression) |

High tension features get priority.

---

## Example Session

```
BOOT:
Features:
- F001 [passing] ✓
- F002 [failing] tension=6.0
- F003 [failing] depends=[F001] ✓
- F004 [failing] depends=[F003] ✗

Actionable: F002, F003
Selected: F002 (priority 10)

ACTION:
- Set F002 status → in_progress
- Implemented src/registration.py
- Updated tests
- pytest tests/test_feature_F002.py -v → 3 passed

EXIT:
- Set F002 status → passing
- Appended: [2025-12-09T12:00:00Z] PASS F002: User registration complete
- STOPPING.
```

---

## Quick Commands

```bash
# Check what's actionable
cat domain_memory/features.json | grep -A5 '"status": "failing"'

# Run specific test
pytest tests/test_feature_F001.py -v

# View progress
cat domain_memory/progress.log

# Get timestamp
date -u +%Y-%m-%dT%H:%M:%SZ
```

---

## Begin the Ritual

1. Read domain_memory/features.json
2. Select ONE failing feature with satisfied dependencies
3. Implement it
4. Run tests
5. Update status
6. Record in progress.log
7. STOP

The loom awaits.
