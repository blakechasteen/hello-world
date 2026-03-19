# Domain Harness Quick Reference

## The Ritual (3 Steps)

```
BOOT → Read features.json, select ONE failing feature
ACTION → Implement, test, update status
EXIT → Append progress.log, STOP
```

## File Locations

| File | Purpose |
|------|---------|
| `domain_memory/features.json` | Feature definitions + status |
| `domain_memory/progress.log` | Append-only history |
| `domain_memory/state.json` | Constraints |
| `domain_memory/tests/` | Test files |

## Status Values

```
pending → failing → in_progress → passing
                 ↓
              blocked (deps not met)
```

## Feature Selection

```python
actionable = [f for f in features 
              if f.status in ("failing", "pending")
              and all_deps_passing(f)]
target = max(actionable, key=lambda f: f.priority)
```

## Progress Log Entry

```
[2025-12-09T12:00:00Z] PASS F001: Brief summary
```

## Tension Scale (Strands)

```
0-2  ░░░░  Stable (passing)
3-5  ▒▒▒▒  Active (in progress)  
6-7  ▓▓▓▓  Attention (failing)
8-9  ████  Urgent (blocked)
10   ████  Critical (regression)
```

## Weave Motions

| Symbol | Motion | When |
|--------|--------|------|
| ⟨⟩ | warp_tug | Tests pass (first) |
| → | weft_advance | Tests stay green |
| ~ | weft_wander | Tests fail |
| ⚡ | tension_spike | Regression |
| ✕ | snag | Blocked |

## Rules

1. ONE feature per session
2. Tests are truth
3. Dependencies first
4. Record everything
5. You are stateless

## CLI

```bash
python run_worker.py init "prompt"  # Initialize
python run_worker.py run            # One cycle
python run_worker.py status         # Check status
python run_worker.py loop           # Until complete
python strands/visualizer.py        # Tension view
```

## Key Files

```
HoloLoom/domain_harness/
├── CLAUDE.md                # Worker instructions (inject)
├── CLAUDE_CODE_HANDOFF.md   # Full context (drop-in)
├── README.md                # Complete documentation
├── bootstrap_harness.py     # Quick-start any project
├── run_worker.py            # CLI orchestrator
└── strands/visualizer.py    # Tension visualization
```
