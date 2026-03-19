# Domain Harness Documentation Index

## Quick Links

| Need | Document |
|------|----------|
| **Get started fast** | [bootstrap_harness.py](bootstrap_harness.py) |
| **Claude Code context** | [CLAUDE_CODE_HANDOFF.md](CLAUDE_CODE_HANDOFF.md) |
| **One-page reference** | [QUICKREF.md](QUICKREF.md) |
| **Full documentation** | [README.md](README.md) |
| **Strands (mythic layer)** | [strands/README.md](strands/README.md) |

---

## For Users

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [QUICKREF.md](QUICKREF.md) | One-page reference | Quick lookup during work |
| [CLAUDE.md](CLAUDE.md) | Worker instructions | Copy to project root |
| [bootstrap_harness.py](bootstrap_harness.py) | Quick-start script | Starting a new project |

## For Deep Context

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [README.md](README.md) | Complete documentation | Understanding the system |
| [HANDOFF.md](HANDOFF.md) | Full reference | Maximum context |
| [CLAUDE_CODE_HANDOFF.md](CLAUDE_CODE_HANDOFF.md) | Self-contained context | Drop into Claude Code |

## For Development

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [SESSION_TRANSCRIPT.md](SESSION_TRANSCRIPT.md) | Implementation history | Design decisions |
| [context.json](context.json) | Machine-readable summary | Programmatic access |

## Mythic Layer (Strands)

| Document | Purpose |
|----------|---------|
| [strands/README.md](strands/README.md) | Complete Strands documentation |

---

## Quick Start

### Option A: New Project (Fastest)

```bash
cd my-project
python path/to/bootstrap_harness.py "Build a REST API with user auth"
# Opens Claude Code, follow CLAUDE.md
```

### Option B: Existing Project

```bash
cp CLAUDE.md /path/to/project/
# Open Claude Code
```

### Option C: CLI Mode

```bash
python run_worker.py init "Your project"
python run_worker.py run
python run_worker.py status
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     PRACTICAL LAYER                         │
│  features.json ─► Worker ─► tests/ ─► progress.log          │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                     SYMBOLIC LAYER                          │
│  TensionModel ─► WeaveDecoder ─► LoomState ─► Visualizer    │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      MYTHIC LAYER                           │
│  Court of Threads │ Ritual Scheduler │ Archive Eternal      │
│  Conflict arbitration │ Cosmic cycles │ Eternal chronicle   │
└─────────────────────────────────────────────────────────────┘
```

---

## File Tree

```
domain_harness/
│
├── 📚 Documentation
│   ├── INDEX.md                 # This file
│   ├── README.md                # Complete docs
│   ├── HANDOFF.md               # Full reference
│   ├── SESSION_TRANSCRIPT.md    # Implementation history
│   ├── QUICKREF.md              # One-page reference
│   └── context.json             # Machine-readable
│
├── 🤖 Claude Code
│   ├── CLAUDE.md                # Worker instructions
│   ├── CLAUDE_CODE_HANDOFF.md   # Drop-in context
│   └── bootstrap_harness.py     # Quick-start
│
├── ⚙️ Implementation
│   ├── __init__.py              # Public API
│   ├── schema.py                # Data types
│   ├── protocol.py              # Lifecycle
│   ├── initializer.py           # Prompt parser
│   ├── worker.py                # Executor
│   ├── templates.py             # File backend
│   ├── run_worker.py            # CLI
│   └── demo.py                  # Demo
│
├── 🧵 Strands (Symbolic)
│   ├── README.md                # Strands documentation
│   ├── __init__.py              # Strands API
│   ├── tension_model.py         # Tension = Gas
│   ├── weave_decoder.py         # Action → Motion
│   ├── decode_weave.py          # Simple decoder
│   ├── update_strands.py        # Integration
│   ├── visualizer.py            # ASCII/Rich/Plot
│   └── git_autocommit.py        # Auto-commits
│
├── 🏛️ Strands (Mythic)
│   ├── loom_court.py            # Court of Threads
│   ├── ritual_scheduler.py      # Cosmic cycles
│   ├── mirror_integration.py    # Archive Eternal
│   ├── loom_diff.py             # Symbolic diff
│   └── weave_svg.py             # SVG knots
│
└── 📝 Agent Prompts
    ├── prompts/initializer_agent.md
    └── prompts/worker_agent.md
```

---

## Core Concept

```
STATELESS MODEL (Claude)
    │ reads/writes
    ▼
PERSISTENT MEMORY (domain_memory/)
    │
    ├── Practical: features.json, tests/
    │
    ├── Symbolic: loom_state.json, weave_history.json
    │
    └── Mythic: ritual_log.json, mirror_canon.json
```

**The agent is not the model. The agent is the transformation of one memory state into another.**

---

## Mythic Components Summary

### Court of Threads
When conflicts arise, the Court convenes for symbolic arbitration.
- Detects: regression clashes, tension deadlocks, dependency cycles
- Issues: arbitration, deferral, dissolution, split, merge, ritual verdicts

### Ritual Scheduler
The loom responds to celestial time.
- Lunar: new moon, full moon, quarters
- Solar: equinoxes, solstices
- Weekly: Monday opening, Friday closing

### Archive Eternal
Every significant event becomes a page in the eternal chronicle.
- Archives: weave motions, rituals, verdicts, milestones
- Queries: by symbol, feature, time range
- Renders: formatted chronicle

### Symbolic Diff
Compares memory states with narrative interpretation.
- Shifts: knot_tremor, warp_tug, fracture, emergence, dissolution

### SVG Knots
Visual representation of strand state.
- Tension → Color (blue=stable, red=critical)
- Complexity → Stroke width
- Failures → Opacity

---

## CLI Quick Reference

```bash
# Practical
python run_worker.py init "prompt"
python run_worker.py run
python run_worker.py status

# Symbolic
python strands/visualizer.py --daemon

# Mythic
python strands/ritual_scheduler.py
python strands/mirror_integration.py
python strands/loom_diff.py old.json new.json
python strands/weave_svg.py
```
