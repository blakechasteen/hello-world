# Strands: The Mythic Layer

> "The loom doesn't just execute—it writes its own story."

The Strands layer transforms the practical domain harness into a living symbolic system. Every worker action, every test result, every regression becomes part of an ongoing narrative.

---

## Table of Contents

1. [Core Components](#core-components)
2. [Tension Model](#tension-model)
3. [Weave Decoder](#weave-decoder)
4. [The Court of Threads](#the-court-of-threads)
5. [Ritual Scheduler](#ritual-scheduler)
6. [Archive Eternal (MirrorLog)](#archive-eternal)
7. [Symbolic Diff](#symbolic-diff)
8. [SVG Knot Visualization](#svg-knot-visualization)
9. [Integration Guide](#integration-guide)

---

## Core Components

```
strands/
├── tension_model.py      # Tension = Gas (attention cost)
├── weave_decoder.py      # Action → Symbolic motion (detailed)
├── decode_weave.py       # Simpler action decoder
│
├── loom_court.py         # Conflict arbitration ("Court of Threads")
├── ritual_scheduler.py   # Cosmic cycles (moon phases, solstices)
├── mirror_integration.py # Archive Eternal (persistent narrative)
│
├── loom_diff.py          # Symbolic memory diffing
├── weave_svg.py          # SVG knot visualization
│
├── update_strands.py     # Integration with domain harness
├── visualizer.py         # ASCII/Rich/Matplotlib
└── git_autocommit.py     # Safe auto-commits
```

---

## Tension Model

**File:** `tension_model.py`

Tension quantifies attention cost—like gas in Ethereum.

### Tension Scale

| Range | State | Symbol | Meaning |
|-------|-------|--------|---------|
| 0-2 | Stable | ░░ | Passing, low maintenance |
| 3-5 | Active | ▒▒ | In progress, normal work |
| 6-7 | Stressed | ▓▓ | Failing, needs focus |
| 8-9 | Critical | ██ | Blocked, stale, urgent |
| 10 | Max | ⚡ | Regression, breaking point |

### Tension Modifiers

```python
# Base tension by status
TENSION_BASE = {
    "passing": 0.5,
    "failing": 6.0,
    "pending": 4.0,
    "in_progress": 5.0,
    "blocked": 8.0,
}

# Modifiers
DEPENDENCY_BLOCKED = +2.0   # Waiting on unmet dependency
REGRESSION = +3.0           # Was passing, now failing
STALE = +1.5                # No progress in 24+ hours
RECENT_PROGRESS = -1.0      # Activity reduces tension
CONSECUTIVE_FAIL = +0.5     # Per consecutive failure
```

### StrandState

```python
@dataclass
class StrandState:
    feature_id: str
    tension: float = 5.0
    
    # Position in weave
    weft_position: float = 0.0   # Progress (0=start, 1=complete)
    
    # History
    regression_count: int = 0
    consecutive_fails: int = 0
    last_motion: Optional[WeaveMotion] = None
    last_updated: datetime
    
    # Flags
    is_stale: bool = False
```

### LoomState

```python
@dataclass
class LoomState:
    strands: Dict[str, StrandState]
    
    total_tension: float
    average_tension: float
    
    is_stable: bool      # All tensions < 7
    is_critical: bool    # Any tension > 9
    is_complete: bool    # All features passing
```

---

## Weave Decoder

**Files:** `weave_decoder.py`, `decode_weave.py`

Translates worker actions into symbolic weave motions.

### Weave Motions

| Motion | Symbol | Trigger | Tension Δ | Meaning |
|--------|--------|---------|-----------|---------|
| `warp_tug` | ⟨⟩ | Tests pass (first time) | -0.5 | "Tightening alignment" |
| `weft_advance` | → | Tests stay green | -0.3 | "Progress through weave" |
| `weft_shift` | ↔ | Refactor, no status change | 0 | "Lateral adjustment" |
| `weft_wander` | ~ | Tests fail | +0.6 | "Introducing slack" |
| `tension_release` | ∿ | Feature stabilizes | -2.0 | "Fabric settling" |
| `regression_spike` | ⚡ | Was passing, now failing | +2.0 | "Unraveling" |
| `knot_tie` | ⊛ | Dependency resolved | -0.5 | "Binding" |
| `thread_snap` | ✕ | Critical failure | +3.0 | "Rupture" |

### DecodedWeave

```python
@dataclass
class DecodedWeave:
    feature_id: str
    motion: WeaveMotion
    symbolic_effect: str      # Human-readable
    tension_delta: float
    new_tension: float
    was_regression: bool
    narrative: str            # Poetic description
```

### Narrative Templates

Each motion has rotating narratives:

```python
MOTION_NARRATIVES = {
    WeaveMotion.WARP_TUG: [
        "The thread pulls taut. Progress.",
        "Another strand finds its place in the pattern.",
        "The weave tightens. Coherence emerges."
    ],
    WeaveMotion.REGRESSION_SPIKE: [
        "The weave unravels. What was done is undone.",
        "A thread snaps back. Ground lost.",
        "The loom remembers. It will remember again."
    ],
    # ...
}
```

---

## The Court of Threads

**File:** `loom_court.py`

When conflicts arise, the Court convenes for symbolic arbitration.

> "Two threads cross and resist the loom. The Court must arbitrate."

### Conflict Types

| Type | Symbol | Description |
|------|--------|-------------|
| `regression_clash` | ⚔️ | Multiple regressions on same feature |
| `tension_deadlock` | 🔄 | Tension oscillating without resolution |
| `dependency_cycle` | 🌀 | Circular dependencies |
| `motion_contradiction` | ↔️ | Conflicting weave motions |
| `worker_dispute` | 👥 | Multiple workers claim same feature |

### Verdict Types

| Verdict | Meaning |
|---------|---------|
| `arbitration` | Court makes a decision |
| `deferral` | More information needed |
| `dissolution` | Feature should be removed |
| `split` | Feature should be split |
| `merge` | Features should be merged |
| `ritual` | Symbolic action required |

### Usage

```python
from HoloLoom.domain_harness.strands.loom_court import CourtOfThreads

court = CourtOfThreads()
verdicts = court.convene(events)

print(court.render_session(verdicts))
```

### Output

```
╔══════════════════════════════════════════════════════════════╗
║            THE COURT OF THREADS CONVENES                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Case: COURT-0001                                            ║
║  The Court has spoken. The thread bends to the verdict.      ║
║                                                              ║
║  Ruling: Split F001 into smaller units                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Ritual Scheduler

**File:** `ritual_scheduler.py`

The loom responds to celestial time.

> "The loom is not just a machine—it breathes with the cosmos."

### Cosmic Rituals

| Ritual | Symbol | Tension Mod | Trigger |
|--------|--------|-------------|---------|
| `new_moon` | 🌑 | -0.5 | New moon phase |
| `full_moon` | 🌕 | 0.0 | Full moon phase |
| `waxing_quarter` | 🌓 | +0.3 | First quarter |
| `waning_quarter` | 🌗 | -0.3 | Last quarter |
| `vernal_equinox` | 🌱 | 0.0 | March 20-21 |
| `summer_solstice` | 🌞 | +0.5 | June 20-21 |
| `autumnal_equinox` | 🍂 | 0.0 | Sept 22-23 |
| `winter_solstice` | ❄️ | -0.5 | Dec 21-22 |
| `monday_opening` | 🌅 | +0.2 | Monday 6-9 AM |
| `friday_closing` | 🌆 | -0.2 | Friday 4-7 PM |

### Proclamations

```python
RITUAL_PROCLAMATIONS = {
    RitualType.NEW_MOON: [
        "🌑 New Moon: The loom begins anew. Seeds planted in darkness.",
        "🌑 Dark of the Moon: A time for quiet planning.",
    ],
    RitualType.WINTER_SOLSTICE: [
        "❄️ Winter Solstice: Longest night. The loom dreams.",
        "🌙 Midwinter: In deepest dark, the light returns.",
    ],
}
```

### Usage

```python
from HoloLoom.domain_harness.strands.ritual_scheduler import RitualScheduler

scheduler = RitualScheduler()
rituals = scheduler.check_and_execute()

for ritual in rituals:
    print(scheduler.render_ritual(ritual))
```

### Moon Phase Calculator

```python
from HoloLoom.domain_harness.strands.ritual_scheduler import get_moon_info

moon = get_moon_info()
print(f"Phase: {moon['name']} ({moon['phase']:.2f})")
print(f"Illumination: {moon['illumination']}")
```

---

## Archive Eternal

**File:** `mirror_integration.py`

Every significant event becomes a page in the eternal chronicle.

> "The loom doesn't just execute—it writes its own story."

### Fragment Types

| Type | Description |
|------|-------------|
| `weave` | Worker action / weave motion |
| `ritual` | Cosmic/scheduled ritual |
| `verdict` | Court verdict |
| `omen` | Warning / prediction |
| `milestone` | Significant achievement |
| `custom` | User-defined entry |

### CanonFragment

```python
@dataclass
class CanonFragment:
    fragment_id: str
    fragment_type: FragmentType
    timestamp: datetime
    narrative: str          # Full narrative text
    
    feature_ids: List[str]  # Related features
    symbols: List[str]      # Symbolic keywords
    tags: List[str]
    
    related_fragments: List[str]  # Cross-references
```

### Usage

```python
from HoloLoom.domain_harness.strands.mirror_integration import MirrorLog

mirror = MirrorLog()

# Archive a weave motion
mirror.archive_weave(decoded_weave)

# Archive a ritual
mirror.archive_ritual(ritual)

# Archive a verdict
mirror.archive_verdict(verdict)

# Custom entry
mirror.archive_custom(
    title="System Initialization",
    narrative="The Archive Eternal awakens.",
    tags=["init"]
)

# Query
fragments = mirror.query_by_symbol("regression")
recent = mirror.recent(limit=10)

# Render chronicle
print(mirror.render_chronicle())
```

### Chronicle Output

```
╔══════════════════════════════════════════════════════════════╗
║              THE ARCHIVE ETERNAL - CHRONICLE                 ║
╚══════════════════════════════════════════════════════════════╝

On 2025-12-09 10:30, the Archive witnessed a turning back.
Thread F001 unraveled what was woven.

    "The loom remembers. It will remember again."

Tension: 0.5 → 3.5 (↑ SPIKE)

────────────────────────────────────────────────────────────────
```

---

## Symbolic Diff

**File:** `loom_diff.py`

Compares two memory states and generates symbolic interpretation.

> "The diff isn't just 'field A changed to B'—it's 'the weave shifted, tension released, a knot tremored.'"

### Symbolic Shifts

| Shift | Symbol | Condition |
|-------|--------|-----------|
| `knot_tremor` | ⚡ | Major disruption (Δ > 1.0) |
| `warp_tug` | ⟡ | Realignment (Δ > 0.3) |
| `soft_pull` | ~ | Minor adjustment (Δ > 0) |
| `stillness` | • | No change |
| `slack_release` | ↓ | Minor relaxation (Δ < 0) |
| `fracture` | ❄ | Major recoil (Δ < -1.0) |
| `status_flip` | ◐ | Pass/fail transition |
| `emergence` | ✦ | New feature appeared |
| `dissolution` | ✧ | Feature removed |

### Usage

```python
from HoloLoom.domain_harness.strands.loom_diff import diff_states

diff = diff_states("snapshot_old.json", "snapshot_new.json")
print(diff.render_ascii())
```

### Output

```
╔══════════════════════════════════════════════════════════════╗
║              SYMBOLIC MEMORY DIFF                            ║
║          2025-12-09 10:30:00                                 ║
╠══════════════════════════════════════════════════════════════╣
║ F001     Δ -0.50 ↓ Slack release: pressure fades.            ║
║ F002     Δ +1.20 ⚡ Knot tremor: weave disrupted.            ║
║ F003     Δ  0.00 • Stillness: the thread rests.              ║
╠══════════════════════════════════════════════════════════════╣
║  Total Tension: 15.0 → 15.7 (Δ+0.7)                          ║
╚══════════════════════════════════════════════════════════════╝

↑ Tension builds. The pattern tightens.
```

---

## SVG Knot Visualization

**File:** `weave_svg.py`

Renders strand tensors as living SVG knots.

> "The knot is the visual soul of the strand."

### Knot Types

| Type | Complexity | When Used |
|------|------------|-----------|
| Trefoil | Low | Default |
| Figure-Eight | Medium | 3+ regressions |
| Lissajous | High | 5+ consecutive fails |
| Spiral | Variable | Custom |

### Visual Encoding

| Property | Encodes |
|----------|---------|
| **Color** | Tension (blue=stable → red=critical) |
| **Stroke Width** | Complexity + regressions |
| **Opacity** | Consecutive failures (fades with failures) |
| **Glow** | Critical tension (>9) |
| **Rotation** | Last motion type |

### Usage

```python
from HoloLoom.domain_harness.strands.weave_svg import render_strand_knot, render_loom_grid

# Single knot
render_strand_knot(strand_state, Path("knot_F001.svg"))

# Full loom
render_loom_grid(loom_state, Path("loom.svg"), title="Project Loom")

# From tensor (GPT handoff format)
tensor = [5.0, 3.0, 2, 1, 0]  # [tension, complexity, deps, retries, regressions]
tensor_to_knot_svg(tensor, "F001", Path("knot.svg"))
```

### Color Mapping

```python
def tension_to_color(tension):
    # 0 → HSL(180, 50%, 60%)  Cyan (stable)
    # 5 → HSL(90, 65%, 50%)   Green-yellow
    # 10 → HSL(0, 80%, 40%)   Red (critical)
```

---

## Integration Guide

### Full Integration Flow

```python
from HoloLoom.domain_harness import FlatFileBackend
from HoloLoom.domain_harness.strands import (
    TensionUpdater, WeaveDecoder, StrandsUpdater,
    integrate_strands_with_worker
)
from HoloLoom.domain_harness.strands.loom_court import CourtOfThreads
from HoloLoom.domain_harness.strands.ritual_scheduler import RitualScheduler
from HoloLoom.domain_harness.strands.mirror_integration import MirrorLog

# After worker run
def post_worker_hook(feature, action_result, previous_status):
    # 1. Update strands
    event = integrate_strands_with_worker(
        domain_memory_dir=Path("./domain_memory"),
        feature=feature,
        action_result=action_result,
        previous_status=previous_status
    )
    
    # 2. Archive to eternal chronicle
    mirror = MirrorLog()
    mirror.archive_weave(event)
    
    # 3. Check for rituals
    scheduler = RitualScheduler()
    rituals = scheduler.check_and_execute()
    for ritual in rituals:
        mirror.archive_ritual(ritual)
    
    # 4. Check for conflicts
    court = CourtOfThreads()
    verdicts = court.convene(recent_events)
    for verdict in verdicts:
        mirror.archive_verdict(verdict)
    
    return event
```

### Minimal Integration

```python
from HoloLoom.domain_harness.strands import integrate_strands_with_worker

# Just update tension and log motion
event = integrate_strands_with_worker(domain_dir, feature, result)
print(f"Motion: {event.symbol} {event.motion.value}")
print(f"Tension: {event.old_tension} → {event.new_tension}")
```

### CLI Commands

```bash
# Visualize tension
python strands/visualizer.py --daemon --mode rich

# Check rituals
python strands/ritual_scheduler.py

# View archive
python strands/mirror_integration.py

# Diff states
python strands/loom_diff.py before.json after.json

# Generate knots
python strands/weave_svg.py
```

---

## The Loom Metaphor

```
Warp (vertical)   = Structure (features, dependencies)
Weft (horizontal) = Progress (implementation)
Tension           = Attention cost
Shuttle           = Worker (one pass per session)
Knot              = Visual soul of a strand
Weave             = Complete system
Court             = Conflict resolution
Ritual            = Cosmic alignment
Archive           = Eternal memory
```

The pattern emerges from many passes of the shuttle, guided by cosmic rhythms, arbitrated by the Court, and preserved forever in the Archive Eternal.

---

## File Quick Reference

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `tension_model.py` | Tension calculation | `TensionModel`, `StrandState`, `LoomState` |
| `weave_decoder.py` | Action → Motion (detailed) | `WeaveDecoder`, `WeaveEvent`, `WeaveHistory` |
| `decode_weave.py` | Action → Motion (simple) | `DecodedWeave`, `decode_worker_result` |
| `loom_court.py` | Conflict resolution | `CourtOfThreads`, `Conflict`, `Verdict` |
| `ritual_scheduler.py` | Cosmic cycles | `RitualScheduler`, `moon_phase`, `solar_event` |
| `mirror_integration.py` | Archive Eternal | `MirrorLog`, `CanonFragment`, `ArchiveEternal` |
| `loom_diff.py` | Symbolic diffing | `diff_states`, `SymbolicDiff`, `FeatureDiff` |
| `weave_svg.py` | SVG visualization | `KnotRenderer`, `LoomRenderer` |
| `update_strands.py` | Harness integration | `StrandsUpdater`, `integrate_strands_with_worker` |
| `visualizer.py` | Terminal visualization | `ASCIIRenderer`, `run_daemon` |
| `git_autocommit.py` | Auto-commits | `GitAutoCommit`, `git_commit_all` |
