# Jenny Generative UI System - Capabilities Summary

**Status**: Production Ready (December 2025)
**Location**: `HoloLoom/visualization/jenny_*.py`
**Total Code**: ~5,200 lines across 9 modules
**Tests**: 257 tests (239 unit + 18 integration)

## Philosophy

> "Disposable pixels, durable decisions."
>
> Every UI panel is temporary and should dissolve when no longer needed.
> Every decision that generated it is permanent and fully replayable.

## Architecture Overview

```
                    ┌──────────────────────┐
                    │    JennyRuntime      │
                    │   (Orchestrator)     │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
   ┌─────────┐          ┌───────────┐          ┌──────────┐
   │Compiler │          │ Lifecycle │          │ Renderer │
   │         │◄────────►│  Manager  │◄────────►│          │
   └─────────┘          └───────────┘          └──────────┘
        │                      │                      │
        │                      ▼                      │
        │               ┌───────────┐                 │
        │               │  Action   │                 │
        └──────────────►│  Handler  │◄────────────────┘
                        └───────────┘
                               │
                               ▼
                        ┌───────────┐
                        │  Spec     │
                        │  Ledger   │ ← Complete Provenance
                        └───────────┘
```

## Core Capabilities

### 1. Panel Types (12 Types)

Jenny supports 12 panel types for visualizing different aspects of query responses:

| Panel Type | Purpose | Auto-Detection Trigger |
|------------|---------|------------------------|
| **TEXT** | Plain text response | Default fallback |
| **CODE** | Code with syntax highlighting | Response contains ``` blocks |
| **GRAPH** | Knowledge graph visualization | >2 threads activated |
| **CONFIDENCE** | Confidence gauge | Confidence < 0.7 |
| **TIMELINE** | Stage timing waterfall | >3 pipeline stages |
| **METRIC** | Single metric display | Duration > 100ms |
| **MEMORY_MAP** | Memory activation map | On request |
| **TABLE** | Tabular data | Structured data detected |
| **REASONING** | Step-by-step reasoning chain | Multi-step reasoning |
| **SOURCES** | Source attribution list | Multiple sources used |
| **ACTIONS** | Action buttons/forms | Interactive workflows |
| **WHY** | Meta-panel explaining UI choice | "Why this UI?" query |

### 2. Lifecycle Management

Panels follow a 4-stage lifecycle with automatic state transitions:

```
compile() → NASCENT → (user pins) → STABLE
                   ↘           ↙
                (timeout/superseded)
                       ↓
                  DISSOLVING
                       ↓
                   ARCHIVED
```

| Stage | Duration | Behavior |
|-------|----------|----------|
| **NASCENT** | 300ms spawn animation | Temporary, auto-dissolves after timeout |
| **STABLE** | Until dismissed | User-pinned, persists across sessions |
| **DISSOLVING** | 300ms animation | Fading out, can be cancelled |
| **ARCHIVED** | Permanent | In SpecLedger, fully replayable |

**Dissolution Triggers**:
- `MANUAL` - User clicked dismiss
- `TIMEOUT` - Idle timeout (default: 5 minutes)
- `CONTEXT_SHIFT` - Query topic changed significantly
- `SUPERSEDED` - New panel replaced this one
- `ORPHAN` - Parent Spacetime deleted
- `MEMORY` - Memory pressure forced dissolution

### 3. Actions System

Users can interact with panels through 4 standard actions:

| Action | Effect | Lifecycle Impact |
|--------|--------|------------------|
| **PIN** | Make panel persistent | NASCENT → STABLE |
| **DISMISS** | Remove panel | → DISSOLVING → ARCHIVED |
| **WHY** | Explain UI choice | Creates meta-panel |
| **COPY** | Copy content to clipboard | None |

Custom actions can be added per-panel for domain-specific workflows.

### 4. Data Binding Modes

Three modes for handling dynamic data:

| Mode | Update Behavior | Use Case |
|------|-----------------|----------|
| **STATIC** | One-time render, no updates | Simple responses |
| **REACTIVE** | Re-render on data changes | Dashboard metrics |
| **STREAMING** | SSE/WebSocket live updates | Real-time data |

### 5. Multi-Target Rendering

Jenny panels can be rendered to multiple formats:

| Target | Implementation | Use Case |
|--------|----------------|----------|
| **HTML** | Static HTML + CSS | Dashboards, web UI |
| **Terminal** | ASCII art | CLI debugging |
| **JSON** | Structured data | API responses |
| **React** | Component props | (Future) Web apps |
| **AR** | Spatial overlays | (Future) AR glasses |

### 6. Provenance Tracking (SpecLedger)

Complete audit trail of all panels and transitions:

- Every panel logged with full context
- Every transition tracked with timestamp
- Session replay capability
- Query by time range, panel type, or spacetime ID

### 7. MRF-Enhanced Intelligence (Phase 2)

**Status**: ✅ Implemented (December 2025)
**Location**: `HoloLoom/visualization/jenny_mrf.py`

Jenny integrates with the Metaprompt Refinement Framework for intelligent panel generation:

#### Thompson Sampling Panel Type Selection

Learns which panel types work best for different query types using Bayesian learning:

```python
from HoloLoom.visualization.jenny_mrf import JennyMRFCompiler

# Create MRF-enhanced compiler with learning
compiler = JennyMRFCompiler(enable_learning=True)

# Compile with MRF enhancement
specs = await compiler.compile(spacetime)

# After user interaction (pin/dismiss), update learning
compiler.update_learning("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)

# View learning statistics
stats = compiler.get_learning_statistics()
print(f"Total selections: {stats['total_selections']}")
print(f"Best for factual: {stats['best_panel_types'].get('factual')}")
```

**Learning Algorithm**:
- Beta(α, β) priors per (query_type, panel_type) pair
- Success: α ← α + confidence
- Failure: β ← β + (1 - confidence)
- Expected value: E[X] = α / (α + β)
- Exploration bonus for encouraging diversity

#### MRF-Enhanced WHY Panels

Uses ELEGANCE strategy for clear, beautiful explanations:

```python
from HoloLoom.visualization.jenny_mrf import generate_why_panel_mrf

# Generate enhanced WHY panel content
content = await generate_why_panel_mrf(
    spec=jenny_spec,
    spacetime=spacetime,
    mrf_strategy="elegance"  # Clarity → Simplicity → Beauty
)
```

| Component | MRF Enhancement | Benefit |
|-----------|-----------------|---------|
| Panel type selection | Thompson Sampling learning | Learn optimal panel per query type |
| WHY meta-panel | ELEGANCE strategy | Clarity → Simplicity → Beauty |
| REASONING panel | VERIFY strategy | Accuracy + Completeness + Consistency |
| Query analysis | MRF prompt analysis | Better intent detection |

#### State Persistence

Learning state persists to disk for continuous improvement:

```python
from HoloLoom.visualization.jenny_mrf import PanelTypeLearner

# Create learner with persistence
learner = PanelTypeLearner(persist_path="./jenny_learning.json")

# State auto-saves after updates
learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.85)

# Load existing state on next startup
learner2 = PanelTypeLearner(persist_path="./jenny_learning.json")
# Continues from previous learning
```

## Quick Start

### Basic Usage

```python
from HoloLoom.visualization import JennyRuntime

async with JennyRuntime() as jenny:
    # Generate panel from query
    panel = await jenny.ask("What is Thompson Sampling?")
    print(panel.html)  # Rendered HTML

    # User pins the panel
    result = await jenny.act(panel, "pin")

    # View complete history
    history = jenny.history()
```

### Integration with WeavingOrchestrator

```python
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator

config = Config.fused()
config.enable_jenny = True

async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(Query(text="What is Thompson Sampling?"))

    # Panel automatically generated based on response
    jenny_panel = spacetime.metadata.get('jenny_panel')
    print(jenny_panel)
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_jenny` | bool | `False` | Enable Jenny panel generation |
| `jenny_persist_path` | str | `"./jenny_specs"` | Path for spec ledger persistence |
| `jenny_default_renderer` | str | `"html"` | Default render target (html/terminal/json) |
| `jenny_max_panels_per_query` | int | `6` | Max panels generated per query |
| `jenny_auto_lifecycle` | bool | `True` | Auto-transition NASCENT → STABLE |
| `jenny_cleanup_interval` | float | `60.0` | Cleanup interval in seconds |

## Panel Type Detection Thresholds

The `_detect_jenny_panel_type()` method uses these thresholds:

| Threshold | Value | Panel Type |
|-----------|-------|------------|
| `JENNY_CONFIDENCE_THRESHOLD` | 0.7 | Below → CONFIDENCE |
| `JENNY_THREADS_THRESHOLD` | 2 | Above → GRAPH |
| `JENNY_STAGES_THRESHOLD` | 3 | Above → TIMELINE |
| `JENNY_DURATION_THRESHOLD_MS` | 100 | Above → METRIC |

**Detection Priority** (first match wins):
1. CODE: Response contains ``` code blocks
2. GRAPH: >2 threads activated in trace
3. CONFIDENCE: Confidence < 0.7
4. TIMELINE: >3 stage durations in trace
5. METRIC: Duration > 100ms
6. TEXT: Default fallback

## Module Reference

| Module | Lines | Purpose |
|--------|-------|---------|
| `jenny_spec.py` | ~450 | Core data structures (JennySpec, enums) |
| `jenny_compiler.py` | ~550 | Query → JennySpec compilation |
| `jenny_lifecycle.py` | ~500 | Panel lifecycle state machine |
| `jenny_actions.py` | ~400 | Action handler system |
| `jenny_streaming.py` | ~350 | Live data bindings |
| `jenny_renderer.py` | ~600 | Multi-target rendering |
| `jenny_runtime.py` | ~700 | Unified orchestrator |
| `spec_ledger.py` | ~500 | Provenance tracking |
| `jenny_mrf.py` | ~745 | MRF integration + Thompson Sampling learning |

## Test Coverage

| Test File | Tests | Focus |
|-----------|-------|-------|
| `test_jenny.py` | 52 | Core spec + compiler |
| `test_jenny_week2.py` | 47 | Lifecycle + renderer |
| `test_jenny_week3.py` | 50 | Actions + streaming |
| `test_jenny_week4.py` | 49 | Runtime integration |
| `test_jenny_mrf.py` | 41 | MRF integration + Thompson Sampling |
| `test_jenny_orchestrator.py` | 18 | WeavingOrchestrator integration |

**Total**: 257 tests (239 unit + 18 integration), all passing

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Panel compilation | <5ms | Query analysis + spec creation |
| Lifecycle transition | <1ms | State machine update |
| HTML rendering | <10ms | Full panel with CSS |
| Terminal rendering | <5ms | ASCII art generation |
| SpecLedger persistence | <2ms | JSON append |

## Future Roadmap

**Completed (Phase 2)**:
- ✅ **MRF-Enhanced Panel Generation** - ELEGANCE strategy for WHY panels
- ✅ **Thompson Sampling Learning** - Learns optimal panel types per query type

**Planned Enhancements (Phase 3+)**:
1. **LLM-Based Panel Compilation** - Replace heuristics with LLM reasoning
2. **ReactRenderer** - First-class React component output
3. **ARRenderer** - Spatial computing overlays for AR glasses
4. **Collaborative Panels** - Multi-user shared panels
5. **Panel Templates** - User-defined panel templates
6. **Animation System** - Smooth transitions between states
7. **Accessibility** - Full ARIA support and screen reader optimization

## Key Files

- **Entry Point**: `HoloLoom/visualization/jenny_runtime.py`
- **MRF Integration**: `HoloLoom/visualization/jenny_mrf.py`
- **Orchestrator Integration**: `HoloLoom/weaving_orchestrator.py` (lines 478-620, 1791-1820)
- **Config**: `HoloLoom/config.py` (line 310)
- **Unit Tests**: `HoloLoom/tests/unit/test_jenny*.py` (239 tests)
- **MRF Tests**: `HoloLoom/tests/unit/test_jenny_mrf.py` (41 tests)
- **Integration Tests**: `HoloLoom/tests/integration/test_jenny_orchestrator.py` (18 tests)

---

*Last Updated: December 2025 (Phase 2.1-2.2 Complete)*
