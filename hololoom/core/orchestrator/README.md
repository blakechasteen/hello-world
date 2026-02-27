# Orchestrator Stages

**Status**: Production Ready (December 2025)
**Location**: `hololoom/orchestrator/stages/`
**Lines**: ~1,539 lines across 3 stage files
**Philosophy**: "Data flows through, orchestrator coordinates"

Modular decomposition of HoloLoom's 9-step weaving cycle into pure function stages for improved maintainability, testability, and composability.

---

## Overview

The Portal Orchestration Stages refactor decomposes the monolithic weaving cycle into discrete, pure-function stages. Each stage:

- Takes `WeavingContext` as first parameter
- Returns `WeavingContext` (mutated or new)
- Has no `self` references (pure functions)
- Receives all dependencies as explicit parameters
- Can be tested in isolation
- Can be composed in different orders

**Benefits**:
- **Easier testing**: Each stage can be unit tested independently
- **Better debugging**: Clear boundaries between stages
- **Flexible composition**: Stages can be reordered or skipped
- **Parallel execution**: Independent stages (4-6) run concurrently for 40-120ms speedup
- **Code clarity**: Each stage has a single, well-defined responsibility

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Weaving Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEPS 0-3: Query Setup & Thread Selection                       │   │
│  │ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────────┐ │   │
│  │ │ Step 0   │ │ Step 1   │ │ Step 2   │ │ Step 3               │ │   │
│  │ │Meta-Prompt│→│Pattern   │→│Chrono    │→│Thread Selection      │ │   │
│  │ │(optional)│ │Selection │ │Trigger   │ │(Shuttle/Yarn Graph)  │ │   │
│  │ └──────────┘ └──────────┘ └──────────┘ └──────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   ↓                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEPS 4-6: Parallel Feature Extraction (40-120ms speedup)       │   │
│  │ ┌──────────────────────────────────────────────────────────────┐│   │
│  │ │         asyncio.gather (concurrent execution)                 ││   │
│  │ │ ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐   ││   │
│  │ │ │ Step 4       │ │ Step 5       │ │ Step 6               │   ││   │
│  │ │ │ Resonance    │ │ Warp Space   │ │ Memory Retrieval     │   ││   │
│  │ │ │ Shed         │ │ Tensioning   │ │ (Multipass Crawl)    │   ││   │
│  │ │ │ (DotPlasma)  │ │              │ │                      │   ││   │
│  │ │ └──────────────┘ └──────────────┘ └──────────────────────┘   ││   │
│  │ └──────────────────────────────────────────────────────────────┘│   │
│  │                                                                  │   │
│  │ Post-parallel (optional):                                        │   │
│  │ ┌──────────────┐ ┌────────────────────────┐                     │   │
│  │ │ Step 5.5     │ │ Step 6.5               │                     │   │
│  │ │ Warp Compute │ │ Beta Wave Packing      │                     │   │
│  │ └──────────────┘ └────────────────────────┘                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   ↓                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEPS 7-9: Convergence, Execution & Output                      │   │
│  │ ┌──────────────┐ ┌──────────────┐ ┌───────────────────────────┐ │   │
│  │ │ Step 7       │ │ Step 8       │ │ Step 9                    │ │   │
│  │ │ Convergence  │→│ Tool         │→│ Spacetime Fabric          │ │   │
│  │ │ Engine       │ │ Execution    │ │ (Final Output + Trace)    │ │   │
│  │ │ (Decision)   │ │ (+ Safety)   │ │                           │ │   │
│  │ └──────────────┘ └──────────────┘ └───────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Lines | Description |
|------|-------|-------------|
| `steps_0_3.py` | ~350 | Query setup and thread selection |
| `steps_4_6.py` | ~674 | Parallel feature extraction |
| `steps_7_9.py` | ~515 | Convergence, execution, output |
| **Total** | **~1,539** | Complete 9-step pipeline |

---

## The 9-Step Weaving Cycle

### Steps 0-3: Query Setup and Thread Selection

**Step 0: Meta-Prompt Enhancement** (optional)

Enhances the query through an LLM call before processing:
- Query rewriting
- Clarification
- Expansion

```python
from hololoom.orchestrator.stages import execute_step0_meta_prompt

ctx = await execute_step0_meta_prompt(
    ctx,
    enable_enhancement=True,
    proto_llm_call=orchestrator.proto_llm_call
)

if ctx.enhancement_applied:
    print(f"Original: {ctx.original_query_text}")
    print(f"Enhanced: {ctx.current_query_text}")
```

---

**Step 1: Pattern Selection** (Loom Command)

Selects the processing pattern based on query complexity:

| Pattern | Target Latency | Use Case |
|---------|----------------|----------|
| **BARE** | <50ms | Simple factual queries |
| **FAST** | <150ms | Standard queries |
| **FUSED** | <300ms | Complex queries |

```python
from hololoom.orchestrator.stages import execute_step1_pattern_selection

ctx = await execute_step1_pattern_selection(ctx, loom_command)
print(f"Selected: {ctx.pattern_spec.name}")
print(f"Timeout: {ctx.pattern_spec.pipeline_timeout}s")
```

---

**Step 2: Chrono Trigger** (Temporal Window)

Creates temporal context for memory retrieval:
- Time window for relevant memories
- Recency bias weighting
- Pipeline timeout from pattern spec

```python
from hololoom.orchestrator.stages import execute_step2_chrono_trigger

ctx = await execute_step2_chrono_trigger(
    ctx,
    lookback_days=365,
    recency_bias=0.5
)
print(f"Window: {ctx.temporal_window.start} to {ctx.temporal_window.end}")
```

---

**Step 3: Thread Selection** (Yarn Graph / Shuttle)

Selects relevant memory threads using:
- **Shuttle**: MCTS-powered Warp<->Yarn intersection (advanced)
- **Yarn Graph**: Simple temporal thread selection (fallback)

```python
from hololoom.orchestrator.stages import execute_step3_thread_selection

ctx = await execute_step3_thread_selection(
    ctx,
    yarn_graph=kg,
    enable_shuttle=False
)
print(f"Selected {len(ctx.threads)} threads")
```

---

### Steps 4-6: Parallel Feature Extraction

These steps execute **concurrently** via `asyncio.gather` for 40-120ms speedup.

**Step 4: Resonance Shed** (DotPlasma creation)

Extracts features through motif, embedding, and spectral threads:
- **Motif Thread**: Symbolic pattern detection
- **Embedding Thread**: Multi-scale vectors (Matryoshka)
- **Spectral Thread**: Graph topology features

Creates **DotPlasma** - the flowing continuous representation.

---

**Step 5: Warp Space** (Continuous manifold tensioning)

Tensions discrete yarn threads into continuous tensor space:
- Transforms discrete → continuous
- Enables tensor operations
- Lifecycle: tension() → compute() → collapse()

---

**Step 6: Memory Retrieval** (Multipass crawl)

Retrieves context with intelligent graph traversal:
- **Multipass crawl**: Graph traversal with gated retrieval
- **Legacy retriever**: Traditional static shard retrieval (fallback)

---

**Main Parallel Executor**:

```python
from hololoom.orchestrator.stages import execute_steps_4_6_parallel

ctx = await execute_steps_4_6_parallel(
    ctx, cfg, embedder,
    memory=memory_backend,
    emit_stage_event=emit_fn
)

print(f"Features: {len(ctx.dot_plasma.get('threads', []))} threads")
print(f"Shards: {len(ctx.shards)}")
print(f"Speedup: {ctx.stage_timings['parallel_speedup']:.2f}x")
```

---

**Step 5.5: Warp Compute** (optional)

Performs tensor operations in continuous manifold:
- Spectral features computation
- Attention entropy calculation
- Context vector generation

```python
from hololoom.orchestrator.stages import execute_step5_5_warp_compute

ctx = await execute_step5_5_warp_compute(ctx)
print(f"Attention entropy: {ctx.warp_compute_results['attention_entropy']:.3f}")
```

---

**Step 6.5: Beta Wave Context Packing** (optional)

Physics-based context optimization using activation spreading:
- Uses spring dynamics for activation propagation
- Achieves 50% token reduction with <1ms overhead
- Requires MultiWaveMemoryEngine with spring_engine

```python
from hololoom.orchestrator.stages import execute_step6_5_beta_wave_packing

ctx = await execute_step6_5_beta_wave_packing(ctx, cfg, memory)
if ctx.packed_context:
    print(f"Included: {ctx.packed_context.elements_included}")
    print(f"Compressed: {ctx.packed_context.elements_compressed}")
```

---

### Steps 7-9: Convergence, Execution, and Output

**Step 7: Convergence Engine** (Decision collapse)

Collapses probability distributions to discrete tool selection:
- Gets neural predictions from policy
- Optionally blends with gradient flow
- Supports multiple strategies

| Strategy | Description |
|----------|-------------|
| **EPSILON_GREEDY** | 90% exploit, 10% explore (default) |
| **BAYESIAN_BLEND** | 70% neural + 30% Thompson Sampling |
| **PURE_THOMPSON** | Pure Thompson Sampling |

```python
from hololoom.orchestrator.stages import execute_step7_convergence

ctx = await execute_step7_convergence(ctx, cfg, policy, tool_executor)
print(f"Selected: {ctx.collapse_result.tool}")
print(f"Confidence: {ctx.collapse_result.confidence:.2f}")
```

---

**Step 8: Tool Execution** (with safety gating)

Gates action through safety guardrails and executes:
- Checks safety before tool execution
- Logs to audit trail
- Graceful degradation if safety check fails

```python
from hololoom.orchestrator.stages import execute_step8_tool_execution

ctx = await execute_step8_tool_execution(
    ctx, tool_executor, guardrails, audit_trail
)

if ctx.safety_blocked:
    print(f"Blocked: {ctx.safety_decision.reason}")
else:
    print(f"Result: {ctx.tool_result}")
```

---

**Step 9: Spacetime Fabric** (Final output)

Weaves final output with complete provenance:
- Detensions Warp Space
- Creates WeavingTrace with full provenance
- Assembles final Spacetime artifact

```python
from hololoom.orchestrator.stages import execute_step9_spacetime_fabric

ctx = await execute_step9_spacetime_fabric(ctx, cfg)
print(f"Response: {ctx.spacetime.response}")
print(f"Confidence: {ctx.spacetime.confidence:.2f}")
print(f"Duration: {ctx.spacetime.trace.duration_ms:.1f}ms")
```

---

## Quick Start

Complete pipeline execution:

```python
from hololoom.orchestrator.stages import (
    execute_step1_pattern_selection,
    execute_step2_chrono_trigger,
    execute_step3_thread_selection,
    execute_steps_4_6_parallel,
    execute_step7_convergence,
    execute_step8_tool_execution,
    execute_step9_spacetime_fabric
)
from hololoom.orchestrator.context import WeavingContext

# Create context
ctx = WeavingContext(query=query)

# Execute stages sequentially
ctx = await execute_step1_pattern_selection(ctx, loom_command)
ctx = await execute_step2_chrono_trigger(ctx)
ctx = await execute_step3_thread_selection(ctx, yarn_graph)

# Steps 4-6 run in parallel (40-120ms speedup)
ctx = await execute_steps_4_6_parallel(ctx, cfg, embedder, memory=memory)

# Continue with convergence and execution
ctx = await execute_step7_convergence(ctx, cfg, policy, tool_executor)
ctx = await execute_step8_tool_execution(ctx, tool_executor, guardrails)
ctx = await execute_step9_spacetime_fabric(ctx, cfg)

# Access final result
print(f"Response: {ctx.spacetime.response}")
print(f"Confidence: {ctx.spacetime.confidence:.2f}")
```

---

## Performance

| Phase | Sequential | Parallel | Speedup |
|-------|-----------|----------|---------|
| **Steps 4-6** | ~150ms | ~70ms | **2.1x** |
| **Total pipeline** | ~300ms | ~220ms | **1.4x** |
| **Step overhead** | <1ms | <2ms | Negligible |

**Typical timings** (FUSED mode):
- Steps 0-3: ~50ms (setup and thread selection)
- Steps 4-6: ~70ms (parallel execution)
- Step 5.5: ~10ms (warp compute)
- Step 6.5: ~1ms (beta wave packing)
- Steps 7-9: ~80ms (convergence, execution, fabric)
- **Total**: ~220ms

---

## WeavingContext

The central data structure that flows through all stages:

```python
@dataclass
class WeavingContext:
    # Input
    query: Query
    pattern_override: Optional[PatternCard] = None
    auto_enhance: Optional[bool] = None

    # Step 0-3 Results
    enhanced_query: Optional[Query] = None
    pattern_spec: Optional[PatternSpec] = None
    chrono: Optional[ChronoTrigger] = None
    temporal_window: Optional[TemporalWindow] = None
    threads: List[MemoryShard] = field(default_factory=list)

    # Step 4-6 Results
    dot_plasma: Optional[Dict] = None
    warp_space: Optional[WarpSpace] = None
    shards: List[MemoryShard] = field(default_factory=list)

    # Step 7-9 Results
    collapse_result: Optional[CollapseResult] = None
    tool_result: Optional[Dict] = None
    spacetime: Optional[Spacetime] = None
    trace: Optional[WeavingTrace] = None

    # Timing & Provenance
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

---

## Stage Event Callbacks

Emit progress events for UI updates:

```python
def emit_stage_event(step: int, name: str, duration: Optional[float]):
    """Called when stage starts (duration=None) or completes."""
    if duration is None:
        print(f"[{step}] {name} started...")
    else:
        print(f"[{step}] {name} completed in {duration:.1f}ms")

ctx = await execute_step1_pattern_selection(
    ctx, loom_command,
    emit_stage_event=emit_stage_event
)
```

---

## Integration with WeavingOrchestrator

Portal stages integrate seamlessly with the existing orchestrator:

```python
# In WeavingOrchestrator.weave()
from hololoom.orchestrator.stages import execute_steps_4_6_parallel

# Replace monolithic feature extraction with parallel stages
ctx = await execute_steps_4_6_parallel(
    ctx, self.cfg, self.embedder,
    memory=self.memory,
    retriever=self.retriever,
    complexity=self.complexity,
    provenance=self.provenance,
    emit_stage_event=self._emit_stage_event
)
```

---

## When to Use

**Use Portal Stages when**:
- Building new orchestrators with custom flows
- Testing individual weaving steps in isolation
- Need flexibility to reorder or skip stages
- Want to monitor progress with stage events
- Debugging specific pipeline issues

**Use Standard Orchestrator when**:
- Using default 9-step flow (most common)
- Don't need stage-level customization
- Prefer higher-level API

---

## Exports

```python
from hololoom.orchestrator.stages import (
    # Steps 0-3
    execute_step0_meta_prompt,
    execute_step1_pattern_selection,
    execute_step2_chrono_trigger,
    execute_step3_thread_selection,

    # Steps 4-6 (parallel)
    execute_steps_4_6_parallel,
    execute_step5_5_warp_compute,
    execute_step6_5_beta_wave_packing,

    # Steps 7-9
    execute_step7_convergence,
    execute_step8_tool_execution,
    execute_step9_spacetime_fabric,

    # Helpers
    create_resonance_shed,
    create_warp_space,
    select_pattern_embedder,
)
```

---

## Related Documentation

- [Weaving Orchestrator](../weaving_orchestrator.py) - Main orchestrator implementation
- [Awareness Graph README](../memory/AWARENESS_GRAPH_README.md) - Consciousness integration
- [Alignment Framework](../alignment/README.md) - Safety guardrails
- [Convergence Engine](../convergence/README.md) - Decision collapse strategies
