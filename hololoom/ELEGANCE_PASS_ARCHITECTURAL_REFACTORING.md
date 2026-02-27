# Elegance Pass: Architectural Refactoring Scope

**Created**: 2025-12-09
**Status**: Scoping Document
**Target File**: `hololoom/weaving_orchestrator.py` (2,561 lines)

## Executive Summary

The `weaving_orchestrator.py` file has grown to 2,561 lines with 325 `self.` references, making it difficult to maintain, test, and extend. This document scopes three architectural refactoring approaches, each with different levels of ambition and risk.

---

## Current State Analysis

### File Metrics
- **Lines of Code**: 2,561
- **`self.` references**: 325 (high coupling to orchestrator state)
- **Constructor parameters**: ~35+ (many feature flags)
- **Methods**: 20+ public and private
- **weave() method**: ~1,200 lines (Steps 0-9 + error handling)

### Key Coupling Points

| Component | `self.` References | Description |
|-----------|-------------------|-------------|
| `self.cfg` | ~50 | Configuration access throughout |
| `self.logger` | ~80 | Logging in every step |
| `self.yarn_graph` | ~10 | Memory graph access |
| `self.embedder` | ~15 | Embedding operations |
| `self.guardrails` | ~20 | Safety checks |
| `self.jenny_*` | ~15 | Jenny UI runtime |
| Feature flags | ~50 | `self.enable_*` checks |

### Step Structure (weave() method)

```
Step 0:  Meta-Prompt Enhancement      (~30 lines)
Step 1:  Loom Command                 (~30 lines)
Step 2:  Chrono Trigger               (~40 lines)
Step 3:  Thread Selection             (~60 lines)
Step 4:  Resonance Shed (Setup)       (~100 lines)
Step 5:  Warp Space (Setup)           (~30 lines)
Steps 4-6: Parallel Execution         (~150 lines) - closures
Step 6.5: Beta Wave Packing           (~50 lines)
Step 7:  Convergence Engine           (~150 lines)
Step 8:  Tool Execution + Safety      (~150 lines)
Step 9:  Spacetime Fabric             (~200 lines)
Jenny UI Generation                   (~100 lines) [EXTRACTED]
Reflection + Caching                  (~100 lines)
Error Handling + Metrics              (~100 lines)
```

### Root Cause of Complexity

1. **Feature accretion**: Each new feature adds constructor params and step logic
2. **Inline closures**: Steps 4-6 define async closures capturing local scope
3. **Data threading**: 15+ variables flow between steps via local scope
4. **Error handling**: Try/except blocks scattered throughout
5. **Metrics collection**: Timing code interleaved with business logic

---

## Refactoring Options

### Option A: Pipeline Context Object (Medium Risk)

**Philosophy**: "Data flows through, orchestrator coordinates"

**Approach**:
- Create a `WeavingContext` dataclass that accumulates all intermediate results
- Each step reads from and writes to this context
- Steps become pure functions: `step_n(context, config) -> context`

**Key Structures**:

```python
@dataclass
class WeavingContext:
    """Accumulated state through the weaving pipeline."""
    # Input
    query: Query
    pattern_override: Optional[PatternCard] = None
    complexity_override: Optional[ComplexityLevel] = None

    # Step 1 outputs
    pattern_spec: Optional[PatternSpec] = None

    # Step 2 outputs
    temporal_window: Optional[TemporalWindow] = None
    chrono: Optional[ChronoTrigger] = None

    # Step 3 outputs
    threads: List[MemoryShard] = field(default_factory=list)
    thread_ids: List[str] = field(default_factory=list)

    # Step 4 outputs
    dot_plasma: Optional[Dict] = None
    features: Optional[Features] = None

    # Step 5 outputs
    warp_space: Optional[WarpSpace] = None
    warp_result: Optional[Dict] = None

    # Step 6 outputs
    retrieval_context: Optional[RetrievalContext] = None

    # Step 7 outputs
    action_plan: Optional[ActionPlan] = None
    collapse_result: Optional[CollapseResult] = None

    # Step 8 outputs
    tool_result: Optional[Dict] = None
    safety_decision: Optional[SafetyDecision] = None

    # Step 9 outputs
    spacetime: Optional[Spacetime] = None

    # Metadata
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    awareness_context: Optional[Dict] = None

    # Timing
    start_time: Optional[datetime] = None
```

**Step Signature**:
```python
async def execute_step_1_pattern_selection(
    ctx: WeavingContext,
    loom_command: LoomCommand,
    logger: Logger
) -> WeavingContext:
    """Select pattern card based on query complexity."""
    step_start = time.time()

    ctx.pattern_spec = loom_command.select_pattern(
        ctx.query.text,
        user_preference=ctx.pattern_override.value if ctx.pattern_override else None
    )

    ctx.stage_timings['pattern_selection'] = (time.time() - step_start) * 1000
    logger.info(f"  [1] Pattern selected: {ctx.pattern_spec.name}")
    return ctx
```

**Benefits**:
- Clear data flow visualization
- Steps become testable in isolation
- Easy to add/remove steps
- Context object documents all intermediate state

**Risks**:
- Large context object (~30 fields)
- Still need orchestrator for component access
- Moderate refactoring effort

**Effort Estimate**: 3-4 days
- Day 1: Create WeavingContext, extract Steps 1-3
- Day 2: Extract Steps 4-6 (parallel execution)
- Day 3: Extract Steps 7-9
- Day 4: Testing, cleanup, documentation

---

### Option B: Stage Executor Classes (Higher Risk)

**Philosophy**: "Each stage is a self-contained unit"

**Approach**:
- Each step becomes its own class with `execute()` method
- Orchestrator becomes a thin coordinator that chains executors
- Dependencies are injected via constructor

**Key Structures**:

```python
# Base protocol
class StageExecutorProtocol(Protocol):
    """Protocol for stage executors."""
    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        ...

# Step 1 Executor
class PatternSelectionStage:
    """Step 1: Loom Command selects Pattern Card."""

    def __init__(self, loom_command: LoomCommand, logger: Logger):
        self.loom_command = loom_command
        self.logger = logger

    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        step_start = time.time()

        ctx.pattern_spec = self.loom_command.select_pattern(
            ctx.query.text,
            user_preference=ctx.pattern_override.value if ctx.pattern_override else None
        )

        ctx.stage_timings['pattern_selection'] = (time.time() - step_start) * 1000
        self.logger.info(f"  [1] Pattern selected: {ctx.pattern_spec.name}")
        return ctx

# Step 4-6 Parallel Executor
class ParallelFeatureStage:
    """Steps 4-6: Parallelized feature extraction, warp tensioning, retrieval."""

    def __init__(self, cfg: Config, embedder, retriever, guardrails, logger: Logger):
        self.cfg = cfg
        self.embedder = embedder
        self.retriever = retriever
        self.guardrails = guardrails
        self.logger = logger

    async def execute(self, ctx: WeavingContext) -> WeavingContext:
        # Create components based on pattern spec
        resonance_shed = self._create_resonance_shed(ctx.pattern_spec)
        warp_space = self._create_warp_space(ctx.pattern_spec)

        # Run in parallel
        results = await asyncio.gather(
            self._step4_features(ctx, resonance_shed),
            self._step5_warp(ctx, warp_space),
            self._step6_retrieval(ctx)
        )

        # Merge results into context
        ctx.dot_plasma = results[0]
        ctx.warp_result = results[1]
        ctx.retrieval_context = results[2]

        return ctx
```

**Orchestrator becomes thin**:

```python
class WeavingOrchestrator:
    """Thin coordinator that chains stage executors."""

    def __init__(self, cfg: Config, ...):
        # Create stage executors
        self.stage_1 = PatternSelectionStage(self.loom_command, self.logger)
        self.stage_2 = ChronoTriggerStage(self.logger)
        self.stage_3 = ThreadSelectionStage(self.yarn_graph, self.shuttle_stage, self.logger)
        self.stage_4_6 = ParallelFeatureStage(cfg, self.embedder, self.retriever, self.guardrails, self.logger)
        self.stage_7 = ConvergenceStage(cfg, self.tool_executor, self.logger)
        self.stage_8 = ToolExecutionStage(self.tool_executor, self.guardrails, self.audit_trail, self.logger)
        self.stage_9 = SpacetimeFabricStage(self.logger)

    async def weave(self, query: Query, ...) -> Spacetime:
        # Create context
        ctx = WeavingContext(query=query, ...)

        # Execute pipeline
        ctx = await self.stage_1.execute(ctx)
        ctx = await self.stage_2.execute(ctx)
        ctx = await self.stage_3.execute(ctx)
        ctx = await self.stage_4_6.execute(ctx)
        ctx = await self.stage_7.execute(ctx)
        ctx = await self.stage_8.execute(ctx)
        ctx = await self.stage_9.execute(ctx)

        return ctx.spacetime
```

**Benefits**:
- Maximum testability (each stage isolated)
- Easy to swap implementations
- Clear separation of concerns
- Stages can be composed differently

**Risks**:
- Significant refactoring effort
- May break existing integrations
- Need to handle error propagation

**Effort Estimate**: 5-7 days
- Day 1-2: Create WeavingContext + base protocols
- Day 3: Extract Steps 1-3 as executors
- Day 4: Extract Steps 4-6 parallel executor
- Day 5: Extract Steps 7-9 as executors
- Day 6: Wire up orchestrator, error handling
- Day 7: Testing, migration, documentation

---

### Option C: Protocol-Based Dependency Injection (Highest Ambition)

**Philosophy**: "Define contracts, inject implementations"

**Approach**:
- Define protocols for each capability (pattern selection, thread retrieval, etc.)
- Orchestrator receives protocol implementations via constructor
- Enable complete swapping of implementations

**Key Protocols**:

```python
# protocols/weaving_protocols.py

class PatternSelectorProtocol(Protocol):
    """Selects processing pattern based on query."""
    def select_pattern(self, query_text: str, user_preference: Optional[str] = None) -> PatternSpec:
        ...

class ThreadSelectorProtocol(Protocol):
    """Selects memory threads for context."""
    async def select_threads(self, temporal_window: TemporalWindow, query: Query) -> List[MemoryShard]:
        ...

class FeatureExtractorProtocol(Protocol):
    """Extracts features from query and threads."""
    async def extract(self, query: Query, threads: List[MemoryShard]) -> Features:
        ...

class WarpSpaceProtocol(Protocol):
    """Tensions threads into continuous manifold."""
    async def tension(self, threads: List[MemoryShard]) -> WarpResult:
        ...
    def collapse(self) -> List[WarpUpdate]:
        ...

class ConvergenceProtocol(Protocol):
    """Collapses probabilities to discrete tool selection."""
    def collapse(self, neural_probs: np.ndarray) -> CollapseResult:
        ...

class ToolExecutorProtocol(Protocol):
    """Executes selected tool."""
    async def execute(self, tool: str, query: Query, context: RetrievalContext) -> Dict:
        ...

class SafetyGateProtocol(Protocol):
    """Gates actions through safety checks."""
    def gate_action(self, request: ActionRequest) -> SafetyDecision:
        ...
```

**Orchestrator with DI**:

```python
class WeavingOrchestrator:
    """Protocol-based orchestrator with dependency injection."""

    def __init__(
        self,
        cfg: Config,
        # Core protocols (required)
        pattern_selector: PatternSelectorProtocol,
        thread_selector: ThreadSelectorProtocol,
        feature_extractor: FeatureExtractorProtocol,
        convergence: ConvergenceProtocol,
        tool_executor: ToolExecutorProtocol,
        # Optional protocols
        warp_space: Optional[WarpSpaceProtocol] = None,
        safety_gate: Optional[SafetyGateProtocol] = None,
        jenny_runtime: Optional[JennyRuntimeProtocol] = None,
        awareness: Optional[AwarenessProtocol] = None,
    ):
        self.cfg = cfg
        self.pattern_selector = pattern_selector
        self.thread_selector = thread_selector
        self.feature_extractor = feature_extractor
        self.convergence = convergence
        self.tool_executor = tool_executor
        self.warp_space = warp_space
        self.safety_gate = safety_gate
        self.jenny_runtime = jenny_runtime
        self.awareness = awareness
```

**Factory for easy instantiation**:

```python
def create_weaving_orchestrator(cfg: Config, shards: List[MemoryShard]) -> WeavingOrchestrator:
    """Factory function with sensible defaults."""
    return WeavingOrchestrator(
        cfg=cfg,
        pattern_selector=LoomCommand(cfg),
        thread_selector=YarnGraphThreadSelector(shards, cfg),
        feature_extractor=ResonanceShedExtractor(cfg),
        convergence=ThompsonSamplingConvergence(cfg),
        tool_executor=StandardToolExecutor(cfg),
        warp_space=WarpSpace(cfg) if cfg.enable_warp_space else None,
        safety_gate=SafetyGuardrails(cfg) if cfg.enable_safety else None,
    )
```

**Benefits**:
- Maximum flexibility and testability
- Clear contracts between components
- Easy to mock for testing
- Supports multiple implementations

**Risks**:
- Largest refactoring effort
- Breaking change for all consumers
- Need migration path
- May over-engineer simple use cases

**Effort Estimate**: 7-10 days
- Day 1-2: Define all protocols
- Day 3-4: Create default implementations
- Day 5-6: Refactor orchestrator to use protocols
- Day 7-8: Create factory + migration helpers
- Day 9-10: Testing, documentation, migration guide

---

## Recommendation

### Recommended Path: Option A (Pipeline Context) + Incremental Option B

**Phase 1** (3-4 days): Implement Option A
- Create `WeavingContext` dataclass
- Extract steps as pure functions in `orchestrator/stages/` module
- Keep orchestrator method signatures unchanged
- **Low risk, immediate wins**

**Phase 2** (4-5 days): Migrate to Option B
- Convert pure functions to Stage Executor classes
- Add protocol definitions
- Update orchestrator to chain executors
- **Higher value, moderate risk**

**Phase 3** (Optional, 3-4 days): Add Option C flexibility
- Define additional protocols as needed
- Enable DI for testing and customization
- **Only if needed**

### Implementation Roadmap

```
Week 1:
├── Day 1: Create WeavingContext dataclass
├── Day 2: Extract Steps 1-3 as functions
├── Day 3: Extract Steps 4-6 (parallel) as functions
├── Day 4: Extract Steps 7-9 as functions
└── Day 5: Testing + documentation

Week 2:
├── Day 1: Define StageExecutorProtocol
├── Day 2: Convert Steps 1-3 to executor classes
├── Day 3: Convert Steps 4-6 to ParallelFeatureStage
├── Day 4: Convert Steps 7-9 to executor classes
└── Day 5: Wire up thin orchestrator + testing
```

---

## File Structure After Refactoring

```
hololoom/orchestrator/
├── __init__.py                    # Package exports
├── context.py                     # WeavingContext dataclass (~150 lines)
├── protocols.py                   # Stage protocols (~100 lines)
├── factory.py                     # Factory functions (~100 lines)
│
├── core/                          # Already extracted
│   ├── complexity_detection.py
│   ├── metrics_collection.py
│   ├── background_tasks.py
│   └── stat_mech_integration.py
│
├── jenny/                         # Already extracted
│   ├── panel_detection.py
│   └── __init__.py
│
└── stages/                        # NEW: Stage executors
    ├── __init__.py
    ├── step1_pattern_selection.py (~80 lines)
    ├── step2_chrono_trigger.py    (~60 lines)
    ├── step3_thread_selection.py  (~100 lines)
    ├── step4_6_parallel.py        (~250 lines)
    ├── step7_convergence.py       (~150 lines)
    ├── step8_tool_execution.py    (~200 lines)
    └── step9_spacetime.py         (~200 lines)
```

**Estimated Final State**:
- `weaving_orchestrator.py`: ~800 lines (from 2,561)
- New files: ~1,200 lines across stages/
- Net code growth: ~0 (refactoring, not adding)

---

## Success Criteria

1. **weave() method < 200 lines** (currently ~1,200)
2. **Each stage testable in isolation**
3. **No breaking changes to public API**
4. **All existing tests pass**
5. **Stage timing preserved for monitoring**

---

## Next Steps

1. **Review this document** with stakeholder
2. **Choose approach** (A, B, or A+B incremental)
3. **Create branch** `elegance-pass/pipeline-refactor`
4. **Implement Phase 1** (WeavingContext + pure functions)
5. **Test thoroughly** before Phase 2
