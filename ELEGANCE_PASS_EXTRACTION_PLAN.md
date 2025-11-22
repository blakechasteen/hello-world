# Elegance Pass: Extraction Plan for weaving_orchestrator.py

**Created**: 2025-11-22
**Goal**: Reduce weaving_orchestrator.py from 3,640 lines → <1,500 lines (58.8% reduction)
**Status**: Planning Complete - Ready for Implementation

---

## File Analysis Summary

**Current State**:
- **Total Lines**: 3,640
- **Total Methods**: 54
- **Classes**: 3 (ToolExecutor, YarnGraph, WeavingOrchestrator)
- **Large Methods (>50 lines)**: 11 methods, 928 lines total (25.5%)

**Target State**:
- **Total Lines**: <1,500 (goal)
- **Methods per File**: <50 lines each
- **Cyclomatic Complexity**: <10 per method
- **Documentation**: 100% docstring coverage

---

## Extraction Strategy

### Phase 1: Extract Standalone Classes (~200 lines)

#### 1.1 Extract ToolExecutor → `HoloLoom/tools/executor.py`
**Lines**: 164-329 (~165 lines)
**Why**: Self-contained tool execution logic, no tight coupling to orchestrator
**Complexity**: Low
**Dependencies**:
- HoloLoom.protocols.types (Query, Context)
- HoloLoom.awareness.llm_integration (OllamaLLM)

**New File Structure**:
```python
# HoloLoom/tools/executor.py
class ToolExecutor:
    """
    Executes tools based on convergence engine decisions.

    Supports multiple tools: answer, search, notion_write, calc
    Falls back gracefully when LLM unavailable.
    """

    def __init__(self, llm: Optional['OllamaLLM'] = None): ...
    async def execute(self, tool: str, query: Query, context: Context) -> Dict: ...
    async def _handle_answer(self, query: Query, context: Context) -> Dict: ...
    async def _handle_search(self, query: Query, context: Context) -> Dict: ...
    async def _handle_notion_write(self, query: Query, context: Context) -> Dict: ...
    async def _handle_calc(self, query: Query, context: Context) -> Dict: ...
    async def _handle_unknown(self, query: Query, context: Context) -> Dict: ...
```

**Import Update in weaving_orchestrator.py**:
```python
from HoloLoom.tools.executor import ToolExecutor
```

---

#### 1.2 Decision: Keep or Extract YarnGraph
**Lines**: 336-371 (~35 lines)
**Options**:
  - **Option A (Recommended)**: Keep as nested class (minimal gain, tight coupling)
  - **Option B**: Extract to `HoloLoom/memory/simple_yarn_graph.py` (for completeness)

**Decision**: Keep as nested class (already in memory/graph.py as KG, this is just a shim)

**Total Phase 1 Extraction**: ~165 lines

---

### Phase 2: Extract Initialization Logic (~600 lines)

Create `HoloLoom/orchestrator/initialization/` module with separate files for each initialization concern.

#### 2.1 Extract `_initialize_config_and_memory` → `config_init.py`
**Lines**: 570-682 (~113 lines)
**Responsibilities**:
- Memory source validation (memory vs yarn_graph vs shards)
- Deprecation warnings
- mythRL protocol setup
- Complexity thresholds configuration
- Multipass crawling configuration
- SafetyGuardrails creation
- Pattern card determination
- Lifecycle management setup

**New File**:
```python
# HoloLoom/orchestrator/initialization/config_init.py
def initialize_config_and_memory(
    orchestrator: 'WeavingOrchestrator',
    memory, yarn_graph, shards,
    pattern_preference,
    enable_complexity_auto_detect
) -> None:
    """Initialize memory configuration and mythRL protocols."""
    ...
```

---

#### 2.2 Extract `_initialize_reflection_and_caching` → `reflection_init.py`
**Lines**: 683-729 (~47 lines)
**Responsibilities**:
- ReflectionBuffer setup
- QueryCache initialization
- Smart query routing (QueryClassifier, FastPathRouter)
- Dashboard constructor (Edward Tufte Machine)

**New File**:
```python
# HoloLoom/orchestrator/initialization/reflection_init.py
def initialize_reflection_and_caching(
    orchestrator: 'WeavingOrchestrator',
    enable_reflection: bool,
    reflection_capacity: int
) -> None:
    """Initialize reflection loop, query cache, and dashboard constructor."""
    ...
```

---

#### 2.3 Extract `_initialize_recursive_learning` → `recursive_init.py`
**Lines**: 730-814 (~85 lines)
**Responsibilities**:
- Lazy initialization of recursive learning components
- Scratchpad (Phase 1)
- Pattern learner (Phase 2)
- Hot pattern tracker (Phase 3)
- Advanced refiner (Phase 4)
- Background learner (Phase 5)

**New File**:
```python
# HoloLoom/orchestrator/initialization/recursive_init.py
def initialize_recursive_learning(
    orchestrator: 'WeavingOrchestrator'
) -> Optional[Dict[str, Any]]:
    """
    Initialize recursive learning components (lazy initialization).

    Returns:
        Dictionary of components or None if initialization failed
    """
    ...
```

---

#### 2.4 Extract `_initialize_components` → `component_init.py`
**Lines**: 815-956 (~142 lines) - **LARGEST INITIALIZATION METHOD**
**Responsibilities**:
- Loom Command creation
- Yarn Graph / Memory Backend setup
- Matryoshka Embeddings
- Semantic Cache (3-tier caching)
- Linguistic Matryoshka Gate
- ToolExecutor
- Gradient Flow Router
- Unified Physics Engine
- Statistical Mechanics Engine
- Retriever (legacy)

**New File**:
```python
# HoloLoom/orchestrator/initialization/component_init.py
def initialize_components(orchestrator: 'WeavingOrchestrator') -> None:
    """Initialize all weaving architecture components."""
    ...
```

---

#### 2.5 Extract `_initialize_production_hardening` → `production_init.py`
**Lines**: 957-1044 (~88 lines)
**Responsibilities**:
- ProductionConfig validation
- SystemMonitor creation
- CircuitBreakerRegistry setup
- RateLimiter initialization
- HealthChecker setup
- ErrorHandler creation

**New File**:
```python
# HoloLoom/orchestrator/initialization/production_init.py
def initialize_production_hardening(
    orchestrator: 'WeavingOrchestrator',
    production_config, rate_limit_qps,
    rate_limit_concurrent, enable_circuit_breakers,
    circuit_breaker_threshold, enable_health_checks
) -> None:
    """Initialize production hardening components (Part 5)."""
    ...
```

---

#### 2.6 Extract `_initialize_semantic_cache` → `cache_init.py`
**Lines**: 1045-1080 (~36 lines)
**Responsibilities**:
- 3-tier semantic caching setup (L1: LITE, L2: FAST, L3: FULL)
- SemanticSpectrum initialization
- 244D projection caching

**New File**:
```python
# HoloLoom/orchestrator/initialization/cache_init.py
def initialize_semantic_cache(orchestrator: 'WeavingOrchestrator') -> None:
    """Initialize three-tier semantic caching for 244D projections."""
    ...
```

---

#### 2.7 Extract `_initialize_linguistic_gate` → `linguistic_init.py`
**Lines**: 1081-1137 (~57 lines)
**Responsibilities**:
- Linguistic Matryoshka Gate setup (Phase 5)
- spaCy NLP pipeline
- Universal grammar chunking
- X-bar theory integration

**New File**:
```python
# HoloLoom/orchestrator/initialization/linguistic_init.py
def initialize_linguistic_gate(orchestrator: 'WeavingOrchestrator') -> None:
    """Initialize Phase 5 Linguistic Matryoshka Gate."""
    ...
```

---

**Total Phase 2 Extraction**: ~568 lines

**New Directory Structure**:
```
HoloLoom/orchestrator/
├── __init__.py                  # Public API
├── initialization/
│   ├── __init__.py              # Initialization orchestration
│   ├── config_init.py           # Memory & config setup (~113 lines)
│   ├── reflection_init.py       # Reflection & caching (~47 lines)
│   ├── recursive_init.py        # Recursive learning (~85 lines)
│   ├── component_init.py        # Core components (~142 lines)
│   ├── production_init.py       # Production hardening (~88 lines)
│   ├── cache_init.py            # Semantic cache (~36 lines)
│   └── linguistic_init.py       # Linguistic gate (~57 lines)
```

**Orchestrator `__init__` becomes**:
```python
def __init__(self, cfg, ...):
    """Initialize the Weaving Shuttle."""
    self.cfg = cfg
    self.logger = logging.getLogger(__name__)

    # Initialize all subsystems
    from HoloLoom.orchestrator.initialization import (
        initialize_config_and_memory,
        initialize_reflection_and_caching,
        initialize_components,
        initialize_production_hardening
    )

    if self.enable_production_hardening:
        initialize_production_hardening(self, ...)

    initialize_config_and_memory(self, ...)
    initialize_components(self)
    initialize_reflection_and_caching(self, ...)

    self.logger.info("WeavingOrchestrator initialization complete")
```

---

### Phase 3: Extract Core Logic (~400 lines)

#### 3.1 Extract `_assess_complexity_level` → `HoloLoom/orchestrator/complexity.py`
**Lines**: 1191-1286 (~95 lines)
**Why**: Complex standalone algorithm for complexity assessment
**Complexity**: Medium

**New File**:
```python
# HoloLoom/orchestrator/complexity.py
"""
Query Complexity Assessment Module

Implements 3-5-7-9 progressive complexity detection:
- LITE (3 steps): Greetings, simple commands (<50ms)
- FAST (5 steps): Standard questions (< 150ms)
- FULL (7 steps): Detailed queries (<300ms)
- RESEARCH (9 steps): Research mode (no limit)
"""

def assess_complexity_level(
    query: Query,
    trace: Optional[ProvenanceTrace],
    thresholds: Dict[str, Any]
) -> ComplexityLevel:
    """
    Assess query complexity using word count + intent patterns.

    Args:
        query: User query
        trace: Optional provenance trace
        thresholds: Complexity detection thresholds

    Returns:
        ComplexityLevel enum (LITE/FAST/FULL/RESEARCH)
    """
    ...
```

---

#### 3.2 Extract Recursive Learning Integration → `HoloLoom/orchestrator/recursive_integration.py`
**Lines**: 2769-2862 (~93 lines) + 2862-2931 (~69 lines) = ~162 lines
**Methods**:
- `_apply_recursive_learning` (93 lines)
- `get_recursive_learning_stats` (69 lines)

**New File**:
```python
# HoloLoom/orchestrator/recursive_integration.py
"""
Recursive Learning Integration

Integrates 5-phase recursive learning system:
- Phase 1: Scratchpad (provenance tracking)
- Phase 2: Pattern learner
- Phase 3: Hot pattern tracker
- Phase 4: Advanced refiner
- Phase 5: Background learner (Thompson Sampling + Policy Weights)
"""

async def apply_recursive_learning(
    orchestrator: 'WeavingOrchestrator',
    spacetime: Spacetime,
    query: Query
) -> Spacetime:
    """Apply recursive learning enhancements to spacetime result."""
    ...

def get_recursive_learning_stats(
    orchestrator: 'WeavingOrchestrator'
) -> Optional[Dict[str, Any]]:
    """Get comprehensive recursive learning statistics."""
    ...
```

---

#### 3.3 Extract Background Tasks → `HoloLoom/orchestrator/background_tasks.py`
**Lines**: 3129-3222 (~93 lines)
**Methods**:
- `_background_consolidation_loop` (93 lines)
- `start_background_consolidation` (23 lines)

**New File**:
```python
# HoloLoom/orchestrator/background_tasks.py
"""
Background Task Management

Manages long-running background processes:
- Statistical mechanics memory consolidation
- Periodic cache cleanup
- Metrics aggregation
"""

async def background_consolidation_loop(
    orchestrator: 'WeavingOrchestrator'
) -> None:
    """Background loop for statistical mechanics consolidation."""
    ...

def start_background_consolidation(
    orchestrator: 'WeavingOrchestrator'
) -> None:
    """Start background consolidation task."""
    ...
```

---

#### 3.4 Extract Stats/Metrics → `HoloLoom/orchestrator/metrics.py`
**Lines**: Various method locations
**Methods**:
- `get_health` (34 lines)
- `get_metrics` (31 lines)
- `get_circuit_breaker_status` (30 lines)
- `cache_stats` (9 lines)

**New File**:
```python
# HoloLoom/orchestrator/metrics.py
"""
Orchestrator Metrics and Health

Provides observability endpoints:
- Health checks (load balancer integration)
- Prometheus metrics
- Circuit breaker status
- Cache statistics
"""

async def get_health(
    orchestrator: 'WeavingOrchestrator'
) -> Optional[Dict[str, Any]]:
    """Get health status for load balancer health checks."""
    ...

def get_metrics(
    orchestrator: 'WeavingOrchestrator'
) -> Optional[Dict[str, Any]]:
    """Get comprehensive system metrics."""
    ...

def get_circuit_breaker_status(
    orchestrator: 'WeavingOrchestrator'
) -> Optional[Dict[str, Any]]:
    """Get circuit breaker status for all backends."""
    ...

def cache_stats(orchestrator: 'WeavingOrchestrator') -> Dict:
    """Get query cache statistics."""
    ...
```

**Total Phase 3 Extraction**: ~400 lines

---

### Phase 4: Move Demo Code (~77 lines)

#### 4.1 Move `main()` → `demos/orchestrator_demo.py`
**Lines**: 3562-3638 (~77 lines)
**Why**: Demo code shouldn't be in production module
**Complexity**: Low

**New File**:
```python
# demos/orchestrator_demo.py
"""
WeavingOrchestrator Demo

Demonstrates the full 9-step weaving cycle with all features enabled.
"""

async def main():
    """Run demonstration of weaving orchestrator."""
    ...

if __name__ == "__main__":
    asyncio.run(main())
```

**Remove from weaving_orchestrator.py**:
- Delete entire `main()` function
- Delete `if __name__ == "__main__":` block

---

### Phase 5: Extract Statistical Mechanics Integration (~52 lines)

#### 5.1 Extract `_shards_to_microstates` and `_macrostates_to_shards` → `HoloLoom/orchestrator/stat_mech_integration.py`
**Lines**: 3033-3129 (~96 lines combined)
**Methods**:
- `_shards_to_microstates` (44 lines)
- `_macrostates_to_shards` (52 lines)

**New File**:
```python
# HoloLoom/orchestrator/stat_mech_integration.py
"""
Statistical Mechanics Integration

Integrates Phase 5 statistical mechanics for memory consolidation.
Converts between memory shards and statistical mechanics states.
"""

def shards_to_microstates(
    shards: List[MemoryShard]
) -> List['Microstate']:
    """Convert memory shards to microstates."""
    ...

def macrostates_to_shards(
    macrostates: List['Macrostate']
) -> List[MemoryShard]:
    """Convert macrostates back to memory shards."""
    ...
```

---

## Extraction Summary

| Phase | Target | Lines Extracted | New Files |
|-------|--------|-----------------|-----------|
| **Phase 1** | Standalone Classes | ~165 | 1 |
| **Phase 2** | Initialization | ~568 | 7 |
| **Phase 3** | Core Logic | ~400 | 4 |
| **Phase 4** | Demo Code | ~77 | 1 |
| **Phase 5** | Stat Mech | ~96 | 1 |
| **Total** | | **~1,306 lines** | **14 files** |

**Remaining Lines**: 3,640 - 1,306 = **~2,334 lines** (still above target)

---

## Additional Extraction Opportunities

To reach <1,500 lines, we need an additional ~834 line reduction.

### Option 1: Extract Weaving Cycle Stages

The main `weave()` method and its helper stages could be extracted into separate modules:

- `HoloLoom/orchestrator/stages/loom_command.py` - Pattern selection stage
- `HoloLoom/orchestrator/stages/chrono_trigger.py` - Temporal window stage
- `HoloLoom/orchestrator/stages/yarn_selection.py` - Thread selection stage
- `HoloLoom/orchestrator/stages/resonance_shed.py` - Feature extraction stage
- `HoloLoom/orchestrator/stages/warp_space.py` - Tensioning stage
- `HoloLoom/orchestrator/stages/convergence.py` - Decision stage
- `HoloLoom/orchestrator/stages/execution.py` - Tool execution stage
- `HoloLoom/orchestrator/stages/spacetime.py` - Result weaving stage
- `HoloLoom/orchestrator/stages/reflection.py` - Learning stage

Each stage would be ~50-100 lines, extracting potentially 450-900 lines total.

### Option 2: Extract Helper Methods

Many smaller helper methods (<50 lines each) could be grouped by functionality:

- `HoloLoom/orchestrator/helpers/memory_query.py` - Memory querying helpers
- `HoloLoom/orchestrator/helpers/semantic.py` - Semantic analysis helpers
- `HoloLoom/orchestrator/helpers/lifecycle.py` - Lifecycle management helpers

Potential extraction: ~200-300 lines

---

## Implementation Order

**Week 1 (Days 1-5): Phases 1-2**
1. Extract ToolExecutor (Day 1)
2. Extract initialization modules (Days 2-5)
   - config_init.py
   - component_init.py
   - production_init.py
   - reflection_init.py
   - Others

**Week 2 (Days 6-10): Phases 3-4**
3. Extract complexity.py (Day 6)
4. Extract recursive_integration.py (Day 7)
5. Extract background_tasks.py (Day 8)
6. Extract metrics.py (Day 9)
7. Move demo code (Day 10)

**Week 3 (Days 11-15): Phase 5 + Refinement**
8. Extract stat_mech_integration.py (Day 11)
9. Test all extractions (Days 12-13)
10. Add comprehensive docstrings (Days 14-15)

**Week 3+ (Optional): Deeper Extraction**
11. Extract weaving cycle stages (if needed to reach <1,500 lines)
12. Extract helper methods (if needed)

---

## Testing Strategy

### Per-Extraction Testing
After each extraction:
1. Run `pytest HoloLoom/tests/integration/test_full_pipeline.py -v`
2. Run `pytest HoloLoom/tests/e2e/test_orchestrator_9_step_cycle.py -v`
3. Verify imports resolve correctly
4. Verify all initialization still works

### Regression Testing
After all extractions:
1. Run full test suite: `pytest HoloLoom/tests/ -v`
2. Run demos:
   - `python demos/orchestrator_demo.py`
   - `python demos/demo_memory_symphony_integration.py`
3. Verify performance (should be same or better due to better code organization)

---

## Benefits

### Code Organization
- **Single Responsibility**: Each module has one clear purpose
- **Discoverability**: Easier to find relevant code
- **Testability**: Smaller units are easier to test
- **Maintainability**: Changes isolated to specific modules

### Performance
- **Faster Imports**: Only load what you need
- **Better Caching**: Python bytecode caching more granular
- **Easier Profiling**: Identify bottlenecks per module

### Documentation
- **API Clarity**: Public API clearly separated from internals
- **Module Docstrings**: Each module documents its purpose
- **Import Simplicity**: `from HoloLoom.orchestrator import WeavingOrchestrator` still works

---

## Risks and Mitigations

### Risk 1: Circular Imports
**Mitigation**: Use TYPE_CHECKING for type hints, runtime imports only when needed

### Risk 2: Broken Tests
**Mitigation**: Test after each extraction, not all at once

### Risk 3: API Changes
**Mitigation**: Keep public API unchanged, only internal refactoring

### Risk 4: Performance Regression
**Mitigation**: Benchmark before/after, profile if needed

---

## Success Criteria

✅ **Primary Goals**:
- weaving_orchestrator.py < 1,500 lines (58.8% reduction)
- All tests passing (120+ tests)
- No public API changes
- 100% docstring coverage on public methods
- Cyclomatic complexity <10 per method

✅ **Secondary Goals**:
- Better code organization (14+ new modules)
- Improved maintainability
- Faster onboarding for new developers
- Clearer separation of concerns

---

**Ready to Begin**: Phase 1 (Extract ToolExecutor) ✓

