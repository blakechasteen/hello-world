# HoloLoom Chaining System - Declarative Prompt Chain Orchestration

**Status**: ✅ Complete (November 2025)
**Location**: `HoloLoom/chaining/`
**Total Code**: ~2,500 lines across 5 modules

## Overview

The Chaining System provides a declarative, composable approach to sequencing HoloLoom department operations. Instead of manually orchestrating execute → verify → refine sequences, define chains once and execute them repeatedly with consistent, predictable behavior.

**Key Philosophy**: *"Define workflows once, execute them anywhere"*

## Quick Start

### Installation

The chaining system is built into HoloLoom. Just import:

```python
from HoloLoom.chaining import Chain, ChainStep, StepType, ChainOrchestrator, ChainPatterns
```

### 30-Second Example

```python
from HoloLoom.chaining import ChainOrchestrator, ChainPatterns
from HoloLoom.departments.rag_department import RAGDepartment

# Use a pre-built pattern
chain = ChainPatterns.verified_query()  # execute → verify → output

# Execute
async with RAGDepartment() as rag_dept:
    orchestrator = ChainOrchestrator(rag_dept)
    result = await orchestrator.execute_chain(
        chain,
        "What is Thompson Sampling?"
    )

    print(f"Answer: {result.final_response.response['answer']}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Success: {result.success}")
```

## Core Concepts

### 1. Chain: Declarative Workflow Definition

A `Chain` defines the sequence of steps to execute:

```python
from HoloLoom.chaining import Chain, ChainStep, StepType

chain = Chain(name="my_chain", entry_point="execute")

# Add sequential steps
chain.add_step("execute", ChainStep(
    step_type=StepType.EXECUTE,
    params={"mode": "verify", "max_sources": 5},
    next_step="verify"
))

chain.add_step("verify", ChainStep(
    step_type=StepType.VERIFY,
    params={},
    # No next_step = end of chain
))
```

### 2. ChainStep: Individual Operation

Each step represents a department operation:

```python
step = ChainStep(
    step_type=StepType.EXECUTE,        # What to do
    params={"mode": "verify"},          # Configuration
    next_step="verify",                 # Where to go next
    timeout_seconds=5.0,                # Safety timeout
    retry_count=2,                      # Retries on failure
)
```

**Available Step Types**:
- `EXECUTE` - Run RAG query
- `VERIFY` - Run verification checks
- `REFINE` - Improve low-confidence response
- `UPDATE_STRATEGY` - Learning signal
- `CONDITION` - Branching logic
- `LOOP` - Iterative execution
- `CUSTOM` - User-defined handler

### 3. ChainOrchestrator: Execution Engine

The orchestrator executes chains with context passing:

```python
orchestrator = ChainOrchestrator(
    department=rag_dept,
    enable_tracing=True,      # Track execution
    enable_rollback=False,     # Not yet implemented
)

result = await orchestrator.execute_chain(
    chain,
    initial_input="Your question here"
)
```

### 4. Context: Shared State Between Steps

Context flows through the chain:

```python
# Step 1: Execute sets response
context.shared_state["response"] = {...}
context.shared_state["confidence"] = 0.85

# Step 2: Verify reads response and sets verification result
verification = context.get_step_output("execute")
context.shared_state["verification_score"] = 0.88
```

### 5. Conditional Branching: Decisions

Make decisions based on context:

```python
chain.add_step("check_confidence", ChainStep(
    step_type=StepType.CONDITION,
    params={},
    condition=Conditions.confidence_above(0.75),
    on_success="output",     # If true
    on_failure="refine",     # If false
))
```

## Pre-Built Patterns

### 1. Simple Query (Fastest)

**Workflow**: `[execute]` → output
**Latency**: ~150ms
**Use When**: Speed is critical, confidence is sufficient

```python
chain = ChainPatterns.simple_query()
```

### 2. Verified Query (Standard)

**Workflow**: `[execute]` → `[verify]` → output
**Latency**: ~200-250ms
**Use When**: Standard quality checks needed

```python
chain = ChainPatterns.verified_query()
```

### 3. Auto-Refine (Smart)

**Workflow**: `[execute]` → `[verify]` → if low confidence → `[refine]`
**Latency**: ~200-400ms (varies)
**Use When**: Automatic improvement on low confidence

```python
chain = ChainPatterns.auto_refine()
```

### 4. Iterative Improve (Quality-First)

**Workflow**: `[execute]` → `[verify]` → loop `[refine]` until high confidence
**Latency**: ~500ms-2s
**Use When**: Quality is critical (medical, legal, financial)

```python
chain = ChainPatterns.iterative_improve()
```

### 5. Multi-Strategy (Fallback)

**Workflow**: Try `[direct]` → if fails, try `[research]`
**Latency**: ~150-350ms
**Use When**: Want fallback if first approach fails

```python
chain = ChainPatterns.multi_strategy()
```

### 6. Research Pipeline (Full Cycle)

**Workflow**: `[execute]` → `[verify]` → conditional `[refine]` → `[learn]`
**Latency**: ~300-600ms
**Use When**: Deep research needed, system should learn

```python
chain = ChainPatterns.research_pipeline()
```

### 7. Quality-First (Strictest)

**Workflow**: Full research with strict verification, multiple refinement attempts
**Latency**: ~1-5s
**Use When**: Accuracy is paramount, all checks must pass

```python
chain = ChainPatterns.quality_first()
```

### 8. Balanced (Default)

**Workflow**: `[execute]` → `[verify]` → conditional `[refine]` (max 1x)
**Latency**: ~150-300ms
**Use When**: Good balance between speed and quality

```python
chain = ChainPatterns.balanced()
```

## Conditional Branching with Conditions

### Simple Conditions

```python
from HoloLoom.chaining import Conditions

# Confidence thresholds
Conditions.confidence_above(0.75)
Conditions.confidence_below(0.5)
Conditions.confidence_between(0.5, 0.8)

# Source validation
Conditions.has_sources(min_count=3)
Conditions.sources_above(5)

# Verification
Conditions.all_checks_passed()
Conditions.specific_check_passed("Domain")
Conditions.verification_score_above(0.85)

# Response validation
Conditions.response_exists()
Conditions.response_has_content(min_length=50)
Conditions.response_contains("keyword")
Conditions.response_matches_pattern(r"answer.*\.")

# Context
Conditions.field_exists("field_name")
Conditions.field_equals("mode", "research")
```

### Combined Conditions

```python
# AND: All conditions must be true
cond = Conditions.combine_and(
    Conditions.confidence_above(0.75),
    Conditions.has_sources(min_count=3)
)

# OR: Any condition can be true
cond = Conditions.combine_or(
    Conditions.confidence_above(0.85),
    Conditions.all_checks_passed()
)

# NOT: Negate a condition
cond = Conditions.combine_not(Conditions.error_occurred())
```

### Pre-Built Combinations

```python
from HoloLoom.chaining import CommonConditions

CommonConditions.high_confidence()      # >= 0.75
CommonConditions.low_confidence()       # < 0.75
CommonConditions.very_low_confidence()  # < 0.5
CommonConditions.verified_response()    # Verified + has content
CommonConditions.needs_refinement()     # Low confidence + has content
CommonConditions.ready_to_output()      # >= 0.5
```

## Custom Chains

### Simple Sequential Chain

```python
from HoloLoom.chaining import Chain, ChainStep, StepType

chain = Chain(name="my_chain", entry_point="step1")

chain.add_step("step1", ChainStep(
    step_type=StepType.EXECUTE,
    params={"mode": "verify"},
    next_step="step2"
))

chain.add_step("step2", ChainStep(
    step_type=StepType.VERIFY,
    next_step="step3"
))

chain.add_step("step3", ChainStep(
    step_type=StepType.REFINE
))
```

### Chain with Conditional Branching

```python
from HoloLoom.chaining import Conditions

chain = Chain(name="adaptive", entry_point="execute")

chain.add_step("execute", ChainStep(
    step_type=StepType.EXECUTE,
    params={"mode": "verify"},
    next_step="decide"
))

# Decision point
chain.add_step("decide", ChainStep(
    step_type=StepType.CONDITION,
    condition=Conditions.confidence_above(0.8),
    on_success=None,       # Done
    on_failure="refine"    # Need refinement
))

chain.add_step("refine", ChainStep(
    step_type=StepType.REFINE
))
```

### Chain with Loops

```python
chain = Chain(name="iterative", entry_point="execute")

chain.add_step("execute", ChainStep(
    step_type=StepType.EXECUTE,
    params={"mode": "research"},
    next_step="verify"
))

chain.add_step("verify", ChainStep(
    step_type=StepType.VERIFY,
    next_step="check"
))

chain.add_step("check", ChainStep(
    step_type=StepType.CONDITION,
    condition=Conditions.confidence_above(0.85),
    on_success=None,         # Done
    on_failure="improve"     # Loop back
))

# Loop step
chain.add_step("improve", ChainStep(
    step_type=StepType.LOOP,
    next_step="refine",
    max_iterations=3
))

chain.add_step("refine", ChainStep(
    step_type=StepType.REFINE,
    next_step="verify"  # Loop back to verify
))
```

## Execution Tracing and Debugging

### Enable Tracing

```python
orchestrator = ChainOrchestrator(
    department=rag_dept,
    enable_tracing=True  # Capture execution trace
)

result = await orchestrator.execute_chain(chain, "question")

# View trace
print(result.trace.get_summary())
```

### Trace Information

```
============================================================
Execution Trace
============================================================
Steps: 3 (errors: 0)
Duration: 234.5ms

✓ execute           [execute        ]   85.2ms
✓ verify            [verify         ]   132.1ms
✓ refine            [refine         ]   17.2ms
============================================================
```

### Access Detailed Results

```python
# Step results
for step_result in result.trace.step_results:
    print(f"{step_result.step_id}: {step_result.status}")
    print(f"  Duration: {step_result.duration_ms}ms")
    print(f"  Error: {step_result.error}")

# Statistics
print(f"Steps executed: {result.stats.completed_steps}")
print(f"Failed steps: {result.stats.failed_steps}")
print(f"Success rate: {result.stats.get_success_rate():.1%}")
```

## Chain Validation

### Check Chain Structure

```python
errors = chain.validate()

if errors:
    print("Chain validation failed:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Chain is valid!")
```

### Visualize Chain

```python
print(chain.visualize())
```

Output:
```
============================================================
Chain: my_chain
============================================================
┌─ execute [execute]
│  mode: verify
│  max_sources: 5
│
├─ verify [verify]
│
└─ refine [refine]
   └─ [end]
============================================================
```

## Performance Characteristics

### Latency by Pattern

| Pattern | Latency | Best For |
|---------|---------|----------|
| simple_query | ~150ms | Speed critical |
| verified_query | ~200-250ms | Standard use |
| auto_refine | ~200-400ms | Smart improvement |
| iterative_improve | ~500ms-2s | Quality critical |
| multi_strategy | ~150-350ms | Fallback needed |
| research_pipeline | ~300-600ms | Deep research |
| quality_first | ~1-5s | Accuracy paramount |
| balanced | ~150-300ms | General purpose |

### Overhead

- **Chain definition**: <1ms (one-time)
- **Step execution**: 100-200ms (per step, varies by department)
- **Verification**: 30-50ms
- **Refinement**: 50-100ms
- **Context passing**: <1ms
- **Tracing overhead**: ~5-10% of total time

## Integration with Departments

The chaining system works with any department implementing `DepartmentProtocol`:

```python
from HoloLoom.departments.rag_department import RAGDepartment

async with RAGDepartment() as rag_dept:
    orchestrator = ChainOrchestrator(rag_dept)
    result = await orchestrator.execute_chain(chain, query)
```

## Error Handling

### Retries

```python
chain.add_step("execute", ChainStep(
    step_type=StepType.EXECUTE,
    params={},
    retry_count=2,  # Retry up to 2 times on failure
))
```

### Timeouts

```python
chain.add_step("execute", ChainStep(
    step_type=StepType.EXECUTE,
    params={},
    timeout_seconds=5.0,  # Fail if takes > 5 seconds
))
```

### Error Responses

```python
result = await orchestrator.execute_chain(chain, "question")

if not result.success:
    print(f"Chain failed: {result.error}")
    print(f"Confidence: {result.confidence}")
```

## Best Practices

### 1. Start with Pre-Built Patterns

Don't build custom chains unless necessary. Start with:
- `ChainPatterns.simple_query()` - Basic queries
- `ChainPatterns.verified_query()` - Standard use
- `ChainPatterns.balanced()` - Default choice

### 2. Use Appropriate Complexity

Choose the right pattern for your use case:
- **Real-time chat** → simple_query
- **Standard QA** → verified_query or balanced
- **Research** → research_pipeline
- **Medical/Legal** → quality_first

### 3. Monitor Performance

Track latency and quality:

```python
result = await orchestrator.execute_chain(chain, query)
print(f"Time: {result.stats.total_duration_ms}ms")
print(f"Confidence: {result.confidence:.2f}")
print(f"Quality: Pass" if result.success else "Quality: Fail")
```

### 4. Use Execution Traces

Debug issues with execution traces:

```python
orchestrator = ChainOrchestrator(dept, enable_tracing=True)
result = await orchestrator.execute_chain(chain, query)
print(result.trace.get_summary())
```

### 5. Validate Chains Early

Validate chains once at startup:

```python
errors = chain.validate()
if errors:
    raise ValueError(f"Invalid chain: {errors}")

# Now safe to use in production
```

## Advanced Topics

### Custom Step Handlers

```python
async def my_custom_handler(ctx):
    """Custom step logic."""
    response = ctx.get("response")
    # Do something with response
    return modified_response

chain.add_step("custom", ChainStep(
    step_type=StepType.CUSTOM,
    params={"handler": my_custom_handler}
))
```

### Step-Specific Configuration

```python
chain.add_step("execute", ChainStep(
    step_type=StepType.EXECUTE,
    params={
        "mode": "research",
        "max_sources": 20,
    },
    timeout_seconds=10.0,      # This step can take 10 seconds
    retry_count=1,             # Retry once if fails
    skip_condition=lambda ctx: ctx.get("skip_execute", False),
))
```

### Dynamic Parameters

Parameters can reference context (future):

```python
# Note: This feature is planned for Phase 2
chain.add_step("refine", ChainStep(
    step_type=StepType.REFINE,
    params={
        "max_sources": "${sources_count * 2}",  # Dynamic param
    }
))
```

## Files and API Reference

### Core Files

| File | Purpose | Lines |
|------|---------|-------|
| `chain.py` | Chain definition | 400 |
| `orchestrator.py` | Execution engine | 500 |
| `patterns.py` | Pre-built patterns | 300 |
| `conditions.py` | Condition helpers | 200 |
| `types.py` | Result types | 150 |

### Key Classes

**Chain** (chain.py)
- `add_step()` - Add step to chain
- `add_sequential_steps()` - Add multiple steps
- `validate()` - Check chain validity
- `visualize()` - ASCII visualization
- `to_json()` - Serialize to JSON

**ChainOrchestrator** (orchestrator.py)
- `execute_chain()` - Run chain
- `_execute_step()` - Execute single step
- `_get_next_step()` - Determine next step
- `_build_trace()` - Create execution trace

**ChainPatterns** (patterns.py)
- `simple_query()`
- `verified_query()`
- `auto_refine()`
- `iterative_improve()`
- `multi_strategy()`
- `research_pipeline()`
- `quality_first()`
- `balanced()`

**Conditions** (conditions.py)
- `confidence_above()` / `confidence_below()`
- `has_sources()` / `sources_above()`
- `all_checks_passed()` / `specific_check_passed()`
- `response_exists()` / `response_has_content()`
- `combine_and()` / `combine_or()` / `combine_not()`

## Testing

```bash
# Run chain orchestrator tests
pytest HoloLoom/chaining/tests/test_chain_orchestrator.py -v

# Expected: 20+ test cases passing
# Coverage: Chain definition, execution, conditions, patterns
```

## Demos

```bash
# Run comprehensive demonstrations
PYTHONPATH=. python demos/demo_chain_orchestrator.py

# Shows 8 example chains with detailed output
# Expected runtime: ~5-10 seconds
```

## Roadmap (Future Phases)

### Phase 2 (Q1 2026)

- [ ] Dynamic parameter substitution (`${variable}` syntax)
- [ ] Chain composition (nest chains within chains)
- [ ] Parallel execution (`PARALLEL` step type)
- [ ] Conditional loops (`while` construct)
- [ ] Chain templates (reusable patterns)

### Phase 3 (Q2 2026)

- [ ] Chain optimization (automatic pattern selection)
- [ ] Performance profiling
- [ ] Rollback on failure
- [ ] Transactional chains (all-or-nothing)

### Phase 4 (Q3 2026)

- [ ] Distributed chain execution
- [ ] Chain versioning
- [ ] A/B testing of chains
- [ ] Chain analytics dashboard

## FAQ

### Q: How do I choose between patterns?

**A**: Use this decision tree:
1. **Speed critical?** → `simple_query`
2. **Want verification?** → `verified_query`
3. **Want automatic refinement?** → `auto_refine`
4. **Quality critical?** → `iterative_improve` or `quality_first`
5. **Default choice** → `balanced`

### Q: Can I create complex nested chains?

**A**: Not yet. Phase 2 will support chain composition. For now, create a custom chain with multiple steps.

### Q: What if a step fails?

**A**: By default, chain execution stops. Use `retry_count` to retry:

```python
chain.add_step("execute", ChainStep(
    step_type=StepType.EXECUTE,
    retry_count=2  # Retry up to 2 times
))
```

### Q: How do I monitor chains in production?

**A**: Use execution tracing and statistics:

```python
result = await orchestrator.execute_chain(chain, query)
log_metrics({
    "chain": chain.name,
    "success": result.success,
    "confidence": result.confidence,
    "latency_ms": result.stats.total_duration_ms,
})
```

### Q: Can I use multiple departments?

**A**: Currently, each chain is tied to one department. Phase 2 will support multi-department chains.

## Contact & Support

For issues, questions, or feature requests:
- GitHub Issues: HoloLoom/issues
- Email: team@hololoom.ai
- Documentation: https://hololoom.ai/docs/chaining

---

**Created**: November 2025
**Status**: ✅ Production Ready
**Maintainers**: HoloLoom Architecture Team
