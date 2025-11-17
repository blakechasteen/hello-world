# Chain Design Patterns

Common patterns and best practices for building robust prompt chains.

## Table of Contents

1. [Structural Patterns](#structural-patterns)
2. [Control Flow Patterns](#control-flow-patterns)
3. [Error Handling Patterns](#error-handling-patterns)
4. [Performance Patterns](#performance-patterns)
5. [Data Flow Patterns](#data-flow-patterns)
6. [Integration Patterns](#integration-patterns)
7. [Anti-Patterns](#anti-patterns)

---

## Structural Patterns

### 1. Pipeline Pattern

**Intent:** Process data through a series of transformation stages.

**Structure:**
```
Input → Stage1 → Stage2 → Stage3 → Output
```

**Implementation:**
```yaml
steps:
  - name: stage1_extract
    type: transform
  - name: stage2_process
    type: execute
    depends_on: [stage1_extract]
  - name: stage3_format
    type: transform
    depends_on: [stage2_process]
```

**Use When:**
- Sequential processing needed
- Each stage adds value
- Intermediate results useful

**Example:** ETL pipeline, content generation, data enrichment

---

### 2. Fan-Out/Fan-In Pattern

**Intent:** Distribute work in parallel, then aggregate results.

**Structure:**
```
       ┌──→ Task1 ──┐
Input ─┼──→ Task2 ──┼──→ Aggregate → Output
       └──→ Task3 ──┘
```

**Implementation:**
```yaml
steps:
  - name: fan_out
    type: parallel
    config:
      tasks:
        - name: task1
        - name: task2
        - name: task3
      aggregation: all

  - name: fan_in
    type: transform
    depends_on: [fan_out]
    config:
      transform_type: aggregate
```

**Use When:**
- Independent subtasks
- Results need aggregation
- Performance critical

**Example:** Multi-source research, ensemble models, A/B testing

---

### 3. Map-Reduce Pattern

**Intent:** Apply operation to each item, then reduce results.

**Structure:**
```
Array → Map(transform each) → Reduce(combine) → Result
```

**Implementation:**
```yaml
steps:
  - name: map_phase
    type: loop
    config:
      loop_type: for_each
      items: input_array
      action:
        type: execute
        prompt: "Process: {item}"
      accumulator: mapped_results

  - name: reduce_phase
    type: transform
    depends_on: [map_phase]
    config:
      transform_type: aggregate
      method: reduce
```

**Use When:**
- Operating on collections
- Results can be combined
- Scalability needed

**Example:** Batch processing, summarization, voting

---

## Control Flow Patterns

### 4. Router Pattern

**Intent:** Route to different processing paths based on input classification.

**Structure:**
```
               ┌──→ Path A
Input → Classify ─┼──→ Path B
               └──→ Path C
```

**Implementation:**
```yaml
steps:
  - name: classify
    type: execute
    config:
      prompt: "Classify input: {input}"

  - name: route
    type: conditional
    depends_on: [classify]
    config:
      conditions:
        - field: category
          keywords: ["type_a"]
          action: {type: chain, chain: path_a}
        - field: category
          keywords: ["type_b"]
          action: {type: chain, chain: path_b}
      default: {type: chain, chain: path_c}
```

**Use When:**
- Different input types
- Specialized processing needed
- Dynamic workflow required

**Example:** Customer support routing, content classification, intent handling

---

### 5. Iterative Refinement Pattern

**Intent:** Improve result through multiple iterations.

**Structure:**
```
Generate → Critique → Refine → Check Quality ─┐
   ↑                                          │
   └──────── (if quality < threshold) ────────┘
```

**Implementation:**
```yaml
steps:
  - name: refinement_loop
    type: loop
    config:
      loop_type: while
      condition: "quality < 0.85 and iteration < 5"
      action:
        type: sequence
        steps:
          - type: execute
            prompt: "Generate: {topic}"
          - type: execute
            prompt: "Critique: {content}"
          - type: execute
            prompt: "Refine based on: {critique}"
          - type: transform
            config:
              operation: calculate_quality
```

**Use When:**
- Quality is critical
- Incremental improvement possible
- Multiple iterations acceptable

**Example:** Content generation, code refactoring, answer refinement

---

### 6. Consensus Pattern

**Intent:** Get consensus from multiple independent sources.

**Structure:**
```
       ┌──→ Source1 ──┐
Input ─┼──→ Source2 ──┼──→ Calculate Consensus → Final
       └──→ Source3 ──┘
```

**Implementation:**
```yaml
steps:
  - name: parallel_sources
    type: parallel
    config:
      tasks:
        - name: source1
        - name: source2
        - name: source3

  - name: calculate_consensus
    type: execute
    depends_on: [parallel_sources]
    config:
      prompt: "Find consensus: {results}"

  - name: check_agreement
    type: conditional
    depends_on: [calculate_consensus]
    config:
      conditions:
        - field: consensus_score
          operator: ge
          value: 0.7
          action: {type: set, field: status, value: agreed}
      default: {type: set, field: status, value: disagreed}
```

**Use When:**
- Accuracy critical
- Multiple sources available
- Validation needed

**Example:** Multi-model consensus, fact-checking, quality assurance

---

## Error Handling Patterns

### 7. Retry with Exponential Backoff

**Intent:** Automatically retry failed operations with increasing delays.

**Implementation:**
```yaml
- name: reliable_step
  type: retry
  config:
    max_attempts: 5
    backoff_strategy: exponential
    initial_delay: 1.0
    max_delay: 30.0
    jitter: true  # Add randomness to prevent thundering herd
```

**Use When:**
- Transient failures expected
- External service calls
- Network operations

**Best Practices:**
- Set max attempts (3-5)
- Use exponential backoff
- Add jitter for distributed systems
- Implement circuit breaker for persistent failures

---

### 8. Circuit Breaker Pattern

**Intent:** Prevent cascading failures by stopping calls to failing services.

**Implementation:**
```yaml
- name: external_service_call
  type: retry
  config:
    circuit_breaker:
      enabled: true
      failure_threshold: 5    # Open after 5 failures
      success_threshold: 2    # Close after 2 successes
      timeout: 60.0           # Try again after 60s
```

**States:**
- **Closed:** Normal operation
- **Open:** Fail fast without attempting call
- **Half-Open:** Allow limited calls to test recovery

**Use When:**
- External dependencies
- Cascading failure risk
- Service degradation possible

---

### 9. Fallback Pattern

**Intent:** Provide alternative when primary path fails.

**Implementation:**
```yaml
- name: primary_with_fallback
  type: retry
  config:
    max_attempts: 3
    action:
      type: execute
      prompt: "Primary approach: {input}"
    fallback:
      type: execute
      prompt: "Fallback approach: {input}"
```

**Use When:**
- Degraded service acceptable
- Alternative approaches available
- Reliability critical

**Example:** Cache fallback, simpler model fallback, default response

---

## Performance Patterns

### 10. Eager Parallelization

**Intent:** Execute independent tasks concurrently.

**Implementation:**
```yaml
- name: parallel_execution
  type: parallel
  config:
    tasks:
      - name: task1
      - name: task2
      - name: task3
    max_workers: 5
    timeout: 30.0
```

**Speedup:** N tasks → ~N/workers speedup

**Use When:**
- Tasks are independent
- I/O bound operations
- Total duration > individual task time

---

### 11. Lazy Evaluation Pattern

**Intent:** Delay expensive computation until result is needed.

**Implementation:**
```yaml
steps:
  - name: check_cache
    type: transform
    config:
      operation: check_cache

  - name: expensive_computation
    type: execute
    depends_on: [check_cache]
    condition:
      field: cache_hit
      operator: eq
      value: false
```

**Use When:**
- Computation expensive
- Result may not be needed
- Cache available

---

### 12. Memoization Pattern

**Intent:** Cache expensive operation results.

**Implementation:**
```yaml
- name: cached_operation
  type: transform
  config:
    transform_type: custom
    cache_enabled: true
    cache_ttl: 3600  # 1 hour
    cache_key_fields: [input_a, input_b]
```

**Cache Hit Rate:** 50-90% typical

**Use When:**
- Same inputs repeated
- Computation expensive
- Results deterministic

---

## Data Flow Patterns

### 13. Context Accumulation Pattern

**Intent:** Build up context through pipeline stages.

**Implementation:**
```yaml
steps:
  - name: add_context_1
    type: execute
    config:
      prompt: "Initial context"

  - name: add_context_2
    type: execute
    depends_on: [add_context_1]
    config:
      prompt: "Enhanced with: {context_1}"

  - name: add_context_3
    type: execute
    depends_on: [add_context_2]
    config:
      prompt: "Further enhanced: {context_1}, {context_2}"
```

**Use When:**
- Context builds incrementally
- Each stage adds information
- Final stage needs full context

---

### 14. Branching Merge Pattern

**Intent:** Branch processing, then merge results.

**Implementation:**
```yaml
steps:
  - name: branch_a
    type: execute

  - name: branch_b
    type: execute

  - name: merge
    type: conditional
    depends_on: [branch_a, branch_b]
    config:
      conditions:
        - field: quality_a
          operator: gt
          value_field: quality_b
          action: {field: result, value_field: result_a}
      default: {field: result, value_field: result_b}
```

**Use When:**
- Multiple approaches available
- Best result selected
- Quality comparison possible

---

## Integration Patterns

### 15. Adapter Pattern

**Intent:** Adapt external systems to chain interface.

**Implementation:**
```yaml
- name: api_adapter
  type: transform
  config:
    transform_type: custom
    operation: adapt_api_response
    source_field: external_api_response
    target_format: chain_format
```

**Use When:**
- External system integration
- Format conversion needed
- Abstraction layer desired

---

### 16. Gateway Pattern

**Intent:** Single entry point for multiple backends.

**Implementation:**
```yaml
- name: gateway
  type: conditional
  config:
    conditions:
      - field: backend_type
        keywords: ["api_v1"]
        action: {type: execute, endpoint: "api/v1"}
      - field: backend_type
        keywords: ["api_v2"]
        action: {type: execute, endpoint: "api/v2"}
```

**Use When:**
- Multiple backend versions
- Load balancing needed
- Migration in progress

---

## Anti-Patterns

### ❌ Sequential Bottleneck

**Problem:** Executing independent tasks sequentially.

**Bad:**
```yaml
steps:
  - name: task1
  - name: task2
    depends_on: [task1]  # Unnecessary dependency
  - name: task3
    depends_on: [task2]  # Unnecessary dependency
```

**Good:**
```yaml
steps:
  - name: parallel_tasks
    type: parallel
    config:
      tasks: [task1, task2, task3]
```

---

### ❌ Retry Without Backoff

**Problem:** Hammering failing service.

**Bad:**
```yaml
- name: retry_no_backoff
  type: loop
  config:
    max_iterations: 10
    action:
      type: execute
```

**Good:**
```yaml
- name: retry_with_backoff
  type: retry
  config:
    max_attempts: 5
    backoff_strategy: exponential
```

---

### ❌ Missing Error Handling

**Problem:** No fallback or error recovery.

**Bad:**
```yaml
- name: fragile_step
  type: execute
  config:
    prompt: "This might fail"
```

**Good:**
```yaml
- name: robust_step
  type: retry
  config:
    max_attempts: 3
    action:
      type: execute
      prompt: "This might fail"
    fallback:
      type: set
      value: "default_response"
```

---

### ❌ Unbounded Loops

**Problem:** Loop without termination condition.

**Bad:**
```yaml
- name: infinite_loop
  type: loop
  config:
    loop_type: while
    condition: "true"  # Always true!
```

**Good:**
```yaml
- name: bounded_loop
  type: loop
  config:
    loop_type: while
    condition: "iteration < 10 and not satisfied"
    max_iterations: 10
```

---

### ❌ Missing Timeouts

**Problem:** Operations can hang indefinitely.

**Bad:**
```yaml
- name: slow_operation
  type: parallel
  config:
    tasks: [long_task1, long_task2]
```

**Good:**
```yaml
- name: timeout_protected
  type: parallel
  config:
    tasks: [long_task1, long_task2]
    timeout: 30.0
```

---

### ❌ Ignoring Costs

**Problem:** Expensive operations without monitoring.

**Bad:**
```yaml
- name: expensive_loop
  type: loop
  config:
    max_iterations: 1000
    action:
      type: execute  # $0.01 per call = $10 total!
```

**Good:**
```yaml
- name: cost_conscious_loop
  type: loop
  config:
    max_iterations: 10  # Reasonable limit
    action:
      type: execute
variables:
  max_cost_per_execution: 0.50  # Cost guard
```

---

## Pattern Selection Guide

| Scenario | Recommended Pattern |
|----------|-------------------|
| Multiple independent tasks | Fan-Out/Fan-In |
| Process collection | Map-Reduce |
| Route by input type | Router |
| Improve quality | Iterative Refinement |
| Verify accuracy | Consensus |
| Handle transient failures | Retry with Backoff |
| Prevent cascade failures | Circuit Breaker |
| External API integration | Adapter + Circuit Breaker |
| Build up context | Context Accumulation |
| Sequential processing | Pipeline |

## Best Practices Summary

1. **Parallelize aggressively** - Default to parallel when possible
2. **Always handle errors** - Use retry + fallback
3. **Set timeouts** - Prevent hanging operations
4. **Monitor costs** - Track and limit spending
5. **Use appropriate backoff** - Exponential for retries
6. **Implement circuit breakers** - For external dependencies
7. **Cache when possible** - Memoize expensive operations
8. **Validate inputs** - Fail fast on bad data
9. **Log comprehensively** - Enable debugging
10. **Test thoroughly** - Unit, integration, load tests

---

**Version:** 1.0
**Last Updated:** 2025-11-17

See also:
- [ADVANCED_CHAINING_GUIDE.md](ADVANCED_CHAINING_GUIDE.md)
- [TROUBLESHOOTING_CHAINS.md](TROUBLESHOOTING_CHAINS.md)
