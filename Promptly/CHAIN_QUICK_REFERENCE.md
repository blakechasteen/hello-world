# Chain Processing Quick Reference

## Processor Quick Reference

### ConditionalProcessor

```yaml
- name: conditional_step
  type: conditional
  config:
    conditions:
      - type: numeric         # or: regex, keyword, custom
        field: score
        operator: gt          # gt, lt, eq, ne, ge, le
        value: 0.7
        action:
          type: set
          field: result
          value: "passed"
    default:
      type: set
      field: result
      value: "failed"
```

### ParallelProcessor

```yaml
- name: parallel_step
  type: parallel
  config:
    tasks:
      - name: task1
        prompt: "Process {input}"
    aggregation: concat       # concat, voting, weighted, first, all, best
    error_strategy: best_effort  # fail_fast, best_effort, retry, fallback
    timeout: 30.0
    max_workers: 5
```

### LoopProcessor

```yaml
- name: loop_step
  type: loop
  config:
    loop_type: for_each      # for_each, while, map, reduce, accumulate
    items: items_field
    max_iterations: 100
    action:
      type: process
    accumulator: results
    break_on:
      field: count
      operator: gt
      value: 10
```

### RetryProcessor

```yaml
- name: retry_step
  type: retry
  config:
    max_attempts: 3
    backoff_strategy: exponential  # constant, linear, exponential, fibonacci, jitter
    initial_delay: 1.0
    circuit_breaker:
      enabled: true
      failure_threshold: 5
      timeout: 60.0
    action:
      type: execute
    fallback:
      type: set
      value: "fallback_value"
```

### TransformProcessor

```yaml
- name: transform_step
  type: transform
  config:
    transform_type: extract   # extract, convert, template, validate, sanitize
    method: json              # json, regex, split, csv, key_value
    source_field: input
    target_field: output
```

## Common Patterns

### Extract and Validate

```yaml
- name: extract
  type: transform
  config:
    transform_type: extract
    method: json
    source_field: input

- name: validate
  type: transform
  depends_on: [extract]
  config:
    transform_type: validate
    validation_rules:
      - type: required
      - type: type
        expected_type: dict
```

### Parallel with Retry

```yaml
- name: parallel_with_retry
  type: parallel
  config:
    tasks:
      - name: task1
        prompt: "Process: {input}"
    aggregation: all
    error_strategy: retry
    retry_attempts: 3
    timeout: 30.0
```

### Loop with Condition

```yaml
- name: conditional_loop
  type: loop
  config:
    loop_type: for_each
    items: items
    action:
      type: process
    continue_on:
      field: skip
      operator: eq
      value: true
    break_on:
      field: error_count
      operator: gt
      value: 5
```

### Chain with Fallback

```yaml
- name: main_process
  type: retry
  config:
    max_attempts: 3
    action:
      type: execute
      prompt: "Main: {input}"
    fallback:
      type: set
      value: "Using cached result"
```

## Python Quick Start

### Basic Usage

```python
from promptly.chain_dsl import ChainDSL

dsl = ChainDSL()
dsl.set_executor(your_model_function)

chain_def = dsl.load_chain("workflow.yaml")
result = dsl.execute_chain(chain_def, {"input": "test"})
```

### With Tracing

```python
from promptly.chain_tracing import create_tracer

tracer = create_tracer(trace_level="standard")

dsl = ChainDSL()
# ... execute with tracing ...

summary = tracer.get_summary()
print(f"Duration: {summary['total_duration']:.2f}s")
```

### Visualization

```python
from promptly.chain_visualization import visualize_chain

mermaid = visualize_chain(chain_def, format="mermaid")
print(mermaid)
```

## Workflow Template

```yaml
name: my_workflow
description: Workflow description
version: "1.0"

variables:
  timeout: 30
  max_retries: 3

steps:
  # Extract and validate
  - name: extract
    type: transform
    config:
      transform_type: extract
      method: json

  - name: validate
    type: transform
    depends_on: [extract]
    config:
      transform_type: validate

  # Process
  - name: process
    type: conditional
    depends_on: [validate]
    config:
      conditions:
        - type: keyword
          field: is_valid
          keywords: ["true"]
          action:
            type: execute

  # Output
  - name: format_output
    type: transform
    depends_on: [process]
    config:
      transform_type: template
      template: "Result: {result}"
```

## Error Handling

### With Fallback

```yaml
- name: with_fallback
  type: retry
  config:
    max_attempts: 3
    fallback:
      type: set
      value: "default_value"
```

### Best Effort

```yaml
- name: best_effort
  type: parallel
  config:
    error_strategy: best_effort
    tasks: [...]
```

### Circuit Breaker

```yaml
- name: with_circuit_breaker
  type: retry
  config:
    circuit_breaker:
      enabled: true
      failure_threshold: 5
      success_threshold: 2
      timeout: 60.0
```

## Validation Rules

```yaml
validation_rules:
  - type: required
  - type: type
    expected_type: string
  - type: regex
    pattern: "^[a-z]+$"
  - type: range
    min: 0
    max: 100
```

## Sanitization Rules

```yaml
sanitize_rules:
  - type: strip
  - type: normalize_whitespace
  - type: html_escape
  - type: lowercase
  - type: truncate
    max_length: 500
  - type: remove_special_chars
    allowed: "._-"
```

## Aggregation Strategies

- **concat**: Join all outputs as string
- **voting**: Majority vote
- **weighted**: Weighted combination
- **first**: First successful result
- **all**: Return all results as list
- **best**: Highest scoring result

## Backoff Strategies

- **constant**: Fixed delay
- **linear**: Linear increase
- **exponential**: Exponential growth
- **fibonacci**: Fibonacci sequence
- **jitter**: Exponential + random jitter

## Trace Levels

- **minimal**: Step names and status only
- **standard**: + timing and summaries
- **detailed**: + full inputs/outputs
- **debug**: + internal state

## Export Formats

### Visualization

- **mermaid**: Mermaid diagram syntax
- **graphviz**: DOT format
- **ascii**: ASCII art
- **json**: JSON graph structure

### Tracing

- **json**: JSON export
- **csv**: CSV format
- **markdown**: Markdown report

## Common Issues

### Issue: Circular dependency

```
Error: Cycle detected in chain dependencies
```

**Solution**: Check `depends_on` - no step should depend on itself or create a cycle.

### Issue: Unknown processor

```
Error: Unknown processor type: 'my_processor'
```

**Solution**: Register custom processor:
```python
dsl.register_processor("my_processor", MyProcessor())
```

### Issue: Circuit breaker open

```
State: open
```

**Solution**: Reset circuit breaker:
```python
processor.reset_circuit("circuit_key")
```

## Performance Tips

1. **Use parallel processing** for independent tasks
2. **Set appropriate timeouts** to prevent hanging
3. **Enable circuit breakers** for unreliable services
4. **Limit iterations** in loops
5. **Use best_effort** error strategy when partial results acceptable
6. **Monitor slow steps** with tracing
7. **Cache results** when possible
8. **Batch similar operations**

## Examples Location

- RAG Pipeline: `examples/workflows/rag_pipeline.yaml`
- A/B Testing: `examples/workflows/ab_testing.yaml`
- Multi-Agent: `examples/workflows/multi_agent.yaml`
- Demo Script: `examples/chain_processing_demo.py`

## Full Documentation

See `CHAIN_PROCESSING.md` for complete documentation.
