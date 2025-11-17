# Advanced Prompt Chaining Guide

Complete guide to creating sophisticated prompt chain workflows with Promptly.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Advanced Chain Examples](#advanced-chain-examples)
4. [Visualization & Debugging](#visualization--debugging)
5. [Performance Optimization](#performance-optimization)
6. [Chain Composition](#chain-composition)
7. [Monitoring & Observability](#monitoring--observability)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)

---

## Introduction

Advanced prompt chaining enables you to build complex, production-ready AI workflows that go beyond simple sequential prompts. This guide covers:

- **Multi-stage pipelines** with sophisticated control flow
- **Parallel execution** for performance
- **Error handling and recovery** for reliability
- **Real-time monitoring** for observability
- **Cost optimization** techniques
- **Debugging tools** for development

## Core Concepts

### Chain Definition Structure

Every chain is defined in YAML with this structure:

```yaml
name: my_chain
description: What this chain does
version: "1.0"

variables:
  # Configuration variables
  timeout: 30
  max_retries: 3

steps:
  # Chain steps
  - name: step1
    type: processor_type
    config: {}
```

### Processor Types

Promptly provides several built-in processors:

| Processor | Purpose | Use Case |
|-----------|---------|----------|
| `transform` | Data transformation | Extract, validate, format data |
| `parallel` | Parallel execution | Run multiple tasks concurrently |
| `conditional` | Branching logic | Execute based on conditions |
| `loop` | Iteration | Process arrays, retry logic |
| `retry` | Error recovery | Automatic retry with backoff |
| `execute` | Prompt execution | Run LLM prompts |

### Dependency Management

Steps can depend on previous steps:

```yaml
steps:
  - name: step1
    type: transform
    # No dependencies - runs first

  - name: step2
    type: execute
    depends_on: [step1]  # Runs after step1

  - name: step3
    type: parallel
    depends_on: [step1]  # Also runs after step1 (parallel with step2)

  - name: step4
    depends_on: [step2, step3]  # Runs after both complete
```

## Advanced Chain Examples

### 1. Research Pipeline

Multi-stage research with parallel source consultation:

```yaml
name: research_pipeline
steps:
  # Decompose query
  - name: decompose_query
    type: retry
    config:
      max_attempts: 3
      action:
        type: execute
        prompt: "Break down: {query} into sub-questions"

  # Parallel research
  - name: parallel_research
    type: parallel
    depends_on: [decompose_query]
    config:
      tasks:
        - name: academic_search
          prompt: "Search academic sources: {sub_questions}"
        - name: web_search
          prompt: "Search web: {sub_questions}"
        - name: kb_search
          prompt: "Search knowledge base: {sub_questions}"
      aggregation: all
      timeout: 60.0

  # Synthesize findings
  - name: synthesize
    type: retry
    depends_on: [parallel_research]
    config:
      action:
        type: execute
        prompt: "Synthesize: {research_results}"
```

**Key Features:**
- Query decomposition for better results
- Parallel source consultation (3x speedup)
- Retry for reliability
- Evidence synthesis

**Estimated Performance:**
- Duration: ~30-45s (vs. 90s sequential)
- Cost: ~$0.03 per query
- Success rate: 95%+

### 2. Adaptive Content Generation

Dynamic content adaptation based on audience:

```yaml
name: adaptive_content
steps:
  # Detect audience
  - name: detect_audience
    type: retry
    config:
      action:
        type: execute
        prompt: "Analyze audience for: {topic}"

  # Select template
  - name: select_template
    type: conditional
    depends_on: [detect_audience]
    config:
      conditions:
        - type: keyword
          field: audience.technical_level
          keywords: ["expert"]
          action: {type: set, field: template, value: technical_deep_dive}
        - type: keyword
          field: audience.technical_level
          keywords: ["intermediate"]
          action: {type: set, field: template, value: balanced_explainer}
      default: {type: set, field: template, value: simple_introduction}

  # Iterative refinement
  - name: refinement_loop
    type: loop
    depends_on: [select_template]
    config:
      loop_type: while
      condition: "quality_score < 0.85 and iteration < 5"
      action:
        type: sequence
        steps:
          - type: execute
            prompt: "Generate content using {template}"
          - type: execute
            prompt: "Critique: {content}"
          - type: execute
            prompt: "Refine based on: {critique}"
```

**Key Features:**
- Audience detection
- Dynamic template selection
- Iterative quality improvement
- Multi-format output

**Estimated Performance:**
- Duration: ~60-90s
- Cost: $0.05-0.08
- Final quality: 0.85+ (on 0-1 scale)

### 3. Code Review Chain

Comprehensive code analysis with security scanning:

```yaml
name: code_review_chain
steps:
  # Parse code
  - name: parse_code
    type: retry
    config:
      action:
        type: execute
        prompt: "Parse {language} code structure: {code}"

  # Parallel analysis
  - name: comprehensive_analysis
    type: parallel
    depends_on: [parse_code]
    config:
      tasks:
        - name: security_scan
          prompt: "Security vulnerabilities in: {code}"
        - name: performance_analysis
          prompt: "Performance issues in: {code}"
        - name: best_practices
          prompt: "Best practices check: {code}"
        - name: documentation_analysis
          prompt: "Documentation gaps in: {code}"
      aggregation: all
      timeout: 90.0

  # Generate suggestions
  - name: generate_suggestions
    type: retry
    depends_on: [comprehensive_analysis]
    config:
      action:
        type: execute
        prompt: "Refactoring suggestions for: {issues}"

  # Generate tests
  - name: generate_tests
    type: loop
    depends_on: [parse_code]
    config:
      loop_type: for_each
      items: testable_components
      action:
        type: execute
        prompt: "Generate unit tests for: {item}"
```

**Key Features:**
- Multi-language support
- Parallel analysis (4 dimensions)
- Automated refactoring
- Test generation

**Estimated Performance:**
- Duration: ~45-60s
- Cost: $0.06-0.10
- Issues detected: 10-20 on average

## Visualization & Debugging

### Interactive HTML Visualization

Generate interactive visualizations with D3.js:

```python
from promptly.chain_viz_advanced import visualize_chain_advanced

# Create visualization
html = visualize_chain_advanced(
    chain_def=my_chain,
    execution_trace=trace,
    title="Research Pipeline"
)

# Save to file
with open('chain_viz.html', 'w') as f:
    f.write(html)
```

**Features:**
- Interactive dependency graph
- Gantt chart timeline
- Bottleneck highlighting
- Cost breakdown
- Real-time metrics

### Debugging with Breakpoints

Use the debugger for step-by-step execution:

```python
from promptly.chain_debugger import create_debugger, add_conditional_breakpoint

# Create debugger
debugger = create_debugger(chain_def)

# Add breakpoints
debugger.add_breakpoint("synthesize_evidence")
add_conditional_breakpoint(debugger, "filter_chunks", "score < 0.5")

# Set callback for breakpoint hits
def on_breakpoint(bp, frame):
    print(f"Breakpoint hit: {bp.step_name}")
    print(f"Variables: {debugger.inspect_variables()}")
    print(f"Call stack: {debugger.get_call_stack()}")

    # Step over or continue
    return DebugAction.STEP_OVER

debugger.on_breakpoint_hit = on_breakpoint

# Execute with debugging
result = execute_chain_with_debugger(chain_def, debugger)
```

**Debugging Features:**
- Step-by-step execution
- Conditional breakpoints
- Variable inspection
- Call stack viewing
- Expression evaluation
- Execution replay

## Performance Optimization

### Automatic Optimization Analysis

```python
from promptly.chain_optimizer import optimize_chain

# Analyze chain
suggestions = optimize_chain(chain_def, execution_trace)

# Print report
optimizer = ChainOptimizer()
optimizer.analyze(chain_def, execution_trace)
print(optimizer.generate_report())
```

**Optimization Categories:**

1. **Parallelization**
   - Find steps that can run in parallel
   - Estimated speedup: 2-4x

2. **Caching**
   - Identify expensive operations to cache
   - Hit rate improvement: 50-90%

3. **Elimination**
   - Remove redundant steps
   - Cost reduction: 10-30%

4. **Cost Optimization**
   - Reduce retry frequency
   - Use circuit breakers
   - Savings: $0.01-0.05 per execution

5. **Performance Tuning**
   - Identify bottlenecks
   - Suggest timeouts
   - Speed improvement: 20-50%

### Manual Optimization Techniques

#### 1. Parallelize Independent Steps

Before:
```yaml
steps:
  - name: step1
    depends_on: [input]
  - name: step2
    depends_on: [input]
  - name: step3
    depends_on: [input]
```

After:
```yaml
steps:
  - name: parallel_group
    type: parallel
    config:
      tasks:
        - name: step1
        - name: step2
        - name: step3
```

**Impact:** 3x speedup

#### 2. Add Circuit Breakers

```yaml
- name: external_api_call
  type: retry
  config:
    max_attempts: 3
    circuit_breaker:
      enabled: true
      failure_threshold: 5
      success_threshold: 2
      timeout: 60.0
```

**Impact:** Prevents cascade failures

#### 3. Use Timeouts

```yaml
- name: slow_step
  type: parallel
  config:
    timeout: 30.0  # Fail fast if taking too long
```

**Impact:** Better user experience

## Chain Composition

### Combining Multiple Chains

```python
from promptly.chain_composer import ChainComposer

composer = ChainComposer()

# Register chains
composer.register_chain(research_chain)
composer.register_chain(summarization_chain)

# Compose sequentially
composed = composer.compose_sequential(
    chain_names=["research_chain", "summarization_chain"],
    name="research_and_summarize",
    share_context=True
)

# Compose in parallel
parallel_composed = composer.compose_parallel(
    chain_names=["chain_a", "chain_b", "chain_c"],
    name="parallel_processing",
    aggregation="all"
)

# Conditional composition
conditional_composed = composer.compose_conditional(
    condition_chain="check_complexity",
    true_chain="complex_processing",
    false_chain="simple_processing",
    name="adaptive_pipeline"
)
```

### Template Chains

```python
# Create template
template = composer.create_template_chain(
    base_chain="content_generation",
    name="content_template",
    parameters={
        "audience": "intermediate",
        "tone": "professional",
        "length": "medium"
    }
)

# Instantiate with different parameters
instance = composer.instantiate_template(
    template_name="content_template",
    instance_name="blog_post",
    parameter_values={
        "audience": "expert",
        "tone": "technical",
        "length": "long"
    }
)
```

## Monitoring & Observability

### Real-time Monitoring

```python
from promptly.chain_monitor import get_global_monitor

monitor = get_global_monitor()

# Record executions
monitor.record_execution(
    chain_name="research_pipeline",
    duration=42.3,
    cost=0.034,
    success=True
)

# Register alert callback
def alert_handler(alert):
    if alert.severity == "critical":
        send_notification(alert.message)

monitor.register_alert_callback(alert_handler)

# Get dashboard
dashboard_data = monitor.get_dashboard_data()
print(f"Success rate: {dashboard_data['summary']['overall_success_rate']}")

# Generate HTML dashboard
html = monitor.generate_html_dashboard()
```

**Monitoring Features:**
- Real-time metrics
- Success/failure rates
- Cost tracking
- Performance trends
- Anomaly detection
- Automated alerts

### Metrics Tracked

- **Execution count** (total, successful, failed)
- **Duration** (total, average, per step)
- **Cost** (total, average, per step)
- **Success rate** (percentage)
- **Error frequency** (by type)
- **Bottlenecks** (slowest steps)

## Best Practices

### 1. Error Handling

Always use retry with backoff:

```yaml
- name: critical_step
  type: retry
  config:
    max_attempts: 3
    backoff_strategy: exponential
    initial_delay: 1.0
    fallback:
      type: set
      value: "default_value"
```

### 2. Timeouts

Set appropriate timeouts:

```yaml
- name: external_call
  type: parallel
  config:
    timeout: 30.0  # Don't wait forever
```

### 3. Validation

Validate inputs and outputs:

```yaml
- name: validate_input
  type: transform
  config:
    transform_type: validate
    validation_rules:
      - type: required
      - type: type
        expected_type: string
```

### 4. Logging

Include metadata for debugging:

```yaml
variables:
  log_level: debug
  trace_id: "{request_id}"
```

### 5. Cost Control

Monitor and limit costs:

```yaml
variables:
  max_cost_per_execution: 0.10
  enable_cost_tracking: true
```

### 6. Progressive Enhancement

Start simple, add complexity:

1. Build basic chain
2. Add error handling
3. Optimize performance
4. Add monitoring
5. Implement caching

### 7. Testing

Test chains before production:

```python
# Unit test individual steps
def test_decompose_query():
    result = execute_step("decompose_query", test_input)
    assert len(result.sub_questions) >= 3

# Integration test full chain
def test_research_pipeline():
    result = execute_chain(research_pipeline, test_query)
    assert result.quality_score > 0.8

# Load test
def test_performance():
    durations = [execute_chain(chain, input) for _ in range(100)]
    assert statistics.mean(durations) < 60.0
```

## API Reference

### Chain Execution

```python
from promptly.chain_dsl import ChainDSL

dsl = ChainDSL()

# Load chain
chain_def = dsl.load_chain("chain.yaml")

# Execute
result = dsl.execute_chain(chain_def, initial_input={"query": "..."})

# Access results
print(result["results"])
print(result["trace"])
```

### Visualization

```python
from promptly.chain_viz_advanced import AdvancedChainVisualizer

viz = AdvancedChainVisualizer()
viz.build_graph_from_chain(chain_def)
viz.set_execution_trace(trace)
html = viz.to_interactive_html()
```

### Debugging

```python
from promptly.chain_debugger import ChainDebugger

debugger = ChainDebugger(chain_def)
debugger.add_breakpoint("step_name")
debugger.step_over()
variables = debugger.inspect_variables()
```

### Optimization

```python
from promptly.chain_optimizer import ChainOptimizer

optimizer = ChainOptimizer()
suggestions = optimizer.analyze(chain_def, trace)
report = optimizer.generate_report()
```

### Composition

```python
from promptly.chain_composer import ChainComposer

composer = ChainComposer()
composed = composer.compose_sequential(["chain1", "chain2"], "combined")
```

### Monitoring

```python
from promptly.chain_monitor import ChainMonitor

monitor = ChainMonitor()
monitor.record_execution("chain", 42.0, 0.03, True)
metrics = monitor.get_metrics("chain")
```

---

## Next Steps

1. Review [CHAIN_PATTERNS.md](CHAIN_PATTERNS.md) for design patterns
2. Check [TROUBLESHOOTING_CHAINS.md](TROUBLESHOOTING_CHAINS.md) for common issues
3. Explore example chains in `examples/advanced_chains/`
4. Run `advanced_chaining_demo.py` to see chains in action

## Support

- **Documentation:** See docs/ directory
- **Examples:** See examples/ directory
- **Issues:** Report on GitHub
- **Community:** Join Discord/Slack

---

**Version:** 1.0
**Last Updated:** 2025-11-17
