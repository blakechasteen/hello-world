# Troubleshooting Chain Execution

Common issues, solutions, and debugging techniques for prompt chains.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Performance Problems](#performance-problems)
3. [Error Handling](#error-handling)
4. [Debugging Techniques](#debugging-techniques)
5. [Monitoring & Alerts](#monitoring--alerts)
6. [FAQ](#faq)

---

## Common Issues

### Issue 1: Chain Hangs Indefinitely

**Symptoms:**
- Chain execution never completes
- No error messages
- CPU/memory usage normal

**Causes:**
1. Missing timeout in parallel processor
2. Infinite loop without termination condition
3. Deadlock in dependencies
4. Waiting for external service

**Solutions:**

```yaml
# Add timeouts to all parallel operations
- name: parallel_step
  type: parallel
  config:
    timeout: 30.0  # Add this

# Add max_iterations to loops
- name: loop_step
  type: loop
  config:
    max_iterations: 100  # Add this
    condition: "..."

# Add timeout to retry
- name: retry_step
  type: retry
  config:
    max_attempts: 3
    timeout_per_attempt: 10.0  # Add this
```

**Debugging:**
```python
from promptly.chain_debugger import create_debugger

debugger = create_debugger(chain_def)
debugger.add_breakpoint("suspected_step")
# Check which step is hanging
```

---

### Issue 2: High Failure Rate

**Symptoms:**
- Success rate < 70%
- Frequent errors in logs
- Inconsistent results

**Causes:**
1. Insufficient retries
2. No error handling
3. Invalid input data
4. Transient service failures

**Solutions:**

```yaml
# Add retry with backoff
- name: unreliable_step
  type: retry
  config:
    max_attempts: 5  # Increase from 3
    backoff_strategy: exponential
    initial_delay: 1.0
    max_delay: 30.0

# Add input validation
- name: validate_input
  type: transform
  config:
    transform_type: validate
    validation_rules:
      - type: required
      - type: type
        expected_type: string
      - type: min_length
        value: 10

# Add circuit breaker for external services
- name: external_call
  type: retry
  config:
    circuit_breaker:
      enabled: true
      failure_threshold: 5
      success_threshold: 2
      timeout: 60.0

# Add fallback for critical steps
- name: critical_step
  type: retry
  config:
    max_attempts: 3
    fallback:
      type: set
      value: "safe_default_value"
```

**Monitoring:**
```python
from promptly.chain_monitor import get_global_monitor

monitor = get_global_monitor()

# Set alert on high failure rate
monitor.failure_rate_threshold = 0.2  # Alert if > 20% failures

# Check metrics
metrics = monitor.get_metrics("my_chain")
print(f"Success rate: {metrics['success_rate']}")
print(f"Recent errors: {metrics['recent_errors']}")
```

---

### Issue 3: Slow Execution

**Symptoms:**
- Chain takes > 60 seconds
- Steps execute sequentially when they could be parallel
- Bottlenecks in specific steps

**Causes:**
1. Sequential execution of independent steps
2. Slow individual steps
3. Too many retries
4. No caching

**Solutions:**

```yaml
# Parallelize independent steps
- name: parallel_optimization
  type: parallel
  config:
    tasks:
      - name: task1  # These were sequential
      - name: task2
      - name: task3
    max_workers: 5  # Increase worker pool

# Add caching
- name: expensive_operation
  type: transform
  config:
    cache_enabled: true
    cache_ttl: 3600

# Reduce unnecessary retries
- name: optimized_retry
  type: retry
  config:
    max_attempts: 3  # Reduced from 5
    initial_delay: 0.5  # Reduced from 1.0
```

**Optimization Analysis:**
```python
from promptly.chain_optimizer import optimize_chain

# Get optimization suggestions
suggestions = optimize_chain(chain_def, execution_trace)

for suggestion in suggestions:
    if suggestion.category == "parallelization":
        print(f"Can parallelize: {suggestion.steps_affected}")
        print(f"Estimated speedup: {suggestion.estimated_improvement}")
```

---

### Issue 4: High Costs

**Symptoms:**
- Bills higher than expected
- Cost per execution > $0.10
- Many expensive operations

**Causes:**
1. Too many iterations in loops
2. No caching of repeated operations
3. Using expensive models unnecessarily
4. Retry without limit

**Solutions:**

```yaml
# Limit loop iterations
- name: expensive_loop
  type: loop
  config:
    max_iterations: 10  # Add limit
    loop_type: for_each

# Add result caching
variables:
  cache_enabled: true
  cache_ttl: 3600

# Use circuit breaker to prevent retry storms
- name: external_api
  type: retry
  config:
    circuit_breaker:
      enabled: true
      failure_threshold: 3

# Set cost budget
variables:
  max_cost_per_execution: 0.05
  enable_cost_tracking: true
```

**Cost Monitoring:**
```python
from promptly.chain_monitor import get_global_monitor

monitor = get_global_monitor()

# Check cost breakdown
dashboard = monitor.get_dashboard_data()
for chain, cost in dashboard['cost_breakdown'].items():
    if cost > 0.05:
        print(f"High cost chain: {chain} = ${cost:.4f}")
```

---

### Issue 5: Circular Dependencies

**Symptoms:**
- Error: "Cycle detected in chain dependencies"
- Chain won't execute
- Validation fails

**Cause:**
```yaml
# Bad - circular dependency
steps:
  - name: step1
    depends_on: [step2]
  - name: step2
    depends_on: [step1]  # Circular!
```

**Solution:**
```yaml
# Good - proper dependency order
steps:
  - name: step1
    # No dependencies
  - name: step2
    depends_on: [step1]
  - name: step3
    depends_on: [step2]
```

**Detection:**
```python
from promptly.chain_dsl import ChainDSL

dsl = ChainDSL()
chain_def = dsl.load_chain("chain.yaml")

# Validate
validation = dsl.validate_chain(chain_def)
if not validation["valid"]:
    print("Errors:", validation["errors"])
```

---

## Performance Problems

### Bottleneck Identification

**Use the optimizer:**
```python
from promptly.chain_optimizer import ChainOptimizer
from promptly.chain_viz_advanced import visualize_chain_advanced

optimizer = ChainOptimizer()
suggestions = optimizer.analyze(chain_def, execution_trace)

# Find bottlenecks
bottleneck_suggestions = [
    s for s in suggestions
    if s.category == "performance" and s.impact == "high"
]

for suggestion in bottleneck_suggestions:
    print(f"Bottleneck: {suggestion.title}")
    print(f"Steps: {suggestion.steps_affected}")
    print(f"Fix: {suggestion.implementation}")

# Visualize bottlenecks
html = visualize_chain_advanced(chain_def, execution_trace)
# Bottlenecks highlighted in red
```

### Parallelization Opportunities

**Detect:**
```python
parallelization_suggestions = [
    s for s in suggestions
    if s.category == "parallelization"
]

for suggestion in parallelization_suggestions:
    print(f"Can parallelize: {suggestion.steps_affected}")
    print(f"Speedup: {suggestion.estimated_improvement}")
```

**Apply:**
```yaml
# Convert to parallel
- name: parallel_group
  type: parallel
  config:
    tasks:
      - name: step1
      - name: step2
      - name: step3
```

---

## Error Handling

### Handling Specific Errors

**API Rate Limits:**
```yaml
- name: api_call
  type: retry
  config:
    max_attempts: 5
    backoff_strategy: exponential
    initial_delay: 2.0  # Start with longer delay
    max_delay: 60.0
    retry_on_errors:
      - "RateLimitError"
      - "429"
```

**Timeout Errors:**
```yaml
- name: slow_operation
  type: retry
  config:
    timeout_per_attempt: 30.0
    max_attempts: 2  # Don't retry timeouts many times
    fallback:
      type: execute
      prompt: "Faster alternative: {input}"
```

**Validation Errors:**
```yaml
- name: validate_and_sanitize
  type: transform
  config:
    transform_type: validate
    on_error: sanitize  # Auto-fix if possible
    validation_rules:
      - type: required
      - type: sanitize
        rules: [strip, normalize_whitespace]
```

### Error Logging

**Enable detailed logging:**
```yaml
variables:
  log_level: debug
  log_errors: true
  error_details: full

steps:
  - name: logged_step
    type: execute
    config:
      log_input: true
      log_output: true
      log_duration: true
```

**Access logs:**
```python
from promptly.chain_tracing import create_tracer

tracer = create_tracer(trace_level="detailed")

# After execution
trace = tracer.get_trace()
for step_trace in trace:
    if step_trace.status == "failed":
        print(f"Failed: {step_trace.step_name}")
        print(f"Error: {step_trace.error}")
        print(f"Input: {step_trace.input_data}")
```

---

## Debugging Techniques

### 1. Step-by-Step Debugging

```python
from promptly.chain_debugger import create_debugger, DebugAction

debugger = create_debugger(chain_def)

# Add breakpoints
debugger.add_breakpoint("critical_step")

# Conditional breakpoint
def condition(variables):
    return variables.get("score", 1.0) < 0.5

debugger.add_breakpoint("quality_check", condition=condition)

# Set callback
def on_breakpoint(bp, frame):
    print(f"\n=== Breakpoint: {bp.step_name} ===")
    print(f"Variables: {debugger.inspect_variables()}")
    print(f"Call stack: {debugger.get_call_stack()}")

    # Interactive prompt
    action = input("Action (c=continue, s=step, i=inspect)? ")
    if action == "c":
        return DebugAction.CONTINUE
    elif action == "s":
        return DebugAction.STEP_OVER
    elif action == "i":
        var = input("Variable to inspect: ")
        print(debugger.evaluate_expression(var))
        return DebugAction.STEP_OVER

debugger.on_breakpoint_hit = on_breakpoint

# Execute with debugging
result = execute_chain_with_debugger(chain_def, debugger, input_data)
```

### 2. Variable Inspection

```python
# At breakpoint
variables = debugger.inspect_variables()
print(json.dumps(variables, indent=2))

# Evaluate expressions
score = debugger.evaluate_expression("quality_score * 100")
is_valid = debugger.evaluate_expression("len(results) > 0")
```

### 3. Execution Replay

```python
# Save execution
debug_session = debugger.export_debug_session()
with open("debug_session.json", "w") as f:
    f.write(debug_session)

# Replay with different input
new_input = {"query": "different test query"}
result = debugger.replay(new_input, executor)
```

### 4. Visualization for Debugging

```python
from promptly.chain_viz_advanced import visualize_chain_advanced

# Generate interactive visualization
html = visualize_chain_advanced(
    chain_def=chain_def,
    execution_trace=trace,
    title="Debug Visualization"
)

# Save and open in browser
with open("debug.html", "w") as f:
    f.write(html)

# Features:
# - Red highlighting for failed steps
# - Dashed lines for error paths
# - Tooltips with error messages
# - Timeline showing where time was spent
```

---

## Monitoring & Alerts

### Setting Up Monitoring

```python
from promptly.chain_monitor import get_global_monitor

monitor = get_global_monitor()

# Configure thresholds
monitor.duration_threshold_multiplier = 2.0
monitor.cost_threshold_multiplier = 2.0
monitor.failure_rate_threshold = 0.3

# Register alert handler
def handle_alert(alert):
    if alert.severity == "critical":
        send_slack_notification(alert.message)
    elif alert.severity == "warning":
        log_warning(alert.message)

monitor.register_alert_callback(handle_alert)

# Record executions
monitor.record_execution(
    chain_name="my_chain",
    duration=45.2,
    cost=0.034,
    success=True
)
```

### Dashboard

```python
# Generate HTML dashboard
html = monitor.generate_html_dashboard()
with open("dashboard.html", "w") as f:
    f.write(html)

# Or get data
dashboard = monitor.get_dashboard_data()
print(f"Total executions: {dashboard['summary']['total_executions']}")
print(f"Success rate: {dashboard['summary']['overall_success_rate']}")
print(f"Total cost: ${dashboard['summary']['total_cost']}")
```

---

## FAQ

### Q: My chain is slow. How do I find the bottleneck?

**A:** Use the optimizer and visualizer:

```python
from promptly.chain_optimizer import optimize_chain
from promptly.chain_viz_advanced import visualize_chain_advanced

# Find bottlenecks
suggestions = optimize_chain(chain_def, trace)
bottlenecks = [s for s in suggestions if "bottleneck" in s.title.lower()]

# Visualize
html = visualize_chain_advanced(chain_def, trace)
# Bottlenecks highlighted in red
```

### Q: How do I reduce costs?

**A:**
1. Add caching
2. Limit loop iterations
3. Use circuit breakers
4. Implement early termination

```yaml
variables:
  max_cost_per_execution: 0.05  # Hard limit

steps:
  - name: cache_check
    type: transform
    config:
      check_cache: true

  - name: expensive_op
    type: execute
    condition:
      field: cache_hit
      value: false  # Only if not cached
```

### Q: Chain fails intermittently. How to make it reliable?

**A:** Add retries, circuit breakers, and fallbacks:

```yaml
- name: reliable_step
  type: retry
  config:
    max_attempts: 5
    backoff_strategy: exponential
    circuit_breaker:
      enabled: true
      failure_threshold: 5
    fallback:
      type: set
      value: "safe_default"
```

### Q: How do I debug a specific step?

**A:** Use breakpoints:

```python
debugger = create_debugger(chain_def)
debugger.add_breakpoint("problem_step")

# Conditional breakpoint
debugger.add_breakpoint("step2", condition=lambda v: v['score'] < 0.5)
```

### Q: Chain works locally but fails in production

**A:** Common causes:
1. **Missing environment variables**
   - Solution: Add validation step
2. **Different timeouts**
   - Solution: Increase timeouts in production
3. **Rate limits**
   - Solution: Add backoff and circuit breakers
4. **Resource constraints**
   - Solution: Reduce parallelism

### Q: How do I test chains before production?

**A:** Comprehensive testing approach:

```python
# 1. Unit test steps
def test_individual_step():
    result = execute_step("step_name", test_input)
    assert result["success"]

# 2. Integration test chain
def test_full_chain():
    result = execute_chain(chain_def, test_input)
    assert result["final_output"]["quality"] > 0.8

# 3. Load test
def test_performance():
    results = [execute_chain(chain_def, test_input) for _ in range(100)]
    avg_duration = statistics.mean(r["duration"] for r in results)
    assert avg_duration < 60.0

# 4. Cost test
def test_cost():
    result = execute_chain(chain_def, test_input)
    assert result["cost"] < 0.10
```

### Q: How do I migrate from v1 to v2 chain?

**A:** Use versioning and gradual rollout:

```python
from promptly.chain_composer import ChainComposer

composer = ChainComposer()
composer.register_chain(chain_v1)
composer.register_chain(chain_v2)

# A/B test approach
def get_chain_for_request(request):
    if random.random() < 0.1:  # 10% to v2
        return chain_v2
    else:
        return chain_v1

# Monitor both versions
monitor.record_execution("chain_v1", ...)
monitor.record_execution("chain_v2", ...)

# Compare metrics
v1_metrics = monitor.get_metrics("chain_v1")
v2_metrics = monitor.get_metrics("chain_v2")
```

---

## Quick Reference

### Common Commands

```python
# Load and execute
from promptly.chain_dsl import ChainDSL
dsl = ChainDSL()
chain = dsl.load_chain("chain.yaml")
result = dsl.execute_chain(chain, {"input": "..."})

# Debug
from promptly.chain_debugger import create_debugger
debugger = create_debugger(chain)
debugger.add_breakpoint("step_name")

# Optimize
from promptly.chain_optimizer import optimize_chain
suggestions = optimize_chain(chain, trace)

# Monitor
from promptly.chain_monitor import get_global_monitor
monitor = get_global_monitor()
monitor.record_execution("chain", 45.0, 0.03, True)

# Visualize
from promptly.chain_viz_advanced import visualize_chain_advanced
html = visualize_chain_advanced(chain, trace)
```

### Error Code Reference

| Error | Cause | Solution |
|-------|-------|----------|
| `CircularDependencyError` | Cycle in dependencies | Remove circular deps |
| `TimeoutError` | Step took too long | Increase timeout or optimize |
| `ValidationError` | Invalid input | Add validation step |
| `RateLimitError` | API rate limit hit | Add backoff |
| `CircuitBreakerOpenError` | Too many failures | Wait for recovery |
| `MaxRetriesExceeded` | All retries failed | Add fallback |

---

**Version:** 1.0
**Last Updated:** 2025-11-17

For more help:
- [ADVANCED_CHAINING_GUIDE.md](ADVANCED_CHAINING_GUIDE.md)
- [CHAIN_PATTERNS.md](CHAIN_PATTERNS.md)
- GitHub Issues: https://github.com/yourrepo/issues
