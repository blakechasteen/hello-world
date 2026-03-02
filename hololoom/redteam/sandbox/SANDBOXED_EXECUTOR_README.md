# SandboxedExecutor: Transparent Sandbox Integration

## Overview

`SandboxedExecutor` wraps `AttackExecutor` with complete sandbox integration, making security isolation transparent and automatic. It's a **drop-in replacement** for `AttackExecutor` that adds:

- ✅ **Transparent isolation**: All attacks automatically sandboxed
- ✅ **Auto-mode selection**: Automatically chooses best available isolation
- ✅ **Resource monitoring**: <5% overhead resource tracking
- ✅ **Batch execution**: Sequential and parallel batch operations
- ✅ **Complete lifecycle**: Proper async setup/cleanup
- ✅ **Graceful degradation**: Works even if isolation unavailable

**Key philosophy**: *"Isolate attacks, understand constraints, learn safely."*

## Quick Start

### Basic Usage (3 lines)

```python
from HoloLoom.redteam.sandbox import create_sandboxed_executor

async with await create_sandboxed_executor() as executor:
    result = await executor.execute_attack(strategy, payload, context)
```

### Drop-in Replacement

```python
# Before (no sandboxing)
executor = AttackExecutor()

# After (full sandboxing - identical interface)
executor = await create_sandboxed_executor()

# All methods work the same:
result = await executor.execute_attack(...)     # Same signature
results = await executor.execute_batch(...)     # Same signature
```

## Architecture

### Components

1. **SandboxedExecutor** - Main wrapper class
   - Wraps AttackExecutor
   - Manages sandbox lifecycle (setup/teardown)
   - Coordinates isolation components
   - Handles resource monitoring

2. **Isolation Modes** - Hardware-specific isolation
   - `NONE` - No isolation (testing only)
   - `SUBPROCESS` - Basic subprocess with resource limits
   - `CGROUPS` - Linux cgroups + seccomp (Linux only)
   - `DOCKER` - Full Docker container (requires Docker)
   - `AUTO` - Auto-detect best available (recommended)

3. **Resource Monitoring**
   - CPU usage tracking
   - Memory usage tracking
   - I/O operations monitoring
   - <5% overhead

4. **Configuration**
   - SandboxConfig object
   - Customizable timeouts, limits, policies
   - Sensible defaults for security

### Data Flow

```
Attack Payload
    ↓
[SandboxedExecutor]
    ├─ Select isolation mode (auto or configured)
    ├─ Setup sandbox environment
    ├─ Start resource monitoring
    ├─ Route to AttackExecutor
    ├─ Monitor execution
    └─ Collect metrics
    ↓
[AttackResult]
    └─ With sandbox metrics
```

## Usage Patterns

### Pattern 1: Async Context Manager (Recommended)

```python
async with await create_sandboxed_executor() as executor:
    result = await executor.execute_attack(
        AttackStrategy.UNICODE_BYPASS,
        "ignore\\u200bore previous",
        {}
    )
```

**Benefits**: Automatic cleanup, exception-safe

### Pattern 2: Manual Lifecycle

```python
executor = create_sandboxed_executor_sync()
await executor.setup()

try:
    result = await executor.execute_attack(...)
finally:
    await executor.close()
```

**Benefits**: Full control over timing

### Pattern 3: One-Off Execution

```python
result = await sandboxed_attack_execution(
    AttackStrategy.UNICODE_BYPASS,
    "payload",
    {},
    config=SandboxConfig(mode=SandboxMode.DOCKER)
)
```

**Benefits**: Simplest for single executions

### Pattern 4: Batch Processing

```python
async with await create_sandboxed_executor() as executor:
    payloads = [AttackPayload(...) for _ in range(100)]

    # Sequential
    results = await executor.execute_batch(payloads, parallel=False)

    # Parallel (max 4 concurrent)
    results = await executor.execute_batch(
        payloads,
        parallel=True,
        max_parallel=4
    )
```

## Configuration

### Basic Configuration

```python
config = SandboxConfig(
    mode=SandboxMode.AUTO,          # Auto-select best mode
    timeout_seconds=30.0,            # 30s execution timeout
    memory_limit_mb=512,             # 512 MB memory limit
    cpu_limit_percent=50,            # 50% CPU limit
    network_enabled=False             # Block network by default
)

executor = await create_sandboxed_executor(config=config)
```

### Advanced Configuration

```python
config = SandboxConfig(
    mode=SandboxMode.DOCKER,         # Force Docker mode

    # Execution limits
    timeout_seconds=60.0,
    memory_limit_mb=1024,
    cpu_limit_percent=75,

    # Network policy
    network_enabled=True,
    allowed_hosts=['api.example.com'],
    allowed_ports=[443],

    # Filesystem policy
    filesystem_readonly=True,
    allowed_read_paths=['/etc'],
    allowed_write_paths=['/tmp'],

    # Docker-specific
    docker_image='python:3.11-slim',
    docker_network='none'
)
```

### Configuration Validation

```python
from HoloLoom.redteam.sandbox import validate_sandbox_config

warnings = validate_sandbox_config(config)
if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")
```

## Monitoring & Statistics

### Resource Summary

```python
resources = executor.get_resource_summary()

print(f"Sandbox mode: {resources.sandbox_mode_used.value}")
print(f"Startup: {resources.startup_time_ms:.1f}ms")
print(f"Execution: {resources.execution_time_ms:.1f}ms")
print(f"Cleanup: {resources.cleanup_time_ms:.1f}ms")
print(f"Peak memory: {resources.max_memory_mb:.1f} MB")
print(f"Avg CPU: {resources.avg_cpu_percent:.1f}%")
print(f"Network violations: {len(resources.network_violations)}")
```

### Execution Statistics

```python
stats = executor.get_execution_stats()

print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Avg time: {stats['avg_execution_time_ms']:.1f}ms")
print(f"Total time: {stats['total_execution_time_ms']:.1f}ms")
```

### Blocked Network Attempts

```python
blocked = executor.get_blocked_network_attempts()

for attempt in blocked:
    print(f"Blocked: {attempt['host']}:{attempt['port']}")
```

## Isolation Modes

### SUBPROCESS (Universal)

- **Availability**: All platforms
- **Overhead**: Low (~5-10%)
- **Isolation**: Process boundary + resource limits
- **Use**: Development, testing, general purpose

```python
config = SandboxConfig(mode=SandboxMode.SUBPROCESS)
```

### CGROUPS (Linux only)

- **Availability**: Linux with cgroups v2
- **Overhead**: Medium (~10-20%)
- **Isolation**: cgroups + seccomp syscall filtering
- **Use**: Production on Linux

```python
config = SandboxConfig(mode=SandboxMode.CGROUPS)
```

### DOCKER (Linux/macOS/Windows with Docker)

- **Availability**: Systems with Docker daemon
- **Overhead**: Medium-High (~20-50%)
- **Isolation**: Full container isolation
- **Use**: Maximum security, production

```python
config = SandboxConfig(
    mode=SandboxMode.DOCKER,
    docker_image='python:3.11-slim',
    docker_network='none'
)
```

### AUTO (Recommended)

- **Availability**: All platforms
- **Selection order**: DOCKER > CGROUPS > SUBPROCESS
- **Best for**: Unknown environment

```python
config = SandboxConfig(mode=SandboxMode.AUTO)
# Automatically selects best available
```

## Timeout Handling

Timeouts are handled gracefully with proper cleanup:

```python
config = SandboxConfig(timeout_seconds=5.0)
executor = await create_sandboxed_executor(config=config)

result = await executor.execute_attack(strategy, payload, context)

if result.outcome.value == "timeout":
    print("Execution timed out")
```

**Timeout behavior**:
- Execution stops after timeout
- Resources are cleaned up
- Result has `outcome=TIMEOUT`
- Process is killed cleanly

## Integration with CARTS

### With Attack Learner

```python
from HoloLoom.redteam.sandbox import create_sandboxed_executor
from HoloLoom.redteam.learning import AttackLearner

executor = await create_sandboxed_executor()
learner = AttackLearner()

payloads = learner.generate_payloads(strategy, count=100)
results = await executor.execute_batch(payloads)

# Learn from sandboxed results
for payload, result in zip(payloads, results):
    learner.update(payload, result)
```

### With Attack Tracker

```python
from HoloLoom.redteam.tracker import AttackTracker

tracker = AttackTracker()
async with await create_sandboxed_executor() as executor:
    result = await executor.execute_attack(strategy, payload, context)
    tracker.record(result, sandbox_mode=executor.get_sandbox_mode())
```

### With Attack Reporter

```python
from HoloLoom.redteam.reporter import AttackReporter

reporter = AttackReporter()
async with await create_sandboxed_executor() as executor:
    results = await executor.execute_batch(payloads)

    report = reporter.generate_report(
        results,
        sandbox_config=executor.get_sandbox_config(),
        resources=executor.get_resource_summary()
    )
```

## Performance Characteristics

### Overhead Analysis

| Operation | Overhead | Notes |
|-----------|----------|-------|
| Setup | ~50-100ms | One-time, mode-dependent |
| Execution | <5% | Resource monitoring only |
| Cleanup | ~10-50ms | Mode-dependent |
| Monitoring | <5% | Background collection |

### Throughput

| Mode | Attacks/sec | Latency per attack |
|------|-------------|-------------------|
| SUBPROCESS | 8-12 | 80-125ms |
| CGROUPS | 6-10 | 100-170ms |
| DOCKER | 2-5 | 200-500ms |

### Memory Usage

| Mode | Base | Per execution |
|------|------|---------------|
| SUBPROCESS | 20-30 MB | 10-50 MB |
| CGROUPS | 25-35 MB | 10-50 MB |
| DOCKER | 50-100 MB | 50-150 MB |

## Graceful Degradation

SandboxedExecutor works even if isolation unavailable:

```python
# If Docker not installed
config = SandboxConfig(mode=SandboxMode.DOCKER)
executor = await create_sandboxed_executor(config=config)
# Falls back to CGROUPS or SUBPROCESS automatically

# If psutil not installed
# Falls back to /proc/self/stat on Linux or basic estimates
resources = executor.get_resource_summary()
# Still returns valid metrics
```

## Error Handling

### Execution Errors

```python
try:
    result = await executor.execute_attack(strategy, payload, context)
except Exception as e:
    print(f"Execution error: {e}")
    # SandboxedExecutor returns AttackResult with error metadata
```

### Setup Errors

```python
try:
    executor = await create_sandboxed_executor()
except Exception as e:
    print(f"Setup failed: {e}")
    # Check logs for details
```

### Cleanup Errors

```python
try:
    await executor.close()
except Exception as e:
    print(f"Cleanup error: {e}")
    # Cleanup is best-effort, exceptions logged but not raised
```

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest HoloLoom/redteam/sandbox/tests/test_sandboxed_executor.py -v

# Specific test
pytest HoloLoom/redteam/sandbox/tests/test_sandboxed_executor.py::test_sandboxed_executor_init -v

# With logging
pytest HoloLoom/redteam/sandbox/tests/test_sandboxed_executor.py -v -s
```

## Demo

Run the comprehensive demo:

```bash
python -m HoloLoom.redteam.sandbox.demo_sandboxed_executor
```

Demonstrates:
1. Basic execution
2. Batch sequential
3. Batch parallel
4. Resource monitoring
5. Isolation mode auto-detection
6. Timeout handling
7. Statistics tracking
8. Manual lifecycle
9. Convenience function
10. Drop-in replacement

## FAQ

### Q: Should I use SUBPROCESS, CGROUPS, or DOCKER?

**A**: Use AUTO mode to let the system decide:
- `SandboxMode.AUTO` (recommended) - Best available
- `SandboxMode.SUBPROCESS` - Development/testing
- `SandboxMode.CGROUPS` - Production Linux
- `SandboxMode.DOCKER` - Maximum security

### Q: How much does sandboxing slow down execution?

**A**: <5% for resource monitoring overhead. Setup/teardown varies:
- SUBPROCESS: 50-100ms
- CGROUPS: 50-150ms
- DOCKER: 500-2000ms

### Q: What happens if Docker is unavailable?

**A**: Falls back to CGROUPS, then SUBPROCESS automatically.

### Q: Can I monitor network attempts?

**A**: Yes! Call `executor.get_blocked_network_attempts()`

### Q: How do I customize resource limits?

**A**: Use SandboxConfig:
```python
config = SandboxConfig(
    memory_limit_mb=1024,
    cpu_limit_percent=75,
    timeout_seconds=60
)
```

### Q: Can I execute code besides attacks?

**A**: Yes, wrap any coroutine:
```python
async def my_function():
    # Your code here
    pass

# Manually call within sandbox context
executor = await create_sandboxed_executor()
# Executor is ready to use
```

## Best Practices

1. **Always use context managers** - Guarantees cleanup
   ```python
   async with await create_sandboxed_executor() as executor:
       result = await executor.execute_attack(...)
   ```

2. **Configure for your use case** - Security vs performance
   ```python
   # Development
   config = SandboxConfig(mode=SandboxMode.SUBPROCESS)

   # Production
   config = SandboxConfig(mode=SandboxMode.DOCKER)
   ```

3. **Monitor resource usage** - Understand constraints
   ```python
   resources = executor.get_resource_summary()
   if resources.max_memory_mb > threshold:
       logger.warning("High memory usage")
   ```

4. **Set appropriate timeouts** - Prevent hangs
   ```python
   config = SandboxConfig(timeout_seconds=30)
   ```

5. **Batch when possible** - Better throughput
   ```python
   # Batch execution with parallelism
   results = await executor.execute_batch(payloads, parallel=True)
   ```

## Security Considerations

### What SandboxedExecutor Protects Against

- ✅ Accidental system damage (resource exhaustion)
- ✅ File system access (mount isolation)
- ✅ Network access (egress blocking)
- ✅ Unauthorized process spawning (resource limits)

### What SandboxedExecutor Does NOT Protect Against

- ❌ Privileged execution attacks
- ❌ Kernel vulnerabilities (except via cgroups/seccomp)
- ❌ Timing attacks across VMs
- ❌ Spectre/Meltdown variants

**Use Docker mode for strongest isolation.**

## Logging

Enable debug logging to see sandbox internals:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("HoloLoom.redteam.sandbox")

# Now see all sandbox operations
```

## Related Classes

- `AttackExecutor` - Wrapped attack executor
- `AttackResult` - Result data structure
- `SandboxConfig` - Configuration
- `SandboxMode` - Isolation modes
- `ResourceMonitor` - Resource tracking
- `ResourceSummary` - Resource metrics

## See Also

- [CARTS Red Team System](../README.md)
- [AttackExecutor Documentation](../executor.py)
- [Sandbox Protocols](./protocols.py)
- [Resource Monitoring](./monitor.py)

---

**Author**: CARTS Team
**Date**: 2025-12-05
**Version**: 0.2.0
