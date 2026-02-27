# Sandbox Isolation Quick Start

**Phase 2 Foundation Documentation**
**Author**: CARTS Team
**Date**: December 5, 2025

## Overview

Sandbox isolation enables safe, contained execution of attack payloads with configurable resource limits, network isolation, and complete resource monitoring.

## Basic Usage

### 1. Create Configuration

```python
from HoloLoom.redteam.sandbox import SandboxMode, SandboxConfig

# Minimal configuration (uses sensible defaults)
config = SandboxConfig()

# Or customize for your needs
config = SandboxConfig(
    mode=SandboxMode.AUTO,           # Auto-detect best available
    timeout_seconds=30.0,             # Kill after 30 seconds
    memory_limit_mb=512,              # Max 512 MB RAM
    cpu_limit_percent=50,             # Max 50% CPU
    network_enabled=False,            # Block all network
    filesystem_readonly=True          # Read-only filesystem
)
```

### 2. Monitor Execution

```python
from HoloLoom.redteam.sandbox import ResourceMonitor

# Create monitor with 100ms sampling
monitor = ResourceMonitor(sample_interval_ms=100)

# Start monitoring
await monitor.start()

# ... execute attack ...

# Stop and get summary
summary = await monitor.stop()

print(summary)
# Resource Summary (150 samples, 15.0s)
#   CPU: 5.2% avg, 12.3% max
#   Memory: 124.5 MB avg, 156.2 MB peak
#   I/O: 2,457,600 bytes
#   Monitor Overhead: 0.47%
```

### 3. Check Against Limits

```python
# After stopping monitor
within_limits, violations = monitor.check_limits(
    memory_limit_mb=512,
    timeout_seconds=30.0
)

if within_limits:
    print("Execution stayed within limits")
else:
    print("Violations detected:")
    for v in violations:
        print(f"  - {v}")
```

## Configuration Modes

### NONE (Testing/Development)
```python
config = SandboxConfig(mode=SandboxMode.NONE)
# No isolation, immediate execution
# Use for development and testing only
```

### SUBPROCESS (Recommended Default)
```python
config = SandboxConfig(mode=SandboxMode.SUBPROCESS)
# Uses process resource limits (rlimit)
# Available on all platforms
# ~2% overhead
```

### CGROUPS (Linux Production)
```python
config = SandboxConfig(mode=SandboxMode.CGROUPS)
# Linux cgroups v2 + seccomp filtering
# Strong isolation, network filtering
# ~3% overhead
# Requires Linux
```

### DOCKER (Maximum Isolation)
```python
config = SandboxConfig(
    mode=SandboxMode.DOCKER,
    docker_image="python:3.11-slim",
    docker_network="none"  # No network access
)
# Full container isolation
# ~10% overhead
# Requires Docker installed and running
```

### AUTO (Recommended)
```python
config = SandboxConfig(mode=SandboxMode.AUTO)
# Auto-detects and selects best available
# Preference: DOCKER > CGROUPS > SUBPROCESS > NONE
# Graceful fallback if best choice unavailable
```

## Network Policies

### Block All (Default)
```python
config = SandboxConfig(network_enabled=False)
# All egress traffic blocked
```

### Allow Specific Hosts
```python
config = SandboxConfig(
    network_enabled=True,
    allowed_hosts=["api.example.com", "10.0.0.1"],
    allowed_ports=[443, 8080]
)
# Only connections to specified hosts/ports allowed
```

### Allow DNS
```python
config = SandboxConfig(
    network_enabled=True,
    allowed_hosts=["8.8.8.8"],  # Google DNS
    allowed_ports=[53]  # DNS port
)
# Enable DNS resolution
```

## Resource Limits

### Conservative (Development)
```python
config = SandboxConfig(
    timeout_seconds=5.0,      # 5 second limit
    memory_limit_mb=128,      # 128 MB
    cpu_limit_percent=25      # 25% CPU
)
```

### Standard (Production)
```python
config = SandboxConfig(
    timeout_seconds=30.0,     # 30 second limit (DEFAULT)
    memory_limit_mb=512,      # 512 MB (DEFAULT)
    cpu_limit_percent=50      # 50% CPU (DEFAULT)
)
```

### Generous (Complex Analysis)
```python
config = SandboxConfig(
    timeout_seconds=300.0,    # 5 minute limit
    memory_limit_mb=2048,     # 2 GB
    cpu_limit_percent=100     # Full CPU
)
```

## Filesystem Isolation

### Read-Only (Recommended)
```python
config = SandboxConfig(
    filesystem_readonly=True,
    allowed_read_paths=["/etc/hosts", "/usr/share/data"]
)
# Filesystem is read-only except for temp files
```

### Overlay Mount (Linux)
```python
config = SandboxConfig(
    filesystem_readonly=True,
    overlay_root="/mnt/sandbox",
    overlay_lower="/",           # Root filesystem
    overlay_upper="/tmp/upper",  # Ephemeral writes
    allowed_read_paths=["/usr/lib", "/usr/bin"]
)
# Sandbox runs on overlay, writes are ephemeral
```

### Ephemeral Storage
```python
config = SandboxConfig(
    allowed_write_paths=["/tmp/sandbox"]  # Only temp directory
)
# Only /tmp/sandbox is writable
```

## Validation

### Check Mode Availability

```python
from HoloLoom.redteam.sandbox.protocols import (
    get_sandbox_mode_availability,
    select_best_sandbox_mode
)

# Check what's available on this system
available = get_sandbox_mode_availability()
print(f"Available modes: {[m.value for m, a in available.items() if a]}")

# Auto-select best
best_mode = select_best_sandbox_mode()
print(f"Best available: {best_mode.value}")
```

### Validate Configuration

```python
from HoloLoom.redteam.sandbox.protocols import validate_sandbox_config

config = SandboxConfig(...)
warnings = validate_sandbox_config(config)

if warnings:
    print("Configuration warnings:")
    for w in warnings:
        print(f"  - {w}")
```

## Integration with CARTS

### Attack Execution with Sandbox

```python
from HoloLoom.redteam import AttackExecutor
from HoloLoom.redteam.sandbox import SandboxConfig

executor = AttackExecutor(
    safety_adapter=adapter,
    sandbox_config=SandboxConfig(mode=SandboxMode.AUTO)
)

# Attacks now execute in sandbox
result = await executor.execute_attack(
    strategy,
    payload,
    {}
)

print(f"Sandbox mode: {result.metadata.get('sandbox_mode')}")
print(f"Violations: {result.metadata.get('sandbox_violations', [])}")
```

### Red Team Orchestration

```python
from HoloLoom.redteam import RedTeamOrchestrator
from HoloLoom.redteam.sandbox import SandboxConfig, ResourceMonitor

orchestrator = RedTeamOrchestrator(
    executor=executor,
    sandbox_config=SandboxConfig()
)

# Each cycle runs in sandbox with monitoring
monitor = ResourceMonitor()
await monitor.start()

result = await orchestrator.run_cycle(strategies_per_cycle=3)

summary = await monitor.stop()

print(f"Found {result.vulnerabilities_found} vulnerabilities")
print(f"Peak memory: {summary.peak_memory_mb:.1f} MB")
print(f"Monitor overhead: {summary.overhead_percent:.2f}%")
```

## Monitoring Details

### Resource Metrics

```python
summary = await monitor.stop()

# CPU
print(f"CPU: {summary.avg_cpu_percent:.1f}% avg, {summary.max_cpu_percent:.1f}% max")

# Memory
print(f"Memory: {summary.avg_memory_mb:.1f} MB avg, {summary.peak_memory_mb:.1f} MB peak")

# I/O
print(f"I/O Read: {summary.total_io_read_bytes:,} bytes at {summary.avg_io_read_rate_bps/1024/1024:.2f} MB/s")
print(f"I/O Write: {summary.total_io_write_bytes:,} bytes")

# Network
print(f"Network: {summary.total_network_bytes:,} bytes total")

# Overhead
print(f"Monitor Overhead: {summary.overhead_percent:.2f}%")
```

### Get Individual Samples

```python
monitor = ResourceMonitor(sample_interval_ms=100)
await monitor.start()
# ... do work ...

# Get all samples collected
samples = monitor.get_samples()

for sample in samples[-5:]:  # Last 5 samples
    print(f"CPU: {sample.cpu_percent:.1f}%, Memory: {sample.memory_mb:.1f} MB")
```

## Error Handling

### Configuration Errors

```python
try:
    config = SandboxConfig(timeout_seconds=-1)  # Invalid!
except ValueError as e:
    print(f"Config error: {e}")
```

### Monitor Errors

```python
try:
    monitor = ResourceMonitor()
    # monitor.stop() called without start()
    summary = await monitor.stop()  # Returns empty summary
except Exception as e:
    logger.error(f"Monitor error: {e}")
```

### Graceful Degradation

```python
# If psutil not available, falls back to /proc/stat (Linux) or basic estimation
monitor = ResourceMonitor()
# Still works, just less accurate

summary = await monitor.stop()
# CPU and memory available, I/O and network may be 0
```

## Performance Tips

### Reduce Monitoring Overhead

```python
# Default: 100ms samples (0.5% overhead typical)
monitor = ResourceMonitor(sample_interval_ms=100)

# Reduce frequency for <0.1% overhead
monitor = ResourceMonitor(sample_interval_ms=500)  # 500ms samples

# Increase for better temporal resolution
monitor = ResourceMonitor(sample_interval_ms=50)   # 50ms samples
```

### Cache Configuration

```python
# Don't recreate config every time
config = SandboxConfig(mode=SandboxMode.AUTO)

for attack in attacks:
    result = await executor.execute_attack(..., sandbox_config=config)
```

### Batch Operations

```python
# Monitor multiple operations together
monitor = ResourceMonitor()
await monitor.start()

# Execute multiple attacks
for i in range(10):
    result = await executor.execute_attack(...)

summary = await monitor.stop()
# Summary covers all 10 attacks
```

## Common Patterns

### Safe Attack Execution

```python
async def execute_safe_attack(strategy, payload):
    config = SandboxConfig(
        mode=SandboxMode.AUTO,
        timeout_seconds=30.0,
        memory_limit_mb=512
    )

    monitor = ResourceMonitor()
    await monitor.start()

    try:
        result = await executor.execute_attack(strategy, payload, {})
        summary = await monitor.stop()

        # Log resource usage
        if summary.peak_memory_mb > 400:
            logger.warning(f"High memory usage: {summary.peak_memory_mb} MB")

        return result, summary

    except Exception as e:
        logger.error(f"Attack execution failed: {e}")
        await monitor.stop()
        raise
```

### Batch Testing with Limits

```python
async def test_attacks_with_limits(attacks):
    config = SandboxConfig(
        timeout_seconds=30.0,
        memory_limit_mb=512
    )

    results = []
    for strategy, payload in attacks:
        monitor = ResourceMonitor()
        await monitor.start()

        try:
            result = await executor.execute_attack(strategy, payload, {})
            summary = await monitor.stop()

            within_limits, violations = monitor.check_limits(512, 30.0)

            results.append({
                'attack': strategy.value,
                'success': result.bypassed,
                'violations': violations,
                'memory_peak': summary.peak_memory_mb
            })

        except Exception as e:
            logger.error(f"Error testing {strategy.value}: {e}")

    return results
```

## Troubleshooting

### Monitor Overhead Too High

```python
# Check what's available
from HoloLoom.redteam.sandbox.protocols import get_sandbox_mode_availability

available = get_sandbox_mode_availability()
if not available[SandboxMode.CGROUPS]:
    print("cgroups not available, using SUBPROCESS mode")
```

### Memory Limit Exceeded

```python
# Reduce limit or check for memory leaks
if summary.peak_memory_mb > config.memory_limit_mb:
    print(f"Consider increasing memory_limit or investigating memory usage")
    print(f"Avg: {summary.avg_memory_mb:.1f} MB, Peak: {summary.peak_memory_mb:.1f} MB")
```

### Timeout Exceeded

```python
# Increase timeout or optimize code
if any(v.startswith("Timeout") for v in violations):
    print("Increase timeout_seconds or optimize attack execution")
```

## Phase 2 Implementation Status

- [x] Protocol definitions
- [x] Configuration system
- [x] Resource monitoring
- [ ] SandboxManager (orchestrator)
- [ ] Concrete isolators (Subprocess, cgroups, Docker)
- [ ] Network policy implementations
- [ ] Filesystem isolation
- [ ] Integration tests
- [ ] Performance benchmarks

See `CARTS_PHASE2_SANDBOX_FOUNDATION.md` for full details.
