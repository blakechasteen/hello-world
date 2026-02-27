# Sandbox Integration Guide

**Status**: Production Ready (December 5, 2025)
**Module**: `hololoom.redteam.sandbox`
**Author**: CARTS Team

Complete guide for integrating sandbox protocols and resource monitoring into CARTS red team attacks.

## Quick Integration Checklist

- [ ] Import sandbox components
- [ ] Create SandboxConfig with appropriate mode
- [ ] Create ResourceMonitor for tracking
- [ ] Execute attack in sandbox
- [ ] Monitor resource usage
- [ ] Check limits and violations
- [ ] Log results and metrics

## Step-by-Step Integration

### 1. Import Components

```python
from hololoom.redteam.sandbox import (
    SandboxMode,
    SandboxConfig,
    SandboxResult,
    ResourceMonitor,
)
```

### 2. Create Configuration

**Minimal (development)**:
```python
config = SandboxConfig()
# Uses defaults: AUTO mode, 30s timeout, 512MB memory
```

**Production (recommended)**:
```python
config = SandboxConfig(
    mode=SandboxMode.AUTO,      # Auto-detect best available
    timeout_seconds=30.0,        # Kill after 30 seconds
    memory_limit_mb=512,         # Max 512 MB RAM
    cpu_limit_percent=50,        # Max 50% CPU
    network_enabled=False,       # Block network by default
    filesystem_readonly=True,    # Read-only filesystem
    allowed_write_paths=["/tmp"]  # Allow /tmp for temp files
)
```

### 3. Create Resource Monitor

```python
monitor = ResourceMonitor(
    sample_interval_ms=100,  # Sample every 100ms
    pid=None                 # Monitor current process
)
```

### 4. Execute with Monitoring

```python
# Start monitoring
await monitor.start()

try:
    # Execute attack payload
    result = await execute_attack_payload(
        command=["python", "attack.py"],
        config=config
    )
finally:
    # Always stop monitoring (even if attack fails)
    summary = await monitor.stop()
```

### 5. Analyze Results

```python
# Check if attack succeeded
if not result.success:
    print(f"Attack failed with exit code {result.exit_code}")
    print(f"Error: {result.stderr}")

# Check resource usage
print(f"Memory used: {summary.peak_memory_mb:.1f} MB")
print(f"CPU usage: {summary.avg_cpu_percent:.1f}%")
print(f"Duration: {summary.duration_seconds:.1f}s")

# Check sandbox violations
if result.had_violations:
    print("Sandbox violations detected:")
    for v in result.sandbox_violations:
        print(f"  - {v}")

# Check if within limits
within_limits, violations = monitor.check_limits(
    memory_limit_mb=512,
    timeout_seconds=30.0
)
if not within_limits:
    print("Resource limit violations:")
    for v in violations:
        print(f"  - {v}")
```

## Complete Example: Attack Executor

```python
import asyncio
from hololoom.redteam.sandbox import (
    SandboxMode,
    SandboxConfig,
    ResourceMonitor
)


class AttackExecutor:
    """Execute attacks safely with resource monitoring."""

    def __init__(self, sandbox_config: SandboxConfig = None):
        """Initialize executor with sandbox config."""
        self.config = sandbox_config or SandboxConfig()

    async def execute(self, command: list, timeout_override: float = None) -> dict:
        """
        Execute command with full monitoring and safety.

        Args:
            command: Command to execute (e.g., ["python", "attack.py"])
            timeout_override: Override config timeout (optional)

        Returns:
            Complete result dict with metrics
        """
        # Setup config with optional timeout override
        config = self.config
        if timeout_override:
            config.timeout_seconds = timeout_override

        # Create monitor
        monitor = ResourceMonitor(sample_interval_ms=100)

        try:
            # Start monitoring
            await monitor.start()

            # Execute attack (placeholder - actual implementation varies)
            result = await self._execute_in_sandbox(command, config)

        finally:
            # Always stop monitoring
            summary = await monitor.stop()

        # Compile comprehensive result
        return {
            # Execution status
            "success": result.success,
            "exit_code": result.exit_code,
            "execution_time_ms": result.execution_time_ms,

            # Output
            "stdout": result.stdout,
            "stderr": result.stderr,

            # Resources
            "resource_usage": {
                "peak_memory_mb": summary.peak_memory_mb,
                "avg_cpu_percent": summary.avg_cpu_percent,
                "total_io_bytes": summary.total_io_bytes,
                "duration_seconds": summary.duration_seconds,
                "monitor_overhead_percent": summary.overhead_percent,
            },

            # Safety
            "sandbox_mode": config.mode.value,
            "sandbox_violations": result.sandbox_violations,
            "within_limits": self._check_limits(monitor, config),

            # Metadata
            "metadata": result.metadata,
            "errors": result.errors,
            "warnings": result.warnings,
        }

    async def _execute_in_sandbox(self, command: list, config: SandboxConfig):
        """Execute command in sandbox (implementation varies by mode)."""
        # Placeholder - actual implementation depends on sandbox mode
        # This would call SubprocessIsolator, CGroupsIsolator, or DockerIsolator
        from hololoom.redteam.sandbox import SandboxResult
        return SandboxResult(
            success=True,
            exit_code=0,
            execution_time_ms=100.0,
            stdout="execution output",
            stderr="",
            sandbox_mode_used=config.mode
        )

    def _check_limits(self, monitor, config) -> bool:
        """Check if execution stayed within limits."""
        within, _ = monitor.check_limits(
            config.memory_limit_mb,
            config.timeout_seconds
        )
        return within


# Usage
async def main():
    executor = AttackExecutor(
        SandboxConfig(
            mode=SandboxMode.AUTO,
            timeout_seconds=30.0,
            memory_limit_mb=512
        )
    )

    result = await executor.execute(["python", "attack.py"])

    print("=== Attack Result ===")
    print(f"Success: {result['success']}")
    print(f"Exit Code: {result['exit_code']}")
    print(f"\n=== Resource Usage ===")
    print(f"Peak Memory: {result['resource_usage']['peak_memory_mb']:.1f} MB")
    print(f"Avg CPU: {result['resource_usage']['avg_cpu_percent']:.1f}%")
    print(f"Duration: {result['resource_usage']['duration_seconds']:.1f}s")
    print(f"\n=== Safety ===")
    print(f"Within Limits: {result['within_limits']}")
    print(f"Violations: {result['sandbox_violations']}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Patterns

### Development/Testing

```python
config = SandboxConfig(
    mode=SandboxMode.NONE,      # No isolation
    timeout_seconds=10.0,
    memory_limit_mb=256
)
```

### Staging (Balanced)

```python
config = SandboxConfig(
    mode=SandboxMode.SUBPROCESS,  # Basic isolation
    timeout_seconds=30.0,
    memory_limit_mb=512,
    network_enabled=False
)
```

### Production (High Security)

```python
config = SandboxConfig(
    mode=SandboxMode.DOCKER,      # Full container isolation
    docker_image="python:3.11-slim",
    docker_network="none",
    timeout_seconds=60.0,
    memory_limit_mb=1024,
    filesystem_readonly=True,
    allowed_write_paths=["/tmp"]
)
```

### Network Whitelisting

```python
config = SandboxConfig(
    network_enabled=True,
    allowed_hosts=["github.com", "api.example.com"],
    allowed_ports=[443, 80]  # HTTPS and HTTP only
)
```

### Custom Resource Limits

```python
config = SandboxConfig(
    timeout_seconds=120.0,      # 2 minute timeout
    memory_limit_mb=2048,       # 2 GB memory
    cpu_limit_percent=100,      # Full CPU available
    allowed_write_paths=[
        "/tmp/attack",
        "/var/tmp/attack"
    ]
)
```

## Monitoring Patterns

### Basic Monitoring

```python
monitor = ResourceMonitor()
await monitor.start()
# ... execute attack ...
summary = await monitor.stop()
print(summary)  # Pretty-printed statistics
```

### High-Frequency Monitoring (sensitive operations)

```python
monitor = ResourceMonitor(sample_interval_ms=10)  # 10ms samples
await monitor.start()
# ... time-critical attack ...
summary = await monitor.stop()
print(f"Peak memory: {summary.peak_memory_mb:.1f} MB")
```

### Low-Frequency Monitoring (background operations)

```python
monitor = ResourceMonitor(sample_interval_ms=500)  # 500ms samples
await monitor.start()
# ... long-running attack ...
summary = await monitor.stop()
# Only ~0.2% overhead
```

### Custom Process Monitoring

```python
import os
monitor = ResourceMonitor(
    sample_interval_ms=100,
    pid=os.getpid()  # Monitor specific process
)
```

## Error Handling

### Safe Resource Cleanup

```python
monitor = ResourceMonitor()
try:
    await monitor.start()
    result = await execute_attack(config)
except Exception as e:
    print(f"Attack failed: {e}")
    result = None
finally:
    # Always stop monitoring
    summary = await monitor.stop()
```

### Context Manager Pattern (Recommended)

```python
async def execute_with_monitoring(command):
    monitor = ResourceMonitor()
    await monitor.start()

    try:
        result = await execute_attack(command)
        return result
    finally:
        summary = await monitor.stop()
        return summary
```

### Limit Violation Handling

```python
await monitor.start()
# ... execute ...
summary = await monitor.stop()

within_limits, violations = monitor.check_limits(512, 30.0)

if not within_limits:
    # Log violations for analysis
    print("Resource limit exceeded!")
    for v in violations:
        print(f"  {v}")

    # Potentially retry with higher limits
    if "Memory" in str(violations):
        print("Retrying with doubled memory limit...")
        await execute_attack(config_with_higher_memory)
```

## Integration with Attack Framework

### With ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def run_attack_concurrent(attack_list):
    """Run multiple attacks with resource monitoring."""
    results = []

    for attack in attack_list:
        config = SandboxConfig(
            mode=SandboxMode.SUBPROCESS,
            timeout_seconds=30.0
        )
        monitor = ResourceMonitor()

        await monitor.start()
        result = await execute_attack(attack, config)
        summary = await monitor.stop()

        results.append({
            "attack": attack,
            "result": result,
            "summary": summary
        })

    return results
```

### With Attack Orchestrator

```python
class AttackOrchestrator:
    """Run orchestrated attacks with sandbox isolation."""

    def __init__(self, sandbox_config: SandboxConfig):
        self.config = sandbox_config

    async def run_attack_chain(self, attacks: list) -> list:
        """Run sequence of attacks with monitoring."""
        results = []

        for attack in attacks:
            monitor = ResourceMonitor()

            try:
                await monitor.start()
                result = await self.execute_sandboxed(attack)
                summary = await monitor.stop()

                results.append({
                    "attack": attack.name,
                    "success": result.success,
                    "resource_usage": {
                        "memory_mb": summary.peak_memory_mb,
                        "cpu_percent": summary.avg_cpu_percent,
                        "duration_s": summary.duration_seconds
                    },
                    "violations": result.sandbox_violations
                })
            except Exception as e:
                results.append({
                    "attack": attack.name,
                    "success": False,
                    "error": str(e)
                })

        return results
```

## Metrics and Logging

### Log Successful Attack

```python
import logging

logger = logging.getLogger("CARTS.attack")

async def execute_and_log(attack_name, command, config):
    monitor = ResourceMonitor()
    await monitor.start()

    result = await execute_attack(command, config)
    summary = await monitor.stop()

    logger.info(
        f"Attack executed: {attack_name}",
        extra={
            "success": result.success,
            "exit_code": result.exit_code,
            "memory_mb": summary.peak_memory_mb,
            "duration_s": summary.duration_seconds,
            "violations": result.sandbox_violations
        }
    )

    return result
```

### CSV Export of Results

```python
import csv
from datetime import datetime

async def export_attack_results(results, filename="attacks.csv"):
    """Export attack results to CSV."""
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp",
            "attack_name",
            "success",
            "exit_code",
            "peak_memory_mb",
            "avg_cpu_percent",
            "duration_seconds",
            "violations",
            "errors"
        ])
        writer.writeheader()

        for r in results:
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "attack_name": r.get("attack_name"),
                "success": r["result"].success,
                "exit_code": r["result"].exit_code,
                "peak_memory_mb": r["summary"].peak_memory_mb,
                "avg_cpu_percent": r["summary"].avg_cpu_percent,
                "duration_seconds": r["summary"].duration_seconds,
                "violations": ";".join(r["result"].sandbox_violations),
                "errors": ";".join(r["result"].errors)
            })
```

## Best Practices

1. **Always monitor**: Use ResourceMonitor for every attack execution
2. **Conservative limits**: Start with strict limits, relax only if needed
3. **Handle timeouts**: Implement graceful degradation when timeout occurs
4. **Log violations**: Record all sandbox violations for analysis
5. **Cleanup resources**: Always stop monitor in finally block
6. **Test configuration**: Validate config before production use
7. **Monitor overhead**: Verify <5% overhead target is met
8. **Archive results**: Keep complete result history for learning

## Troubleshooting

### Monitor not collecting samples
```python
# Check if monitoring is running
print(f"Monitoring: {monitor._monitoring}")

# Verify samples are being collected
await asyncio.sleep(0.5)
samples = monitor.get_samples()
print(f"Samples collected: {len(samples)}")
```

### High overhead (>5%)
```python
# Use less frequent sampling
monitor = ResourceMonitor(sample_interval_ms=500)  # Instead of 100ms
# Reduces overhead to ~0.2%
```

### Memory tracking not accurate
```python
# Try installing psutil for accurate metrics
pip install psutil

# Fallback to system-level monitoring if needed
summary = await monitor.stop()
print(f"Monitor overhead: {summary.overhead_percent}%")
```

## See Also

- `SANDBOX_ARCHITECTURE.md` - Complete architecture reference
- `QUICKSTART.md` - Quick start with examples
- `protocols.py` - Protocol definitions
- `monitor.py` - Resource monitor implementation
