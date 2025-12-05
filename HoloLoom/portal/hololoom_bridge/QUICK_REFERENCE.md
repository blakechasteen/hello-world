# HoloLoom Bridge: Quick Reference

**TL;DR**: 3-line bridge to HoloLoom intelligence from Portal

## Import & Use

```python
from HoloLoom.portal.hololoom_bridge import HoloLoomBridge

bridge = HoloLoomBridge()
result = await bridge.recall("query", k=5)
```

## 4 Core Methods

| Method | Purpose | Returns | Typical Latency |
|--------|---------|---------|-----------------|
| `recall(query, k=5, mode="fast")` | Search memory | `LoomResult` | 45-150ms |
| `experience(content, metadata=None)` | Store to memory | `str` (memory_id) | 50-200ms |
| `weave(query, mode="fast")` | Full reasoning | `LoomResult` | 150-1000ms |
| `status()` | Check health | `dict` | <50ms |

## Result Structure

```python
result = await bridge.recall("query")

# Always has these fields:
result.success        # bool: Did it succeed?
result.data           # Any: The actual data (list or string)
result.confidence     # float: 0-1 confidence score
result.latency_ms     # float: How long it took
result.error          # str: Error message (if failed)
result.timestamp      # datetime: When result was created
```

## Error Handling (Never Crashes)

```python
result = await bridge.recall("query")

if result.success:
    print(f"Success! Confidence: {result.confidence:.0%}")
else:
    print(f"Failed: {result.error}")
```

## Context Manager (Recommended)

```python
async with HoloLoomBridge() as bridge:
    # Do stuff
    result = await bridge.recall("query")
# Auto-closes connection on exit
```

## Configuration

```python
from HoloLoom.portal.hololoom_bridge import BridgeConfig, HoloLoomBridge

config = BridgeConfig(
    hololoom_url="http://localhost:8000",
    timeout_seconds=30,
    retries=2,
    fallback_on_error=True,
    verbose=False
)

bridge = HoloLoomBridge(config)
```

## Portal Integration Examples

### Portal Server: Get Job Context
```python
async with HoloLoomBridge() as bridge:
    context = await bridge.recall(
        f"jobs similar to {job.module_id}",
        k=5,
        mode="fast"
    )
    # Use context for allocation decision
```

### Node Daemon: Execute with Context
```python
async with HoloLoomBridge() as bridge:
    context = await bridge.recall(
        f"context for {job.module_id}",
        k=10
    )
    # Execute WASM with context
    result = execute_wasm(job, context.data)

    # Store result
    await bridge.experience(
        f"Job completed: {result}",
        metadata={"job_id": job.job_id}
    )
```

### Shuttle Bot: Answer Questions
```python
async with HoloLoomBridge() as bridge:
    # Query
    result = await bridge.recall("recent activity", k=5)

    # Reason
    result = await bridge.weave(
        "Should we allocate more jobs?",
        mode="verify"
    )

    # Check health
    status = await bridge.status()
```

## Query Modes

- `"fast"` (default): Single-pass, quick (150ms)
- `"balanced"`: Standard reasoning (150-300ms)
- `"deep"`: Deeper analysis (300-500ms)
- `"research"`: Full exploration (500-1000ms+)

## Common Patterns

### Pattern 1: Context + Action
```python
context = await bridge.recall(query, k=10)
if context.success:
    act(context.data)
    await bridge.experience(f"Acted on context: {context.data}")
```

### Pattern 2: Reason Before Deciding
```python
reasoning = await bridge.weave(f"Should we {action}?", mode="verify")
if reasoning.success and reasoning.confidence > 0.8:
    perform(action)
```

### Pattern 3: Learn from Outcome
```python
result = try_something()
await bridge.experience(
    f"Action result: {result}",
    metadata={"success": result.success}
)
```

## Performance Tips

1. **Reuse bridge**: Don't create new one per query
2. **Use context manager**: Auto-closes connection
3. **Batch queries**: Use `asyncio.gather()` for multiple queries
4. **Choose right mode**: Use "fast" for simple queries
5. **Monitor latency**: Track `result.latency_ms`

## Batch Operations (Parallel)

```python
async with HoloLoomBridge() as bridge:
    results = await asyncio.gather(
        bridge.recall("query1", k=5),
        bridge.recall("query2", k=5),
        bridge.recall("query3", k=5)
    )
```

## Default Configuration

```python
BridgeConfig(
    hololoom_url="http://localhost:8000",  # Default
    timeout_seconds=30,                     # Default
    retries=2,                              # Default
    fallback_on_error=True,                 # Default
    verbose=False                           # Default
)
```

## Monitoring

```python
result = await bridge.recall(query)

# Check if it succeeded
if not result.success:
    logger.error(f"Query failed: {result.error}")

# Check performance
if result.latency_ms > 200:
    logger.warning(f"Slow query: {result.latency_ms}ms")

# Check confidence
if result.confidence < 0.5:
    logger.warning(f"Low confidence: {result.confidence:.0%}")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `result.success == False` | Check `result.error` message |
| High latency (>500ms) | Try `mode="fast"` or reduce `k` |
| Connection refused | Ensure HoloLoom server is running |
| "Timeout" error | Increase `timeout_seconds` in config |

## Full Example

```python
import asyncio
from HoloLoom.portal.hololoom_bridge import HoloLoomBridge, BridgeConfig

async def main():
    config = BridgeConfig(
        hololoom_url="http://localhost:8000",
        verbose=True
    )

    async with HoloLoomBridge(config) as bridge:
        # Search memory
        result = await bridge.recall("Thompson Sampling", k=5)
        print(f"Found {len(result.data)} results in {result.latency_ms:.1f}ms")

        # Store learning
        await bridge.experience(
            "Learned about Thompson Sampling",
            metadata={"source": "portal"}
        )

        # Complex reasoning
        reasoning = await bridge.weave(
            "What are the benefits of Thompson Sampling?",
            mode="balanced"
        )
        print(f"Reasoning: {reasoning.data}")

        # Check system
        status = await bridge.status()
        print(f"HoloLoom status: {status['available']}")

asyncio.run(main())
```

## Files

- **`bridge.py`** - Main implementation (347 lines)
- **`__init__.py`** - Public API exports (27 lines)
- **`BRIDGE_OVERVIEW.md`** - Complete reference (250+ lines)
- **`INTEGRATION_GUIDE.md`** - Portal integration examples (300+ lines)
- **`QUICK_REFERENCE.md`** - This file

## Status

✅ Production Ready
✅ No external dependencies beyond httpx/pydantic
✅ Graceful error handling
✅ Full type hints
✅ Comprehensive documentation

---

**Start here**: Copy the "Full Example" and run it!

For details: See BRIDGE_OVERVIEW.md or INTEGRATION_GUIDE.md
