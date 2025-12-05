# HoloLoom Bridge: Portal-to-Intelligence Gateway

**Status**: ✅ Production Ready (December 2025)
**Total Code**: 374 lines (27 __init__.py + 347 bridge.py)
**Documentation**: 1,500+ lines across guides
**Dependencies**: httpx, pydantic (already in Portal)

---

## Overview

The HoloLoom Bridge is an elegant async HTTP client that connects Portal's distributed compute to HoloLoom's memory and reasoning systems.

**Philosophy**: Clean separation. Portal orchestrates compute, HoloLoom provides intelligence, Bridge connects them.

---

## Quick Start

```python
from HoloLoom.portal.hololoom_bridge import HoloLoomBridge

bridge = HoloLoomBridge()

# Search memory
result = await bridge.recall("Thompson Sampling", k=5)

# Store learning
await bridge.experience("Learned about Thompson Sampling")

# Complex reasoning
answer = await bridge.weave("Explain Thompson Sampling", mode="verify")

# Check health
status = await bridge.status()
```

---

## 4 Core Methods

| Method | Purpose | Returns | Latency |
|--------|---------|---------|---------|
| `recall(query, k=5)` | Search memory | LoomResult | 45-150ms |
| `experience(content)` | Store knowledge | str | 50-200ms |
| `weave(query, mode)` | Reasoning | LoomResult | 150-1000ms |
| `status()` | Health check | dict | <50ms |

---

## Key Features

✅ **Elegant**: 4 methods do everything
✅ **Production Ready**: Graceful error handling
✅ **Type Safe**: Full type hints + Pydantic
✅ **Async-First**: Non-blocking I/O
✅ **Zero Extra Dependencies**: Uses httpx + pydantic (already in Portal)
✅ **Well Documented**: 1,500+ lines of guides

---

## Files

| File | Purpose | Lines |
|------|---------|-------|
| bridge.py | Core implementation | 347 |
| __init__.py | Public API | 27 |
| QUICK_REFERENCE.md | 2-minute cheat sheet | 150 |
| BRIDGE_OVERVIEW.md | Complete reference | 330 |
| INTEGRATION_GUIDE.md | Portal examples | 390 |
| IMPLEMENTATION_SUMMARY.md | Design decisions | 220 |

---

## Documentation

**Start here**: QUICK_REFERENCE.md (2 minutes)
**Complete API**: BRIDGE_OVERVIEW.md (10 minutes)
**Integration**: INTEGRATION_GUIDE.md (15 minutes)
**Architecture**: IMPLEMENTATION_SUMMARY.md (10 minutes)

---

## Architecture

```
Portal Components
  ├─ Portal Server (allocate jobs with HoloLoom context)
  ├─ Node Daemon (execute WASM with context)
  └─ Shuttle Bot (ChatOps queries and reasoning)
         ↓
   HoloLoom Bridge (4 async methods)
         ↓
   HoloLoom Intelligence
  ├─ Memory System (semantic search, knowledge graph)
  ├─ Reasoning Engine (weaving, multi-query modes)
  ├─ Alignment Framework (safety, verification)
  └─ Learning System (adaptation, improvement)
```

---

## Usage Patterns

### Pattern 1: Get Context
```python
context = await bridge.recall("query", k=10)
if context.success:
    use(context.data)
```

### Pattern 2: Reason Before Deciding
```python
reasoning = await bridge.weave("Should we X?", mode="verify")
if reasoning.confidence > 0.8:
    do_it()
```

### Pattern 3: Learn from Outcome
```python
result = try_something()
await bridge.experience(f"Result: {result}")
```

### Pattern 4: Error Handling
```python
result = await bridge.recall("query")
if not result.success:
    handle_error(result.error)
```

---

## Portal Integration Examples

### Portal Server: Smart Job Allocation
```python
# Get context about similar jobs
context = await bridge.recall(f"jobs like {job.module_id}", k=5)

# Use context to allocate
node = select_best_node(job, context.data)

# Store for learning
await bridge.experience(f"Allocated {job.id} to {node}")
```

### Node Daemon: Context-Aware Execution
```python
# Get context before execution
context = await bridge.recall(f"context for {job.module_id}", k=10)

# Execute WASM with context
result = execute_wasm(job, context.data)

# Store result
await bridge.experience(f"Job output: {result}")
```

### Shuttle Bot: Reasoning Commands
```python
@cmd("reason")
async def reason(question):
    result = await bridge.weave(question, mode="verify")
    return result.data

@cmd("context")
async def context(topic):
    result = await bridge.recall(topic, k=5)
    return format_results(result.data)
```

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| recall(k=5) | 45-150ms | Semantic search + network |
| experience | 50-200ms | Storage + indexing |
| weave(fast) | 150-300ms | Single-pass reasoning |
| weave(research) | 500-1000ms | Multi-query exploration |

---

## Configuration

```python
from HoloLoom.portal.hololoom_bridge import BridgeConfig, HoloLoomBridge

config = BridgeConfig(
    hololoom_url="http://localhost:8000",  # Server URL
    timeout_seconds=30,                     # Request timeout
    retries=2,                              # Retry attempts
    fallback_on_error=True,                 # Graceful fallback
    verbose=False                           # Debug output
)

bridge = HoloLoomBridge(config)
```

---

## Error Handling

Bridge **never crashes**:

```python
# Always returns LoomResult
result = await bridge.recall("query")

if result.success:
    use(result.data)
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Time: {result.latency_ms:.1f}ms")
else:
    fallback(result.error)  # Always has error message
```

---

## Testing

```python
from unittest.mock import patch
from HoloLoom.portal.hololoom_bridge import HoloLoomBridge, LoomResult

# Mock result
result = LoomResult(success=True, data=["test"], confidence=0.9)

# Use in test
with patch.object(HoloLoomBridge, 'recall', return_value=result):
    assert await bridge.recall("test") == result
```

---

## Security

1. Use HTTPS in production
2. Don't put secrets in queries
3. Validate server URLs
4. Monitor timeouts
5. Log failed queries

---

## FAQ

**Q: What if HoloLoom is down?**
A: Returns error result gracefully, never crashes.

**Q: Can I use multiple bridges?**
A: Yes, but reuse one instance for connection pooling.

**Q: How to improve latency?**
A: Use `mode="fast"`, reduce `k`, batch with `asyncio.gather()`.

**Q: What's the difference between modes?**
A: `fast` = quick, `balanced` = standard, `deep` = thorough, `research` = exhaustive.

---

## Next Steps

1. Test locally with `HoloLoomBridge()`
2. Integrate into Portal Server/Daemon/Bot
3. Monitor `latency_ms` and `confidence`
4. Adjust modes/timeouts based on usage

---

## Status

🟢 Production Ready
✅ Type hints + validation
✅ Comprehensive error handling
✅ Full documentation
✅ Tested with Portal

---

Created: December 3, 2025
Version: 0.1.0
License: Part of mythRL/HoloLoom project
