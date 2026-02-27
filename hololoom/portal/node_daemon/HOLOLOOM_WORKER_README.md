# HoloLoom Worker - Distributed Intelligence Module

## Overview

The **HoloLoomWorker** enables Node Daemon to execute HoloLoom operations **directly** (not via HTTP), bringing distributed intelligence to the Portal's distributed computing network.

### Key Features

- **Local Execution**: No HTTP overhead - true in-process performance
- **Lazy Initialization**: HoloLoom initialized on first use, minimal startup cost
- **Graceful Degradation**: Returns error dicts if HoloLoom unavailable (never crashes)
- **Operation Timing**: Complete execution metrics for every operation
- **Async-Native**: Full async/await support for concurrent operations
- **Generic Dispatch**: Single `execute()` method for all operations

## Architecture

```
Node Daemon
    ↓
HoloLoomWorker (this module)
    ├─ recall() → Local memory search
    ├─ experience() → Local memory store
    ├─ weave() → Local reasoning cycle
    └─ execute() → Generic dispatch
        ↓
HoloLoom (direct import, not HTTP)
    ├─ Memory System (Yarn Graph + Vector Store)
    ├─ Awareness Graph (activation tracking)
    ├─ Semantic Calculus (228D projections)
    └─ Embedding Layer (Matryoshka representations)
```

## Installation

Files:
- `hololoom_worker.py` - Main worker implementation (253 lines)
- `hololoom_worker_demo.py` - Usage examples and demos (180 lines)

No additional dependencies required beyond HoloLoom itself.

## Usage

### Basic Operations

```python
from hololoom_worker import HoloLoomWorker

# Create worker in "fast" mode
worker = HoloLoomWorker(mode="fast")

try:
    # Store a memory
    result = await worker.experience("Thompson Sampling balances exploration")
    memory_id = result["memory_id"]

    # Search memories
    result = await worker.recall("Thompson Sampling", k=5)
    memories = result["memories"]

    # Reason about a query
    result = await worker.weave("Explain Thompson Sampling", mode="fast")
    response = result["response"]
    confidence = result["confidence"]

finally:
    await worker.close()
```

### Generic Dispatch

The `execute()` method provides a unified interface for all operations:

```python
worker = HoloLoomWorker(mode="fast")

# All three are equivalent:
result = await worker.execute("recall", {
    "query": "Thompson Sampling",
    "k": 5
})
```

### Operation Parameters

#### recall(query: str, k: int = 5) → Dict

Search local memory for relevant items.

**Parameters:**
- `query` (str): Search query
- `k` (int): Number of results to return (default: 5)

**Returns:**
```python
{
    "status": "success",
    "query": "Thompson Sampling",
    "memories": [
        {"id": "mem-123", "text": "...", "relevance": 0.92, "timestamp": "..."},
        ...
    ],
    "count": 3,
    "execution_time_ms": 45.2
}
```

#### experience(content: str) → Dict

Store new memory.

**Parameters:**
- `content` (str): Content to remember

**Returns:**
```python
{
    "status": "success",
    "memory_id": "mem-456",
    "content": "...",
    "timestamp": "2025-12-03T22:44:00Z",
    "execution_time_ms": 12.5
}
```

#### weave(query: str, mode: str = "fast") → Dict

Execute full reasoning cycle.

**Parameters:**
- `query` (str): Query to reason about
- `mode` (str): "bare", "fast" (default), or "fused"

**Returns:**
```python
{
    "status": "success",
    "query": "How does Thompson Sampling work?",
    "mode": "fast",
    "response": "Based on memory: ...",
    "memories_used": 3,
    "confidence": 0.75,
    "execution_time_ms": 150.3
}
```

#### execute(operation: str, params: Dict) → Dict

Generic operation dispatch.

**Parameters:**
- `operation` (str): Operation name ("recall", "experience", "weave")
- `params` (dict): Operation parameters

**Returns:** Same as individual operation methods

**Errors:**
```python
{
    "status": "error",
    "error": "Unknown operation: invalid",
    "execution_time_ms": 1.2
}
```

## Execution Modes

The worker supports three HoloLoom execution modes:

| Mode | Use Case | Speed | Quality |
|------|----------|-------|---------|
| **bare** | Ultra-fast, minimal processing | ~45ms | Good |
| **fast** | Balanced (recommended default) | ~50-150ms | Good |
| **fused** | Maximum quality, all features | ~200-300ms | Excellent |

Configure at initialization:
```python
worker = HoloLoomWorker(mode="fused")  # Maximum quality
worker = HoloLoomWorker(mode="fast")   # Balanced (default)
worker = HoloLoomWorker(mode="bare")   # Ultra-fast
```

## Implementation Details

### Lazy Initialization

HoloLoom is created only when the first operation is executed:

```python
worker = HoloLoomWorker()  # Fast - no HoloLoom init yet
result = await worker.recall("query")  # Slow - HoloLoom initialized here
result = await worker.recall("another")  # Fast - HoloLoom already initialized
```

Thread-safe: Uses asyncio.Lock to ensure only one coroutine initializes.

### Graceful Degradation

If HoloLoom is not available (import fails), operations return error responses:

```python
worker = HoloLoomWorker()
result = await worker.recall("query")
# Returns: {"status": "error", "error": "HoloLoom not available", "execution_time_ms": 0.5}
# Never raises an exception
```

### Operation Timing

Every result includes `execution_time_ms` for performance monitoring:

```python
result = await worker.recall("query", k=5)
print(f"Recall took {result['execution_time_ms']:.1f}ms")
```

## Integration with Node Daemon

### As a Job Worker

The HoloLoomWorker can be used in the Node Daemon's job queue system:

```python
# In node_daemon/main.py
from hololoom_worker import HoloLoomWorker

class NodeDaemon:
    def __init__(self):
        self.hololoom_worker = HoloLoomWorker(mode="fast")

    async def handle_job(self, job_request):
        # Job is a HoloLoom operation
        result = await self.hololoom_worker.execute(
            operation=job_request.operation,
            params=job_request.params
        )

        return JobResult(
            job_id=job_request.job_id,
            status=JobStatus.COMPLETED,
            output_json=result,
            execution_time_ms=result.get("execution_time_ms")
        )
```

### Registering HoloLoom Operations

Add to Node Daemon's capabilities:

```python
# In node_daemon/module_registry.py
from hololoom_worker import HoloLoomWorker

registry.register_module({
    "name": "hololoom",
    "operations": ["recall", "experience", "weave"],
    "worker": HoloLoomWorker(mode="fast")
})
```

## Performance Characteristics

Typical operation latencies (cold cache):

| Operation | Latency | Notes |
|-----------|---------|-------|
| **recall (k=5)** | 45-100ms | In-memory search, no HTTP |
| **experience** | 10-25ms | Memory store operation |
| **weave (fast)** | 150-250ms | Includes recall + reasoning |
| **First operation** | +30-50ms | One-time HoloLoom init |

With query cache (warm):
- Repeated recall: <5ms (100x+ speedup)
- Repeated weave: <10ms (20x+ speedup)

## Error Handling

All operations return structured error responses:

```python
result = await worker.recall("query")

if result["status"] == "error":
    error_msg = result["error"]
    time_ms = result["execution_time_ms"]
    # Handle error gracefully
else:
    data = result  # Use result data
```

## Examples

### Example 1: Simple Q&A

```python
async def simple_qa():
    worker = HoloLoomWorker(mode="fast")
    try:
        # Store knowledge
        await worker.experience("Python decorators modify functions")
        await worker.experience("Java uses annotations similarly")

        # Query knowledge
        result = await worker.recall("decorators", k=3)
        print(f"Found {result['count']} memories about decorators")

    finally:
        await worker.close()
```

### Example 2: Batch Operations

```python
async def batch_operations():
    worker = HoloLoomWorker(mode="fast")
    try:
        # Store batch
        for content in ["...", "...", "..."]:
            await worker.experience(content)

        # Query batch
        queries = ["topic1", "topic2", "topic3"]
        results = await asyncio.gather(*[
            worker.recall(q, k=3)
            for q in queries
        ])

        total_memories = sum(r.get("count", 0) for r in results)
        print(f"Total matches: {total_memories}")

    finally:
        await worker.close()
```

### Example 3: Intelligent Reasoning

```python
async def intelligent_qa():
    worker = HoloLoomWorker(mode="fused")  # Maximum quality
    try:
        # Multi-step reasoning
        result = await worker.weave(
            "What are the tradeoffs of Thompson Sampling vs UCB?",
            mode="fused"
        )

        print(f"Response: {result['response']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Used {result['memories_used']} memories")
        print(f"Execution: {result['execution_time_ms']:.1f}ms")

    finally:
        await worker.close()
```

## Running the Demo

```bash
cd hololoom/portal/node_daemon

# Run all demos
python hololoom_worker_demo.py

# Or import in Python
from hololoom_worker_demo import demo_basic_operations
import asyncio
asyncio.run(demo_basic_operations())
```

## Roadmap

**Phase 1** (Current):
- ✅ Local memory operations (recall, experience)
- ✅ Basic weaving (context-based reasoning)
- ✅ Lazy initialization
- ✅ Graceful degradation

**Phase 2** (Planned):
- Full weaving orchestrator integration
- Multi-step reasoning chains
- Policy-based tool selection
- Integration with alignment framework

**Phase 3** (Future):
- Streaming results (token-by-token)
- Distributed context expansion
- Cross-node memory federation
- Performance optimization

## Troubleshooting

### HoloLoom not available

**Symptom:** All operations return "HoloLoom not available" error

**Cause:** HoloLoom module not importable

**Fix:**
```bash
# Ensure HoloLoom is installed
pip install -e HoloLoom

# Check import
python -c "from hololoom import hololoom; print('OK')"
```

### Slow initial operation

**Symptom:** First operation takes 30-50ms longer than expected

**Cause:** HoloLoom lazy initialization

**Fix:** This is normal and expected. Subsequent operations will be faster.

### Memory growing over time

**Symptom:** Process memory increases with each operation

**Cause:** HoloLoom internal caching

**Fix:** Call `worker.close()` to cleanup, or restart worker periodically.

## Related Files

- `wasm_runner.py` - WASM module execution (similar pattern)
- `module_registry.py` - Node capability registration
- `main.py` - Node daemon entry point
- `../shared/types.py` - Shared data models (JobResult, JobStatus)
- `../shared/logging.py` - Logging utilities

## Questions?

See HoloLoom documentation:
- [CLAUDE.md](../../../CLAUDE.md) - Complete system guide
- [hololoom/hololoom.py](../../hololoom.py) - Core API
- [hololoom/config.py](../../config.py) - Configuration options
