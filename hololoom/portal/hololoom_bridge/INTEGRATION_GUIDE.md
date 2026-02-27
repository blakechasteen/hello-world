# HoloLoom Bridge Integration Guide for Portal

**Quick Start**: 3 lines of code to connect Portal to HoloLoom intelligence.

## Installation

No additional dependencies required! Bridge uses `httpx` and `pydantic` which Portal already has.

```python
from hololoom.portal.hololoom_bridge import HoloLoomBridge
```

## Portal Server Integration

**Use Case**: Portal Server needs to provide job context or verify reasoning about task allocation.

```python
# portal_server/main.py
from hololoom.portal.hololoom_bridge import HoloLoomBridge, BridgeConfig

class PortalServer:
    def __init__(self):
        self.bridge_config = BridgeConfig(
            hololoom_url="http://localhost:8000",
            timeout_seconds=10
        )

    async def allocate_job(self, job_request):
        """Allocate job to best node with HoloLoom intelligence."""
        async with HoloLoomBridge(self.bridge_config) as bridge:
            # Get intelligence about job type
            context = await bridge.recall(
                f"jobs similar to {job_request.module_id}",
                k=5,
                mode="fast"
            )

            if context.success:
                # Store job request to HoloLoom for learning
                await bridge.experience(
                    f"Job {job_request.job_id} allocated: {job_request.module_id}",
                    metadata={
                        "job_id": job_request.job_id,
                        "module_id": job_request.module_id,
                        "source": "portal_server"
                    }
                )

            # Existing allocation logic...
            return self.select_best_node(job_request, context.data)
```

## Node Daemon Integration

**Use Case**: Node needs to understand job context before execution.

```python
# portal/node_daemon/wasm_runner.py
from hololoom.portal.hololoom_bridge import HoloLoomBridge

class WASMRunner:
    def __init__(self):
        self.bridge = HoloLoomBridge()

    async def run_job(self, job_request):
        """Execute WASM job with HoloLoom context."""

        # Get context from hololoom memory
        context = await self.bridge.recall(
            f"context for {job_request.module_id}",
            k=10
        )

        # Prepare input with context
        job_input = {
            "original": job_request.input_json,
            "context": [m['text'] for m in context.data] if context.success else []
        }

        # Execute WASM
        result = await self.execute_wasm(
            job_request.wasm_base64,
            job_request.entry_function,
            job_input,
            job_request.timeout_seconds
        )

        # Store results to HoloLoom
        await self.bridge.experience(
            f"Job {job_request.job_id} output: {result['summary']}",
            metadata={
                "job_id": job_request.job_id,
                "status": result['status'],
                "duration_ms": result['duration']
            }
        )

        return result

    async def execute_wasm(self, wasm_b64, func, input_data, timeout):
        # ... existing WASM execution logic ...
        pass
```

## Shuttle Bot Integration

**Use Case**: Shuttle Bot answers questions by querying HoloLoom memory.

```python
# portal/shuttle_bot/commands.py
from hololoom.portal.hololoom_bridge import HoloLoomBridge

class ShuttleBot:
    def __init__(self):
        self.bridge = HoloLoomBridge()

    async def cmd_status(self, room, args):
        """!status - Get Portal status with HoloLoom intelligence."""

        # Query HoloLoom for recent activity
        activity = await self.bridge.recall(
            "recent job activity",
            k=10,
            mode="fast"
        )

        # Get system status
        status = await self.bridge.status()

        message = f"""
Portal Status:
- HoloLoom: {'🟢 Online' if status['available'] else '🔴 Offline'}
- Recent Activity: {len(activity.data)} entries

Recent Jobs:
{self._format_jobs(activity.data)}
        """.strip()

        await self.send_message(room, message)

    async def cmd_query(self, room, args):
        """!query <text> - Ask HoloLoom a question."""

        query_text = ' '.join(args)

        # Simple recall
        result = await self.bridge.recall(query_text, k=5, mode="balanced")

        if not result.success:
            await self.send_message(room, f"Error: {result.error}")
            return

        message = f"""
Query: {query_text}
Confidence: {result.confidence:.0%}
Time: {result.latency_ms:.1f}ms

Results:
{self._format_results(result.data)}
        """.strip()

        await self.send_message(room, message)

    async def cmd_reason(self, room, args):
        """!reason <question> - Get HoloLoom reasoning."""

        question = ' '.join(args)

        # Full weaving cycle
        result = await self.bridge.weave(question, mode="verify")

        if not result.success:
            await self.send_message(room, f"Reasoning failed: {result.error}")
            return

        message = f"""
Question: {question}
Confidence: {result.confidence:.0%}
Time: {result.latency_ms:.1f}ms

Answer:
{result.data}
        """.strip()

        await self.send_message(room, message)

    def _format_jobs(self, jobs):
        if not jobs:
            return "(none)"
        return '\n'.join([f"- {j.get('text', j)}" for j in jobs[:5]])

    def _format_results(self, data):
        if not data:
            return "(no results)"
        return '\n'.join([f"- {d.get('text', d)}" for d in data[:3]])

    async def send_message(self, room, text):
        # ... existing message sending logic ...
        pass
```

## Common Patterns

### Pattern 1: Context + Action

```python
# Get context, then act
context = await bridge.recall(query, k=10)
if context.success:
    result = perform_action(context.data)
    await bridge.experience(f"Action result: {result}")
```

### Pattern 2: Reasoning Before Decision

```python
# Reason about decision
reasoning = await bridge.weave(
    f"Should we allocate {job.module_id} to {node.node_id}?",
    mode="verify"
)

if reasoning.success and reasoning.confidence > 0.8:
    allocate_job(job, node)
```

### Pattern 3: Learning from Outcomes

```python
# Try something
result = execute_job(job)

# Store to HoloLoom for future learning
await bridge.experience(
    f"Job {job.module_id} on {node.node_id}: {result.status}",
    metadata={"duration": result.duration, "success": result.status == "success"}
)
```

### Pattern 4: Error Recovery

```python
# Try primary path
result = primary_operation()

if not result.success:
    # Ask HoloLoom for alternatives
    alternatives = await bridge.recall(
        f"alternatives for {primary_operation.__name__}",
        k=3
    )

    if alternatives.success:
        # Try alternative based on HoloLoom suggestion
        result = try_alternative(alternatives.data[0])
```

## Configuration for Different Portal Scenarios

### Single-Machine Development
```python
config = BridgeConfig(
    hololoom_url="http://localhost:8000",
    timeout_seconds=30,
    fallback_on_error=True,
    verbose=True  # Debug output
)
```

### Multi-Machine Production
```python
config = BridgeConfig(
    hololoom_url="http://hololoom-server.local:8000",
    timeout_seconds=10,  # Stricter timeout
    retries=2,
    fallback_on_error=True,
    verbose=False  # No debug output
)
```

### Research/Exploration
```python
config = BridgeConfig(
    hololoom_url="http://localhost:8000",
    timeout_seconds=300,  # Long timeout for deep reasoning
    retries=3,
    fallback_on_error=False,  # Strict - fail on errors
    verbose=True
)
```

## Error Handling

Always check `result.success`:

```python
result = await bridge.recall("query")

if result.success:
    # Use result.data
    process(result.data)
else:
    # Handle error gracefully
    logger.warning(f"Recall failed: {result.error}")
    use_fallback()
```

## Performance Tips

1. **Reuse bridge instance** across multiple queries (use context manager)
2. **Use appropriate modes**:
   - `fast`: Simple factual queries
   - `balanced`: Standard queries (default)
   - `deep`: Complex reasoning
3. **Limit k**: Don't request more results than needed (default 5 is good)
4. **Batch operations**: Make multiple queries in parallel

```python
# Good: Reuse bridge, batch queries
async with HoloLoomBridge() as bridge:
    results = await asyncio.gather(
        bridge.recall("query1", k=5),
        bridge.recall("query2", k=5),
        bridge.recall("query3", k=5)
    )
```

## Monitoring & Debugging

Check `result.latency_ms` to detect slow queries:

```python
result = await bridge.recall(query)

if result.latency_ms > 200:
    logger.warning(f"Slow recall: {result.latency_ms}ms")

print(f"Confidence: {result.confidence:.0%} | Time: {result.latency_ms:.1f}ms")
```

Use `verbose=True` in config for detailed output:

```python
config = BridgeConfig(verbose=True)
# Will print all HTTP requests/responses
```

## Testing

Simple mock for testing without HoloLoom running:

```python
# tests/test_portal_with_bridge.py
import pytest
from unittest.mock import AsyncMock, patch
from hololoom.portal.hololoom_bridge import HoloLoomBridge, LoomResult

@pytest.mark.asyncio
async def test_job_allocation_with_hololoom():
    """Test job allocation uses HoloLoom for context."""

    # Mock HoloLoom response
    mock_context = LoomResult(
        success=True,
        data=[{"text": "similar job A", "score": 0.9}],
        confidence=0.9,
        latency_ms=50
    )

    with patch.object(HoloLoomBridge, 'recall', return_value=mock_context):
        server = PortalServer()
        result = await server.allocate_job(create_test_job())
        assert result is not None
```

## Next Steps

1. ✅ Add bridge imports to Portal components
2. ✅ Test with local HoloLoom instance
3. ✅ Deploy HoloLoom server to your network
4. ✅ Monitor latencies and adjust timeouts
5. ✅ Integrate learning signals into Portal decisions

---

**Questions?** Check BRIDGE_OVERVIEW.md for detailed API documentation.

**Issue?** Enable `verbose=True` to see HTTP requests and responses.
