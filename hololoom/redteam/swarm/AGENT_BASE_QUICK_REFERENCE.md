# BaseAgent Quick Reference Guide

**Last Updated**: November 2025
**Status**: Production Ready
**File**: `hololoom/redteam/swarm/agent_base.py`

## Quick Start (5 minutes)

### Import
```python
from hololoom.redteam.swarm.agent_base import BaseAgent
from hololoom.redteam.swarm.communication import MessageBus
from hololoom.redteam.swarm.protocols import AgentRole, AgentTask, AgentResult
```

### Create Agent
```python
bus = MessageBus()
agent = BaseAgent(
    agent_id="scout_1",
    role=AgentRole.SCOUT,
    message_bus=bus,
)
```

### Use Agent
```python
async with agent:
    # Execute task
    task = AgentTask(
        task_type="probe",
        target="example.com",
        parameters={},
    )
    result = await agent.execute_task(task)
    print(f"Success: {result.success}")

    # Get metrics
    metrics = agent.get_metrics()
    print(f"Tasks: {metrics['tasks_completed']}")
```

## Agent Properties

```python
agent.agent_id       # "scout_1"
agent.role           # AgentRole.SCOUT
agent.state          # AgentState.ACTIVE
```

## Agent Lifecycle

```python
# Start
await agent.start()
assert agent.state == AgentState.ACTIVE

# Do work
result = await agent.execute_task(task)

# Stop
await agent.stop()
assert agent.state == AgentState.SHUTDOWN
```

## Communication

### Send Message
```python
await agent.send_message(
    recipient="coordinator",
    message_type="status",
    payload={"status": "ready"},
    priority=MessagePriority.NORMAL,
    requires_ack=False,
)
```

### Broadcast Message
```python
count = await agent.broadcast(
    message_type="alert",
    payload={"alert": "intrusion"},
    priority=MessagePriority.CRITICAL,
)
```

## Task Execution

### Override in Subclass
```python
class MyAgent(BaseAgent):
    async def execute_task(self, task: AgentTask) -> AgentResult:
        start_time = time.time()
        try:
            # Do work
            result = await self._do_work(task)

            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=True,
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
```

## Message Types

| Type | Purpose | Handler |
|------|---------|---------|
| `task` | Assign task | `_handle_task_message()` |
| `result` | Return result | `_handle_result_message()` |
| `status` | Status update | `_handle_status_message()` |
| `discovery` | Vulnerability | `_handle_discovery_message()` |
| `command` | Coordinator command | `_handle_command_message()` |

## Message Handlers (Override in Subclass)

```python
class MyAgent(BaseAgent):
    async def _handle_status_message(self, message: AgentMessage):
        """Custom status handling."""
        status = message.payload.get("status")
        # Handle status

    async def _handle_discovery_message(self, message: AgentMessage):
        """Custom discovery handling."""
        discovery = message.payload.get("discovery")
        # Handle discovery
```

## Metrics

### Get All Metrics
```python
metrics = agent.get_metrics()
```

### Metrics Contents
```python
{
    "agent_id": "scout_1",
    "role": "scout",
    "state": "active",
    "tasks_completed": 42,
    "tasks_failed": 3,
    "tasks_total": 45,
    "success_rate": 0.933,
    "messages_sent": 150,
    "messages_received": 145,
    "messages_acked": 50,
    "message_throughput": 29.5,  # msgs/sec
    "avg_task_duration_ms": 245.3,
    "max_task_duration_ms": 1234.5,
    "min_task_duration_ms": 10.0,
    "uptime_seconds": 300.5,
    "error_count": 5,
    "recovery_count": 5,
}
```

## State Machine

```
IDLE → start() → ACTIVE
               ↓
       [message_loop running]
               ↓
       execute_task() → EXECUTING
               ↓
       COMPLETED/FAILED
               ↓
       back to ACTIVE
               ↓
       stop() → SHUTDOWN
```

## Commands (From Coordinator)

```python
# Pause agent
pause_cmd = AgentMessage(
    sender="coordinator",
    recipient="scout_1",
    message_type="command",
    payload={"command": "pause"},
)
# Agent transitions to WAITING

# Resume agent
resume_cmd = AgentMessage(
    sender="coordinator",
    recipient="scout_1",
    message_type="command",
    payload={"command": "resume"},
)
# Agent transitions back to ACTIVE

# Shutdown agent
shutdown_cmd = AgentMessage(
    sender="coordinator",
    recipient="scout_1",
    message_type="command",
    payload={"command": "shutdown"},
)
# Agent calls stop()
```

## Error Handling

### Task Execution Errors
Always return `AgentResult`, never raise:

```python
async def execute_task(self, task):
    try:
        # Do work
    except Exception as e:
        return AgentResult(
            task_id=task.task_id,
            agent_id=self._agent_id,
            success=False,
            error=str(e),
            execution_time_ms=...,
        )
```

### Message Handler Errors
Automatically handled (continues despite errors):

```python
# BaseAgent._message_loop handles all exceptions
# and continues running
```

## Performance Tuning

### Adjust Message Handler Timeout
```python
agent = BaseAgent(
    agent_id="scout_1",
    role=AgentRole.SCOUT,
    message_bus=bus,
    message_handler_timeout=2.0,  # Longer timeout
)
```

### Adjust Shutdown Timeout
```python
agent = BaseAgent(
    agent_id="scout_1",
    role=AgentRole.SCOUT,
    message_bus=bus,
    shutdown_timeout=10.0,  # Longer shutdown timeout
)
```

## Best Practices

### 1. Always Use Context Manager
```python
# Good ✅
async with agent:
    await agent.send_message(...)

# Bad ❌
agent = BaseAgent(...)
await agent.start()
# ... might crash here ...
await agent.stop()  # Might not execute
```

### 2. Always Measure Task Duration
```python
# Good ✅
start_time = time.time()
try:
    # Do work
    elapsed_ms = (time.time() - start_time) * 1000
    return AgentResult(..., execution_time_ms=elapsed_ms)
except:
    elapsed_ms = (time.time() - start_time) * 1000
    return AgentResult(..., execution_time_ms=elapsed_ms)

# Bad ❌
return AgentResult(...)  # execution_time_ms = 0.0
```

### 3. Always Return AgentResult
```python
# Good ✅
return AgentResult(
    task_id=task.task_id,
    agent_id=self._agent_id,
    success=True,
    result=data,
    execution_time_ms=elapsed_ms,
)

# Bad ❌
return data  # Wrong type
raise Exception("Failed")  # Should return result
```

### 4. Monitor Metrics Regularly
```python
async def monitor_loop():
    while True:
        metrics = agent.get_metrics()
        if metrics['error_count'] > 10:
            log.warning("High error rate")
        await asyncio.sleep(60)
```

### 5. Handle Acknowledgments
```python
# Good ✅
if message.requires_ack:
    # BaseAgent.handle_message() sends ack automatically

# Custom ack (if needed)
await self._message_bus.acknowledge(message.id)
```

## Common Patterns

### Pattern 1: Simple Task Execution
```python
task = AgentTask(
    task_type="probe_surface",
    target="example.com",
    parameters={"timeout": 30},
)

result = await agent.execute_task(task)
if result.success:
    print(f"Discovered: {result.result}")
else:
    print(f"Failed: {result.error}")
```

### Pattern 2: Send Result to Coordinator
```python
result = await agent.execute_task(task)
await agent.send_message(
    recipient="coordinator",
    message_type="result",
    payload={"result": result.to_dict()},
    requires_ack=True,
)
```

### Pattern 3: Broadcast Discovery
```python
discoveries = [
    {"type": "service", "port": 80},
    {"type": "service", "port": 443},
]

await agent.broadcast(
    message_type="discovery",
    payload={"discoveries": discoveries},
    priority=MessagePriority.HIGH,
)
```

### Pattern 4: Monitor Agent Health
```python
while running:
    metrics = agent.get_metrics()
    print(f"Status: {metrics['state']}")
    print(f"Tasks: {metrics['tasks_completed']}/{metrics['tasks_total']}")
    print(f"Errors: {metrics['error_count']}")
    await asyncio.sleep(60)
```

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now BaseAgent will log all operations at DEBUG level
```

### Check Agent State
```python
print(f"Agent state: {agent.state}")
print(f"Pending tasks: {len(agent._pending_tasks)}")
print(f"Message handler: {agent._message_handler_task}")
```

### Inspect Metrics
```python
metrics = agent.get_metrics()
print(json.dumps(metrics, indent=2))
```

## Common Issues

### Issue 1: Agent Not Receiving Messages
**Solution**: Start the agent first
```python
await agent.start()  # Required!
message = await bus.receive("agent_id")
```

### Issue 2: Messages Dropped (Queue Full)
**Solution**: Increase queue size
```python
bus = MessageBus(max_queue_size=20000)  # Default: 10000
```

### Issue 3: Timeout on Shutdown
**Solution**: Increase shutdown timeout
```python
agent = BaseAgent(
    ...,
    shutdown_timeout=10.0,  # Increase from default 5.0
)
```

### Issue 4: High Memory Usage
**Solution**: Limit pending tasks
```python
# In execute_task():
if len(self._pending_tasks) > 1000:
    # Don't accept more tasks
    return AgentResult(..., success=False, error="Queue full")
```

## Testing

### Simple Test
```python
@pytest.mark.asyncio
async def test_agent():
    bus = MessageBus()
    agent = BaseAgent("test_1", AgentRole.SCOUT, bus)

    async with agent:
        assert agent.state == AgentState.ACTIVE
        metrics = agent.get_metrics()
        assert metrics["uptime_seconds"] >= 0
```

### Full Test Example
```python
from hololoom.redteam.swarm.tests.test_agent_base import TestAgent

@pytest.mark.asyncio
async def test_custom_agent():
    bus = MessageBus()
    agent = TestAgent("agent_1", AgentRole.SCOUT, bus)

    async with agent:
        task = AgentTask(
            task_type="probe",
            target="example.com",
            parameters={},
        )

        result = await agent.execute_task(task)
        assert result.success

        metrics = agent.get_metrics()
        assert metrics["tasks_completed"] == 1
```

## Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `agent_base.py` | Core implementation | 350 |
| `test_agent_base.py` | Test suite | 500 |
| `AGENT_BASE_DOCUMENTATION.md` | Full documentation | 600+ |
| `AGENT_BASE_QUICK_REFERENCE.md` | This file | 300+ |

## Key Classes

### AgentMetrics
Tracks agent performance:
```python
metrics = AgentMetrics()
metrics.record_success()
metrics.record_failure()
metrics.update_task_timing(245.3)
print(metrics.to_dict())
```

### BaseAgent
Main agent class:
```python
agent = BaseAgent(agent_id, role, message_bus)
await agent.start()
await agent.stop()
```

## See Also

- **Full Documentation**: `AGENT_BASE_DOCUMENTATION.md`
- **Implementation Summary**: `AGENT_BASE_IMPLEMENTATION_SUMMARY.md`
- **Wave 2 Report**: `WAVE_2_PHASE_4_REPORT.md`
- **Protocol Definitions**: `protocols.py`
- **Message Bus**: `communication.py`
- **Tests**: `tests/test_agent_base.py`
