"""Tests for BaseAgent class and agent lifecycle management.

CARTS Phase 4: Agent Base Tests
Status: Comprehensive Test Suite (November 2025)

Test Coverage:
- Agent lifecycle (creation, start, stop, shutdown)
- Message handling and routing
- Task execution and metrics
- State transitions
- Error handling and recovery
- Async patterns and resource cleanup
"""

import asyncio
import pytest
from typing import Optional

from hololoom.redteam.swarm.agent_base import BaseAgent, AgentMetrics
from hololoom.redteam.swarm.communication import MessageBus
from hololoom.redteam.swarm.protocols import (
    AgentMessage,
    AgentRole,
    AgentState,
    AgentTask,
    AgentResult,
    MessagePriority,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def message_bus():
    """Create a message bus for testing."""
    return MessageBus(max_queue_size=1000)


@pytest.fixture
def base_agent(message_bus):
    """Create a base agent for testing."""
    return BaseAgent(
        agent_id="test_agent_1",
        role=AgentRole.SCOUT,
        message_bus=message_bus,
        message_handler_timeout=0.5,
        shutdown_timeout=2.0,
    )


class TestAgent(BaseAgent):
    """Concrete agent implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_tasks = []
        self.received_messages = []

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a test task."""
        start_time = task.created_at
        self.executed_tasks.append(task)

        # Simulate some work
        await asyncio.sleep(0.01)

        if task.parameters.get("fail"):
            return AgentResult(
                task_id=task.task_id,
                agent_id=self._agent_id,
                success=False,
                result=None,
                error="Task failed intentionally",
                execution_time_ms=10.0,
            )

        return AgentResult(
            task_id=task.task_id,
            agent_id=self._agent_id,
            success=True,
            result={"executed": True},
            discoveries=[],
            execution_time_ms=10.0,
        )

    async def _handle_status_message(self, message: AgentMessage) -> None:
        """Track received status messages."""
        self.received_messages.append(message)


@pytest.fixture
def test_agent(message_bus):
    """Create a test agent implementation."""
    return TestAgent(
        agent_id="test_agent_1",
        role=AgentRole.SCOUT,
        message_bus=message_bus,
        message_handler_timeout=0.5,
        shutdown_timeout=2.0,
    )


# ============================================================================
# Lifecycle Tests
# ============================================================================


@pytest.mark.asyncio
async def test_agent_creation(base_agent):
    """Test agent creation and initial state."""
    assert base_agent.agent_id == "test_agent_1"
    assert base_agent.role == AgentRole.SCOUT
    assert base_agent.state == AgentState.IDLE


@pytest.mark.asyncio
async def test_agent_start(base_agent):
    """Test agent startup."""
    await base_agent.start()
    assert base_agent.state == AgentState.ACTIVE
    assert base_agent._message_handler_task is not None
    assert not base_agent._message_handler_task.done()

    await base_agent.stop()


@pytest.mark.asyncio
async def test_agent_stop(base_agent):
    """Test agent shutdown."""
    await base_agent.start()
    assert base_agent.state == AgentState.ACTIVE

    await base_agent.stop()
    assert base_agent.state == AgentState.SHUTDOWN

    # Message handler should be done
    if base_agent._message_handler_task:
        assert base_agent._message_handler_task.done()


@pytest.mark.asyncio
async def test_agent_context_manager(base_agent):
    """Test agent as async context manager."""
    assert base_agent.state == AgentState.IDLE

    async with base_agent:
        assert base_agent.state == AgentState.ACTIVE

    assert base_agent.state == AgentState.SHUTDOWN


@pytest.mark.asyncio
async def test_agent_start_invalid_state(base_agent):
    """Test starting agent from non-IDLE state."""
    await base_agent.start()

    with pytest.raises(RuntimeError):
        await base_agent.start()

    await base_agent.stop()


@pytest.mark.asyncio
async def test_agent_stop_graceful(base_agent):
    """Test graceful shutdown."""
    await base_agent.start()

    # Let message handler run briefly
    await asyncio.sleep(0.1)

    # Stop should complete without error
    await base_agent.stop()
    assert base_agent.state == AgentState.SHUTDOWN


# ============================================================================
# Message Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_send_message(test_agent, message_bus):
    """Test sending a message."""
    success = await test_agent.send_message(
        recipient="agent_2",
        message_type="status",
        payload={"status": "idle"},
    )

    assert success
    assert test_agent._metrics.messages_sent == 1

    # Verify message in bus
    msg = await message_bus.receive("agent_2", timeout=1.0)
    assert msg is not None
    assert msg.sender == "test_agent_1"
    assert msg.message_type == "status"


@pytest.mark.asyncio
async def test_send_message_queue_full(message_bus):
    """Test sending message when queue is full."""
    # Create bus with small queue
    small_bus = MessageBus(max_queue_size=1)
    agent = BaseAgent(
        agent_id="test_agent",
        role=AgentRole.SCOUT,
        message_bus=small_bus,
    )

    # Fill the queue
    await small_bus.send(
        AgentMessage(
            sender="other",
            recipient="target",
            message_type="status",
            payload={},
        )
    )

    # This should fail
    success = await agent.send_message(
        recipient="target",
        message_type="status",
        payload={},
    )

    # Second message should fail (queue full)
    # Note: First message creates queue, second fills it
    assert not success


@pytest.mark.asyncio
async def test_broadcast_message(test_agent, message_bus):
    """Test broadcasting a message."""
    # Create multiple agents
    agent2 = BaseAgent("agent_2", AgentRole.ATTACKER, message_bus)
    agent3 = BaseAgent("agent_3", AgentRole.EXPLOITER, message_bus)

    # Start agents so they have queues
    await agent2.start()
    await agent3.start()

    # Broadcast message
    count = await test_agent.broadcast(
        message_type="alert",
        payload={"alert": "intrusion"},
    )

    assert count >= 2  # At least agent_2 and agent_3
    assert test_agent._metrics.messages_sent == 1

    await agent2.stop()
    await agent3.stop()


@pytest.mark.asyncio
async def test_handle_message_routing(test_agent):
    """Test message routing in handle_message."""
    # Create messages of different types
    task_message = AgentMessage(
        sender="coordinator",
        recipient="test_agent_1",
        message_type="task",
        payload={"task": {
            "task_type": "probe_surface",
            "target": "example.com",
            "parameters": {},
            "task_id": "task_1",
            "priority": "NORMAL",
            "timeout_seconds": 30.0,
        }},
    )

    status_message = AgentMessage(
        sender="agent_2",
        recipient="test_agent_1",
        message_type="status",
        payload={"status": "idle"},
    )

    # Handle messages
    await test_agent.handle_message(task_message)
    await test_agent.handle_message(status_message)

    assert len(test_agent._pending_tasks) == 1
    assert len(test_agent.received_messages) == 1


@pytest.mark.asyncio
async def test_message_acknowledgment(test_agent, message_bus):
    """Test message acknowledgment."""
    message = AgentMessage(
        sender="agent_2",
        recipient="test_agent_1",
        message_type="status",
        payload={},
        requires_ack=True,
    )

    await test_agent.handle_message(message)
    assert test_agent._metrics.messages_acked == 1

    # Check ack was sent
    ack = await message_bus.receive("agent_2", timeout=1.0)
    assert ack is not None
    assert ack.message_type == "ack"


@pytest.mark.asyncio
async def test_command_message_handling(test_agent):
    """Test coordinator command message handling."""
    # Test pause command
    pause_cmd = AgentMessage(
        sender="coordinator",
        recipient="test_agent_1",
        message_type="command",
        payload={"command": "pause"},
    )

    await test_agent.handle_message(pause_cmd)
    assert test_agent.state == AgentState.WAITING

    # Test resume command
    resume_cmd = AgentMessage(
        sender="coordinator",
        recipient="test_agent_1",
        message_type="command",
        payload={"command": "resume"},
    )

    await test_agent.handle_message(resume_cmd)
    assert test_agent.state == AgentState.ACTIVE


# ============================================================================
# Task Execution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_execute_task_success(test_agent):
    """Test successful task execution."""
    task = AgentTask(
        task_type="probe_surface",
        target="example.com",
        parameters={},
        task_id="task_1",
    )

    result = await test_agent.execute_task(task)

    assert result.success
    assert result.agent_id == "test_agent_1"
    assert result.execution_time_ms >= 0
    assert task in test_agent.executed_tasks


@pytest.mark.asyncio
async def test_execute_task_failure(test_agent):
    """Test failed task execution."""
    task = AgentTask(
        task_type="attack",
        target="example.com",
        parameters={"fail": True},
        task_id="task_2",
    )

    result = await test_agent.execute_task(task)

    assert not result.success
    assert result.error is not None
    assert task in test_agent.executed_tasks


@pytest.mark.asyncio
async def test_default_execute_task(base_agent):
    """Test default execute_task (not implemented)."""
    task = AgentTask(
        task_type="probe",
        target="example.com",
        parameters={},
    )

    result = await base_agent.execute_task(task)

    assert not result.success
    assert "not implemented" in result.error.lower()


# ============================================================================
# Metrics Tests
# ============================================================================


@pytest.mark.asyncio
async def test_metrics_task_tracking(test_agent):
    """Test metrics track task execution."""
    task1 = AgentTask(
        task_type="probe",
        target="example.com",
        parameters={},
    )

    task2 = AgentTask(
        task_type="attack",
        target="example.com",
        parameters={"fail": True},
    )

    # Execute tasks
    result1 = await test_agent.execute_task(task1)
    test_agent._update_metrics(task1, result1)

    result2 = await test_agent.execute_task(task2)
    test_agent._update_metrics(task2, result2)

    # Check metrics
    metrics = test_agent.get_metrics()
    assert metrics["tasks_completed"] == 1
    assert metrics["tasks_failed"] == 1
    assert metrics["tasks_total"] == 2
    assert metrics["success_rate"] == 0.5


@pytest.mark.asyncio
async def test_metrics_message_tracking(test_agent):
    """Test metrics track message handling."""
    await test_agent.send_message("agent_2", "status", {})
    await test_agent.send_message("agent_3", "status", {})

    msg = AgentMessage(
        sender="agent_2",
        recipient="test_agent_1",
        message_type="status",
        payload={},
    )
    await test_agent.handle_message(msg)

    metrics = test_agent.get_metrics()
    assert metrics["messages_sent"] == 2
    assert metrics["messages_received"] == 1


@pytest.mark.asyncio
async def test_metrics_uptime(test_agent):
    """Test metrics uptime tracking."""
    await asyncio.sleep(0.1)
    metrics = test_agent.get_metrics()

    assert metrics["uptime_seconds"] >= 0.1


@pytest.mark.asyncio
async def test_metrics_throughput(test_agent):
    """Test metrics message throughput."""
    await test_agent.send_message("agent_2", "status", {})
    await asyncio.sleep(0.1)

    metrics = test_agent.get_metrics()
    throughput = metrics["message_throughput"]

    # Should be at least 10 msgs/sec (1 msg / 0.1 sec)
    assert throughput >= 5


# ============================================================================
# State Management Tests
# ============================================================================


@pytest.mark.asyncio
async def test_state_transitions(test_agent):
    """Test agent state transitions."""
    assert test_agent.state == AgentState.IDLE

    await test_agent.start()
    assert test_agent.state == AgentState.ACTIVE

    # Create and process task
    task = AgentTask(
        task_type="probe",
        target="example.com",
        parameters={},
    )

    result = await test_agent.execute_task(task)
    test_agent._update_metrics(task, result)

    await test_agent.stop()
    assert test_agent.state == AgentState.SHUTDOWN


@pytest.mark.asyncio
async def test_state_pause_resume(test_agent):
    """Test pause and resume state transitions."""
    await test_agent.start()
    assert test_agent.state == AgentState.ACTIVE

    # Pause
    pause_msg = AgentMessage(
        sender="coordinator",
        recipient="test_agent_1",
        message_type="command",
        payload={"command": "pause"},
    )
    await test_agent.handle_message(pause_msg)
    assert test_agent.state == AgentState.WAITING

    # Resume
    resume_msg = AgentMessage(
        sender="coordinator",
        recipient="test_agent_1",
        message_type="command",
        payload={"command": "resume"},
    )
    await test_agent.handle_message(resume_msg)
    assert test_agent.state == AgentState.ACTIVE

    await test_agent.stop()


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_error_in_task_execution(test_agent):
    """Test error handling in task execution."""
    task = AgentTask(
        task_type="attack",
        target="example.com",
        parameters={"fail": True},
    )

    result = await test_agent.execute_task(task)
    test_agent._update_metrics(task, result)

    metrics = test_agent.get_metrics()
    assert metrics["tasks_failed"] == 1


@pytest.mark.asyncio
async def test_error_recovery(test_agent):
    """Test error recovery tracking."""
    test_agent._metrics.record_failure()
    assert test_agent._metrics.error_count == 1

    test_agent._metrics.record_recovery()
    assert test_agent._metrics.recovery_count == 1


@pytest.mark.asyncio
async def test_message_handler_error_recovery(test_agent, message_bus):
    """Test message handler continues despite errors."""
    await test_agent.start()

    # Send a message with bad task data (should be handled gracefully)
    bad_task_msg = AgentMessage(
        sender="coordinator",
        recipient="test_agent_1",
        message_type="task",
        payload={"task": None},  # Invalid task
    )

    await message_bus.send(bad_task_msg)

    # Send a valid message
    valid_msg = AgentMessage(
        sender="agent_2",
        recipient="test_agent_1",
        message_type="status",
        payload={"status": "idle"},
    )

    await message_bus.send(valid_msg)

    # Wait for processing
    await asyncio.sleep(0.2)

    # Valid message should have been received
    assert len(test_agent.received_messages) == 1

    await test_agent.stop()


# ============================================================================
# Concurrent Operations Tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_message_handling(test_agent, message_bus):
    """Test agent handles multiple concurrent messages."""
    await test_agent.start()

    # Send multiple messages
    for i in range(5):
        msg = AgentMessage(
            sender=f"agent_{i}",
            recipient="test_agent_1",
            message_type="status",
            payload={"status": "active"},
        )
        await message_bus.send(msg)

    # Wait for processing
    await asyncio.sleep(0.5)

    # All messages should be received
    assert len(test_agent.received_messages) == 5

    await test_agent.stop()


@pytest.mark.asyncio
async def test_concurrent_task_execution(test_agent):
    """Test agent can handle concurrent task requests."""
    tasks = [
        AgentTask(
            task_type="probe",
            target=f"target_{i}",
            parameters={},
        )
        for i in range(3)
    ]

    # Execute all tasks concurrently
    results = await asyncio.gather(*[test_agent.execute_task(t) for t in tasks])

    assert len(results) == 3
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_concurrent_send_receive(test_agent, message_bus):
    """Test concurrent message sending and receiving."""
    async def send_messages():
        for i in range(5):
            await test_agent.send_message(
                f"agent_{i}",
                "status",
                {"status": "active"},
            )
            await asyncio.sleep(0.01)

    async def receive_messages():
        received = []
        for i in range(5):
            msg = await message_bus.receive(f"agent_{i}", timeout=1.0)
            if msg:
                received.append(msg)
        return received

    # Run send and receive concurrently
    sent, received = await asyncio.gather(
        send_messages(),
        receive_messages(),
    )

    assert len(received) == 5


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_agent_full_lifecycle(test_agent, message_bus):
    """Test complete agent lifecycle."""
    # Start
    await test_agent.start()
    assert test_agent.state == AgentState.ACTIVE

    # Send and receive messages
    await test_agent.send_message("agent_2", "status", {"status": "active"})

    # Create and execute task
    task = AgentTask(
        task_type="probe",
        target="example.com",
        parameters={},
    )
    result = await test_agent.execute_task(task)
    test_agent._update_metrics(task, result)

    # Process incoming message
    msg = AgentMessage(
        sender="agent_2",
        recipient="test_agent_1",
        message_type="status",
        payload={"status": "working"},
    )
    await test_agent.handle_message(msg)

    # Get metrics
    metrics = test_agent.get_metrics()
    assert metrics["tasks_completed"] == 1
    assert metrics["messages_sent"] == 1
    assert metrics["messages_received"] == 1

    # Stop
    await test_agent.stop()
    assert test_agent.state == AgentState.SHUTDOWN


@pytest.mark.asyncio
async def test_multi_agent_coordination(message_bus):
    """Test coordination between multiple agents."""
    agent1 = TestAgent("agent_1", AgentRole.SCOUT, message_bus)
    agent2 = TestAgent("agent_2", AgentRole.ATTACKER, message_bus)

    await agent1.start()
    await agent2.start()

    # Agent 1 sends to Agent 2
    await agent1.send_message("agent_2", "status", {"status": "ready"})

    # Agent 2 receives and responds
    await asyncio.sleep(0.2)

    await agent2.send_message("agent_1", "status", {"status": "acknowledged"})

    # Check metrics
    metrics1 = agent1.get_metrics()
    metrics2 = agent2.get_metrics()

    assert metrics1["messages_sent"] >= 1
    assert metrics2["messages_received"] >= 1

    await agent1.stop()
    await agent2.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
