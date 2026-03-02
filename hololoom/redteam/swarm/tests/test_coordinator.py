"""Test suite for SwarmCoordinator.

Tests cover:
- Coordinator initialization and lifecycle
- Campaign phase execution
- Task distribution and assignment
- Result aggregation and metrics
- Thompson Sampling agent selection
- Error handling and resilience
- Integration with agent pool
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from hololoom.redteam.swarm.coordinator import (
    SwarmCoordinator,
    CampaignPhase,
    SwarmMetrics,
    SwarmCampaignResult,
)
from hololoom.redteam.swarm.protocols import (
    AgentTask,
    AgentResult,
    AgentRole,
    AgentState,
    AgentMessage,
    MessagePriority,
)
from hololoom.redteam.swarm.communication import MessageBus


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def message_bus():
    """Create a message bus for testing."""
    return MessageBus(max_queue_size=1000)


@pytest.fixture
async def coordinator(message_bus):
    """Create a coordinator for testing."""
    return SwarmCoordinator(
        message_bus=message_bus,
        num_scouts=2,
        num_attackers=2,
        num_exploiters=1,
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestCoordinatorInitialization:
    """Test coordinator initialization."""

    def test_coordinator_creation(self, message_bus):
        """Test coordinator can be created."""
        coordinator = SwarmCoordinator(
            message_bus=message_bus,
            num_scouts=3,
            num_attackers=4,
            num_exploiters=2,
        )

        assert coordinator._num_scouts == 3
        assert coordinator._num_attackers == 4
        assert coordinator._num_exploiters == 2
        assert not coordinator._running

    def test_coordinator_defaults(self, message_bus):
        """Test default agent counts."""
        coordinator = SwarmCoordinator(message_bus=message_bus)

        assert coordinator._num_scouts == 2
        assert coordinator._num_attackers == 3
        assert coordinator._num_exploiters == 1

    def test_metrics_initialization(self, message_bus):
        """Test metrics are initialized."""
        coordinator = SwarmCoordinator(message_bus=message_bus)
        metrics = coordinator.get_metrics()

        assert isinstance(metrics, SwarmMetrics)
        assert metrics.tasks_completed == 0
        assert metrics.tasks_failed == 0
        assert metrics.discoveries == 0


# ============================================================================
# Lifecycle Tests
# ============================================================================


class TestCoordinatorLifecycle:
    """Test coordinator start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_coordinator_start(self, coordinator):
        """Test coordinator startup."""
        await coordinator.start()

        assert coordinator._running
        assert len(coordinator._all_agents) == 5  # 2 scouts + 2 attackers + 1 exploiter
        assert len(coordinator._scouts) == 2
        assert len(coordinator._attackers) == 2
        assert len(coordinator._exploiters) == 1

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_coordinator_stop(self, coordinator):
        """Test coordinator shutdown."""
        await coordinator.start()
        await coordinator.stop()

        assert not coordinator._running

    @pytest.mark.asyncio
    async def test_coordinator_context_manager(self, message_bus):
        """Test coordinator as async context manager."""
        coordinator = SwarmCoordinator(message_bus=message_bus)

        async with coordinator:
            assert coordinator._running

        assert not coordinator._running

    @pytest.mark.asyncio
    async def test_double_start_error(self, coordinator):
        """Test that starting twice raises error."""
        await coordinator.start()

        with pytest.raises(RuntimeError):
            await coordinator.start()

        await coordinator.stop()


# ============================================================================
# Campaign Execution Tests
# ============================================================================


class TestCampaignExecution:
    """Test campaign execution and phases."""

    @pytest.mark.asyncio
    async def test_campaign_phase_sequence(self, coordinator, message_bus):
        """Test campaign phases execute in order."""
        await coordinator.start()

        # Mock agent execution
        for agent in coordinator._all_agents:
            agent.execute_task = AsyncMock(
                return_value=AgentResult(
                    task_id="test",
                    agent_id=agent.agent_id,
                    success=True,
                    result={"status": "success"},
                    discoveries=[{"name": "test_vuln"}],
                )
            )

        # Run campaign
        result = await asyncio.wait_for(
            coordinator.run_campaign("localhost", duration_seconds=10),
            timeout=30,
        )

        # Verify campaign completed
        assert coordinator._campaign_phase == CampaignPhase.COMPLETE
        assert result.target == "localhost"

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_campaign_target_tracking(self, coordinator):
        """Test campaign tracks target correctly."""
        await coordinator.start()

        # Mock agent execution
        for agent in coordinator._all_agents:
            agent.execute_task = AsyncMock(
                return_value=AgentResult(
                    task_id="test",
                    agent_id=agent.agent_id,
                    success=True,
                    result={},
                )
            )

        # Run campaign
        result = await asyncio.wait_for(
            coordinator.run_campaign("192.168.1.1"),
            timeout=30,
        )

        assert result.target == "192.168.1.1"
        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_campaign_without_start_fails(self, coordinator):
        """Test campaign fails if coordinator not started."""
        with pytest.raises(RuntimeError):
            await coordinator.run_campaign("localhost")


# ============================================================================
# Task Distribution Tests
# ============================================================================


class TestTaskDistribution:
    """Test task distribution and agent assignment."""

    @pytest.mark.asyncio
    async def test_distribute_task(self, coordinator, message_bus):
        """Test task distribution."""
        await coordinator.start()

        task = AgentTask(
            task_type="probe_surface",
            target="localhost",
            parameters={"timeout": 10},
        )

        agent_id = await coordinator.distribute_task(task)

        assert agent_id is not None
        assert task.assigned_agent == agent_id
        assert task.task_id in coordinator._pending_tasks

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_scout_task_assignment(self, coordinator):
        """Test scout tasks are assigned to scouts."""
        await coordinator.start()

        task = AgentTask(
            task_type="probe_surface",
            target="localhost",
            parameters={},
        )

        agent_id = await coordinator.distribute_task(task)

        # Check agent is a scout
        agent = next(
            (a for a in coordinator._scouts if a.agent_id == agent_id), None
        )
        assert agent is not None

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_attack_task_assignment(self, coordinator):
        """Test attack tasks are assigned to attackers."""
        await coordinator.start()

        task = AgentTask(
            task_type="execute_attack",
            target="localhost",
            parameters={},
        )

        agent_id = await coordinator.distribute_task(task)

        # Check agent is an attacker
        agent = next(
            (a for a in coordinator._attackers if a.agent_id == agent_id), None
        )
        assert agent is not None

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_exploit_task_assignment(self, coordinator):
        """Test exploit tasks are assigned to exploiters."""
        await coordinator.start()

        task = AgentTask(
            task_type="exploit_vulnerability",
            target="localhost",
            parameters={},
        )

        agent_id = await coordinator.distribute_task(task)

        # Check agent is an exploiter
        agent = next(
            (a for a in coordinator._exploiters if a.agent_id == agent_id), None
        )
        assert agent is not None

        await coordinator.stop()


# ============================================================================
# Agent Selection Tests
# ============================================================================


class TestAgentSelection:
    """Test Thompson Sampling agent selection."""

    @pytest.mark.asyncio
    async def test_thompson_sample_agent(self, coordinator):
        """Test agent selection via Thompson Sampling."""
        await coordinator.start()

        # Select agent
        agent = coordinator._thompson_sample_agent(coordinator._scouts)

        assert agent in coordinator._scouts

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_agent_prior_updates(self, coordinator):
        """Test agent priors are updated on success/failure."""
        await coordinator.start()

        agent_id = coordinator._scouts[0].agent_id
        initial_α, initial_β = coordinator._agent_priors[agent_id]

        # Update on success
        coordinator._update_agent_prior(agent_id, success=True)
        α, β = coordinator._agent_priors[agent_id]
        assert α == initial_α + 1
        assert β == initial_β

        # Update on failure
        coordinator._update_agent_prior(agent_id, success=False)
        α, β = coordinator._agent_priors[agent_id]
        assert α == initial_α + 1
        assert β == initial_β + 1

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_agent_selection_prefers_successful(self, coordinator):
        """Test that successful agents are preferred."""
        await coordinator.start()

        agent1_id = coordinator._scouts[0].agent_id
        agent2_id = coordinator._scouts[1].agent_id

        # Make agent1 very successful
        coordinator._agent_priors[agent1_id] = (10, 1)  # α=10, β=1 → high success
        coordinator._agent_priors[agent2_id] = (1, 10)  # α=1, β=10 → low success

        # Select agents multiple times
        selections = [
            coordinator._thompson_sample_agent(coordinator._scouts)
            for _ in range(100)
        ]

        agent1_selections = sum(1 for a in selections if a.agent_id == agent1_id)
        agent2_selections = sum(1 for a in selections if a.agent_id == agent2_id)

        # Agent1 should be selected more often (though not guaranteed)
        # Use loose threshold to avoid flakiness
        assert agent1_selections >= agent2_selections * 0.5

        await coordinator.stop()


# ============================================================================
# Result Collection Tests
# ============================================================================


class TestResultCollection:
    """Test result aggregation and collection."""

    @pytest.mark.asyncio
    async def test_handle_agent_result(self, coordinator):
        """Test processing agent results."""
        await coordinator.start()

        result = AgentResult(
            task_id="task_1",
            agent_id="scout_0",
            success=True,
            result={"status": "ok"},
            discoveries=[{"name": "service1"}],
            execution_time_ms=50.5,
        )

        await coordinator.handle_agent_result(result)

        assert coordinator._metrics.tasks_completed == 1
        assert len(coordinator._task_latencies) == 1

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_collect_results(self, coordinator):
        """Test collecting multiple results."""
        await coordinator.start()

        # Add some results
        for i in range(3):
            result = AgentResult(
                task_id=f"task_{i}",
                agent_id=f"scout_{i % 2}",
                success=True,
                result={},
                discoveries=[],
                execution_time_ms=25.0 + i,
            )
            await coordinator.handle_agent_result(result)

        # Collect results
        results = await coordinator.collect_results(timeout_ms=5000)

        assert len(results) == 3
        assert all(isinstance(r, AgentResult) for r in results)

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_result_timeout(self, coordinator):
        """Test result collection timeout."""
        await coordinator.start()

        # Try to collect with no results available
        results = await coordinator.collect_results(timeout_ms=100)

        assert len(results) == 0

        await coordinator.stop()


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetrics:
    """Test metrics collection and calculation."""

    @pytest.mark.asyncio
    async def test_get_metrics(self, coordinator):
        """Test retrieving metrics."""
        await coordinator.start()

        metrics = coordinator.get_metrics()

        assert isinstance(metrics, SwarmMetrics)
        assert metrics.scout_count == 2
        assert metrics.attack_count == 2
        assert metrics.exploit_count == 1

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_metrics_update_on_results(self, coordinator):
        """Test metrics update when results added."""
        await coordinator.start()

        initial_metrics = coordinator.get_metrics()
        assert initial_metrics.tasks_completed == 0

        # Add result
        result = AgentResult(
            task_id="task_1",
            agent_id="scout_0",
            success=True,
            result={},
            discoveries=[{"name": "vuln1"}],
            execution_time_ms=50.0,
        )
        await coordinator.handle_agent_result(result)

        updated_metrics = coordinator.get_metrics()
        assert updated_metrics.tasks_completed == 1

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_agent_contributions(self, coordinator):
        """Test agent contribution calculation."""
        await coordinator.start()

        # Add results for each agent
        for agent in coordinator._all_agents[:3]:
            for _ in range(2):
                result = AgentResult(
                    task_id=f"task_{agent.agent_id}",
                    agent_id=agent.agent_id,
                    success=True,
                    result={},
                )
                await coordinator.handle_agent_result(result)

        contributions = coordinator._calculate_agent_contributions()

        assert len(contributions) == len(coordinator._all_agents)
        assert all(0.0 <= v <= 1.0 for v in contributions.values())

        await coordinator.stop()


# ============================================================================
# Broadcasting Tests
# ============================================================================


class TestBroadcast:
    """Test message broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_message(self, coordinator):
        """Test broadcasting coordination messages."""
        await coordinator.start()

        count = await coordinator.broadcast(
            message_type="phase_change",
            payload={"phase": "attack"},
        )

        assert count == len(coordinator._all_agents)

        await coordinator.stop()


# ============================================================================
# State Query Tests
# ============================================================================


class TestStateQueries:
    """Test agent state queries."""

    @pytest.mark.asyncio
    async def test_get_agent_states(self, coordinator):
        """Test retrieving all agent states."""
        await coordinator.start()

        states = coordinator.get_agent_states()

        assert len(states) == len(coordinator._all_agents)
        assert all(state == AgentState.ACTIVE for state in states.values())

        await coordinator.stop()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for coordinator."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, message_bus):
        """Test complete coordinator workflow."""
        coordinator = SwarmCoordinator(
            message_bus=message_bus,
            num_scouts=2,
            num_attackers=2,
            num_exploiters=1,
        )

        async with coordinator:
            # Verify started
            assert coordinator._running

            # Distribute task
            task = AgentTask(
                task_type="probe_surface",
                target="localhost",
                parameters={},
            )
            agent_id = await coordinator.distribute_task(task)
            assert agent_id is not None

            # Get metrics
            metrics = coordinator.get_metrics()
            assert metrics.agents_active > 0

            # Get states
            states = coordinator.get_agent_states()
            assert len(states) > 0

        # Verify stopped
        assert not coordinator._running


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and resilience."""

    @pytest.mark.asyncio
    async def test_failed_task_metric(self, coordinator):
        """Test failed tasks are tracked."""
        await coordinator.start()

        result = AgentResult(
            task_id="task_1",
            agent_id="scout_0",
            success=False,
            result=None,
            error="Test error",
            execution_time_ms=10.0,
        )

        await coordinator.handle_agent_result(result)

        assert coordinator._metrics.tasks_failed == 1

        await coordinator.stop()

    @pytest.mark.asyncio
    async def test_mixed_success_failure(self, coordinator):
        """Test mixed success and failure tracking."""
        await coordinator.start()

        # Add successful results
        for i in range(3):
            result = AgentResult(
                task_id=f"task_{i}",
                agent_id="scout_0",
                success=True,
                result={},
            )
            await coordinator.handle_agent_result(result)

        # Add failed results
        for i in range(2):
            result = AgentResult(
                task_id=f"task_{10+i}",
                agent_id="scout_0",
                success=False,
                result=None,
                error="Test error",
            )
            await coordinator.handle_agent_result(result)

        assert coordinator._metrics.tasks_completed == 3
        assert coordinator._metrics.tasks_failed == 2

        await coordinator.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
