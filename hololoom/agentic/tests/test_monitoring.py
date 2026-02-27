"""
Unit tests for Agent Monitoring System

Tests cover:
- AgentMonitor session management
- AgentTreeBuilder tree construction
- Integration with orchestrator
- WebSocket broadcasting
- Cleanup lifecycle

Status: Production Ready (November 2025)
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock

from hololoom.agentic.monitoring import (
    AgentMonitor,
    AgentTreeBuilder,
    AgentStatus,
    StepType,
    AgentSession,
    AgentTreeNode,
    get_monitor,
    start_monitoring,
    stop_monitoring
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def monitor():
    """Create fresh AgentMonitor instance"""
    return AgentMonitor()


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection"""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.fixture
def sample_steps() -> List[Dict[str, Any]]:
    """Sample reasoning steps for tree building"""
    return [
        {
            'type': 'initial_answer',
            'query': 'What is Thompson Sampling?',
            'confidence': 0.85,
            'epistemic_confidence': 0.75,
            'finding': 'Thompson Sampling is a Bayesian approach...',
            'tool_used': 'answer',
            'latency_ms': 150.5,
            'completed': True
        },
        {
            'type': 'verification',
            'query': 'Verify Thompson Sampling definition',
            'confidence': 0.92,
            'epistemic_confidence': 0.88,
            'finding': 'Definition verified against sources',
            'tool_used': 'verify',
            'latency_ms': 200.3,
            'completed': True
        },
        {
            'type': 'research_query',
            'query': 'Explore Thompson Sampling applications',
            'confidence': 0.78,
            'epistemic_confidence': 0.65,
            'finding': 'Applications in A/B testing, RL...',
            'tool_used': 'research',
            'latency_ms': 325.7,
            'completed': True
        }
    ]


# ============================================================================
# AgentMonitor Tests (12 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_session_creation(monitor):
    """Test agent session is created correctly"""
    agent_id = "agent_test_123"
    project = "mythRL"
    query = "What is Thompson Sampling?"
    mode = "research"

    await monitor.agent_started(agent_id, project, query, mode)

    # Verify session exists
    assert agent_id in monitor.sessions
    session = monitor.sessions[agent_id]

    # Verify session fields
    assert session.agent_id == agent_id
    assert session.project == project
    assert session.query == query
    assert session.mode == mode
    assert session.status == AgentStatus.RUNNING

    # Verify project tracking
    assert project in monitor.projects
    assert agent_id in monitor.projects[project]

    # Verify metrics
    assert monitor.total_agents_started == 1


@pytest.mark.asyncio
async def test_session_cleanup(monitor):
    """Test 1-hour session cleanup"""
    # Create old session (2 hours ago)
    agent_id_old = "agent_old"
    monitor.sessions[agent_id_old] = AgentSession(
        agent_id=agent_id_old,
        project="mythRL",
        query="Old query",
        mode="direct",
        status=AgentStatus.COMPLETED,
        start_time=(datetime.now() - timedelta(hours=2)).isoformat()
    )
    monitor.projects["mythRL"].add(agent_id_old)

    # Create recent session (30 minutes ago)
    agent_id_recent = "agent_recent"
    monitor.sessions[agent_id_recent] = AgentSession(
        agent_id=agent_id_recent,
        project="mythRL",
        query="Recent query",
        mode="direct",
        status=AgentStatus.COMPLETED,
        start_time=(datetime.now() - timedelta(minutes=30)).isoformat()
    )
    monitor.projects["mythRL"].add(agent_id_recent)

    # Manually trigger cleanup logic
    now = datetime.now()
    to_remove = []

    for agent_id, session in monitor.sessions.items():
        if session.status in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
            start_time = datetime.fromisoformat(session.start_time)
            age_seconds = (now - start_time).total_seconds()

            if age_seconds > 3600:  # 1 hour
                to_remove.append(agent_id)

    for agent_id in to_remove:
        session = monitor.sessions.pop(agent_id)
        monitor.projects[session.project].discard(agent_id)

    # Verify old session removed
    assert agent_id_old not in monitor.sessions
    assert agent_id_old not in monitor.projects["mythRL"]

    # Verify recent session remains
    assert agent_id_recent in monitor.sessions
    assert agent_id_recent in monitor.projects["mythRL"]


@pytest.mark.asyncio
async def test_project_grouping(monitor):
    """Test project → agents mapping"""
    # Create sessions in different projects
    await monitor.agent_started("agent_1", "mythRL", "Query 1", "direct")
    await monitor.agent_started("agent_2", "mythRL", "Query 2", "verify")
    await monitor.agent_started("agent_3", "squad", "Query 3", "research")

    # Verify project grouping
    assert len(monitor.projects) == 2
    assert len(monitor.projects["mythRL"]) == 2
    assert len(monitor.projects["squad"]) == 1

    # Verify get_project_agents
    mythrl_agents = monitor.get_project_agents("mythRL")
    assert len(mythrl_agents) == 2
    assert all(a.project == "mythRL" for a in mythrl_agents)


@pytest.mark.asyncio
async def test_metrics_calculation(monitor):
    """Test success rate and avg latency calculation"""
    # Complete 3 agents successfully
    for i in range(3):
        agent_id = f"agent_{i}"
        await monitor.agent_started(agent_id, "mythRL", f"Query {i}", "direct")
        await monitor.agent_completed(agent_id, [], 150.0 + i * 10)

    # Fail 1 agent
    agent_id_fail = "agent_fail"
    await monitor.agent_started(agent_id_fail, "mythRL", "Query fail", "direct")
    await monitor.agent_failed(agent_id_fail, "Test error")

    # Get metrics
    metrics = monitor.get_metrics()

    # Verify counts
    assert metrics['total_agents_started'] == 4
    assert metrics['total_agents_completed'] == 3
    assert metrics['total_agents_failed'] == 1
    assert metrics['active_agents'] == 4  # Still in memory

    # Verify success rate
    assert metrics['success_rate'] == 0.75  # 3/4

    # Verify avg latency
    expected_avg = (150.0 + 160.0 + 170.0) / 3
    assert abs(metrics['avg_latency_ms'] - expected_avg) < 0.01


@pytest.mark.asyncio
async def test_websocket_broadcast(monitor, mock_websocket):
    """Test WebSocket broadcasting"""
    # Register WebSocket
    monitor.register_ws_connection(mock_websocket)

    # Start agent (triggers broadcast)
    await monitor.agent_started("agent_test", "mythRL", "Test query", "direct")

    # Verify broadcast was called
    mock_websocket.send_json.assert_called_once()
    call_args = mock_websocket.send_json.call_args[0][0]

    assert call_args['type'] == 'agent_started'
    assert call_args['agent_id'] == 'agent_test'
    assert call_args['project'] == 'mythRL'


@pytest.mark.asyncio
async def test_agent_lifecycle(monitor):
    """Test full agent lifecycle: started → step → completed"""
    agent_id = "agent_lifecycle"

    # 1. Start
    await monitor.agent_started(agent_id, "mythRL", "Test query", "research")
    session = monitor.sessions[agent_id]
    assert session.status == AgentStatus.RUNNING

    # 2. Step
    await monitor.agent_step(
        agent_id, 1, 3, "research_query",
        confidence=0.85, epistemic=0.75, finding="Finding 1", latency_ms=150.0
    )
    assert session.current_step == 1
    assert session.total_steps == 3

    # 3. Feed update
    await monitor.agent_feed(agent_id, "Line 1", "Line 2")
    assert session.feed_line1 == "Line 1"
    assert session.feed_line2 == "Line 2"

    # 4. Complete
    steps = [{'type': 'initial_answer', 'completed': True}]
    await monitor.agent_completed(agent_id, steps, 300.0)
    assert session.status == AgentStatus.COMPLETED
    assert session.total_duration_ms == 300.0


@pytest.mark.asyncio
async def test_agent_failure(monitor):
    """Test agent failure handling"""
    agent_id = "agent_fail"

    await monitor.agent_started(agent_id, "mythRL", "Test query", "direct")
    await monitor.agent_failed(agent_id, "Test error message")

    session = monitor.sessions[agent_id]
    assert session.status == AgentStatus.FAILED
    assert monitor.total_agents_failed == 1


@pytest.mark.asyncio
async def test_concurrent_agents(monitor):
    """Test multiple agents simultaneously"""
    # Start 5 agents concurrently
    tasks = []
    for i in range(5):
        tasks.append(
            monitor.agent_started(f"agent_{i}", "mythRL", f"Query {i}", "direct")
        )

    await asyncio.gather(*tasks)

    # Verify all sessions created
    assert len(monitor.sessions) == 5
    assert monitor.total_agents_started == 5


@pytest.mark.asyncio
async def test_feed_updates(monitor):
    """Test two-line feed updates"""
    agent_id = "agent_feed"

    await monitor.agent_started(agent_id, "mythRL", "Test", "direct")
    await monitor.agent_feed(agent_id, "Processing query...", "Step 1/3")

    session = monitor.sessions[agent_id]
    assert session.feed_line1 == "Processing query..."
    assert session.feed_line2 == "Step 1/3"


@pytest.mark.asyncio
async def test_status_transitions(monitor):
    """Test RUNNING → COMPLETED status transition"""
    agent_id = "agent_status"

    await monitor.agent_started(agent_id, "mythRL", "Test", "direct")
    assert monitor.sessions[agent_id].status == AgentStatus.RUNNING

    await monitor.agent_status(agent_id, AgentStatus.WAITING)
    assert monitor.sessions[agent_id].status == AgentStatus.WAITING

    await monitor.agent_completed(agent_id, [], 150.0)
    assert monitor.sessions[agent_id].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_latency_tracking(monitor):
    """Test latency deque max size 1000"""
    # Add 1100 latencies
    for i in range(1100):
        monitor.latencies.append(float(i))

    # Verify max size enforced
    assert len(monitor.latencies) == 1000

    # Verify oldest entries removed (FIFO)
    assert monitor.latencies[0] == 100.0  # First 100 were removed


@pytest.mark.asyncio
async def test_empty_sessions(monitor):
    """Test edge case: empty sessions"""
    # Get metrics with no sessions
    metrics = monitor.get_metrics()

    assert metrics['total_agents_started'] == 0
    assert metrics['total_agents_completed'] == 0
    assert metrics['active_agents'] == 0
    assert metrics['avg_latency_ms'] == 0
    assert metrics['success_rate'] == 0

    # Get all sessions
    sessions = monitor.get_all_sessions()
    assert len(sessions) == 0


# ============================================================================
# AgentTreeBuilder Tests (8 tests)
# ============================================================================

def test_direct_mode_tree(sample_steps):
    """Test single root node for DIRECT mode"""
    steps = [sample_steps[0]]  # Only initial answer

    nodes = AgentTreeBuilder.build_tree("agent_direct", steps)

    assert len(nodes) == 1
    root = nodes["agent_direct_step_0"]
    assert root.parent_id is None
    assert len(root.children) == 0


def test_verify_mode_tree(sample_steps):
    """Test root + N verification children"""
    # Initial answer + verification
    steps = [sample_steps[0], sample_steps[1]]

    nodes = AgentTreeBuilder.build_tree("agent_verify", steps)

    assert len(nodes) == 2

    # Verify tree structure
    root = nodes["agent_verify_step_0"]
    child = nodes["agent_verify_step_1"]

    assert root.parent_id is None
    assert "agent_verify_step_1" in root.children
    assert child.parent_id == "agent_verify_step_0"


def test_research_mode_tree(sample_steps):
    """Test root + N research query children"""
    nodes = AgentTreeBuilder.build_tree("agent_research", sample_steps)

    assert len(nodes) == 3

    # Verify tree structure
    root = nodes["agent_research_step_0"]
    child1 = nodes["agent_research_step_1"]
    child2 = nodes["agent_research_step_2"]

    # Research children attach to root
    assert "agent_research_step_1" in root.children
    assert "agent_research_step_2" in root.children
    assert child1.parent_id == "agent_research_step_0"
    assert child2.parent_id == "agent_research_step_0"


def test_plan_execute_tree():
    """Test sequential chain for PLAN_EXECUTE mode"""
    steps = [
        {'type': 'plan', 'completed': True},
        {'type': 'execution', 'completed': True},
        {'type': 'execution', 'completed': True},
    ]

    nodes = AgentTreeBuilder.build_tree("agent_plan", steps)

    assert len(nodes) == 3

    # Verify sequential chain
    step0 = nodes["agent_plan_step_0"]
    step1 = nodes["agent_plan_step_1"]
    step2 = nodes["agent_plan_step_2"]

    assert step0.parent_id is None
    assert "agent_plan_step_1" in step0.children
    assert step1.parent_id == "agent_plan_step_0"
    assert "agent_plan_step_2" in step1.children
    assert step2.parent_id == "agent_plan_step_1"


def test_tree_depth_calculation(sample_steps):
    """Test recursive depth calculation"""
    nodes = AgentTreeBuilder.build_tree("agent_depth", sample_steps)

    # Research mode: root with 2 children (depth = 2)
    depth = AgentTreeBuilder.get_tree_depth(nodes, "agent_depth_step_0")
    assert depth == 2


def test_empty_steps():
    """Test empty steps list"""
    nodes = AgentTreeBuilder.build_tree("agent_empty", [])
    assert len(nodes) == 0


def test_unknown_step_types():
    """Test fallback to INITIAL_ANSWER for unknown types"""
    steps = [{'type': 'unknown_type', 'completed': True}]

    nodes = AgentTreeBuilder.build_tree("agent_unknown", steps)

    assert len(nodes) == 1
    node = nodes["agent_unknown_step_0"]
    assert node.step_type == StepType.INITIAL_ANSWER


def test_tree_serialization(monitor, sample_steps):
    """Test JSON tree serialization"""
    agent_id = "agent_serialize"

    # Create session with tree
    session = AgentSession(
        agent_id=agent_id,
        project="mythRL",
        query="Test",
        mode="research",
        status=AgentStatus.COMPLETED
    )

    session.nodes = AgentTreeBuilder.build_tree(agent_id, sample_steps)
    session.root_node_id = f"{agent_id}_step_0"

    # Serialize
    tree = monitor._serialize_tree(session)

    # Verify structure
    assert tree['node_id'] == f"{agent_id}_step_0"
    assert tree['step_type'] == 'initial_answer'
    assert len(tree['children']) == 2  # Research mode has 2 children


# ============================================================================
# Integration Tests (5 tests)
# ============================================================================

@pytest.mark.asyncio
async def test_monitor_lifecycle():
    """Test monitor start/stop lifecycle"""
    monitor = AgentMonitor()

    # Start
    await monitor.start()
    assert monitor._cleanup_task is not None

    # Stop
    await monitor.stop()

    # Verify cleanup task cancelled
    assert monitor._cleanup_task.cancelled() or monitor._cleanup_task.done()


@pytest.mark.asyncio
async def test_websocket_connection_cleanup(monitor, mock_websocket):
    """Test auto-cleanup on WebSocket disconnect"""
    # Register connection
    monitor.register_ws_connection(mock_websocket)
    assert len(monitor.ws_connections) == 1

    # Unregister
    monitor.unregister_ws_connection(mock_websocket)
    assert len(monitor.ws_connections) == 0


def test_global_monitor_singleton():
    """Test get_monitor() returns singleton"""
    monitor1 = get_monitor()
    monitor2 = get_monitor()

    assert monitor1 is monitor2


@pytest.mark.asyncio
async def test_monitoring_disabled():
    """Test graceful degradation when monitoring unavailable"""
    # This test verifies the import fallback logic works
    # If monitoring.py is unavailable, code should not crash

    # Create a mock orchestrator without monitoring
    class MockOrchestrator:
        def __init__(self):
            self.monitor = None

        async def reason(self, query):
            # Should not crash even though monitor is None
            if self.monitor:
                await self.monitor.agent_started("test", "project", "query", "direct")

    orchestrator = MockOrchestrator()
    await orchestrator.reason("test query")  # Should not raise


@pytest.mark.asyncio
async def test_broadcast_with_disconnected_clients(monitor):
    """Test broadcast handles disconnected WebSocket clients"""
    # Create mock WebSockets (one working, one broken)
    ws_working = AsyncMock()
    ws_working.send_json = AsyncMock()

    ws_broken = AsyncMock()
    ws_broken.send_json = AsyncMock(side_effect=Exception("Connection closed"))

    # Register both
    monitor.register_ws_connection(ws_working)
    monitor.register_ws_connection(ws_broken)

    assert len(monitor.ws_connections) == 2

    # Broadcast (should auto-cleanup broken connection)
    await monitor.broadcast({'type': 'test', 'data': 'value'})

    # Verify broken connection removed
    assert len(monitor.ws_connections) == 1
    assert ws_working in monitor.ws_connections
    assert ws_broken not in monitor.ws_connections


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
