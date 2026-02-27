"""
Tests for Phase 7.1: Multi-Agent Communication Protocol
=======================================================

Comprehensive test suite covering:
- Data types (enums, dataclasses)
- AgentMessage creation and serialization
- AgentRegistry registration and discovery
- MessageBus pub/sub and request/response
- SharedMemory with namespaces and TTL
- MultiAgentCoordinator high-level operations

Author: Claude Code
Date: 2025-12-09
"""

import pytest
import asyncio
from typing import List
from datetime import datetime, timedelta

from hololoom.agentic.multi_agent import (
    # Enums
    AgentCapability,
    MessageType,
    MessagePriority,
    AgentStatus,
    # Data classes
    AgentMessage,
    AgentInfo,
    SharedMemoryEntry,
    # Core classes
    AgentRegistry,
    MessageBus,
    SharedMemory,
    MultiAgentCoordinator,
    # Convenience functions
    create_coordinator,
    create_agent_message,
    create_research_agent,
    create_synthesis_agent,
    create_code_agent,
)

from hololoom.agentic.context_handoff import ContextItem


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def research_agent() -> AgentInfo:
    """Create a research agent for testing."""
    return AgentInfo(
        id="research-001",
        name="Research Bot",
        capabilities={AgentCapability.RESEARCH, AgentCapability.ANALYSIS, AgentCapability.VERIFICATION}
    )


@pytest.fixture
def synthesis_agent() -> AgentInfo:
    """Create a synthesis agent for testing."""
    return AgentInfo(
        id="synthesis-001",
        name="Synthesis Bot",
        capabilities={AgentCapability.SUMMARIZATION, AgentCapability.SYNTHESIS, AgentCapability.GENERATION}
    )


@pytest.fixture
def code_agent() -> AgentInfo:
    """Create a code review agent for testing."""
    return AgentInfo(
        id="code-001",
        name="Code Reviewer",
        capabilities={AgentCapability.CODE_REVIEW, AgentCapability.CODE_GENERATION, AgentCapability.DEBUGGING}
    )


@pytest.fixture
def sample_message() -> AgentMessage:
    """Create a sample message for testing."""
    return create_agent_message(
        from_agent="agent-a",
        to_agent="agent-b",
        msg_type=MessageType.REQUEST,
        content={"task": "analyze", "query": "What is Thompson Sampling?"},
        priority=MessagePriority.NORMAL
    )


@pytest.fixture
def context_items() -> List[ContextItem]:
    """Create sample context items."""
    return [
        ContextItem(
            content="Thompson Sampling is a Bayesian approach to exploration.",
            source_agent="research",
            mi_score=0.9,
        ),
        ContextItem(
            content="It balances exploration and exploitation.",
            source_agent="research",
            mi_score=0.8,
        ),
    ]


@pytest.fixture
def registry() -> AgentRegistry:
    """Create a fresh agent registry."""
    return AgentRegistry()


@pytest.fixture
def message_bus(registry) -> MessageBus:
    """Create a message bus with registry."""
    return MessageBus(registry)


@pytest.fixture
def shared_memory() -> SharedMemory:
    """Create shared memory instance."""
    return SharedMemory()


@pytest.fixture
def coordinator() -> MultiAgentCoordinator:
    """Create a coordinator instance."""
    return create_coordinator()


# ============================================================================
# Enum Tests
# ============================================================================

class TestEnums:
    """Tests for enum types."""

    def test_agent_capability_values(self):
        """AgentCapability should have expected values."""
        assert AgentCapability.RESEARCH.value == "research"
        assert AgentCapability.ANALYSIS.value == "analysis"
        assert AgentCapability.CODE_REVIEW.value == "code_review"
        assert AgentCapability.SUMMARIZATION.value == "summarization"
        # Check total count (19 capabilities including GENERAL, DATA_ANALYSIS, CONTENT_CREATION, REASONING)
        assert len(AgentCapability) == 19

    def test_message_type_values(self):
        """MessageType should have expected values."""
        assert MessageType.REQUEST.value == "request"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.HANDOFF.value == "handoff"
        assert MessageType.CONTEXT.value == "context"

    def test_message_priority_ordering(self):
        """MessagePriority should be comparable."""
        assert MessagePriority.CRITICAL.value > MessagePriority.HIGH.value
        assert MessagePriority.HIGH.value > MessagePriority.NORMAL.value
        assert MessagePriority.NORMAL.value > MessagePriority.LOW.value

    def test_agent_status_values(self):
        """AgentStatus should have expected values."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.BUSY.value == "busy"
        assert AgentStatus.OFFLINE.value == "offline"


# ============================================================================
# AgentMessage Tests
# ============================================================================

class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_create_message(self, sample_message):
        """Should create message with required fields."""
        assert sample_message.from_agent == "agent-a"
        assert sample_message.to_agent == "agent-b"
        assert sample_message.type == MessageType.REQUEST
        assert sample_message.priority == MessagePriority.NORMAL
        assert "task" in sample_message.content

    def test_message_has_id(self, sample_message):
        """Message should have auto-generated ID."""
        assert sample_message.id is not None
        assert len(sample_message.id) > 0

    def test_message_has_timestamp(self, sample_message):
        """Message should have timestamp."""
        assert sample_message.timestamp is not None

    def test_message_not_expired(self, sample_message):
        """Fresh message should not be expired."""
        assert not sample_message.is_expired()

    def test_message_expiration(self):
        """Message should expire after ttl_seconds."""
        msg = AgentMessage(
            from_agent="a",
            to_agent="b",
            type=MessageType.REQUEST,
            content={},
            ttl_seconds=1  # 1 second TTL
        )
        assert not msg.is_expired()
        import time
        time.sleep(1.5)
        assert msg.is_expired()

    def test_create_response(self, sample_message):
        """Should create response to message."""
        response = sample_message.create_response(
            content={"answer": "Thompson Sampling uses Bayesian priors"},
            from_agent="agent-b"
        )

        assert response.correlation_id == sample_message.id
        assert response.from_agent == "agent-b"
        assert response.to_agent == sample_message.from_agent
        assert response.type == MessageType.RESPONSE
        assert "answer" in response.content

    def test_message_serialization(self, sample_message, context_items):
        """Message should serialize and deserialize correctly."""
        sample_message.context = context_items

        data = sample_message.to_dict()
        restored = AgentMessage.from_dict(data)

        assert restored.id == sample_message.id
        assert restored.from_agent == sample_message.from_agent
        assert restored.to_agent == sample_message.to_agent
        assert restored.type == sample_message.type
        assert restored.content == sample_message.content
        assert len(restored.context) == len(sample_message.context)

    def test_message_with_context(self, sample_message, context_items):
        """Message should carry context items."""
        sample_message.context = context_items

        assert len(sample_message.context) == 2
        assert sample_message.context[0].mi_score == 0.9


# ============================================================================
# AgentInfo Tests
# ============================================================================

class TestAgentInfo:
    """Tests for AgentInfo dataclass."""

    def test_create_research_agent(self, research_agent):
        """Should create research agent with correct capabilities."""
        assert research_agent.id == "research-001"
        assert research_agent.name == "Research Bot"
        assert AgentCapability.RESEARCH in research_agent.capabilities
        assert AgentCapability.ANALYSIS in research_agent.capabilities
        assert research_agent.status == AgentStatus.IDLE

    def test_create_synthesis_agent(self, synthesis_agent):
        """Should create synthesis agent with correct capabilities."""
        assert AgentCapability.SUMMARIZATION in synthesis_agent.capabilities
        assert AgentCapability.SYNTHESIS in synthesis_agent.capabilities

    def test_create_code_agent(self, code_agent):
        """Should create code agent with correct capabilities."""
        assert AgentCapability.CODE_REVIEW in code_agent.capabilities
        assert AgentCapability.CODE_GENERATION in code_agent.capabilities

    def test_agent_performance_tracking(self, research_agent):
        """Should track agent performance metrics."""
        # Initial state
        assert research_agent.total_requests == 0
        assert research_agent.success_rate == 1.0  # Default is 1.0
        assert research_agent.avg_response_time_ms == 0.0

        # Record success
        research_agent.update_metrics(response_time_ms=150.0, success=True)
        assert research_agent.total_requests == 1
        assert research_agent.success_rate == 1.0
        assert research_agent.avg_response_time_ms == 150.0

        # Record failure
        research_agent.update_metrics(response_time_ms=200.0, success=False)
        assert research_agent.total_requests == 2
        assert research_agent.success_rate == 0.5
        assert research_agent.avg_response_time_ms == 175.0

    def test_agent_availability(self, research_agent):
        """Should check agent availability correctly."""
        assert research_agent.is_available()

        research_agent.status = AgentStatus.BUSY
        assert not research_agent.is_available()

        research_agent.status = AgentStatus.OFFLINE
        assert not research_agent.is_available()

    def test_agent_serialization(self, research_agent):
        """Agent should serialize and deserialize correctly."""
        research_agent.update_metrics(response_time_ms=100.0, success=True)

        data = research_agent.to_dict()

        assert data["id"] == research_agent.id
        assert data["name"] == research_agent.name
        assert data["total_requests"] == research_agent.total_requests


# ============================================================================
# AgentRegistry Tests
# ============================================================================

class TestAgentRegistry:
    """Tests for AgentRegistry."""

    @pytest.mark.asyncio
    async def test_register_agent(self, registry, research_agent):
        """Should register agent successfully."""
        result = await registry.register(research_agent)
        assert result is True
        assert registry.get(research_agent.id) is not None

    @pytest.mark.asyncio
    async def test_register_duplicate_updates(self, registry, research_agent):
        """Should update existing registration."""
        await registry.register(research_agent)
        # Modify agent
        research_agent.name = "Updated Name"
        result = await registry.register(research_agent)
        assert result is True
        assert registry.get(research_agent.id).name == "Updated Name"

    @pytest.mark.asyncio
    async def test_unregister_agent(self, registry, research_agent):
        """Should unregister agent successfully."""
        await registry.register(research_agent)
        result = await registry.unregister(research_agent.id)
        assert result is True
        assert registry.get(research_agent.id) is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, registry):
        """Should return False for nonexistent agent."""
        result = await registry.unregister("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_find_by_capability(self, registry, research_agent, synthesis_agent):
        """Should find agents by capability."""
        await registry.register(research_agent)
        await registry.register(synthesis_agent)

        # Find research agents
        research_agents = registry.find_by_capability(AgentCapability.RESEARCH)
        assert len(research_agents) == 1
        assert research_agents[0].id == "research-001"

        # Find summarization agents
        summarization_agents = registry.find_by_capability(AgentCapability.SUMMARIZATION)
        assert len(summarization_agents) == 1
        assert summarization_agents[0].id == "synthesis-001"

    @pytest.mark.asyncio
    async def test_find_by_capability_only_available(self, registry, research_agent):
        """Should filter unavailable agents."""
        await registry.register(research_agent)
        research_agent.status = AgentStatus.BUSY

        # With only_available=True
        agents = registry.find_by_capability(AgentCapability.RESEARCH, only_available=True)
        assert len(agents) == 0

        # With only_available=False
        agents = registry.find_by_capability(AgentCapability.RESEARCH, only_available=False)
        assert len(agents) == 1

    @pytest.mark.asyncio
    async def test_get_best_agent_fastest(self, registry):
        """Should return fastest agent."""
        agent1 = AgentInfo(
            id="r1",
            name="Research 1",
            capabilities={AgentCapability.RESEARCH}
        )
        agent2 = AgentInfo(
            id="r2",
            name="Research 2",
            capabilities={AgentCapability.RESEARCH}
        )
        agent1.update_metrics(response_time_ms=200.0, success=True)
        agent2.update_metrics(response_time_ms=100.0, success=True)  # Faster

        await registry.register(agent1)
        await registry.register(agent2)

        best = registry.get_best_agent(AgentCapability.RESEARCH, strategy="fastest")
        assert best.id == "r2"  # Lower response time

    @pytest.mark.asyncio
    async def test_get_best_agent_reliable(self, registry):
        """Should prefer most reliable agent."""
        agent1 = AgentInfo(
            id="r1",
            name="Research 1",
            capabilities={AgentCapability.RESEARCH}
        )
        agent2 = AgentInfo(
            id="r2",
            name="Research 2",
            capabilities={AgentCapability.RESEARCH}
        )
        agent1.update_metrics(response_time_ms=100.0, success=True)
        agent1.update_metrics(response_time_ms=100.0, success=False)  # 50% success
        agent2.update_metrics(response_time_ms=100.0, success=True)
        agent2.update_metrics(response_time_ms=100.0, success=True)  # 100% success

        await registry.register(agent1)
        await registry.register(agent2)

        best = registry.get_best_agent(AgentCapability.RESEARCH, strategy="reliable")
        assert best.id == "r2"  # Higher success rate

    @pytest.mark.asyncio
    async def test_get_all_agents(self, registry, research_agent, synthesis_agent):
        """Should list all registered agents."""
        await registry.register(research_agent)
        await registry.register(synthesis_agent)

        all_agents = registry.get_all_agents()
        assert len(all_agents) == 2


# ============================================================================
# MessageBus Tests
# ============================================================================

class TestMessageBus:
    """Tests for MessageBus."""

    @pytest.mark.asyncio
    async def test_subscribe_and_send(self, message_bus, research_agent):
        """Should deliver message to subscriber."""
        await message_bus.registry.register(research_agent)

        received_messages = []

        async def handler(msg: AgentMessage):
            received_messages.append(msg)
            return None

        message_bus.subscribe(research_agent.id, handler)

        msg = create_agent_message(
            from_agent="sender",
            to_agent=research_agent.id,
            msg_type=MessageType.REQUEST,
            content={"task": "analyze"}
        )

        result = await message_bus.send(msg)
        await asyncio.sleep(0.1)  # Let async handler run

        assert result is True
        assert len(received_messages) == 1
        assert received_messages[0].content["task"] == "analyze"

    @pytest.mark.asyncio
    async def test_unsubscribe(self, message_bus, research_agent):
        """Should stop delivering after unsubscribe."""
        await message_bus.registry.register(research_agent)

        received = []
        async def handler(msg):
            received.append(msg)
            return None

        message_bus.subscribe(research_agent.id, handler)
        message_bus.unsubscribe(research_agent.id)

        msg = create_agent_message(
            from_agent="sender",
            to_agent=research_agent.id,
            msg_type=MessageType.REQUEST,
            content={}
        )

        # Send fails because no handler
        result = await message_bus.send(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_request_response(self, message_bus, research_agent):
        """Should support request/response pattern."""
        await message_bus.registry.register(research_agent)

        async def handler(msg: AgentMessage):
            # Simulate processing and responding
            return msg.create_response(content={"answer": "42"}, from_agent=research_agent.id)

        message_bus.subscribe(research_agent.id, handler)

        request = create_agent_message(
            from_agent="requester",
            to_agent=research_agent.id,
            msg_type=MessageType.REQUEST,
            content={"question": "What is the answer?"}
        )

        response = await message_bus.request(request, timeout=5.0)

        assert response is not None
        assert response.type == MessageType.RESPONSE
        assert response.content["answer"] == "42"
        assert response.correlation_id == request.id

    @pytest.mark.asyncio
    async def test_request_timeout(self, message_bus, research_agent):
        """Should timeout if no response."""
        await message_bus.registry.register(research_agent)

        # Handler that doesn't respond
        async def handler(msg):
            return None

        message_bus.subscribe(research_agent.id, handler)

        request = create_agent_message(
            from_agent="requester",
            to_agent=research_agent.id,
            msg_type=MessageType.REQUEST,
            content={}
        )

        response = await message_bus.request(request, timeout=0.1)
        assert response is None

    @pytest.mark.asyncio
    async def test_route_by_capability(self, message_bus, research_agent):
        """Should route to agent by capability."""
        await message_bus.registry.register(research_agent)

        async def handler(msg: AgentMessage):
            return msg.create_response(content={"researched": True}, from_agent=research_agent.id)

        message_bus.subscribe(research_agent.id, handler)

        request = create_agent_message(
            from_agent="requester",
            to_agent="",  # Will be filled by routing
            msg_type=MessageType.REQUEST,
            content={"task": "research"}
        )

        response = await message_bus.route_by_capability(
            request,
            AgentCapability.RESEARCH,
            strategy="balanced"
        )

        assert response is not None
        assert response.content["researched"] is True


# ============================================================================
# SharedMemory Tests
# ============================================================================

class TestSharedMemory:
    """Tests for SharedMemory."""

    @pytest.mark.asyncio
    async def test_put_and_get(self, shared_memory):
        """Should store and retrieve values."""
        await shared_memory.put("key1", {"data": "value"}, owner="agent-a")

        result = await shared_memory.get("key1")
        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, shared_memory):
        """Should return None for nonexistent key."""
        result = await shared_memory.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, shared_memory):
        """Should isolate keys by namespace."""
        await shared_memory.put("key1", "value_a", owner="a", namespace="ns_a")
        await shared_memory.put("key1", "value_b", owner="b", namespace="ns_b")

        result_a = await shared_memory.get("key1", namespace="ns_a")
        result_b = await shared_memory.get("key1", namespace="ns_b")

        assert result_a == "value_a"
        assert result_b == "value_b"

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, shared_memory):
        """Should expire entries after TTL."""
        await shared_memory.put("key1", "value", owner="a", ttl_seconds=1)

        # Should exist initially
        result = await shared_memory.get("key1")
        assert result == "value"

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired
        result = await shared_memory.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, shared_memory):
        """Should delete entries."""
        await shared_memory.put("key1", "value", owner="a")
        assert await shared_memory.get("key1") == "value"

        result = await shared_memory.delete("key1")
        assert result is True
        assert await shared_memory.get("key1") is None

    @pytest.mark.asyncio
    async def test_get_namespace(self, shared_memory):
        """Should get all entries in namespace."""
        await shared_memory.put("key1", "v1", owner="a", namespace="test")
        await shared_memory.put("key2", "v2", owner="a", namespace="test")
        await shared_memory.put("key3", "v3", owner="a", namespace="other")

        entries = await shared_memory.get_namespace("test")
        assert len(entries) == 2
        assert "key1" in entries
        assert "key2" in entries
        assert entries["key1"] == "v1"
        assert entries["key2"] == "v2"

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, shared_memory):
        """Should cleanup expired entries."""
        await shared_memory.put("key1", "v1", owner="a", namespace="test", ttl_seconds=1)
        await shared_memory.put("key2", "v2", owner="a", namespace="test")  # No TTL

        await asyncio.sleep(1.5)

        count = await shared_memory.cleanup_expired()
        assert count == 1

        # key1 should be gone
        assert await shared_memory.get("key1", namespace="test") is None
        # key2 should still exist
        assert await shared_memory.get("key2", namespace="test") == "v2"

    @pytest.mark.asyncio
    async def test_statistics(self, shared_memory):
        """Should return statistics."""
        await shared_memory.put("key1", "v1", owner="agent-a")
        await shared_memory.put("key2", "v2", owner="agent-b")

        stats = shared_memory.get_statistics()
        assert stats["total_entries"] == 2
        assert "agent-a" in stats["entries_by_owner"]
        assert "agent-b" in stats["entries_by_owner"]


# ============================================================================
# MultiAgentCoordinator Tests
# ============================================================================

class TestMultiAgentCoordinator:
    """Tests for MultiAgentCoordinator."""

    @pytest.mark.asyncio
    async def test_register_agent(self, coordinator, research_agent):
        """Should register agent with handler."""
        received = []

        async def handler(msg):
            received.append(msg)
            return None

        result = await coordinator.register_agent(research_agent, handler)
        assert result is True

        # Agent should be in registry
        agent = coordinator.registry.get(research_agent.id)
        assert agent is not None

    @pytest.mark.asyncio
    async def test_delegate_task(self, coordinator, research_agent):
        """Should delegate task to capable agent."""
        async def handler(msg: AgentMessage):
            return msg.create_response(content={"result": "researched"}, from_agent=research_agent.id)

        await coordinator.register_agent(research_agent, handler)

        response = await coordinator.delegate_task(
            task={"query": "What is Thompson Sampling?"},
            capability=AgentCapability.RESEARCH,
            from_agent="coordinator",
            timeout=5.0
        )

        assert response is not None
        assert response.content["result"] == "researched"

    @pytest.mark.asyncio
    async def test_delegate_no_capable_agent(self, coordinator):
        """Should return None if no capable agent."""
        response = await coordinator.delegate_task(
            task={"query": "test"},
            capability=AgentCapability.RESEARCH,
            from_agent="coordinator"
        )

        assert response is None

    @pytest.mark.asyncio
    async def test_broadcast_context(self, coordinator, research_agent, synthesis_agent):
        """Should broadcast context to shared memory."""
        async def handler(msg):
            return None

        await coordinator.register_agent(research_agent, handler)
        await coordinator.register_agent(synthesis_agent, handler)

        await coordinator.broadcast_context(
            key="research_results",
            value={"findings": ["A", "B", "C"]},
            owner="research-001",
            namespace="session-123"
        )

        # Context should be in shared memory
        result = await coordinator.memory.get(
            "research_results",
            namespace="session-123"
        )
        assert result == {"findings": ["A", "B", "C"]}

    @pytest.mark.asyncio
    async def test_get_statistics(self, coordinator, research_agent):
        """Should return comprehensive statistics."""
        async def handler(msg: AgentMessage):
            return msg.create_response(content={"done": True}, from_agent=research_agent.id)

        await coordinator.register_agent(research_agent, handler)

        stats = coordinator.get_statistics()

        assert "registry" in stats
        assert "memory" in stats
        assert "bus" in stats


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_coordinator(self):
        """Should create coordinator with defaults."""
        coord = create_coordinator()
        assert coord.registry is not None
        assert coord.bus is not None
        assert coord.memory is not None

    def test_create_agent_message(self):
        """Should create message with convenience function."""
        msg = create_agent_message(
            from_agent="a",
            to_agent="b",
            msg_type=MessageType.REQUEST,
            content={"task": "test"}
        )

        assert msg.from_agent == "a"
        assert msg.to_agent == "b"
        assert msg.type == MessageType.REQUEST

    def test_create_research_agent(self):
        """Should create research agent."""
        agent = create_research_agent()
        assert agent.id.startswith("research_")
        assert AgentCapability.RESEARCH in agent.capabilities

    def test_create_synthesis_agent(self):
        """Should create synthesis agent."""
        agent = create_synthesis_agent()
        assert AgentCapability.SYNTHESIS in agent.capabilities

    def test_create_code_agent(self):
        """Should create code agent."""
        agent = create_code_agent()
        assert AgentCapability.CODE_REVIEW in agent.capabilities


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full workflows."""

    @pytest.mark.asyncio
    async def test_multi_agent_research_workflow(self, coordinator):
        """Test complete multi-agent research workflow."""
        # Create specialized agents
        research_agent = AgentInfo(
            id="research",
            name="Researcher",
            capabilities={AgentCapability.RESEARCH, AgentCapability.ANALYSIS}
        )
        synthesis_agent = AgentInfo(
            id="synthesis",
            name="Synthesizer",
            capabilities={AgentCapability.SUMMARIZATION, AgentCapability.SYNTHESIS}
        )

        research_findings = []

        async def research_handler(msg: AgentMessage):
            # Simulate research
            findings = {
                "topic": msg.content.get("query"),
                "findings": ["Finding A", "Finding B", "Finding C"]
            }
            research_findings.append(findings)

            # Store in shared memory
            await coordinator.broadcast_context(
                key="research_findings",
                value=findings,
                owner=research_agent.id,
                namespace="session"
            )

            return msg.create_response(content=findings, from_agent=research_agent.id)

        synthesis_results = []

        async def synthesis_handler(msg: AgentMessage):
            # Get research findings from shared memory
            findings = await coordinator.memory.get(
                "research_findings",
                namespace="session"
            )

            # Synthesize
            result = {
                "summary": f"Synthesized {len(findings.get('findings', []))} findings",
                "context_received": len(msg.context)
            }
            synthesis_results.append(result)

            return msg.create_response(content=result, from_agent=synthesis_agent.id)

        # Register agents
        await coordinator.register_agent(research_agent, research_handler)
        await coordinator.register_agent(synthesis_agent, synthesis_handler)

        # Step 1: Research
        research_response = await coordinator.delegate_task(
            task={"query": "Thompson Sampling"},
            capability=AgentCapability.RESEARCH,
            from_agent="coordinator"
        )

        assert research_response is not None
        assert len(research_response.content["findings"]) == 3

        # Step 2: Synthesis
        synthesis_response = await coordinator.delegate_task(
            task={"action": "synthesize"},
            capability=AgentCapability.SUMMARIZATION,
            from_agent="coordinator"
        )

        assert synthesis_response is not None
        assert "Synthesized 3 findings" in synthesis_response.content["summary"]

    @pytest.mark.asyncio
    async def test_agent_selection_strategy(self, coordinator):
        """Test different agent selection strategies."""
        # Create multiple agents with different metrics
        agent1 = AgentInfo(
            id="r1",
            name="Research 1",
            capabilities={AgentCapability.RESEARCH}
        )
        agent2 = AgentInfo(
            id="r2",
            name="Research 2",
            capabilities={AgentCapability.RESEARCH}
        )

        # Agent1: slow but reliable
        agent1.update_metrics(response_time_ms=200.0, success=True)
        agent1.update_metrics(response_time_ms=200.0, success=True)  # 100% success

        # Agent2: fast but less reliable
        agent2.update_metrics(response_time_ms=50.0, success=True)
        agent2.update_metrics(response_time_ms=50.0, success=False)  # 50% success

        async def handler(msg: AgentMessage):
            return msg.create_response(content={"agent": msg.to_agent}, from_agent=msg.to_agent)

        await coordinator.register_agent(agent1, handler)
        await coordinator.register_agent(agent2, handler)

        # Test reliable strategy (should prefer agent1)
        response = await coordinator.delegate_task(
            task={"query": "test"},
            capability=AgentCapability.RESEARCH,
            strategy="reliable"
        )
        assert response.content["agent"] == "r1"

        # Test fastest strategy (should prefer agent2)
        response = await coordinator.delegate_task(
            task={"query": "test"},
            capability=AgentCapability.RESEARCH,
            strategy="fastest"
        )
        assert response.content["agent"] == "r2"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
