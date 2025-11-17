"""
Multi-Agent System Tests
=========================

Comprehensive test suite for multi-agent collaboration.

**Date**: November 17, 2025

Run with:
    pytest HoloLoom/multi_agent/tests/test_multi_agent.py -v
"""

import pytest
import asyncio
import networkx as nx
from datetime import datetime, timedelta

from HoloLoom.multi_agent.communication import (
    AgentMessage,
    AgentCommunicator,
    DirectTransport,
    MessageType,
    get_registry
)

from HoloLoom.multi_agent.consensus import (
    GraphOperation,
    YarnGraphConsensus,
    ConflictResolver,
    OperationType
)

from HoloLoom.multi_agent.delegation import (
    QueryClassifier,
    TaskRouter,
    AgentCapability,
    Domain
)

from HoloLoom.multi_agent.specialized_agents import (
    BaseAgent,
    MathAgent,
    CodeAgent,
    WritingAgent,
    ResearchAgent,
    create_agent
)

from HoloLoom.Documentation.types import Query
from HoloLoom.memory.graph import KGEdge


# ============================================================================
# Communication Tests
# ============================================================================

class TestCommunication:
    """Test agent-to-agent communication."""

    @pytest.mark.asyncio
    async def test_direct_transport_send_receive(self):
        """Test direct transport message passing."""
        comm1 = AgentCommunicator("agent-1", transport="direct")
        comm2 = AgentCommunicator("agent-2", transport="direct")

        await comm1.start()
        await comm2.start()

        try:
            # Send message from agent-1 to agent-2
            success = await comm1.send_message(
                receiver_id="agent-2",
                message_type=MessageType.QUERY.value,
                payload={"test": "data"}
            )

            assert success, "Message send should succeed"

            # Receive message at agent-2
            messages = await comm2.receive_messages(timeout=1.0)

            assert len(messages) == 1, "Should receive one message"
            assert messages[0].sender_id == "agent-1"
            assert messages[0].payload["test"] == "data"

        finally:
            await comm1.stop()
            await comm2.stop()

    @pytest.mark.asyncio
    async def test_request_response_pattern(self):
        """Test request-response messaging."""
        comm1 = AgentCommunicator("agent-1", transport="direct")
        comm2 = AgentCommunicator("agent-2", transport="direct")

        await comm1.start()
        await comm2.start()

        try:
            # Background task to handle request and send response
            async def responder():
                messages = await comm2.receive_messages(timeout=2.0)
                if messages:
                    request = messages[0]
                    response = AgentMessage(
                        sender_id="agent-2",
                        receiver_id="agent-1",
                        message_type=MessageType.RESULT.value,
                        payload={"result": "processed"},
                        correlation_id=request.correlation_id
                    )
                    await comm2.transport.send(response)

            responder_task = asyncio.create_task(responder())

            # Send request and wait for response
            response = await comm1.request_response(
                receiver_id="agent-2",
                message_type=MessageType.QUERY.value,
                payload={"query": "test"},
                timeout=3.0
            )

            await responder_task

            assert response is not None, "Should receive response"
            assert response.payload["result"] == "processed"

        finally:
            await comm1.stop()
            await comm2.stop()

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcast messaging."""
        comm1 = AgentCommunicator("agent-1", transport="direct")
        comm2 = AgentCommunicator("agent-2", transport="direct")
        comm3 = AgentCommunicator("agent-3", transport="direct")

        await comm1.start()
        await comm2.start()
        await comm3.start()

        try:
            # Broadcast from agent-1
            success = await comm1.broadcast(
                message_type=MessageType.HEARTBEAT.value,
                payload={"status": "alive"}
            )

            assert success, "Broadcast should succeed"

            # Both other agents should receive
            messages_2 = await comm2.receive_messages(timeout=1.0)
            messages_3 = await comm3.receive_messages(timeout=1.0)

            assert len(messages_2) == 1, "Agent-2 should receive broadcast"
            assert len(messages_3) == 1, "Agent-3 should receive broadcast"

        finally:
            await comm1.stop()
            await comm2.stop()
            await comm3.stop()


# ============================================================================
# Consensus Tests
# ============================================================================

class TestConsensus:
    """Test Yarn Graph consensus and conflict resolution."""

    def test_conflict_resolver_lww(self):
        """Test Last-Write-Wins conflict resolution."""
        resolver = ConflictResolver(strategy="lww")

        now = datetime.now()
        earlier = now - timedelta(seconds=10)

        edge1 = KGEdge("python", "language", "IS_A", 0.9)
        edge2 = KGEdge("python", "language", "IS_A", 0.95)

        op_old = GraphOperation(
            op_type=OperationType.UPDATE_WEIGHT.value,
            agent_id="agent-1",
            timestamp=earlier,
            lamport_clock=5,
            edge=edge1
        )

        op_new = GraphOperation(
            op_type=OperationType.UPDATE_WEIGHT.value,
            agent_id="agent-2",
            timestamp=now,
            lamport_clock=10,
            edge=edge2
        )

        winner, reason = resolver.resolve(op_old, op_new)

        assert winner == op_new, "Newer operation should win"
        assert "newer" in reason.lower()

    def test_conflict_resolver_hww(self):
        """Test Highest-Weight-Wins conflict resolution."""
        resolver = ConflictResolver(strategy="hww")

        now = datetime.now()

        edge_high = KGEdge("python", "language", "IS_A", 0.99)
        edge_low = KGEdge("python", "language", "IS_A", 0.7)

        op_high = GraphOperation(
            op_type=OperationType.UPDATE_WEIGHT.value,
            agent_id="expert",
            timestamp=now,
            lamport_clock=5,
            edge=edge_high
        )

        op_low = GraphOperation(
            op_type=OperationType.UPDATE_WEIGHT.value,
            agent_id="novice",
            timestamp=now,
            lamport_clock=6,
            edge=edge_low
        )

        winner, reason = resolver.resolve(op_low, op_high)

        assert winner == op_high, "Higher weight should win"
        assert winner.edge.weight == 0.99

    @pytest.mark.asyncio
    async def test_graph_merge(self):
        """Test CRDT-based graph merging."""
        graph1 = nx.MultiDiGraph()
        graph2 = nx.MultiDiGraph()

        # Graph 1: Math knowledge
        graph1.add_edge("calculus", "math", key="IS_A", weight=1.0)
        graph1.add_edge("derivative", "calculus", key="PART_OF", weight=0.9)

        # Graph 2: Code knowledge + conflicting edge
        graph2.add_edge("python", "programming", key="IS_A", weight=1.0)
        graph2.add_edge("calculus", "math", key="IS_A", weight=0.8)  # Conflict

        consensus = YarnGraphConsensus(
            agent_id="agent-1",
            local_graph=graph2,
            strategy="lww"
        )

        result = await consensus.merge_graph(graph1, "agent-2")

        # Should add new edges
        assert result.edges_added > 0, "Should add edges from remote graph"

        # Should detect conflict
        assert result.conflicts_detected > 0, "Should detect conflicting edge"

        # Final graph should have edges from both
        assert graph2.number_of_edges() >= 3, "Should have edges from both graphs"

    @pytest.mark.asyncio
    async def test_operation_application(self):
        """Test applying graph operations."""
        graph = nx.MultiDiGraph()

        consensus = YarnGraphConsensus(
            agent_id="agent-1",
            local_graph=graph,
            strategy="lww"
        )

        # Create add_edge operation
        edge = KGEdge("python", "programming", "IS_A", 1.0)
        op = GraphOperation(
            op_type=OperationType.ADD_EDGE.value,
            agent_id="agent-1",
            timestamp=datetime.now(),
            lamport_clock=1,
            edge=edge
        )

        success = await consensus.apply_operation(op)

        assert success, "Operation should apply successfully"
        assert graph.has_edge("python", "programming", key="IS_A")


# ============================================================================
# Delegation Tests
# ============================================================================

class TestDelegation:
    """Test task delegation and routing."""

    def test_query_classifier(self):
        """Test domain classification."""
        classifier = QueryClassifier()

        # Math query
        math_query = Query(text="Solve the equation x^2 + 5x + 6 = 0")
        assert classifier.classify(math_query) == Domain.MATH

        # Code query
        code_query = Query(text="Write a Python function to sort a list")
        assert classifier.classify(code_query) == Domain.CODE

        # Writing query
        writing_query = Query(text="Explain quantum computing in simple terms")
        assert classifier.classify(writing_query) == Domain.WRITING

        # Research query
        research_query = Query(text="Compare machine learning frameworks")
        assert classifier.classify(research_query) == Domain.RESEARCH

    def test_specialization_scoring(self):
        """Test agent specialization scoring."""
        comm = AgentCommunicator("test", transport="direct")
        router = TaskRouter(comm, strategy="best_match")

        # Create agent capabilities
        math_cap = AgentCapability(
            agent_id="math-01",
            domain=Domain.MATH.value,
            expertise_areas=["algebra", "calculus"]
        )

        code_cap = AgentCapability(
            agent_id="code-01",
            domain=Domain.CODE.value,
            expertise_areas=["python", "javascript"]
        )

        # Math query should score higher for math agent
        math_query = Query(text="Calculate derivative of x^2")

        math_score = router.calculate_specialization_score(math_cap, math_query)
        code_score = router.calculate_specialization_score(code_cap, math_query)

        assert math_score > code_score, "Math agent should score higher for math query"

    @pytest.mark.asyncio
    async def test_task_decomposition(self):
        """Test complex task decomposition."""
        comm = AgentCommunicator("test", transport="direct")
        router = TaskRouter(comm, strategy="best_match")

        # Register some agents
        await router.register_agent(AgentCapability(
            agent_id="math-01",
            domain=Domain.MATH.value,
            expertise_areas=[]
        ))

        await router.register_agent(AgentCapability(
            agent_id="code-01",
            domain=Domain.CODE.value,
            expertise_areas=[]
        ))

        # Complex query
        query = Query(text="Compare Python and JavaScript")

        subtasks = await router.decompose_task(query, max_subtasks=5)

        # Should decompose into multiple subtasks
        assert len(subtasks) >= 2, "Should create multiple subtasks"

        # Should have dependency structure
        has_dependency = any(len(st.dependencies) > 0 for st in subtasks)
        assert has_dependency, "Should have dependent subtasks"


# ============================================================================
# Specialized Agent Tests
# ============================================================================

class TestSpecializedAgents:
    """Test specialized agent implementations."""

    @pytest.mark.asyncio
    async def test_agent_creation(self):
        """Test creating specialized agents."""
        async with MathAgent("math-01", transport="direct") as agent:
            assert agent.agent_id == "math-01"
            assert agent.domain == Domain.MATH.value
            assert len(agent.expertise_areas) > 0

    @pytest.mark.asyncio
    async def test_agent_factory(self):
        """Test agent factory function."""
        math_agent = create_agent("math")
        code_agent = create_agent("code")
        writing_agent = create_agent("writing")
        research_agent = create_agent("research")

        assert isinstance(math_agent, MathAgent)
        assert isinstance(code_agent, CodeAgent)
        assert isinstance(writing_agent, WritingAgent)
        assert isinstance(research_agent, ResearchAgent)

    @pytest.mark.asyncio
    async def test_agent_knowledge_initialization(self):
        """Test domain-specific knowledge initialization."""
        async with MathAgent("math-01", transport="direct") as agent:
            # Should have math concepts in graph
            assert agent.graph.number_of_nodes() > 0
            assert agent.graph.number_of_edges() > 0

    @pytest.mark.asyncio
    async def test_agent_message_handling(self):
        """Test agent message processing."""
        async with MathAgent("math-01", transport="direct") as math_agent, \
                   CodeAgent("code-01", transport="direct") as code_agent:

            # Wait for agents to start
            await asyncio.sleep(0.1)

            # Send query from math to code
            response = await math_agent.communicator.request_response(
                receiver_id="code-01",
                message_type=MessageType.QUERY.value,
                payload={"query": {"text": "Test query"}},
                timeout=2.0
            )

            # Should receive response
            assert response is not None
            assert response.sender_id == "code-01"

    @pytest.mark.asyncio
    async def test_agent_capability_metadata(self):
        """Test agent capability reporting."""
        async with MathAgent("math-01", transport="direct") as agent:
            capability = agent.get_capability()

            assert capability.agent_id == "math-01"
            assert capability.domain == Domain.MATH.value
            assert capability.expertise_score > 0
            assert capability.max_capacity > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self):
        """Test complete multi-agent collaboration workflow."""
        async with MathAgent("math-01", transport="direct") as math_agent, \
                   CodeAgent("code-01", transport="direct") as code_agent:

            # Create router
            router = TaskRouter(math_agent.communicator, strategy="best_match")

            # Register agents
            await router.register_agent(math_agent.get_capability())
            await router.register_agent(code_agent.get_capability())

            # Math agent adds knowledge
            math_agent.graph.add_edge(
                "calculus", "math",
                key="IS_A",
                weight=1.0
            )

            # Sync knowledge
            result = await code_agent.consensus.merge_graph(
                math_agent.graph,
                "math-01"
            )

            # Should merge successfully
            assert result.edges_added > 0 or result.edges_updated > 0

    @pytest.mark.asyncio
    async def test_distributed_consensus(self):
        """Test distributed consensus across multiple agents."""
        agents = [
            MathAgent("math-01", transport="direct"),
            CodeAgent("code-01", transport="direct"),
            WritingAgent("writing-01", transport="direct")
        ]

        async with agents[0] as agent1, agents[1] as agent2, agents[2] as agent3:
            # Each agent adds knowledge
            agent1.graph.add_edge("math", "science", key="IS_A", weight=1.0)
            agent2.graph.add_edge("code", "engineering", key="IS_A", weight=1.0)
            agent3.graph.add_edge("writing", "communication", key="IS_A", weight=1.0)

            # Merge all graphs into agent1
            result2 = await agent1.consensus.merge_graph(agent2.graph, "code-01")
            result3 = await agent1.consensus.merge_graph(agent3.graph, "writing-01")

            # Agent1 should have knowledge from all agents
            assert agent1.graph.number_of_edges() >= 3


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and scalability tests."""

    @pytest.mark.asyncio
    async def test_message_throughput(self):
        """Test message passing throughput."""
        comm1 = AgentCommunicator("agent-1", transport="direct")
        comm2 = AgentCommunicator("agent-2", transport="direct")

        await comm1.start()
        await comm2.start()

        try:
            # Send 100 messages
            for i in range(100):
                await comm1.send_message(
                    receiver_id="agent-2",
                    message_type=MessageType.HEARTBEAT.value,
                    payload={"count": i}
                )

            # Receive all messages
            all_messages = []
            for _ in range(10):  # Try up to 10 batches
                messages = await comm2.receive_messages(timeout=0.1)
                all_messages.extend(messages)
                if len(all_messages) >= 100:
                    break

            assert len(all_messages) == 100, "Should receive all messages"

        finally:
            await comm1.stop()
            await comm2.stop()

    @pytest.mark.asyncio
    async def test_graph_merge_scalability(self):
        """Test graph merging with large graphs."""
        graph1 = nx.MultiDiGraph()
        graph2 = nx.MultiDiGraph()

        # Create graphs with 100 edges each
        for i in range(100):
            graph1.add_edge(f"node_{i}", f"node_{i+1}", key="CONNECTS", weight=1.0)
            graph2.add_edge(f"node_{i+50}", f"node_{i+51}", key="CONNECTS", weight=1.0)

        consensus = YarnGraphConsensus(
            agent_id="agent-1",
            local_graph=graph1,
            strategy="lww"
        )

        import time
        start = time.time()

        result = await consensus.merge_graph(graph2, "agent-2")

        duration = time.time() - start

        # Should complete in reasonable time (<1 second)
        assert duration < 1.0, f"Merge took {duration:.3f}s (should be <1s)"

        # Should have edges from both graphs
        assert graph1.number_of_edges() > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
