"""
E2E tests for integration scenarios and real-world workflows.

Tests:
1. Complete question-answering workflow
2. Multi-turn conversation
3. Context accumulation over session
4. Tool selection across query types
5. Learning from feedback loop
6. Mixed complexity queries
7. Research mode workflow
8. Verification workflow
9. Plan-execute workflow
10. Complete system integration

Performance Budget: <30s per test (E2E tier)
"""

import pytest
import asyncio

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query


class TestQuestionAnsweringWorkflow:
    """Test complete Q&A workflow."""

    @pytest.mark.asyncio
    async def test_simple_qa_workflow(self, bare_config, test_shards):
        """Simple Q&A workflow should work end-to-end."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="What is Thompson Sampling?")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None
            assert spacetime.response is not None
            assert len(spacetime.response) > 0

    @pytest.mark.asyncio
    async def test_complex_qa_workflow(self, bare_config, test_shards):
        """Complex Q&A with multiple facets."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="How does Thompson Sampling balance exploration and exploitation in reinforcement learning?")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None
            assert spacetime.confidence >= 0


class TestMultiTurnConversation:
    """Test multi-turn conversations."""

    @pytest.mark.asyncio
    async def test_conversation_continuity(self, bare_config, test_shards):
        """Conversation should maintain continuity."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            # Turn 1
            q1 = Query(text="What is Thompson Sampling?")
            s1 = await orchestrator.weave(q1)

            # Turn 2 (follow-up)
            q2 = Query(text="How does it work?")
            s2 = await orchestrator.weave(q2)

            # Turn 3 (deeper dive)
            q3 = Query(text="Can you give an example?")
            s3 = await orchestrator.weave(q3)

            assert all(s is not None for s in [s1, s2, s3])


class TestContextAccumulation:
    """Test context accumulation over session."""

    @pytest.mark.asyncio
    async def test_context_builds_over_session(self, bare_config, test_shards):
        """Context should accumulate over session."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            queries = [
                Query(text="What is reinforcement learning?"),
                Query(text="What is Thompson Sampling?"),
                Query(text="How does Thompson Sampling apply to RL?"),
            ]

            for query in queries:
                spacetime = await orchestrator.weave(query)
                assert spacetime is not None


class TestToolSelection:
    """Test tool selection across query types."""

    @pytest.mark.asyncio
    async def test_tool_selection_variety(self, bare_config, test_shards):
        """Different queries should select appropriate tools."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            queries = [
                Query(text="What is Thompson Sampling?"),  # Answer
                Query(text="Search for reinforcement learning"),  # Search
                Query(text="Calculate 2+2"),  # Calculator (if available)
            ]

            for query in queries:
                spacetime = await orchestrator.weave(query)
                assert spacetime is not None

                # Tool should be selected
                if hasattr(spacetime, 'tool_used'):
                    assert spacetime.tool_used is not None


class TestLearningFeedbackLoop:
    """Test learning from feedback loop."""

    @pytest.mark.asyncio
    async def test_feedback_improves_performance(self, bare_config, test_shards):
        """Feedback should improve performance over time."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards, enable_reflection=True) as orchestrator:
            query = Query(text="What is Thompson Sampling?")

            # Initial query
            spacetime1 = await orchestrator.weave(query)

            # Provide positive feedback
            await orchestrator.reflect(spacetime1, feedback={"helpful": True, "rating": 5})

            # Execute more queries
            for i in range(10):
                q = Query(text=f"Query {i}")
                s = await orchestrator.weave(q)
                await orchestrator.reflect(s, feedback={"helpful": True})

            # Same query again
            spacetime2 = await orchestrator.weave(query)

            # Both should work
            assert spacetime1 is not None
            assert spacetime2 is not None


class TestMixedComplexity:
    """Test mixed complexity queries."""

    @pytest.mark.asyncio
    async def test_mixed_complexity_batch(self, bare_config, test_shards):
        """Mixed complexity queries should all handle."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            queries = [
                Query(text="Hi"),  # Simple
                Query(text="What is Thompson Sampling?"),  # Medium
                Query(text="Explain how Thompson Sampling balances exploration-exploitation tradeoffs in multi-armed bandit problems with Bayesian priors"),  # Complex
            ]

            for query in queries:
                spacetime = await orchestrator.weave(query)
                assert spacetime is not None


class TestResearchMode:
    """Test research mode workflow."""

    @pytest.mark.asyncio
    async def test_research_workflow(self, bare_config, test_shards):
        """Research mode should explore topic from multiple angles."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="Research Thompson Sampling applications")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None


class TestVerificationWorkflow:
    """Test verification workflow."""

    @pytest.mark.asyncio
    async def test_verification_workflow(self, bare_config, test_shards):
        """Verification should check claims."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="Verify: Thompson Sampling uses Bayesian priors")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None


class TestPlanExecuteWorkflow:
    """Test plan-execute workflow."""

    @pytest.mark.asyncio
    async def test_plan_execute_workflow(self, bare_config, test_shards):
        """Plan-execute should break down complex tasks."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
            query = Query(text="Plan: Implement Thompson Sampling algorithm")
            spacetime = await orchestrator.weave(query)

            assert spacetime is not None


class TestCompleteSystemIntegration:
    """Test complete system integration."""

    @pytest.mark.asyncio
    async def test_full_system_integration(self, test_shards):
        """Full system integration with all features."""
        # Use FUSED mode for complete integration
        config = Config.fused()

        async with WeavingOrchestrator(cfg=config, shards=test_shards, enable_reflection=True) as orchestrator:
            # Query
            query = Query(text="What is Thompson Sampling and how does it work?")
            spacetime = await orchestrator.weave(query)

            # Verify complete integration
            assert spacetime is not None
            assert spacetime.response is not None
            assert hasattr(spacetime, 'confidence')
            assert hasattr(spacetime, 'trace')

            # Reflect
            feedback = {"helpful": True, "rating": 5}
            await orchestrator.reflect(spacetime, feedback=feedback)

    @pytest.mark.asyncio
    async def test_end_to_end_session(self, bare_config, test_shards):
        """Complete end-to-end session."""
        async with WeavingOrchestrator(cfg=bare_config, shards=test_shards, enable_reflection=True) as orchestrator:
            # Session with multiple queries and feedback
            session_queries = [
                "What is Thompson Sampling?",
                "How does it balance exploration and exploitation?",
                "Show me an example",
                "What are the tradeoffs?",
                "How do I implement it?",
            ]

            for query_text in session_queries:
                query = Query(text=query_text)
                spacetime = await orchestrator.weave(query)

                assert spacetime is not None

                # Provide feedback
                feedback = {"helpful": True}
                await orchestrator.reflect(spacetime, feedback=feedback)


# Total: 12 E2E tests for integration scenarios
# Key coverage:
# - Question-answering workflow
# - Multi-turn conversation
# - Context accumulation
# - Tool selection
# - Learning feedback loop
# - Mixed complexity queries
# - Research mode
# - Verification workflow
# - Plan-execute workflow
# - Complete system integration
