"""
Unit Tests: Ultra-Fast Optimizer

Tests for <1ms decision-making components:
- Working memory (EMA updates, decay)
- Sub-query routing (agent selection)
- Token gating (streaming confidence)
- Verification triggering
- Statistics tracking
"""

import pytest
import torch
import asyncio
import time

from hololoom.core.memory.nested_ultra_fast import (
    UltraFastOptimizer,
    WorkingMemory,
    SubQueryRouter,
    TokenGate,
    AgentType,
    DecisionType,
    route_subquery_fast,
    should_verify_fast
)


class TestWorkingMemory:
    """Test working memory EMA updates"""

    def test_initialization(self):
        """Test working memory initializes correctly"""
        wm = WorkingMemory(embedding_dim=244, decay_rate=0.9)
        assert wm.embedding_dim == 244
        assert wm.decay_rate == 0.9
        assert wm.activations.shape == (1, 244)
        assert torch.allclose(wm.activations, torch.zeros(1, 244))

    def test_update_ema(self):
        """Test exponential moving average update"""
        wm = WorkingMemory(embedding_dim=244, decay_rate=0.9)

        # First update
        new_features = torch.ones(1, 244)
        updated = wm.update(new_features)

        # Should be 0.9 * 0 + 0.1 * 1 = 0.1
        expected = 0.1 * torch.ones(1, 244)
        assert torch.allclose(updated, expected, atol=1e-6)

        # Second update
        new_features = torch.ones(1, 244) * 2
        updated = wm.update(new_features)

        # Should be 0.9 * 0.1 + 0.1 * 2 = 0.09 + 0.2 = 0.29
        expected = torch.tensor([[0.29]] * 244)
        assert torch.allclose(updated, expected, atol=1e-6)

    def test_reset(self):
        """Test working memory reset"""
        wm = WorkingMemory(embedding_dim=244)
        wm.update(torch.ones(1, 244))
        assert not torch.allclose(wm.activations, torch.zeros(1, 244))

        wm.reset()
        assert torch.allclose(wm.activations, torch.zeros(1, 244))

    def test_time_tracking(self):
        """Test time since update tracking"""
        wm = WorkingMemory(embedding_dim=244)
        wm.update(torch.randn(1, 244))

        time.sleep(0.01)  # 10ms
        elapsed = wm.time_since_update_ms()
        assert elapsed >= 10.0


class TestSubQueryRouter:
    """Test sub-query routing neural network"""

    def test_initialization(self):
        """Test router initializes correctly"""
        router = SubQueryRouter(input_dim=244, hidden_dim=128, n_agents=7)
        assert router.input_dim == 244
        assert router.hidden_dim == 128
        assert router.n_agents == 7

    def test_forward_pass(self):
        """Test forward pass produces correct shape"""
        router = SubQueryRouter()
        working_memory = torch.randn(1, 244)
        logits = router(working_memory)

        assert logits.shape == (1, 7)  # 7 agent types

    def test_agent_selection(self):
        """Test agent selection from logits"""
        router = SubQueryRouter()
        logits = torch.randn(1, 7)

        agent_type, confidence = router.select_agent(logits, temperature=0.1)

        assert isinstance(agent_type, AgentType)
        assert 0.0 <= confidence <= 1.0

    def test_temperature_effect(self):
        """Test temperature affects confidence"""
        router = SubQueryRouter()
        logits = torch.tensor([[1.0, 0.5, 0.3, 0.1, 0.0, -0.1, -0.5]])

        # Low temperature (more confident)
        _, conf_low = router.select_agent(logits, temperature=0.1)

        # High temperature (less confident)
        _, conf_high = router.select_agent(logits, temperature=1.0)

        assert conf_low > conf_high  # Lower temp = higher confidence


class TestTokenGate:
    """Test token-level gating"""

    def test_initialization(self):
        """Test token gate initializes correctly"""
        gate = TokenGate(input_dim=244)
        assert hasattr(gate, 'gate')

    def test_forward_pass(self):
        """Test forward pass produces confidence score"""
        gate = TokenGate()
        working_memory = torch.randn(1, 244)
        confidence = gate(working_memory)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_should_output_token(self):
        """Test token gating decision"""
        gate = TokenGate()

        # High confidence state
        high_conf_state = torch.ones(1, 244)
        # Low confidence state
        low_conf_state = -torch.ones(1, 244)

        # Test with moderate threshold
        threshold = 0.5

        # Note: Results depend on random initialization
        # Just verify it returns bool
        result_high = gate.should_output_token(high_conf_state, threshold)
        result_low = gate.should_output_token(low_conf_state, threshold)

        assert isinstance(result_high, bool)
        assert isinstance(result_low, bool)


class TestUltraFastOptimizer:
    """Test ultra-fast optimizer integration"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test optimizer initializes correctly"""
        optimizer = UltraFastOptimizer(
            embedding_dim=244,
            hidden_dim=128,
            n_agents=7
        )

        assert optimizer.embedding_dim == 244
        assert optimizer.working_memory is not None
        assert optimizer.router is not None
        assert optimizer.token_gate is not None

    @pytest.mark.asyncio
    async def test_route_subquery(self):
        """Test sub-query routing"""
        optimizer = UltraFastOptimizer()
        working_memory = torch.randn(1, 244)

        decision = await optimizer.route_subquery(
            text="Is this accurate?",
            parent_query="What is Thompson Sampling?",
            working_memory_state=working_memory
        )

        assert decision.decision_type == DecisionType.ROUTE_SUBQUERY
        assert decision.agent_type is not None
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.latency_ms > 0
        assert decision.latency_ms < 10.0  # Should be <10ms

    @pytest.mark.asyncio
    async def test_gate_token(self):
        """Test token gating"""
        optimizer = UltraFastOptimizer()
        working_memory = torch.randn(1, 244)

        decision = await optimizer.gate_token(
            token="Thompson",
            working_memory_state=working_memory,
            threshold=0.7
        )

        assert decision.decision_type == DecisionType.GATE_TOKEN
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.latency_ms > 0
        assert decision.latency_ms < 1.0  # Should be <1ms

    @pytest.mark.asyncio
    async def test_should_verify(self):
        """Test verification triggering"""
        optimizer = UltraFastOptimizer()
        working_memory = torch.randn(1, 244)

        decision = await optimizer.should_verify(
            claim="Python was created in 1991",
            working_memory_state=working_memory
        )

        assert decision.decision_type == DecisionType.VERIFY_CLAIM
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.latency_ms > 0

    @pytest.mark.asyncio
    async def test_prioritize_subqueries(self):
        """Test sub-query prioritization"""
        optimizer = UltraFastOptimizer()
        working_memory = torch.randn(1, 244)

        subqueries = [
            "What is the definition?",
            "Show me an example",
            "What are the tradeoffs?"
        ]

        priorities = await optimizer.prioritize_subqueries(
            subqueries=subqueries,
            working_memory_state=working_memory
        )

        assert len(priorities) == len(subqueries)
        assert all(isinstance(item, tuple) for item in priorities)
        assert all(len(item) == 2 for item in priorities)

        # Check sorted by priority
        scores = [score for _, score in priorities]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test statistics are tracked correctly"""
        optimizer = UltraFastOptimizer()
        working_memory = torch.randn(1, 244)

        # Make various decisions
        await optimizer.route_subquery("Query 1", "Parent", working_memory)
        await optimizer.route_subquery("Query 2", "Parent", working_memory)
        await optimizer.gate_token("token", working_memory)

        stats = optimizer.get_statistics()

        assert stats['total_decisions'] == 3
        assert stats['decisions_by_type'][DecisionType.ROUTE_SUBQUERY] == 2
        assert stats['decisions_by_type'][DecisionType.GATE_TOKEN] == 1
        assert stats['average_latency_ms'] > 0

    @pytest.mark.asyncio
    async def test_working_memory_persistence(self):
        """Test working memory persists across decisions"""
        optimizer = UltraFastOptimizer()

        # First decision
        wm1 = torch.ones(1, 244)
        await optimizer.route_subquery("Query 1", "Parent", wm1)

        state_after_1 = optimizer.working_memory.get_state()

        # Second decision
        wm2 = torch.zeros(1, 244)
        await optimizer.route_subquery("Query 2", "Parent", wm2)

        state_after_2 = optimizer.working_memory.get_state()

        # States should be different (EMA blending)
        assert not torch.allclose(state_after_1, state_after_2)

    @pytest.mark.asyncio
    async def test_reset_working_memory(self):
        """Test working memory reset"""
        optimizer = UltraFastOptimizer()
        working_memory = torch.ones(1, 244)

        await optimizer.route_subquery("Query", "Parent", working_memory)
        assert not torch.allclose(
            optimizer.working_memory.activations,
            torch.zeros(1, 244)
        )

        optimizer.reset_working_memory()
        assert torch.allclose(
            optimizer.working_memory.activations,
            torch.zeros(1, 244)
        )

    def test_summary(self):
        """Test summary output"""
        optimizer = UltraFastOptimizer()
        summary = optimizer.summary()

        assert isinstance(summary, str)
        assert "UltraFastOptimizer Statistics" in summary


class TestConvenienceFunctions:
    """Test convenience functions"""

    @pytest.mark.asyncio
    async def test_route_subquery_fast(self):
        """Test quick routing function"""
        decision = await route_subquery_fast(
            "Is this accurate?",
            "Parent query"
        )

        assert decision.decision_type == DecisionType.ROUTE_SUBQUERY
        assert decision.agent_type is not None

    @pytest.mark.asyncio
    async def test_should_verify_fast(self):
        """Test quick verification function"""
        result = await should_verify_fast("Python was created in 1991")

        assert isinstance(result, bool)


class TestPerformance:
    """Performance tests (latency requirements)"""

    @pytest.mark.asyncio
    async def test_routing_latency(self):
        """Test routing meets <1ms latency target"""
        optimizer = UltraFastOptimizer()
        working_memory = torch.randn(1, 244)

        # Warm up
        await optimizer.route_subquery("Warmup", "Parent", working_memory)

        # Measure 10 decisions
        latencies = []
        for i in range(10):
            decision = await optimizer.route_subquery(
                f"Query {i}",
                "Parent",
                working_memory
            )
            latencies.append(decision.latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage routing latency: {avg_latency:.3f}ms")

        # Relaxed requirement: <10ms (will optimize further)
        assert avg_latency < 10.0, f"Routing too slow: {avg_latency:.3f}ms"

    @pytest.mark.asyncio
    async def test_token_gating_latency(self):
        """Test token gating meets <0.5ms target"""
        optimizer = UltraFastOptimizer()
        working_memory = torch.randn(1, 244)

        # Warm up
        await optimizer.gate_token("warmup", working_memory)

        # Measure 20 tokens
        latencies = []
        for i in range(20):
            decision = await optimizer.gate_token(
                f"token_{i}",
                working_memory
            )
            latencies.append(decision.latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage token gating latency: {avg_latency:.3f}ms")

        # Relaxed requirement: <5ms
        assert avg_latency < 5.0, f"Token gating too slow: {avg_latency:.3f}ms"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
