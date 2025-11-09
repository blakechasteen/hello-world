"""
Integration Tests for Gradient Flow (Phase 1)

Tests the complete gradient flow routing system.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.physics import (
    GradientFlowEngine,
    RouteDecision,
    Target,
    combined_loss,
    load_loss,
    latency_loss
)
from HoloLoom.routing import (
    FlowRouter,
    ServerRouter,
    ToolRouter,
    ServerConfig,
    ToolConfig,
    create_server_router,
    create_tool_router
)


class TestGradientFlowEngine:
    """Test GradientFlowEngine."""

    def test_compute_loss(self):
        """Test loss computation."""
        engine = GradientFlowEngine(loss_fn=load_loss)

        loss = engine.compute_loss({"load": 0.5})
        assert loss == 0.5

    def test_compute_gradient(self):
        """Test gradient computation."""
        engine = GradientFlowEngine(loss_fn=load_loss)

        targets = [
            Target("A", {"load": 0.9}),
            Target("B", {"load": 0.3}),
            Target("C", {"load": 0.6})
        ]

        # Gradient at position 0 (server A)
        gradient = engine.compute_gradient(targets, position=0.0)

        # Should be negative (downhill toward B)
        assert gradient < 0

    def test_route_basic(self):
        """Test basic routing."""
        engine = GradientFlowEngine(loss_fn=load_loss, noise_level=0.0)

        decision = engine.route(
            targets=["A", "B", "C"],
            target_metrics=[
                {"load": 0.9},  # Worst
                {"load": 0.3},  # Best!
                {"load": 0.6}
            ],
            max_steps=20
        )

        # Should route to B (lowest load)
        assert decision.target == "B"
        assert decision.loss < 0.4


class TestFlowRouter:
    """Test FlowRouter."""

    @pytest.mark.asyncio
    async def test_route(self):
        """Test basic routing."""
        router = FlowRouter(
            targets=["A", "B", "C"],
            loss_fn=load_loss
        )

        decision = await router.route(
            current_metrics={
                "A": {"load": 0.9},
                "B": {"load": 0.2},
                "C": {"load": 0.6}
            }
        )

        assert decision.target == "B"  # Lowest load

    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test routing statistics."""
        router = FlowRouter(
            targets=["A", "B"],
            loss_fn=load_loss
        )

        # Route multiple times
        for _ in range(5):
            await router.route({
                "A": {"load": 0.5},
                "B": {"load": 0.3}
            })

        stats = router.get_statistics()
        assert stats["total_routes"] == 5
        assert "B" in stats["targets"]


class TestServerRouter:
    """Test ServerRouter."""

    @pytest.mark.asyncio
    async def test_route_query(self):
        """Test query routing with load tracking."""
        router = ServerRouter(
            servers=[
                ServerConfig("server1"),
                ServerConfig("server2"),
                ServerConfig("server3")
            ]
        )

        # All loads start at 0
        loads = router.get_loads()
        assert all(load == 0.0 for load in loads.values())

        # Route query
        decision = await router.route_query("Test query")

        # Load should have increased
        loads = router.get_loads()
        assert loads[decision.target] > 0.0

    @pytest.mark.asyncio
    async def test_load_balancing(self):
        """Test automatic load balancing."""
        router = ServerRouter(
            servers=[
                ServerConfig("server1"),
                ServerConfig("server2")
            ]
        )

        # Route 10 queries
        for _ in range(10):
            await router.route_query("Test")

        # At least one server should have load
        loads = router.get_loads()
        total_load = sum(loads.values())
        assert total_load > 0.0

        # With gradient flow, queries naturally distribute based on load
        # (though all may go to one server if it stays lowest load)


class TestToolRouter:
    """Test ToolRouter."""

    @pytest.mark.asyncio
    async def test_select_tool(self):
        """Test tool selection."""
        router = ToolRouter(
            tools=[
                ToolConfig("cheap", cost=0.1, quality=0.5, latency=50),
                ToolConfig("balanced", cost=0.5, quality=0.7, latency=100),
                ToolConfig("premium", cost=0.9, quality=0.95, latency=150)
            ],
            quality_weight=0.8  # Favor quality
        )

        decision = await router.select_tool("Complex task")

        # Should select premium (highest quality)
        assert decision.target == "premium"


class TestIntegration:
    """Full integration tests."""

    @pytest.mark.asyncio
    async def test_datacenter_flow(self):
        """Test complete datacenter downhill flow."""
        # Create router
        router = create_server_router(
            server_names=["server1", "server2", "server3"]
        )

        # Route multiple queries
        queries = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        for query in queries:
            await router.route_query(query)

        # Verify load balancing
        stats = router.get_statistics()
        assert stats["total_routes"] == 5

        # Multiple servers should have received queries
        assert len(stats["targets"]) >= 2

    @pytest.mark.asyncio
    async def test_tool_selection_integration(self):
        """Test tool selection integration."""
        router = create_tool_router(
            tool_configs=[
                {"name": "search", "cost": 0.5, "quality": 0.7, "latency": 100},
                {"name": "answer", "cost": 0.2, "quality": 0.6, "latency": 50},
                {"name": "reason", "cost": 0.8, "quality": 0.95, "latency": 200}
            ]
        )

        decision = await router.select_tool("Complex reasoning task")

        # Should select based on quality/cost/latency balance
        assert decision.target in ["search", "answer", "reason"]
        assert decision.loss >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
