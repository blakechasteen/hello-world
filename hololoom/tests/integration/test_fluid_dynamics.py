"""
Integration Tests for Fluid Dynamics (Phase 2)

Tests the complete fluid dynamics system:
    - PressureField
    - VelocityField
    - ContextFlowEngine
    - AdaptivePacker
"""

import pytest
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hololoom.physics import (
    PressureField,
    VelocityField,
    ContextFlowEngine,
    AdaptivePacker,
    PressureSource
)


class TestPressureField:
    """Test PressureField functionality."""

    def test_set_and_get_pressure(self):
        """Test basic pressure operations."""
        field = PressureField()

        field.set_pressure("node1", 0.8)
        assert field.get_pressure("node1") == 0.8
        assert field.get_pressure("node2") == 0.0  # Default

    def test_inject_pressure(self):
        """Test pressure injection."""
        field = PressureField()

        field.inject_pressure("node1", importance=0.9, tokens=50, text="test")
        assert field.get_pressure("node1") == 0.9
        assert "node1" in field.sources
        assert field.sources["node1"].tokens == 50

    def test_compute_gradient(self):
        """Test gradient computation."""
        field = PressureField()

        field.set_pressure("A", 0.9)
        field.set_pressure("B", 0.3)
        field.set_pressure("C", 0.6)

        gradients = field.compute_gradient("A", ["B", "C"])

        # Gradient = neighbor_pressure - node_pressure
        assert gradients["B"] == pytest.approx(0.3 - 0.9)  # -0.6 (flow away)
        assert gradients["C"] == pytest.approx(0.6 - 0.9)  # -0.3 (flow away)

    def test_detect_sparse_regions(self):
        """Test sparse region detection."""
        field = PressureField()

        field.set_pressure("A", 0.9)
        field.set_pressure("B", 0.2)  # Sparse!
        field.set_pressure("C", 0.1)  # Sparse!

        sparse = field.detect_sparse_regions(threshold=0.3)
        assert "B" in sparse
        assert "C" in sparse
        assert "A" not in sparse


class TestVelocityField:
    """Test VelocityField functionality."""

    def test_set_and_get_velocity(self):
        """Test basic velocity operations."""
        field = VelocityField()

        field.set_velocity("A", "B", magnitude=0.5)
        assert field.get_velocity("A", "B") == 0.5
        assert field.get_velocity("B", "C") == 0.0  # Default

    def test_update_from_pressure_gradient(self):
        """Test velocity update from gradient."""
        field = VelocityField(viscosity=0.01)

        # Initial velocity = 0
        field.set_velocity("A", "B", 0.0)

        # Pressure gradient (high -> low = positive gradient)
        gradient = 0.5
        field.update_from_pressure_gradient("A", "B", gradient, dt=0.01)

        # Velocity should increase (flow from high to low)
        v = field.get_velocity("A", "B")
        assert v < 0  # Negative because we negate gradient

    def test_get_net_flow(self):
        """Test net flow calculation."""
        field = VelocityField()

        field.set_velocity("A", "B", 0.3)  # Flow from A to B (inflow to B)
        field.set_velocity("C", "B", 0.5)  # Flow from C to B (inflow to B)

        # Net flow = inflow - outflow
        # Both are inflows to B, no outflows
        net = field.get_net_flow("B")
        assert net == pytest.approx(0.8)  # 0.3 + 0.5 = 0.8 inflow


class TestContextFlowEngine:
    """Test ContextFlowEngine functionality."""

    def test_inject_and_step(self):
        """Test basic injection and propagation."""
        engine = ContextFlowEngine(viscosity=0.01)

        # Build simple graph
        engine.add_edge("A", "B")
        engine.add_edge("B", "C")

        # Inject high pressure
        engine.inject_context("A", importance=0.9, tokens=50)

        assert engine.tokens_used == 50
        assert engine.pressure.get_pressure("A") == 0.9

        # Propagate
        for _ in range(10):
            engine.step(dt=0.01)

        # Pressure should still be high at A
        assert engine.pressure.get_pressure("A") > 0.8

    def test_detect_sparse_regions(self):
        """Test sparse region detection."""
        engine = ContextFlowEngine(sparse_threshold=0.3)

        engine.pressure.set_pressure("A", 0.9)
        engine.pressure.set_pressure("B", 0.2)  # Sparse

        sparse = engine.detect_sparse_regions()
        assert "B" in sparse

    def test_generate_reverse_prompt(self):
        """Test reverse prompt generation."""
        engine = ContextFlowEngine()

        # Build graph
        engine.add_edge("A", "B")
        engine.pressure.set_pressure("A", 0.9)  # High
        engine.pressure.set_pressure("B", 0.2)  # Low (sparse)

        # Generate reverse prompt for sparse node
        prompt = engine.generate_reverse_prompt("B")
        assert "A" in prompt or "B" in prompt

    def test_extract_context(self):
        """Test context extraction."""
        engine = ContextFlowEngine()

        # Inject multiple contexts
        engine.inject_context("A", 0.9, 100, "Context A")
        engine.inject_context("B", 0.7, 100, "Context B")
        engine.inject_context("C", 0.5, 100, "Context C")

        # Extract with token limit
        packed = engine.extract_context(max_tokens=250)

        # Should pack top 2 (A and B)
        assert len(packed) == 2
        assert packed[0][0] == "A"  # Highest pressure first
        assert packed[1][0] == "B"


class TestAdaptivePacker:
    """Test AdaptivePacker functionality."""

    def test_pack_sync(self):
        """Test synchronous packing."""
        packer = AdaptivePacker(max_tokens=1000, viscosity=0.01)

        # Build graph
        packer.add_edge("A", "B")
        packer.add_edge("B", "C")

        # Inject query
        packer.inject("A", importance=0.95, tokens=100, text="Query")

        # Pack
        result = packer.pack_sync(max_iterations=10)

        assert result.tokens_used == 100
        assert result.tokens_available == 1000
        assert len(result.nodes) > 0
        assert len(result.flow_states) == 10

    @pytest.mark.asyncio
    async def test_pack_async(self):
        """Test async packing with reverse prompting."""

        # Mock reverse prompt callback
        async def mock_callback(prompt):
            return (50, f"Answer to: {prompt}")

        packer = AdaptivePacker(
            max_tokens=1000,
            sparse_threshold=0.4,
            reverse_prompt_callback=mock_callback
        )

        # Build graph with sparse regions
        packer.add_edge("A", "B")
        packer.inject("A", 0.9, 100)

        # Pack with reverse prompting
        result = await packer.pack(
            max_iterations=5,
            enable_reverse_prompting=True
        )

        assert result.tokens_used >= 100
        # Reverse prompts may or may not execute depending on sparse detection

    def test_visualize_pressure_field(self):
        """Test pressure field visualization."""
        packer = AdaptivePacker(max_tokens=1000)

        packer.inject("A", 0.9, 100)
        packer.inject("B", 0.5, 100)

        viz = packer.visualize_pressure_field()
        assert "A" in viz
        assert "B" in viz
        assert "0.90" in viz or "0.9" in viz


class TestIntegration:
    """Full integration tests."""

    def test_end_to_end_packing(self):
        """Test complete end-to-end packing workflow."""
        # Create packer
        packer = AdaptivePacker(max_tokens=8000, viscosity=0.01)

        # Build knowledge graph
        edges = [
            ("ThompsonSampling", "Bayesian"),
            ("ThompsonSampling", "MultiArmedBandit"),
            ("Bayesian", "PriorDistribution"),
            ("MultiArmedBandit", "Regret"),
        ]

        for src, tgt in edges:
            packer.add_edge(src, tgt)

        # Inject query
        packer.inject(
            "ThompsonSampling",
            importance=0.95,
            tokens=100,
            text="What is Thompson Sampling?"
        )

        # Pack
        result = packer.pack_sync(max_iterations=20, dt=0.01)

        # Verify results
        assert result.tokens_used == 100
        assert len(result.nodes) > 0
        assert result.nodes[0][0] == "ThompsonSampling"  # Highest pressure

        # Get summary
        summary = packer.get_flow_summary()
        assert summary["iterations"] == 20
        assert summary["tokens_used"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
