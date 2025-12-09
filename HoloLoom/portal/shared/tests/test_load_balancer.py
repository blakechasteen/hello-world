"""
Unit tests for Portal Load Balancer.

Tests all 4 strategies and edge cases.
"""

import pytest
from datetime import datetime

from ..load_balancer import (
    LoadBalancer,
    LoadBalanceStrategy,
    SelectionResult,
    LoadBalancerStats,
)
from ..types import NodeRecord, NodeCapabilities, NodeStatus


def mock_node(
    node_id: str,
    cpu_cores: int = 4,
    memory_mb: int = 8192,
    gpu: bool = False,
    current_jobs: int = 0,
) -> NodeRecord:
    """Create a mock node for testing."""
    return NodeRecord(
        node_id=node_id,
        address=f"192.168.1.{node_id[-1]}:9090",
        capabilities=NodeCapabilities(
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            gpu=gpu,
            wasm_runtime="wasmtime",
        ),
        status=NodeStatus.ONLINE,
        last_heartbeat=datetime.utcnow(),
        current_jobs=current_jobs,
    )


class TestRoundRobinStrategy:
    """Tests for ROUND_ROBIN strategy."""

    def test_round_robin_cycles(self):
        """Nodes are selected in sequence."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)
        nodes = [mock_node("a"), mock_node("b"), mock_node("c")]

        assert lb.select(nodes).node.node_id == "a"
        assert lb.select(nodes).node.node_id == "b"
        assert lb.select(nodes).node.node_id == "c"
        assert lb.select(nodes).node.node_id == "a"  # Cycles back

    def test_round_robin_single_node(self):
        """Single node always selected."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)
        nodes = [mock_node("a")]

        assert lb.select(nodes).node.node_id == "a"
        assert lb.select(nodes).node.node_id == "a"
        assert lb.select(nodes).node.node_id == "a"


class TestLeastLoadedStrategy:
    """Tests for LEAST_LOADED strategy."""

    def test_least_loaded_picks_idle(self):
        """Node with fewest jobs is selected."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOADED)
        nodes = [
            mock_node("a", current_jobs=5),
            mock_node("b", current_jobs=0),
            mock_node("c", current_jobs=3),
        ]

        assert lb.select(nodes).node.node_id == "b"

    def test_least_loaded_picks_first_on_tie(self):
        """First node selected when tied."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOADED)
        nodes = [
            mock_node("a", current_jobs=0),
            mock_node("b", current_jobs=0),
        ]

        assert lb.select(nodes).node.node_id == "a"

    def test_least_loaded_with_all_busy(self):
        """Picks least busy even when all have jobs."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOADED)
        nodes = [
            mock_node("a", current_jobs=10),
            mock_node("b", current_jobs=5),
            mock_node("c", current_jobs=15),
        ]

        assert lb.select(nodes).node.node_id == "b"


class TestWeightedStrategy:
    """Tests for WEIGHTED strategy."""

    def test_weighted_prefers_capacity(self):
        """High-capacity nodes preferred when equally loaded."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.WEIGHTED)
        nodes = [
            mock_node("a", cpu_cores=4, memory_mb=8192, current_jobs=0),
            mock_node("b", cpu_cores=16, memory_mb=32768, current_jobs=0),
        ]

        assert lb.select(nodes).node.node_id == "b"

    def test_weighted_penalizes_load(self):
        """Loaded nodes penalized despite high capacity."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.WEIGHTED)
        nodes = [
            mock_node("a", cpu_cores=4, memory_mb=8192, current_jobs=0),
            mock_node("b", cpu_cores=16, memory_mb=32768, current_jobs=10),
        ]

        # Node A: score = (4*1.0 + 8.192*0.5) / 1 = 8.096
        # Node B: score = (16*1.0 + 32.768*0.5) / 11 = 2.94
        assert lb.select(nodes).node.node_id == "a"

    def test_weighted_custom_weights(self):
        """Custom CPU/memory weights work correctly."""
        lb = LoadBalancer(
            strategy=LoadBalanceStrategy.WEIGHTED,
            cpu_weight=0.0,  # Ignore CPU
            memory_weight=1.0,  # Only memory
        )
        nodes = [
            mock_node("a", cpu_cores=16, memory_mb=8192),
            mock_node("b", cpu_cores=4, memory_mb=32768),
        ]

        # With only memory weight, node B wins
        assert lb.select(nodes).node.node_id == "b"


class TestCapabilityStrategy:
    """Tests for CAPABILITY strategy."""

    def test_capability_filters_gpu(self):
        """GPU requirement filters non-GPU nodes."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.CAPABILITY)
        nodes = [
            mock_node("a", gpu=False),
            mock_node("b", gpu=True),
        ]

        result = lb.select(nodes, job_requirements={"gpu": True})
        assert result.node.node_id == "b"

    def test_capability_filters_memory(self):
        """Memory requirement filters low-memory nodes."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.CAPABILITY)
        nodes = [
            mock_node("a", memory_mb=4096),
            mock_node("b", memory_mb=16384),
        ]

        result = lb.select(nodes, job_requirements={"min_memory_mb": 8000})
        assert result.node.node_id == "b"

    def test_capability_filters_cpu(self):
        """CPU requirement filters low-CPU nodes."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.CAPABILITY)
        nodes = [
            mock_node("a", cpu_cores=2),
            mock_node("b", cpu_cores=8),
        ]

        result = lb.select(nodes, job_requirements={"min_cpu_cores": 4})
        assert result.node.node_id == "b"

    def test_capability_combined_requirements(self):
        """Combined requirements filter correctly."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.CAPABILITY)
        nodes = [
            mock_node("a", cpu_cores=8, memory_mb=16384, gpu=False),
            mock_node("b", cpu_cores=4, memory_mb=32768, gpu=True),
            mock_node("c", cpu_cores=8, memory_mb=32768, gpu=True),
        ]

        result = lb.select(nodes, job_requirements={
            "gpu": True,
            "min_cpu_cores": 6,
            "min_memory_mb": 20000,
        })
        assert result.node.node_id == "c"

    def test_capability_fallback_when_no_match(self):
        """Falls back to all nodes when no capability match."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.CAPABILITY)
        nodes = [
            mock_node("a", gpu=False),
            mock_node("b", gpu=False),
        ]

        # Require GPU but none have it - should fallback
        result = lb.select(nodes, job_requirements={"gpu": True})
        assert result.node is not None  # Fallback occurred
        assert lb.stats.fallbacks == 1


class TestEmptyNodesHandling:
    """Tests for edge cases with empty nodes."""

    def test_empty_nodes_returns_none(self):
        """Empty node list returns None gracefully."""
        lb = LoadBalancer()
        result = lb.select([])

        assert result.node is None
        assert result.reason == "No nodes available"
        assert result.candidates_considered == 0


class TestStatsTracking:
    """Tests for statistics tracking."""

    def test_stats_tracking(self):
        """Selection stats are tracked."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)
        nodes = [mock_node("a")]

        lb.select(nodes)
        lb.select(nodes)

        stats = lb.get_stats()
        assert stats["selections_total"] == 2
        assert stats["selections_by_strategy"]["round_robin"] == 2
        assert stats["current_strategy"] == "round_robin"

    def test_stats_reset(self):
        """Stats can be reset."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)
        nodes = [mock_node("a"), mock_node("b")]

        lb.select(nodes)
        lb.select(nodes)

        lb.reset_stats()

        stats = lb.get_stats()
        assert stats["selections_total"] == 0
        # Round-robin index also reset
        assert lb.select(nodes).node.node_id == "a"


class TestSelectionResult:
    """Tests for SelectionResult metadata."""

    def test_selection_result_metadata(self):
        """SelectionResult contains correct metadata."""
        lb = LoadBalancer(strategy=LoadBalanceStrategy.LEAST_LOADED)
        nodes = [
            mock_node("a", current_jobs=5),
            mock_node("b", current_jobs=0),
        ]

        result = lb.select(nodes)

        assert result.node.node_id == "b"
        assert result.strategy_used == LoadBalanceStrategy.LEAST_LOADED
        assert "0 jobs" in result.reason
        assert result.candidates_considered == 2
        assert result.selection_time_ms >= 0

    def test_selection_timing(self):
        """Selection time is measured."""
        lb = LoadBalancer()
        nodes = [mock_node("a") for _ in range(100)]  # Larger list

        result = lb.select(nodes)

        # Should have positive timing (even if very small)
        assert result.selection_time_ms >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
