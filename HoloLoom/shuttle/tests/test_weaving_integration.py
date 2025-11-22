"""
Unit Tests for Shuttle Weaving Integration

Tests the integration adapters that connect Shuttle to WeavingOrchestrator:
- HoloLoomWarpAdapter (Qdrant integration)
- HoloLoomYarnAdapter (Neo4j/NetworkX integration)
- ShuttleStage (drop-in Step 3 replacement)
- Configuration derivation (ExecutionMode → ShuttleMode)

Created: 2025-01-21
"""

import pytest
from datetime import datetime, timedelta
from typing import List

# HoloLoom types
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.config import Config, ExecutionMode
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.chrono.trigger import TemporalWindow

# Shuttle components
from HoloLoom.shuttle.weaving_integration import (
    HoloLoomWarpAdapter,
    HoloLoomYarnAdapter,
    ShuttleStage,
    create_shuttle_stage
)
from HoloLoom.shuttle.config import ShuttleConfig, ShuttleMode


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_retriever():
    """Mock retriever that returns test memory shards."""
    class MockRetriever:
        def retrieve(self, query: str, k: int = 10) -> List[MemoryShard]:
            # Return mock shards
            return [
                MemoryShard(
                    id=f"shard_{i}",
                    text=f"Memory {i}: {query}",
                    metadata={'source': 'mock', 'score': 0.8 - i * 0.1}
                )
                for i in range(min(k, 3))
            ]
    return MockRetriever()


@pytest.fixture
def mock_kg():
    """Mock KG with test graph structure."""
    kg = KG()
    # Add test edges
    kg.add_edges([
        KGEdge("thompson_sampling", "bandit", "USES", 1.0),
        KGEdge("thompson_sampling", "bayesian", "IS_A", 1.0),
        KGEdge("bandit", "exploration", "LEADS_TO", 0.8),
        KGEdge("bandit", "exploitation", "LEADS_TO", 0.8),
        KGEdge("bayesian", "beta_distribution", "USES", 0.9),
    ])

    # Add node descriptions
    for node in kg.G.nodes():
        kg.G.nodes[node]['description'] = f"Description of {node}"

    return kg


# ============================================================================
# HoloLoomWarpAdapter Tests
# ============================================================================

def test_warp_adapter_initialization(mock_retriever):
    """Test HoloLoomWarpAdapter initializes correctly."""
    adapter = HoloLoomWarpAdapter(mock_retriever)
    assert adapter.retriever is mock_retriever


def test_warp_adapter_search(mock_retriever):
    """Test HoloLoomWarpAdapter.search() returns correct format."""
    adapter = HoloLoomWarpAdapter(mock_retriever)

    results = adapter.search("What is Thompson Sampling?", top_k=5)

    # Should return list of dicts
    assert isinstance(results, list)
    assert len(results) == 3  # Mock returns 3 shards

    # Check format
    for result in results:
        assert 'id' in result
        assert 'text' in result
        assert 'score' in result
        assert 'metadata' in result
        assert 'entities' in result


def test_warp_adapter_top_k(mock_retriever):
    """Test HoloLoomWarpAdapter respects top_k parameter."""
    adapter = HoloLoomWarpAdapter(mock_retriever)

    results_5 = adapter.search("test", top_k=5)
    results_2 = adapter.search("test", top_k=2)

    # Both should work (mock returns 3 max)
    assert len(results_5) == 3
    assert len(results_2) == 2


# ============================================================================
# HoloLoomYarnAdapter Tests
# ============================================================================

def test_yarn_adapter_initialization(mock_kg):
    """Test HoloLoomYarnAdapter initializes correctly."""
    adapter = HoloLoomYarnAdapter(mock_kg)
    assert adapter.kg is mock_kg


def test_yarn_adapter_build_neighbor_map(mock_kg):
    """Test HoloLoomYarnAdapter.build_neighbor_map() traverses graph."""
    adapter = HoloLoomYarnAdapter(mock_kg)

    # Start from thompson_sampling node
    anchors = ["thompson_sampling"]
    allowed_edge_types = ["USES", "IS_A", "LEADS_TO"]

    neighbor_map, all_nodes = adapter.build_neighbor_map(
        anchors=anchors,
        allowed_edge_types=allowed_edge_types,
        max_depth=2,
        max_nodes=10
    )

    # Should have found neighbors
    assert isinstance(neighbor_map, dict)
    assert isinstance(all_nodes, list)
    assert len(all_nodes) > 1  # At least anchor + neighbors

    # Anchor should be in map
    assert "thompson_sampling" in neighbor_map

    # Should have neighbors (bandit, bayesian)
    assert "bandit" in all_nodes or "bayesian" in all_nodes


def test_yarn_adapter_max_depth(mock_kg):
    """Test HoloLoomYarnAdapter respects max_depth."""
    adapter = HoloLoomYarnAdapter(mock_kg)

    anchors = ["thompson_sampling"]
    allowed_edge_types = ["USES", "IS_A", "LEADS_TO"]

    # Depth 1: only direct neighbors
    _, nodes_1 = adapter.build_neighbor_map(
        anchors=anchors,
        allowed_edge_types=allowed_edge_types,
        max_depth=1,
        max_nodes=100
    )

    # Depth 2: neighbors + their neighbors
    _, nodes_2 = adapter.build_neighbor_map(
        anchors=anchors,
        allowed_edge_types=allowed_edge_types,
        max_depth=2,
        max_nodes=100
    )

    # Depth 2 should find more nodes
    assert len(nodes_2) >= len(nodes_1)


def test_yarn_adapter_get_node_description(mock_kg):
    """Test HoloLoomYarnAdapter.get_node_description() retrieves text."""
    adapter = HoloLoomYarnAdapter(mock_kg)

    node_ids = ["thompson_sampling", "bandit"]
    descriptions = adapter.get_node_description(node_ids)

    # Should return dict
    assert isinstance(descriptions, dict)
    assert len(descriptions) == 2

    # Should have descriptions
    assert "thompson_sampling" in descriptions
    assert "Description of thompson_sampling" in descriptions["thompson_sampling"]


# ============================================================================
# ShuttleStage Tests
# ============================================================================

def test_shuttle_stage_initialization(mock_kg, mock_retriever):
    """Test ShuttleStage initializes correctly."""
    config = Config.fast()

    stage = ShuttleStage(
        config=config,
        kg=mock_kg,
        retriever=mock_retriever
    )

    assert stage.config is config
    assert stage.kg is mock_kg
    assert stage.retriever is mock_retriever
    assert stage.warp is not None
    assert stage.yarn is not None
    assert stage.shuttle is not None


def test_shuttle_stage_config_derivation_bare():
    """Test ShuttleStage derives MINIMAL mode from BARE."""
    config = Config.bare()

    shuttle_config = ShuttleStage._derive_shuttle_config(None, config)

    assert shuttle_config.mode == ShuttleMode.MINIMAL
    assert shuttle_config.mcts_simulations == 16  # BARE uses lighter settings
    assert shuttle_config.max_graph_depth == 1


def test_shuttle_stage_config_derivation_fast():
    """Test ShuttleStage derives LITE mode from FAST."""
    config = Config.fast()

    shuttle_config = ShuttleStage._derive_shuttle_config(None, config)

    assert shuttle_config.mode == ShuttleMode.LITE
    assert shuttle_config.mcts_simulations == 16
    assert shuttle_config.max_graph_depth == 1


def test_shuttle_stage_config_derivation_fused():
    """Test ShuttleStage derives FULL mode from FUSED."""
    config = Config.fused()

    shuttle_config = ShuttleStage._derive_shuttle_config(None, config)

    assert shuttle_config.mode == ShuttleMode.FULL
    assert shuttle_config.mcts_simulations == 32
    assert shuttle_config.max_graph_depth == 2


@pytest.mark.asyncio
async def test_shuttle_stage_select_threads(mock_kg, mock_retriever):
    """Test ShuttleStage.select_threads() returns MemoryShard format."""
    config = Config.fast()

    stage = ShuttleStage(
        config=config,
        kg=mock_kg,
        retriever=mock_retriever
    )

    # Create temporal window
    temporal_window = TemporalWindow(
        start=datetime.now() - timedelta(days=1),
        end=datetime.now()
    )

    query = Query(text="What is Thompson Sampling?")

    # Select threads
    threads = await stage.select_threads(temporal_window, query)

    # Should return list of MemoryShard
    assert isinstance(threads, list)
    assert all(isinstance(t, MemoryShard) for t in threads)

    # Should have some threads (from Warp + Yarn)
    assert len(threads) > 0


@pytest.mark.asyncio
async def test_shuttle_stage_fallback_on_error(mock_kg, mock_retriever):
    """Test ShuttleStage falls back to simple retrieval on error."""
    config = Config.fast()

    # Create stage with graceful degradation
    stage = ShuttleStage(
        config=config,
        kg=mock_kg,
        retriever=mock_retriever
    )
    stage.shuttle_config.enable_graceful_degradation = True

    # Break the shuttle (simulate error)
    original_shuttle = stage.shuttle
    stage.shuttle = None

    temporal_window = TemporalWindow(
        start=datetime.now() - timedelta(days=1),
        end=datetime.now()
    )

    query = Query(text="test")

    # Should fall back gracefully
    threads = await stage.select_threads(temporal_window, query)

    # Should still return threads (from fallback)
    assert isinstance(threads, list)
    # Note: May be empty if retriever also fails, but shouldn't crash


# ============================================================================
# Factory Function Tests
# ============================================================================

def test_create_shuttle_stage(mock_kg, mock_retriever):
    """Test create_shuttle_stage() factory function."""
    config = Config.fast()

    stage = create_shuttle_stage(
        config=config,
        kg=mock_kg,
        retriever=mock_retriever
    )

    assert isinstance(stage, ShuttleStage)
    assert stage.config is config


def test_create_shuttle_stage_custom_config(mock_kg, mock_retriever):
    """Test create_shuttle_stage() with custom shuttle config."""
    config = Config.fast()

    custom_shuttle_config = ShuttleConfig(
        mode=ShuttleMode.FULL,
        mcts_simulations=64,
        max_graph_depth=3
    )

    stage = create_shuttle_stage(
        config=config,
        kg=mock_kg,
        retriever=mock_retriever,
        shuttle_config=custom_shuttle_config
    )

    # Should use custom config
    assert stage.shuttle_config.mcts_simulations == 64
    assert stage.shuttle_config.max_graph_depth == 3


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
