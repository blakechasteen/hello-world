"""
Unit tests for HoloLoom/memory/multi_wave_engine.py
====================================================

Comprehensive test coverage for the Multi-Wave Memory Engine:
- BrainWaveMode enum
- ActivationPattern dataclass
- ThetaWaveConsolidator
- DeltaWavePruner
- REMDreamer
- MultiWaveMemoryEngine

Created 2026-01-28
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from HoloLoom.memory.multi_wave_engine import (
    BrainWaveMode,
    ActivationPattern,
    ThetaWaveConsolidator,
    DeltaWavePruner,
    REMDreamer,
    MultiWaveMemoryEngine,
)
from HoloLoom.memory.spring_dynamics_engine import (
    SpringEngineConfig,
    MemoryNode,
    BetaWaveRecallResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8  # small embedding dimension for fast tests


def _make_engine(num_nodes: int = 0) -> MultiWaveMemoryEngine:
    """Create an engine and optionally populate it with nodes."""
    config = SpringEngineConfig(dt=0.1, damping=0.95)
    engine = MultiWaveMemoryEngine(config=config)
    for i in range(num_nodes):
        emb = np.random.default_rng(i).standard_normal(DIM).astype(np.float32)
        engine.add_node(
            node_id=f"node_{i}",
            embedding=emb,
            content=f"content {i}",
            neighbors=set(),
            initial_spring_constant=1.0,
        )
    return engine


def _make_node(node_id: str = "n1", k: float = 1.0, neighbors: set = None,
               last_accessed: datetime = None, decay_rate: float = 0.01) -> MemoryNode:
    """Create a standalone MemoryNode for direct testing."""
    emb = np.zeros(DIM, dtype=np.float32)
    node = MemoryNode(
        id=node_id,
        content="test",
        position=emb.copy(),
        rest_position=emb.copy(),
        velocity=np.zeros(DIM, dtype=np.float32),
        spring_constant=k,
        neighbors=neighbors or set(),
        decay_rate=decay_rate,
    )
    if last_accessed is not None:
        node.last_accessed = last_accessed
    return node


# ===========================================================================
# 1. BrainWaveMode enum
# ===========================================================================

class TestBrainWaveMode:
    """Tests for BrainWaveMode enum values and behaviour."""

    def test_enum_values(self):
        """All five expected brain-wave modes exist with correct string values."""
        assert BrainWaveMode.BETA.value == "beta"
        assert BrainWaveMode.ALPHA.value == "alpha"
        assert BrainWaveMode.THETA.value == "theta"
        assert BrainWaveMode.DELTA.value == "delta"
        assert BrainWaveMode.REM.value == "rem"

    def test_enum_member_count(self):
        """There should be exactly 5 brain-wave modes."""
        assert len(BrainWaveMode) == 5


# ===========================================================================
# 2. ActivationPattern dataclass
# ===========================================================================

class TestActivationPattern:
    """Tests for the ActivationPattern dataclass."""

    def test_creation(self):
        """ActivationPattern stores nodes, timestamp, and source."""
        ts = datetime(2026, 1, 28, 12, 0, 0)
        pattern = ActivationPattern(nodes=["a", "b"], timestamp=ts, source="query")
        assert pattern.nodes == ["a", "b"]
        assert pattern.timestamp == ts
        assert pattern.source == "query"

    def test_creation_ingestion_source(self):
        """Source can also be 'ingestion'."""
        pattern = ActivationPattern(nodes=["x"], timestamp=datetime.now(), source="ingestion")
        assert pattern.source == "ingestion"


# ===========================================================================
# 3. ThetaWaveConsolidator
# ===========================================================================

class TestThetaWaveConsolidator:
    """Tests for ThetaWaveConsolidator recording and consolidation."""

    def test_record_activation_pattern_above_threshold(self):
        """Patterns with >=2 active nodes above threshold are recorded."""
        engine = _make_engine(num_nodes=3)
        consolidator = engine.theta_consolidator

        activations = {"node_0": 0.5, "node_1": 0.8, "node_2": 0.1}
        consolidator.record_activation_pattern(activations, source="query", threshold=0.3)

        assert len(consolidator.co_activation_history) == 1
        recorded_nodes = consolidator.co_activation_history[0].nodes
        assert "node_0" in recorded_nodes
        assert "node_1" in recorded_nodes
        # node_2 is below threshold
        assert "node_2" not in recorded_nodes

    def test_record_activation_pattern_below_threshold_ignored(self):
        """Patterns where fewer than 2 nodes are active are NOT recorded."""
        engine = _make_engine(num_nodes=2)
        consolidator = engine.theta_consolidator

        # Only one node above threshold
        activations = {"node_0": 0.5, "node_1": 0.1}
        consolidator.record_activation_pattern(activations, source="query", threshold=0.3)
        assert len(consolidator.co_activation_history) == 0

    def test_max_history_trimming(self):
        """History is capped at max_history entries."""
        engine = _make_engine(num_nodes=3)
        consolidator = engine.theta_consolidator
        consolidator.max_history = 5

        for _ in range(10):
            activations = {"node_0": 0.5, "node_1": 0.8}
            consolidator.record_activation_pattern(activations, source="query")

        assert len(consolidator.co_activation_history) == 5

    def test_theta_consolidation_update_creates_neighbors(self):
        """After enough co-activations, nodes become mutual neighbors."""
        engine = _make_engine(num_nodes=2)
        consolidator = engine.theta_consolidator
        consolidator.consolidation_threshold = 3

        # Record the same pair 4 times (above threshold of 3)
        for _ in range(4):
            activations = {"node_0": 0.5, "node_1": 0.8}
            consolidator.record_activation_pattern(activations, source="query")

        consolidated = consolidator.theta_consolidation_update()
        assert consolidated >= 1

        # Nodes should now be mutual neighbors
        assert "node_1" in engine.nodes["node_0"].neighbors
        assert "node_0" in engine.nodes["node_1"].neighbors

    def test_theta_consolidation_update_no_patterns(self):
        """Consolidation with no recorded patterns returns 0."""
        engine = _make_engine(num_nodes=2)
        consolidated = engine.theta_consolidator.theta_consolidation_update()
        assert consolidated == 0

    def test_strengthen_connection_missing_node(self):
        """_strengthen_connection returns False when a node is missing."""
        engine = _make_engine(num_nodes=1)
        result = engine.theta_consolidator._strengthen_connection("node_0", "nonexistent", 1.0)
        assert result is False


# ===========================================================================
# 4. DeltaWavePruner
# ===========================================================================

class TestDeltaWavePruner:
    """Tests for DeltaWavePruner pruning weak and strengthening strong nodes."""

    def test_prune_weak_old_node(self):
        """Weak spring constant + old access time leads to pruning (neighbors cleared, k halved)."""
        engine = _make_engine(num_nodes=0)
        pruner = engine.delta_pruner
        pruner.prune_after_hours = 72

        # Manually add a weak, old node with a neighbor
        old_time = datetime.now() - timedelta(hours=100)
        node_weak = _make_node("weak", k=0.2, neighbors={"other"}, last_accessed=old_time)
        node_other = _make_node("other", k=2.0, neighbors={"weak"})
        engine.nodes["weak"] = node_weak
        engine.nodes["other"] = node_other

        pruned, strengthened = pruner.delta_pruning_update()

        assert pruned >= 1
        # weak node should have no neighbors now
        assert len(engine.nodes["weak"].neighbors) == 0
        # other should no longer list weak as neighbor
        assert "weak" not in engine.nodes["other"].neighbors

    def test_strengthen_strong_node(self):
        """Nodes with high spring constant get slightly stronger and decay slower."""
        engine = _make_engine(num_nodes=0)
        pruner = engine.delta_pruner

        strong_node = _make_node("strong", k=6.0)
        engine.nodes["strong"] = strong_node
        original_k = strong_node.spring_constant
        original_decay = strong_node.decay_rate

        pruner.delta_pruning_update()

        # spring_constant should have increased by factor 1.05
        assert engine.nodes["strong"].spring_constant == pytest.approx(original_k * 1.05, rel=1e-4)
        # decay_rate should have decreased by factor 0.95
        assert engine.nodes["strong"].decay_rate == pytest.approx(original_decay * 0.95, rel=1e-4)

    def test_velocity_damping_during_delta(self):
        """All nodes get dramatic velocity damping (* 0.1) during delta update."""
        engine = _make_engine(num_nodes=1)
        engine.nodes["node_0"].velocity = np.ones(DIM, dtype=np.float32) * 5.0
        original_v = engine.nodes["node_0"].velocity.copy()

        engine.delta_pruner.delta_pruning_update()

        np.testing.assert_allclose(
            engine.nodes["node_0"].velocity,
            original_v * 0.1,
            atol=1e-6,
        )

    def test_no_pruning_when_recently_accessed(self):
        """Weak but recently accessed nodes are NOT pruned."""
        engine = _make_engine(num_nodes=0)
        pruner = engine.delta_pruner
        pruner.prune_after_hours = 72

        recent_node = _make_node("recent", k=0.2, last_accessed=datetime.now())
        engine.nodes["recent"] = recent_node

        pruned, _ = pruner.delta_pruning_update()
        assert pruned == 0


# ===========================================================================
# 5. REMDreamer
# ===========================================================================

class TestREMDreamer:
    """Tests for REMDreamer dream cycle."""

    @pytest.mark.asyncio
    async def test_dream_cycle_too_few_nodes(self):
        """Dream cycle exits immediately when fewer nodes than random_seed_count."""
        engine = _make_engine(num_nodes=1)  # Only 1 node, need 3 seeds
        bridges = await engine.rem_dreamer.dream_cycle(duration_seconds=0.1)
        assert bridges == 0

    @pytest.mark.asyncio
    async def test_dream_cycle_creates_bridges(self):
        """Dream cycle can create bridge connections between distant nodes."""
        engine = _make_engine(num_nodes=0)
        dreamer = engine.rem_dreamer
        dreamer.random_seed_count = 2
        dreamer.bridge_distance_threshold = 0.5  # low threshold so bridges are likely

        # Create nodes with distant rest positions
        for i in range(4):
            emb = np.zeros(DIM, dtype=np.float32)
            emb[0] = float(i) * 10.0  # spread far apart
            engine.add_node(
                node_id=f"far_{i}",
                embedding=emb,
                content=f"far {i}",
                neighbors=set(),
            )
        # Make all nodes neighbors so activation can spread
        for n in engine.nodes.values():
            n.neighbors = set(engine.nodes.keys()) - {n.id}

        bridges = await dreamer.dream_cycle(duration_seconds=0.5)
        # We can't guarantee exact count due to randomness,
        # but the function should run without error.
        assert isinstance(bridges, int)

    @pytest.mark.asyncio
    async def test_dream_cycle_returns_int(self):
        """dream_cycle always returns an integer (bridges_created)."""
        engine = _make_engine(num_nodes=5)
        result = await engine.rem_dreamer.dream_cycle(duration_seconds=0.3)
        assert isinstance(result, int)
        assert result >= 0


# ===========================================================================
# 6. MultiWaveMemoryEngine initialisation and mode switching
# ===========================================================================

class TestMultiWaveMemoryEngine:
    """Tests for the main MultiWaveMemoryEngine class."""

    def test_initial_state(self):
        """Engine starts in BETA mode with correct sub-components."""
        engine = _make_engine()
        assert engine.mode == BrainWaveMode.BETA
        assert isinstance(engine.theta_consolidator, ThetaWaveConsolidator)
        assert isinstance(engine.delta_pruner, DeltaWavePruner)
        assert isinstance(engine.rem_dreamer, REMDreamer)
        assert engine.ingestion_active is False
        assert engine.total_ingested == 0

    def test_mode_stays_beta_during_ingestion(self):
        """While ingestion is active, mode is forced to BETA."""
        engine = _make_engine()
        engine.ingestion_active = True
        engine.last_query_time = datetime.now() - timedelta(hours=5)
        engine._update_mode()
        assert engine.mode == BrainWaveMode.BETA

    def test_mode_beta_when_recently_active(self):
        """Mode is BETA when last query was < 5 minutes ago."""
        engine = _make_engine()
        engine.last_query_time = datetime.now() - timedelta(minutes=2)
        engine._update_mode()
        assert engine.mode == BrainWaveMode.BETA

    def test_mode_alpha_when_idle_5_to_30_min(self):
        """Mode switches to ALPHA between 5 and 30 minutes idle."""
        engine = _make_engine()
        engine.last_query_time = datetime.now() - timedelta(minutes=15)
        engine._update_mode()
        assert engine.mode == BrainWaveMode.ALPHA

    def test_mode_theta_when_idle_30_to_120_min(self):
        """Mode switches to THETA between 30 and 120 minutes idle."""
        engine = _make_engine()
        engine.last_query_time = datetime.now() - timedelta(minutes=60)
        engine._update_mode()
        assert engine.mode == BrainWaveMode.THETA

    def test_mode_delta_or_rem_after_2_hours(self):
        """Mode is DELTA or REM after >2 hours idle."""
        engine = _make_engine()
        engine.last_query_time = datetime.now() - timedelta(hours=3)
        engine._update_mode()
        assert engine.mode in (BrainWaveMode.DELTA, BrainWaveMode.REM)

    def test_on_query_resets_to_beta(self):
        """Calling on_query switches mode back to BETA and records pattern."""
        engine = _make_engine(num_nodes=3)
        engine.mode = BrainWaveMode.THETA  # simulate idle state

        query_emb = np.random.default_rng(42).standard_normal(DIM).astype(np.float32)
        result = engine.on_query(query_emb)

        assert engine.mode == BrainWaveMode.BETA
        assert isinstance(result, BetaWaveRecallResult)

    def test_encode_new_memory_adds_node(self):
        """encode_new_memory creates a node in the engine."""
        engine = _make_engine(num_nodes=0)
        emb = np.ones(DIM, dtype=np.float32)
        engine.encode_new_memory(
            node_id="new_1",
            content="hello world",
            embedding=emb,
            initial_k=2.0,
        )
        assert "new_1" in engine.nodes
        assert engine.nodes["new_1"].spring_constant == 2.0

    def test_get_statistics_includes_mode(self):
        """get_statistics returns mode, ingested count, and consolidation history size."""
        engine = _make_engine(num_nodes=2)
        stats = engine.get_statistics()
        assert "mode" in stats
        assert stats["mode"] == "beta"
        assert "total_ingested" in stats
        assert "consolidation_history_size" in stats

    def test_alpha_filtering_suppresses_weak_velocities(self):
        """Alpha filtering dampens nodes with low velocity magnitude."""
        engine = _make_engine(num_nodes=1)
        node = engine.nodes["node_0"]
        node.velocity = np.ones(DIM, dtype=np.float32) * 0.05  # below alpha threshold 0.2
        original_k = node.spring_constant

        engine.active_nodes.add("node_0")
        engine._alpha_filtering_update()

        # velocity should be damped (* 0.9)
        assert np.linalg.norm(node.velocity) < np.linalg.norm(np.ones(DIM) * 0.05)
        # spring_constant should be slightly weakened (* 0.99)
        assert node.spring_constant == pytest.approx(original_k * 0.99, rel=1e-4)

    def test_alpha_filtering_boosts_strong_signals(self):
        """Alpha filtering slightly boosts nodes with high spring constant and velocity."""
        engine = _make_engine(num_nodes=1)
        node = engine.nodes["node_0"]
        node.velocity = np.ones(DIM, dtype=np.float32) * 5.0  # above alpha threshold
        node.spring_constant = 4.0  # above 3.0 strong threshold
        original_k = node.spring_constant

        engine.active_nodes.add("node_0")
        engine._alpha_filtering_update()

        assert node.spring_constant == pytest.approx(original_k * 1.005, rel=1e-4)

    def test_empty_engine_statistics(self):
        """Statistics work correctly on an engine with no nodes."""
        engine = _make_engine(num_nodes=0)
        stats = engine.get_statistics()
        assert stats["total_nodes"] == 0
        assert stats["active_nodes"] == 0
        assert stats["mode"] == "beta"
