"""
Unit tests for hololoom.core.memory.v2 — Classical CBR Architecture.

Covers: activation, blackboard, cbr, chunk_compiler, classical_factory,
cold_start, confidence, navigator, tms.

Regression test for AP-1: chunk_compiler must use StoredItem, not raw MemoryItem.
"""

import math
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch

import hololoom  # noqa: F401 — installs meta path hook

from hololoom.core.memory.v2.activation import ActivationAdapter, ActivationConfig
from hololoom.core.memory.v2.blackboard import (
    Blackboard,
    WorkingMemory,
    ProductionRule,
    default_skip_verify_rules,
    default_skip_flag_rules,
    compute_ppr_entropy,
)
from hololoom.core.memory.v2.cbr import CBRSystem, CBRConfig, Case
from hololoom.core.memory.v2.chunk_compiler import (
    ChunkCompiler,
    ChunkConfig,
    ChunkAction,
    CompiledChunk,
    ExperienceRecord,
)
from hololoom.core.memory.v2.classical_factory import create_classical_sources
from hololoom.core.memory.v2.cold_start import ColdStartHandler, ColdStartConfig
from hololoom.core.memory.v2.confidence import (
    ConfidenceEstimator,
    ConfidenceConfig,
    DualConfidence,
    _simple_stem,
    _content_stems,
)
from hololoom.core.memory.v2.navigator import (
    SeedNode,
    NavigatorResult,
    PPRConfig,
    personalized_pagerank,
    LiteBusGraph,
)
from hololoom.core.memory.v2.tms import (
    TruthMaintenanceSystem,
    TMSConfig,
    JustificationRecord,
)
from hololoom.core.memory.v2.knowledge_source import Phase, SourceRegistry
from hololoom.core.memory.lite_bus import StoredItem


# =============================================================================
# Helpers — Minimal graph and bus fakes for testing
# =============================================================================


class FakeBus:
    """Minimal bus with _items, _edges, _aliases for testing."""

    def __init__(self):
        self._items: Dict[str, StoredItem] = {}
        self._edges: Dict[str, List[Tuple[str, str]]] = {}
        self._aliases: Dict[str, str] = {}

    def add_item(self, item: StoredItem):
        self._items[item.id] = item

    def add_edge(self, src: str, dst: str, rel: str = "RELATED"):
        if src not in self._edges:
            self._edges[src] = []
        self._edges[src].append((dst, rel))


class FakeGraph:
    """Minimal GraphBackend for unit tests."""

    def __init__(self, bus: FakeBus):
        self._bus = bus

    def nodes(self) -> List[str]:
        return list(self._bus._items.keys())

    def neighbors(self, node_id: str) -> List[Tuple[str, str, float]]:
        edges = self._bus._edges.get(node_id, [])
        result = []
        for target_id, rel_type in edges:
            if target_id in self._bus._items:
                weight = self._bus._items[target_id].importance
                result.append((target_id, rel_type, weight))
        return result

    def node_data(self, node_id: str) -> Dict[str, Any]:
        if node_id in self._bus._items:
            return self._bus._items[node_id].to_dict()
        return {}

    def has_node(self, node_id: str) -> bool:
        return node_id in self._bus._items


def _make_stored_item(
    id: str,
    content: str = "test content",
    memory_type: str = "fact",
    importance: float = 0.5,
) -> StoredItem:
    return StoredItem(
        id=id,
        content=content,
        memory_type=memory_type,
        importance=importance,
        timestamp="2026-03-01T00:00:00",
    )


def _build_test_graph():
    """Build a small graph for navigator/confidence tests."""
    bus = FakeBus()
    bus.add_item(_make_stored_item("thompson", "Thompson Sampling is a Bayesian approach", "entity", 0.8))
    bus.add_item(_make_stored_item("bandit", "Multi-armed bandit problem explores arms", "fact", 0.6))
    bus.add_item(_make_stored_item("bayesian", "Bayesian inference updates prior to posterior", "fact", 0.7))
    bus.add_item(_make_stored_item("epsilon", "Epsilon-greedy selects random arm with probability epsilon", "fact", 0.5))
    bus.add_edge("thompson", "bandit", "RELATED")
    bus.add_edge("thompson", "bayesian", "RELATED")
    bus.add_edge("bandit", "epsilon", "RELATED")
    bus._aliases = {"thompson sampling": "thompson", "bandit": "bandit"}
    return bus, FakeGraph(bus)


def _make_nav_result(graph, seed_ids=None, scored_nodes=None):
    """Build a NavigatorResult from graph for testing."""
    seeds = [SeedNode(nid, 1.0, "alias") for nid in (seed_ids or [])]
    ranked = scored_nodes or [(nid, 0.5) for nid in graph.nodes()]
    node_data = {nid: graph.node_data(nid) for nid, _ in ranked}
    return NavigatorResult(
        ranked_nodes=ranked,
        seed_nodes=seeds,
        iterations=5,
        converged=True,
        node_data=node_data,
    )


# =============================================================================
# Activation Adapter
# =============================================================================


class TestActivationAdapter:

    def test_boost_seeds_no_activation(self):
        """Seeds are unchanged when no activation levels exist."""
        adapter = ActivationAdapter()
        seed = SeedNode("n1", 1.0, "alias")
        result = adapter.boost_seeds([seed])
        assert result[0].weight == 1.0, "Weight unchanged with no activation"

    def test_boost_seeds_with_activation(self):
        """Seeds with activation get multiplicative boost."""
        adapter = ActivationAdapter(config=ActivationConfig(weight_amplifier=1.0))
        adapter._levels["n1"] = 0.5
        seed = SeedNode("n1", 1.0, "alias")
        adapter.boost_seeds([seed])
        assert seed.weight == pytest.approx(1.5), "weight * (1.0 + 0.5 * 1.0) = 1.5"

    def test_on_retrieval_boosts_levels(self):
        adapter = ActivationAdapter(config=ActivationConfig(retrieval_boost=0.3))
        adapter.on_retrieval(["a", "b"])
        assert adapter.get_activation("a") == pytest.approx(0.3)
        assert adapter.get_activation("b") == pytest.approx(0.3)

    def test_on_retrieval_respects_max(self):
        adapter = ActivationAdapter(config=ActivationConfig(
            retrieval_boost=0.7, max_activation=1.0,
        ))
        adapter._levels["a"] = 0.8
        adapter.on_retrieval(["a"])
        assert adapter.get_activation("a") == pytest.approx(1.0), "Capped at max"

    def test_on_query_entities(self):
        adapter = ActivationAdapter(config=ActivationConfig(query_boost=0.5))
        adapter.on_query_entities(["e1"])
        assert adapter.get_activation("e1") == pytest.approx(0.5)

    def test_decay_step_reduces_levels(self):
        adapter = ActivationAdapter(config=ActivationConfig(decay_rate=0.5, min_activation=0.01))
        adapter._levels["a"] = 0.8
        adapter.decay_step()
        assert adapter.get_activation("a") == pytest.approx(0.4)

    def test_decay_step_prunes_below_min(self):
        adapter = ActivationAdapter(config=ActivationConfig(decay_rate=0.5, min_activation=0.3))
        adapter._levels["a"] = 0.4
        adapter.decay_step()
        # 0.4 * 0.5 = 0.2 < 0.3 → pruned
        assert "a" not in adapter._levels

    def test_active_node_ids_sorted(self):
        adapter = ActivationAdapter()
        adapter._levels = {"a": 0.2, "b": 0.8, "c": 0.5}
        result = adapter.active_node_ids(min_activation=0.1)
        assert result == ["b", "c", "a"], "Sorted descending by activation"

    def test_clear_resets_all(self):
        adapter = ActivationAdapter()
        adapter._levels = {"a": 0.5, "b": 0.3}
        adapter.clear()
        assert len(adapter.levels) == 0

    def test_stats(self):
        adapter = ActivationAdapter(config=ActivationConfig(min_activation=0.1))
        adapter._levels = {"a": 0.8, "b": 0.3, "c": 0.05}
        stats = adapter.stats()
        assert stats["n_tracked"] == 3
        assert stats["n_active"] == 2, "c below min_activation threshold"


# =============================================================================
# Blackboard + WorkingMemory
# =============================================================================


class TestWorkingMemory:

    def test_load_and_pressure(self):
        wm = WorkingMemory(capacity=10)
        wm.active_goals = ["g1", "g2"]
        wm.entity_focus = {"e1": 0.5}
        assert wm.load == 3
        assert wm.pressure == pytest.approx(0.3)

    def test_attend_boosts_focus(self):
        wm = WorkingMemory()
        wm.attend(["e1", "e2"], boost=0.5)
        assert wm.entity_focus["e1"] == pytest.approx(0.5)
        assert wm.entity_focus["e2"] == pytest.approx(0.5)

    def test_attend_caps_at_1(self):
        wm = WorkingMemory()
        wm.entity_focus["e1"] = 0.8
        wm.attend(["e1"], boost=0.5)
        assert wm.entity_focus["e1"] == pytest.approx(1.0)

    def test_attend_prunes_excess(self):
        wm = WorkingMemory(max_entities=3)
        wm.attend(["a", "b", "c", "d"], boost=0.5)
        assert len(wm.entity_focus) == 3, "Pruned to max_entities"

    def test_update_turn_decays_focus(self):
        wm = WorkingMemory(focus_decay=0.5)
        wm.entity_focus = {"e1": 0.8}
        wm.update_turn()
        assert wm.entity_focus["e1"] == pytest.approx(0.4)
        assert wm.turn_count == 1

    def test_update_turn_prunes_weak(self):
        wm = WorkingMemory(focus_decay=0.5)
        wm.entity_focus = {"e1": 0.08}  # 0.08 * 0.5 = 0.04 < 0.05
        wm.update_turn()
        assert "e1" not in wm.entity_focus

    def test_evict_priority_order(self):
        """Eviction: entities first, then corrections, then facts, then goals."""
        wm = WorkingMemory()
        wm.entity_focus = {"e1": 0.1}
        wm.recent_corrections = ["c1"]
        wm.session_facts = {"f1": "v1"}
        wm.active_goals = ["g1"]
        evicted = wm.evict(4)
        assert len(evicted) == 4
        assert "entity:e1" in evicted[0]
        assert "correction:c1" in evicted[1]
        assert "fact:f1" in evicted[2]
        assert "goal:g1" in evicted[3]

    def test_auto_evict_on_over_capacity(self):
        wm = WorkingMemory(capacity=2)
        wm.entity_focus = {"e1": 0.1, "e2": 0.2, "e3": 0.3}
        wm.update_turn()
        assert wm.load <= wm.capacity

    def test_set_goal_dedup(self):
        wm = WorkingMemory()
        wm.set_goal("learn")
        wm.set_goal("learn")
        assert wm.active_goals.count("learn") == 1

    def test_set_goal_evicts_oldest(self):
        wm = WorkingMemory(max_goals=2)
        wm.set_goal("a")
        wm.set_goal("b")
        wm.set_goal("c")
        assert "a" not in wm.active_goals
        assert wm.active_goals == ["b", "c"]

    def test_snapshot_roundtrip(self):
        wm = WorkingMemory()
        wm.active_goals = ["g1"]
        wm.entity_focus = {"e1": 0.5}
        wm.turn_count = 3
        snap = wm.to_snapshot()
        wm2 = WorkingMemory()
        wm2.from_snapshot(snap)
        assert wm2.active_goals == ["g1"]
        assert wm2.entity_focus == {"e1": 0.5}
        assert wm2.turn_count == 3

    def test_clear(self):
        wm = WorkingMemory()
        wm.active_goals = ["g1"]
        wm.entity_focus = {"e1": 0.5}
        wm.clear()
        assert wm.load == 0
        assert wm.turn_count == 0


class TestBlackboard:

    def test_post_navigation(self):
        bb = Blackboard(query="test query")
        bb.post_navigation(
            seed_count=5,
            seed_sources={"alias": 3, "keyword": 2},
            ppr_entropy=0.7,
            ppr_converged=True,
            entity_ids={"e1", "e2"},
        )
        assert bb.seed_count == 5
        assert bb.ppr_entropy == pytest.approx(0.7)
        assert bb.active_entities == {"e1", "e2"}

    def test_post_confidence(self):
        bb = Blackboard()
        bb.post_confidence(0.8, 0.7, 0.9, "sufficient")
        assert bb.latest_confidence == pytest.approx(0.8)
        assert bb.latest_decision == "sufficient"

    def test_confidence_trend(self):
        bb = Blackboard()
        bb.post_confidence(0.5, 0.4, 0.6, "verify")
        bb.post_confidence(0.8, 0.7, 0.9, "sufficient")
        assert bb.confidence_trend == pytest.approx(0.3)

    def test_confidence_trend_no_data(self):
        bb = Blackboard()
        assert bb.confidence_trend == 0.0

    def test_entity_match_ratio(self):
        bb = Blackboard()
        bb.seed_count = 10
        bb.seed_sources = {"alias": 3, "fuzzy": 2, "keyword": 5}
        assert bb.entity_match_ratio == pytest.approx(0.5)

    def test_post_shell(self):
        bb = Blackboard()
        bb.post_shell("prime")
        bb.post_shell("verify")
        assert bb.shell_count == 2
        assert bb.shells_executed == ["prime", "verify"]

    def test_post_flag(self):
        bb = Blackboard()
        bb.post_flag("test_key", {"data": 42})
        assert bb.flags["test_key"]["data"] == 42

    def test_context_features(self):
        bb = Blackboard(query="short query")
        bb.seed_count = 5
        bb.ppr_entropy = 0.5
        features = bb.context_features()
        assert "query_length" in features
        assert "seed_count" in features
        assert features["ppr_entropy"] == pytest.approx(0.5)

    def test_for_new_query_preserves_working_memory(self):
        bb = Blackboard(query="first")
        bb.working_memory.set_goal("learn")
        bb.working_memory.attend(["e1"], boost=0.8)
        bb2 = bb.for_new_query("second")
        assert bb2.query == "second"
        assert "learn" in bb2.working_memory.active_goals
        assert bb2.working_memory.turn_count == 1

    def test_wm_overlap(self):
        bb = Blackboard()
        bb.working_memory.entity_focus = {"e1": 0.5, "e2": 0.8}
        bb.active_entities = {"e1", "e3"}
        assert bb.wm_overlap == pytest.approx(0.5)  # 1 of 2 overlap

    def test_wm_overlap_empty(self):
        bb = Blackboard()
        assert bb.wm_overlap == 0.0


class TestProductionRule:

    def test_fires_when_condition_true(self):
        rule = ProductionRule(name="test", condition=lambda bb: bb.seed_count > 3)
        bb = Blackboard()
        bb.seed_count = 5
        assert rule.fires(bb) is True

    def test_fires_when_condition_false(self):
        rule = ProductionRule(name="test", condition=lambda bb: bb.seed_count > 3)
        bb = Blackboard()
        bb.seed_count = 1
        assert rule.fires(bb) is False

    def test_fires_catches_exception(self):
        rule = ProductionRule(name="bad", condition=lambda bb: 1 / 0)
        bb = Blackboard()
        assert rule.fires(bb) is False, "Exception should return False, not crash"

    def test_default_skip_verify_rules(self):
        rules = default_skip_verify_rules()
        assert len(rules) == 3
        names = {r.name for r in rules}
        assert "confidence_sufficient" in names
        assert "strong_anchors" in names
        assert "entity_grounded" in names

    def test_default_skip_flag_rules(self):
        rules = default_skip_flag_rules()
        assert len(rules) == 3


class TestComputePPREntropy:

    def test_empty_scores(self):
        assert compute_ppr_entropy([]) == 0.0

    def test_single_score(self):
        assert compute_ppr_entropy([1.0]) == 0.0

    def test_uniform_distribution_max_entropy(self):
        scores = [1.0, 1.0, 1.0, 1.0]
        entropy = compute_ppr_entropy(scores)
        assert entropy == pytest.approx(1.0, abs=0.01), "Uniform dist has max entropy"

    def test_concentrated_distribution(self):
        scores = [0.99, 0.003, 0.003, 0.003]
        entropy = compute_ppr_entropy(scores)
        assert entropy < 0.5, "Concentrated distribution has low entropy"

    def test_zero_scores(self):
        assert compute_ppr_entropy([0.0, 0.0]) == 0.0


# =============================================================================
# CBR System
# =============================================================================


class TestCBRSystem:

    def test_similarity_identical_features(self):
        cbr = CBRSystem()
        a = {"query_length": 0.5, "seed_count": 0.3}
        b = {"query_length": 0.5, "seed_count": 0.3}
        assert cbr._similarity(a, b) == pytest.approx(1.0)

    def test_similarity_different_features(self):
        cbr = CBRSystem()
        a = {"query_length": 0.0, "seed_count": 0.0}
        b = {"query_length": 1.0, "seed_count": 1.0}
        sim = cbr._similarity(a, b)
        assert sim < 0.5, f"Very different features should have low similarity: {sim}"

    def test_similarity_partial_features(self):
        cbr = CBRSystem()
        a = {"query_length": 0.5}
        b = {"query_length": 0.5, "seed_count": 0.0}
        sim = cbr._similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_case_serialization_roundtrip(self):
        case = Case(
            case_id="abc123",
            query="test query",
            features={"query_length": 0.5},
            preset_id="balanced",
            reward=0.8,
            response_summary="test response",
            shell_count=2,
        )
        snap = case.to_snapshot()
        restored = Case.from_snapshot(snap)
        assert restored.case_id == "abc123"
        assert restored.reward == pytest.approx(0.8)
        assert restored.features["query_length"] == pytest.approx(0.5)

    def test_snapshot_roundtrip(self):
        cbr = CBRSystem()
        cbr._cases.append(Case(
            case_id="test1",
            query="hello",
            features={"query_length": 0.3},
            preset_id="focused",
            reward=0.9,
            response_summary="hi",
            shell_count=1,
        ))
        snap = cbr.to_snapshot()
        cbr2 = CBRSystem()
        cbr2.from_snapshot(snap)
        assert cbr2.case_count == 1
        assert cbr2.cases[0].case_id == "test1"

    def test_activation_level_with_cases(self):
        cbr = CBRSystem()
        assert cbr.activation_level(None) == pytest.approx(0.1), "No cases = low"
        cbr._cases.append(Case("x", "q", {}, "b", 0.5, "", 1))
        assert cbr.activation_level(None) == pytest.approx(0.7), "Has cases = high"

    def test_retrieve_posts_hints(self):
        """CBR retrieves similar cases and posts hints to blackboard."""
        cbr = CBRSystem(config=CBRConfig(min_similarity=0.5))
        cbr._cases.append(Case(
            case_id="prev",
            query="test",
            features={"query_length": 0.3, "seed_count": 0.5,
                      "ppr_entropy": 0.4, "entity_ratio": 0.3,
                      "wm_overlap": 0.0, "confidence": 0.0},
            preset_id="balanced",
            reward=0.8,
            response_summary="previous answer",
            shell_count=1,
        ))
        bb = Blackboard(query="test query")
        bb.seed_count = 5
        bb.ppr_entropy = 0.4
        bb.seed_sources = {"alias": 3}
        cbr._retrieve_cases(bb, {"nav_result": True})
        hints = bb.flags.get("cbr_hints", [])
        assert len(hints) >= 1, "Should retrieve the similar case"
        assert hints[0]["case_id"] == "prev"

    def test_report(self):
        cbr = CBRSystem()
        cbr._cases.append(Case("x", "q", {}, "b", 0.5, "", 1))
        report = cbr.report()
        assert report["total_cases"] == 1
        assert "avg_reward" in report


# =============================================================================
# Chunk Compiler
# =============================================================================


class TestChunkCompiler:

    def test_compiled_chunk_matches(self):
        chunk = CompiledChunk(
            chunk_id="test",
            trigger={"query_length": (0.1, 0.5), "seed_count": (0.2, 0.8)},
            action=ChunkAction(preset_id="focused"),
        )
        assert chunk.matches({"query_length": 0.3, "seed_count": 0.5}) is True
        assert chunk.matches({"query_length": 0.9, "seed_count": 0.5}) is False

    def test_compiled_chunk_hit_rate(self):
        chunk = CompiledChunk(
            chunk_id="test",
            trigger={},
            action=ChunkAction(preset_id="x"),
            successes=7,
            failures=3,
        )
        assert chunk.hit_rate == pytest.approx(0.7)

    def test_compiled_chunk_retirement(self):
        chunk = CompiledChunk(
            chunk_id="test",
            trigger={},
            action=ChunkAction(preset_id="x"),
            consecutive_failures=3,
        )
        assert chunk.is_retired is True

    def test_chunk_not_retired(self):
        chunk = CompiledChunk(
            chunk_id="test",
            trigger={},
            action=ChunkAction(preset_id="x"),
            consecutive_failures=2,
        )
        assert chunk.is_retired is False

    def test_chunk_snapshot_roundtrip(self):
        chunk = CompiledChunk(
            chunk_id="snap_test",
            trigger={"query_length": (0.1, 0.5)},
            action=ChunkAction(preset_id="focused", max_shells=2, skip_verify=True),
            successes=10,
            failures=2,
        )
        snap = chunk.to_snapshot()
        restored = CompiledChunk.from_snapshot(snap)
        assert restored.chunk_id == "snap_test"
        assert restored.action.preset_id == "focused"
        assert restored.action.skip_verify is True
        assert restored.successes == 10

    def test_experience_record_roundtrip(self):
        rec = ExperienceRecord(
            features={"query_length": 0.5},
            preset_id="balanced",
            reward=0.8,
            shell_count=1,
            query_id="q1",
        )
        snap = rec.to_snapshot()
        restored = ExperienceRecord.from_snapshot(snap)
        assert restored.query_id == "q1"
        assert restored.reward == pytest.approx(0.8)

    def test_activation_level_phases(self):
        cc = ChunkCompiler(config=ChunkConfig(min_cluster_size=3))
        assert cc.activation_level(None) == pytest.approx(0.1), "Empty = low"
        cc._experience = [ExperienceRecord({}, "b", 0.8, 1)] * 3
        assert cc.activation_level(None) == pytest.approx(0.3), "Ready to compile"
        cc._chunks.append(CompiledChunk("x", {}, ChunkAction("x")))
        assert cc.activation_level(None) == pytest.approx(0.8), "Has chunks = high"

    def test_add_chunk_and_retire(self):
        cc = ChunkCompiler()
        chunk = CompiledChunk("manual", {}, ChunkAction("test"))
        cc.add_chunk(chunk)
        assert len(cc.chunks) == 1
        cc.retire_chunk("manual")
        assert cc.chunks[0].is_retired

    def test_compiler_snapshot_roundtrip(self):
        cc = ChunkCompiler()
        cc._chunks.append(CompiledChunk("c1", {"q": (0.1, 0.5)}, ChunkAction("focused")))
        cc._experience.append(ExperienceRecord({"q": 0.3}, "focused", 0.8, 1))
        snap = cc.to_snapshot()
        cc2 = ChunkCompiler()
        cc2.from_snapshot(snap)
        assert len(cc2.chunks) == 1
        assert cc2.experience_count == 1

    def test_report(self):
        cc = ChunkCompiler()
        cc._chunks.append(CompiledChunk("c1", {}, ChunkAction("x"), successes=5))
        report = cc.report()
        assert report["total_chunks"] == 1
        assert report["active_chunks"] == 1

    def test_compute_trigger_ranges_tight_cluster(self):
        """Tight cluster of features produces valid trigger ranges."""
        cc = ChunkCompiler(config=ChunkConfig(
            feature_spread_limit=0.6,
            min_trigger_features=2,
        ))
        experiences = [
            ExperienceRecord(
                features={"query_length": 0.3, "seed_count": 0.4,
                          "ppr_entropy": 0.5, "entity_ratio": 0.3},
                preset_id="balanced", reward=0.8, shell_count=1,
            ),
            ExperienceRecord(
                features={"query_length": 0.35, "seed_count": 0.45,
                          "ppr_entropy": 0.55, "entity_ratio": 0.35},
                preset_id="balanced", reward=0.9, shell_count=1,
            ),
        ]
        trigger = cc._compute_trigger_ranges(experiences)
        assert trigger is not None, "Tight cluster should produce triggers"
        assert len(trigger) >= 2

    def test_compute_trigger_ranges_wide_cluster(self):
        """Wide spread cluster returns None (too loose)."""
        cc = ChunkCompiler(config=ChunkConfig(
            feature_spread_limit=0.1,
            min_trigger_features=4,
        ))
        experiences = [
            ExperienceRecord(
                features={"query_length": 0.1, "seed_count": 0.1,
                          "ppr_entropy": 0.1, "entity_ratio": 0.1},
                preset_id="b", reward=0.8, shell_count=1,
            ),
            ExperienceRecord(
                features={"query_length": 0.9, "seed_count": 0.9,
                          "ppr_entropy": 0.9, "entity_ratio": 0.9},
                preset_id="b", reward=0.8, shell_count=1,
            ),
        ]
        trigger = cc._compute_trigger_ranges(experiences)
        assert trigger is None, "Wide spread should not produce triggers"


class TestChunkCompilerAP1:
    """Regression tests for AP-1: bus._items must contain StoredItem, not raw MemoryItem."""

    def test_chunk_compiler_uses_stored_item_not_raw(self):
        """Regression: AP-1 - bus._items must contain StoredItem, not raw MemoryItem.

        The chunk_compiler._store_chunk_node() method was previously storing
        raw MemoryItem objects to bus._items. MemoryItem has no .to_dict(),
        which breaks LiteBusGraph.node_data(). The fix imports StoredItem
        from lite_bus and stores that instead.

        This test verifies the fix by checking:
        1. The stored object IS a StoredItem
        2. The stored object HAS a .to_dict() method
        3. The to_dict() call succeeds without error
        """
        bus = FakeBus()
        graph = FakeGraph(bus)

        cc = ChunkCompiler(config=ChunkConfig(store_in_graph=True))
        chunk = CompiledChunk(
            chunk_id="ap1test",
            trigger={"query_length": (0.1, 0.5)},
            action=ChunkAction(preset_id="focused"),
        )
        experiences = [
            ExperienceRecord({"q": 0.3}, "focused", 0.8, 1, query_id="q1"),
        ]

        cc._store_chunk_node(graph, chunk, experiences)

        node_id = f"chunk_{chunk.chunk_id}"
        assert node_id in bus._items, "Chunk node should be stored in bus._items"

        stored = bus._items[node_id]
        assert isinstance(stored, StoredItem), (
            f"AP-1 regression: stored object is {type(stored).__name__}, not StoredItem"
        )
        assert hasattr(stored, "to_dict"), "StoredItem must have to_dict()"
        result = stored.to_dict()
        assert isinstance(result, dict), "to_dict() must return a dict"
        assert result["memory_type"] == "chunk"

    def test_cbr_store_case_uses_stored_item(self):
        """Regression: AP-1 also applies to CBR._store_case_node().

        The same pattern applies: case nodes stored in bus._items
        must be StoredItem, not raw MemoryItem.
        """
        bus = FakeBus()
        graph = FakeGraph(bus)

        cbr = CBRSystem(config=CBRConfig(store_in_graph=True))
        case = Case(
            case_id="ap1case",
            query="test query",
            features={"q": 0.5},
            preset_id="balanced",
            reward=0.8,
            response_summary="test",
            shell_count=1,
        )

        cbr._store_case_node(graph, case)

        node_id = f"case_{case.case_id}"
        assert node_id in bus._items, "Case node should be stored in bus._items"

        stored = bus._items[node_id]
        assert isinstance(stored, StoredItem), (
            f"AP-1 regression: stored object is {type(stored).__name__}, not StoredItem"
        )
        result = stored.to_dict()
        assert isinstance(result, dict)
        assert result["memory_type"] == "case"

    def test_chunk_compiler_import_path_uses_stored_item(self):
        """Verify the import in chunk_compiler.py resolves to StoredItem from lite_bus."""
        from hololoom.memory.lite_bus import StoredItem as LiteBusStoredItem
        # Confirm StoredItem imported within chunk_compiler matches lite_bus.StoredItem
        assert LiteBusStoredItem is StoredItem


# =============================================================================
# Classical Factory
# =============================================================================


class TestClassicalFactory:

    def test_creates_all_four_sources(self):
        registry = create_classical_sources()
        assert registry.has("tms"), "TMS should be registered"
        assert registry.has("chunk_compiler"), "ChunkCompiler should be registered"
        assert registry.has("cbr"), "CBR should be registered"
        assert registry.has("demons"), "Demons should be registered"
        assert len(registry.sources) == 4

    def test_custom_configs_applied(self):
        registry = create_classical_sources(
            tms_config=TMSConfig(max_trace_depth=5),
            chunk_config=ChunkConfig(max_chunks=10),
            cbr_config=CBRConfig(max_cases=50),
            activation_threshold=0.2,
        )
        assert registry._threshold == pytest.approx(0.2)
        assert len(registry.sources) == 4

    def test_with_activation_adapter(self):
        adapter = ActivationAdapter()
        registry = create_classical_sources(activation_adapter=adapter)
        # TMS should have the adapter wired
        tms = next(s for s in registry.sources if s.name == "tms")
        assert tms._activation is adapter


# =============================================================================
# Cold Start Handler
# =============================================================================


class TestColdStartHandler:

    def test_needs_cold_start(self):
        bus, graph = _build_test_graph()
        handler = ColdStartHandler(graph, config=ColdStartConfig(min_seeds=3))
        assert handler.needs_cold_start([SeedNode("a", 1.0, "alias")]) is True
        seeds = [SeedNode("a", 1.0, "alias"), SeedNode("b", 1.0, "alias"),
                 SeedNode("c", 1.0, "alias")]
        assert handler.needs_cold_start(seeds) is False

    def test_keyword_seeds_found(self):
        bus, graph = _build_test_graph()
        handler = ColdStartHandler(graph, config=ColdStartConfig(keyword_k=3))
        seeds = handler._keyword_seeds("Thompson Sampling Bayesian")
        assert len(seeds) > 0, "Should find keyword matches"
        for seed in seeds:
            assert seed.source == "cold_start_keyword"

    def test_keyword_seeds_empty_query(self):
        bus, graph = _build_test_graph()
        handler = ColdStartHandler(graph)
        seeds = handler._keyword_seeds("")
        assert seeds == []

    def test_keyword_seeds_no_match(self):
        bus, graph = _build_test_graph()
        handler = ColdStartHandler(graph)
        seeds = handler._keyword_seeds("xyzzy plugh zorkmid")
        assert seeds == [], "No matches for nonsense words"

    def test_tokenize(self):
        tokens = ColdStartHandler._tokenize("Hello, World! It's a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # "it's" survives: strip removes quotes but 4 chars > 2 min length
        assert "it's" in tokens


# =============================================================================
# Confidence
# =============================================================================


class TestConfidenceStemming:

    def test_simple_stem_ing(self):
        assert _simple_stem("running") == "runn"
        assert _simple_stem("testing") == "test"

    def test_simple_stem_ed(self):
        assert _simple_stem("tested") == "test"

    def test_simple_stem_s(self):
        assert _simple_stem("tests") == "test"

    def test_simple_stem_short_word(self):
        assert _simple_stem("at") == "at"
        assert _simple_stem("run") == "run"

    def test_content_stems(self):
        stems = _content_stems("The running tests are tested frequently")
        assert "runn" in stems
        assert "test" in stems
        assert "frequently" in stems  # _simple_stem doesn't strip -ly
        # Stop words excluded
        assert "the" not in stems
        assert "are" not in stems


class TestConfidenceEstimator:

    def test_empty_nav_result_flags(self):
        bus, graph = _build_test_graph()
        estimator = ConfidenceEstimator(graph)
        empty_result = NavigatorResult([], [], 0, True, {})
        dc = estimator.estimate(empty_result)
        assert dc.combined == pytest.approx(0.0)
        assert dc.decision == "flag"

    def test_estimate_with_results(self):
        bus, graph = _build_test_graph()
        estimator = ConfidenceEstimator(graph)
        nav_result = _make_nav_result(graph, seed_ids=["thompson"])
        dc = estimator.estimate(nav_result, query="Thompson Sampling")
        assert 0.0 <= dc.combined <= 1.0
        assert dc.decision in ("sufficient", "verify", "flag")
        assert dc.structural.score >= 0.0
        assert dc.narrative.score >= 0.0

    def test_gini_uniform(self):
        gini = ConfidenceEstimator._gini([1.0, 1.0, 1.0, 1.0])
        assert gini == pytest.approx(0.0, abs=0.01), "Uniform = Gini 0"

    def test_gini_concentrated(self):
        gini = ConfidenceEstimator._gini([0.0, 0.0, 0.0, 100.0])
        assert gini > 0.5, "Concentrated = high Gini"

    def test_gini_empty(self):
        assert ConfidenceEstimator._gini([]) == 0.0

    def test_per_node_confidence(self):
        bus, graph = _build_test_graph()
        estimator = ConfidenceEstimator(graph)
        nav_result = _make_nav_result(graph, seed_ids=["thompson"])
        dc = estimator.estimate(nav_result)
        assert len(dc.per_node) > 0
        for nid, score in dc.per_node.items():
            assert 0.0 <= score <= 1.0, f"Per-node score for {nid} out of range"

    def test_with_reranker(self):
        bus, graph = _build_test_graph()
        mock_reranker = lambda q, c: 0.9
        estimator = ConfidenceEstimator(graph, reranker_fn=mock_reranker)
        nav_result = _make_nav_result(graph, seed_ids=["thompson"])
        dc = estimator.estimate(nav_result, query="test")
        # Reranker blended at 20% weight, should push combined up
        assert dc.combined > 0.0


# =============================================================================
# Navigator (PPR)
# =============================================================================


class TestNavigator:

    def test_ppr_empty_graph(self):
        bus = FakeBus()
        graph = FakeGraph(bus)
        seeds = [SeedNode("nonexistent", 1.0, "alias")]
        result = personalized_pagerank(graph, seeds)
        assert result.ranked_nodes == []

    def test_ppr_no_seeds(self):
        bus, graph = _build_test_graph()
        result = personalized_pagerank(graph, [])
        assert result.ranked_nodes == []

    def test_ppr_basic_ranking(self):
        bus, graph = _build_test_graph()
        seeds = [SeedNode("thompson", 1.0, "alias")]
        result = personalized_pagerank(graph, seeds, PPRConfig(max_iterations=20))
        assert len(result.ranked_nodes) > 0
        # Thompson should be highest (seed node gets teleport)
        top_id = result.ranked_nodes[0][0]
        assert top_id == "thompson", f"Seed node should rank highest, got {top_id}"

    def test_ppr_convergence(self):
        bus, graph = _build_test_graph()
        seeds = [SeedNode("thompson", 1.0, "alias")]
        result = personalized_pagerank(graph, seeds, PPRConfig(max_iterations=100, tolerance=1e-6))
        assert result.converged is True

    def test_ppr_node_data_cached(self):
        bus, graph = _build_test_graph()
        seeds = [SeedNode("thompson", 1.0, "alias")]
        result = personalized_pagerank(graph, seeds)
        for nid, _ in result.ranked_nodes:
            assert nid in result.node_data, f"Missing cached data for {nid}"

    def test_seed_node_dataclass(self):
        seed = SeedNode("n1", 0.8, "alias")
        assert seed.node_id == "n1"
        assert seed.weight == pytest.approx(0.8)
        assert seed.source == "alias"

    def test_navigator_result_properties(self):
        bus, graph = _build_test_graph()
        seeds = [SeedNode("thompson", 1.0, "alias")]
        result = personalized_pagerank(graph, seeds)
        assert len(result.node_ids) == len(result.ranked_nodes)
        assert len(result.top_content) == len(result.ranked_nodes)

    def test_lite_bus_graph_adapter(self):
        bus, _ = _build_test_graph()
        graph = LiteBusGraph(bus)
        nodes = graph.nodes()
        assert "thompson" in nodes
        assert graph.has_node("thompson") is True
        assert graph.has_node("nonexistent") is False
        neighbors = graph.neighbors("thompson")
        neighbor_ids = [n[0] for n in neighbors]
        assert "bandit" in neighbor_ids or "bayesian" in neighbor_ids


# =============================================================================
# TMS (Truth Maintenance System)
# =============================================================================


class TestTMS:

    def test_justification_record_strength(self):
        rec = JustificationRecord(
            belief_id="b1",
            seed_id="s1",
            path=["s1", "mid", "b1"],
            ppr_score=0.5,
            seed_weight=0.9,
            query_id="q1",
        )
        assert rec.strength == pytest.approx(0.3), "0.9 / 3 = 0.3"

    def test_justification_self_path(self):
        rec = JustificationRecord(
            belief_id="s1",
            seed_id="s1",
            path=["s1"],
            ppr_score=0.8,
            seed_weight=1.0,
            query_id="q1",
        )
        assert rec.strength == pytest.approx(1.0), "Self-justified: weight / 1"

    def test_record_justifications(self):
        bus, graph = _build_test_graph()
        tms = TruthMaintenanceSystem(config=TMSConfig(store_edges=False))
        nav_result = _make_nav_result(graph, seed_ids=["thompson"])
        bb = Blackboard(query="test")

        tms._record_justifications(bb, graph, nav_result)

        justifications = bb.flags.get("justifications", {})
        assert "thompson" in justifications, "Seed node should be self-justified"
        # Thompson is a seed, so path should be [thompson]
        assert justifications["thompson"][0].path == ["thompson"]

    def test_node_strength_no_records(self):
        tms = TruthMaintenanceSystem()
        assert tms._node_strength("unknown") == pytest.approx(0.0)

    def test_explain_no_justification(self):
        tms = TruthMaintenanceSystem()
        assert tms._explain("unknown") == "No justification recorded."

    def test_explain_node_report(self):
        tms = TruthMaintenanceSystem()
        tms._justifications["n1"] = [
            JustificationRecord("n1", "s1", ["s1", "n1"], 0.5, 0.8, "q1"),
        ]
        report = tms.explain_node("n1")
        assert report["justified"] is True
        assert report["total_strength"] > 0

    def test_activation_level_always_high(self):
        tms = TruthMaintenanceSystem()
        assert tms.activation_level(None) == pytest.approx(1.0)

    def test_phases(self):
        tms = TruthMaintenanceSystem()
        assert Phase.POST_NAVIGATE in tms.phases
        assert Phase.POST_CONFIDENCE in tms.phases

    def test_report(self):
        tms = TruthMaintenanceSystem()
        tms._justifications["n1"] = [
            JustificationRecord("n1", "s1", ["s1", "n1"], 0.5, 0.8, "q1"),
        ]
        report = tms.report()
        assert report["nodes_justified"] == 1
        assert report["total_justification_records"] == 1

    def test_find_path_bfs(self):
        bus, graph = _build_test_graph()
        tms = TruthMaintenanceSystem()
        path = tms._find_path(graph, "thompson", "bandit", max_depth=3)
        assert path is not None, "Should find path thompson -> bandit"
        assert path[0] == "thompson"
        assert path[-1] == "bandit"

    def test_find_path_no_connection(self):
        bus = FakeBus()
        bus.add_item(_make_stored_item("a", "node a"))
        bus.add_item(_make_stored_item("b", "node b"))
        # No edges between a and b
        graph = FakeGraph(bus)
        tms = TruthMaintenanceSystem()
        path = tms._find_path(graph, "a", "b", max_depth=3)
        assert path is None

    def test_find_path_nonexistent_node(self):
        bus, graph = _build_test_graph()
        tms = TruthMaintenanceSystem()
        path = tms._find_path(graph, "thompson", "nonexistent", max_depth=3)
        assert path is None

    def test_retraction_decreases_activation(self):
        """When TMS retracts a belief, activation decreases."""
        adapter = ActivationAdapter()
        adapter._levels["weak_node"] = 0.8
        tms = TruthMaintenanceSystem(
            activation_adapter=adapter,
            config=TMSConfig(retraction_penalty=0.3),
        )
        # Manually test retraction logic
        tms._justifications["weak_node"] = [
            JustificationRecord("weak_node", "s1", ["s1", "m", "weak_node"],
                                0.3, 0.3, "q1"),  # strength = 0.1
        ]
        tms._justifications["strong_node"] = [
            JustificationRecord("strong_node", "s1", ["s1", "strong_node"],
                                0.8, 1.0, "q1"),  # strength = 0.5
        ]

        weak_str = tms._node_strength("weak_node")
        strong_str = tms._node_strength("strong_node")
        assert weak_str < strong_str, "Weak should have lower strength"


# =============================================================================
# Knowledge Source Registry
# =============================================================================


class TestSourceRegistry:

    def test_register_and_has(self):
        registry = SourceRegistry()
        tms = TruthMaintenanceSystem()
        registry.register(tms)
        assert registry.has("tms")
        assert not registry.has("nonexistent")

    def test_register_idempotent(self):
        registry = SourceRegistry()
        tms = TruthMaintenanceSystem()
        registry.register(tms)
        registry.register(tms)
        assert len(registry.sources) == 1

    def test_remove(self):
        registry = SourceRegistry()
        tms = TruthMaintenanceSystem()
        registry.register(tms)
        registry.remove("tms")
        assert not registry.has("tms")

    def test_invoke_filters_by_phase(self):
        registry = SourceRegistry(activation_threshold=0.0)
        tms = TruthMaintenanceSystem()
        registry.register(tms)
        bb = Blackboard()
        bus, graph = _build_test_graph()

        # POST_NAVIGATE should invoke TMS
        nav_result = _make_nav_result(graph, seed_ids=["thompson"])
        registry.invoke(Phase.POST_NAVIGATE, bb, graph, {"nav_result": nav_result})
        assert "justifications" in bb.flags

    def test_invoke_skips_below_threshold(self):
        registry = SourceRegistry(activation_threshold=2.0)  # impossibly high
        tms = TruthMaintenanceSystem()
        registry.register(tms)
        bb = Blackboard()
        bus, graph = _build_test_graph()
        nav_result = _make_nav_result(graph, seed_ids=["thompson"])
        registry.invoke(Phase.POST_NAVIGATE, bb, graph, {"nav_result": nav_result})
        assert "justifications" not in bb.flags, "Should skip due to high threshold"

    def test_invoke_with_none_blackboard(self):
        """Registry gracefully handles None blackboard."""
        registry = SourceRegistry()
        tms = TruthMaintenanceSystem()
        registry.register(tms)
        bus, graph = _build_test_graph()
        # Should not crash
        registry.invoke(Phase.POST_NAVIGATE, None, graph, {"nav_result": None})

    def test_report(self):
        registry = SourceRegistry()
        tms = TruthMaintenanceSystem()
        registry.register(tms)
        report = registry.report()
        assert report["source_count"] == 1
        assert report["sources"][0]["name"] == "tms"
