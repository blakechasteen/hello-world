"""
Unit tests for hololoom.core.memory.v2 — training, confidence, chunk_compiler.

Target: 60%+ coverage on training.py (452 stmts), confidence.py (314 stmts),
chunk_compiler.py (231 stmts).
"""

import json
import math
import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import hololoom  # noqa: F401 — installs meta path hook

from hololoom.core.memory.v2.orchestrator import ShellConfig, ShellType, DEFAULT_SHELLS
from hololoom.core.memory.v2.navigator import NavigatorResult, SeedNode, GraphBackend
from hololoom.core.memory.v2.knowledge_source import Phase


# =============================================================================
# Helpers / Fixtures
# =============================================================================

class FakeGraph:
    """Minimal GraphBackend for testing."""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[str, List[Tuple[str, str, float]]] = {}

    def add_node(self, nid: str, **data):
        self._nodes[nid] = data

    def add_edge(self, src: str, tgt: str, rel: str = "RELATED", weight: float = 1.0):
        self._edges.setdefault(src, []).append((tgt, rel, weight))

    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def neighbors(self, node_id: str) -> List[Tuple[str, str, float]]:
        return self._edges.get(node_id, [])

    def node_data(self, node_id: str) -> Dict[str, Any]:
        return self._nodes.get(node_id, {})

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes


def make_nav_result(
    ranked_nodes: List[Tuple[str, float]],
    node_data: Dict[str, Dict[str, Any]],
    seed_nodes: Optional[List[SeedNode]] = None,
) -> NavigatorResult:
    """Build a NavigatorResult for testing."""
    return NavigatorResult(
        ranked_nodes=ranked_nodes,
        node_data=node_data,
        seed_nodes=seed_nodes or [],
        iterations=5,
        converged=True,
    )


# =============================================================================
# CONFIDENCE TESTS
# =============================================================================

from hololoom.core.memory.v2.confidence import (
    ConfidenceEstimator,
    ConfidenceConfig,
    DualConfidence,
    StructuralConfidence,
    NarrativeConfidence,
    _simple_stem,
    _content_stems,
    find_contradicting_pairs,
)


class TestSimpleStem:
    def test_short_word_unchanged(self):
        assert _simple_stem("the") == "the"

    def test_strip_ing(self):
        assert _simple_stem("running") == "runn"

    def test_strip_ed(self):
        assert _simple_stem("walked") == "walk"

    def test_strip_s(self):
        assert _simple_stem("cats") == "cat"

    def test_no_strip_ss(self):
        assert _simple_stem("grass") == "grass"

    def test_short_ing_unchanged(self):
        # "bring" has len 5, not > 5
        assert _simple_stem("bring") == "bring"

    def test_short_ed_unchanged(self):
        # "used" has len 4, not > 4
        assert _simple_stem("used") == "used"


class TestContentStems:
    def test_removes_stop_words(self):
        stems = _content_stems("the cat is on the mat")
        assert "the" not in stems
        assert "cat" in stems
        assert "mat" in stems

    def test_removes_short_words(self):
        stems = _content_stems("I am a big dog")
        # "am" and "a" filtered (len <= 2 or stop words)
        assert "big" in stems
        assert "dog" in stems

    def test_stems_words(self):
        stems = _content_stems("running dogs walked")
        assert "runn" in stems
        assert "dog" in stems
        assert "walk" in stems


class TestFindContradictingPairs:
    def test_no_contradictions_no_negation(self):
        ranked = [("a", 1.0), ("b", 0.8)]
        data = {
            "a": {"content": "The sky is blue and clear"},
            "b": {"content": "The sky is bright and clear"},
        }
        pairs = find_contradicting_pairs(ranked, data)
        assert pairs == []

    def test_no_contradiction_same_negation(self):
        ranked = [("a", 1.0), ("b", 0.8)]
        data = {
            "a": {"content": "Python is not slow for web development"},
            "b": {"content": "Python is not bad for web development"},
        }
        pairs = find_contradicting_pairs(ranked, data)
        assert pairs == []

    def test_contradiction_with_graph_entities(self):
        graph = FakeGraph()
        graph.add_node("entity1", memory_type="entity", content="Python language")
        graph.add_node("a", memory_type="fact", content="python language is fast for computation")
        graph.add_node("b", memory_type="fact", content="python language is not fast for computation")
        graph.add_edge("a", "entity1", "MENTIONS")
        graph.add_edge("b", "entity1", "MENTIONS")

        ranked = [("a", 1.0), ("b", 0.8)]
        data = {
            "a": {"content": "python language is fast for computation"},
            "b": {"content": "python language is not fast for computation"},
        }
        pairs = find_contradicting_pairs(ranked, data, graph=graph)
        assert len(pairs) == 1
        assert pairs[0] == ("a", "b")

    def test_empty_content_skipped(self):
        ranked = [("a", 1.0), ("b", 0.8)]
        data = {"a": {"content": ""}, "b": {"content": "something"}}
        pairs = find_contradicting_pairs(ranked, data)
        assert pairs == []

    def test_fallback_entity_mention_detection(self):
        graph = FakeGraph()
        graph.add_node("entity1", memory_type="entity", content="machine learning")
        # Nodes are NOT connected to entity via edges, but mention it in content
        ranked = [("a", 1.0), ("b", 0.8)]
        data = {
            "a": {"content": "machine learning algorithms perform well on structured data"},
            "b": {"content": "machine learning algorithms do not perform well on structured data"},
        }
        pairs = find_contradicting_pairs(ranked, data, graph=graph)
        assert len(pairs) == 1


class TestConfidenceEstimator:
    def _make_graph_with_nodes(self):
        g = FakeGraph()
        g.add_node("e1", memory_type="entity", content="TestEntity")
        g.add_node("n1", memory_type="fact", content="A" * 30)
        g.add_node("n2", memory_type="episode", content="B" * 30)
        g.add_node("n3", memory_type="fact", content="C" * 30)
        g.add_edge("n1", "e1", "MENTIONS")
        g.add_edge("n2", "e1", "MENTIONS")
        g.add_edge("n1", "n2", "RELATED")
        return g

    def test_empty_result_flags(self):
        g = FakeGraph()
        est = ConfidenceEstimator(graph=g)
        result = est.estimate(make_nav_result([], {}))
        assert result.combined == 0.0
        assert result.decision == "flag"
        assert result.per_node == {}

    def test_estimate_with_diverse_types(self):
        g = self._make_graph_with_nodes()
        est = ConfidenceEstimator(graph=g)
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.8), ("n2", 0.4), ("n3", 0.2)],
            node_data={
                "n1": {"content": "A" * 30, "memory_type": "fact"},
                "n2": {"content": "B" * 30, "memory_type": "episode"},
                "n3": {"content": "C" * 30, "memory_type": "fact"},
            },
            seed_nodes=[SeedNode(node_id="n1", weight=1.0, source="keyword")],
        )
        result = est.estimate(nav)
        assert 0.0 <= result.combined <= 1.0
        assert result.decision in ("sufficient", "verify", "flag")
        assert result.structural.score >= 0.0
        assert result.narrative.score >= 0.0

    def test_structural_seed_coverage(self):
        g = FakeGraph()
        g.add_node("s1", memory_type="fact", content="X" * 30)
        g.add_node("s2", memory_type="fact", content="Y" * 30)
        est = ConfidenceEstimator(graph=g)
        nav = make_nav_result(
            ranked_nodes=[("s1", 0.5)],
            node_data={"s1": {"content": "X" * 30}},
            seed_nodes=[
                SeedNode(node_id="s1", weight=1.0, source="keyword"),
                SeedNode(node_id="s2", weight=1.0, source="keyword"),
            ],
        )
        result = est.estimate(nav)
        # Only 1 of 2 seeds in results
        assert result.structural.seed_coverage == 0.5

    def test_gini_uniform(self):
        # Equal values → Gini near 0
        g = ConfidenceEstimator._gini([1.0, 1.0, 1.0, 1.0])
        assert g == pytest.approx(0.0, abs=0.01)

    def test_gini_concentrated(self):
        # One dominant → Gini near 1
        g = ConfidenceEstimator._gini([0.0, 0.0, 0.0, 100.0])
        assert g > 0.5

    def test_gini_empty(self):
        assert ConfidenceEstimator._gini([]) == 0.0

    def test_gini_single(self):
        assert ConfidenceEstimator._gini([5.0]) == 0.0

    def test_gini_all_zeros(self):
        assert ConfidenceEstimator._gini([0.0, 0.0]) == 0.0

    def test_decision_sufficient(self):
        g = self._make_graph_with_nodes()
        cfg = ConfidenceConfig(high_threshold=0.01, low_threshold=0.0)
        est = ConfidenceEstimator(graph=g, config=cfg)
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.8), ("n2", 0.6)],
            node_data={
                "n1": {"content": "A" * 30, "memory_type": "fact"},
                "n2": {"content": "B" * 30, "memory_type": "episode"},
            },
            seed_nodes=[SeedNode(node_id="n1", weight=1.0, source="keyword")],
        )
        result = est.estimate(nav)
        assert result.decision == "sufficient"

    def test_decision_flag_low_confidence(self):
        g = FakeGraph()
        g.add_node("n1", memory_type="fact", content="short")
        cfg = ConfidenceConfig(high_threshold=0.99, low_threshold=0.99)
        est = ConfidenceEstimator(graph=g, config=cfg)
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.1)],
            node_data={"n1": {"content": "short"}},
        )
        result = est.estimate(nav)
        assert result.decision == "flag"

    def test_per_node_confidence(self):
        g = self._make_graph_with_nodes()
        est = ConfidenceEstimator(graph=g)
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.9), ("n2", 0.3)],
            node_data={
                "n1": {"content": "A" * 30, "memory_type": "fact"},
                "n2": {"content": "B" * 30, "memory_type": "episode"},
            },
        )
        result = est.estimate(nav)
        assert "n1" in result.per_node
        assert "n2" in result.per_node
        # n1 has higher PPR, should have higher per-node confidence
        assert result.per_node["n1"] >= result.per_node["n2"]

    def test_reranker_blended(self):
        g = FakeGraph()
        g.add_node("n1", memory_type="fact", content="A" * 30)
        reranker = lambda q, t: 0.9
        est = ConfidenceEstimator(graph=g, reranker_fn=reranker)
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.5)],
            node_data={"n1": {"content": "A" * 30}},
        )
        result_with = est.estimate(nav, query="test query")

        est2 = ConfidenceEstimator(graph=g)
        result_without = est2.estimate(nav)

        # Reranker should modify the combined score
        assert result_with.combined != result_without.combined

    def test_reranker_returns_none_without_query(self):
        g = FakeGraph()
        g.add_node("n1", memory_type="fact", content="A" * 30)
        reranker = lambda q, t: 0.9
        est = ConfidenceEstimator(graph=g, reranker_fn=reranker)
        # No query → reranker should return None
        score = est._reranker_confidence("", make_nav_result([("n1", 0.5)], {"n1": {"content": "A" * 30}}))
        assert score is None

    def test_reranker_handles_exception(self):
        g = FakeGraph()
        g.add_node("n1", memory_type="fact", content="A" * 30)

        def bad_reranker(q, t):
            raise ValueError("boom")

        est = ConfidenceEstimator(graph=g, reranker_fn=bad_reranker)
        score = est._reranker_confidence("test", make_nav_result(
            [("n1", 0.5)], {"n1": {"content": "A" * 30}}
        ))
        assert score is None

    def test_narrative_topical_coherence(self):
        g = FakeGraph()
        g.add_node("n1", memory_type="fact", content="X" * 30)
        g.add_node("n2", memory_type="fact", content="X" * 30)
        est = ConfidenceEstimator(graph=g)
        # Identical content → high topical coherence
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.5), ("n2", 0.5)],
            node_data={
                "n1": {"content": "machine learning algorithms for classification tasks"},
                "n2": {"content": "machine learning algorithms for classification problems"},
            },
        )
        result = est.estimate(nav)
        assert result.narrative.topical_coherence > 0.0

    def test_narrative_source_diversity(self):
        g = FakeGraph()
        g.add_node("n1", memory_type="fact", content="A" * 30)
        g.add_node("n2", memory_type="episode", content="B" * 30)
        g.add_node("n3", memory_type="entity", content="C" * 30)
        est = ConfidenceEstimator(graph=g)
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.5), ("n2", 0.4), ("n3", 0.3)],
            node_data={
                "n1": {"content": "A" * 30, "memory_type": "fact"},
                "n2": {"content": "B" * 30, "memory_type": "episode"},
                "n3": {"content": "C" * 30, "memory_type": "entity"},
            },
        )
        result = est.estimate(nav)
        # 3 distinct types → source diversity > 0
        assert result.narrative.temporal_order > 0.0

    def test_narrative_contradiction_detected(self):
        g = FakeGraph()
        g.add_node("entity1", memory_type="entity", content="neural networks")
        g.add_node("n1", memory_type="fact", content="neural networks are efficient for image tasks")
        g.add_node("n2", memory_type="fact", content="neural networks are not efficient for image tasks")
        g.add_edge("n1", "entity1", "MENTIONS")
        g.add_edge("n2", "entity1", "MENTIONS")
        est = ConfidenceEstimator(graph=g)
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.5), ("n2", 0.4)],
            node_data={
                "n1": {"content": "neural networks are efficient for image tasks"},
                "n2": {"content": "neural networks are not efficient for image tasks"},
            },
        )
        result = est.estimate(nav)
        assert result.narrative.contradiction_count >= 1

    def test_structural_entity_linkage(self):
        g = FakeGraph()
        g.add_node("e1", memory_type="entity", content="Entity1")
        g.add_node("n1", memory_type="fact", content="X" * 30)
        g.add_node("n2", memory_type="fact", content="Y" * 30)
        g.add_edge("n1", "e1", "MENTIONS")
        # n2 is not linked to any entity
        est = ConfidenceEstimator(graph=g)
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.5), ("n2", 0.4)],
            node_data={
                "n1": {"content": "X" * 30},
                "n2": {"content": "Y" * 30},
            },
        )
        result = est.estimate(nav)
        # Only n1 is linked → entity_grounding = 0.5
        assert result.structural.entity_grounding == pytest.approx(0.5)

    def test_structural_connectivity(self):
        g = FakeGraph()
        g.add_node("n1", memory_type="fact", content="X" * 30)
        g.add_node("n2", memory_type="fact", content="Y" * 30)
        g.add_node("n3", memory_type="fact", content="Z" * 30)
        g.add_edge("n1", "n2", "RELATED")
        # n3 not connected to anyone
        est = ConfidenceEstimator(graph=g)
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.5), ("n2", 0.4), ("n3", 0.3)],
            node_data={
                "n1": {"content": "X" * 30},
                "n2": {"content": "Y" * 30},
                "n3": {"content": "Z" * 30},
            },
        )
        result = est.estimate(nav)
        # n1 has neighbor n2 in results → connected
        # n2 is a neighbor of n1 but the edge is one-directional from n1
        # n3 has no connections
        assert result.structural.graph_connectivity > 0.0

    def test_content_density(self):
        g = FakeGraph()
        g.add_node("n1", memory_type="fact", content="X" * 30)
        g.add_node("n2", memory_type="fact", content="Y")  # too short
        est = ConfidenceEstimator(graph=g, config=ConfidenceConfig(min_content_length=20))
        nav = make_nav_result(
            ranked_nodes=[("n1", 0.5), ("n2", 0.4)],
            node_data={
                "n1": {"content": "X" * 30},
                "n2": {"content": "Y"},
            },
        )
        result = est.estimate(nav)
        assert result.structural.content_density == pytest.approx(0.5)


# =============================================================================
# CHUNK COMPILER TESTS
# =============================================================================

from hololoom.core.memory.v2.chunk_compiler import (
    ChunkCompiler,
    ChunkConfig,
    ChunkAction,
    CompiledChunk,
    ExperienceRecord,
)


class TestCompiledChunk:
    def test_matches_in_range(self):
        chunk = CompiledChunk(
            chunk_id="test1",
            trigger={"query_length": (0.1, 0.5), "seed_count": (0.2, 0.6)},
            action=ChunkAction(preset_id="focused"),
        )
        assert chunk.matches({"query_length": 0.3, "seed_count": 0.4})

    def test_matches_out_of_range(self):
        chunk = CompiledChunk(
            chunk_id="test1",
            trigger={"query_length": (0.1, 0.5)},
            action=ChunkAction(preset_id="focused"),
        )
        assert not chunk.matches({"query_length": 0.8})

    def test_matches_missing_feature_defaults_zero(self):
        chunk = CompiledChunk(
            chunk_id="test1",
            trigger={"query_length": (0.1, 0.5)},
            action=ChunkAction(preset_id="focused"),
        )
        # Missing feature defaults to 0.0, which is < 0.1
        assert not chunk.matches({})

    def test_hit_rate(self):
        chunk = CompiledChunk(
            chunk_id="test1",
            trigger={},
            action=ChunkAction(preset_id="balanced"),
            successes=7,
            failures=3,
        )
        assert chunk.hit_rate == pytest.approx(0.7)

    def test_hit_rate_no_evals(self):
        chunk = CompiledChunk(
            chunk_id="test1",
            trigger={},
            action=ChunkAction(preset_id="balanced"),
        )
        assert chunk.hit_rate == 0.0

    def test_is_retired(self):
        chunk = CompiledChunk(
            chunk_id="test1",
            trigger={},
            action=ChunkAction(preset_id="balanced"),
            consecutive_failures=3,
        )
        assert chunk.is_retired

    def test_not_retired(self):
        chunk = CompiledChunk(
            chunk_id="test1",
            trigger={},
            action=ChunkAction(preset_id="balanced"),
            consecutive_failures=2,
        )
        assert not chunk.is_retired

    def test_to_dict(self):
        chunk = CompiledChunk(
            chunk_id="abc",
            trigger={"query_length": (0.1, 0.5)},
            action=ChunkAction(preset_id="focused", max_shells=2, skip_verify=True),
            successes=5,
            failures=1,
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "abc"
        assert d["trigger"]["query_length"] == [0.1, 0.5]
        assert d["action"]["preset_id"] == "focused"
        assert d["hit_rate"] == pytest.approx(5 / 6)

    def test_snapshot_roundtrip(self):
        chunk = CompiledChunk(
            chunk_id="roundtrip",
            trigger={"seed_count": (0.2, 0.8)},
            action=ChunkAction(preset_id="wide", max_shells=3, skip_verify=False),
            node_id="chunk_roundtrip",
            successes=10,
            failures=2,
            consecutive_failures=1,
            last_tuned_at=8,
        )
        snap = chunk.to_snapshot()
        restored = CompiledChunk.from_snapshot(snap)
        assert restored.chunk_id == "roundtrip"
        assert restored.trigger["seed_count"] == (0.2, 0.8)
        assert restored.action.preset_id == "wide"
        assert restored.action.max_shells == 3
        assert restored.successes == 10
        assert restored.failures == 2
        assert restored.consecutive_failures == 1
        assert restored.last_tuned_at == 8


class TestExperienceRecord:
    def test_snapshot_roundtrip(self):
        rec = ExperienceRecord(
            features={"query_length": 0.3, "seed_count": 0.5},
            preset_id="balanced",
            reward=0.8,
            shell_count=2,
            query_id="q1",
        )
        snap = rec.to_snapshot()
        restored = ExperienceRecord.from_snapshot(snap)
        assert restored.features == {"query_length": 0.3, "seed_count": 0.5}
        assert restored.preset_id == "balanced"
        assert restored.reward == 0.8
        assert restored.shell_count == 2
        assert restored.query_id == "q1"

    def test_from_snapshot_defaults(self):
        restored = ExperienceRecord.from_snapshot({})
        assert restored.features == {}
        assert restored.preset_id == "balanced"
        assert restored.reward == 0.0
        assert restored.shell_count == 1


class TestChunkCompiler:
    def test_name_and_phases(self):
        cc = ChunkCompiler()
        assert cc.name == "chunk_compiler"
        assert Phase.PRE_SHELL in cc.phases
        assert Phase.POST_LOOP in cc.phases

    def test_activation_level_no_chunks_no_experience(self):
        cc = ChunkCompiler()
        assert cc.activation_level(None) == pytest.approx(0.1)

    def test_activation_level_with_experience(self):
        cc = ChunkCompiler(config=ChunkConfig(min_cluster_size=3))
        for i in range(3):
            cc._experience.append(ExperienceRecord(
                features={}, preset_id="balanced", reward=0.8, shell_count=1,
            ))
        assert cc.activation_level(None) == pytest.approx(0.3)

    def test_activation_level_with_chunks(self):
        cc = ChunkCompiler()
        cc._chunks.append(CompiledChunk(
            chunk_id="c1", trigger={}, action=ChunkAction(preset_id="focused"),
        ))
        assert cc.activation_level(None) == pytest.approx(0.8)

    def test_add_chunk_and_retire(self):
        cc = ChunkCompiler()
        chunk = CompiledChunk(
            chunk_id="manual", trigger={}, action=ChunkAction(preset_id="balanced"),
        )
        cc.add_chunk(chunk)
        assert len(cc.chunks) == 1
        assert len(cc.active_chunks) == 1

        cc.retire_chunk("manual")
        assert len(cc.chunks) == 1
        assert len(cc.active_chunks) == 0

    def test_retire_nonexistent_chunk(self):
        cc = ChunkCompiler()
        cc.retire_chunk("nonexistent")  # Should not raise

    def test_experience_count(self):
        cc = ChunkCompiler()
        assert cc.experience_count == 0
        cc._experience.append(ExperienceRecord(
            features={}, preset_id="x", reward=0.5, shell_count=1,
        ))
        assert cc.experience_count == 1

    def test_report(self):
        cc = ChunkCompiler()
        cc.add_chunk(CompiledChunk(
            chunk_id="r1", trigger={}, action=ChunkAction(preset_id="balanced"),
        ))
        report = cc.report()
        assert report["total_chunks"] == 1
        assert report["active_chunks"] == 1
        assert report["retired_chunks"] == 0
        assert len(report["chunks"]) == 1

    def test_snapshot_roundtrip(self):
        cc = ChunkCompiler()
        cc.add_chunk(CompiledChunk(
            chunk_id="snap1",
            trigger={"query_length": (0.1, 0.5)},
            action=ChunkAction(preset_id="focused", max_shells=2),
            successes=5,
        ))
        cc._experience.append(ExperienceRecord(
            features={"query_length": 0.3},
            preset_id="focused",
            reward=0.9,
            shell_count=1,
        ))

        snap = cc.to_snapshot()
        cc2 = ChunkCompiler()
        cc2.from_snapshot(snap)
        assert len(cc2.chunks) == 1
        assert cc2.chunks[0].chunk_id == "snap1"
        assert cc2.experience_count == 1

    def test_compute_trigger_ranges_tight_cluster(self):
        cc = ChunkCompiler(config=ChunkConfig(
            feature_spread_limit=0.6,
            min_trigger_features=2,
        ))
        experiences = [
            ExperienceRecord(
                features={"query_length": 0.2, "seed_count": 0.3, "ppr_entropy": 0.4, "entity_ratio": 0.5},
                preset_id="balanced", reward=0.8, shell_count=1,
            ),
            ExperienceRecord(
                features={"query_length": 0.25, "seed_count": 0.35, "ppr_entropy": 0.45, "entity_ratio": 0.55},
                preset_id="balanced", reward=0.9, shell_count=1,
            ),
            ExperienceRecord(
                features={"query_length": 0.22, "seed_count": 0.32, "ppr_entropy": 0.42, "entity_ratio": 0.52},
                preset_id="balanced", reward=0.85, shell_count=1,
            ),
        ]
        trigger = cc._compute_trigger_ranges(experiences)
        assert trigger is not None
        assert len(trigger) >= 2
        for fname, (lo, hi) in trigger.items():
            assert lo >= 0.0
            assert hi <= 1.0
            assert lo < hi

    def test_compute_trigger_ranges_wide_spread_returns_none(self):
        cc = ChunkCompiler(config=ChunkConfig(
            feature_spread_limit=0.1,
            min_trigger_features=2,
        ))
        experiences = [
            ExperienceRecord(
                features={"query_length": 0.0, "seed_count": 0.0, "ppr_entropy": 0.0, "entity_ratio": 0.0},
                preset_id="balanced", reward=0.8, shell_count=1,
            ),
            ExperienceRecord(
                features={"query_length": 1.0, "seed_count": 1.0, "ppr_entropy": 1.0, "entity_ratio": 1.0},
                preset_id="balanced", reward=0.9, shell_count=1,
            ),
        ]
        trigger = cc._compute_trigger_ranges(experiences)
        assert trigger is None

    def test_attempt_compilation_creates_chunk(self):
        cc = ChunkCompiler(config=ChunkConfig(
            min_cluster_size=3,
            feature_spread_limit=0.6,
            min_trigger_features=2,
            store_in_graph=False,
        ))
        # Add enough successful experiences with tight features
        for i in range(5):
            cc._experience.append(ExperienceRecord(
                features={"query_length": 0.3 + i * 0.01, "seed_count": 0.4 + i * 0.01,
                           "ppr_entropy": 0.5, "entity_ratio": 0.6},
                preset_id="focused",
                reward=0.8,
                shell_count=1,
            ))
        cc._attempt_compilation(graph=None)
        assert len(cc.chunks) >= 1
        assert cc.chunks[0].action.preset_id == "focused"

    def test_attempt_compilation_skips_existing_preset(self):
        cc = ChunkCompiler(config=ChunkConfig(
            min_cluster_size=2,
            feature_spread_limit=1.0,
            min_trigger_features=1,
            store_in_graph=False,
        ))
        # Add existing non-retired chunk for "focused"
        cc.add_chunk(CompiledChunk(
            chunk_id="existing",
            trigger={"query_length": (0.0, 1.0)},
            action=ChunkAction(preset_id="focused"),
        ))
        for i in range(3):
            cc._experience.append(ExperienceRecord(
                features={"query_length": 0.3, "seed_count": 0.4,
                           "ppr_entropy": 0.5, "entity_ratio": 0.6},
                preset_id="focused",
                reward=0.8,
                shell_count=1,
            ))
        initial_count = len(cc.chunks)
        cc._attempt_compilation(graph=None)
        assert len(cc.chunks) == initial_count  # No new chunk

    def test_auto_tune_tighten(self):
        cc = ChunkCompiler(config=ChunkConfig(auto_tune_interval=5))
        chunk = CompiledChunk(
            chunk_id="tune1",
            trigger={"query_length": (0.2, 0.8)},
            action=ChunkAction(preset_id="balanced"),
            successes=9,
            failures=1,
            last_tuned_at=0,
        )
        # total = 10, last_tuned_at = 0, diff = 10 >= interval (5)
        # hit_rate = 0.9 >= 0.8 → tighten
        cc._auto_tune_chunk(chunk)
        lo, hi = chunk.trigger["query_length"]
        # Should have shrunk (centroid stays at 0.5)
        assert lo > 0.2
        assert hi < 0.8

    def test_auto_tune_widen(self):
        cc = ChunkCompiler(config=ChunkConfig(auto_tune_interval=5))
        chunk = CompiledChunk(
            chunk_id="tune2",
            trigger={"query_length": (0.3, 0.7)},
            action=ChunkAction(preset_id="balanced"),
            successes=2,
            failures=8,
            last_tuned_at=0,
        )
        # hit_rate = 0.2 < 0.6 → widen
        cc._auto_tune_chunk(chunk)
        lo, hi = chunk.trigger["query_length"]
        assert lo < 0.3
        assert hi > 0.7

    def test_auto_tune_skips_if_interval_not_reached(self):
        cc = ChunkCompiler(config=ChunkConfig(auto_tune_interval=10))
        chunk = CompiledChunk(
            chunk_id="tune3",
            trigger={"query_length": (0.3, 0.7)},
            action=ChunkAction(preset_id="balanced"),
            successes=3,
            failures=1,
            last_tuned_at=0,
        )
        # total = 4, interval = 10 → skip
        cc._auto_tune_chunk(chunk)
        assert chunk.trigger["query_length"] == (0.3, 0.7)  # Unchanged

    def test_auto_tune_recompute_mid_range(self):
        cc = ChunkCompiler(config=ChunkConfig(
            auto_tune_interval=5,
            feature_spread_limit=1.0,
            min_trigger_features=1,
        ))
        chunk = CompiledChunk(
            chunk_id="tune4",
            trigger={"query_length": (0.2, 0.8)},
            action=ChunkAction(preset_id="balanced"),
            successes=4,
            failures=3,
            last_tuned_at=0,
        )
        # hit_rate = 4/7 ≈ 0.57, in mid range → recompute
        # Add matching experiences
        for i in range(3):
            cc._experience.append(ExperienceRecord(
                features={"query_length": 0.4 + i * 0.01, "seed_count": 0.5,
                           "ppr_entropy": 0.5, "entity_ratio": 0.5},
                preset_id="balanced",
                reward=0.8,
                shell_count=1,
            ))
        cc._auto_tune_chunk(chunk)
        assert chunk.last_tuned_at == 7

    def test_try_chunk_match_posts_to_blackboard(self):
        cc = ChunkCompiler()
        chunk = CompiledChunk(
            chunk_id="match1",
            trigger={"query_length": (0.0, 1.0)},
            action=ChunkAction(preset_id="focused", max_shells=2, skip_verify=True),
            successes=8,
            failures=1,
        )
        cc.add_chunk(chunk)

        bb = MagicMock()
        bb.context_features.return_value = {"query_length": 0.5}
        flags = {}
        bb.flags = flags
        bb.post_flag = lambda key, val: flags.__setitem__(key, val)

        cc._try_chunk_match(bb, {"shell_type": ShellType.PRIME})
        assert "chunk_action" in flags
        assert flags["chunk_action"]["preset_id"] == "focused"

    def test_try_chunk_match_skips_non_prime(self):
        cc = ChunkCompiler()
        cc.add_chunk(CompiledChunk(
            chunk_id="c1",
            trigger={"query_length": (0.0, 1.0)},
            action=ChunkAction(preset_id="focused"),
            successes=10,
        ))
        bb = MagicMock()
        bb.flags = {}
        cc._try_chunk_match(bb, {"shell_type": ShellType.VERIFY})
        assert "chunk_action" not in bb.flags

    def test_try_chunk_match_skips_retired(self):
        cc = ChunkCompiler()
        cc.add_chunk(CompiledChunk(
            chunk_id="retired",
            trigger={"query_length": (0.0, 1.0)},
            action=ChunkAction(preset_id="focused"),
            successes=1,
            consecutive_failures=3,  # retired
        ))
        bb = MagicMock()
        bb.context_features.return_value = {"query_length": 0.5}
        bb.flags = {}
        bb.post_flag = lambda key, val: bb.flags.__setitem__(key, val)
        cc._try_chunk_match(bb, {"shell_type": ShellType.PRIME})
        assert "chunk_action" not in bb.flags

    def test_try_chunk_match_none_blackboard(self):
        cc = ChunkCompiler()
        cc._try_chunk_match(None, {"shell_type": ShellType.PRIME})  # No crash

    def test_contribute_routes_to_pre_shell(self):
        cc = ChunkCompiler()
        bb = MagicMock()
        bb.context_features.return_value = {}
        bb.flags = {}
        cc.contribute(bb, None, {"shell_type": ShellType.PRIME})
        # Should have called context_features
        bb.context_features.assert_called()

    def test_contribute_routes_to_post_loop(self):
        cc = ChunkCompiler(config=ChunkConfig(store_in_graph=False, min_cluster_size=100))
        loop_result = MagicMock()
        loop_result.final_confidence.combined = 0.75
        loop_result.shell_count = 1
        bb = MagicMock()
        bb.context_features.return_value = {"query_length": 0.3}
        bb.flags = {}
        bb.query_id = "q1"

        cc.contribute(bb, None, {"loop_result": loop_result})
        assert cc.experience_count == 1

    def test_max_buffer_trim(self):
        cc = ChunkCompiler(config=ChunkConfig(max_buffer_size=5, store_in_graph=False, min_cluster_size=100))
        for i in range(10):
            loop_result = MagicMock()
            loop_result.final_confidence.combined = 0.5
            loop_result.shell_count = 1
            bb = MagicMock()
            bb.context_features.return_value = {"query_length": 0.3}
            bb.flags = {}
            bb.query_id = f"q{i}"
            cc._record_experience(bb, {"loop_result": loop_result})
        assert len(cc._experience) <= 5


# =============================================================================
# TRAINING TESTS
# =============================================================================

from hololoom.core.memory.v2.training import (
    TrainingExample,
    AuditSummary,
    AuditCollector,
    BanditArm,
    ParameterBandit,
    PromptRefiner,
    RefinementResult,
    Signature,
    SignatureOptimizer,
    OptimizationResult,
    TrainingPipeline,
    RewardJudge,
    ContextualBanditAdapter,
    CONFIG_PRESETS,
)


class TestTrainingExample:
    def test_is_good(self):
        ex = TrainingExample(
            query="test", shell_type="prime", confidence=0.8,
            structural=0.7, narrative=0.6, reward=0.7,
            context_tokens=100, context_items=5, duration_ms=50.0,
        )
        assert ex.is_good

    def test_is_poor(self):
        ex = TrainingExample(
            query="test", shell_type="prime", confidence=0.2,
            structural=0.1, narrative=0.1, reward=0.3,
            context_tokens=100, context_items=5, duration_ms=50.0,
        )
        assert ex.is_poor

    def test_neither_good_nor_poor(self):
        ex = TrainingExample(
            query="test", shell_type="prime", confidence=0.5,
            structural=0.5, narrative=0.5, reward=0.5,
            context_tokens=100, context_items=5, duration_ms=50.0,
        )
        assert not ex.is_good
        assert not ex.is_poor


class TestBanditArm:
    def test_initial_mean(self):
        arm = BanditArm(value=0.5)
        assert arm.mean == pytest.approx(0.5)

    def test_update_success(self):
        arm = BanditArm(value=0.5)
        arm.update(0.8)
        assert arm.alpha > 1.0
        assert arm.pulls == 1

    def test_update_failure(self):
        arm = BanditArm(value=0.5)
        arm.update(0.3)
        assert arm.beta > 1.0
        assert arm.pulls == 1

    def test_sample_in_range(self):
        arm = BanditArm(value=0.5, alpha=2.0, beta=2.0)
        for _ in range(100):
            s = arm.sample()
            assert 0.0 <= s <= 1.0


class TestParameterBandit:
    def test_select_returns_valid_index(self):
        bandit = ParameterBandit("test", [0.1, 0.2, 0.3])
        idx, val = bandit.select()
        assert 0 <= idx < 3
        assert val in [0.1, 0.2, 0.3]

    def test_update(self):
        bandit = ParameterBandit("test", [0.1, 0.2, 0.3])
        bandit.update(0, 0.9)
        assert bandit.arms[0].pulls == 1

    def test_best(self):
        bandit = ParameterBandit("test", [0.1, 0.2, 0.3])
        # Train arm 2 heavily
        for _ in range(10):
            bandit.update(2, 0.95)
        idx, val = bandit.best()
        assert idx == 2
        assert val == 0.3

    def test_train_from_examples(self):
        bandit = ParameterBandit("test.alpha", [0.10, 0.15, 0.20, 0.25])
        examples = [
            TrainingExample(
                query="q1", shell_type="prime", confidence=0.8,
                structural=0.7, narrative=0.6, reward=0.9,
                context_tokens=100, context_items=5, duration_ms=50.0,
            ),
            TrainingExample(
                query="q2", shell_type="prime", confidence=0.3,
                structural=0.2, narrative=0.1, reward=0.2,
                context_tokens=100, context_items=5, duration_ms=50.0,
            ),
        ]
        bandit.train_from_examples(examples, lambda e: 0.15)
        # Arm closest to 0.15 (index 1) should have been updated
        assert bandit.arms[1].pulls == 2

    def test_report(self):
        bandit = ParameterBandit("test", [0.1, 0.2])
        report = bandit.report()
        assert len(report) == 2
        assert "value" in report[0]
        assert "mean" in report[0]


class TestAuditCollector:
    def test_collect_empty(self):
        bus = MagicMock()
        bus._audit = []
        collector = AuditCollector(bus)
        summary = collector.collect()
        assert summary.total == 0
        assert summary.examples == []

    def test_collect_filters_type(self):
        bus = MagicMock()
        bus._audit = [
            {"type": "other_event", "query": "skip"},
            {"type": "v2_weave_cycle", "query": "keep", "v2_confidence": 0.8,
             "reward": 0.7, "context_tokens": 100, "context_items": 5,
             "duration_ms": 50.0},
        ]
        collector = AuditCollector(bus)
        summary = collector.collect()
        assert summary.total == 1
        assert summary.examples[0].query == "keep"

    def test_collect_aggregates(self):
        bus = MagicMock()
        bus._audit = [
            {"type": "v2_weave_cycle", "query": f"q{i}", "v2_confidence": 0.8,
             "reward": 0.8, "context_tokens": 100, "context_items": 5,
             "duration_ms": 50.0, "tool_used": "shell:prime"}
            for i in range(5)
        ]
        collector = AuditCollector(bus)
        summary = collector.collect()
        assert summary.total == 5
        assert summary.avg_confidence == pytest.approx(0.8)
        assert summary.avg_reward == pytest.approx(0.8)
        assert summary.good_rate == pytest.approx(1.0)
        assert summary.poor_rate == pytest.approx(0.0)

    def test_collect_shell_parsing(self):
        bus = MagicMock()
        bus._audit = [
            {"type": "v2_weave_cycle", "query": "q1", "v2_confidence": 0.5,
             "reward": 0.5, "context_tokens": 100, "context_items": 5,
             "duration_ms": 50.0, "tool_used": "shell:verify"},
        ]
        collector = AuditCollector(bus)
        summary = collector.collect()
        assert summary.examples[0].shell_type == "verify"

    def test_collect_no_audit_attr(self):
        bus = MagicMock(spec=[])  # No _audit attribute
        collector = AuditCollector(bus)
        summary = collector.collect()
        assert summary.total == 0


class TestPromptRefiner:
    @pytest.mark.asyncio
    async def test_refine_no_poor_examples(self):
        llm_fn = AsyncMock(return_value="improved template {context} {query}")
        refiner = PromptRefiner(llm_fn=llm_fn)
        examples = [
            TrainingExample(
                query="good query", shell_type="prime", confidence=0.9,
                structural=0.8, narrative=0.7, reward=0.9,
                context_tokens=100, context_items=5, duration_ms=50.0,
            ),
        ]
        result = await refiner.refine(
            shell_type=ShellType.PRIME,
            current_template="original {context} {query}",
            examples=examples,
        )
        assert result.refined_template == "original {context} {query}"
        assert result.improvement_score == 0.0
        assert result.iterations == 0

    @pytest.mark.asyncio
    async def test_refine_with_poor_examples(self):
        call_count = 0

        async def mock_llm(prompt, max_tokens):
            nonlocal call_count
            call_count += 1
            if "critique" in prompt.lower() or "weakness" in prompt.lower():
                return "The template lacks specificity"
            elif "improve" in prompt.lower():
                return "Improved: {context}\n\nAnswer {query} thoroughly."
            else:
                return "0.8"

        refiner = PromptRefiner(llm_fn=mock_llm, max_iterations=1)
        examples = [
            TrainingExample(
                query="bad query", shell_type="prime", confidence=0.2,
                structural=0.1, narrative=0.1, reward=0.2,
                context_tokens=100, context_items=5, duration_ms=50.0,
            ),
        ]
        result = await refiner.refine(
            shell_type=ShellType.PRIME,
            current_template="Original {context} {query}",
            examples=examples,
        )
        assert result.examples_used == 1
        assert result.iterations >= 1

    @pytest.mark.asyncio
    async def test_refine_rejects_missing_placeholders(self):
        async def mock_llm(prompt, max_tokens):
            if "improve" in prompt.lower():
                return "Bad template without placeholders"
            elif "rate" in prompt.lower():
                return "0.5"
            return "Missing specifics"

        refiner = PromptRefiner(llm_fn=mock_llm, max_iterations=1)
        examples = [
            TrainingExample(
                query="q", shell_type="prime", confidence=0.1,
                structural=0.1, narrative=0.1, reward=0.1,
                context_tokens=100, context_items=5, duration_ms=50.0,
            ),
        ]
        result = await refiner.refine(
            shell_type=ShellType.PRIME,
            current_template="Original {context} {query}",
            examples=examples,
        )
        # Should keep original because improved is missing placeholders
        assert "{context}" in result.refined_template

    @pytest.mark.asyncio
    async def test_score_parse_error_returns_zero(self):
        async def mock_llm(prompt, max_tokens):
            if "rate" in prompt.lower():
                return "not a number at all"
            return "critique text {context} {query}"

        refiner = PromptRefiner(llm_fn=mock_llm, max_iterations=1)
        score = await refiner._score(ShellType.PRIME, "a", "b", [])
        assert score == 0.0


class TestSignatureOptimizer:
    @pytest.mark.asyncio
    async def test_optimize_empty_examples(self):
        llm_fn = AsyncMock(return_value="0.8")
        optimizer = SignatureOptimizer(llm_fn=llm_fn)
        sig = Signature(
            name="prime", input_fields=["context", "query"],
            output_fields=["response"], instruction="Answer the question.",
        )
        result = await optimizer.optimize(sig, [])
        assert result.instruction == "Answer the question."

    @pytest.mark.asyncio
    async def test_optimize_returns_best_instruction(self):
        async def mock_llm(prompt, max_tokens):
            if "alternative" in prompt.lower() or "generate" in prompt.lower():
                return "1. Better instruction for answering\n2. Even more specific instruction\n3. Detailed approach"
            elif "rate" in prompt.lower():
                return "0.7"
            return "0.5"

        optimizer = SignatureOptimizer(llm_fn=mock_llm, n_candidates=3)
        sig = Signature(
            name="prime", input_fields=["context", "query"],
            output_fields=["response"], instruction="Answer the question.",
        )
        examples = [
            TrainingExample(
                query="q1", shell_type="prime", confidence=0.9,
                structural=0.8, narrative=0.7, reward=0.9,
                context_tokens=100, context_items=5, duration_ms=50.0,
            ),
            TrainingExample(
                query="q2", shell_type="prime", confidence=0.1,
                structural=0.1, narrative=0.1, reward=0.1,
                context_tokens=100, context_items=5, duration_ms=50.0,
            ),
        ]
        result = await optimizer.optimize(sig, examples)
        assert result.name == "prime"
        assert len(result.instruction) > 0

    def test_select_demonstrations(self):
        optimizer = SignatureOptimizer(llm_fn=AsyncMock())
        examples = [
            TrainingExample(
                query="q1", shell_type="prime", confidence=0.9,
                structural=0.8, narrative=0.7, reward=0.9,
                context_tokens=100, context_items=5, duration_ms=50.0,
            ),
            TrainingExample(
                query="q2", shell_type="prime", confidence=0.8,
                structural=0.7, narrative=0.6, reward=0.8,
                context_tokens=100, context_items=5, duration_ms=50.0,
            ),
        ]
        demos = optimizer._select_demonstrations(examples, max_demos=2)
        assert len(demos) == 2
        assert demos[0]["query"] == "q1"  # Highest reward first

    def test_select_demonstrations_empty(self):
        optimizer = SignatureOptimizer(llm_fn=AsyncMock())
        demos = optimizer._select_demonstrations([], max_demos=3)
        assert demos == []


class TestRewardJudge:
    @pytest.mark.asyncio
    async def test_score_without_llm(self):
        judge = RewardJudge(llm_fn=None)
        score = await judge.score(
            query="test query",
            response="A" * 200,
            confidence=0.8,
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_empty_response(self):
        judge = RewardJudge(llm_fn=None)
        score = await judge.score(query="test", response="", confidence=0.8)
        # substance = 0, only structural contributes
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_with_llm(self):
        async def mock_llm(prompt, max_tokens):
            return "0.85"

        judge = RewardJudge(llm_fn=mock_llm)
        score = await judge.score(
            query="test query",
            response="A" * 200,
            confidence=0.7,
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_llm_failure_fallback(self):
        async def bad_llm(prompt, max_tokens):
            raise RuntimeError("LLM down")

        judge = RewardJudge(llm_fn=bad_llm)
        score = await judge.score(
            query="test", response="A" * 200, confidence=0.7,
        )
        # Falls back to 0.5 for LLM part
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_llm_unparseable(self):
        async def bad_parse_llm(prompt, max_tokens):
            return "I think it's pretty good overall"

        judge = RewardJudge(llm_fn=bad_parse_llm)
        score = await judge.score(
            query="test", response="A" * 200, confidence=0.7,
        )
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_short_response_skips_llm(self):
        call_count = 0

        async def counting_llm(prompt, max_tokens):
            nonlocal call_count
            call_count += 1
            return "0.9"

        judge = RewardJudge(llm_fn=counting_llm)
        score = await judge.score(query="q", response="short", confidence=0.8)
        # Response < 50 chars → should not call LLM
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_substance_scoring(self):
        judge = RewardJudge(llm_fn=None)
        short = await judge.score(query="q", response="A" * 50, confidence=0.5)
        long = await judge.score(query="q", response="A" * 2000, confidence=0.5)
        assert long > short

    @pytest.mark.asyncio
    async def test_zero_weights_fallback(self):
        judge = RewardJudge(
            llm_fn=None, structural_weight=0.0, substance_weight=0.0, llm_weight=1.0,
        )
        score = await judge.score(query="q", response="A" * 100, confidence=0.8)
        assert score == pytest.approx(0.5)


class TestTrainingPipeline:
    def test_init_creates_bandits(self):
        pipeline = TrainingPipeline()
        assert "prime.alpha" in pipeline.bandits
        assert "verify.alpha" in pipeline.bandits
        assert "prime.token_budget" in pipeline.bandits
        assert "confidence.high_threshold" in pipeline.bandits

    def test_init_disables_llm_features_without_llm(self):
        pipeline = TrainingPipeline(llm_fn=None)
        assert not pipeline.enable_prompt_refinement
        assert not pipeline.enable_signature_optimization

    def test_build_configs(self):
        pipeline = TrainingPipeline()
        configs = pipeline._build_configs()
        assert ShellType.PRIME in configs
        assert ShellType.VERIFY in configs
        assert ShellType.FLAG in configs

    def test_filter_examples_by_shell(self):
        pipeline = TrainingPipeline()
        examples = [
            TrainingExample(query="q1", shell_type="prime", confidence=0.5,
                            structural=0.5, narrative=0.5, reward=0.5,
                            context_tokens=100, context_items=5, duration_ms=50.0),
            TrainingExample(query="q2", shell_type="verify", confidence=0.5,
                            structural=0.5, narrative=0.5, reward=0.5,
                            context_tokens=100, context_items=5, duration_ms=50.0),
        ]
        prime_only = pipeline._filter_examples("prime.alpha", examples)
        assert len(prime_only) == 1
        assert prime_only[0].query == "q1"

        all_examples = pipeline._filter_examples("confidence.high_threshold", examples)
        assert len(all_examples) == 2

    def test_value_fn_for(self):
        pipeline = TrainingPipeline()
        ex = TrainingExample(
            query="q", shell_type="prime", confidence=0.8,
            structural=0.7, narrative=0.6, reward=0.9,
            context_tokens=1500, context_items=10, duration_ms=100.0,
        )
        assert pipeline._value_fn_for("prime.alpha")(ex) == 0.15
        assert pipeline._value_fn_for("prime.token_budget")(ex) == 1500.0
        assert pipeline._value_fn_for("confidence.high_threshold")(ex) == 0.8
        assert pipeline._value_fn_for("nonexistent")(ex) == 0.5

    def test_extract_instruction(self):
        pipeline = TrainingPipeline()
        config = ShellConfig(
            shell_type=ShellType.PRIME,
            token_budget=2048,
            prompt_template="Answer the question.\n\n{context}\n\n{query}",
        )
        instruction = pipeline._extract_instruction(config)
        assert instruction == "Answer the question."

    def test_get_configs(self):
        pipeline = TrainingPipeline()
        configs = pipeline.get_configs()
        assert ShellType.PRIME in configs

    def test_update_online(self):
        pipeline = TrainingPipeline()
        ex = TrainingExample(
            query="q", shell_type="prime", confidence=0.8,
            structural=0.7, narrative=0.6, reward=0.9,
            context_tokens=100, context_items=5, duration_ms=50.0,
        )
        # Should not raise
        pipeline.update_online(ex)
        # Verify prime bandits got updated
        total_pulls = sum(a.pulls for a in pipeline.bandits["prime.alpha"].arms)
        assert total_pulls > 0

    def test_save_and_load(self, tmp_path):
        pipeline = TrainingPipeline()
        # Train a bit
        pipeline.bandits["prime.alpha"].update(0, 0.9)
        pipeline.bandits["prime.alpha"].update(0, 0.8)

        path = str(tmp_path / "training_state.json")
        pipeline.save(path)

        pipeline2 = TrainingPipeline()
        loaded = pipeline2.load(path)
        assert loaded is True
        assert pipeline2.bandits["prime.alpha"].arms[0].pulls == 2

    def test_load_nonexistent(self, tmp_path):
        pipeline = TrainingPipeline()
        loaded = pipeline.load(str(tmp_path / "nonexistent.json"))
        assert loaded is False

    def test_load_corrupt_json(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json{{{")
        pipeline = TrainingPipeline()
        loaded = pipeline.load(str(path))
        assert loaded is False

    @pytest.mark.asyncio
    async def test_run_without_llm(self):
        pipeline = TrainingPipeline(llm_fn=None)
        bus = MagicMock()
        bus._audit = [
            {"type": "v2_weave_cycle", "query": "test", "v2_confidence": 0.7,
             "reward": 0.7, "context_tokens": 100, "context_items": 5,
             "duration_ms": 50.0, "tool_used": "shell:prime"}
            for _ in range(10)
        ]
        result = await pipeline.run(bus)
        assert result.summary.total == 10
        assert result.elapsed_seconds >= 0
        assert isinstance(result.shell_configs, dict)
        assert result.refinements == []

    @pytest.mark.asyncio
    async def test_run_with_llm(self):
        async def mock_llm(prompt, max_tokens):
            if "weakness" in prompt.lower() or "critique" in prompt.lower():
                return "Needs more specificity"
            elif "improve" in prompt.lower():
                return "Better template {context} {query} {previous_response}"
            elif "rate" in prompt.lower() or "score" in prompt.lower():
                return "0.4"
            elif "alternative" in prompt.lower() or "generate" in prompt.lower():
                return "1. Better instruction approach\n2. More focused instruction\n3. Detailed approach"
            return "0.5"

        pipeline = TrainingPipeline(llm_fn=mock_llm)
        bus = MagicMock()
        bus._audit = [
            {"type": "v2_weave_cycle", "query": f"test query {i}",
             "v2_confidence": 0.3, "reward": 0.2,
             "context_tokens": 100, "context_items": 5,
             "duration_ms": 50.0, "tool_used": "shell:prime"}
            for i in range(5)
        ]
        result = await pipeline.run(bus)
        assert result.summary.total == 5
        # Should have attempted refinement
        assert len(result.refinements) >= 1


class TestOptimizationResult:
    def test_changes_summary(self):
        result = OptimizationResult(
            shell_configs={},
            refinements=[
                RefinementResult(
                    shell_type=ShellType.PRIME,
                    original_template="old",
                    refined_template="new",
                    critique="better",
                    improvement_score=0.5,
                    examples_used=3,
                    iterations=2,
                ),
            ],
            bandit_reports={
                "prime.alpha": [
                    {"value": 0.15, "mean": 0.8, "pulls": 10, "alpha": 5, "beta": 2},
                    {"value": 0.20, "mean": 0.6, "pulls": 5, "alpha": 3, "beta": 2},
                ],
            },
            signature_updates={
                "prime": Signature(
                    name="prime",
                    input_fields=["context", "query"],
                    output_fields=["response"],
                    instruction="Better instruction",
                    demonstrations=[{"query": "q1"}],
                ),
            },
            summary=AuditSummary(
                total=10, by_shell={}, by_decision={},
                avg_confidence=0.7, avg_reward=0.7, avg_duration_ms=50.0,
                good_rate=0.8, poor_rate=0.1, examples=[],
            ),
            elapsed_seconds=1.5,
        )
        changes = result.changes()
        assert len(changes) >= 1
        assert any("prime.alpha" in c for c in changes)
        assert any("refined" in c for c in changes)
        assert any("demonstration" in c for c in changes)


class TestContextualBanditAdapter:
    def test_no_policy_no_fallback_returns_defaults(self):
        adapter = ContextualBanditAdapter()
        bb = MagicMock()
        configs = adapter.select_config(bb)
        assert ShellType.PRIME in configs

    def test_has_neural_policy_false(self):
        adapter = ContextualBanditAdapter()
        assert not adapter.has_neural_policy

    def test_fallback_selection(self):
        fallback = {
            "prime.alpha": ParameterBandit("prime.alpha", [0.1, 0.2, 0.3]),
            "prime.token_budget": ParameterBandit("prime.token_budget", [1024, 2048]),
        }
        adapter = ContextualBanditAdapter(fallback_bandits=fallback)
        bb = MagicMock()
        configs = adapter.select_config(bb)
        assert ShellType.PRIME in configs
        assert configs[ShellType.PRIME].ppr_alpha in [0.1, 0.2, 0.3]

    def test_get_preset(self):
        adapter = ContextualBanditAdapter()
        preset = adapter._get_preset("balanced")
        assert preset is not None
        assert preset["id"] == "balanced"

    def test_get_preset_nonexistent(self):
        adapter = ContextualBanditAdapter()
        assert adapter._get_preset("nonexistent") is None

    def test_preset_to_configs(self):
        adapter = ContextualBanditAdapter()
        configs = adapter._preset_to_configs("focused")
        assert configs[ShellType.PRIME].ppr_alpha == 0.25
        assert configs[ShellType.PRIME].token_budget == 1536

    def test_preset_to_configs_invalid(self):
        adapter = ContextualBanditAdapter()
        configs = adapter._preset_to_configs("nonexistent")
        # Falls back to defaults
        assert ShellType.PRIME in configs

    def test_report(self):
        adapter = ContextualBanditAdapter()
        report = adapter.report()
        assert "has_neural_policy" in report
        assert "has_fallback" in report
        assert "presets" in report

    def test_update_without_policy(self):
        adapter = ContextualBanditAdapter()
        bb = MagicMock()
        adapter.update(bb, reward=0.8)  # Should not raise

    def test_update_with_fallback(self):
        fallback = {
            "prime.alpha": ParameterBandit("prime.alpha", [0.1, 0.2, 0.3]),
            "prime.token_budget": ParameterBandit("prime.token_budget", [1024, 2048]),
        }
        adapter = ContextualBanditAdapter(fallback_bandits=fallback)
        adapter._last_action_id = "focused"
        bb = MagicMock()
        adapter.update(bb, reward=0.8)
        # Should have updated some arms
        total_pulls = sum(a.pulls for b in fallback.values() for a in b.arms)
        assert total_pulls > 0


class TestConfigPresets:
    def test_all_presets_have_required_keys(self):
        for preset in CONFIG_PRESETS:
            assert "id" in preset
            assert "alpha" in preset
            assert "budget" in preset
            assert "seeds" in preset

    def test_preset_values_reasonable(self):
        for preset in CONFIG_PRESETS:
            assert 0.0 < preset["alpha"] < 1.0
            assert preset["budget"] > 0
            assert preset["seeds"] > 0
