"""
Tests for scored_observation — domain-agnostic additive scoring pipeline output.

Validates ScoringTerm, band_from_score, score_terms, scored_observation,
and advance_observation lifecycle transitions.
"""

import asyncio
import pytest

from hololoom.core.memory.scored_observation import (
    ScoringTerm,
    band_from_score,
    score_terms,
    scored_observation,
    advance_observation,
)
from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.bus import MemoryItem


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# =============================================================================
# Test scoring terms
# =============================================================================

def always_one(ctx):
    return 1.0

def always_half(ctx):
    return 0.5

def always_zero(ctx):
    return 0.0

def reads_value(ctx):
    return ctx.get("value", 0.0)


SAMPLE_TERMS = [
    ScoringTerm("w_alpha", 40, always_one),
    ScoringTerm("w_beta", 30, always_half),
    ScoringTerm("w_gamma", 20, always_zero),
]


class TestScoringTerm:
    def test_frozen(self):
        term = ScoringTerm("test", 10, always_one)
        with pytest.raises(AttributeError):
            term.name = "changed"
        with pytest.raises(AttributeError):
            term.weight = 99


class TestBandFromScore:
    @pytest.mark.parametrize("score,expected", [
        (100, "P0"), (80, "P0"), (79.99, "P1"),
        (50, "P1"), (35, "P1"), (34.99, "P2"),
        (25, "P2"), (18, "P2"), (17.99, "P3"),
        (10, "P3"), (0, "P3"),
    ])
    def test_boundaries(self, score, expected):
        assert band_from_score(score) == expected


class TestScoreTerms:
    def test_basic_scoring(self):
        composite, breakdown = score_terms(SAMPLE_TERMS, {})
        # w_alpha: 40 * 1.0 = 40, w_beta: 30 * 0.5 = 15, w_gamma: 20 * 0.0 = 0
        assert composite == pytest.approx(55.0)
        assert breakdown["w_alpha"] == pytest.approx(40.0)
        assert breakdown["w_beta"] == pytest.approx(15.0)
        assert breakdown["w_gamma"] == pytest.approx(0.0)

    def test_mode_weights(self):
        mode_weights = {"w_alpha": 2.0, "w_gamma": 0.5}
        composite, breakdown = score_terms(SAMPLE_TERMS, {}, mode_weights)
        # w_alpha: 40 * 2.0 * 1.0 = 80, w_beta: 30 * 0.5 = 15, w_gamma: 20 * 0.5 * 0.0 = 0
        assert composite == pytest.approx(95.0)
        assert breakdown["w_alpha"] == pytest.approx(80.0)

    def test_context_passed_to_fn(self):
        terms = [ScoringTerm("w_val", 50, reads_value)]
        composite, breakdown = score_terms(terms, {"value": 0.6})
        assert composite == pytest.approx(30.0)  # 50 * 0.6


class TestScoredObservation:
    def test_returns_memory_item(self):
        obs = scored_observation(
            content="test observation",
            domain="hive",
            category="pest",
            terms=SAMPLE_TERMS,
            context={},
            entity_ids=["hive-3"],
            mode="inspection",
        )
        assert isinstance(obs, MemoryItem)
        assert obs.memory_type == "hive"
        assert obs.entity_ids == ["hive-3"]

    def test_properties_shape(self):
        obs = scored_observation(
            content="test",
            domain="kitchen",
            category="recipe",
            terms=SAMPLE_TERMS,
            context={},
        )
        props = obs.properties
        assert "band" in props
        assert "status" in props
        assert props["status"] == "open"
        assert "composite" in props
        assert "breakdown" in props
        assert "mode" in props
        assert "observed_at" in props
        assert isinstance(props["breakdown"], dict)

    def test_importance_normalized(self):
        # SAMPLE_TERMS composite = 55, so importance = 0.55
        obs = scored_observation("test", "d", "c", SAMPLE_TERMS, {})
        assert obs.importance == pytest.approx(0.55)

    def test_band_derived_from_composite(self):
        obs = scored_observation("test", "d", "c", SAMPLE_TERMS, {})
        assert obs.properties["band"] == "P1"  # 55 → P1


class TestAdvanceObservation:
    def test_creates_followed_by_edge(self):
        bus = LiteMemoryBus()

        async def _test():
            await bus.initialize()
            obs = scored_observation("varroa high", "hive", "pest", SAMPLE_TERMS, {})
            obs_id = await bus.store(obs)

            res_id = await advance_observation(bus, obs_id, "resolved", "Treated with oxalic")

            # Verify edge exists
            edges = bus._edges.get(obs_id, [])
            targets = [t for t, r in edges if r == "FOLLOWED_BY"]
            assert res_id in targets

            # Verify resolution item exists
            res_item = bus.get_item(res_id)
            assert res_item is not None
            assert res_item.properties["status"] == "resolved"
            assert res_item.properties["original_id"] == obs_id

        run(_test())

    def test_updates_original_status(self):
        bus = LiteMemoryBus()

        async def _test():
            await bus.initialize()
            obs = scored_observation("stores low", "hive", "stores", SAMPLE_TERMS, {})
            obs_id = await bus.store(obs)

            await advance_observation(bus, obs_id, "acted", "Added feeder")

            original = bus.get_item(obs_id)
            assert original.properties["status"] == "acted"

        run(_test())

    def test_missing_id_raises(self):
        bus = LiteMemoryBus()

        async def _test():
            await bus.initialize()
            with pytest.raises(KeyError, match="not found"):
                await advance_observation(bus, "nonexistent", "resolved")

        run(_test())
