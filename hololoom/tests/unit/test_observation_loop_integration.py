"""
Integration smoke test — the full observe → score → store → query → resolve → learn loop.

Wires all three pieces together:
1. properties_match on MemoryQuery
2. scored_observation (ScoringTerm, score_terms, scored_observation, advance_observation)
3. ObservationReflector (Thompson Sampling over (mode, term_name))
"""

import asyncio
import pytest

from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.bus import MemoryQuery
from hololoom.core.memory.scored_observation import (
    ScoringTerm,
    scored_observation,
    advance_observation,
)
from hololoom.core.memory.observation_reflector import ObservationReflector


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# =============================================================================
# Domain-specific scoring terms (simulated hive inspection)
# =============================================================================

def varroa_score(ctx):
    """Score varroa mite count. Higher count = higher score."""
    count = ctx.get("varroa_per_100", 0)
    if count >= 8:
        return 1.0
    elif count >= 5:
        return 0.7
    elif count >= 3:
        return 0.4
    return 0.0

def stores_score(ctx):
    """Score honey stores. Lower stores = higher score (needs attention)."""
    level = ctx.get("stores_level", "adequate")
    return {"light": 0.8, "adequate": 0.2, "heavy": 0.0}.get(level, 0.3)

def queen_score(ctx):
    """Score queen status. No queen seen + no eggs = high score."""
    if not ctx.get("queen_seen", True) and not ctx.get("eggs_seen", True):
        return 1.0
    elif not ctx.get("queen_seen", True):
        return 0.5
    return 0.0

HIVE_TERMS = [
    ScoringTerm("w_varroa", 30, varroa_score),
    ScoringTerm("w_stores", 15, stores_score),
    ScoringTerm("w_queen", 25, queen_score),
]


class TestFullLoop:
    def test_observe_score_store_query_resolve_learn(self, tmp_path):
        """End-to-end: the complete learning loop."""
        bus = LiteMemoryBus()
        reflector = ObservationReflector()

        async def _test():
            await bus.initialize()

            # --- Cycle 1: Score a hive inspection ---
            ctx = {"varroa_per_100": 8, "stores_level": "adequate", "queen_seen": True, "eggs_seen": True}
            mode = "fall"

            # Get learned adjustments (fresh = all 1.0)
            adjustments = reflector.suggest_weight_adjustments(mode, HIVE_TERMS)
            assert all(v == 1.0 for v in adjustments.values()), "Fresh reflector should return 1.0"

            obs = scored_observation(
                content="Varroa count 8/100 in Hive 3 — emergency threshold",
                domain="hive",
                category="pest",
                terms=HIVE_TERMS,
                context=ctx,
                entity_ids=["hive-3"],
                mode=mode,
                mode_weights=adjustments,
            )

            # Verify observation shape
            assert obs.memory_type == "hive"
            # varroa=1.0*30=30, stores=0.2*15=3, queen=0.0*25=0 → composite=33 → P2
            assert obs.properties["band"] == "P2"
            assert obs.properties["status"] == "open"

            # Store in bus
            obs_id = await bus.store(obs)

            # --- Query back via properties_match ---
            result = await bus.query(MemoryQuery(
                intent="open pest issues",
                memory_type="hive",
                properties_match={"status": "open", "band": "P2"},
            ))
            assert len(result.items) == 1
            assert result.resolution_path == "structured"
            assert "Varroa" in result.items[0]["content"]

            # --- Resolve the observation (treatment applied) ---
            res_id = await advance_observation(
                bus, obs_id, "resolved", "Oxalic acid treatment applied"
            )

            # Verify temporal chain
            edges = bus._edges.get(obs_id, [])
            followed_by = [t for t, r in edges if r == "FOLLOWED_BY"]
            assert res_id in followed_by

            # Verify original status updated
            original = bus.get_item(obs_id)
            assert original.properties["status"] == "resolved"

            # --- Feed outcome to reflector ---
            reflector.record_outcome(
                mode=mode,
                breakdown=obs.properties["breakdown"],
                success=True,  # treatment worked
            )

            # --- Cycle 2: Score again with learned weights ---
            adjustments2 = reflector.suggest_weight_adjustments(mode, HIVE_TERMS)
            # w_varroa should now have multiplier > 1.0 (it predicted correctly)
            assert adjustments2["w_varroa"] > 1.0, (
                f"After success, w_varroa multiplier should increase, got {adjustments2['w_varroa']}"
            )

            # Score with updated weights
            ctx2 = {"varroa_per_100": 6, "stores_level": "light", "queen_seen": True, "eggs_seen": True}
            obs2 = scored_observation(
                content="Varroa 6/100 in Hive 3 — still elevated",
                domain="hive",
                category="pest",
                terms=HIVE_TERMS,
                context=ctx2,
                entity_ids=["hive-3"],
                mode=mode,
                mode_weights=adjustments2,
            )

            # The learned weights should amplify w_varroa's contribution
            base_varroa = 30 * 0.7  # Without adjustment: 21
            adjusted_varroa = obs2.properties["breakdown"]["w_varroa"]
            assert adjusted_varroa > base_varroa, (
                f"Adjusted varroa contribution ({adjusted_varroa}) should exceed "
                f"base ({base_varroa}) due to learned multiplier"
            )

            # --- Persistence roundtrip ---
            state_path = tmp_path / "reflector.json"
            reflector.save(state_path)

            reflector2 = ObservationReflector()
            reflector2.load(state_path)
            assert reflector2.multiplier(mode, "w_varroa") == pytest.approx(
                reflector.multiplier(mode, "w_varroa"), abs=0.01
            )

        run(_test())

    def test_multiple_domains_coexist(self):
        """Different domains (hive, kitchen) coexist in the same bus."""
        bus = LiteMemoryBus()

        async def _test():
            await bus.initialize()

            hive_obs = scored_observation(
                content="Queen cells found",
                domain="hive",
                category="queen",
                terms=[ScoringTerm("w_cells", 40, lambda ctx: 1.0)],
                context={},
                entity_ids=["hive-1"],
            )
            kitchen_obs = scored_observation(
                content="Tomatoes overripe",
                domain="kitchen",
                category="inventory",
                terms=[ScoringTerm("w_freshness", 50, lambda ctx: 0.8)],
                context={},
            )
            await bus.store(hive_obs)
            await bus.store(kitchen_obs)

            # Query hive only
            hive_result = await bus.query(MemoryQuery(
                intent="hive issues",
                memory_type="hive",
                properties_match={"status": "open"},
            ))
            assert len(hive_result.items) == 1
            assert "Queen" in hive_result.items[0]["content"]

            # Query kitchen only
            kitchen_result = await bus.query(MemoryQuery(
                intent="kitchen issues",
                memory_type="kitchen",
                properties_match={"status": "open"},
            ))
            assert len(kitchen_result.items) == 1
            assert "Tomatoes" in kitchen_result.items[0]["content"]

            # Query across domains: all open P0+P1
            all_open = await bus.query(MemoryQuery(
                intent="all urgent",
                properties_match={"status": "open"},
                max_results=10,
            ))
            assert len(all_open.items) == 2

        run(_test())

    def test_reflector_learns_different_modes(self):
        """Reflector tracks modes independently."""
        reflector = ObservationReflector()
        breakdown = {"w_varroa": 30.0, "w_stores": 3.0}

        # Spring: varroa treatment fails (too early in season)
        for _ in range(10):
            reflector.record_outcome("spring", breakdown, success=False)

        # Fall: varroa treatment succeeds
        for _ in range(10):
            reflector.record_outcome("fall", breakdown, success=True)

        spring_mult = reflector.multiplier("spring", "w_varroa")
        fall_mult = reflector.multiplier("fall", "w_varroa")

        assert spring_mult < 1.0, f"Spring varroa should be dampened: {spring_mult}"
        assert fall_mult > 1.0, f"Fall varroa should be amplified: {fall_mult}"
