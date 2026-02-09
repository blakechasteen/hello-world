"""
Unit tests for pressure engine — 2026-02-09

Tests tier computation and tier-specific behavior toggles.
"""

from memory_bus.models import PressureTier
from memory_bus.pressure import PressureEngine


class TestPressureEngine:
    def test_initial_tier0(self):
        engine = PressureEngine(context_window=100_000)
        assert engine.tier == PressureTier.TIER0

    def test_update_tier1(self):
        engine = PressureEngine(context_window=100_000)
        tier = engine.update(context_tokens_used=60_000)
        assert tier == PressureTier.TIER1

    def test_update_tier2(self):
        engine = PressureEngine(context_window=100_000)
        tier = engine.update(context_tokens_used=80_000)
        assert tier == PressureTier.TIER2

    def test_update_tier3(self):
        engine = PressureEngine(context_window=100_000)
        tier = engine.update(context_tokens_used=95_000)
        assert tier == PressureTier.TIER3


class TestPressurePolicies:
    def test_semantic_allowed_tier0(self):
        engine = PressureEngine()
        engine.update(context_tokens_used=0)
        assert engine.is_semantic_allowed() is True

    def test_semantic_allowed_tier1(self):
        engine = PressureEngine(context_window=100_000)
        engine.update(context_tokens_used=55_000)
        assert engine.is_semantic_allowed() is True

    def test_semantic_disabled_tier2(self):
        engine = PressureEngine(context_window=100_000)
        engine.update(context_tokens_used=80_000)
        assert engine.is_semantic_allowed() is False

    def test_semantic_disabled_tier3(self):
        engine = PressureEngine(context_window=100_000)
        engine.update(context_tokens_used=95_000)
        assert engine.is_semantic_allowed() is False

    def test_min_importance_increases_with_tier(self):
        engine = PressureEngine(context_window=100_000)

        engine.update(context_tokens_used=0)
        imp0 = engine.min_importance()

        engine.update(context_tokens_used=55_000)
        imp1 = engine.min_importance()

        engine.update(context_tokens_used=80_000)
        imp2 = engine.min_importance()

        engine.update(context_tokens_used=95_000)
        imp3 = engine.min_importance()

        assert imp0 < imp1 < imp2 < imp3

    def test_token_cap_tier0_none(self):
        engine = PressureEngine()
        assert engine.get_token_cap() is None

    def test_token_cap_tier3_strict(self):
        engine = PressureEngine(context_window=100_000)
        engine.update(context_tokens_used=95_000)
        cap = engine.get_token_cap()
        assert cap == 500

    def test_token_cap_min_of_tier_and_requested(self):
        engine = PressureEngine(context_window=100_000)
        engine.update(context_tokens_used=80_000)  # Tier2: cap=2000
        assert engine.get_token_cap(requested=1000) == 1000
        assert engine.get_token_cap(requested=5000) == 2000


class TestPressureStatus:
    def test_status_structure(self):
        engine = PressureEngine(context_window=100_000)
        engine.update(context_tokens_used=60_000)
        status = engine.status()
        assert status.tier == PressureTier.TIER1
        assert status.signals.context_tokens_used == 60_000
        assert "priming_strategy" in status.policy
        assert "min_importance" in status.policy
        assert "semantic_fallback_enabled" in status.policy
        assert "plan_write_through" in status.policy

    def test_plan_write_through_always_true(self):
        """Invariant #4: Plans always write-through at all tiers."""
        engine = PressureEngine(context_window=100_000)
        for tokens in [0, 55_000, 80_000, 95_000]:
            engine.update(context_tokens_used=tokens)
            policy = engine.current_policy()
            assert policy["plan_write_through"] is True
