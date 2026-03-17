"""
Tests for ObservationReflector — Thompson Sampling learning loop.

Validates that the reflector learns from resolved observations and
provides meaningful weight adjustments for scoring pipelines.
"""

import json
import pytest

from hololoom.core.memory.observation_reflector import ObservationReflector
from hololoom.core.memory.scored_observation import ScoringTerm


# =============================================================================
# Helpers
# =============================================================================

def always_one(ctx):
    return 1.0

def always_half(ctx):
    return 0.5

TERMS = [
    ScoringTerm("w_alpha", 40, always_one),
    ScoringTerm("w_beta", 30, always_half),
]

BREAKDOWN = {"w_alpha": 40.0, "w_beta": 15.0}


# =============================================================================
# Core Behavior
# =============================================================================

class TestObservationReflector:
    def test_fresh_multiplier_is_one(self):
        r = ObservationReflector()
        assert r.multiplier("fall", "w_varroa") == 1.0

    def test_success_increases_multiplier(self):
        r = ObservationReflector()
        for _ in range(10):
            r.record_outcome("fall", BREAKDOWN, success=True)
        m = r.multiplier("fall", "w_alpha")
        assert m > 1.0

    def test_failure_decreases_multiplier(self):
        r = ObservationReflector()
        for _ in range(10):
            r.record_outcome("fall", BREAKDOWN, success=False)
        m = r.multiplier("fall", "w_alpha")
        assert m < 1.0

    def test_multiplier_clamped(self):
        r = ObservationReflector()
        # Extreme success
        for _ in range(1000):
            r.record_outcome("x", {"t": 1.0}, success=True)
        assert r.multiplier("x", "t") <= 2.0

        # Extreme failure
        r2 = ObservationReflector()
        for _ in range(1000):
            r2.record_outcome("x", {"t": 1.0}, success=False)
        assert r2.multiplier("x", "t") >= 0.5

    def test_firing_threshold(self):
        r = ObservationReflector()
        # w_beta has lower contribution (15 vs 40), normalized = 15/40 = 0.375
        # With threshold=0.5, w_beta should be skipped
        r.record_outcome("fall", BREAKDOWN, success=True, firing_threshold=0.5)
        assert ("fall", "w_alpha") in r._arms
        assert ("fall", "w_beta") not in r._arms

    def test_suggest_weight_adjustments(self):
        r = ObservationReflector()
        r.record_outcome("fall", BREAKDOWN, success=True)
        adjustments = r.suggest_weight_adjustments("fall", TERMS)
        assert "w_alpha" in adjustments
        assert "w_beta" in adjustments
        assert isinstance(adjustments["w_alpha"], float)

    def test_independent_modes(self):
        r = ObservationReflector()
        for _ in range(10):
            r.record_outcome("spring", BREAKDOWN, success=True)
            r.record_outcome("fall", BREAKDOWN, success=False)

        m_spring = r.multiplier("spring", "w_alpha")
        m_fall = r.multiplier("fall", "w_alpha")
        assert m_spring > 1.0
        assert m_fall < 1.0
        assert m_spring != m_fall

    def test_normalized_weight_update(self):
        r = ObservationReflector()
        # w_alpha contributes 40, w_beta contributes 15
        # Normalized: w_alpha=1.0, w_beta=0.375
        # So w_alpha's arm should get stronger updates
        r.record_outcome("x", BREAKDOWN, success=True, weight=1.0)
        arm_a = r._arms[("x", "w_alpha")]
        arm_b = r._arms[("x", "w_beta")]
        # alpha got update weight=1.0*1.0=1.0, beta got 1.0*0.375=0.375
        assert arm_a.alpha > arm_b.alpha

    def test_empty_breakdown(self):
        r = ObservationReflector()
        r.record_outcome("x", {}, success=True)
        assert r.arm_count == 0

    def test_zero_breakdown(self):
        r = ObservationReflector()
        r.record_outcome("x", {"t": 0.0}, success=True)
        assert r.arm_count == 0


# =============================================================================
# Observability
# =============================================================================

class TestStats:
    def test_stats_shape(self):
        r = ObservationReflector()
        r.record_outcome("fall", {"w_x": 1.0}, success=True)
        s = r.stats()
        assert "fall:w_x" in s
        entry = s["fall:w_x"]
        assert "alpha" in entry
        assert "beta" in entry
        assert "mean" in entry
        assert "certainty" in entry
        assert "total_pulls" in entry
        assert "multiplier" in entry


# =============================================================================
# Persistence
# =============================================================================

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        r = ObservationReflector()
        for _ in range(5):
            r.record_outcome("fall", BREAKDOWN, success=True)
            r.record_outcome("spring", {"w_alpha": 20.0}, success=False)

        path = tmp_path / "reflector.json"
        r.save(path)

        r2 = ObservationReflector()
        r2.load(path)

        assert r2.arm_count == r.arm_count
        assert r2.multiplier("fall", "w_alpha") == pytest.approx(
            r.multiplier("fall", "w_alpha"), abs=0.01
        )

    def test_load_missing_file(self, tmp_path):
        r = ObservationReflector()
        r.load(tmp_path / "nonexistent.json")
        assert r.arm_count == 0

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json {{{")
        r = ObservationReflector()
        r.load(path)
        assert r.arm_count == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        r = ObservationReflector()
        r.record_outcome("x", {"t": 1.0}, success=True)
        path = tmp_path / "deep" / "nested" / "reflector.json"
        r.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "x:t" in data
