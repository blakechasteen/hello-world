"""Tests for the ConfidenceModel."""

import math
import pytest

from hololoom.core.ritual.confidence import (
    ConfidenceModel,
    ConfidenceSignal,
    DOMAIN_DECAY_RATES,
    DOMAIN_ADJACENCY,
    DEFAULT_ADJACENCY_WEIGHT,
)


class TestConfidenceSignalValues:
    def test_positive_signals(self):
        assert ConfidenceSignal.TOOL_SUCCESS.value > 0
        assert ConfidenceSignal.RETRIEVAL_HIT.value > 0
        assert ConfidenceSignal.REFLECTION_PASS.value > 0

    def test_negative_signals(self):
        assert ConfidenceSignal.TOOL_FAILURE.value < 0
        assert ConfidenceSignal.RETRIEVAL_MISS.value < 0
        assert ConfidenceSignal.CONTRADICTION.value < 0
        assert ConfidenceSignal.ESCALATION_USED.value < 0
        assert ConfidenceSignal.HUMAN_CORRECTION.value < 0

    def test_human_correction_is_strongest_penalty(self):
        penalties = [s.value for s in ConfidenceSignal if s.value < 0]
        assert ConfidenceSignal.HUMAN_CORRECTION.value == min(penalties)


class TestConfidenceModelUpdate:
    def test_update_increases(self):
        m = ConfidenceModel(value=0.5)
        result = m.update(ConfidenceSignal.TOOL_SUCCESS)
        assert result == pytest.approx(0.58)
        assert m.value == pytest.approx(0.58)

    def test_update_decreases(self):
        m = ConfidenceModel(value=0.5)
        result = m.update(ConfidenceSignal.TOOL_FAILURE)
        assert result == pytest.approx(0.35)

    def test_clamps_to_zero(self):
        m = ConfidenceModel(value=0.1)
        m.update(ConfidenceSignal.HUMAN_CORRECTION)  # -0.30
        assert m.value == 0.0

    def test_clamps_to_one(self):
        m = ConfidenceModel(value=0.95)
        m.update(ConfidenceSignal.REFLECTION_PASS)  # +0.10
        assert m.value == 1.0

    def test_history_tracked(self):
        m = ConfidenceModel(value=0.5)
        m.update(ConfidenceSignal.TOOL_SUCCESS)
        m.update(ConfidenceSignal.TOOL_FAILURE)
        assert len(m.history) == 2
        assert m.history[0] == pytest.approx(0.58)
        assert m.history[1] == pytest.approx(0.43)

    def test_multiple_updates_accumulate(self):
        m = ConfidenceModel(value=0.5)
        for _ in range(3):
            m.update(ConfidenceSignal.TOOL_SUCCESS)
        assert m.value == pytest.approx(0.74)


class TestConfidenceModelDecay:
    def test_decay_reduces_value(self):
        m = ConfidenceModel(value=1.0, domain="coding")
        m.decay(elapsed_ms=10000)  # 10 seconds
        # exp(-0.001 * 10) = exp(-0.01) ≈ 0.990
        assert m.value == pytest.approx(math.exp(-0.01), abs=1e-6)

    def test_weather_decays_fast(self):
        m = ConfidenceModel(value=1.0, domain="weather")
        m.decay(elapsed_ms=60000)  # 1 minute
        # exp(-0.020 * 60) = exp(-1.2) ≈ 0.301
        assert m.value == pytest.approx(math.exp(-1.2), abs=1e-6)

    def test_coding_decays_slow(self):
        m = ConfidenceModel(value=1.0, domain="coding")
        m.decay(elapsed_ms=60000)
        # exp(-0.001 * 60) = exp(-0.06) ≈ 0.942
        assert m.value == pytest.approx(math.exp(-0.06), abs=1e-6)

    def test_unknown_domain_uses_default(self):
        m = ConfidenceModel(value=1.0, domain="alien")
        m.decay(elapsed_ms=10000)
        # Default rate 0.002: exp(-0.002 * 10) = exp(-0.02)
        assert m.value == pytest.approx(math.exp(-0.02), abs=1e-6)

    def test_decay_history_tracked(self):
        m = ConfidenceModel(value=1.0, domain="coding")
        m.decay(elapsed_ms=5000)
        assert len(m.history) == 1

    def test_zero_elapsed_no_decay(self):
        m = ConfidenceModel(value=0.8, domain="coding")
        m.decay(elapsed_ms=0)
        assert m.value == pytest.approx(0.8)


class TestDomainWeight:
    def test_same_domain_full_weight(self):
        m = ConfidenceModel(domain="coding")
        assert m.domain_weight("coding") == 1.0

    def test_adjacent_domains(self):
        m = ConfidenceModel(domain="farming")
        assert m.domain_weight("beekeeping") == 0.7
        assert m.domain_weight("weather") == 0.8

    def test_non_adjacent_default(self):
        m = ConfidenceModel(domain="coding")
        assert m.domain_weight("beekeeping") == DEFAULT_ADJACENCY_WEIGHT

    def test_adjacency_is_directional(self):
        # farming→weather = 0.8, weather→farming = 0.5
        m_farming = ConfidenceModel(domain="farming")
        m_weather = ConfidenceModel(domain="weather")
        assert m_farming.domain_weight("weather") == 0.8
        assert m_weather.domain_weight("farming") == 0.5


class TestConfidenceThresholds:
    def test_is_low(self):
        assert ConfidenceModel(value=0.59).is_low
        assert not ConfidenceModel(value=0.6).is_low

    def test_is_critical(self):
        assert ConfidenceModel(value=0.34).is_critical
        assert not ConfidenceModel(value=0.35).is_critical

    def test_critical_implies_low(self):
        m = ConfidenceModel(value=0.2)
        assert m.is_critical
        assert m.is_low
