"""
Test Suite for Nap Mechanic
============================

Comprehensive tests for energy system, nap/sleep mechanics, and dream triggers.

Test Coverage:
- Energy depletion (time-based, activities)
- Energy restoration (naps, sleep)
- Dream trigger logic (thresholds, types, intensity)
- Edge cases (negative energy, over-restoration)
- State tracking and history
"""

import pytest
from datetime import datetime, timedelta
from nap_mechanic import (
    NapMechanic,
    EnergyState,
    NapEvent,
    ActivityType,
    DreamType,
    create_nap_mechanic,
    BASE_HOURLY_DEPLETION,
    ACTIVITY_COSTS,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def nap_mechanic():
    """Create a fresh NapMechanic for testing."""
    return create_nap_mechanic("test_resident")


@pytest.fixture
def high_energy_mechanic():
    """Create NapMechanic with high energy."""
    mechanic = create_nap_mechanic("high_energy")
    mechanic.energy_state.current_energy = 95.0
    return mechanic


@pytest.fixture
def low_energy_mechanic():
    """Create NapMechanic with low energy."""
    mechanic = create_nap_mechanic("low_energy")
    mechanic.energy_state.current_energy = 25.0
    return mechanic


@pytest.fixture
def critical_energy_mechanic():
    """Create NapMechanic at critical energy."""
    mechanic = create_nap_mechanic("critical_energy")
    mechanic.energy_state.current_energy = 15.0
    return mechanic


# ============================================================================
# Energy Depletion Tests
# ============================================================================


class TestEnergyDepletion:
    """Test energy depletion mechanics."""

    def test_passive_depletion(self, nap_mechanic):
        """Test passive hourly energy depletion."""
        initial_energy = nap_mechanic.energy_state.current_energy
        nap_mechanic.update_energy(hours_elapsed=1.0)
        expected = initial_energy - BASE_HOURLY_DEPLETION
        assert abs(nap_mechanic.energy_state.current_energy - expected) < 0.01

    def test_passive_depletion_multiple_hours(self, nap_mechanic):
        """Test depletion over multiple hours."""
        initial_energy = nap_mechanic.energy_state.current_energy
        nap_mechanic.update_energy(hours_elapsed=5.0)
        expected = initial_energy - (BASE_HOURLY_DEPLETION * 5.0)
        assert abs(nap_mechanic.energy_state.current_energy - expected) < 0.01

    def test_activity_work(self, nap_mechanic):
        """Test work activity depletion."""
        initial = nap_mechanic.energy_state.current_energy
        nap_mechanic.deplete_energy(ActivityType.WORK)
        expected_cost = abs(ACTIVITY_COSTS[ActivityType.WORK])
        assert abs(nap_mechanic.energy_state.current_energy - (initial - expected_cost)) < 0.01

    def test_activity_conflict(self, nap_mechanic):
        """Test conflict activity depletion (high cost)."""
        initial = nap_mechanic.energy_state.current_energy
        nap_mechanic.deplete_energy(ActivityType.CONFLICT)
        expected_cost = abs(ACTIVITY_COSTS[ActivityType.CONFLICT])
        assert abs(nap_mechanic.energy_state.current_energy - (initial - expected_cost)) < 0.01

    def test_activity_social(self, nap_mechanic):
        """Test social activity (low cost)."""
        initial = nap_mechanic.energy_state.current_energy
        nap_mechanic.deplete_energy(ActivityType.SOCIAL)
        expected_cost = abs(ACTIVITY_COSTS[ActivityType.SOCIAL])
        assert abs(nap_mechanic.energy_state.current_energy - (initial - expected_cost)) < 0.01

    def test_activity_with_stress_multiplier(self, nap_mechanic):
        """Test activity cost modified by stress."""
        initial = nap_mechanic.energy_state.current_energy
        base_cost = abs(ACTIVITY_COSTS[ActivityType.WORK])
        nap_mechanic.deplete_energy(ActivityType.WORK, multiplier=1.5)
        expected = initial - (base_cost * 1.5)
        assert abs(nap_mechanic.energy_state.current_energy - expected) < 0.01

    def test_energy_cannot_go_negative(self, nap_mechanic):
        """Test that energy is clamped at 0."""
        nap_mechanic.energy_state.current_energy = 5.0
        nap_mechanic.update_energy(hours_elapsed=10.0)
        assert nap_mechanic.energy_state.current_energy >= 0.0

    def test_negative_time_elapsed_ignored(self, nap_mechanic):
        """Test that negative time is gracefully handled."""
        initial = nap_mechanic.energy_state.current_energy
        nap_mechanic.update_energy(hours_elapsed=-5.0)
        assert nap_mechanic.energy_state.current_energy == initial

    def test_activity_history_tracked(self, nap_mechanic):
        """Test that activities are tracked."""
        nap_mechanic.deplete_energy(ActivityType.WORK)
        nap_mechanic.deplete_energy(ActivityType.CONFLICT)
        assert len(nap_mechanic.activity_history) == 2
        assert nap_mechanic.activity_history[0][1] == ActivityType.WORK
        assert nap_mechanic.activity_history[1][1] == ActivityType.CONFLICT


# ============================================================================
# Energy Restoration Tests
# ============================================================================


class TestEnergyRestoration:
    """Test nap and sleep mechanics."""

    def test_short_nap_restores_energy(self, low_energy_mechanic):
        """Test that a 1.5 hour nap restores energy."""
        initial = low_energy_mechanic.energy_state.current_energy
        state, event = low_energy_mechanic.restore_energy(sleep_hours=1.5)
        assert state.current_energy > initial
        assert event.duration_hours == 1.5

    def test_full_sleep_restores_most_energy(self, low_energy_mechanic):
        """Test that 8 hour sleep restores to near-max."""
        initial = low_energy_mechanic.energy_state.current_energy
        state, event = low_energy_mechanic.restore_energy(sleep_hours=8.0)
        assert state.current_energy > initial + 50.0  # Significant restoration
        assert state.current_energy <= 100.0

    def test_very_long_sleep_capped_at_max(self, low_energy_mechanic):
        """Test that energy cannot exceed max."""
        low_energy_mechanic.restore_energy(sleep_hours=12.0)
        assert low_energy_mechanic.energy_state.current_energy <= 100.0

    def test_poor_quality_sleep_reduces_restoration(self, low_energy_mechanic):
        """Test that poor sleep quality (0.5x) reduces restoration."""
        initial = low_energy_mechanic.energy_state.current_energy
        state_good, _ = low_energy_mechanic.restore_energy(sleep_hours=4.0, quality_modifier=1.0)
        good_energy = state_good.current_energy

        low_energy_mechanic.energy_state.current_energy = initial  # Reset
        state_bad, _ = low_energy_mechanic.restore_energy(sleep_hours=4.0, quality_modifier=0.5)

        # With 0.5x quality, restoration should be less
        assert state_bad.current_energy < good_energy

    def test_excellent_sleep_boosts_restoration(self, low_energy_mechanic):
        """Test that excellent sleep (1.5x) boosts restoration."""
        initial = low_energy_mechanic.energy_state.current_energy
        state_normal, _ = low_energy_mechanic.restore_energy(sleep_hours=4.0, quality_modifier=1.0)
        normal_energy = state_normal.current_energy

        low_energy_mechanic.energy_state.current_energy = initial  # Reset
        state_excellent, _ = low_energy_mechanic.restore_energy(sleep_hours=4.0, quality_modifier=1.5)

        # With 1.5x quality, restoration should be more
        assert state_excellent.current_energy > normal_energy

    def test_invalid_sleep_duration_ignored(self, nap_mechanic):
        """Test that invalid sleep durations are handled."""
        initial = nap_mechanic.energy_state.current_energy
        result = nap_mechanic.restore_energy(sleep_hours=-5.0)
        # Should return None or same state
        assert result[1] is None or result[0].current_energy == initial

    def test_nap_history_recorded(self, nap_mechanic):
        """Test that naps are added to history."""
        nap_mechanic.restore_energy(sleep_hours=2.0)
        nap_mechanic.restore_energy(sleep_hours=8.0)
        assert len(nap_mechanic.nap_history) == 2

    def test_nap_event_contains_correct_data(self, nap_mechanic):
        """Test NapEvent has all required fields."""
        initial_energy = nap_mechanic.energy_state.current_energy
        state, event = nap_mechanic.restore_energy(sleep_hours=6.0)

        assert event.resident_id == "test_resident"
        assert event.duration_hours == 6.0
        assert event.energy_before == pytest.approx(initial_energy, abs=0.1)
        assert event.energy_after == pytest.approx(state.current_energy, abs=0.1)

    def test_forced_sleep_flag_set_for_critical_energy(self, critical_energy_mechanic):
        """Test that forced_sleep flag is set when energy < sleep_threshold."""
        state, event = critical_energy_mechanic.restore_energy(sleep_hours=2.0)
        assert event.forced_sleep is True

    def test_forced_sleep_flag_unset_for_normal_energy(self, high_energy_mechanic):
        """Test that forced_sleep flag is False when energy >= sleep_threshold."""
        state, event = high_energy_mechanic.restore_energy(sleep_hours=2.0)
        assert event.forced_sleep is False


# ============================================================================
# Dream Trigger Tests
# ============================================================================


class TestDreamTriggers:
    """Test dream trigger logic."""

    def test_no_dream_for_short_nap(self, nap_mechanic):
        """Test that naps < 1 hour don't trigger dreams."""
        state, event = nap_mechanic.restore_energy(sleep_hours=0.5)
        assert event.dream_triggered is False

    def test_dream_triggered_for_critical_energy(self, critical_energy_mechanic):
        """Test mandatory dream when energy < sleep_threshold."""
        state, event = critical_energy_mechanic.restore_energy(sleep_hours=1.5)
        assert event.dream_triggered is True
        assert event.dream_type == DreamType.EXHAUSTION_DREAM

    def test_dream_guaranteed_for_long_sleep(self, low_energy_mechanic):
        """Test that 6+ hour sleep always triggers dream."""
        state, event = low_energy_mechanic.restore_energy(sleep_hours=8.0)
        assert event.dream_triggered is True

    def test_nap_dream_for_recovering_low_energy(self, nap_mechanic):
        """Test that nap triggers dream when energy is recovering from low."""
        nap_mechanic.energy_state.current_energy = 35.0
        state, event = nap_mechanic.restore_energy(sleep_hours=1.5)
        assert event.dream_triggered is True
        assert event.dream_type == DreamType.NAP_DREAM

    def test_dream_type_high_energy(self, high_energy_mechanic):
        """Test dream type for high energy."""
        state, event = high_energy_mechanic.restore_energy(sleep_hours=8.0)
        assert event.dream_type == DreamType.VIVID or event.dream_type == DreamType.NORMAL

    def test_dream_intensity_for_critical_energy(self, critical_energy_mechanic):
        """Test dream intensity is high for critical energy."""
        state, event = critical_energy_mechanic.restore_energy(sleep_hours=2.0)
        assert event.dream_intensity > 0.6

    def test_dream_intensity_for_high_energy(self, high_energy_mechanic):
        """Test dream intensity is high for high energy."""
        state, event = high_energy_mechanic.restore_energy(sleep_hours=8.0)
        assert event.dream_intensity > 0.6

    def test_dream_intensity_for_low_energy(self, low_energy_mechanic):
        """Test dream intensity is high for low energy (exhaustion dream)."""
        state, event = low_energy_mechanic.restore_energy(sleep_hours=8.0)
        # Low energy (25%) triggers exhaustion dream with high intensity
        assert event.dream_type == DreamType.EXHAUSTION_DREAM
        assert event.dream_intensity > 0.6


# ============================================================================
# Energy Threshold Tests
# ============================================================================


class TestEnergyThresholds:
    """Test energy threshold management."""

    def test_needs_sleep_flag_set_below_threshold(self, nap_mechanic):
        """Test needs_sleep flag when energy drops below threshold."""
        nap_mechanic.energy_state.current_energy = 25.0
        nap_mechanic.update_energy(hours_elapsed=0.1)  # Small update to trigger check
        assert nap_mechanic.energy_state.needs_sleep is True

    def test_needs_sleep_flag_cleared_after_rest(self, critical_energy_mechanic):
        """Test needs_sleep flag clears after adequate rest."""
        # Ensure needs_sleep is set
        critical_energy_mechanic.energy_state.needs_sleep = True
        assert critical_energy_mechanic.energy_state.needs_sleep is True
        critical_energy_mechanic.restore_energy(sleep_hours=4.0)
        assert critical_energy_mechanic.energy_state.needs_sleep is False

    def test_fragmented_dream_below_quality_threshold(self, nap_mechanic):
        """Test fragmented dream when energy < dream_quality_threshold."""
        nap_mechanic.energy_state.current_energy = 50.0
        state, event = nap_mechanic.restore_energy(sleep_hours=8.0)
        # 50% energy (low) + 8h sleep classifies as FRAGMENTED based on energy_before
        assert event.dream_type == DreamType.FRAGMENTED

    def test_normal_dream_at_medium_energy(self, nap_mechanic):
        """Test normal dream at medium energy."""
        nap_mechanic.energy_state.current_energy = 70.0
        state, event = nap_mechanic.restore_energy(sleep_hours=8.0)
        # 70% energy (medium-high) + 8h sleep classifies as NORMAL
        assert event.dream_type in (DreamType.NORMAL, DreamType.VIVID)

    def test_vivid_dream_at_high_energy(self, nap_mechanic):
        """Test vivid dream at high energy."""
        nap_mechanic.energy_state.current_energy = 95.0
        state, event = nap_mechanic.restore_energy(sleep_hours=8.0)
        assert event.dream_type == DreamType.VIVID


# ============================================================================
# Dream Quality Metrics Tests
# ============================================================================


class TestDreamQualityMetrics:
    """Test dream quality calculation."""

    def test_clarity_improves_with_energy(self, nap_mechanic):
        """Test that clarity increases with energy."""
        nap_mechanic.energy_state.current_energy = 30.0
        metrics_low = nap_mechanic.get_dream_quality_metrics()

        nap_mechanic.energy_state.current_energy = 90.0
        metrics_high = nap_mechanic.get_dream_quality_metrics()

        assert metrics_high['clarity'] > metrics_low['clarity']

    def test_emotional_intensity_decreases_with_energy(self, nap_mechanic):
        """Test that emotional intensity is inverse to energy."""
        nap_mechanic.energy_state.current_energy = 20.0
        metrics_low = nap_mechanic.get_dream_quality_metrics()

        nap_mechanic.energy_state.current_energy = 90.0
        metrics_high = nap_mechanic.get_dream_quality_metrics()

        assert metrics_low['emotional_intensity'] > metrics_high['emotional_intensity']

    def test_all_metrics_in_valid_range(self, nap_mechanic):
        """Test that all quality metrics are 0.0-1.0."""
        nap_mechanic.energy_state.current_energy = 50.0
        metrics = nap_mechanic.get_dream_quality_metrics()

        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Test state tracking and persistence."""

    def test_status_includes_all_fields(self, nap_mechanic):
        """Test that get_status returns all required fields."""
        status = nap_mechanic.get_status()
        assert 'resident_id' in status
        assert 'energy_state' in status
        assert 'dream_quality' in status
        assert 'nap_count' in status
        assert 'total_sleep_hours' in status

    def test_energy_state_serialization(self, nap_mechanic):
        """Test that energy state can be serialized."""
        state_dict = nap_mechanic.energy_state.to_dict()
        assert 'current_energy' in state_dict
        assert 'max_energy' in state_dict
        assert 'needs_sleep' in state_dict

    def test_nap_event_serialization(self, nap_mechanic):
        """Test that NapEvent can be serialized."""
        state, event = nap_mechanic.restore_energy(sleep_hours=2.0)
        event_dict = event.to_dict()
        assert 'resident_id' in event_dict
        assert 'dream_triggered' in event_dict
        assert 'dream_type' in event_dict

    def test_total_sleep_hours_accumulated(self, nap_mechanic):
        """Test that total sleep hours accumulates."""
        nap_mechanic.restore_energy(sleep_hours=2.0)
        nap_mechanic.restore_energy(sleep_hours=4.0)
        nap_mechanic.restore_energy(sleep_hours=3.0)

        status = nap_mechanic.get_status()
        assert abs(status['total_sleep_hours'] - 9.0) < 0.01


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests simulating real scenarios."""

    def test_day_cycle_with_work_and_nap(self, nap_mechanic):
        """Simulate a day: work, energy depletes, nap restores."""
        # Morning: work (8 hours)
        nap_mechanic.deplete_energy(ActivityType.WORK)  # -10%
        nap_mechanic.update_energy(hours_elapsed=8.0)   # -16%
        energy_after_work = nap_mechanic.energy_state.current_energy

        assert energy_after_work < 90.0

        # Afternoon: nap (1.5 hours)
        state, event = nap_mechanic.restore_energy(sleep_hours=1.5)
        assert state.current_energy > energy_after_work
        assert event.dream_triggered is not None

    def test_stress_spiral_requires_sleep(self, nap_mechanic):
        """Simulate stress spiral requiring sleep."""
        # Multiple conflicts in a row
        nap_mechanic.deplete_energy(ActivityType.CONFLICT, multiplier=1.5)
        nap_mechanic.deplete_energy(ActivityType.CONFLICT, multiplier=1.5)
        nap_mechanic.deplete_energy(ActivityType.CONFLICT, multiplier=1.5)
        nap_mechanic.update_energy(hours_elapsed=4.0)

        # Energy should be critical
        assert nap_mechanic.energy_state.needs_sleep is True

        # Full sleep should recover
        state, event = nap_mechanic.restore_energy(sleep_hours=8.0)
        assert state.current_energy > 50.0
        assert event.forced_sleep is True

    def test_week_simulation(self, nap_mechanic):
        """Simulate a full week of activity and sleep."""
        for day in range(7):
            # Daytime: work and activities
            nap_mechanic.deplete_energy(ActivityType.WORK)
            nap_mechanic.update_energy(hours_elapsed=10.0)

            # Evening: nap or sleep
            if nap_mechanic.energy_state.needs_sleep:
                nap_mechanic.restore_energy(sleep_hours=8.0)
            else:
                nap_mechanic.restore_energy(sleep_hours=2.0)

        # After a week, check history
        assert len(nap_mechanic.nap_history) >= 7
        assert nap_mechanic.energy_state.current_energy > 20.0

    def test_continuous_stress_without_recovery(self, nap_mechanic):
        """Test what happens under continuous stress."""
        for _ in range(5):
            nap_mechanic.deplete_energy(ActivityType.STRESS_EVENT, multiplier=1.5)

        # Energy should be depleted
        assert nap_mechanic.energy_state.needs_sleep is True
        assert nap_mechanic.energy_state.current_energy < 30.0

        # Recovery possible with sleep
        state, event = nap_mechanic.restore_energy(sleep_hours=10.0)
        assert state.current_energy > 60.0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_energy_state_validation(self):
        """Test that EnergyState validates energy."""
        # Create with invalid energy (should clamp)
        state = EnergyState(
            current_energy=150.0,  # Over max
            max_energy=100.0,
            depletion_rate=2.0,
            last_update=datetime.now(),
            dream_quality_threshold=60.0,
            sleep_threshold=30.0,
        )
        assert state.current_energy <= 100.0

    def test_zero_energy_resident(self):
        """Test resident at zero energy."""
        mechanic = create_nap_mechanic("zero_energy")
        mechanic.energy_state.current_energy = 0.0
        state, event = mechanic.restore_energy(sleep_hours=8.0)
        assert state.current_energy > 0.0

    def test_very_long_sleep_duration(self, nap_mechanic):
        """Test extremely long sleep (unrealistic but valid)."""
        state, event = nap_mechanic.restore_energy(sleep_hours=24.0)
        assert state.current_energy == 100.0  # Capped at max

    def test_multiple_naps_in_sequence(self, nap_mechanic):
        """Test taking multiple naps without full recovery between."""
        nap_mechanic.energy_state.current_energy = 50.0
        state1, event1 = nap_mechanic.restore_energy(sleep_hours=1.0)
        state1_energy = state1.current_energy
        state2, event2 = nap_mechanic.restore_energy(sleep_hours=1.0)

        assert len(nap_mechanic.nap_history) == 2
        assert state2.current_energy >= state1_energy  # Second nap should restore some energy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
