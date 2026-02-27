"""
Tests for Phase 2D: Conscience Calibrator with Thompson Sampling

Tests that ConscienceCalibrator correctly calibrates conscience decisions
using Bayesian/Thompson Sampling updates based on actual outcomes.

Created: Dec 2025
Phase: 2D - Thompson Sampling Integration
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock
from datetime import datetime, timedelta

from hololoom.agentic.conscience_calibrator import (
    ConscienceCalibrator,
    CalibrationPrior,
    CalibrationEvent,
    DriftAlert,
    create_conscience_calibrator,
)
from hololoom.protocols.conscience import (
    ConscienceDecision,
    StepType,
    RiskLevel,
)


# =============================================================================
# Test: CalibrationPrior Dataclass
# =============================================================================

class TestCalibrationPrior:
    """Tests for CalibrationPrior dataclass."""

    def test_default_initialization(self):
        """Test default initialization with uniform prior."""
        prior = CalibrationPrior()
        assert prior.alpha == 1.0
        assert prior.beta == 1.0
        assert prior.updates == 0
        assert prior.last_update is None

    def test_mean_calculation(self):
        """Test mean calculation (α / (α + β))."""
        prior = CalibrationPrior(alpha=3.0, beta=1.0)
        assert prior.mean == 0.75  # 3 / (3 + 1)

        prior2 = CalibrationPrior(alpha=1.0, beta=3.0)
        assert prior2.mean == 0.25  # 1 / (1 + 3)

        prior3 = CalibrationPrior(alpha=1.0, beta=1.0)
        assert prior3.mean == 0.5  # Uniform prior

    def test_variance_calculation(self):
        """Test variance calculation."""
        prior = CalibrationPrior(alpha=1.0, beta=1.0)
        # Variance = αβ / ((α+β)²(α+β+1)) = 1*1 / (4*3) = 1/12
        expected = 1.0 / 12.0
        assert abs(prior.variance - expected) < 0.001

    def test_confidence_with_few_updates(self):
        """Test confidence is low with few updates."""
        prior = CalibrationPrior(alpha=1.0, beta=1.0)
        assert prior.confidence == 0.0  # No updates (subtracts initial 1+1)

    def test_confidence_scales_with_updates(self):
        """Test confidence scales with observations."""
        prior = CalibrationPrior(alpha=11.0, beta=11.0)  # 20 observations
        # (11 + 11 - 2) / 100 = 20 / 100 = 0.2
        assert prior.confidence == 0.2

    def test_confidence_saturates_at_one(self):
        """Test confidence saturates at 1.0."""
        prior = CalibrationPrior(alpha=60.0, beta=50.0)  # 108 observations
        assert prior.confidence == 1.0

    def test_sample_returns_valid_value(self):
        """Test sample returns value in [0, 1]."""
        prior = CalibrationPrior(alpha=2.0, beta=3.0)
        for _ in range(100):
            sample = prior.sample()
            assert 0.0 <= sample <= 1.0

    def test_update_success(self):
        """Test update with success increases alpha."""
        prior = CalibrationPrior()
        prior.update(success=True)
        assert prior.alpha == 2.0
        assert prior.beta == 1.0
        assert prior.updates == 1
        assert prior.last_update is not None

    def test_update_failure(self):
        """Test update with failure increases beta."""
        prior = CalibrationPrior()
        prior.update(success=False)
        assert prior.alpha == 1.0
        assert prior.beta == 2.0
        assert prior.updates == 1

    def test_update_with_weight(self):
        """Test update with custom weight."""
        prior = CalibrationPrior()
        prior.update(success=True, weight=0.5)
        assert prior.alpha == 1.5
        assert prior.beta == 1.0

    def test_to_dict(self):
        """Test serialization to dict."""
        prior = CalibrationPrior(alpha=2.0, beta=3.0, updates=5)
        prior.last_update = datetime(2025, 12, 5, 10, 0, 0)

        d = prior.to_dict()
        assert d["alpha"] == 2.0
        assert d["beta"] == 3.0
        assert d["updates"] == 5
        assert d["last_update"] == "2025-12-05T10:00:00"
        assert "mean" in d
        assert "variance" in d
        assert "confidence" in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "alpha": 3.0,
            "beta": 2.0,
            "updates": 10,
            "last_update": "2025-12-05T10:00:00",
        }
        prior = CalibrationPrior.from_dict(d)
        assert prior.alpha == 3.0
        assert prior.beta == 2.0
        assert prior.updates == 10
        assert prior.last_update == datetime(2025, 12, 5, 10, 0, 0)


# =============================================================================
# Test: CalibrationEvent Dataclass
# =============================================================================

class TestCalibrationEvent:
    """Tests for CalibrationEvent dataclass."""

    def test_creation(self):
        """Test event creation."""
        event = CalibrationEvent(
            timestamp=datetime.now(),
            step_type=StepType.VERIFICATION,
            decision_allowed=True,
            actual_success=True,
            outcome_correct=True,
            confidence_before=0.1,
            confidence_after=0.15,
            prior_mean_before=0.5,
            prior_mean_after=0.55,
        )
        assert event.step_type == StepType.VERIFICATION
        assert event.decision_allowed is True
        assert event.outcome_correct is True

    def test_to_dict(self):
        """Test serialization."""
        event = CalibrationEvent(
            timestamp=datetime(2025, 12, 5, 10, 0, 0),
            step_type=StepType.RESEARCH,
            decision_allowed=False,
            actual_success=False,
            outcome_correct=True,
            confidence_before=0.2,
            confidence_after=0.25,
            prior_mean_before=0.6,
            prior_mean_after=0.55,
            metadata={"key": "value"},
        )
        d = event.to_dict()
        assert d["timestamp"] == "2025-12-05T10:00:00"
        assert d["step_type"] == "RESEARCH"
        assert d["decision_allowed"] is False
        assert d["outcome_correct"] is True
        assert d["metadata"]["key"] == "value"


# =============================================================================
# Test: DriftAlert Dataclass
# =============================================================================

class TestDriftAlert:
    """Tests for DriftAlert dataclass."""

    def test_creation(self):
        """Test alert creation."""
        alert = DriftAlert(
            timestamp=datetime.now(),
            step_type=StepType.PLAN_STEP,
            old_mean=0.6,
            new_mean=0.45,
            drift_magnitude=0.15,
            direction="stricter",
        )
        assert alert.step_type == StepType.PLAN_STEP
        assert alert.direction == "stricter"

    def test_to_dict(self):
        """Test serialization."""
        alert = DriftAlert(
            timestamp=datetime(2025, 12, 5, 10, 0, 0),
            step_type=StepType.QUERY,
            old_mean=0.5,
            new_mean=0.65,
            drift_magnitude=0.15,
            direction="more_permissive",
        )
        d = alert.to_dict()
        assert d["timestamp"] == "2025-12-05T10:00:00"
        assert d["step_type"] == "QUERY"
        assert d["direction"] == "more_permissive"


# =============================================================================
# Test: ConscienceCalibrator Initialization
# =============================================================================

class TestConscienceCalibratorInit:
    """Tests for ConscienceCalibrator initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        calibrator = ConscienceCalibrator()
        assert calibrator._drift_threshold == 0.1
        assert calibrator._min_updates == 10
        assert calibrator._decay_rate == 0.0
        assert calibrator._total_updates == 0

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        calibrator = ConscienceCalibrator(
            drift_threshold=0.2,
            min_updates_for_calibration=20,
            decay_rate=0.1,
        )
        assert calibrator._drift_threshold == 0.2
        assert calibrator._min_updates == 20
        assert calibrator._decay_rate == 0.1

    def test_priors_initialized_for_all_step_types(self):
        """Test that priors exist for all step types."""
        calibrator = ConscienceCalibrator()
        for step_type in StepType:
            assert step_type in calibrator._priors
            assert isinstance(calibrator._priors[step_type], CalibrationPrior)


# =============================================================================
# Test: ConscienceCalibrator.update()
# =============================================================================

class TestConscienceCalibratorUpdate:
    """Tests for update method."""

    @pytest.fixture
    def calibrator(self):
        """Create a calibrator for testing."""
        return ConscienceCalibrator(min_updates_for_calibration=5)

    @pytest.fixture
    def mock_decision(self):
        """Create a mock ConscienceDecision."""
        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Safe action",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )
        return decision

    def test_update_with_success(self, calibrator, mock_decision):
        """Test update with successful outcome."""
        event = calibrator.update(mock_decision, success=True)

        assert event.decision_allowed is True
        assert event.actual_success is True
        assert event.outcome_correct is True
        assert event.step_type == StepType.VERIFICATION

        prior = calibrator._priors[StepType.VERIFICATION]
        assert prior.alpha == 2.0  # Initial 1 + 1
        assert prior.beta == 1.0  # Initial 1

    def test_update_with_failure(self, calibrator, mock_decision):
        """Test update with failed outcome (allowed but failed)."""
        event = calibrator.update(mock_decision, success=False)

        assert event.decision_allowed is True
        assert event.actual_success is False
        assert event.outcome_correct is False

        prior = calibrator._priors[StepType.VERIFICATION]
        assert prior.alpha == 1.0
        assert prior.beta == 2.0

    def test_update_with_blocked_decision(self, calibrator):
        """Test update when decision was blocked."""
        decision = ConscienceDecision(
            allowed=False,  # Blocked
            risk_level=RiskLevel.HIGH,
            reason="Blocked for safety",
            voice="stop",
            concerns=["Risk detected"],
            guidance="Denied",
            step_type=StepType.RESEARCH,
        )
        # When blocked, we assume it would have failed (correct block)
        event = calibrator.update(decision, success=False)

        assert event.decision_allowed is False
        assert event.outcome_correct is True  # Blocking was correct

    def test_update_increments_total_updates(self, calibrator, mock_decision):
        """Test that total updates are tracked."""
        calibrator.update(mock_decision, success=True)
        calibrator.update(mock_decision, success=False)
        calibrator.update(mock_decision, success=True)

        assert calibrator._total_updates == 3

    def test_update_with_weight(self, calibrator, mock_decision):
        """Test update with custom weight."""
        event = calibrator.update(mock_decision, success=True, weight=0.5)

        prior = calibrator._priors[StepType.VERIFICATION]
        assert prior.alpha == 1.5  # Initial 1 + 0.5

    def test_update_tracks_false_negatives(self, calibrator, mock_decision):
        """Test that false negatives are tracked."""
        # Allowed but failed = false negative
        calibrator.update(mock_decision, success=False)

        assert calibrator._false_negatives == 1

    def test_update_returns_event_with_metadata(self, calibrator, mock_decision):
        """Test that update returns event with optional metadata."""
        event = calibrator.update(
            mock_decision,
            success=True,
            metadata={"key": "value"},
        )
        assert event.metadata["key"] == "value"

    def test_history_limited_to_max(self, calibrator, mock_decision):
        """Test that history is limited to max_history."""
        calibrator._max_history = 10

        for _ in range(20):
            calibrator.update(mock_decision, success=True)

        assert len(calibrator._history) == 10


# =============================================================================
# Test: ConscienceCalibrator.get_threshold()
# =============================================================================

class TestConscienceCalibratorThreshold:
    """Tests for threshold methods."""

    def test_get_threshold_returns_default_without_updates(self):
        """Test threshold returns 0.5 without enough updates."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=10)
        threshold = calibrator.get_threshold(StepType.VERIFICATION)
        assert threshold == 0.5

    def test_get_threshold_returns_mean_after_updates(self):
        """Test threshold returns mean after enough updates."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        # Do 10 updates: 8 success, 2 failure
        for _ in range(8):
            calibrator.update(decision, success=True)
        for _ in range(2):
            calibrator.update(decision, success=False)

        threshold = calibrator.get_threshold(StepType.VERIFICATION)
        # alpha = 1 + 8 = 9, beta = 1 + 2 = 3
        # mean = 9 / 12 = 0.75
        assert abs(threshold - 0.75) < 0.01

    def test_sample_threshold_returns_valid_value(self):
        """Test sample_threshold returns value in [0, 1]."""
        calibrator = ConscienceCalibrator()
        for _ in range(100):
            sample = calibrator.sample_threshold(StepType.QUERY)
            assert 0.0 <= sample <= 1.0


# =============================================================================
# Test: ConscienceCalibrator.get_confidence()
# =============================================================================

class TestConscienceCalibratorConfidence:
    """Tests for confidence and readiness methods."""

    def test_get_confidence_zero_without_updates(self):
        """Test confidence is zero without updates."""
        calibrator = ConscienceCalibrator()
        assert calibrator.get_confidence(StepType.VERIFICATION) == 0.0

    def test_get_confidence_increases_with_updates(self):
        """Test confidence increases with updates."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        for _ in range(10):
            calibrator.update(decision, success=True)

        confidence = calibrator.get_confidence(StepType.VERIFICATION)
        assert confidence > 0.0

    def test_should_use_calibration_false_without_updates(self):
        """Test should_use_calibration returns False without enough updates."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=10)
        assert calibrator.should_use_calibration(StepType.QUERY) is False

    def test_should_use_calibration_true_after_updates(self):
        """Test should_use_calibration returns True after enough updates."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.QUERY,
        )

        for _ in range(6):
            calibrator.update(decision, success=True)

        assert calibrator.should_use_calibration(StepType.QUERY) is True


# =============================================================================
# Test: ConscienceCalibrator.get_statistics()
# =============================================================================

class TestConscienceCalibratorStatistics:
    """Tests for statistics methods."""

    def test_get_statistics_structure(self):
        """Test statistics structure."""
        calibrator = ConscienceCalibrator()
        stats = calibrator.get_statistics()

        assert "total_updates" in stats
        assert "correct_predictions" in stats
        assert "accuracy" in stats
        assert "false_negatives" in stats
        assert "drift_alerts" in stats
        assert "per_step_type" in stats
        assert "drift_threshold" in stats
        assert "min_updates_for_calibration" in stats

    def test_get_statistics_accuracy(self):
        """Test accuracy calculation in statistics."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        # 8 correct, 2 incorrect
        for _ in range(8):
            calibrator.update(decision, success=True)
        for _ in range(2):
            calibrator.update(decision, success=False)

        stats = calibrator.get_statistics()
        assert stats["total_updates"] == 10
        assert stats["correct_predictions"] == 8
        assert stats["accuracy"] == 0.8

    def test_get_priors(self):
        """Test get_priors returns all step type priors."""
        calibrator = ConscienceCalibrator()
        priors = calibrator.get_priors()

        for step_type in StepType:
            assert step_type.name in priors
            assert "alpha" in priors[step_type.name]
            assert "beta" in priors[step_type.name]


# =============================================================================
# Test: ConscienceCalibrator Drift Detection
# =============================================================================

class TestConscienceCalibratorDrift:
    """Tests for drift detection."""

    def test_drift_alert_triggered_on_large_change(self):
        """Test drift alert is triggered on large mean change."""
        # Use a lower threshold to detect smaller per-update drifts
        calibrator = ConscienceCalibrator(
            drift_threshold=0.05,  # Lower threshold for per-update detection
            min_updates_for_calibration=5,
        )

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        # Build up a strong prior with many successes
        for _ in range(10):
            calibrator.update(decision, success=True)

        # Now apply high-weight failure to trigger significant drift
        # Use weight=5.0 to cause a larger single-update change
        event = calibrator.update(decision, success=False, weight=5.0)

        # The single large-weight update should trigger drift
        alerts = calibrator.get_drift_alerts()
        # Should have at least one alert
        assert len(alerts) > 0
        assert alerts[0].step_type == StepType.VERIFICATION
        assert alerts[0].direction == "stricter"  # Mean decreased

    def test_get_drift_alerts_since_filter(self):
        """Test filtering drift alerts by time."""
        calibrator = ConscienceCalibrator()

        # Create some alerts manually
        now = datetime.now()
        old_alert = DriftAlert(
            timestamp=now - timedelta(hours=2),
            step_type=StepType.QUERY,
            old_mean=0.5,
            new_mean=0.6,
            drift_magnitude=0.1,
            direction="more_permissive",
        )
        new_alert = DriftAlert(
            timestamp=now,
            step_type=StepType.VERIFICATION,
            old_mean=0.6,
            new_mean=0.5,
            drift_magnitude=0.1,
            direction="stricter",
        )
        calibrator._drift_alerts = [old_alert, new_alert]

        filtered = calibrator.get_drift_alerts(since=now - timedelta(hours=1))
        assert len(filtered) == 1
        assert filtered[0].step_type == StepType.VERIFICATION


# =============================================================================
# Test: ConscienceCalibrator Persistence
# =============================================================================

class TestConscienceCalibratorPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load(self):
        """Test saving and loading calibration state."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        # Do some updates
        for _ in range(10):
            calibrator.update(decision, success=True)
        for _ in range(3):
            calibrator.update(decision, success=False)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_path = Path(f.name)

        calibrator.save(save_path)

        # Load into new calibrator
        calibrator2 = ConscienceCalibrator()
        calibrator2.load(save_path)

        # Verify state was restored
        assert calibrator2._total_updates == 13
        assert calibrator2._correct_predictions == calibrator._correct_predictions

        prior1 = calibrator._priors[StepType.VERIFICATION]
        prior2 = calibrator2._priors[StepType.VERIFICATION]
        assert prior2.alpha == prior1.alpha
        assert prior2.beta == prior1.beta

        # Cleanup
        save_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file doesn't crash."""
        calibrator = ConscienceCalibrator()
        calibrator.load(Path("/nonexistent/path/calibration.json"))
        # Should just warn, not crash


# =============================================================================
# Test: ConscienceCalibrator.calibrate_decision()
# =============================================================================

class TestCalibrateDecision:
    """Tests for calibrate_decision method."""

    def test_calibrate_decision_returns_original_without_updates(self):
        """Test calibrate returns original decision without enough updates."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=10)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        result = calibrator.calibrate_decision(decision)
        # Should be the same decision (no calibration metadata)
        assert result.allowed == decision.allowed
        assert "calibration" not in result.metadata

    def test_calibrate_decision_adds_metadata_after_updates(self):
        """Test calibrate adds metadata after enough updates."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        # Do enough updates to enable calibration
        for _ in range(50):  # Need enough for high confidence
            calibrator.update(decision, success=True)

        result = calibrator.calibrate_decision(decision)
        assert "calibration" in result.metadata
        assert "threshold" in result.metadata["calibration"]
        assert "confidence" in result.metadata["calibration"]

    def test_calibrate_decision_with_sampling(self):
        """Test calibrate with sampling enabled."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        # Do enough updates
        for _ in range(50):
            calibrator.update(decision, success=True)

        result = calibrator.calibrate_decision(decision, use_sampling=True)
        if "calibration" in result.metadata:
            assert result.metadata["calibration"]["sampling"] is True


# =============================================================================
# Test: Factory Function
# =============================================================================

class TestFactoryFunction:
    """Tests for create_conscience_calibrator factory."""

    def test_create_with_defaults(self):
        """Test factory with default parameters."""
        calibrator = create_conscience_calibrator()
        assert isinstance(calibrator, ConscienceCalibrator)
        assert calibrator._drift_threshold == 0.1
        assert calibrator._min_updates == 10

    def test_create_with_custom_params(self):
        """Test factory with custom parameters."""
        calibrator = create_conscience_calibrator(
            drift_threshold=0.2,
            min_updates=20,
            decay_rate=0.05,
        )
        assert calibrator._drift_threshold == 0.2
        assert calibrator._min_updates == 20
        assert calibrator._decay_rate == 0.05

    def test_create_with_load_path(self):
        """Test factory with load_path loads state."""
        # First create and save a calibrator
        calibrator1 = create_conscience_calibrator()

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.QUERY,
        )
        for _ in range(5):
            calibrator1.update(decision, success=True)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_path = Path(f.name)

        calibrator1.save(save_path)

        # Now create with load_path
        calibrator2 = create_conscience_calibrator(load_path=save_path)
        assert calibrator2._total_updates == 5

        # Cleanup
        save_path.unlink()


# =============================================================================
# Test: Integration with Thompson Sampling Behavior
# =============================================================================

class TestThompsonSamplingBehavior:
    """Tests verifying correct Thompson Sampling behavior."""

    def test_exploration_via_sampling(self):
        """Test that sampling provides exploration."""
        calibrator = ConscienceCalibrator()

        # With uniform prior (1,1), samples should vary
        samples = [calibrator.sample_threshold(StepType.QUERY) for _ in range(100)]

        # Should have variance in samples
        sample_std = (sum((s - 0.5) ** 2 for s in samples) / len(samples)) ** 0.5
        assert sample_std > 0.1  # Should have significant variance

    def test_exploitation_converges_to_mean(self):
        """Test that get_threshold returns stable mean for exploitation."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        # Many successes should converge to high threshold
        for _ in range(20):
            calibrator.update(decision, success=True)

        threshold = calibrator.get_threshold(StepType.VERIFICATION)
        # alpha = 1 + 20 = 21, beta = 1
        # mean = 21/22 ≈ 0.95
        assert threshold > 0.9

    def test_prior_updates_correctly_on_mixed_outcomes(self):
        """Test prior updates correctly with mixed success/failure."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.RESEARCH,
        )

        # 6 successes, 4 failures
        for _ in range(6):
            calibrator.update(decision, success=True)
        for _ in range(4):
            calibrator.update(decision, success=False)

        prior = calibrator._priors[StepType.RESEARCH]
        # alpha = 1 + 6 = 7, beta = 1 + 4 = 5
        assert prior.alpha == 7.0
        assert prior.beta == 5.0
        # mean = 7/12 ≈ 0.583
        assert abs(prior.mean - 7/12) < 0.01


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_calibrator_statistics(self):
        """Test statistics on empty calibrator."""
        calibrator = ConscienceCalibrator()
        stats = calibrator.get_statistics()
        assert stats["total_updates"] == 0
        assert stats["accuracy"] == 0.0

    def test_recent_history_on_empty(self):
        """Test get_recent_history on empty calibrator."""
        calibrator = ConscienceCalibrator()
        history = calibrator.get_recent_history()
        assert history == []

    def test_multiple_step_types_independent(self):
        """Test that different step types are calibrated independently."""
        calibrator = ConscienceCalibrator(min_updates_for_calibration=5)

        query_decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.QUERY,
        )
        verification_decision = ConscienceDecision(
            allowed=True,
            risk_level=RiskLevel.LOW,
            reason="Test",
            voice="proceed",
            concerns=[],
            guidance="Continue",
            step_type=StepType.VERIFICATION,
        )

        # QUERY gets many successes
        for _ in range(10):
            calibrator.update(query_decision, success=True)

        # VERIFICATION gets many failures
        for _ in range(10):
            calibrator.update(verification_decision, success=False)

        # Should have different thresholds
        query_threshold = calibrator.get_threshold(StepType.QUERY)
        verification_threshold = calibrator.get_threshold(StepType.VERIFICATION)

        assert query_threshold > 0.8  # High success rate
        assert verification_threshold < 0.3  # Low success rate
