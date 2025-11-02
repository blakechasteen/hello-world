"""
Unit tests for validation utilities.

Tests rigor of validation, error handling, and type safety.
"""

import pytest
from datetime import datetime, timedelta

from apps.keep.models import Hive, Colony, Inspection, HarvestRecord
from apps.keep.types import HiveType, HealthStatus, QueenStatus, InspectionType
from apps.keep.validation import (
    HiveValidator,
    ColonyValidator,
    InspectionValidator,
    HarvestValidator,
    ValidationError,
    InvalidHiveError,
    InvalidColonyError,
    sanitize_string,
    sanitize_numeric,
    assert_valid_hive,
    assert_valid_colony,
    is_healthy_colony,
    is_queenless_colony,
)


class TestHiveValidator:
    """Test hive validation."""

    def test_valid_hive(self):
        """Test that valid hive passes validation."""
        hive = Hive(
            hive_id="h1",
            name="Test Hive",
            hive_type=HiveType.LANGSTROTH,
            location="Test Location",
        )

        errors = HiveValidator.validate(hive)
        assert len(errors) == 0

    def test_future_installation_date(self):
        """Test that future installation date is caught."""
        hive = Hive(
            hive_id="h1",
            installation_date=datetime.now() + timedelta(days=1)
        )

        errors = HiveValidator.validate(hive)
        assert any("future" in err.lower() for err in errors)

    def test_name_too_long(self):
        """Test that overly long name is caught."""
        hive = Hive(
            hive_id="h1",
            name="x" * 101  # Over limit
        )

        errors = HiveValidator.validate(hive)
        assert any("name" in err.lower() for err in errors)

    def test_validate_strict_raises(self):
        """Test that validate_strict raises exception."""
        hive = Hive(
            hive_id="h1",
            installation_date=datetime.now() + timedelta(days=1)
        )

        with pytest.raises(InvalidHiveError):
            HiveValidator.validate_strict(hive)


class TestColonyValidator:
    """Test colony validation."""

    def test_valid_colony(self):
        """Test that valid colony passes validation."""
        colony = Colony(
            colony_id="c1",
            hive_id="h1",
            health_status=HealthStatus.GOOD,
            queen_status=QueenStatus.PRESENT_LAYING,
            population_estimate=50000,
        )

        errors = ColonyValidator.validate(colony)
        assert len(errors) == 0

    def test_negative_population(self):
        """Test that negative population is caught."""
        colony = Colony(
            colony_id="c1",
            hive_id="h1",
            population_estimate=-1000
        )

        errors = ColonyValidator.validate(colony)
        assert any("negative" in err.lower() for err in errors)

    def test_unrealistic_population(self):
        """Test that unrealistic population is caught."""
        colony = Colony(
            colony_id="c1",
            hive_id="h1",
            population_estimate=150000  # Too high
        )

        errors = ColonyValidator.validate(colony)
        assert any("unrealistic" in err.lower() for err in errors)

    def test_queen_age_validation(self):
        """Test queen age validation."""
        colony = Colony(
            colony_id="c1",
            hive_id="h1",
            queen_age_months=-5
        )

        errors = ColonyValidator.validate(colony)
        assert any("negative" in err.lower() for err in errors)

        colony.queen_age_months = 70  # Too old
        errors = ColonyValidator.validate(colony)
        assert any("unrealistic" in err.lower() for err in errors)

    def test_logical_validation(self):
        """Test logical validations."""
        # Laying queen with zero population is illogical
        colony = Colony(
            colony_id="c1",
            hive_id="h1",
            queen_status=QueenStatus.PRESENT_LAYING,
            population_estimate=0
        )

        errors = ColonyValidator.validate(colony)
        assert any("population" in err.lower() for err in errors)


class TestInspectionValidator:
    """Test inspection validation."""

    def test_valid_inspection(self):
        """Test that valid inspection passes validation."""
        inspection = Inspection(
            inspection_id="i1",
            hive_id="h1",
            inspection_type=InspectionType.ROUTINE,
            timestamp=datetime.now(),
            temperature=72.0,
            duration_minutes=30,
            findings={
                "queen_seen": True,
                "frames_with_brood": 8,
            }
        )

        errors = InspectionValidator.validate(inspection)
        assert len(errors) == 0

    def test_future_timestamp(self):
        """Test that future timestamp is caught."""
        inspection = Inspection(
            inspection_id="i1",
            hive_id="h1",
            timestamp=datetime.now() + timedelta(days=1)
        )

        errors = InspectionValidator.validate(inspection)
        assert any("future" in err.lower() for err in errors)

    def test_unrealistic_temperature(self):
        """Test that unrealistic temperature is caught."""
        inspection = Inspection(
            inspection_id="i1",
            hive_id="h1",
            temperature=200.0  # Way too hot
        )

        errors = InspectionValidator.validate(inspection)
        assert any("unrealistic" in err.lower() for err in errors)

    def test_negative_duration(self):
        """Test that negative duration is caught."""
        inspection = Inspection(
            inspection_id="i1",
            hive_id="h1",
            duration_minutes=-10
        )

        errors = InspectionValidator.validate(inspection)
        assert any("negative" in err.lower() for err in errors)

    def test_findings_validation(self):
        """Test findings validation."""
        inspection = Inspection(
            inspection_id="i1",
            hive_id="h1",
            findings={
                "frames_with_brood": -5,  # Invalid
                "frames_with_honey": 30,  # Too many
                "population_estimate": -1000,  # Invalid
            }
        )

        errors = InspectionValidator.validate(inspection)
        assert len(errors) > 0


class TestHarvestValidator:
    """Test harvest validation."""

    def test_valid_harvest(self):
        """Test that valid harvest passes validation."""
        harvest = HarvestRecord(
            harvest_id="h1",
            hive_id="hive1",
            product_type="honey",
            quantity=45.0,
            moisture_content=17.5,
        )

        errors = HarvestValidator.validate(harvest)
        assert len(errors) == 0

    def test_negative_quantity(self):
        """Test that negative quantity is caught."""
        harvest = HarvestRecord(
            harvest_id="h1",
            hive_id="hive1",
            quantity=-10.0
        )

        errors = HarvestValidator.validate(harvest)
        assert any("negative" in err.lower() for err in errors)

    def test_unrealistic_honey_harvest(self):
        """Test that unrealistic honey harvest is caught."""
        harvest = HarvestRecord(
            harvest_id="h1",
            hive_id="hive1",
            product_type="honey",
            quantity=250.0  # Too much from one hive
        )

        errors = HarvestValidator.validate(harvest)
        assert any("unrealistic" in err.lower() for err in errors)

    def test_moisture_content_validation(self):
        """Test moisture content validation."""
        harvest = HarvestRecord(
            harvest_id="h1",
            hive_id="hive1",
            moisture_content=5.0  # Too low
        )

        errors = HarvestValidator.validate(harvest)
        assert any("moisture" in err.lower() for err in errors)


class TestSanitization:
    """Test sanitization utilities."""

    def test_sanitize_string(self):
        """Test string sanitization."""
        assert sanitize_string("  hello  ") == "hello"
        assert sanitize_string("test", max_length=2) == "te"
        assert sanitize_string("") == ""

    def test_sanitize_numeric(self):
        """Test numeric sanitization."""
        assert sanitize_numeric("42") == 42.0
        assert sanitize_numeric("42.5") == 42.5
        assert sanitize_numeric(100, min_val=0, max_val=50) == 50
        assert sanitize_numeric(-10, min_val=0) == 0

        with pytest.raises(ValueError):
            sanitize_numeric("not a number")


class TestTypeGuards:
    """Test type guard functions."""

    def test_is_healthy_colony(self):
        """Test healthy colony detection."""
        good_colony = Colony(
            colony_id="c1",
            hive_id="h1",
            health_status=HealthStatus.GOOD
        )
        assert is_healthy_colony(good_colony)

        poor_colony = Colony(
            colony_id="c2",
            hive_id="h2",
            health_status=HealthStatus.POOR
        )
        assert not is_healthy_colony(poor_colony)

    def test_is_queenless_colony(self):
        """Test queenless colony detection."""
        absent = Colony(
            colony_id="c1",
            hive_id="h1",
            queen_status=QueenStatus.ABSENT
        )
        assert is_queenless_colony(absent)

        not_laying = Colony(
            colony_id="c2",
            hive_id="h2",
            queen_status=QueenStatus.PRESENT_NOT_LAYING
        )
        assert is_queenless_colony(not_laying)

        laying = Colony(
            colony_id="c3",
            hive_id="h3",
            queen_status=QueenStatus.PRESENT_LAYING
        )
        assert not is_queenless_colony(laying)


class TestAssertHelpers:
    """Test assertion helpers."""

    def test_assert_valid_hive_passes(self):
        """Test that assert passes for valid hive."""
        hive = Hive(hive_id="h1", name="Test")
        assert_valid_hive(hive)  # Should not raise

    def test_assert_valid_hive_fails(self):
        """Test that assert raises for invalid hive."""
        hive = Hive(
            hive_id="h1",
            installation_date=datetime.now() + timedelta(days=1)
        )

        with pytest.raises(InvalidHiveError):
            assert_valid_hive(hive)

    def test_assert_valid_colony_passes(self):
        """Test that assert passes for valid colony."""
        colony = Colony(colony_id="c1", hive_id="h1")
        assert_valid_colony(colony)  # Should not raise

    def test_assert_valid_colony_fails(self):
        """Test that assert raises for invalid colony."""
        colony = Colony(
            colony_id="c1",
            hive_id="h1",
            population_estimate=-1000
        )

        with pytest.raises(InvalidColonyError):
            assert_valid_colony(colony)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
