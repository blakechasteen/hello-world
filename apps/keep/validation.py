"""
Validation and error handling utilities for Keep.

Provides rigorous validation, type checking, and error handling
for all domain objects and operations.
"""

from typing import Any, List, Optional, Callable, Type
from dataclasses import is_dataclass, fields
from datetime import datetime

from apps.keep.models import Hive, Colony, Inspection, HarvestRecord, Alert
from apps.keep.types import (
    HealthStatus,
    QueenStatus,
    HiveType,
    InspectionType,
    AlertLevel,
    InspectionData,
)


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class InvalidHiveError(ValidationError):
    """Hive validation failed."""
    pass


class InvalidColonyError(ValidationError):
    """Colony validation failed."""
    pass


class InvalidInspectionError(ValidationError):
    """Inspection validation failed."""
    pass


class InvalidHarvestError(ValidationError):
    """Harvest validation failed."""
    pass


# =============================================================================
# Domain Object Validators
# =============================================================================

class HiveValidator:
    """Validates Hive objects."""

    @staticmethod
    def validate(hive: Hive) -> List[str]:
        """
        Validate a hive and return list of errors.

        Args:
            hive: Hive to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Required fields
        if not hive.hive_id:
            errors.append("hive_id is required")

        # Type validation
        if not isinstance(hive.hive_type, HiveType):
            errors.append(f"Invalid hive_type: {hive.hive_type}")

        # Date validation
        if hive.installation_date > datetime.now():
            errors.append("installation_date cannot be in the future")

        # Name validation (if provided)
        if hive.name and len(hive.name) > 100:
            errors.append("name must be <= 100 characters")

        # Location validation (if provided)
        if hive.location and len(hive.location) > 200:
            errors.append("location must be <= 200 characters")

        return errors

    @staticmethod
    def validate_strict(hive: Hive) -> None:
        """
        Validate a hive and raise exception if invalid.

        Args:
            hive: Hive to validate

        Raises:
            InvalidHiveError: If hive is invalid
        """
        errors = HiveValidator.validate(hive)
        if errors:
            raise InvalidHiveError(f"Invalid hive: {', '.join(errors)}")


class ColonyValidator:
    """Validates Colony objects."""

    @staticmethod
    def validate(colony: Colony) -> List[str]:
        """
        Validate a colony and return list of errors.

        Args:
            colony: Colony to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Required fields
        if not colony.colony_id:
            errors.append("colony_id is required")

        if not colony.hive_id:
            errors.append("hive_id is required")

        # Type validation
        if not isinstance(colony.health_status, HealthStatus):
            errors.append(f"Invalid health_status: {colony.health_status}")

        if not isinstance(colony.queen_status, QueenStatus):
            errors.append(f"Invalid queen_status: {colony.queen_status}")

        # Range validation
        if colony.population_estimate < 0:
            errors.append("population_estimate cannot be negative")

        if colony.population_estimate > 100000:
            errors.append("population_estimate seems unrealistically high (>100k)")

        if colony.queen_age_months is not None:
            if colony.queen_age_months < 0:
                errors.append("queen_age_months cannot be negative")
            if colony.queen_age_months > 60:
                errors.append("queen_age_months seems unrealistic (>5 years)")

        # Date validation
        if colony.established_date > datetime.now():
            errors.append("established_date cannot be in the future")

        # Logical validation
        if colony.queen_status == QueenStatus.PRESENT_LAYING and colony.population_estimate == 0:
            errors.append("Colony with laying queen should have population > 0")

        return errors

    @staticmethod
    def validate_strict(colony: Colony) -> None:
        """
        Validate a colony and raise exception if invalid.

        Args:
            colony: Colony to validate

        Raises:
            InvalidColonyError: If colony is invalid
        """
        errors = ColonyValidator.validate(colony)
        if errors:
            raise InvalidColonyError(f"Invalid colony: {', '.join(errors)}")


class InspectionValidator:
    """Validates Inspection objects."""

    @staticmethod
    def validate(inspection: Inspection) -> List[str]:
        """
        Validate an inspection and return list of errors.

        Args:
            inspection: Inspection to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Required fields
        if not inspection.inspection_id:
            errors.append("inspection_id is required")

        if not inspection.hive_id:
            errors.append("hive_id is required")

        # Type validation
        if not isinstance(inspection.inspection_type, InspectionType):
            errors.append(f"Invalid inspection_type: {inspection.inspection_type}")

        # Date validation
        if inspection.timestamp > datetime.now():
            errors.append("timestamp cannot be in the future")

        # Temperature validation
        if inspection.temperature is not None:
            if inspection.temperature < -20 or inspection.temperature > 150:
                errors.append("temperature seems unrealistic")

        # Duration validation
        if inspection.duration_minutes is not None:
            if inspection.duration_minutes < 0:
                errors.append("duration_minutes cannot be negative")
            if inspection.duration_minutes > 300:
                errors.append("duration_minutes seems unrealistic (>5 hours)")

        # Findings validation
        if inspection.findings:
            finding_errors = InspectionValidator.validate_findings(inspection.findings)
            errors.extend(finding_errors)

        return errors

    @staticmethod
    def validate_findings(findings: InspectionData) -> List[str]:
        """Validate inspection findings data."""
        errors = []

        # Frame counts
        for field in ["frames_with_brood", "frames_with_honey", "frames_with_pollen"]:
            if field in findings:
                value = findings[field]
                if not isinstance(value, int):
                    errors.append(f"{field} must be an integer")
                elif value < 0:
                    errors.append(f"{field} cannot be negative")
                elif value > 20:
                    errors.append(f"{field} seems unrealistic (>20 frames)")

        # Cell counts
        for field in ["swarm_cells", "supersedure_cells"]:
            if field in findings:
                value = findings[field]
                if not isinstance(value, int):
                    errors.append(f"{field} must be an integer")
                elif value < 0:
                    errors.append(f"{field} cannot be negative")
                elif value > 50:
                    errors.append(f"{field} seems unrealistic (>50 cells)")

        # Population estimate
        if "population_estimate" in findings:
            pop = findings["population_estimate"]
            if pop < 0:
                errors.append("population_estimate cannot be negative")
            elif pop > 100000:
                errors.append("population_estimate seems unrealistic (>100k)")

        # Logical validation
        if findings.get("eggs_seen") and not findings.get("queen_seen"):
            # This is actually OK - queen might not be seen but eggs present
            pass

        if findings.get("capped_brood_seen") and not findings.get("eggs_seen"):
            # This is suspicious but possible (late season, etc.)
            pass

        return errors

    @staticmethod
    def validate_strict(inspection: Inspection) -> None:
        """
        Validate an inspection and raise exception if invalid.

        Args:
            inspection: Inspection to validate

        Raises:
            InvalidInspectionError: If inspection is invalid
        """
        errors = InspectionValidator.validate(inspection)
        if errors:
            raise InvalidInspectionError(f"Invalid inspection: {', '.join(errors)}")


class HarvestValidator:
    """Validates HarvestRecord objects."""

    @staticmethod
    def validate(harvest: HarvestRecord) -> List[str]:
        """
        Validate a harvest record and return list of errors.

        Args:
            harvest: Harvest to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Required fields
        if not harvest.harvest_id:
            errors.append("harvest_id is required")

        if not harvest.hive_id:
            errors.append("hive_id is required")

        # Quantity validation
        if harvest.quantity < 0:
            errors.append("quantity cannot be negative")

        if harvest.product_type == "honey" and harvest.quantity > 200:
            errors.append("honey harvest seems unrealistic (>200 lbs from single hive)")

        # Date validation
        if harvest.timestamp > datetime.now():
            errors.append("timestamp cannot be in the future")

        # Moisture content validation (for honey)
        if harvest.moisture_content is not None:
            if harvest.moisture_content < 10 or harvest.moisture_content > 25:
                errors.append("moisture_content seems unrealistic (should be 10-25%)")

        return errors

    @staticmethod
    def validate_strict(harvest: HarvestRecord) -> None:
        """
        Validate a harvest and raise exception if invalid.

        Args:
            harvest: Harvest to validate

        Raises:
            InvalidHarvestError: If harvest is invalid
        """
        errors = HarvestValidator.validate(harvest)
        if errors:
            raise InvalidHarvestError(f"Invalid harvest: {', '.join(errors)}")


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_dataclass(obj: Any, validator_class: Optional[Type] = None) -> List[str]:
    """
    Generic dataclass validator.

    Args:
        obj: Object to validate
        validator_class: Optional validator class to use

    Returns:
        List of validation errors
    """
    if not is_dataclass(obj):
        return [f"{obj} is not a dataclass"]

    errors = []

    # Validate field types
    for field in fields(obj):
        value = getattr(obj, field.name)

        # Check for None in required fields
        if value is None and field.default is None and field.default_factory is None:
            errors.append(f"{field.name} is required but is None")

    # Use specific validator if provided
    if validator_class:
        errors.extend(validator_class.validate(obj))

    return errors


def validate_business_rules(obj: Any) -> List[str]:
    """
    Validate business rules for domain objects.

    Args:
        obj: Domain object to validate

    Returns:
        List of validation errors
    """
    if isinstance(obj, Hive):
        return HiveValidator.validate(obj)
    elif isinstance(obj, Colony):
        return ColonyValidator.validate(obj)
    elif isinstance(obj, Inspection):
        return InspectionValidator.validate(obj)
    elif isinstance(obj, HarvestRecord):
        return HarvestValidator.validate(obj)
    else:
        return [f"Unknown object type: {type(obj)}"]


def validate_all(objects: List[Any], strict: bool = False) -> dict[str, List[str]]:
    """
    Validate multiple objects and return all errors.

    Args:
        objects: List of objects to validate
        strict: If True, raise exception on first error

    Returns:
        Dict mapping object ID to list of errors

    Raises:
        ValidationError: If strict=True and any validation fails
    """
    all_errors = {}

    for obj in objects:
        errors = validate_business_rules(obj)

        if errors:
            obj_id = getattr(obj, f"{type(obj).__name__.lower()}_id", id(obj))
            all_errors[str(obj_id)] = errors

            if strict:
                raise ValidationError(
                    f"Validation failed for {type(obj).__name__}: {', '.join(errors)}"
                )

    return all_errors


# =============================================================================
# Sanitization Utilities
# =============================================================================

def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize string input.

    Args:
        value: String to sanitize
        max_length: Optional max length

    Returns:
        Sanitized string
    """
    # Strip whitespace
    sanitized = value.strip()

    # Truncate if needed
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def sanitize_numeric(value: Any, min_val: float = 0.0, max_val: Optional[float] = None) -> float:
    """
    Sanitize numeric input.

    Args:
        value: Value to sanitize
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Sanitized numeric value

    Raises:
        ValueError: If value cannot be converted to float
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Cannot convert {value} to number")

    # Clamp to range
    if num < min_val:
        num = min_val
    if max_val is not None and num > max_val:
        num = max_val

    return num


# =============================================================================
# Assertion Helpers
# =============================================================================

def assert_valid_hive(hive: Hive) -> None:
    """Assert that hive is valid (raises if not)."""
    HiveValidator.validate_strict(hive)


def assert_valid_colony(colony: Colony) -> None:
    """Assert that colony is valid (raises if not)."""
    ColonyValidator.validate_strict(colony)


def assert_valid_inspection(inspection: Inspection) -> None:
    """Assert that inspection is valid (raises if not)."""
    InspectionValidator.validate_strict(inspection)


def assert_valid_harvest(harvest: HarvestRecord) -> None:
    """Assert that harvest is valid (raises if not)."""
    HarvestValidator.validate_strict(harvest)


# =============================================================================
# Validation Decorators
# =============================================================================

def validate_args(**validators: Callable):
    """
    Decorator to validate function arguments.

    Example:
        @validate_args(hive=assert_valid_hive, colony=assert_valid_colony)
        def add_colony_to_hive(hive: Hive, colony: Colony):
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate each argument
            for arg_name, validator in validators.items():
                if arg_name in bound.arguments:
                    validator(bound.arguments[arg_name])

            return func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# Type Guards
# =============================================================================

def is_valid_health_status(value: Any) -> bool:
    """Check if value is valid HealthStatus."""
    return isinstance(value, HealthStatus)


def is_valid_queen_status(value: Any) -> bool:
    """Check if value is valid QueenStatus."""
    return isinstance(value, QueenStatus)


def is_healthy_colony(colony: Colony) -> bool:
    """Check if colony is in healthy state."""
    return colony.health_status in [HealthStatus.EXCELLENT, HealthStatus.GOOD]


def is_queenless_colony(colony: Colony) -> bool:
    """Check if colony is potentially queenless."""
    return colony.queen_status in [
        QueenStatus.ABSENT,
        QueenStatus.PRESENT_NOT_LAYING
    ]
