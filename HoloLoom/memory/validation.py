"""
Memory System Input Validation & Sanitization
==============================================

Week 8A: Production-grade error handling for memory systems.

Provides comprehensive input validation, sanitization, and type checking
for all memory operations. Ensures defensive programming across the board.

Key Features:
- Input validation (query strings, confidence scores, timestamps)
- Sanitization (remove control chars, enforce length limits)
- Type checking (coerce types, handle edge cases)
- Error recovery strategies

Philosophy:
"Validate early, fail fast, recover gracefully"

Author: HoloLoom Memory Team
Date: 2025-11-18 (Week 8A: Error Handling & Defensive Programming)
"""

from typing import Any, Optional, List, Dict, Union
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class ValidationConfig:
    """Configuration for validation behavior."""

    # Query validation
    MAX_QUERY_LENGTH: int = 10000
    MIN_QUERY_LENGTH: int = 1

    # Concept validation
    MAX_CONCEPT_LENGTH: int = 1000
    MIN_CONCEPT_LENGTH: int = 1

    # Confidence validation
    MIN_CONFIDENCE: float = 0.0
    MAX_CONFIDENCE: float = 1.0

    # Timestamp validation
    MAX_TIMESTAMP_AGE_DAYS: int = 3650  # 10 years
    ALLOW_FUTURE_TIMESTAMPS: bool = False

    # Memory ID validation
    MAX_MEMORY_ID_LENGTH: int = 256
    MEMORY_ID_PATTERN: str = r'^[a-zA-Z0-9_\-\.]+$'

    # Entity validation
    MAX_ENTITIES_PER_MEMORY: int = 100
    MAX_ENTITY_LENGTH: int = 200

    # General
    STRICT_MODE: bool = False  # Raise errors vs log warnings


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(ValueError):
    """Base validation error."""
    pass


class QueryValidationError(ValidationError):
    """Query validation failed."""
    pass


class ConfidenceValidationError(ValidationError):
    """Confidence validation failed."""
    pass


class TimestampValidationError(ValidationError):
    """Timestamp validation failed."""
    pass


class MemoryIdValidationError(ValidationError):
    """Memory ID validation failed."""
    pass


class EntityValidationError(ValidationError):
    """Entity validation failed."""
    pass


# ============================================================================
# Core Validators
# ============================================================================

class MemoryValidator:
    """
    Memory system input validator.

    Provides static validation methods for all memory inputs.

    Usage:
        >>> validator = MemoryValidator()
        >>> query = validator.validate_query("What is Thompson Sampling?")
        >>> confidence = validator.validate_confidence(0.95)
        >>> timestamp = validator.validate_timestamp(datetime.now())
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator.

        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or ValidationConfig()

    # ========================================================================
    # Query Validation
    # ========================================================================

    def validate_query(self, query: Any, allow_empty: bool = False) -> str:
        """
        Validate and sanitize query text.

        Args:
            query: Query text (any type, will be coerced to string)
            allow_empty: Allow empty queries (default: False)

        Returns:
            Validated and sanitized query string

        Raises:
            QueryValidationError: If validation fails in strict mode
        """
        # Type coercion
        if query is None:
            if allow_empty:
                return ""
            else:
                self._handle_error(
                    QueryValidationError("Query cannot be None"),
                    default=""
                )
                return ""

        # Convert to string
        if not isinstance(query, str):
            logger.warning(f"Query is not string (type: {type(query)}), converting")
            query = str(query)

        # Sanitize (remove control characters)
        query = self._sanitize_text(query)

        # Strip whitespace
        query = query.strip()

        # Check empty
        if not query and not allow_empty:
            self._handle_error(
                QueryValidationError("Query cannot be empty"),
                default="[empty query]"
            )
            return "[empty query]"

        # Check length
        if len(query) > self.config.MAX_QUERY_LENGTH:
            logger.warning(
                f"Query too long ({len(query)} > {self.config.MAX_QUERY_LENGTH}), truncating"
            )
            query = query[:self.config.MAX_QUERY_LENGTH]

        if len(query) < self.config.MIN_QUERY_LENGTH and not allow_empty:
            self._handle_error(
                QueryValidationError(
                    f"Query too short ({len(query)} < {self.config.MIN_QUERY_LENGTH})"
                ),
                default="[invalid query]"
            )
            return "[invalid query]"

        return query

    # ========================================================================
    # Confidence Validation
    # ========================================================================

    def validate_confidence(self, confidence: Any) -> float:
        """
        Validate confidence score.

        Ensures confidence is in [0.0, 1.0] range.

        Args:
            confidence: Confidence value (any numeric type)

        Returns:
            Validated confidence in [0.0, 1.0]

        Raises:
            ConfidenceValidationError: If validation fails in strict mode
        """
        # Type coercion
        if confidence is None:
            logger.warning("Confidence is None, defaulting to 0.0")
            return 0.0

        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as e:
            self._handle_error(
                ConfidenceValidationError(f"Cannot convert confidence to float: {e}"),
                default=0.0
            )
            return 0.0

        # Check for NaN or Inf
        if not self._is_finite(confidence):
            logger.warning(f"Confidence is not finite ({confidence}), defaulting to 0.0")
            return 0.0

        # Clamp to [0.0, 1.0]
        if confidence < self.config.MIN_CONFIDENCE:
            logger.warning(
                f"Confidence below minimum ({confidence} < {self.config.MIN_CONFIDENCE}), "
                f"clamping to {self.config.MIN_CONFIDENCE}"
            )
            confidence = self.config.MIN_CONFIDENCE

        if confidence > self.config.MAX_CONFIDENCE:
            logger.warning(
                f"Confidence above maximum ({confidence} > {self.config.MAX_CONFIDENCE}), "
                f"clamping to {self.config.MAX_CONFIDENCE}"
            )
            confidence = self.config.MAX_CONFIDENCE

        return confidence

    # ========================================================================
    # Timestamp Validation
    # ========================================================================

    def validate_timestamp(self, timestamp: Any) -> datetime:
        """
        Validate timestamp.

        Checks:
        - Not in the future (configurable)
        - Not too old (> MAX_TIMESTAMP_AGE_DAYS)
        - Valid datetime object

        Args:
            timestamp: Timestamp (datetime, float, int, or string)

        Returns:
            Validated datetime object

        Raises:
            TimestampValidationError: If validation fails in strict mode
        """
        # Type coercion
        if timestamp is None:
            logger.warning("Timestamp is None, using current time")
            return datetime.now()

        # Convert to datetime
        if isinstance(timestamp, datetime):
            dt = timestamp
        elif isinstance(timestamp, (int, float)):
            try:
                dt = datetime.fromtimestamp(timestamp)
            except (ValueError, OSError) as e:
                self._handle_error(
                    TimestampValidationError(f"Invalid timestamp value: {e}"),
                    default=datetime.now()
                )
                return datetime.now()
        elif isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
            except ValueError as e:
                self._handle_error(
                    TimestampValidationError(f"Invalid ISO timestamp: {e}"),
                    default=datetime.now()
                )
                return datetime.now()
        else:
            self._handle_error(
                TimestampValidationError(f"Unknown timestamp type: {type(timestamp)}"),
                default=datetime.now()
            )
            return datetime.now()

        # Check if future
        now = datetime.now()
        if dt > now and not self.config.ALLOW_FUTURE_TIMESTAMPS:
            logger.warning(
                f"Timestamp is in the future ({dt} > {now}), using current time"
            )
            return now

        # Check if too old
        max_age = timedelta(days=self.config.MAX_TIMESTAMP_AGE_DAYS)
        if now - dt > max_age:
            logger.warning(
                f"Timestamp is very old ({dt}), may be invalid"
            )
            # Don't fail, just warn (old memories are valid)

        return dt

    # ========================================================================
    # Memory ID Validation
    # ========================================================================

    def validate_memory_id(self, memory_id: Any) -> str:
        """
        Validate and sanitize memory ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Validated memory ID

        Raises:
            MemoryIdValidationError: If validation fails in strict mode
        """
        # Type coercion
        if memory_id is None:
            self._handle_error(
                MemoryIdValidationError("Memory ID cannot be None"),
                default="invalid_id"
            )
            return "invalid_id"

        if not isinstance(memory_id, str):
            logger.warning(f"Memory ID is not string (type: {type(memory_id)}), converting")
            memory_id = str(memory_id)

        # Sanitize
        memory_id = memory_id.strip()

        # Check empty
        if not memory_id:
            self._handle_error(
                MemoryIdValidationError("Memory ID cannot be empty"),
                default="invalid_id"
            )
            return "invalid_id"

        # Check length
        if len(memory_id) > self.config.MAX_MEMORY_ID_LENGTH:
            logger.warning(
                f"Memory ID too long ({len(memory_id)} > {self.config.MAX_MEMORY_ID_LENGTH}), "
                f"truncating"
            )
            memory_id = memory_id[:self.config.MAX_MEMORY_ID_LENGTH]

        # Check pattern (alphanumeric + _ - .)
        if not re.match(self.config.MEMORY_ID_PATTERN, memory_id):
            logger.warning(
                f"Memory ID contains invalid characters: {memory_id}, sanitizing"
            )
            memory_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', memory_id)

        return memory_id

    # ========================================================================
    # Concept Validation
    # ========================================================================

    def validate_concept_text(self, text: Any, allow_empty: bool = False) -> str:
        """
        Validate and sanitize concept text.

        Args:
            text: Concept text
            allow_empty: Allow empty text

        Returns:
            Validated concept text

        Raises:
            ValidationError: If validation fails in strict mode
        """
        # Type coercion
        if text is None:
            if allow_empty:
                return ""
            else:
                self._handle_error(
                    ValidationError("Concept text cannot be None"),
                    default=""
                )
                return ""

        if not isinstance(text, str):
            logger.warning(f"Concept text is not string (type: {type(text)}), converting")
            text = str(text)

        # Sanitize
        text = self._sanitize_text(text)
        text = text.strip()

        # Check empty
        if not text and not allow_empty:
            self._handle_error(
                ValidationError("Concept text cannot be empty"),
                default="[empty concept]"
            )
            return "[empty concept]"

        # Check length
        if len(text) > self.config.MAX_CONCEPT_LENGTH:
            logger.warning(
                f"Concept text too long ({len(text)} > {self.config.MAX_CONCEPT_LENGTH}), "
                f"truncating"
            )
            text = text[:self.config.MAX_CONCEPT_LENGTH]

        if len(text) < self.config.MIN_CONCEPT_LENGTH and not allow_empty:
            self._handle_error(
                ValidationError(
                    f"Concept text too short ({len(text)} < {self.config.MIN_CONCEPT_LENGTH})"
                ),
                default="[invalid concept]"
            )
            return "[invalid concept]"

        return text

    # ========================================================================
    # Entity Validation
    # ========================================================================

    def validate_entities(self, entities: Any) -> List[str]:
        """
        Validate entity list.

        Args:
            entities: List of entities (or single entity)

        Returns:
            Validated entity list

        Raises:
            EntityValidationError: If validation fails in strict mode
        """
        # Type coercion
        if entities is None:
            return []

        # Convert single entity to list
        if isinstance(entities, str):
            entities = [entities]

        # Check if iterable
        if not isinstance(entities, (list, tuple, set)):
            logger.warning(f"Entities is not iterable (type: {type(entities)}), converting")
            entities = [str(entities)]

        # Validate each entity
        validated_entities = []
        for entity in entities:
            if entity is None:
                continue

            if not isinstance(entity, str):
                entity = str(entity)

            # Sanitize
            entity = self._sanitize_text(entity).strip()

            if not entity:
                continue  # Skip empty entities

            # Check length
            if len(entity) > self.config.MAX_ENTITY_LENGTH:
                logger.warning(
                    f"Entity too long ({len(entity)} > {self.config.MAX_ENTITY_LENGTH}), "
                    f"truncating"
                )
                entity = entity[:self.config.MAX_ENTITY_LENGTH]

            validated_entities.append(entity)

        # Check count
        if len(validated_entities) > self.config.MAX_ENTITIES_PER_MEMORY:
            logger.warning(
                f"Too many entities ({len(validated_entities)} > "
                f"{self.config.MAX_ENTITIES_PER_MEMORY}), truncating"
            )
            validated_entities = validated_entities[:self.config.MAX_ENTITIES_PER_MEMORY]

        return validated_entities

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _sanitize_text(self, text: str) -> str:
        """
        Remove control characters and normalize whitespace.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        # Remove control characters (except newline, tab)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text

    def _is_finite(self, value: float) -> bool:
        """
        Check if float is finite (not NaN or Inf).

        Args:
            value: Float value

        Returns:
            True if finite
        """
        import math
        return not (math.isnan(value) or math.isinf(value))

    def _handle_error(
        self,
        error: Exception,
        default: Any = None
    ) -> None:
        """
        Handle validation error based on strict mode.

        Args:
            error: Validation error
            default: Default value (for logging)

        Raises:
            error: If strict mode is enabled
        """
        if self.config.STRICT_MODE:
            raise error
        else:
            logger.warning(f"Validation error (non-strict): {error}, using default: {default}")


# ============================================================================
# Batch Validation
# ============================================================================

class BatchValidator:
    """
    Batch validation utilities.

    Validates collections of inputs efficiently.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize batch validator."""
        self.validator = MemoryValidator(config)

    def validate_queries(
        self,
        queries: List[Any],
        allow_empty: bool = False
    ) -> List[str]:
        """
        Validate multiple queries.

        Args:
            queries: List of queries
            allow_empty: Allow empty queries

        Returns:
            List of validated queries
        """
        validated = []
        for i, query in enumerate(queries):
            try:
                validated.append(self.validator.validate_query(query, allow_empty))
            except Exception as e:
                logger.error(f"Failed to validate query {i}: {e}")
                validated.append("[invalid query]")

        return validated

    def validate_confidences(self, confidences: List[Any]) -> List[float]:
        """
        Validate multiple confidence scores.

        Args:
            confidences: List of confidence values

        Returns:
            List of validated confidences
        """
        validated = []
        for i, confidence in enumerate(confidences):
            try:
                validated.append(self.validator.validate_confidence(confidence))
            except Exception as e:
                logger.error(f"Failed to validate confidence {i}: {e}")
                validated.append(0.0)

        return validated

    def validate_timestamps(self, timestamps: List[Any]) -> List[datetime]:
        """
        Validate multiple timestamps.

        Args:
            timestamps: List of timestamps

        Returns:
            List of validated timestamps
        """
        validated = []
        for i, timestamp in enumerate(timestamps):
            try:
                validated.append(self.validator.validate_timestamp(timestamp))
            except Exception as e:
                logger.error(f"Failed to validate timestamp {i}: {e}")
                validated.append(datetime.now())

        return validated


# ============================================================================
# Factory Functions
# ============================================================================

def create_validator(strict: bool = False) -> MemoryValidator:
    """
    Create memory validator with configuration.

    Args:
        strict: Enable strict mode (raise errors instead of warnings)

    Returns:
        MemoryValidator instance
    """
    config = ValidationConfig()
    config.STRICT_MODE = strict
    return MemoryValidator(config)


def create_batch_validator(strict: bool = False) -> BatchValidator:
    """
    Create batch validator with configuration.

    Args:
        strict: Enable strict mode

    Returns:
        BatchValidator instance
    """
    config = ValidationConfig()
    config.STRICT_MODE = strict
    return BatchValidator(config)
