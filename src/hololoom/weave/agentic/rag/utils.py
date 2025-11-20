"""
Shared utilities for RAG features.

Provides common patterns to reduce code duplication:
- Result serialization
- Error formatting
- Validation helpers
- Statistics tracking
- Async execution patterns
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Coroutine
from dataclasses import dataclass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)


# ============================================================================
# Result Serialization
# ============================================================================

def result_to_dict_base(result) -> Dict[str, Any]:
    """
    Base serialization for all RAGResult subclasses.

    Args:
        result: RAGResult or subclass instance

    Returns:
        Dictionary with base fields

    Example:
        >>> base = result_to_dict_base(result)
        >>> base.update({'custom_field': value})
        >>> return base
    """
    return {
        'response': result.response,
        'sources': result.sources,
        'confidence': result.confidence,
        'reasoning_mode': result.reasoning_mode,
        'metadata': result.metadata,
    }


# ============================================================================
# Error Formatting
# ============================================================================

def format_import_error(package: str, feature: str) -> str:
    """
    Format standard import error message.

    Args:
        package: Package name (e.g., "sentence-transformers")
        feature: Feature name (e.g., "reranking")

    Returns:
        Formatted error message

    Example:
        >>> raise ImportError(format_import_error("openai", "OpenAI embeddings"))
    """
    return (
        f"{package} not installed. Install with:\n"
        f"    pip install {package}\n"
        f"Or disable {feature} via configuration."
    )


def format_runtime_error(component: str, reason: str, recovery: str) -> str:
    """
    Format runtime error with recovery suggestion.

    Args:
        component: Component name
        reason: Why it failed
        recovery: How to fix

    Returns:
        Formatted error message

    Example:
        >>> raise RuntimeError(format_runtime_error(
        ...     "Database", "Connection timeout", "Check server is running"
        ... ))
    """
    return (
        f"{component} failed: {reason}\n"
        f"Recovery: {recovery}"
    )


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_array_shape(
    array: Any,
    expected_shape: tuple,
    name: str
) -> bool:
    """
    Validate numpy array shape with logging.

    Args:
        array: Array to validate
        expected_shape: Expected shape tuple
        name: Array name for error messages

    Returns:
        True if valid, False otherwise

    Example:
        >>> embeddings = provider.encode(texts)
        >>> if not validate_array_shape(embeddings, (2, 384), "embeddings"):
        ...     return False
    """
    if not NUMPY_AVAILABLE:
        logger.warning("NumPy not available, skipping array validation")
        return True

    if not isinstance(array, np.ndarray):
        logger.error(f"{name} should be np.ndarray, got {type(array)}")
        return False

    if array.shape != expected_shape:
        logger.error(
            f"{name} shape incorrect. "
            f"Expected {expected_shape}, got {array.shape}"
        )
        return False

    return True


def clamp_top_k(top_k: int, max_value: int, min_value: int = 1) -> int:
    """
    Clamp top_k to valid range.

    Args:
        top_k: Requested top_k
        max_value: Maximum allowed value
        min_value: Minimum allowed value (default: 1)

    Returns:
        Clamped value

    Example:
        >>> top_k = clamp_top_k(100, len(documents))  # Won't exceed doc count
    """
    return max(min_value, min(top_k, max_value))


def validate_not_empty(
    items: List[Any],
    name: str,
    min_length: int = 1
) -> bool:
    """
    Validate list is not empty.

    Args:
        items: List to validate
        name: List name for error messages
        min_length: Minimum required length

    Returns:
        True if valid, False otherwise

    Example:
        >>> if not validate_not_empty(documents, "documents"):
        ...     return []
    """
    if not items or len(items) < min_length:
        logger.error(
            f"{name} must have at least {min_length} items, "
            f"got {len(items) if items else 0}"
        )
        return False
    return True


# ============================================================================
# Statistics Tracking
# ============================================================================

@dataclass
class QueryStats:
    """
    Track query statistics.

    Provides automatic computation of derived metrics like
    average latency and success rate.

    Example:
        >>> stats = QueryStats()
        >>> stats.update(success=True, latency_ms=150.0)
        >>> print(f"Success rate: {stats.success_rate:.1%}")
        >>> print(f"Avg latency: {stats.avg_latency_ms:.1f}ms")
    """
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency across all queries."""
        return (
            self.total_latency_ms / self.total_queries
            if self.total_queries > 0 else 0.0
        )

    @property
    def success_rate(self) -> float:
        """Success rate (0.0-1.0)."""
        return (
            self.successful_queries / self.total_queries
            if self.total_queries > 0 else 0.0
        )

    def update(self, success: bool, latency_ms: float) -> None:
        """
        Update statistics with new query result.

        Args:
            success: Whether query succeeded
            latency_ms: Query latency in milliseconds
        """
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        self.total_latency_ms += latency_ms

    def to_dict(self) -> Dict[str, Any]:
        """Serialize statistics."""
        return {
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'failed_queries': self.failed_queries,
            'avg_latency_ms': self.avg_latency_ms,
            'success_rate': self.success_rate,
        }


# ============================================================================
# Async Execution Patterns
# ============================================================================

async def run_parallel_with_timeout(
    tasks: List[Coroutine],
    timeout: float,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Run tasks in parallel with timeout.

    Args:
        tasks: List of coroutines to run
        timeout: Timeout in seconds
        return_exceptions: Whether to return exceptions or raise

    Returns:
        List of results (or exceptions if return_exceptions=True)

    Raises:
        asyncio.TimeoutError: If timeout exceeded

    Example:
        >>> results = await run_parallel_with_timeout(
        ...     [query1(), query2(), query3()],
        ...     timeout=30.0
        ... )
    """
    try:
        return await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=return_exceptions),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Parallel execution timed out after {timeout}s")
        raise


async def run_with_fallback(
    primary: Coroutine,
    fallback: Coroutine,
    timeout: Optional[float] = None
) -> Any:
    """
    Run primary task with fallback if it fails.

    Args:
        primary: Primary coroutine
        fallback: Fallback coroutine (called if primary fails)
        timeout: Optional timeout for primary task

    Returns:
        Result from primary or fallback

    Example:
        >>> result = await run_with_fallback(
        ...     stream_from_llm(query),
        ...     regular_query(query),
        ...     timeout=5.0
        ... )
    """
    try:
        if timeout:
            return await asyncio.wait_for(primary, timeout=timeout)
        else:
            return await primary
    except Exception as e:
        logger.info(f"Primary execution failed: {e}, trying fallback")
        return await fallback


# ============================================================================
# String Utilities
# ============================================================================

def build_path_string(entities: List[str], relationships: List[str]) -> str:
    """
    Build graph path string efficiently.

    Uses list concatenation + join instead of repeated string concatenation
    for O(n) instead of O(n²) complexity.

    Args:
        entities: List of entity names
        relationships: List of edge types

    Returns:
        Formatted path string

    Example:
        >>> build_path_string(
        ...     ["A", "B", "C"],
        ...     ["USES", "IS_A"]
        ... )
        "A -[USES]-> B -[IS_A]-> C"
    """
    if not entities:
        return ""

    parts = [entities[0]]
    for i, rel in enumerate(relationships):
        if i + 1 < len(entities):
            parts.extend([f" -[{rel}]-> ", entities[i+1]])

    return "".join(parts)


def deduplicate_preserving_order(items: List[Any]) -> List[Any]:
    """
    Deduplicate list while preserving order.

    Uses set for O(1) lookup instead of O(n) `in` operator.

    Args:
        items: List with potential duplicates

    Returns:
        List with duplicates removed, order preserved

    Example:
        >>> deduplicate_preserving_order([1, 2, 2, 3, 1])
        [1, 2, 3]
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Result serialization
    'result_to_dict_base',

    # Error formatting
    'format_import_error',
    'format_runtime_error',

    # Validation
    'validate_array_shape',
    'clamp_top_k',
    'validate_not_empty',

    # Statistics
    'QueryStats',

    # Async patterns
    'run_parallel_with_timeout',
    'run_with_fallback',

    # String utilities
    'build_path_string',
    'deduplicate_preserving_order',
]
