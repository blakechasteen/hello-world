"""
HoloLoom Shuttle - Exception Hierarchy

Custom exceptions for Shuttle operations with clear error messages
and graceful degradation support.

Created: 2025-01-21
"""

from typing import Optional


class ShuttleError(Exception):
    """
    Base exception for all Shuttle errors.

    All Shuttle-specific exceptions inherit from this to enable
    targeted exception handling.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(ShuttleError):
    """
    Raised when Shuttle configuration is invalid or incomplete.

    Examples:
    - Required backend not available (HYBRID mode but Neo4j down)
    - Invalid trajectory registry
    - Missing required parameters
    """
    pass


class BackendUnavailableError(ShuttleError):
    """
    Raised when a required backend service is unavailable.

    Examples:
    - Neo4j connection refused
    - Qdrant not responding
    - Network timeout

    This error triggers graceful degradation to a fallback mode.
    """

    def __init__(
        self,
        backend_name: str,
        reason: str,
        fallback_mode: Optional[str] = None
    ):
        self.backend_name = backend_name
        self.reason = reason
        self.fallback_mode = fallback_mode

        message = f"Backend '{backend_name}' unavailable: {reason}"
        if fallback_mode:
            message += f" (falling back to {fallback_mode} mode)"

        super().__init__(
            message,
            details={
                "backend": backend_name,
                "reason": reason,
                "fallback": fallback_mode,
            }
        )


class TimeoutError(ShuttleError):
    """
    Raised when an operation exceeds its time budget.

    Examples:
    - MCTS search exceeds timeout
    - Warp search takes too long
    - Yarn graph query times out
    """

    def __init__(
        self,
        operation: str,
        timeout_ms: int,
        elapsed_ms: Optional[float] = None
    ):
        self.operation = operation
        self.timeout_ms = timeout_ms
        self.elapsed_ms = elapsed_ms

        message = f"Operation '{operation}' exceeded {timeout_ms}ms timeout"
        if elapsed_ms is not None:
            message += f" (took {elapsed_ms:.1f}ms)"

        super().__init__(
            message,
            details={
                "operation": operation,
                "timeout_ms": timeout_ms,
                "elapsed_ms": elapsed_ms,
            }
        )


class EntityExtractionError(ShuttleError):
    """
    Raised when entity extraction from Warp results fails.

    Examples:
    - No entities found in search results
    - Entity extraction backend unavailable (spaCy not installed)
    - Malformed entity data
    """

    def __init__(
        self,
        reason: str,
        extraction_method: Optional[str] = None,
        fallback_available: bool = False
    ):
        self.reason = reason
        self.extraction_method = extraction_method
        self.fallback_available = fallback_available

        message = f"Entity extraction failed: {reason}"
        if extraction_method:
            message += f" (method: {extraction_method})"
        if fallback_available:
            message += " (fallback available)"

        super().__init__(
            message,
            details={
                "reason": reason,
                "method": extraction_method,
                "fallback": fallback_available,
            }
        )


class TraversalError(ShuttleError):
    """
    Raised when graph traversal encounters an error.

    Examples:
    - Neighbor map is empty
    - MCTS gets stuck in infinite loop
    - All trajectories fail
    """
    pass


class WarpSearchError(ShuttleError):
    """
    Raised when Warp (vector search) fails.

    Examples:
    - Qdrant connection lost
    - Invalid query embedding
    - Search timeout
    """
    pass


class YarnTraversalError(ShuttleError):
    """
    Raised when Yarn (graph) traversal fails.

    Examples:
    - Neo4j query error
    - Invalid Cypher syntax
    - Graph traversal timeout
    """
    pass
