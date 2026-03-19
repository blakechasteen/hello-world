"""
Retrieval Result Types for HoloLoom Memory System
==================================================

Design Principles:
- Never return empty containers without explanation
- All retrieval failures have informative messages
- Status enum enables graceful degradation handling
- Metadata captures debugging context

Created: 2025-12-30
Status: Production Ready
W9: Memory Retrieval Stub Remediation (SWOT Weakness)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hololoom.protocols.types import MemoryShard

logger = logging.getLogger(__name__)


class RetrievalStatus(Enum):
    """
    Status codes for retrieval operations.

    Enables callers to understand WHY results are empty or partial,
    not just THAT they are.
    """
    SUCCESS = "success"           # Full results returned
    PARTIAL = "partial"           # Some results, but degraded quality
    FALLBACK = "fallback"         # Using fallback method (e.g., BM25 instead of semantic)
    UNAVAILABLE = "unavailable"   # Backend/service not available
    EMPTY = "empty"               # Query executed but no matches found
    ERROR = "error"               # Unexpected error during retrieval


@dataclass
class RetrievalResultEnhanced:
    """
    Result of memory retrieval with status and explanation.

    Design: Never return bare empty containers without context.

    Attributes:
        shards: Retrieved memory shards (may be empty with explanation in message)
        status: RetrievalStatus indicating result quality
        message: Human-readable explanation (empty for SUCCESS)
        metadata: Additional debugging context
        retrieval_time_ms: Time taken for retrieval operation
        timestamp: When the retrieval occurred

    Examples:
        >>> result = RetrievalResultEnhanced.success(shards)
        >>> result = RetrievalResultEnhanced.fallback(shards, "Embedder unavailable, using BM25")
        >>> result = RetrievalResultEnhanced.empty("obscure query")
        >>> result = RetrievalResultEnhanced.unavailable("Neo4j connection failed")
    """

    shards: list["MemoryShard"]
    status: RetrievalStatus
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # --- Factory Methods (Elegant API) ---

    @classmethod
    def success(
        cls,
        shards: list["MemoryShard"],
        retrieval_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None
    ) -> "RetrievalResultEnhanced":
        """
        Create successful retrieval result.

        Args:
            shards: Retrieved memory shards
            retrieval_time_ms: Time taken for retrieval
            metadata: Optional additional context

        Returns:
            RetrievalResultEnhanced with SUCCESS status
        """
        return cls(
            shards=shards,
            status=RetrievalStatus.SUCCESS,
            message="",
            metadata=metadata or {},
            retrieval_time_ms=retrieval_time_ms
        )

    @classmethod
    def partial(
        cls,
        shards: list["MemoryShard"],
        reason: str,
        retrieval_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None
    ) -> "RetrievalResultEnhanced":
        """
        Create partial result (some results, but degraded).

        Args:
            shards: Retrieved memory shards (partial set)
            reason: Why results are partial
            retrieval_time_ms: Time taken for retrieval
            metadata: Optional additional context

        Returns:
            RetrievalResultEnhanced with PARTIAL status
        """
        meta = metadata or {}
        meta['partial_reason'] = reason
        return cls(
            shards=shards,
            status=RetrievalStatus.PARTIAL,
            message=f"Partial results: {reason}",
            metadata=meta,
            retrieval_time_ms=retrieval_time_ms
        )

    @classmethod
    def fallback(
        cls,
        shards: list["MemoryShard"],
        reason: str,
        fallback_method: str = "unknown",
        retrieval_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None
    ) -> "RetrievalResultEnhanced":
        """
        Create fallback result (using alternative retrieval method).

        Args:
            shards: Retrieved memory shards from fallback method
            reason: Why fallback was used
            fallback_method: Name of fallback method (e.g., "BM25", "keyword")
            retrieval_time_ms: Time taken for retrieval
            metadata: Optional additional context

        Returns:
            RetrievalResultEnhanced with FALLBACK status
        """
        meta = metadata or {}
        meta['fallback_reason'] = reason
        meta['fallback_method'] = fallback_method
        return cls(
            shards=shards,
            status=RetrievalStatus.FALLBACK,
            message=f"Using fallback ({fallback_method}): {reason}",
            metadata=meta,
            retrieval_time_ms=retrieval_time_ms
        )

    @classmethod
    def unavailable(
        cls,
        reason: str,
        backend: str = "unknown",
        metadata: dict[str, Any] | None = None
    ) -> "RetrievalResultEnhanced":
        """
        Create unavailable result (backend/service not accessible).

        Args:
            reason: Why backend is unavailable
            backend: Name of unavailable backend
            metadata: Optional additional context

        Returns:
            RetrievalResultEnhanced with UNAVAILABLE status and empty shards
        """
        meta = metadata or {}
        meta['unavailable_reason'] = reason
        meta['backend'] = backend
        logger.warning(f"Retrieval unavailable ({backend}): {reason}")
        return cls(
            shards=[],
            status=RetrievalStatus.UNAVAILABLE,
            message=f"Backend unavailable ({backend}): {reason}",
            metadata=meta
        )

    @classmethod
    def empty(
        cls,
        query: str,
        retrieval_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None
    ) -> "RetrievalResultEnhanced":
        """
        Create empty result (query executed but no matches).

        Args:
            query: The query that returned no results
            retrieval_time_ms: Time taken for retrieval
            metadata: Optional additional context

        Returns:
            RetrievalResultEnhanced with EMPTY status
        """
        meta = metadata or {}
        meta['query'] = query[:100] if len(query) > 100 else query  # Truncate long queries
        return cls(
            shards=[],
            status=RetrievalStatus.EMPTY,
            message=f"No results for query: {query[:50]}{'...' if len(query) > 50 else ''}",
            metadata=meta,
            retrieval_time_ms=retrieval_time_ms
        )

    @classmethod
    def error(
        cls,
        exception: Exception,
        context: dict[str, Any] | None = None
    ) -> "RetrievalResultEnhanced":
        """
        Create error result (unexpected failure during retrieval).

        Args:
            exception: The exception that occurred
            context: Optional context about what was being attempted

        Returns:
            RetrievalResultEnhanced with ERROR status
        """
        meta = context or {}
        meta['error_type'] = type(exception).__name__
        meta['error_message'] = str(exception)
        logger.error(f"Retrieval error: {exception}", exc_info=True)
        return cls(
            shards=[],
            status=RetrievalStatus.ERROR,
            message=f"Retrieval error: {type(exception).__name__}: {exception}",
            metadata=meta
        )

    # --- Properties for Convenient Status Checking ---

    @property
    def is_ok(self) -> bool:
        """True if retrieval succeeded (SUCCESS status)."""
        return self.status == RetrievalStatus.SUCCESS

    @property
    def is_degraded(self) -> bool:
        """True if results are partial, fallback, or from error recovery."""
        return self.status in (
            RetrievalStatus.PARTIAL,
            RetrievalStatus.FALLBACK
        )

    @property
    def is_empty(self) -> bool:
        """True if no results were found (EMPTY or UNAVAILABLE)."""
        return self.status in (
            RetrievalStatus.EMPTY,
            RetrievalStatus.UNAVAILABLE,
            RetrievalStatus.ERROR
        )

    @property
    def has_results(self) -> bool:
        """True if shards list is non-empty."""
        return len(self.shards) > 0

    @property
    def count(self) -> int:
        """Number of shards returned."""
        return len(self.shards)

    # --- Utility Methods ---

    def unwrap_or(self, default: list["MemoryShard"]) -> list["MemoryShard"]:
        """
        Return shards if successful, otherwise return default.

        Rust-style unwrap pattern for graceful handling.

        Args:
            default: Default value if retrieval failed

        Returns:
            Retrieved shards or default
        """
        return self.shards if self.has_results else default

    def log_status(self, logger_instance: logging.Logger | None = None):
        """
        Log retrieval status at appropriate level.

        Args:
            logger_instance: Logger to use (defaults to module logger)
        """
        log = logger_instance or logger

        if self.status == RetrievalStatus.SUCCESS:
            log.debug(f"Retrieval success: {self.count} results in {self.retrieval_time_ms:.1f}ms")
        elif self.status == RetrievalStatus.EMPTY:
            log.info(f"Retrieval empty: {self.message}")
        elif self.status == RetrievalStatus.FALLBACK:
            log.warning(f"Retrieval fallback: {self.message}")
        elif self.status == RetrievalStatus.PARTIAL:
            log.warning(f"Retrieval partial: {self.message}")
        elif self.status == RetrievalStatus.UNAVAILABLE:
            log.error(f"Retrieval unavailable: {self.message}")
        elif self.status == RetrievalStatus.ERROR:
            log.error(f"Retrieval error: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary (JSON-safe).

        Returns:
            Dictionary representation
        """
        return {
            "status": self.status.value,
            "message": self.message,
            "count": self.count,
            "retrieval_time_ms": self.retrieval_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            # Note: shards not included by default (may be large)
        }

    def __repr__(self) -> str:
        return (
            f"RetrievalResultEnhanced("
            f"status={self.status.value}, "
            f"count={self.count}, "
            f"message={self.message!r})"
        )


# --- Graph Traversal Result Type ---

@dataclass
class TraversalResult:
    """
    Result of graph traversal operation.

    Addresses W9: GraphRetriever.traverse() returning bare {} silently.
    """

    nodes: dict[str, float]  # Node ID → score
    status: RetrievalStatus
    message: str
    hops_executed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        nodes: dict[str, float],
        hops_executed: int = 0,
        metadata: dict[str, Any] | None = None
    ) -> "TraversalResult":
        """Create successful traversal result."""
        return cls(
            nodes=nodes,
            status=RetrievalStatus.SUCCESS,
            message="",
            hops_executed=hops_executed,
            metadata=metadata or {}
        )

    @classmethod
    def empty_graph(
        cls,
        start_entity: str,
        metadata: dict[str, Any] | None = None
    ) -> "TraversalResult":
        """Create result when start entity not in graph."""
        meta = metadata or {}
        meta['start_entity'] = start_entity
        return cls(
            nodes={},
            status=RetrievalStatus.EMPTY,
            message=f"Start entity '{start_entity}' not found in knowledge graph",
            metadata=meta
        )

    @classmethod
    def unavailable(
        cls,
        reason: str,
        metadata: dict[str, Any] | None = None
    ) -> "TraversalResult":
        """Create result when graph is unavailable."""
        meta = metadata or {}
        meta['unavailable_reason'] = reason
        return cls(
            nodes={},
            status=RetrievalStatus.UNAVAILABLE,
            message=f"Knowledge graph unavailable: {reason}",
            metadata=meta
        )

    @property
    def has_nodes(self) -> bool:
        """True if traversal found any nodes."""
        return len(self.nodes) > 0

    @property
    def is_ok(self) -> bool:
        """True if traversal succeeded."""
        return self.status == RetrievalStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        result = {
            "nodes": list(self.nodes.keys()),
            "scores": self.nodes,
            "status": self.status.value,
            "message": self.message,
            "hops_executed": self.hops_executed,
        }
        if self.message:
            result["warning"] = self.message
        return result


# --- Convenience Functions ---

def ensure_retrieval_result(
    result: Any,
    default_status: RetrievalStatus = RetrievalStatus.EMPTY
) -> RetrievalResultEnhanced:
    """
    Ensure a result is wrapped in RetrievalResultEnhanced.

    Useful for migrating legacy code that returns bare lists.

    Args:
        result: Result to wrap (list, RetrievalResultEnhanced, or other)
        default_status: Status to use if wrapping a list

    Returns:
        RetrievalResultEnhanced instance
    """
    if isinstance(result, RetrievalResultEnhanced):
        return result

    if isinstance(result, list):
        if len(result) > 0:
            return RetrievalResultEnhanced.success(result)
        else:
            return RetrievalResultEnhanced(
                shards=[],
                status=default_status,
                message="Wrapped bare empty list (legacy code)",
                metadata={"wrapped": True}
            )

    # Unknown type - log warning and return empty
    logger.warning(f"ensure_retrieval_result: Unknown result type {type(result)}")
    return RetrievalResultEnhanced.error(
        ValueError(f"Cannot wrap result of type {type(result)}")
    )


# --- Exports ---

__all__ = [
    "RetrievalStatus",
    "RetrievalResultEnhanced",
    "TraversalResult",
    "ensure_retrieval_result",
]
