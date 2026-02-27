"""
Retriever Registry with Automatic Fallback Chain
=================================================

Design Principles:
- Register retrievers with priority ordering
- Automatic fallback when primary retriever fails
- Track retriever health and availability
- Never fail silently - always report fallback path taken

Created: 2025-12-30
Status: Production Ready
W9: Memory Retrieval Stub Remediation (SWOT Weakness)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable, Union, Tuple, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import logging
import asyncio

if TYPE_CHECKING:
    from HoloLoom.core.protocols.types import MemoryShard

from HoloLoom.core.memory.retrieval_result import (
    RetrievalResultEnhanced,
    RetrievalStatus,
    ensure_retrieval_result
)

logger = logging.getLogger(__name__)


class RetrieverType(Enum):
    """Types of retrievers in the registry."""
    SEMANTIC = "semantic"      # sentence-transformers based
    GRAPH = "graph"            # Knowledge graph traversal
    KEYWORD = "keyword"        # BM25/TF-IDF keyword search
    HYBRID = "hybrid"          # Combined retrieval
    CUSTOM = "custom"          # User-defined retriever


class RetrieverHealth(Enum):
    """Health status of a retriever."""
    HEALTHY = "healthy"          # Working normally
    DEGRADED = "degraded"        # Partial functionality
    UNAVAILABLE = "unavailable"  # Cannot be used
    UNKNOWN = "unknown"          # Not yet tested


@dataclass
class RetrieverInfo:
    """Information about a registered retriever."""
    name: str
    retriever_type: RetrieverType
    priority: int  # Lower = higher priority (1 is highest)
    retrieve_fn: Callable[..., Awaitable[Any]]
    health: RetrieverHealth = RetrieverHealth.UNKNOWN
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    total_calls: int = 0
    total_successes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)."""
        if self.total_calls == 0:
            return 0.0
        return self.total_successes / self.total_calls

    @property
    def is_available(self) -> bool:
        """Check if retriever can be used."""
        return self.health in (RetrieverHealth.HEALTHY, RetrieverHealth.DEGRADED, RetrieverHealth.UNKNOWN)


@dataclass
class FallbackResult:
    """Result of retrieval with fallback chain tracking."""
    result: RetrievalResultEnhanced
    retriever_used: str
    fallback_path: List[str]  # Chain of retrievers tried
    fallback_reasons: Dict[str, str]  # Why each retriever failed
    total_time_ms: float

    @property
    def used_fallback(self) -> bool:
        """True if primary retriever was not used."""
        return len(self.fallback_path) > 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "retriever_used": self.retriever_used,
            "used_fallback": self.used_fallback,
            "fallback_path": self.fallback_path,
            "fallback_reasons": self.fallback_reasons,
            "total_time_ms": self.total_time_ms,
            "result_status": self.result.status.value,
            "result_count": self.result.count
        }


class RetrieverRegistry:
    """
    Registry for retrieval methods with automatic fallback.

    Design: Never fail silently. Always try fallback chain.

    Example:
        >>> registry = RetrieverRegistry()
        >>> registry.register("semantic", semantic_retriever.retrieve, priority=1)
        >>> registry.register("keyword", keyword_retriever.retrieve, priority=2)
        >>> result = await registry.retrieve(query, memories)
        >>> if result.used_fallback:
        ...     print(f"Fell back to: {result.retriever_used}")
    """

    def __init__(
        self,
        max_consecutive_failures: int = 3,
        health_check_interval_seconds: float = 300.0,
        enable_auto_recovery: bool = True
    ):
        """
        Initialize the registry.

        Args:
            max_consecutive_failures: Mark retriever unavailable after N failures
            health_check_interval_seconds: Time between health checks
            enable_auto_recovery: Re-enable retrievers after cooldown
        """
        self._retrievers: Dict[str, RetrieverInfo] = {}
        self._priority_order: List[str] = []  # Sorted by priority
        self.max_consecutive_failures = max_consecutive_failures
        self.health_check_interval = health_check_interval_seconds
        self.enable_auto_recovery = enable_auto_recovery
        self._lock = asyncio.Lock()

    def register(
        self,
        name: str,
        retrieve_fn: Callable[..., Awaitable[Any]],
        retriever_type: RetrieverType = RetrieverType.CUSTOM,
        priority: int = 10,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a retriever.

        Args:
            name: Unique identifier for the retriever
            retrieve_fn: Async function that performs retrieval
            retriever_type: Type classification
            priority: Lower = higher priority (1 is highest)
            metadata: Additional information about the retriever
        """
        if name in self._retrievers:
            logger.warning(f"Overwriting existing retriever: {name}")

        info = RetrieverInfo(
            name=name,
            retriever_type=retriever_type,
            priority=priority,
            retrieve_fn=retrieve_fn,
            metadata=metadata or {}
        )

        self._retrievers[name] = info
        self._update_priority_order()
        logger.info(f"Registered retriever: {name} (priority={priority}, type={retriever_type.value})")

    def unregister(self, name: str) -> bool:
        """
        Remove a retriever from the registry.

        Returns:
            True if retriever was removed, False if not found
        """
        if name in self._retrievers:
            del self._retrievers[name]
            self._update_priority_order()
            logger.info(f"Unregistered retriever: {name}")
            return True
        return False

    def _update_priority_order(self) -> None:
        """Update the priority-sorted list of retriever names."""
        self._priority_order = sorted(
            self._retrievers.keys(),
            key=lambda n: self._retrievers[n].priority
        )

    def get_available_retrievers(self) -> List[str]:
        """Get list of available retrievers in priority order."""
        return [
            name for name in self._priority_order
            if self._retrievers[name].is_available
        ]

    def get_retriever_info(self, name: str) -> Optional[RetrieverInfo]:
        """Get information about a specific retriever."""
        return self._retrievers.get(name)

    def set_health(self, name: str, health: RetrieverHealth) -> None:
        """Manually set retriever health status."""
        if name in self._retrievers:
            self._retrievers[name].health = health
            self._retrievers[name].last_health_check = datetime.utcnow()
            if health == RetrieverHealth.HEALTHY:
                self._retrievers[name].consecutive_failures = 0

    async def retrieve(
        self,
        query: str,
        memories: List["MemoryShard"],
        limit: int = 10,
        preferred_retriever: Optional[str] = None,
        **kwargs
    ) -> FallbackResult:
        """
        Retrieve with automatic fallback.

        Tries retrievers in priority order until one succeeds.
        Never fails silently - always returns informative result.

        Args:
            query: The search query
            memories: Pool of memories to search
            limit: Maximum results to return
            preferred_retriever: Try this retriever first (overrides priority)
            **kwargs: Additional arguments passed to retrievers

        Returns:
            FallbackResult with complete fallback chain tracking
        """
        start_time = datetime.utcnow()
        fallback_path: List[str] = []
        fallback_reasons: Dict[str, str] = {}

        # Build retriever order
        retriever_order = self.get_available_retrievers()

        if not retriever_order:
            # No retrievers available!
            return FallbackResult(
                result=RetrievalResultEnhanced.unavailable(
                    reason="No retrievers available in registry",
                    backend="RetrieverRegistry",
                    metadata={
                        "registered_retrievers": list(self._retrievers.keys()),
                        "health_statuses": {
                            n: r.health.value for n, r in self._retrievers.items()
                        }
                    }
                ),
                retriever_used="none",
                fallback_path=[],
                fallback_reasons={"registry": "No available retrievers"},
                total_time_ms=0.0
            )

        # If preferred retriever specified and available, put it first
        if preferred_retriever and preferred_retriever in retriever_order:
            retriever_order.remove(preferred_retriever)
            retriever_order.insert(0, preferred_retriever)

        # Try each retriever in order
        async with self._lock:
            for retriever_name in retriever_order:
                fallback_path.append(retriever_name)
                info = self._retrievers[retriever_name]

                try:
                    # Call the retriever
                    info.total_calls += 1
                    raw_result = await info.retrieve_fn(
                        query=query,
                        memories=memories,
                        limit=limit,
                        **kwargs
                    )

                    # Convert to RetrievalResultEnhanced if needed
                    result = ensure_retrieval_result(raw_result)

                    # Check if retrieval succeeded
                    if result.has_results or result.status == RetrievalStatus.SUCCESS:
                        # Success!
                        info.total_successes += 1
                        info.consecutive_failures = 0
                        info.health = RetrieverHealth.HEALTHY

                        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                        return FallbackResult(
                            result=result,
                            retriever_used=retriever_name,
                            fallback_path=fallback_path,
                            fallback_reasons=fallback_reasons,
                            total_time_ms=total_time
                        )

                    elif result.status == RetrievalStatus.EMPTY:
                        # Query executed but no results - this is "success" for health
                        info.total_successes += 1
                        info.consecutive_failures = 0

                        # Only return EMPTY if this is the last retriever
                        if retriever_name == retriever_order[-1]:
                            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                            return FallbackResult(
                                result=result,
                                retriever_used=retriever_name,
                                fallback_path=fallback_path,
                                fallback_reasons=fallback_reasons,
                                total_time_ms=total_time
                            )
                        else:
                            # Try next retriever for results
                            fallback_reasons[retriever_name] = f"No results for query (status={result.status.value})"
                            continue

                    else:
                        # Degraded result (FALLBACK, PARTIAL, etc.) - record and try next
                        fallback_reasons[retriever_name] = result.message or f"Status: {result.status.value}"
                        info.consecutive_failures += 1
                        self._check_and_update_health(info)
                        continue

                except Exception as e:
                    # Exception during retrieval
                    logger.error(f"Retriever {retriever_name} failed with exception: {e}")
                    fallback_reasons[retriever_name] = f"Exception: {type(e).__name__}: {str(e)}"
                    info.consecutive_failures += 1
                    self._check_and_update_health(info)
                    continue

        # All retrievers exhausted
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return FallbackResult(
            result=RetrievalResultEnhanced.unavailable(
                reason=f"All {len(retriever_order)} retrievers failed or returned no results",
                backend="RetrieverRegistry",
                metadata={
                    "fallback_path": fallback_path,
                    "fallback_reasons": fallback_reasons
                }
            ),
            retriever_used="none",
            fallback_path=fallback_path,
            fallback_reasons=fallback_reasons,
            total_time_ms=total_time
        )

    def _check_and_update_health(self, info: RetrieverInfo) -> None:
        """Update health status based on consecutive failures."""
        if info.consecutive_failures >= self.max_consecutive_failures:
            info.health = RetrieverHealth.UNAVAILABLE
            logger.warning(
                f"Retriever {info.name} marked UNAVAILABLE after "
                f"{info.consecutive_failures} consecutive failures"
            )
        elif info.consecutive_failures > 0:
            info.health = RetrieverHealth.DEGRADED

    async def health_check(self, retriever_name: Optional[str] = None) -> Dict[str, RetrieverHealth]:
        """
        Perform health check on retrievers.

        Args:
            retriever_name: Check specific retriever, or all if None

        Returns:
            Dict mapping retriever names to health status
        """
        results = {}

        names_to_check = (
            [retriever_name] if retriever_name
            else list(self._retrievers.keys())
        )

        for name in names_to_check:
            if name not in self._retrievers:
                continue

            info = self._retrievers[name]

            # Auto-recovery: If enough time has passed, reset to UNKNOWN
            if self.enable_auto_recovery and info.health == RetrieverHealth.UNAVAILABLE:
                if info.last_health_check:
                    elapsed = (datetime.utcnow() - info.last_health_check).total_seconds()
                    if elapsed >= self.health_check_interval:
                        info.health = RetrieverHealth.UNKNOWN
                        info.consecutive_failures = 0
                        logger.info(f"Retriever {name} reset to UNKNOWN for retry")

            info.last_health_check = datetime.utcnow()
            results[name] = info.health

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all registered retrievers."""
        stats = {
            "total_retrievers": len(self._retrievers),
            "available_count": len(self.get_available_retrievers()),
            "priority_order": self._priority_order,
            "retrievers": {}
        }

        for name, info in self._retrievers.items():
            stats["retrievers"][name] = {
                "type": info.retriever_type.value,
                "priority": info.priority,
                "health": info.health.value,
                "total_calls": info.total_calls,
                "total_successes": info.total_successes,
                "success_rate": f"{info.success_rate:.1%}",
                "consecutive_failures": info.consecutive_failures,
                "is_available": info.is_available
            }

        return stats

    def __len__(self) -> int:
        """Number of registered retrievers."""
        return len(self._retrievers)

    def __contains__(self, name: str) -> bool:
        """Check if retriever is registered."""
        return name in self._retrievers


# --- Factory Function ---

def create_retriever_registry(
    max_consecutive_failures: int = 3,
    health_check_interval_seconds: float = 300.0,
    enable_auto_recovery: bool = True
) -> RetrieverRegistry:
    """
    Create a new retriever registry.

    Args:
        max_consecutive_failures: Mark retriever unavailable after N failures
        health_check_interval_seconds: Time between health checks (for auto-recovery)
        enable_auto_recovery: Re-enable retrievers after cooldown

    Returns:
        Configured RetrieverRegistry instance
    """
    return RetrieverRegistry(
        max_consecutive_failures=max_consecutive_failures,
        health_check_interval_seconds=health_check_interval_seconds,
        enable_auto_recovery=enable_auto_recovery
    )


# --- Exports ---

__all__ = [
    "RetrieverType",
    "RetrieverHealth",
    "RetrieverInfo",
    "FallbackResult",
    "RetrieverRegistry",
    "create_retriever_registry"
]
