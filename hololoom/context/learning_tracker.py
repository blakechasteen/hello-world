#!/usr/bin/env python3
"""
Learning Tracker for Hybrid Routing

Part 4: Learning Mechanisms (Days 16-20)

Tracks routing decisions and performance metrics for continuous learning.
Integrates with ReflectionBuffer to persist learning signals.

Key Metrics Tracked:
- Backend selection and outcomes
- Predicted vs. actual confidence
- Latency per backend
- Fallback rate
- Cache hit rate

Validated in Demo 4 (learning improves accuracy over time)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class RoutingEvent:
    """Single routing decision event"""
    session_id: str
    query: str
    backend: str
    predicted_confidence: float
    actual_confidence: float
    confidence_error: float
    latency_ms: float
    cache_hit: bool = False
    fallback_used: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "backend": self.backend,
            "predicted_confidence": self.predicted_confidence,
            "actual_confidence": self.actual_confidence,
            "confidence_error": self.confidence_error,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "fallback_used": self.fallback_used,
            "timestamp": self.timestamp.isoformat()
        }

    def is_success(self, confidence_threshold: float = 0.75) -> bool:
        """Check if routing was successful"""
        return (
            not self.fallback_used and
            self.actual_confidence >= confidence_threshold
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics for a backend"""
    backend: str
    count: int
    avg_confidence: float
    avg_latency_ms: float
    fallback_rate: float
    cache_hit_rate: float
    confidence_calibration: float  # Avg confidence error
    success_rate: float

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "backend": self.backend,
            "count": self.count,
            "avg_confidence": self.avg_confidence,
            "avg_latency_ms": self.avg_latency_ms,
            "fallback_rate": self.fallback_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "confidence_calibration": self.confidence_calibration,
            "success_rate": self.success_rate
        }


# ============================================================================
# Learning Tracker
# ============================================================================

class LearningTracker:
    """
    Track routing decisions for continuous learning

    Records routing events and computes performance metrics to inform
    strategy updates and backend selection.

    Usage:
        tracker = LearningTracker()

        # Record routing decision
        await tracker.record_routing(
            session_id="session_001",
            query="What is the policy?",
            backend="sql",
            predicted_confidence=0.85,
            actual_confidence=0.92,
            latency_ms=15.3
        )

        # Get performance metrics
        metrics = tracker.get_recent_performance("sql", window=100)
        print(f"SQL avg confidence: {metrics.avg_confidence:.2f}")
    """

    def __init__(self, reflection_buffer=None):
        """
        Initialize learning tracker

        Args:
            reflection_buffer: Optional ReflectionBuffer for persistence
        """
        self.reflection_buffer = reflection_buffer
        self.routing_history: list[RoutingEvent] = []

        # Statistics
        self.total_events = 0
        self.backend_counts = {}

    async def record_routing(
        self,
        session_id: str,
        query: str,
        backend: str,
        predicted_confidence: float,
        actual_confidence: float,
        latency_ms: float,
        cache_hit: bool = False,
        fallback_used: bool = False
    ):
        """
        Record routing decision

        Args:
            session_id: Session identifier
            query: Query text
            backend: Backend used (sql, neo4j, qdrant)
            predicted_confidence: Predicted confidence from classifier
            actual_confidence: Actual confidence from result
            latency_ms: Query latency
            cache_hit: Whether result was cached
            fallback_used: Whether fallback was used
        """
        confidence_error = abs(predicted_confidence - actual_confidence)

        event = RoutingEvent(
            session_id=session_id,
            query=query,
            backend=backend,
            predicted_confidence=predicted_confidence,
            actual_confidence=actual_confidence,
            confidence_error=confidence_error,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            fallback_used=fallback_used
        )

        self.routing_history.append(event)
        self.total_events += 1

        # Update backend counts
        self.backend_counts[backend] = self.backend_counts.get(backend, 0) + 1

        logger.debug(
            f"Recorded routing: backend={backend}, "
            f"predicted={predicted_confidence:.3f}, actual={actual_confidence:.3f}, "
            f"latency={latency_ms:.1f}ms, fallback={fallback_used}"
        )

        # Store in ReflectionBuffer if available
        if self.reflection_buffer:
            await self.reflection_buffer.store(
                spacetime=None,  # Router doesn't create Spacetime directly
                feedback={
                    "routing": event.to_dict(),
                    "success": event.is_success()
                }
            )

    def get_recent_performance(
        self,
        backend: str,
        window: int = 100
    ) -> PerformanceMetrics:
        """
        Get recent performance stats for a backend

        Args:
            backend: Backend name (sql, neo4j, qdrant)
            window: Number of recent events to consider

        Returns:
            PerformanceMetrics for the backend
        """
        # Get recent events for this backend
        backend_events = [
            e for e in self.routing_history[-window:]
            if e.backend == backend
        ]

        if not backend_events:
            # No data - return default metrics
            return PerformanceMetrics(
                backend=backend,
                count=0,
                avg_confidence=0.5,
                avg_latency_ms=100.0,
                fallback_rate=0.0,
                cache_hit_rate=0.0,
                confidence_calibration=0.0,
                success_rate=0.5
            )

        # Compute metrics
        count = len(backend_events)
        avg_confidence = sum(e.actual_confidence for e in backend_events) / count
        avg_latency = sum(e.latency_ms for e in backend_events) / count
        fallback_rate = sum(1 for e in backend_events if e.fallback_used) / count
        cache_hit_rate = sum(1 for e in backend_events if e.cache_hit) / count
        confidence_calibration = sum(e.confidence_error for e in backend_events) / count
        success_rate = sum(1 for e in backend_events if e.is_success()) / count

        return PerformanceMetrics(
            backend=backend,
            count=count,
            avg_confidence=avg_confidence,
            avg_latency_ms=avg_latency,
            fallback_rate=fallback_rate,
            cache_hit_rate=cache_hit_rate,
            confidence_calibration=confidence_calibration,
            success_rate=success_rate
        )

    def get_overall_performance(self, window: int = 100) -> dict:
        """
        Get overall performance across all backends

        Args:
            window: Number of recent events

        Returns:
            Dict with overall metrics
        """
        recent_events = self.routing_history[-window:]

        if not recent_events:
            return {
                "count": 0,
                "avg_confidence": 0.5,
                "avg_latency_ms": 100.0,
                "fallback_rate": 0.0,
                "cache_hit_rate": 0.0,
                "success_rate": 0.5
            }

        count = len(recent_events)
        avg_confidence = sum(e.actual_confidence for e in recent_events) / count
        avg_latency = sum(e.latency_ms for e in recent_events) / count
        fallback_rate = sum(1 for e in recent_events if e.fallback_used) / count
        cache_hit_rate = sum(1 for e in recent_events if e.cache_hit) / count
        success_rate = sum(1 for e in recent_events if e.is_success()) / count

        return {
            "count": count,
            "avg_confidence": avg_confidence,
            "avg_latency_ms": avg_latency,
            "fallback_rate": fallback_rate,
            "cache_hit_rate": cache_hit_rate,
            "success_rate": success_rate
        }

    def get_backend_comparison(self, window: int = 100) -> dict[str, PerformanceMetrics]:
        """
        Compare performance across all backends

        Args:
            window: Number of recent events

        Returns:
            Dict mapping backend name to PerformanceMetrics
        """
        backends = list(self.backend_counts.keys())

        if not backends:
            # Default backends
            backends = ["sql", "neo4j", "qdrant"]

        return {
            backend: self.get_recent_performance(backend, window)
            for backend in backends
        }

    def get_confidence_trends(
        self,
        backend: str | None = None,
        window: int = 100
    ) -> list[float]:
        """
        Get confidence trend over time

        Args:
            backend: Specific backend (None for all)
            window: Number of recent events

        Returns:
            List of confidence values (oldest to newest)
        """
        recent = self.routing_history[-window:]

        if backend:
            recent = [e for e in recent if e.backend == backend]

        return [e.actual_confidence for e in recent]

    def get_latency_trends(
        self,
        backend: str | None = None,
        window: int = 100
    ) -> list[float]:
        """
        Get latency trend over time

        Args:
            backend: Specific backend (None for all)
            window: Number of recent events

        Returns:
            List of latency values (oldest to newest)
        """
        recent = self.routing_history[-window:]

        if backend:
            recent = [e for e in recent if e.backend == backend]

        return [e.latency_ms for e in recent]

    def get_summary(self) -> str:
        """
        Get learning tracker summary

        Returns:
            Human-readable summary
        """
        summary = "Learning Tracker Summary:\n"
        summary += f"  Total events: {self.total_events}\n\n"

        # Overall performance
        overall = self.get_overall_performance(window=100)
        summary += "Overall (last 100 events):\n"
        summary += f"  Avg confidence: {overall['avg_confidence']:.3f}\n"
        summary += f"  Avg latency: {overall['avg_latency_ms']:.1f}ms\n"
        summary += f"  Fallback rate: {overall['fallback_rate']*100:.1f}%\n"
        summary += f"  Cache hit rate: {overall['cache_hit_rate']*100:.1f}%\n"
        summary += f"  Success rate: {overall['success_rate']*100:.1f}%\n\n"

        # Per-backend performance
        comparison = self.get_backend_comparison(window=100)
        for backend, metrics in sorted(comparison.items()):
            if metrics.count > 0:
                summary += f"{backend.upper()}:\n"
                summary += f"  Count: {metrics.count}\n"
                summary += f"  Avg confidence: {metrics.avg_confidence:.3f}\n"
                summary += f"  Avg latency: {metrics.avg_latency_ms:.1f}ms\n"
                summary += f"  Fallback rate: {metrics.fallback_rate*100:.1f}%\n"
                summary += f"  Success rate: {metrics.success_rate*100:.1f}%\n\n"

        return summary

    def get_stats(self) -> dict:
        """
        Get learning tracker statistics

        Returns:
            Dict with tracker stats
        """
        overall = self.get_overall_performance(window=100)
        comparison = self.get_backend_comparison(window=100)

        return {
            "total_events": self.total_events,
            "backend_counts": self.backend_counts.copy(),
            "overall": overall,
            "backends": {
                backend: metrics.to_dict()
                for backend, metrics in comparison.items()
            }
        }

    def reset_stats(self):
        """Reset learning tracker statistics (for testing)"""
        self.routing_history = []
        self.total_events = 0
        self.backend_counts = {}


# ============================================================================
# Helper Functions
# ============================================================================

def create_learning_tracker(
    reflection_buffer=None
) -> LearningTracker:
    """
    Create learning tracker

    Args:
        reflection_buffer: Optional ReflectionBuffer

    Returns:
        LearningTracker instance
    """
    return LearningTracker(reflection_buffer=reflection_buffer)
