"""
Federation Observability Metrics - Prometheus-compatible metrics for federation layer.

Provides comprehensive metrics for:
- Federated query performance (latency, throughput, errors)
- Alignment verification (L1/L2/L3 layer metrics)
- Gossip protocol health (membership, message rates)
- Cluster health (nodes, capacity)

Quick Start:
    from hololoom.federation.metrics import (
        get_federation_collector,
        record_federation_query,
        record_alignment_check,
        record_gossip_message,
    )

    # Record federated query
    with record_federation_query(complexity="complex") as ctx:
        result = await federation.query(...)
        ctx.set_result("success")

    # Record alignment check
    record_alignment_check(layer="L2", result="aligned", latency_seconds=0.015)

    # Record gossip message
    record_gossip_message(type="PING", status="success")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  PROMETHEUS METRICS (optional dependency)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from hololoom.telemetry import (
        Counter,
        Gauge,
        Histogram,
        counter,
        gauge,
        get_registry,
        histogram,
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    Counter = None
    Gauge = None
    Histogram = None


# ═══════════════════════════════════════════════════════════════════════════════
#  METRIC DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Federated Query Metrics
_federation_queries_total: Counter | None = None
_federation_query_latency: Histogram | None = None
_federation_query_errors: Counter | None = None

# Alignment Verification Metrics
_alignment_checks_total: Counter | None = None
_alignment_latency: Histogram | None = None
_alignment_failures: Counter | None = None

# Gossip Protocol Metrics
_gossip_messages_total: Counter | None = None
_gossip_latency: Histogram | None = None
_gossip_membership_changes: Counter | None = None

# Cluster Health Metrics
_cluster_nodes_total: Gauge | None = None
_cluster_healthy_nodes: Gauge | None = None
_cluster_pending_queries: Gauge | None = None


def _init_metrics() -> None:
    """Initialize Prometheus metrics (lazy initialization)."""
    global _federation_queries_total, _federation_query_latency, _federation_query_errors
    global _alignment_checks_total, _alignment_latency, _alignment_failures
    global _gossip_messages_total, _gossip_latency, _gossip_membership_changes
    global _cluster_nodes_total, _cluster_healthy_nodes, _cluster_pending_queries

    if not _PROMETHEUS_AVAILABLE:
        return

    # Skip if already initialized
    if _federation_queries_total is not None:
        return

    # Federated Query Metrics
    _federation_queries_total = counter(
        "federation_queries_total",
        "Total federated queries",
        ["result", "complexity"],
    )
    _federation_query_latency = histogram(
        "federation_query_latency_seconds",
        "Federated query latency in seconds",
        ["result"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    _federation_query_errors = counter(
        "federation_query_errors_total",
        "Total federated query errors",
        ["error_type"],
    )

    # Alignment Verification Metrics
    _alignment_checks_total = counter(
        "federation_alignment_checks_total",
        "Total alignment verification checks",
        ["layer", "result"],
    )
    _alignment_latency = histogram(
        "federation_alignment_latency_seconds",
        "Alignment verification latency in seconds",
        ["layer"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
    _alignment_failures = counter(
        "federation_alignment_failures_total",
        "Total alignment verification failures",
        ["layer", "reason"],
    )

    # Gossip Protocol Metrics
    _gossip_messages_total = counter(
        "federation_gossip_messages_total",
        "Total gossip protocol messages",
        ["type", "status"],
    )
    _gossip_latency = histogram(
        "federation_gossip_latency_seconds",
        "Gossip message latency in seconds",
        ["type"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
    )
    _gossip_membership_changes = counter(
        "federation_gossip_membership_changes_total",
        "Total gossip membership changes",
        ["event"],  # join, leave, suspect, dead
    )

    # Cluster Health Metrics
    _cluster_nodes_total = gauge(
        "federation_cluster_nodes_total",
        "Total nodes in federation cluster",
    )
    _cluster_healthy_nodes = gauge(
        "federation_cluster_healthy_nodes",
        "Number of healthy nodes in cluster",
    )
    _cluster_pending_queries = gauge(
        "federation_cluster_pending_queries",
        "Number of pending federated queries",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  QUERY METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QueryMetricsContext:
    """Context manager for recording query metrics."""

    complexity: str = "unknown"
    result: str = "unknown"
    _start_time: float = field(default_factory=time.perf_counter)

    def set_result(self, result: str) -> None:
        """Set the query result (success/failure/timeout)."""
        self.result = result


@contextmanager
def record_federation_query(
    complexity: str = "unknown",
) -> Generator[QueryMetricsContext, None, None]:
    """
    Context manager to record federation query metrics.

    Args:
        complexity: Query complexity (simple/moderate/complex/research)

    Yields:
        QueryMetricsContext to set result

    Example:
        with record_federation_query(complexity="complex") as ctx:
            result = await federation.query(...)
            ctx.set_result("success" if result else "failure")
    """
    _init_metrics()
    ctx = QueryMetricsContext(complexity=complexity)

    try:
        yield ctx
    finally:
        latency = time.perf_counter() - ctx._start_time

        if _federation_queries_total:
            _federation_queries_total.inc(
                labels={"result": ctx.result, "complexity": ctx.complexity}
            )

        if _federation_query_latency:
            _federation_query_latency.observe(
                latency,
                labels={"result": ctx.result},
            )


def record_query_error(error_type: str) -> None:
    """Record a federation query error."""
    _init_metrics()
    if _federation_query_errors:
        _federation_query_errors.inc(labels={"error_type": error_type})


# ═══════════════════════════════════════════════════════════════════════════════
#  ALIGNMENT METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def record_alignment_check(
    layer: str,
    result: str,
    latency_seconds: float,
) -> None:
    """
    Record an alignment verification check.

    Args:
        layer: Alignment layer (L1_LOCAL, L2_RESPONSE, L3_CONSENSUS)
        result: Check result (aligned, misaligned, timeout, error)
        latency_seconds: Check latency in seconds
    """
    _init_metrics()

    if _alignment_checks_total:
        _alignment_checks_total.inc(labels={"layer": layer, "result": result})

    if _alignment_latency:
        _alignment_latency.observe(latency_seconds, labels={"layer": layer})


def record_alignment_failure(layer: str, reason: str) -> None:
    """
    Record an alignment verification failure.

    Args:
        layer: Alignment layer (L1_LOCAL, L2_RESPONSE, L3_CONSENSUS)
        reason: Failure reason
    """
    _init_metrics()
    if _alignment_failures:
        _alignment_failures.inc(labels={"layer": layer, "reason": reason})


# ═══════════════════════════════════════════════════════════════════════════════
#  GOSSIP METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def record_gossip_message(
    message_type: str,
    status: str,
    latency_seconds: float | None = None,
) -> None:
    """
    Record a gossip protocol message.

    Args:
        message_type: Message type (PING, ACK, ALIVE, SUSPECT, DEAD, etc.)
        status: Message status (success, failure, timeout)
        latency_seconds: Optional message latency in seconds
    """
    _init_metrics()

    if _gossip_messages_total:
        _gossip_messages_total.inc(labels={"type": message_type, "status": status})

    if latency_seconds is not None and _gossip_latency:
        _gossip_latency.observe(latency_seconds, labels={"type": message_type})


def record_membership_change(event: str) -> None:
    """
    Record a gossip membership change event.

    Args:
        event: Event type (join, leave, suspect, dead)
    """
    _init_metrics()
    if _gossip_membership_changes:
        _gossip_membership_changes.inc(labels={"event": event})


# ═══════════════════════════════════════════════════════════════════════════════
#  CLUSTER METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def set_cluster_nodes(total: int, healthy: int) -> None:
    """
    Set cluster node counts.

    Args:
        total: Total nodes in cluster
        healthy: Number of healthy nodes
    """
    _init_metrics()

    if _cluster_nodes_total:
        _cluster_nodes_total.set(total)

    if _cluster_healthy_nodes:
        _cluster_healthy_nodes.set(healthy)


def set_pending_queries(count: int) -> None:
    """
    Set the number of pending federated queries.

    Args:
        count: Number of pending queries
    """
    _init_metrics()
    if _cluster_pending_queries:
        _cluster_pending_queries.set(count)


def inc_pending_queries() -> None:
    """Increment the pending query count."""
    _init_metrics()
    if _cluster_pending_queries:
        _cluster_pending_queries.inc()


def dec_pending_queries() -> None:
    """Decrement the pending query count."""
    _init_metrics()
    if _cluster_pending_queries:
        _cluster_pending_queries.dec()


# ═══════════════════════════════════════════════════════════════════════════════
#  COLLECTOR (for custom scraping)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FederationMetricsSnapshot:
    """Snapshot of federation metrics for custom reporting."""

    queries_total: int = 0
    queries_success: int = 0
    queries_failed: int = 0
    avg_latency_seconds: float = 0.0
    alignment_checks: int = 0
    alignment_failures: int = 0
    gossip_messages: int = 0
    cluster_nodes: int = 0
    healthy_nodes: int = 0


class FederationCollector:
    """
    Federation metrics collector for Prometheus scraping.

    Tracks all federation metrics and provides a snapshot for export.
    """

    def __init__(self) -> None:
        self._queries_total: int = 0
        self._queries_success: int = 0
        self._queries_failed: int = 0
        self._latencies: list[float] = []
        self._alignment_checks: int = 0
        self._alignment_failures: int = 0
        self._gossip_messages: int = 0
        self._cluster_nodes: int = 0
        self._healthy_nodes: int = 0

    def record_query(
        self,
        result: str,
        latency_seconds: float,
        complexity: str = "unknown",
    ) -> None:
        """Record a federated query."""
        self._queries_total += 1
        self._latencies.append(latency_seconds)

        if result == "success":
            self._queries_success += 1
        else:
            self._queries_failed += 1

        # Keep last 1000 latencies for averaging
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]

    def record_alignment(self, layer: str, result: str) -> None:
        """Record an alignment check."""
        self._alignment_checks += 1
        if result in ("misaligned", "failure", "error"):
            self._alignment_failures += 1

    def record_gossip(self, message_type: str, status: str) -> None:
        """Record a gossip message."""
        self._gossip_messages += 1

    def set_cluster_state(self, total: int, healthy: int) -> None:
        """Set cluster node state."""
        self._cluster_nodes = total
        self._healthy_nodes = healthy

    def get_snapshot(self) -> FederationMetricsSnapshot:
        """Get current metrics snapshot."""
        avg_latency = (
            sum(self._latencies) / len(self._latencies)
            if self._latencies
            else 0.0
        )

        return FederationMetricsSnapshot(
            queries_total=self._queries_total,
            queries_success=self._queries_success,
            queries_failed=self._queries_failed,
            avg_latency_seconds=avg_latency,
            alignment_checks=self._alignment_checks,
            alignment_failures=self._alignment_failures,
            gossip_messages=self._gossip_messages,
            cluster_nodes=self._cluster_nodes,
            healthy_nodes=self._healthy_nodes,
        )


# Global collector instance
_federation_collector: FederationCollector | None = None


def get_federation_collector() -> FederationCollector:
    """Get the global federation metrics collector."""
    global _federation_collector
    if _federation_collector is None:
        _federation_collector = FederationCollector()
    return _federation_collector


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Query metrics
    "record_federation_query",
    "record_query_error",
    "QueryMetricsContext",
    # Alignment metrics
    "record_alignment_check",
    "record_alignment_failure",
    # Gossip metrics
    "record_gossip_message",
    "record_membership_change",
    # Cluster metrics
    "set_cluster_nodes",
    "set_pending_queries",
    "inc_pending_queries",
    "dec_pending_queries",
    # Collector
    "FederationCollector",
    "FederationMetricsSnapshot",
    "get_federation_collector",
]
