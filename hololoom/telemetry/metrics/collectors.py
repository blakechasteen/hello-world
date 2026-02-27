"""
HoloLoom Metrics Collectors - Pre-defined metrics for HoloLoom systems.

Provides standardized metrics for:
- Query processing (latency, throughput)
- Memory operations (cache hits, graph traversals)
- Federation (consensus, gossip, alignment)
- Learning systems (hot patterns, policy updates)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hololoom.telemetry.metrics.prometheus import (
    PrometheusRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    get_registry,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  HOLOLOOM COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HoloLoomCollector:
    """Pre-defined metrics for HoloLoom systems."""

    registry: PrometheusRegistry = field(default_factory=get_registry)

    # Query metrics
    queries_total: Counter = field(init=False)
    query_latency_seconds: Histogram = field(init=False)
    query_confidence: Summary = field(init=False)
    active_queries: Gauge = field(init=False)

    # Memory metrics
    memory_operations_total: Counter = field(init=False)
    memory_cache_hits: Counter = field(init=False)
    memory_cache_misses: Counter = field(init=False)
    memory_nodes: Gauge = field(init=False)
    memory_edges: Gauge = field(init=False)

    # Federation metrics
    federation_nodes_total: Gauge = field(init=False)
    federation_consensus_rounds: Counter = field(init=False)
    federation_gossip_messages: Counter = field(init=False)
    federation_alignment_checks: Counter = field(init=False)
    federation_trust_score: Summary = field(init=False)

    # Learning metrics
    learning_updates_total: Counter = field(init=False)
    hot_patterns_count: Gauge = field(init=False)
    policy_confidence: Summary = field(init=False)
    thompson_samples: Counter = field(init=False)

    # System metrics
    system_uptime_seconds: Gauge = field(init=False)
    system_errors_total: Counter = field(init=False)

    def __post_init__(self) -> None:
        """Initialize all metrics."""
        # Query metrics
        self.queries_total = self.registry.get_counter(
            "queries_total",
            "Total number of queries processed",
            ["mode", "status"],
        )
        self.query_latency_seconds = self.registry.get_histogram(
            "query_latency_seconds",
            "Query processing latency in seconds",
            ["mode"],
            [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )
        self.query_confidence = self.registry.get_summary(
            "query_confidence",
            "Query confidence scores",
            ["mode"],
            [0.5, 0.9, 0.99],
        )
        self.active_queries = self.registry.get_gauge(
            "active_queries",
            "Number of currently processing queries",
        )

        # Memory metrics
        self.memory_operations_total = self.registry.get_counter(
            "memory_operations_total",
            "Total memory operations",
            ["operation"],
        )
        self.memory_cache_hits = self.registry.get_counter(
            "memory_cache_hits_total",
            "Total cache hits",
        )
        self.memory_cache_misses = self.registry.get_counter(
            "memory_cache_misses_total",
            "Total cache misses",
        )
        self.memory_nodes = self.registry.get_gauge(
            "memory_nodes",
            "Number of nodes in memory graph",
        )
        self.memory_edges = self.registry.get_gauge(
            "memory_edges",
            "Number of edges in memory graph",
        )

        # Federation metrics
        self.federation_nodes_total = self.registry.get_gauge(
            "federation_nodes_total",
            "Number of nodes in federation",
            ["status"],
        )
        self.federation_consensus_rounds = self.registry.get_counter(
            "federation_consensus_rounds_total",
            "Total consensus rounds",
            ["outcome"],
        )
        self.federation_gossip_messages = self.registry.get_counter(
            "federation_gossip_messages_total",
            "Total gossip messages",
            ["type"],
        )
        self.federation_alignment_checks = self.registry.get_counter(
            "federation_alignment_checks_total",
            "Total alignment verification checks",
            ["level", "result"],
        )
        self.federation_trust_score = self.registry.get_summary(
            "federation_trust_score",
            "Node trust scores",
        )

        # Learning metrics
        self.learning_updates_total = self.registry.get_counter(
            "learning_updates_total",
            "Total learning updates",
            ["type"],
        )
        self.hot_patterns_count = self.registry.get_gauge(
            "hot_patterns_count",
            "Number of hot patterns",
        )
        self.policy_confidence = self.registry.get_summary(
            "policy_confidence",
            "Policy decision confidence",
        )
        self.thompson_samples = self.registry.get_counter(
            "thompson_samples_total",
            "Total Thompson Sampling samples",
            ["arm", "outcome"],
        )

        # System metrics
        self.system_uptime_seconds = self.registry.get_gauge(
            "uptime_seconds",
            "System uptime in seconds",
        )
        self.system_errors_total = self.registry.get_counter(
            "errors_total",
            "Total errors",
            ["type"],
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  QUERY METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def record_query(
        self,
        mode: str,
        latency_seconds: float,
        confidence: float,
        success: bool = True,
    ) -> None:
        """Record a query execution."""
        status = "success" if success else "error"
        self.queries_total.inc(labels={"mode": mode, "status": status})
        self.query_latency_seconds.observe(latency_seconds, labels={"mode": mode})
        self.query_confidence.observe(confidence, labels={"mode": mode})

    def query_started(self) -> None:
        """Record a query starting."""
        self.active_queries.inc()

    def query_finished(self) -> None:
        """Record a query finishing."""
        self.active_queries.dec()

    # ═══════════════════════════════════════════════════════════════════════════
    #  MEMORY METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def record_memory_operation(self, operation: str) -> None:
        """Record a memory operation."""
        self.memory_operations_total.inc(labels={"operation": operation})

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.memory_cache_hits.inc()

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.memory_cache_misses.inc()

    def update_graph_size(self, nodes: int, edges: int) -> None:
        """Update graph size metrics."""
        self.memory_nodes.set(nodes)
        self.memory_edges.set(edges)

    # ═══════════════════════════════════════════════════════════════════════════
    #  FEDERATION METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def update_federation_nodes(self, online: int, offline: int = 0) -> None:
        """Update federation node counts."""
        self.federation_nodes_total.set(online, labels={"status": "online"})
        self.federation_nodes_total.set(offline, labels={"status": "offline"})

    def record_consensus_round(self, success: bool) -> None:
        """Record a consensus round."""
        outcome = "success" if success else "failure"
        self.federation_consensus_rounds.inc(labels={"outcome": outcome})

    def record_gossip_message(self, msg_type: str) -> None:
        """Record a gossip message."""
        self.federation_gossip_messages.inc(labels={"type": msg_type})

    def record_alignment_check(self, level: str, passed: bool) -> None:
        """Record an alignment check."""
        result = "pass" if passed else "fail"
        self.federation_alignment_checks.inc(labels={"level": level, "result": result})

    def record_trust_score(self, score: float) -> None:
        """Record a trust score."""
        self.federation_trust_score.observe(score)

    # ═══════════════════════════════════════════════════════════════════════════
    #  LEARNING METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def record_learning_update(self, update_type: str) -> None:
        """Record a learning update."""
        self.learning_updates_total.inc(labels={"type": update_type})

    def update_hot_patterns_count(self, count: int) -> None:
        """Update hot patterns count."""
        self.hot_patterns_count.set(count)

    def record_policy_decision(self, confidence: float) -> None:
        """Record a policy decision confidence."""
        self.policy_confidence.observe(confidence)

    def record_thompson_sample(self, arm: str, success: bool) -> None:
        """Record a Thompson Sampling sample."""
        outcome = "success" if success else "failure"
        self.thompson_samples.inc(labels={"arm": arm, "outcome": outcome})

    # ═══════════════════════════════════════════════════════════════════════════
    #  SYSTEM METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def update_uptime(self, seconds: float) -> None:
        """Update system uptime."""
        self.system_uptime_seconds.set(seconds)

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        self.system_errors_total.inc(labels={"type": error_type})


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════


_global_collector: Optional[HoloLoomCollector] = None


def create_default_collector(
    registry: Optional[PrometheusRegistry] = None,
) -> HoloLoomCollector:
    """Create and set the global HoloLoom collector."""
    global _global_collector
    _global_collector = HoloLoomCollector(registry or get_registry())
    return _global_collector


def get_collector() -> HoloLoomCollector:
    """Get the global collector, creating one if needed."""
    global _global_collector
    if _global_collector is None:
        _global_collector = HoloLoomCollector()
    return _global_collector
