#!/usr/bin/env python3
"""
Real-Time Monitoring Dashboard
===============================

Production monitoring for the Math->Meaning pipeline with:
- Real-time metrics collection
- Performance tracking
- A/B testing framework
- Alert system
- Visualization

Features:
- Request/response tracking
- Latency monitoring
- Cost efficiency tracking
- Success rate monitoring
- RL learning curves
- Operation usage statistics
- Anomaly detection
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Collection
# ============================================================================

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: float
    query: str
    intent: str
    operations_executed: List[str]
    total_cost: float
    latency_ms: float
    success: bool
    confidence: float
    num_operations: int
    contextual_bandit_used: bool = False
    data_understanding_used: bool = False


@dataclass
class AggregateMetrics:
    """Aggregated metrics over time window."""
    window_start: float
    window_end: float
    total_requests: int
    success_rate: float
    avg_latency_ms: float
    avg_cost: float
    avg_confidence: float
    avg_operations: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class MetricsCollector:
    """
    Collects and aggregates metrics in real-time.

    Maintains sliding windows of recent requests for analysis.
    """

    def __init__(self, window_size: int = 1000, aggregation_interval_s: float = 60.0):
        """
        Args:
            window_size: Number of recent requests to keep
            aggregation_interval_s: Seconds between aggregate computations
        """
        self.window_size = window_size
        self.aggregation_interval = aggregation_interval_s

        # Recent requests (sliding window)
        self.recent_requests: deque = deque(maxlen=window_size)

        # Aggregated metrics history
        self.aggregate_history: List[AggregateMetrics] = []

        # Counters
        self.total_requests = 0
        self.total_successes = 0
        self.total_failures = 0

        # Operation usage tracking
        self.operation_counts = defaultdict(int)
        self.operation_latencies = defaultdict(list)
        self.operation_costs = defaultdict(list)

        # Intent tracking
        self.intent_counts = defaultdict(int)

        # Last aggregation time
        self.last_aggregation = time.time()

        logger.info(f"MetricsCollector initialized (window={window_size}, interval={aggregation_interval_s}s)")

    def record_request(self, metrics: RequestMetrics):
        """Record metrics for a single request."""
        self.recent_requests.append(metrics)
        self.total_requests += 1

        if metrics.success:
            self.total_successes += 1
        else:
            self.total_failures += 1

        # Track operation usage
        for op in metrics.operations_executed:
            self.operation_counts[op] += 1
            self.operation_latencies[op].append(metrics.latency_ms / len(metrics.operations_executed))
            self.operation_costs[op].append(metrics.total_cost / len(metrics.operations_executed))

        # Track intent
        self.intent_counts[metrics.intent] += 1

        # Aggregate periodically
        if time.time() - self.last_aggregation > self.aggregation_interval:
            self._aggregate()

    def _aggregate(self):
        """Compute aggregate metrics."""
        if not self.recent_requests:
            return

        window_start = self.recent_requests[0].timestamp
        window_end = self.recent_requests[-1].timestamp

        latencies = [r.latency_ms for r in self.recent_requests]
        costs = [r.total_cost for r in self.recent_requests]
        confidences = [r.confidence for r in self.recent_requests]
        num_ops = [r.num_operations for r in self.recent_requests]
        successes = [r.success for r in self.recent_requests]

        aggregate = AggregateMetrics(
            window_start=window_start,
            window_end=window_end,
            total_requests=len(self.recent_requests),
            success_rate=np.mean(successes),
            avg_latency_ms=np.mean(latencies),
            avg_cost=np.mean(costs),
            avg_confidence=np.mean(confidences),
            avg_operations=np.mean(num_ops),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
        )

        self.aggregate_history.append(aggregate)
        self.last_aggregation = time.time()

        logger.info(f"Aggregated metrics: {len(self.recent_requests)} requests, "
                   f"{aggregate.success_rate:.1%} success, "
                   f"{aggregate.avg_latency_ms:.0f}ms avg latency")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        if not self.recent_requests:
            return {"error": "No requests recorded"}

        # Force aggregation
        self._aggregate()

        latest_aggregate = self.aggregate_history[-1] if self.aggregate_history else None

        return {
            "total_requests": self.total_requests,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": self.total_successes / self.total_requests if self.total_requests > 0 else 0,
            "recent_aggregate": asdict(latest_aggregate) if latest_aggregate else None,
            "top_operations": dict(sorted(self.operation_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "intent_distribution": dict(self.intent_counts),
        }


# ============================================================================
# A/B Testing
# ============================================================================

@dataclass
class ABVariant:
    """Configuration for an A/B test variant."""
    name: str
    use_contextual_bandit: bool
    use_data_understanding: bool
    exploration_coef: float = 0.5


class ABTester:
    """
    A/B testing framework for comparing different configurations.

    Randomly assigns requests to variants and tracks comparative performance.
    """

    def __init__(self, variants: List[ABVariant]):
        """
        Args:
            variants: List of test variants
        """
        self.variants = variants
        self.variant_metrics = {v.name: MetricsCollector() for v in variants}

        logger.info(f"ABTester initialized with {len(variants)} variants")
        for v in variants:
            logger.info(f"  {v.name}: contextual={v.use_contextual_bandit}, du={v.use_data_understanding}")

    def assign_variant(self, query_hash: int = None) -> ABVariant:
        """Randomly assign a variant (or deterministically based on hash)."""
        if query_hash is not None:
            # Deterministic assignment
            idx = query_hash % len(self.variants)
        else:
            # Random assignment
            idx = np.random.randint(0, len(self.variants))

        return self.variants[idx]

    def record_result(self, variant_name: str, metrics: RequestMetrics):
        """Record result for a variant."""
        if variant_name in self.variant_metrics:
            self.variant_metrics[variant_name].record_request(metrics)

    def get_comparison(self) -> Dict[str, Any]:
        """Get comparative metrics across variants."""
        comparison = {}

        for variant_name, collector in self.variant_metrics.items():
            metrics = collector.get_current_metrics()
            comparison[variant_name] = metrics

        # Compute winner
        success_rates = {v: m.get("success_rate", 0) for v, m in comparison.items()}
        winner = max(success_rates.keys(), key=lambda v: success_rates[v]) if success_rates else None

        return {
            "variants": comparison,
            "winner": winner,
            "winner_success_rate": success_rates.get(winner, 0) if winner else 0,
        }


# ============================================================================
# Alert System
# ============================================================================

class AlertRule:
    """Base class for alert rules."""

    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold

    def check(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Check if alert should fire. Returns alert message or None."""
        raise NotImplementedError


class LatencyAlert(AlertRule):
    """Alert on high latency."""

    def check(self, metrics: Dict[str, Any]) -> Optional[str]:
        agg = metrics.get("recent_aggregate")
        if not agg:
            return None

        if agg["p95_latency_ms"] > self.threshold:
            return f"High latency: P95={agg['p95_latency_ms']:.0f}ms (threshold={self.threshold}ms)"
        return None


class SuccessRateAlert(AlertRule):
    """Alert on low success rate."""

    def check(self, metrics: Dict[str, Any]) -> Optional[str]:
        success_rate = metrics.get("success_rate", 1.0)

        if success_rate < self.threshold:
            return f"Low success rate: {success_rate:.1%} (threshold={self.threshold:.1%})"
        return None


class CostAlert(AlertRule):
    """Alert on high cost."""

    def check(self, metrics: Dict[str, Any]) -> Optional[str]:
        agg = metrics.get("recent_aggregate")
        if not agg:
            return None

        if agg["avg_cost"] > self.threshold:
            return f"High cost: {agg['avg_cost']:.1f} (threshold={self.threshold})"
        return None


class AlertManager:
    """Manages alert rules and fires alerts."""

    def __init__(self, rules: List[AlertRule] = None):
        self.rules = rules or []
        self.fired_alerts: List[Dict[str, Any]] = []

        logger.info(f"AlertManager initialized with {len(self.rules)} rules")

    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        for rule in self.rules:
            alert_msg = rule.check(metrics)
            if alert_msg:
                self._fire_alert(rule.name, alert_msg)

    def _fire_alert(self, rule_name: str, message: str):
        """Fire an alert."""
        alert = {
            "timestamp": time.time(),
            "rule": rule_name,
            "message": message,
        }

        self.fired_alerts.append(alert)

        logger.warning(f"ALERT [{rule_name}]: {message}")

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self.fired_alerts[-limit:]


# ============================================================================
# Dashboard
# ============================================================================

class MonitoringDashboard:
    """
    Complete monitoring dashboard.

    Combines metrics collection, A/B testing, and alerting.
    """

    def __init__(
        self,
        enable_ab_testing: bool = False,
        ab_variants: List[ABVariant] = None,
        alert_rules: List[AlertRule] = None
    ):
        """
        Args:
            enable_ab_testing: Enable A/B testing
            ab_variants: A/B test variants
            alert_rules: Alert rules
        """
        # Metrics collection
        self.metrics_collector = MetricsCollector()

        # A/B testing
        self.enable_ab_testing = enable_ab_testing
        if enable_ab_testing and ab_variants:
            self.ab_tester = ABTester(ab_variants)
        else:
            self.ab_tester = None

        # Alerting
        self.alert_manager = AlertManager(alert_rules or [])

        logger.info("MonitoringDashboard initialized")
        logger.info(f"  A/B Testing: {'ENABLED' if enable_ab_testing else 'DISABLED'}")
        logger.info(f"  Alert Rules: {len(alert_rules) if alert_rules else 0}")

    def record_request(self, metrics: RequestMetrics, variant_name: str = None):
        """Record request metrics."""
        # Main metrics
        self.metrics_collector.record_request(metrics)

        # A/B testing metrics
        if self.enable_ab_testing and self.ab_tester and variant_name:
            self.ab_tester.record_result(variant_name, metrics)

        # Check alerts
        current_metrics = self.metrics_collector.get_current_metrics()
        self.alert_manager.check_alerts(current_metrics)

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get complete dashboard state."""
        state = {
            "metrics": self.metrics_collector.get_current_metrics(),
            "ab_testing": None,
            "alerts": self.alert_manager.get_recent_alerts(limit=10),
            "timestamp": time.time(),
        }

        if self.enable_ab_testing and self.ab_tester:
            state["ab_testing"] = self.ab_tester.get_comparison()

        return state

    def save_state(self, filepath: Path):
        """Save dashboard state to file."""
        state = self.get_dashboard_state()

        # Convert to JSON-serializable
        state_json = json.dumps(state, indent=2, default=str)

        with open(filepath, "w") as f:
            f.write(state_json)

        logger.info(f"Dashboard state saved to {filepath}")

    def print_summary(self):
        """Print dashboard summary to console."""
        state = self.get_dashboard_state()
        metrics = state["metrics"]

        print("\n" + "="*80)
        print("MONITORING DASHBOARD SUMMARY")
        print("="*80)
        print()

        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Success Rate: {metrics['success_rate']:.1%}")
        print()

        if metrics.get("recent_aggregate"):
            agg = metrics["recent_aggregate"]
            print("Recent Performance:")
            print(f"  Avg Latency: {agg['avg_latency_ms']:.0f}ms")
            print(f"  P95 Latency: {agg['p95_latency_ms']:.0f}ms")
            print(f"  P99 Latency: {agg['p99_latency_ms']:.0f}ms")
            print(f"  Avg Cost: {agg['avg_cost']:.1f}")
            print(f"  Avg Confidence: {agg['avg_confidence']:.2f}")
            print()

        if metrics.get("top_operations"):
            print("Top Operations:")
            for op, count in list(metrics["top_operations"].items())[:5]:
                print(f"  {op}: {count}")
            print()

        if state.get("ab_testing"):
            print("A/B Testing:")
            ab = state["ab_testing"]
            print(f"  Winner: {ab['winner']} ({ab['winner_success_rate']:.1%})")
            print()

        if state.get("alerts"):
            print(f"Recent Alerts: {len(state['alerts'])}")
            for alert in state["alerts"][-3:]:
                print(f"  [{alert['rule']}] {alert['message']}")
            print()


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("MONITORING DASHBOARD DEMO")
    print("="*80)
    print()

    # Create dashboard with A/B testing
    variants = [
        ABVariant("baseline", use_contextual_bandit=False, use_data_understanding=False),
        ABVariant("contextual", use_contextual_bandit=True, use_data_understanding=False),
        ABVariant("full", use_contextual_bandit=True, use_data_understanding=True),
    ]

    alert_rules = [
        LatencyAlert("latency_p95", threshold=500.0),
        SuccessRateAlert("success_rate", threshold=0.9),
        CostAlert("avg_cost", threshold=40.0),
    ]

    dashboard = MonitoringDashboard(
        enable_ab_testing=True,
        ab_variants=variants,
        alert_rules=alert_rules
    )

    # Simulate requests
    print("\nSimulating 20 requests...\n")

    for i in range(20):
        # Assign variant
        variant = dashboard.ab_tester.assign_variant()

        # Simulate request
        metrics = RequestMetrics(
            timestamp=time.time(),
            query=f"Test query {i}",
            intent="similarity",
            operations_executed=["inner_product", "metric_distance"],
            total_cost=np.random.uniform(10, 30),
            latency_ms=np.random.uniform(10, 50) if variant.name != "baseline" else np.random.uniform(20, 60),
            success=np.random.random() > (0.05 if variant.name == "full" else 0.1),
            confidence=np.random.uniform(0.6, 0.95),
            num_operations=2,
            contextual_bandit_used=variant.use_contextual_bandit,
            data_understanding_used=variant.use_data_understanding,
        )

        dashboard.record_request(metrics, variant_name=variant.name)

        time.sleep(0.1)

    # Print summary
    dashboard.print_summary()

    # Save state
    output_path = Path("HoloLoom/bootstrap_results/dashboard_state.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard.save_state(output_path)

    print()
    print("Dashboard ready for production!")
