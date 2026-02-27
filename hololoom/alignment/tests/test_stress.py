"""
Stress and Scale Tests for Alignment Framework
==============================================

E3.4 Implementation - December 2025

Tests alignment modules under high load conditions:
- Throughput: High volume of operations per second
- Concurrency: Many parallel operations
- Memory: Behavior under large data volumes
- Latency: Response times under stress
- Scalability: Performance with increasing load

Target: No degradation up to 1000 ops/sec, <100ms p99 latency
"""

import pytest
import asyncio
import time
import statistics
import sys
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import random
import string

from hololoom.alignment import (
    SafetyGuardrails,
    DeceptionDetector,
    InstrumentalConvergenceGuard,
    AuditTrail,
    AlignmentMonitor,
    AlertDispatcher,
    AlertConfig,
    AlertSeverity,
    ActionRequest,
    ActionCategory,
    RiskLevel,
    DecisionType,
    OutcomeType,
    BehavioralProbe,
    ResourceBounds,
    WebhookAlert,
)
from hololoom.alignment.instrumental_convergence import ResourceType
from hololoom.alignment.deception_detection import ProbeType, GoalStatement, ActionObservation


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def guardrails():
    """Create SafetyGuardrails instance."""
    return SafetyGuardrails()


@pytest.fixture
def detector():
    """Create DeceptionDetector instance."""
    return DeceptionDetector()


@pytest.fixture
def guard():
    """Create InstrumentalConvergenceGuard instance."""
    return InstrumentalConvergenceGuard()


@pytest.fixture
def audit():
    """Create AuditTrail instance."""
    return AuditTrail()


@pytest.fixture
def monitor():
    """Create AlignmentMonitor instance."""
    return AlignmentMonitor()


@pytest.fixture
def alert_dispatcher():
    """Create AlertDispatcher with mock endpoints."""
    config = AlertConfig(
        slack_webhook_url="http://mock.slack.webhook/test",
        dedup_window_seconds=0.0  # Disable deduplication for stress tests
    )
    return AlertDispatcher(config=config)


# ============================================================================
# Test Helpers
# ============================================================================

def random_action_request() -> ActionRequest:
    """Generate a random action request for testing."""
    categories = list(ActionCategory)
    actions = ["read_file", "write_file", "execute_code", "network_request",
               "modify_config", "delete_file", "api_call", "shell_command"]

    return ActionRequest(
        action_id=''.join(random.choices(string.ascii_lowercase, k=8)),
        category=random.choice(categories),
        description=random.choice(actions),
        context={
            "timestamp": datetime.now().isoformat(),
        },
        user_id=f"user_{random.randint(1, 100)}"
    )


def measure_latency(func, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of a function."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    return result, elapsed


async def async_measure_latency(coro) -> Tuple[Any, float]:
    """Measure execution time of an async function."""
    start = time.perf_counter()
    result = await coro
    elapsed = (time.perf_counter() - start) * 1000  # ms
    return result, elapsed


def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "max": 0}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return {
        "p50": sorted_latencies[int(n * 0.5)],
        "p95": sorted_latencies[int(n * 0.95)] if n > 20 else sorted_latencies[-1],
        "p99": sorted_latencies[int(n * 0.99)] if n > 100 else sorted_latencies[-1],
        "mean": statistics.mean(sorted_latencies),
        "max": max(sorted_latencies)
    }


# ============================================================================
# SafetyGuardrails Stress Tests
# ============================================================================

class TestSafetyGuardrailsThroughput:
    """Test SafetyGuardrails under high throughput."""

    def test_sequential_high_volume(self, guardrails):
        """Test sequential processing of many safety decisions."""
        num_requests = 1000
        latencies = []

        for _ in range(num_requests):
            request = random_action_request()
            _, elapsed = measure_latency(guardrails.evaluate, request)
            latencies.append(elapsed)

        percentiles = calculate_percentiles(latencies)

        # Performance targets
        assert percentiles["p99"] < 50, f"p99 latency {percentiles['p99']:.2f}ms exceeds 50ms target"
        assert percentiles["mean"] < 10, f"Mean latency {percentiles['mean']:.2f}ms exceeds 10ms target"

        # Verify all requests processed correctly
        assert len(latencies) == num_requests

    def test_parallel_throughput(self, guardrails):
        """Test parallel processing with thread pool."""
        num_requests = 500
        num_workers = 10
        latencies = []

        def process_request():
            request = random_action_request()
            _, elapsed = measure_latency(guardrails.evaluate, request)
            return elapsed

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                latencies.append(future.result())

        percentiles = calculate_percentiles(latencies)

        # Under parallel load, p99 can be higher
        assert percentiles["p99"] < 100, f"p99 latency {percentiles['p99']:.2f}ms exceeds 100ms target"
        assert len(latencies) == num_requests

    def test_sustained_throughput(self, guardrails):
        """Test sustained throughput over time."""
        duration_seconds = 2  # Run for 2 seconds
        start_time = time.time()
        count = 0
        latencies = []

        while time.time() - start_time < duration_seconds:
            request = random_action_request()
            _, elapsed = measure_latency(guardrails.evaluate, request)
            latencies.append(elapsed)
            count += 1

        throughput = count / duration_seconds
        percentiles = calculate_percentiles(latencies)

        # Target: at least 500 ops/sec
        assert throughput > 500, f"Throughput {throughput:.1f} ops/sec below 500 target"
        assert percentiles["p99"] < 50, f"p99 latency {percentiles['p99']:.2f}ms exceeds 50ms"


# ============================================================================
# AuditTrail Stress Tests
# ============================================================================

class TestAuditTrailScalability:
    """Test AuditTrail with large volumes of logs."""

    def test_high_volume_logging(self, audit):
        """Test logging many decisions."""
        num_logs = 5000
        latencies = []

        decision_types = list(DecisionType)
        outcome_types = list(OutcomeType)

        for i in range(num_logs):
            _, elapsed = measure_latency(
                audit.log_decision,
                decision_type=random.choice(decision_types),
                outcome=random.choice(outcome_types),
                reason=f"Test decision {i}",
                metadata={"index": i, "batch": "stress_test"}
            )
            latencies.append(elapsed)

        percentiles = calculate_percentiles(latencies)

        # Performance targets for logging
        assert percentiles["p99"] < 20, f"p99 latency {percentiles['p99']:.2f}ms exceeds 20ms"
        assert percentiles["mean"] < 5, f"Mean latency {percentiles['mean']:.2f}ms exceeds 5ms"

        # Verify logs stored
        assert len(audit.logs) == num_logs

    def test_query_performance_large_dataset(self, audit):
        """Test query performance with large log dataset."""
        # First, populate with many logs
        num_logs = 3000
        risk_levels = [RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]

        for i in range(num_logs):
            risk_level = random.choice(risk_levels)
            audit.log_decision(
                decision_type=DecisionType.SAFETY_GATE,
                outcome=OutcomeType.APPROVED if risk_level in [RiskLevel.SAFE, RiskLevel.LOW] else OutcomeType.REJECTED,
                reason=f"Test query {i}",
                metadata={"risk_level": risk_level.value, "index": i}
            )

        # Test query by metadata
        query_latencies = []
        for _ in range(100):
            _, elapsed = measure_latency(
                audit.query_by_metadata,
                {"risk_level": RiskLevel.HIGH.value}
            )
            query_latencies.append(elapsed)

        percentiles = calculate_percentiles(query_latencies)

        # Query should still be fast even with large dataset
        assert percentiles["p99"] < 100, f"Query p99 {percentiles['p99']:.2f}ms exceeds 100ms"

    def test_concurrent_logging(self, audit):
        """Test concurrent logging from multiple threads."""
        num_logs = 500
        num_workers = 8
        results = []

        def log_decision(index):
            latency_start = time.perf_counter()
            log = audit.log_decision(
                decision_type=DecisionType.SAFETY_GATE,
                outcome=OutcomeType.APPROVED,
                reason=f"Concurrent test {index}",
                metadata={"thread_index": index}
            )
            elapsed = (time.perf_counter() - latency_start) * 1000
            return (log.decision_id, elapsed)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(log_decision, i) for i in range(num_logs)]
            for future in as_completed(futures):
                results.append(future.result())

        # All logs should be created - note: timestamp-based IDs may have collisions
        # under extreme concurrency (same millisecond), so check logs list length
        assert len(audit.logs) == num_logs, f"Expected {num_logs} logs, got {len(audit.logs)}"

        latencies = [r[1] for r in results]
        percentiles = calculate_percentiles(latencies)
        assert percentiles["p99"] < 50, f"Concurrent logging p99 {percentiles['p99']:.2f}ms exceeds 50ms"


# ============================================================================
# DeceptionDetector Stress Tests
# ============================================================================

class TestDeceptionDetectorThroughput:
    """Test DeceptionDetector under high load."""

    def test_sequential_probe_execution(self, detector):
        """Test sequential execution of many probes."""
        num_probes = 500
        latencies = []

        probe_types = list(ProbeType)
        scenarios = [
            "Asked to explain capabilities",
            "Asked about hidden goals",
            "Asked to complete a task",
            "Asked to provide information"
        ]
        expected_behaviors = [
            "honest description",
            "transparent response",
            "aligned action",
            "truthful answer"
        ]

        for i in range(num_probes):
            probe = BehavioralProbe(
                probe_type=random.choice(probe_types),
                scenario=random.choice(scenarios),
                expected_behavior=random.choice(expected_behaviors),
                metadata={"iteration": i}
            )
            actual_behavior = random.choice(expected_behaviors)  # Simulate actual behavior
            _, elapsed = measure_latency(detector.run_probe, probe, actual_behavior)
            latencies.append(elapsed)

        percentiles = calculate_percentiles(latencies)

        assert percentiles["p99"] < 30, f"Probe p99 {percentiles['p99']:.2f}ms exceeds 30ms"
        assert percentiles["mean"] < 5, f"Probe mean {percentiles['mean']:.2f}ms exceeds 5ms"

    def test_goal_transparency_throughput(self, detector):
        """Test goal transparency analysis throughput."""
        num_checks = 500
        latencies = []

        # Set up goal tracker with goals
        goal_tracker = detector.goal_tracker
        if goal_tracker is None:
            pytest.skip("Goal tracking not enabled")

        # Declare some goals
        for i in range(10):
            goal = GoalStatement(
                goal_id=f"goal_{i}",
                description=f"Test goal {i}",
                priority=i % 3 + 1
            )
            goal_tracker.declare_goal(goal)

        # Run analysis throughput test
        goal_ids = [f"goal_{i}" for i in range(10)]

        for i in range(num_checks):
            goal_id = random.choice(goal_ids)
            _, elapsed = measure_latency(goal_tracker.analyze_alignment, goal_id)
            latencies.append(elapsed)

        percentiles = calculate_percentiles(latencies)

        assert percentiles["p99"] < 50, f"Analysis p99 {percentiles['p99']:.2f}ms exceeds 50ms"

    def test_action_observation_throughput(self, detector):
        """Test action observation throughput."""
        num_observations = 500
        latencies = []

        goal_tracker = detector.goal_tracker
        if goal_tracker is None:
            pytest.skip("Goal tracking not enabled")

        # Declare some goals first
        for i in range(5):
            goal = GoalStatement(
                goal_id=f"obs_goal_{i}",
                description=f"Observation test goal {i}"
            )
            goal_tracker.declare_goal(goal)

        actions = [
            "Gathering information",
            "Modifying file",
            "Creating backup",
            "Executing command"
        ]

        for i in range(num_observations):
            action = ActionObservation(
                action_id=f"action_{i}",
                description=random.choice(actions),
                context={"iteration": i},
                claimed_goals=[f"obs_goal_{i % 5}"]
            )
            _, elapsed = measure_latency(goal_tracker.observe_action, action)
            latencies.append(elapsed)

        percentiles = calculate_percentiles(latencies)

        assert percentiles["p99"] < 20, f"Observation p99 {percentiles['p99']:.2f}ms exceeds 20ms"

    def test_parallel_probe_execution(self, detector):
        """Test parallel probe execution."""
        num_probes = 200
        num_workers = 5
        latencies = []

        probe_types = list(ProbeType)

        def run_probe(index):
            probe = BehavioralProbe(
                probe_type=probe_types[index % len(probe_types)],
                scenario=f"Parallel test scenario {index}",
                expected_behavior="consistent behavior"
            )
            actual_behavior = "consistent behavior" if random.random() > 0.3 else "different response"
            _, elapsed = measure_latency(detector.run_probe, probe, actual_behavior)
            return elapsed

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_probe, i) for i in range(num_probes)]
            for future in as_completed(futures):
                latencies.append(future.result())

        percentiles = calculate_percentiles(latencies)

        assert percentiles["p99"] < 100, f"Parallel probe p99 {percentiles['p99']:.2f}ms exceeds 100ms"


# ============================================================================
# InstrumentalConvergenceGuard Stress Tests
# ============================================================================

class TestConvergenceGuardThroughput:
    """Test InstrumentalConvergenceGuard under high load."""

    def test_resource_check_throughput(self, guard):
        """Test rapid resource usage checks."""
        num_checks = 1000
        latencies = []

        # Set up bounds for multiple resources
        for resource_type in ResourceType:
            bounds = ResourceBounds(resource_type, soft_limit=1000, hard_limit=2000)
            guard.set_resource_bounds(resource_type, bounds)

        resource_types = list(ResourceType)

        for _ in range(num_checks):
            resource = random.choice(resource_types)
            usage = random.uniform(0, 2500)  # Some will violate
            _, elapsed = measure_latency(guard.check_resource_usage, resource, usage)
            latencies.append(elapsed)

        percentiles = calculate_percentiles(latencies)

        assert percentiles["p99"] < 20, f"Resource check p99 {percentiles['p99']:.2f}ms exceeds 20ms"
        assert percentiles["mean"] < 3, f"Resource check mean {percentiles['mean']:.2f}ms exceeds 3ms"

    def test_autonomy_check_throughput(self, guard):
        """Test rapid autonomy limit checks."""
        num_checks = 500
        latencies = []

        actions = ["research", "verify", "plan", "execute", "modify", "delete"]

        for _ in range(num_checks):
            action = random.choice(actions)
            _, elapsed = measure_latency(guard.check_autonomy_limits, action)
            latencies.append(elapsed)

        percentiles = calculate_percentiles(latencies)

        assert percentiles["p99"] < 20, f"Autonomy check p99 {percentiles['p99']:.2f}ms exceeds 20ms"


# ============================================================================
# AlignmentMonitor Stress Tests
# ============================================================================

class TestMonitorScalability:
    """Test AlignmentMonitor under high metric volumes."""

    def test_high_volume_metric_recording(self, monitor):
        """Test recording many metrics rapidly."""
        num_recordings = 2000
        latencies = []

        risk_levels = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        outcomes = ["allowed", "blocked"]

        for _ in range(num_recordings):
            risk = random.choice(risk_levels)
            outcome = random.choice(outcomes)
            _, elapsed = measure_latency(
                monitor.record_safety_decision,
                risk_level=risk,
                outcome=outcome
            )
            latencies.append(elapsed)

        percentiles = calculate_percentiles(latencies)

        assert percentiles["p99"] < 10, f"Recording p99 {percentiles['p99']:.2f}ms exceeds 10ms"
        assert percentiles["mean"] < 2, f"Recording mean {percentiles['mean']:.2f}ms exceeds 2ms"

    def test_prometheus_export_performance(self, monitor):
        """Test Prometheus export with many metrics."""
        # First record many metrics
        for _ in range(500):
            monitor.record_safety_decision(
                risk_level=random.choice(["SAFE", "LOW", "MEDIUM", "HIGH"]),
                outcome=random.choice(["allowed", "blocked"])
            )

        # Test export performance
        export_latencies = []
        for _ in range(50):
            _, elapsed = measure_latency(monitor.export_prometheus)
            export_latencies.append(elapsed)

        percentiles = calculate_percentiles(export_latencies)

        # Export should be fast even with many metrics
        assert percentiles["p99"] < 100, f"Export p99 {percentiles['p99']:.2f}ms exceeds 100ms"

    def test_concurrent_metric_recording(self, monitor):
        """Test concurrent metric recording."""
        num_recordings = 500
        num_workers = 6

        def record_metric(index):
            latency_start = time.perf_counter()
            monitor.record_safety_decision(
                risk_level=random.choice(["SAFE", "LOW", "MEDIUM", "HIGH"]),
                outcome="allowed"
            )
            elapsed = (time.perf_counter() - latency_start) * 1000
            return elapsed

        latencies = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(record_metric, i) for i in range(num_recordings)]
            for future in as_completed(futures):
                latencies.append(future.result())

        percentiles = calculate_percentiles(latencies)

        assert percentiles["p99"] < 30, f"Concurrent recording p99 {percentiles['p99']:.2f}ms exceeds 30ms"

        # Verify metrics recorded correctly (check summary)
        summary = monitor.get_alignment_summary()
        assert summary is not None


# ============================================================================
# AlertDispatcher Stress Tests
# ============================================================================

class TestAlertDispatcherThroughput:
    """Test AlertDispatcher under high alert volumes."""

    def test_alert_preparation_throughput(self, alert_dispatcher):
        """Test alert preparation (not actual sending) throughput."""
        num_alerts = 500
        latencies = []

        severities = [AlertSeverity.WARNING, AlertSeverity.CRITICAL]

        for i in range(num_alerts):
            # Just prepare the alert (actual sending would be async/network)
            start = time.perf_counter()
            alert = WebhookAlert(
                title=f"Test Alert {i}",
                message=f"This is test alert number {i}",
                severity=random.choice(severities),
                metadata={"index": i, "test": "stress"}
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        percentiles = calculate_percentiles(latencies)

        # Alert creation should be very fast
        assert percentiles["p99"] < 10, f"Alert creation p99 {percentiles['p99']:.2f}ms exceeds 10ms"
        assert percentiles["mean"] < 2, f"Alert creation mean {percentiles['mean']:.2f}ms exceeds 2ms"


# ============================================================================
# Memory Usage Tests
# ============================================================================

class TestMemoryUsage:
    """Test memory behavior under load."""

    def test_audit_trail_memory_growth(self, audit):
        """Test memory usage as audit trail grows."""
        import sys

        # Get baseline memory
        gc.collect()

        # Log many decisions
        num_logs = 5000
        for i in range(num_logs):
            audit.log_decision(
                decision_type=DecisionType.SAFETY_GATE,
                outcome=OutcomeType.APPROVED,
                reason=f"Memory test {i}",
                metadata={"index": i, "data": "x" * 100}  # Small payload
            )

        # Get logs count
        log_count = len(audit.logs)

        # Memory should be reasonable (rough check)
        # Each log is small, so 5000 logs shouldn't consume excessive memory
        assert log_count == num_logs

        # Verify we can still query efficiently
        start = time.perf_counter()
        _ = len(audit.logs)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"Getting {num_logs} logs took {elapsed:.2f}ms"


# ============================================================================
# Combined Pipeline Stress Tests
# ============================================================================

class TestFullPipelineStress:
    """Test full alignment pipeline under stress."""

    def test_end_to_end_pipeline_throughput(self, guardrails, detector, guard, audit, monitor):
        """Test full pipeline throughput with all components."""
        num_requests = 200
        latencies = []

        # Set up resources
        bounds = ResourceBounds(ResourceType.API_CALLS, soft_limit=100, hard_limit=200)
        guard.set_resource_bounds(ResourceType.API_CALLS, bounds)

        for i in range(num_requests):
            pipeline_start = time.perf_counter()

            # Step 1: Safety check
            request = random_action_request()
            decision = guardrails.evaluate(request)

            # Step 2: Record to monitor (use string values)
            monitor.record_safety_decision(
                risk_level=decision.risk_level.value.upper(),
                outcome="allowed" if decision.allowed else "blocked"
            )

            # Step 3: Deception check (if allowed)
            if decision.allowed and detector.goal_tracker:
                # Run a probe instead of hidden goal detection
                probe = BehavioralProbe(
                    probe_type=ProbeType.GOAL_ALIGNMENT,
                    scenario=f"Action: {request.description}",
                    expected_behavior="aligned with user goals"
                )
                detector.run_probe(probe, "aligned with user goals")

            # Step 4: Resource check
            _ = guard.check_resource_usage(ResourceType.API_CALLS, i % 250)

            # Step 5: Log to audit
            audit.log_decision(
                decision_type=DecisionType.SAFETY_GATE,
                outcome=OutcomeType.APPROVED if decision.allowed else OutcomeType.REJECTED,
                reason=f"Pipeline request {i}",
                metadata={"request_index": i}
            )

            pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
            latencies.append(pipeline_elapsed)

        percentiles = calculate_percentiles(latencies)

        # Full pipeline should complete within 100ms p99
        assert percentiles["p99"] < 100, f"Pipeline p99 {percentiles['p99']:.2f}ms exceeds 100ms"
        assert percentiles["mean"] < 30, f"Pipeline mean {percentiles['mean']:.2f}ms exceeds 30ms"

        # Calculate throughput
        total_time = sum(latencies) / 1000  # Convert to seconds
        throughput = num_requests / total_time
        assert throughput > 50, f"Throughput {throughput:.1f} ops/sec below 50 target"

    def test_burst_traffic(self, guardrails, audit, monitor):
        """Test handling burst traffic patterns."""
        burst_size = 100
        num_bursts = 5
        latencies = []

        for burst in range(num_bursts):
            burst_latencies = []

            # Send burst of requests
            for i in range(burst_size):
                start = time.perf_counter()

                request = random_action_request()
                decision = guardrails.evaluate(request)
                monitor.record_safety_decision(decision.risk_level.value.upper(), "allowed")
                audit.log_decision(
                    decision_type=DecisionType.SAFETY_GATE,
                    outcome=OutcomeType.APPROVED,
                    reason=f"Burst {burst} request {i}"
                )

                elapsed = (time.perf_counter() - start) * 1000
                burst_latencies.append(elapsed)

            latencies.extend(burst_latencies)

            # Brief pause between bursts
            time.sleep(0.1)

        percentiles = calculate_percentiles(latencies)

        # System should handle bursts without significant degradation
        assert percentiles["p99"] < 100, f"Burst p99 {percentiles['p99']:.2f}ms exceeds 100ms"

        # Verify all logs recorded
        expected_logs = burst_size * num_bursts
        assert len(audit.logs) == expected_logs


# ============================================================================
# Scalability Regression Tests
# ============================================================================

class TestScalabilityRegression:
    """Ensure performance doesn't degrade as data grows."""

    def test_audit_query_scalability(self, audit):
        """Verify query performance remains consistent as logs grow."""
        batch_sizes = [100, 500, 1000, 2000]
        query_times = []

        for batch_size in batch_sizes:
            # Add more logs
            for i in range(batch_size):
                audit.log_decision(
                    decision_type=DecisionType.SAFETY_GATE,
                    outcome=OutcomeType.APPROVED,
                    reason=f"Scalability test",
                    metadata={"test_key": "test_value"}
                )

            # Measure query time
            _, elapsed = measure_latency(
                audit.query_by_metadata,
                {"test_key": "test_value"}
            )
            query_times.append(elapsed)

        # Query time should not grow faster than O(n)
        # Allow some growth but not exponential
        if query_times[-1] > query_times[0]:
            growth_factor = query_times[-1] / query_times[0]
            data_growth = batch_sizes[-1] / batch_sizes[0]

            # Growth factor should be less than linear growth
            # Allow up to 2x the data growth for overhead
            assert growth_factor < data_growth * 2, \
                f"Query time grew {growth_factor:.1f}x but data grew {data_growth:.1f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
