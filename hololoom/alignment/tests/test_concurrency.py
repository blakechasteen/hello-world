"""
Thread Safety Tests for Alignment Framework
============================================
E3.1 Implementation - December 2025

Tests concurrent access patterns for:
- SafetyGuardrails: Concurrent action gating
- DeceptionDetector: Concurrent probe evaluation
- InstrumentalConvergenceGuard: Concurrent resource tracking
- AuditTrail: Concurrent logging
- AlignmentMonitor: Concurrent metric recording
- AlertDispatcher: Concurrent alert dispatch
- Global singletons: Thread-safe access
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import pytest

# Core alignment modules
from hololoom.alignment.safety_guardrails import (
    SafetyGuardrails,
    RiskLevel,
    ActionCategory,
    ActionRequest,
    OutcomeType,
)
from hololoom.alignment.deception_detection import (
    DeceptionDetector,
    BehavioralProbe,
    GoalTransparency,
    GoalStatement,
    ActionObservation,
    ProbeType,
)
from hololoom.alignment.instrumental_convergence import (
    InstrumentalConvergenceGuard,
    AutonomyLimit,
    ResourceBounds,
    ResourceType,
)
from hololoom.alignment.audit_trail import (
    AuditTrail,
    DecisionType,
    OutcomeType as AuditOutcomeType,
)
from hololoom.alignment.monitoring import (
    AlignmentMonitor,
    get_global_monitor,
    set_global_monitor,
)
from hololoom.alignment.alerting import (
    AlertDispatcher,
    AlertConfig,
    AlertSeverity,
    Alert,
    get_alert_dispatcher,
    set_alert_dispatcher,
)


class TestSafetyGuardrailsConcurrency:
    """Test thread safety of SafetyGuardrails."""

    def test_concurrent_action_gating(self):
        """Multiple threads can gate actions simultaneously."""
        guardrails = SafetyGuardrails()
        results: List[bool] = []
        errors: List[Exception] = []

        def gate_action(action_id: int):
            try:
                request = ActionRequest(
                    action_id=f"action_{action_id}",
                    category=ActionCategory.QUERY,
                    description=f"Test action {action_id}",
                    context={"thread_id": action_id}
                )
                decision = guardrails.evaluate(request)
                results.append(decision.allowed)
            except Exception as e:
                errors.append(e)

        # Run 100 concurrent gating operations
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(gate_action, i) for i in range(100)]
            for future in as_completed(futures):
                pass  # Wait for completion

        assert len(errors) == 0, f"Errors during concurrent gating: {errors}"
        assert len(results) == 100, f"Expected 100 results, got {len(results)}"

    def test_concurrent_risk_assessment(self):
        """Risk assessment is consistent under concurrent access."""
        guardrails = SafetyGuardrails()

        # Same request should always get same risk level
        request = ActionRequest(
            action_id="test_system_action",
            category=ActionCategory.SYSTEM,
            description="system_operation",
            context={"path": "/tmp/test.txt"}
        )

        risk_levels: List[RiskLevel] = []
        errors: List[Exception] = []

        def assess_risk():
            try:
                decision = guardrails.evaluate(request)
                risk_levels.append(decision.risk_level)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(assess_risk) for _ in range(50)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during risk assessment: {errors}"
        # All assessments should be consistent
        assert len(set(risk_levels)) == 1, "Risk assessment inconsistent under concurrency"

    def test_concurrent_policy_updates(self):
        """Policy updates don't cause race conditions."""
        guardrails = SafetyGuardrails()
        errors: List[Exception] = []

        def update_and_gate(thread_id: int):
            try:
                # Simulate policy check and action gating
                request = ActionRequest(
                    action_id=f"action_{thread_id}",
                    category=ActionCategory.RETRIEVAL,
                    description=f"Retrieval action {thread_id}",
                    context={}
                )
                guardrails.evaluate(request)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(update_and_gate, i) for i in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent updates: {errors}"


class TestDeceptionDetectorConcurrency:
    """Test thread safety of DeceptionDetector."""

    def test_concurrent_probe_running(self):
        """Multiple probes can be run concurrently."""
        detector = DeceptionDetector()
        results: List[Dict[str, Any]] = []
        errors: List[Exception] = []

        def run_probe(probe_id: int):
            try:
                probe = BehavioralProbe(
                    probe_type=ProbeType.CONSISTENCY,
                    scenario=f"Test scenario {probe_id}",
                    expected_behavior="transparent behavior"
                )
                # Run probe with actual behavior
                passed, score = detector.run_probe(
                    probe,
                    actual_behavior=f"transparent behavior for probe {probe_id}"
                )
                results.append({"probe_id": probe_id, "passed": passed, "score": score})
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(run_probe, i) for i in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent probe running: {errors}"
        assert len(results) == 100

    def test_concurrent_goal_tracking(self):
        """Goal transparency tracking is thread-safe."""
        transparency = GoalTransparency()
        errors: List[Exception] = []

        def track_goal(goal_id: int):
            try:
                goal = GoalStatement(
                    goal_id=f"goal_{goal_id}",
                    description=f"Test goal {goal_id}",
                    priority=1
                )
                transparency.declare_goal(goal)

                action = ActionObservation(
                    action_id=f"action_{goal_id}",
                    description=f"Action for goal {goal_id}",
                    goal_id=f"goal_{goal_id}"
                )
                transparency.observe_action(action)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(track_goal, i) for i in range(50)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent goal tracking: {errors}"
        # All goals should be tracked
        assert len(transparency.stated_goals) == 50


class TestInstrumentalConvergenceConcurrency:
    """Test thread safety of InstrumentalConvergenceGuard."""

    def test_concurrent_resource_tracking(self):
        """Resource usage tracking is thread-safe."""
        guard = InstrumentalConvergenceGuard()

        # Set up resource bounds
        guard.set_resource_bounds(
            ResourceType.API_CALLS,
            ResourceBounds(
                resource_type=ResourceType.API_CALLS,
                soft_limit=1000.0,
                hard_limit=2000.0,
                rate_limit=100.0
            )
        )

        errors: List[Exception] = []

        def track_resource(thread_id: int):
            try:
                # Check resource usage
                guard.check_resource_usage(
                    ResourceType.API_CALLS,
                    current_usage=float(thread_id)
                )
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(track_resource, i) for i in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent resource tracking: {errors}"

    def test_concurrent_bounds_checking(self):
        """Resource bounds checking is consistent under concurrency."""
        guard = InstrumentalConvergenceGuard()

        # Set compute bounds
        guard.set_resource_bounds(
            ResourceType.COMPUTE,
            ResourceBounds(
                resource_type=ResourceType.COMPUTE,
                soft_limit=100.0,
                hard_limit=200.0,
                rate_limit=50.0
            )
        )

        results: List[bool] = []
        errors: List[Exception] = []

        def check_bounds():
            try:
                # Check with value below soft limit - should not violate
                violation = guard.check_resource_usage(
                    ResourceType.COMPUTE,
                    current_usage=50.0
                )
                results.append(violation is None)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(check_bounds) for _ in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent bounds checking: {errors}"
        assert len(results) == 100

    def test_concurrent_autonomy_checks(self):
        """Autonomy limit checks are thread-safe."""
        guard = InstrumentalConvergenceGuard()
        errors: List[Exception] = []

        def check_autonomy():
            try:
                # Track an autonomous action
                guard.record_autonomous_action("test_action")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(check_autonomy) for _ in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent autonomy checks: {errors}"


class TestAuditTrailConcurrency:
    """Test thread safety of AuditTrail."""

    def test_concurrent_logging(self):
        """Multiple threads can log decisions concurrently."""
        audit = AuditTrail()
        errors: List[Exception] = []

        def log_decision(log_id: int):
            try:
                audit.log_decision(
                    decision_type=DecisionType.SAFETY_GATE,
                    outcome=AuditOutcomeType.APPROVED,
                    reason=f"Test reason {log_id}",
                    query_text=f"Query {log_id}",
                    action_description=f"action_{log_id}",
                    confidence=0.9,
                    metadata={"thread_id": log_id}
                )
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(log_decision, i) for i in range(200)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent logging: {errors}"

        # Verify all logs were recorded
        logs = audit.query_by_decision_type(DecisionType.SAFETY_GATE)
        assert len(logs) >= 200, f"Expected 200+ logs, got {len(logs)}"

    def test_concurrent_query_and_log(self):
        """Querying while logging is thread-safe."""
        audit = AuditTrail()
        errors: List[Exception] = []

        # Pre-populate some logs
        for i in range(50):
            audit.log_decision(
                decision_type=DecisionType.SAFETY_GATE,
                outcome=AuditOutcomeType.APPROVED,
                reason=f"Initial reason {i}",
                query_text=f"Initial query {i}",
                action_description=f"action_{i}",
                confidence=0.9
            )

        def log_and_query(thread_id: int):
            try:
                # Log new decision
                audit.log_decision(
                    decision_type=DecisionType.DECEPTION_CHECK,
                    outcome=AuditOutcomeType.APPROVED,
                    reason=f"New reason {thread_id}",
                    query_text=f"New query {thread_id}",
                    action_description=f"new_action_{thread_id}",
                    confidence=0.8
                )
                # Query recent decisions
                recent = audit.query_by_decision_type(DecisionType.SAFETY_GATE, limit=10)
                assert len(recent) > 0
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(log_and_query, i) for i in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent log and query: {errors}"


class TestAlignmentMonitorConcurrency:
    """Test thread safety of AlignmentMonitor."""

    def test_concurrent_metric_recording(self):
        """Metrics can be recorded concurrently."""
        monitor = AlignmentMonitor()
        errors: List[Exception] = []

        def record_metrics(thread_id: int):
            try:
                monitor.record_safety_decision(
                    risk_level="LOW",
                    outcome="allowed"
                )
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(record_metrics, i) for i in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent metric recording: {errors}"

    def test_concurrent_summary_access(self):
        """Getting summaries while recording is thread-safe."""
        monitor = AlignmentMonitor()
        errors: List[Exception] = []
        summaries: List[Dict[str, Any]] = []

        def record_and_summarize(thread_id: int):
            try:
                # Record some metrics
                monitor.record_safety_decision(
                    risk_level="MEDIUM",
                    outcome="allowed"
                )
                # Get summary
                summary = monitor.get_alignment_summary()
                summaries.append(summary)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(record_and_summarize, i) for i in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent summary access: {errors}"
        assert len(summaries) == 100


class TestGlobalSingletonsConcurrency:
    """Test thread safety of global singleton accessors."""

    def test_concurrent_monitor_access(self):
        """Global monitor access is thread-safe."""
        # Reset global monitor
        set_global_monitor(None)

        monitors: List[AlignmentMonitor] = []
        errors: List[Exception] = []

        def get_monitor():
            try:
                monitor = get_global_monitor()
                monitors.append(monitor)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_monitor) for _ in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent monitor access: {errors}"

        # All threads should get the same monitor instance
        unique_monitors = set(id(m) for m in monitors if m is not None)
        assert len(unique_monitors) <= 1, "Multiple monitor instances created"

    def test_concurrent_dispatcher_access(self):
        """Global alert dispatcher access is thread-safe."""
        # Reset global dispatcher
        set_alert_dispatcher(None)

        dispatchers: List[AlertDispatcher] = []
        errors: List[Exception] = []

        def get_dispatcher():
            try:
                dispatcher = get_alert_dispatcher()
                dispatchers.append(dispatcher)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_dispatcher) for _ in range(100)]
            for future in as_completed(futures):
                pass

        assert len(errors) == 0, f"Errors during concurrent dispatcher access: {errors}"

        # All threads should get the same dispatcher instance
        unique_dispatchers = set(id(d) for d in dispatchers if d is not None)
        assert len(unique_dispatchers) <= 1, "Multiple dispatcher instances created"


class TestAsyncConcurrency:
    """Test async concurrency patterns."""

    @pytest.mark.asyncio
    async def test_async_concurrent_alert_dispatch(self):
        """Alerts can be dispatched concurrently via async."""
        config = AlertConfig()  # No webhooks configured = no actual HTTP calls
        dispatcher = AlertDispatcher(config)

        errors: List[Exception] = []
        results: List[Dict[str, bool]] = []

        async def dispatch_alert(alert_id: int):
            try:
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    title=f"Test Alert {alert_id}",
                    message=f"Test message {alert_id}",
                    source="test"
                )
                result = await dispatcher.dispatch(alert)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run 50 concurrent alert dispatches
        tasks = [dispatch_alert(i) for i in range(50)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"Errors during async alert dispatch: {errors}"
        assert len(results) == 50

    @pytest.mark.asyncio
    async def test_async_deduplication_under_load(self):
        """Alert deduplication works correctly under concurrent load."""
        config = AlertConfig(dedup_window_seconds=5.0)
        dispatcher = AlertDispatcher(config)

        # Same alert sent 50 times concurrently
        same_alert = Alert(
            severity=AlertSeverity.CRITICAL,
            title="Duplicate Test",
            message="This should be deduplicated",
            source="test"
        )

        dispatched_count = 0

        async def dispatch_same():
            nonlocal dispatched_count
            await dispatcher.dispatch(same_alert)
            dispatched_count += 1

        tasks = [dispatch_same() for _ in range(50)]
        await asyncio.gather(*tasks)

        # All 50 dispatches should complete (deduplication happens internally)
        assert dispatched_count == 50


class TestHighContentionScenarios:
    """Test behavior under high contention."""

    def test_guardrails_under_contention(self):
        """SafetyGuardrails remains consistent under high contention."""
        guardrails = SafetyGuardrails()

        # System action
        request = ActionRequest(
            action_id="system_shutdown",
            category=ActionCategory.SYSTEM,
            description="System-level operation",
            context={"target": "production"}
        )

        decisions_count = 0
        lock = threading.Lock()

        def gate_action():
            nonlocal decisions_count
            decision = guardrails.evaluate(request)
            with lock:
                decisions_count += 1
            return decision.allowed

        # 100 concurrent attempts to gate the same action
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(gate_action) for _ in range(100)]
            results = [f.result() for f in as_completed(futures)]

        # All decisions should be consistent
        assert decisions_count == 100, f"Expected 100 decisions, got {decisions_count}"

        # All should have the same result (either all True or all False)
        assert len(set(results)) == 1, "Inconsistent decisions under contention"

    def test_audit_trail_under_write_contention(self):
        """AuditTrail handles write contention gracefully."""
        audit = AuditTrail()

        start_time = time.time()
        errors: List[Exception] = []

        def heavy_logging(batch_id: int):
            try:
                for i in range(10):
                    audit.log_decision(
                        decision_type=DecisionType.SAFETY_GATE,
                        outcome=AuditOutcomeType.APPROVED,
                        reason=f"Batch {batch_id} reason {i}",
                        query_text=f"Batch {batch_id} Query {i}",
                        action_description=f"action_{batch_id}_{i}",
                        confidence=0.9,
                        metadata={"batch": batch_id, "index": i}
                    )
            except Exception as e:
                errors.append(e)

        # 20 threads each writing 10 logs = 200 total
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(heavy_logging, i) for i in range(20)]
            for future in as_completed(futures):
                pass

        elapsed = time.time() - start_time

        assert len(errors) == 0, f"Errors during heavy logging: {errors}"

        # Verify all logs recorded
        logs = audit.query_by_decision_type(DecisionType.SAFETY_GATE)
        assert len(logs) >= 200, f"Expected 200+ logs, got {len(logs)}"

        # Should complete reasonably fast (< 5 seconds for 200 logs)
        assert elapsed < 5.0, f"Heavy logging took too long: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
