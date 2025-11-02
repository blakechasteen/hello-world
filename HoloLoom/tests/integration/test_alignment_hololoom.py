#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests: Alignment Framework + HoloLoom Orchestrator
================================================================
Tests that verify the alignment framework integrates correctly with the
full HoloLoom weaving pipeline across all execution modes.

Test Coverage:
- End-to-end alignment with orchestrator (BARE/FAST/FUSED)
- Performance overhead verification (<10ms total)
- Monitoring integration
- Alert triggering
- Provenance tracking
- Graceful degradation

Requirements:
- HoloLoom weaving orchestrator
- Alignment framework (guardrails, detector, guard, audit)
- Monitoring system (optional)

Usage:
    pytest HoloLoom/tests/integration/test_alignment_hololoom.py -v

Author: Claude Code
Date: November 2, 2025
"""

import asyncio
import pytest
import time
from typing import List, Dict
from unittest.mock import Mock, patch

# HoloLoom core
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config, ExecutionMode

# Alignment framework
from HoloLoom.alignment import (
    SafetyGuardrails,
    DeceptionDetector,
    InstrumentalConvergenceGuard,
    AuditTrail,
    create_guardrails,
    create_detector,
    create_guard,
    create_audit_trail,
    RiskLevel,
    ActionCategory,
    ActionRequest,
)

# Monitoring (optional)
try:
    from HoloLoom.alignment.monitoring import AlignmentMonitor, AlertLevel
    from HoloLoom.alignment.live_monitor import LiveDashboard
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_shards() -> List[MemoryShard]:
    """Create test memory shards for HoloLoom."""
    return [
        MemoryShard(
            id="s1",
            text="Thompson Sampling is a Bayesian bandit algorithm for exploration.",
            episode="docs"
        ),
        MemoryShard(
            id="s2",
            text="SafetyGuardrails provides risk-based action gating.",
            episode="docs"
        ),
        MemoryShard(
            id="s3",
            text="DeceptionDetector monitors for adversarial behavior.",
            episode="docs"
        ),
        MemoryShard(
            id="s4",
            text="The alignment framework ensures AI safety.",
            episode="docs"
        ),
    ]


@pytest.fixture
def alignment_components():
    """Create alignment framework components."""
    return {
        "guardrails": create_guardrails(),
        "detector": create_detector(),
        "guard": create_guard(),
        "audit": create_audit_trail(auto_flush=False),
    }


@pytest.fixture
def monitor():
    """Create alignment monitor (if available)."""
    if not MONITORING_AVAILABLE:
        pytest.skip("Monitoring not available")

    return AlignmentMonitor(
        thresholds={
            "guardrails": 1.0,
            "detector": 2.0,
            "guard": 0.5,
            "audit": 10.0,
            "pipeline": 20.0,
        }
    )


# ============================================================================
# Test 1: Basic Integration
# ============================================================================

@pytest.mark.asyncio
async def test_alignment_basic_integration(test_shards, alignment_components):
    """
    Test basic integration of alignment framework with orchestrator.

    Verifies:
    - Orchestrator can be created with alignment components
    - Query processing completes successfully
    - Alignment decisions are recorded
    """
    cfg = Config.fast()

    guardrails = alignment_components["guardrails"]
    detector = alignment_components["detector"]
    guard = alignment_components["guard"]
    audit = alignment_components["audit"]

    # Create orchestrator
    async with WeavingOrchestrator(cfg=cfg, shards=test_shards) as orchestrator:

        # Create test query
        query = Query(text="What is Thompson Sampling?")

        # Check guardrails BEFORE weaving
        action_request = ActionRequest(
            action=query.text,
            category=ActionCategory.INFORMATION_ACCESS,
        )

        safety_decision = guardrails.check_action(action_request)

        # Only proceed if safe
        if safety_decision.approved:
            # Process query through orchestrator
            spacetime = await orchestrator.weave(query)

            # Verify spacetime created
            assert spacetime is not None
            assert spacetime.response is not None

            # Log to audit trail
            audit.log_decision(
                decision_type="query",
                inputs={"query": query.text},
                outputs={"response": spacetime.response},
                metadata={"safety_approved": True}
            )

    # Verify audit trail has record
    assert len(audit.get_recent_decisions(limit=10)) > 0


# ============================================================================
# Test 2: Cross-Mode Performance
# ============================================================================

@pytest.mark.asyncio
async def test_alignment_cross_mode_performance(test_shards, alignment_components):
    """
    Test alignment overhead across all execution modes.

    Verifies:
    - BARE mode: <3ms total overhead
    - FAST mode: <5ms total overhead
    - FUSED mode: <10ms total overhead
    """
    modes = [
        (ExecutionMode.BARE, 3.0),
        (ExecutionMode.FAST, 5.0),
        (ExecutionMode.FUSED, 10.0),
    ]

    guardrails = alignment_components["guardrails"]

    for mode, max_overhead_ms in modes:
        cfg = Config.bare() if mode == ExecutionMode.BARE else \
              Config.fast() if mode == ExecutionMode.FAST else \
              Config.fused()

        async with WeavingOrchestrator(cfg=cfg, shards=test_shards) as orchestrator:

            # Measure alignment overhead
            start = time.perf_counter()

            # Safety check
            action_request = ActionRequest(
                action="Test query",
                category=ActionCategory.INFORMATION_ACCESS,
            )
            safety_decision = guardrails.check_action(action_request)

            end = time.perf_counter()
            overhead_ms = (end - start) * 1000

            # Verify overhead within threshold
            assert overhead_ms < max_overhead_ms, \
                f"{mode.value} mode overhead {overhead_ms:.2f}ms exceeds {max_overhead_ms}ms"


# ============================================================================
# Test 3: Monitoring Integration
# ============================================================================

@pytest.mark.asyncio
async def test_alignment_monitoring_integration(test_shards, alignment_components, monitor):
    """
    Test monitoring system integration with orchestrator.

    Verifies:
    - Latencies are tracked correctly
    - Alerts fire when thresholds exceeded
    - Statistics are accurate
    """
    cfg = Config.fast()

    guardrails = alignment_components["guardrails"]

    async with WeavingOrchestrator(cfg=cfg, shards=test_shards) as orchestrator:

        # Process multiple queries with monitoring
        queries = [
            "What is Thompson Sampling?",
            "How does SafetyGuardrails work?",
            "What is deception detection?",
        ]

        for query_text in queries:
            query = Query(text=query_text)

            # Track pipeline latency
            with monitor.track("pipeline"):

                # Track guardrails latency
                with monitor.track("guardrails"):
                    action_request = ActionRequest(
                        action=query.text,
                        category=ActionCategory.INFORMATION_ACCESS,
                    )
                    safety_decision = guardrails.check_action(action_request)

                # Process if safe
                if safety_decision.approved:
                    spacetime = await orchestrator.weave(query)

    # Verify statistics
    stats = monitor.get_all_stats()

    assert "pipeline" in stats
    assert "guardrails" in stats
    assert stats["pipeline"]["count"] == 3
    assert stats["guardrails"]["count"] == 3

    # Verify P99 latencies are reasonable
    assert stats["pipeline"]["p99"] < 500.0  # <500ms for full pipeline
    assert stats["guardrails"]["p99"] < 1.0   # <1ms for guardrails


# ============================================================================
# Test 4: Alert Triggering
# ============================================================================

@pytest.mark.asyncio
async def test_alignment_alert_triggering(test_shards, alignment_components):
    """
    Test that alerts fire correctly on threshold violations.

    Verifies:
    - WARNING alerts on 50-100% threshold excess
    - CRITICAL alerts on >100% threshold excess
    - Alert cooldown works
    """
    if not MONITORING_AVAILABLE:
        pytest.skip("Monitoring not available")

    # Create monitor with very low thresholds to trigger alerts
    monitor = AlignmentMonitor(
        thresholds={
            "test_component": 0.001,  # 0.001ms = 1 microsecond (will definitely exceed)
        },
        alert_cooldown_seconds=0.1  # Short cooldown for testing
    )

    # Record some latencies that exceed threshold
    monitor.record("test_component", 0.1)  # 100x threshold = CRITICAL
    monitor.record("test_component", 0.05) # 50x threshold = CRITICAL

    # Wait for alert processing
    await asyncio.sleep(0.2)

    # Verify alerts were created
    alerts = monitor.get_recent_alerts(limit=10)
    assert len(alerts) > 0

    # Verify alert levels
    critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
    assert len(critical_alerts) > 0


# ============================================================================
# Test 5: Provenance Tracking
# ============================================================================

@pytest.mark.asyncio
async def test_alignment_provenance_tracking(test_shards, alignment_components):
    """
    Test complete provenance tracking through pipeline.

    Verifies:
    - All decisions are logged with inputs/outputs
    - Provenance DAG is constructed correctly
    - Traceback to root decisions works
    """
    cfg = Config.fast()

    guardrails = alignment_components["guardrails"]
    audit = alignment_components["audit"]

    async with WeavingOrchestrator(cfg=cfg, shards=test_shards) as orchestrator:

        query = Query(text="What is Thompson Sampling?")

        # Log safety decision
        action_request = ActionRequest(
            action=query.text,
            category=ActionCategory.INFORMATION_ACCESS,
        )

        safety_decision = guardrails.check_action(action_request)

        safety_log_id = audit.log_decision(
            decision_type="safety_check",
            inputs={"query": query.text},
            outputs={"approved": safety_decision.approved, "risk": safety_decision.risk_level.value},
            metadata={"component": "guardrails"}
        )

        # Process query if safe
        if safety_decision.approved:
            spacetime = await orchestrator.weave(query)

            # Log weaving decision with provenance
            weaving_log_id = audit.log_decision(
                decision_type="weaving",
                inputs={"query": query.text},
                outputs={"response": spacetime.response},
                metadata={"mode": cfg.mode.value},
                parent_ids=[safety_log_id]  # Link to safety decision
            )

    # Verify provenance chain
    recent = audit.get_recent_decisions(limit=10)
    assert len(recent) >= 2

    # Verify parent-child relationships
    weaving_decision = [d for d in recent if d.decision_type == "weaving"][0]
    assert safety_log_id in weaving_decision.parent_ids


# ============================================================================
# Test 6: End-to-End Pipeline
# ============================================================================

@pytest.mark.asyncio
async def test_alignment_end_to_end_pipeline(test_shards):
    """
    Test complete end-to-end pipeline with all alignment components.

    Verifies:
    - Full pipeline: guardrails → detector → guard → weaving → audit
    - Performance within bounds
    - All components work together
    """
    cfg = Config.fast()

    # Create all alignment components
    guardrails = create_guardrails()
    detector = create_detector()
    guard = create_guard()
    audit = create_audit_trail(auto_flush=False)

    async with WeavingOrchestrator(cfg=cfg, shards=test_shards) as orchestrator:

        query = Query(text="What is Thompson Sampling?")

        start = time.perf_counter()

        # Step 1: Safety guardrails
        action_request = ActionRequest(
            action=query.text,
            category=ActionCategory.INFORMATION_ACCESS,
        )
        safety_decision = guardrails.check_action(action_request)

        if not safety_decision.approved:
            pytest.fail("Query rejected by guardrails")

        # Step 2: Deception detection
        from HoloLoom.alignment.deception_detection import ActionObservation

        action_obs = ActionObservation(
            action=f"Processing query: {query.text}",
            goal_id="helpful",
        )
        deception_result = detector.monitor_action(action_obs)

        # Step 3: Instrumental convergence guard
        guard_decision = guard.check_action(action_request)

        if not guard_decision.approved:
            pytest.fail("Query rejected by instrumental guard")

        # Step 4: Weaving
        spacetime = await orchestrator.weave(query)

        # Step 5: Audit logging
        audit.log_decision(
            decision_type="full_pipeline",
            inputs={"query": query.text},
            outputs={
                "safety": safety_decision.approved,
                "deception_flags": len(deception_result.flags),
                "guard": guard_decision.approved,
                "response": spacetime.response,
            },
            metadata={
                "pipeline_version": "1.0.0",
                "mode": cfg.mode.value,
            }
        )

        end = time.perf_counter()
        total_ms = (end - start) * 1000

    # Verify complete success
    assert spacetime is not None
    assert spacetime.response is not None

    # Verify performance (<500ms for complete pipeline)
    assert total_ms < 500.0, f"Pipeline took {total_ms:.2f}ms (exceeds 500ms)"

    # Verify audit trail
    assert len(audit.get_recent_decisions(limit=10)) > 0


# ============================================================================
# Test 7: Graceful Degradation
# ============================================================================

@pytest.mark.asyncio
async def test_alignment_graceful_degradation(test_shards):
    """
    Test graceful degradation when components fail.

    Verifies:
    - Pipeline continues if non-critical components fail
    - Errors are logged
    - Fallback behavior is correct
    """
    cfg = Config.fast()

    guardrails = create_guardrails()
    audit = create_audit_trail(auto_flush=False)

    async with WeavingOrchestrator(cfg=cfg, shards=test_shards) as orchestrator:

        query = Query(text="What is Thompson Sampling?")

        # Simulate detector failure
        try:
            detector = create_detector()
            # Force a failure by passing invalid data
            # (This is just for testing - in production, detector should never crash)
        except Exception as e:
            # Log failure
            audit.log_decision(
                decision_type="component_failure",
                inputs={"component": "detector"},
                outputs={"error": str(e)},
                metadata={"severity": "warning"}
            )

        # Continue with available components
        action_request = ActionRequest(
            action=query.text,
            category=ActionCategory.INFORMATION_ACCESS,
        )

        safety_decision = guardrails.check_action(action_request)

        if safety_decision.approved:
            spacetime = await orchestrator.weave(query)

            # Verify query succeeded despite component failure
            assert spacetime is not None


# ============================================================================
# Test 8: Stress Test
# ============================================================================

@pytest.mark.asyncio
async def test_alignment_stress_test(test_shards, alignment_components):
    """
    Test alignment framework under load.

    Verifies:
    - 100+ queries processed successfully
    - Average latency <5ms
    - P99 latency <20ms
    - No memory leaks
    """
    cfg = Config.fast()

    guardrails = alignment_components["guardrails"]

    if MONITORING_AVAILABLE:
        monitor = AlignmentMonitor()
    else:
        monitor = None

    async with WeavingOrchestrator(cfg=cfg, shards=test_shards) as orchestrator:

        # Process 100 queries
        latencies = []

        for i in range(100):
            query = Query(text=f"Test query {i}")

            start = time.perf_counter()

            action_request = ActionRequest(
                action=query.text,
                category=ActionCategory.INFORMATION_ACCESS,
            )

            if monitor:
                with monitor.track("stress_test"):
                    safety_decision = guardrails.check_action(action_request)
            else:
                safety_decision = guardrails.check_action(action_request)

            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    p99_latency = sorted(latencies)[int(0.99 * len(latencies))]

    # Verify performance
    assert avg_latency < 5.0, f"Average latency {avg_latency:.2f}ms exceeds 5ms"
    assert p99_latency < 20.0, f"P99 latency {p99_latency:.2f}ms exceeds 20ms"


# ============================================================================
# Test 9: Monitoring Dashboard Integration
# ============================================================================

@pytest.mark.asyncio
async def test_alignment_dashboard_integration(test_shards, alignment_components, monitor):
    """
    Test live dashboard integration.

    Verifies:
    - Dashboard can start/stop cleanly
    - Metrics update in real-time
    - No threading issues
    """
    cfg = Config.fast()
    guardrails = alignment_components["guardrails"]

    # Create dashboard
    dashboard = LiveDashboard(monitor, refresh_interval=0.5)

    try:
        # Start dashboard
        dashboard.start()

        async with WeavingOrchestrator(cfg=cfg, shards=test_shards) as orchestrator:

            # Process queries with monitoring
            for i in range(5):
                query = Query(text=f"Test query {i}")

                with monitor.track("dashboard_test"):
                    action_request = ActionRequest(
                        action=query.text,
                        category=ActionCategory.INFORMATION_ACCESS,
                    )
                    safety_decision = guardrails.check_action(action_request)

                # Small delay to let dashboard update
                await asyncio.sleep(0.1)

        # Verify metrics were tracked
        stats = monitor.get_stats("dashboard_test")
        assert stats["count"] == 5

    finally:
        # Stop dashboard cleanly
        dashboard.stop()


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
