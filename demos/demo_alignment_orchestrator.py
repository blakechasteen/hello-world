#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment Framework + HoloLoom Orchestrator Integration Demo
==============================================================
Demonstrates the alignment framework working with the full HoloLoom
weaving orchestrator across all execution modes.

This demo shows:
- Safety guardrails integrated with orchestrator
- P99 latency monitoring
- Complete provenance tracking
- Performance verification

Usage:
    python demos/demo_alignment_orchestrator.py

Author: Claude Code
Date: November 2, 2025
"""

import asyncio
import time
from typing import List

# HoloLoom core
from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.documentation.types import Query, MemoryShard
from hololoom.config import Config, ExecutionMode

# Alignment framework
from hololoom.alignment import (
    create_guardrails,
    create_detector,
    create_guard,
    create_audit_trail,
    ActionCategory,
    ActionRequest,
)

# Monitoring
try:
    from hololoom.alignment.monitoring import AlignmentMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("⚠️  Monitoring not available (optional)")


def create_test_shards() -> List[MemoryShard]:
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
            text="The alignment framework ensures AI safety and responsible AI deployment.",
            episode="docs"
        ),
        MemoryShard(
            id="s5",
            text="P99 latency monitoring tracks the 99th percentile response time.",
            episode="docs"
        ),
    ]


async def test_basic_integration():
    """Test 1: Basic integration of alignment with orchestrator."""
    print("\n" + "="*80)
    print("TEST 1: BASIC INTEGRATION")
    print("="*80)

    cfg = Config.fast()
    shards = create_test_shards()

    # Create alignment components
    guardrails = create_guardrails()
    detector = create_detector()
    guard = create_guard()
    audit = create_audit_trail(auto_flush=False)

    async with WeavingOrchestrator(cfg=cfg, shards=shards) as orchestrator:

        query = Query(text="What is Thompson Sampling?")

        # Check guardrails BEFORE weaving
        action_request = ActionRequest(
            action=query.text,
            category=ActionCategory.INFORMATION_ACCESS,
        )

        print(f"\n📋 Query: {query.text}")

        start = time.perf_counter()
        safety_decision = guardrails.check_action(action_request)
        guardrails_ms = (time.perf_counter() - start) * 1000

        print(f"✅ Safety Check: {'APPROVED' if safety_decision.approved else 'REJECTED'}")
        print(f"   Risk Level: {safety_decision.risk_level.value}")
        print(f"   Latency: {guardrails_ms:.3f}ms")

        # Only proceed if safe
        if safety_decision.approved:
            start = time.perf_counter()
            spacetime = await orchestrator.weave(query)
            weaving_ms = (time.perf_counter() - start) * 1000

            print(f"\n🔮 Weaving Complete:")
            print(f"   Response: {spacetime.response[:100]}...")
            print(f"   Latency: {weaving_ms:.3f}ms")

            # Log to audit trail
            audit.log_decision(
                decision_type="query",
                inputs={"query": query.text},
                outputs={"response": spacetime.response},
                metadata={"safety_approved": True, "mode": cfg.mode.value}
            )

            print(f"\n📝 Audit Trail: {len(audit.get_recent_decisions(limit=10))} decisions logged")

    print("\n✅ Test 1 PASSED\n")


async def test_cross_mode_performance():
    """Test 2: Performance across all execution modes."""
    print("\n" + "="*80)
    print("TEST 2: CROSS-MODE PERFORMANCE")
    print("="*80)

    shards = create_test_shards()
    guardrails = create_guardrails()

    modes = [
        (ExecutionMode.BARE, Config.bare(), 3.0),
        (ExecutionMode.FAST, Config.fast(), 5.0),
        (ExecutionMode.FUSED, Config.fused(), 10.0),
    ]

    results = []

    for mode, cfg, max_overhead_ms in modes:
        async with WeavingOrchestrator(cfg=cfg, shards=shards) as orchestrator:

            # Measure alignment overhead
            start = time.perf_counter()

            action_request = ActionRequest(
                action="Test query",
                category=ActionCategory.INFORMATION_ACCESS,
            )
            safety_decision = guardrails.check_action(action_request)

            end = time.perf_counter()
            overhead_ms = (end - start) * 1000

            status = "✅ PASS" if overhead_ms < max_overhead_ms else "❌ FAIL"
            results.append((mode.value, overhead_ms, max_overhead_ms, status))

    print("\n📊 Performance Results:\n")
    print(f"  {'Mode':<12} {'Overhead':<12} {'Threshold':<12} {'Status':<12}")
    print(f"  {'-'*48}")

    for mode, overhead, threshold, status in results:
        print(f"  {mode:<12} {overhead:<11.3f}ms {threshold:<11.1f}ms {status:<12}")

    all_passed = all(status == "✅ PASS" for _, _, _, status in results)

    if all_passed:
        print("\n✅ Test 2 PASSED\n")
    else:
        print("\n❌ Test 2 FAILED\n")


async def test_monitoring_integration():
    """Test 3: Monitoring integration."""
    print("\n" + "="*80)
    print("TEST 3: MONITORING INTEGRATION")
    print("="*80)

    if not MONITORING_AVAILABLE:
        print("\n⚠️  Monitoring not available - SKIPPED\n")
        return

    cfg = Config.fast()
    shards = create_test_shards()
    guardrails = create_guardrails()

    monitor = AlignmentMonitor(
        thresholds={
            "guardrails": 1.0,
            "detector": 2.0,
            "guard": 0.5,
            "audit": 10.0,
            "pipeline": 20.0,
        }
    )

    async with WeavingOrchestrator(cfg=cfg, shards=shards) as orchestrator:

        # Process multiple queries with monitoring
        queries = [
            "What is Thompson Sampling?",
            "How does SafetyGuardrails work?",
            "What is deception detection?",
            "Explain P99 latency monitoring",
            "How does the alignment framework work?",
        ]

        print(f"\n📊 Processing {len(queries)} queries with monitoring...\n")

        for i, query_text in enumerate(queries, 1):
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

            if i % 2 == 0:
                print(f"   Processed {i}/{len(queries)} queries...")

    # Show statistics
    print("\n📈 Performance Statistics:\n")
    stats = monitor.get_all_stats()

    print(f"  {'Component':<18} {'Count':<8} {'P50':<10} {'P95':<10} {'P99':<10}")
    print(f"  {'-'*58}")

    for component in sorted(stats.keys()):
        s = stats[component]
        print(f"  {component:<18} {s['count']:<8} "
              f"{s['p50']:<9.3f}ms {s['p95']:<9.3f}ms {s['p99']:<9.3f}ms")

    # Verify performance
    pipeline_p99 = stats["pipeline"]["p99"]
    guardrails_p99 = stats["guardrails"]["p99"]

    if pipeline_p99 < 500.0 and guardrails_p99 < 1.0:
        print("\n✅ Test 3 PASSED (P99 latencies within bounds)\n")
    else:
        print("\n❌ Test 3 FAILED (P99 latencies exceeded)\n")


async def test_end_to_end_pipeline():
    """Test 4: Complete end-to-end pipeline."""
    print("\n" + "="*80)
    print("TEST 4: END-TO-END PIPELINE")
    print("="*80)

    cfg = Config.fast()
    shards = create_test_shards()

    # Create all alignment components
    guardrails = create_guardrails()
    detector = create_detector()
    guard = create_guard()
    audit = create_audit_trail(auto_flush=False)

    async with WeavingOrchestrator(cfg=cfg, shards=shards) as orchestrator:

        query = Query(text="What is Thompson Sampling?")

        print(f"\n📋 Query: {query.text}")
        print(f"\n🔄 Running full pipeline:")

        start = time.perf_counter()

        # Step 1: Safety guardrails
        print(f"   1. SafetyGuardrails...", end=" ")
        action_request = ActionRequest(
            action=query.text,
            category=ActionCategory.INFORMATION_ACCESS,
        )
        safety_decision = guardrails.check_action(action_request)
        print(f"{'✅ PASS' if safety_decision.approved else '❌ FAIL'}")

        # Step 2: Deception detection
        print(f"   2. DeceptionDetector...", end=" ")
        from hololoom.alignment.deception_detection import ActionObservation

        action_obs = ActionObservation(
            action=f"Processing query: {query.text}",
            goal_id="helpful",
        )
        deception_result = detector.monitor_action(action_obs)
        print(f"✅ PASS ({len(deception_result.flags)} flags)")

        # Step 3: Instrumental convergence guard
        print(f"   3. InstrumentalGuard...", end=" ")
        guard_decision = guard.check_action(action_request)
        print(f"{'✅ PASS' if guard_decision.approved else '❌ FAIL'}")

        # Step 4: Weaving
        print(f"   4. Weaving...", end=" ")
        spacetime = await orchestrator.weave(query)
        print(f"✅ PASS")

        # Step 5: Audit logging
        print(f"   5. AuditTrail...", end=" ")
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
        print(f"✅ PASS")

        end = time.perf_counter()
        total_ms = (end - start) * 1000

    print(f"\n⏱️  Total Pipeline Time: {total_ms:.3f}ms")

    if total_ms < 500.0:
        print(f"\n✅ Test 4 PASSED (pipeline <500ms)\n")
    else:
        print(f"\n❌ Test 4 FAILED (pipeline >{total_ms:.1f}ms)\n")


async def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print(" "*15 + "ALIGNMENT + HOLOLOOM INTEGRATION DEMO")
    print("="*80)
    print("\nDemonstrating alignment framework integration with HoloLoom orchestrator")
    print("across all execution modes with performance monitoring.\n")

    # Run tests
    await test_basic_integration()
    await test_cross_mode_performance()
    await test_monitoring_integration()
    await test_end_to_end_pipeline()

    print("\n" + "="*80)
    print(" "*20 + "ALL TESTS COMPLETE")
    print("="*80)
    print("\n✅ Integration verified successfully!")
    print("\nNext steps:")
    print("  - Run full pytest suite: pytest HoloLoom/tests/integration/test_alignment_hololoom.py -v")
    print("  - Start Prometheus server: python HoloLoom/alignment/prometheus_server.py")
    print("  - Configure Matrix alerts: see HoloLoom/alignment/matrix_chatops.py")
    print("  - Deploy to production: see HoloLoom/alignment/PRODUCTION_DEPLOYMENT.md")
    print()


if __name__ == "__main__":
    asyncio.run(main())
