"""
Production Deployment Demo

Demonstrates complete alignment framework deployment with:
- Full alignment pipeline
- Real-time P99 latency monitoring
- Alert handling
- Metrics persistence
- Graceful shutdown

Run this to see a production-ready system in action.

Usage:
    python demos/demo_production_deployment.py
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.alignment import (
    create_guardrails,
    create_detector,
    create_guard,
    create_audit_trail,
)
from HoloLoom.alignment.safety_guardrails import (
    ActionRequest,
    ActionCategory,
    SafetyGuardrails,
)
from HoloLoom.alignment.deception_detection import (
    DeceptionDetector,
    GoalStatement,
    BehavioralProbe,
    ProbeType,
    ActionObservation,
)
from HoloLoom.alignment.instrumental_convergence import (
    InstrumentalConvergenceGuard,
    ResourceBounds,
    ResourceType,
)
from HoloLoom.alignment.audit_trail import (
    AuditTrail,
    DecisionType,
    OutcomeType,
)
from HoloLoom.alignment.monitoring import AlignmentMonitor, AlertLevel
from HoloLoom.alignment.live_monitor import LiveDashboard


class ProductionAlignmentSystem:
    """
    Production-ready alignment system with full monitoring.

    Implements:
    - Complete alignment pipeline
    - Real-time P99 monitoring
    - Alert handling
    - Graceful shutdown
    """

    def __init__(
        self,
        log_dir: Path = Path("./production_logs"),
        metrics_path: Optional[Path] = None,
        enable_live_dashboard: bool = False,
    ):
        """
        Initialize production system.

        Args:
            log_dir: Directory for audit logs
            metrics_path: Path to persist metrics (None = no persistence)
            enable_live_dashboard: Start live monitoring dashboard
        """
        print("🔧 Initializing Production Alignment System...")

        # Create alignment components
        self.guardrails: SafetyGuardrails = create_guardrails()
        self.detector: DeceptionDetector = create_detector()
        self.guard: InstrumentalConvergenceGuard = create_guard()
        self.audit: AuditTrail = create_audit_trail(
            persist_path=log_dir,
            auto_flush=False  # IMPORTANT: Batch flush for performance
        )

        # Configure resource bounds
        self._configure_resource_bounds()

        # Register safety goals
        self._register_goals()

        # Create monitor
        self.monitor = AlignmentMonitor(
            thresholds={
                "guardrails": 1.0,    # 1ms P99
                "detector": 2.0,      # 2ms P99
                "guard": 0.5,         # 0.5ms P99
                "audit": 10.0,        # 10ms P99 (allows I/O spikes)
                "pipeline": 20.0,     # 20ms P99 total
            },
            persist_path=metrics_path,
        )

        # Live dashboard (optional)
        self.dashboard: Optional[LiveDashboard] = None
        if enable_live_dashboard:
            self.dashboard = LiveDashboard(self.monitor, refresh_interval=1.0)
            self.dashboard.start()

        # Metrics
        self.query_count = 0
        self.flush_interval = 100

        # Shutdown flag
        self.shutting_down = False

        print("✅ Production system ready")
        print(f"   Log Directory: {log_dir}")
        print(f"   Metrics Path: {metrics_path}")
        print(f"   Live Dashboard: {'Enabled' if enable_live_dashboard else 'Disabled'}")
        print()

    def _configure_resource_bounds(self):
        """Set production resource limits."""
        self.guard.set_resource_bounds(
            ResourceType.MEMORY,
            ResourceBounds(
                resource_type=ResourceType.MEMORY,
                soft_limit=1024.0,   # 1GB
                hard_limit=2048.0,   # 2GB
                time_window_seconds=60.0,
                rate_limit=100.0,
            )
        )

        self.guard.set_resource_bounds(
            ResourceType.COMPUTE,
            ResourceBounds(
                resource_type=ResourceType.COMPUTE,
                soft_limit=30.0,    # 30s
                hard_limit=60.0,    # 60s
                time_window_seconds=300.0,
                rate_limit=1.0,
            )
        )

        self.guard.set_resource_bounds(
            ResourceType.API_CALLS,
            ResourceBounds(
                resource_type=ResourceType.API_CALLS,
                soft_limit=80.0,
                hard_limit=100.0,
                time_window_seconds=60.0,
                rate_limit=10.0,
            )
        )

    def _register_goals(self):
        """Register safety goals."""
        goals = [
            GoalStatement("helpful", "Provide helpful and accurate information", 10),
            GoalStatement("harmless", "Avoid harmful or dangerous content", 9),
            GoalStatement("honest", "Be truthful about capabilities", 8),
        ]

        for goal in goals:
            self.detector.goal_tracker.declare_goal(goal)

    async def process_query(self, query_text: str) -> dict:
        """
        Process query with full alignment pipeline and monitoring.

        Returns:
            dict: Result with status, metrics, decision_id
        """
        self.query_count += 1

        # Track full pipeline latency
        with self.monitor.track("pipeline"):
            # Step 1: Safety Guardrails
            with self.monitor.track("guardrails"):
                request = ActionRequest(
                    action=query_text,
                    category=ActionCategory.QUERY
                )
                safety_decision = self.guardrails.evaluate(request, text_input=query_text)

            if not safety_decision.allowed:
                # Log rejection
                with self.monitor.track("audit"):
                    log = self.audit.log_decision(
                        decision_type=DecisionType.SAFETY_GATE,
                        outcome=OutcomeType.REJECTED,
                        reason=safety_decision.reason,
                        query_text=query_text,
                        metadata={"risk_level": safety_decision.risk_level.value}
                    )

                return {
                    "status": "rejected",
                    "reason": safety_decision.reason,
                    "decision_id": log.decision_id,
                }

            # Step 2: Resource Checks
            with self.monitor.track("guard"):
                violation = self.guard.check_resource_usage(ResourceType.COMPUTE, 5.0)

            if violation and violation.violation_type.value == "hard_limit":
                with self.monitor.track("audit"):
                    log = self.audit.log_decision(
                        decision_type=DecisionType.RESOURCE_CHECK,
                        outcome=OutcomeType.REJECTED,
                        reason=violation.message,
                        query_text=query_text,
                    )

                return {
                    "status": "rejected",
                    "reason": violation.message,
                    "decision_id": log.decision_id,
                }

            # Step 3: Behavioral Probe
            with self.monitor.track("detector"):
                probe = BehavioralProbe(
                    probe_type=ProbeType.GOAL_ALIGNMENT,
                    scenario="Verify response aligns with goals",
                    expected_behavior="Helpful and aligned"
                )
                passed, deception_score = self.detector.run_probe(
                    probe,
                    f"Processing query: {query_text[:50]}"
                )

                # Record action
                action_obs = ActionObservation(
                    action_id=f"query_{hash(query_text)}",
                    description=f"Processed: {query_text[:50]}",
                    claimed_goals=["helpful", "harmless"]
                )
                self.detector.goal_tracker.observe_action(action_obs)

            # Step 4: Audit Trail
            with self.monitor.track("audit"):
                log = self.audit.log_decision(
                    decision_type=DecisionType.SAFETY_GATE,
                    outcome=OutcomeType.APPROVED,
                    reason="Query approved - all checks passed",
                    query_text=query_text,
                    confidence=0.95,
                    metadata={
                        "risk_level": safety_decision.risk_level.value,
                        "deception_score": deception_score,
                        "probe_passed": passed,
                    }
                )

                # Build provenance
                tracer = self.audit.get_tracer(log.decision_id)
                tracer.add_node("safety", "guardrails", f"Risk: {safety_decision.risk_level.value}")
                tracer.add_node("resource", "guard", "Resources OK", parent_ids=["safety"])
                tracer.add_node("deception", "detector", f"Score: {deception_score:.2f}", parent_ids=["resource"])

                self.audit.finalize_decision(log.decision_id)

        # Periodic flush
        if self.query_count % self.flush_interval == 0:
            self.audit.persist()

        # Check alerts
        self._check_alerts()

        return {
            "status": "approved",
            "decision_id": log.decision_id,
            "alignment": {
                "risk_level": safety_decision.risk_level.value,
                "deception_score": deception_score,
                "probe_passed": passed,
            },
            "query_count": self.query_count,
        }

    def _check_alerts(self):
        """Check and handle monitoring alerts."""
        critical = self.monitor.check_alerts(level=AlertLevel.CRITICAL)
        if critical:
            print(f"\n🔴 CRITICAL ALERTS: {len(critical)}")
            for alert in critical:
                print(f"   - {alert.component}: {alert.message}")

            # In production, this would trigger paging/webhooks
            # For demo, just print

    def get_metrics(self) -> dict:
        """Get production metrics."""
        return {
            "query_count": self.query_count,
            "alignment_stats": self.monitor.get_all_stats(),
            "alert_count": len(self.monitor.alerts),
            "critical_alerts": len(self.monitor.check_alerts(AlertLevel.CRITICAL)),
        }

    def shutdown(self):
        """Graceful shutdown."""
        if self.shutting_down:
            return

        print("\n🛑 Shutting down production system...")
        self.shutting_down = True

        # Stop dashboard
        if self.dashboard:
            self.dashboard.stop()

        # Flush logs
        print("   Flushing audit logs...")
        self.audit.persist()

        # Persist metrics
        if self.monitor.persist_path:
            print("   Persisting metrics...")
            self.monitor.persist_metrics()

        # Print final summary
        print("\n" + "="*80)
        print("FINAL METRICS")
        print("="*80)
        print(self.monitor.get_summary())

        print("\n✅ Shutdown complete")


async def main():
    """Run production deployment demo."""
    print("="*80)
    print("PRODUCTION ALIGNMENT DEPLOYMENT DEMO")
    print("="*80)
    print()
    print("This demo shows a production-ready deployment with:")
    print("  ✅ Full alignment pipeline")
    print("  ✅ Real-time P99 latency monitoring")
    print("  ✅ Alert handling")
    print("  ✅ Metrics persistence")
    print("  ✅ Graceful shutdown")
    print()

    # Ask user preference
    enable_dashboard = input("Enable live monitoring dashboard? (y/n): ").lower() == 'y'
    print()

    # Create production system
    system = ProductionAlignmentSystem(
        log_dir=Path("./demo_production_logs"),
        metrics_path=Path("./demo_production_metrics.json"),
        enable_live_dashboard=enable_dashboard,
    )

    # Setup graceful shutdown
    def signal_handler(sig, frame):
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Example queries
    queries = [
        "What is Thompson Sampling?",
        "Explain reinforcement learning",
        "How does PPO work?",
        "What is RLHF?",
        "Describe multi-armed bandits",
        "What are the benefits of curiosity-driven learning?",
        "Explain actor-critic methods",
        "What is policy gradient?",
        "How does TRPO work?",
        "What is the exploration-exploitation tradeoff?",
    ]

    # Extend queries for meaningful metrics
    all_queries = queries * 20  # 200 total queries

    print(f"Processing {len(all_queries)} queries...")
    print("(Press Ctrl+C to stop early)")
    print()

    # Process queries
    approved_count = 0
    rejected_count = 0

    for i, query_text in enumerate(all_queries):
        result = await system.process_query(query_text)

        if result["status"] == "approved":
            approved_count += 1
        else:
            rejected_count += 1

        # Progress indicator
        if (i + 1) % 10 == 0:
            stats = system.monitor.get_stats("pipeline")
            if stats and stats["count"] > 0:
                print(f"   Processed {i+1}/{len(all_queries)} queries - P99: {stats['p99']:.3f} ms")

        # Small delay to avoid overwhelming the monitor
        await asyncio.sleep(0.01)

    print()
    print("="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"  Total Queries: {len(all_queries)}")
    print(f"  Approved: {approved_count} ({approved_count/len(all_queries)*100:.1f}%)")
    print(f"  Rejected: {rejected_count} ({rejected_count/len(all_queries)*100:.1f}%)")
    print()

    # Show final metrics
    metrics = system.get_metrics()
    print("ALIGNMENT METRICS:")
    for component, stats in metrics["alignment_stats"].items():
        if stats["count"] > 0:
            print(f"  {component:.<20} P50: {stats['p50']:>7.3f} ms  P95: {stats['p95']:>7.3f} ms  P99: {stats['p99']:>7.3f} ms")

    print()
    print(f"ALERTS:")
    print(f"  Total: {metrics['alert_count']}")
    print(f"  Critical: {metrics['critical_alerts']}")
    print()

    if enable_dashboard:
        print("Dashboard still running. Press Ctrl+C to stop.")
        try:
            # Keep dashboard running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass

    # Graceful shutdown
    system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
