"""
Live Monitoring Dashboard for Alignment Framework

Real-time terminal-based dashboard showing latency metrics,
alerts, and system health.

Usage:
    python HoloLoom/alignment/live_monitor.py

Or programmatically:
    from hololoom.alignment.live_monitor import LiveDashboard

    dashboard = LiveDashboard(monitor)
    dashboard.start()  # Runs in background
    # ... process queries ...
    dashboard.stop()
"""

import sys
import threading
import time
from datetime import datetime, timedelta

from hololoom.alignment.monitoring import AlertLevel, AlignmentMonitor


class LiveDashboard:
    """
    Real-time terminal dashboard for alignment monitoring.

    Displays continuously updating metrics in the terminal.
    """

    def __init__(
        self,
        monitor: AlignmentMonitor,
        refresh_interval: float = 1.0,  # seconds
    ):
        """
        Initialize dashboard.

        Args:
            monitor: AlignmentMonitor instance to display
            refresh_interval: Update frequency in seconds
        """
        self.monitor = monitor
        self.refresh_interval = refresh_interval

        # Control flags
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self):
        """Start dashboard in background thread."""
        if self.running:
            print("⚠️  Dashboard already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("📊 Live dashboard started (background thread)")

    def stop(self):
        """Stop dashboard."""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

        print("\n📊 Live dashboard stopped")

    def _run_loop(self):
        """Main dashboard loop."""
        while self.running:
            self._render()
            time.sleep(self.refresh_interval)

    def _render(self):
        """Render dashboard to terminal."""
        # Clear screen (ANSI escape code)
        print("\033[2J\033[H", end="")

        # Header
        print("="*80)
        print(" "*20 + "ALIGNMENT FRAMEWORK LIVE MONITOR")
        print("="*80)

        now = datetime.now()
        uptime = now - self.monitor.session_start

        print(f"  Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Session Start: {self.monitor.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Uptime: {self._format_duration(uptime)}")
        print()

        # Component metrics table
        print("  COMPONENT LATENCIES (milliseconds)")
        print("  " + "-"*76)
        print(f"  {'Component':<18} {'Count':<10} {'P50':<10} {'P95':<10} {'P99':<10} {'Status':<12}")
        print("  " + "-"*76)

        all_stats = self.monitor.get_all_stats()

        if not all_stats:
            print("  " + " "*20 + "No data yet - waiting for queries...")
        else:
            for component in sorted(all_stats.keys()):
                stats = all_stats[component]

                if stats["count"] == 0:
                    continue

                # Determine status
                status = self._get_status(component, stats)
                status_symbol = self._get_status_symbol(status)

                print(
                    f"  {component:<18} {stats['count']:<10} "
                    f"{stats['p50']:<10.3f} {stats['p95']:<10.3f} "
                    f"{stats['p99']:<10.3f} {status_symbol} {status:<10}"
                )

        print("  " + "-"*76)

        # Alerts section
        print()
        print("  RECENT ALERTS")
        print("  " + "-"*76)

        recent_alerts = self.monitor.alerts[-5:]  # Last 5 alerts

        if not recent_alerts:
            print("  " + " "*25 + "✅ No alerts - all systems nominal")
        else:
            for alert in recent_alerts:
                symbol = "🔴" if alert.level == AlertLevel.CRITICAL else "⚠️"
                time_ago = self._format_duration(now - alert.timestamp)

                print(f"  {symbol} [{alert.level.value.upper()}] {alert.component}")
                print(f"     {alert.message}")
                print(f"     {time_ago} ago ({alert.timestamp.strftime('%H:%M:%S')})")
                print()

        print("  " + "-"*76)

        # Summary statistics
        print()
        print("  SUMMARY")
        print("  " + "-"*76)

        total_samples = sum(stats["count"] for stats in all_stats.values())
        total_alerts = len(self.monitor.alerts)
        critical_alerts = len([a for a in self.monitor.alerts if a.level == AlertLevel.CRITICAL])
        warning_alerts = len([a for a in self.monitor.alerts if a.level == AlertLevel.WARNING])

        print(f"  Total Queries Processed: {total_samples}")
        print(f"  Total Alerts: {total_alerts} (🔴 {critical_alerts} critical, ⚠️  {warning_alerts} warnings)")

        if total_samples > 0:
            # Calculate overall P99
            pipeline_stats = all_stats.get("pipeline", None)
            if pipeline_stats and pipeline_stats["count"] > 0:
                overall_p99 = pipeline_stats["p99"]
                print(f"  Overall P99 Latency: {overall_p99:.3f} ms")

        print("  " + "-"*76)

        # Footer
        print()
        print("  Press Ctrl+C to stop monitoring")
        print("="*80)

        sys.stdout.flush()

    def _get_status(self, component: str, stats: dict) -> str:
        """Determine component status."""
        if component not in self.monitor.thresholds:
            return "OK"

        p99 = stats["p99"]
        threshold = self.monitor.thresholds[component]
        warning_threshold = threshold * self.monitor.WARNING_MULTIPLIER

        if p99 > threshold:
            return "CRITICAL"
        elif p99 > warning_threshold:
            return "WARNING"
        else:
            return "OK"

    def _get_status_symbol(self, status: str) -> str:
        """Get emoji for status."""
        symbols = {
            "OK": "✅",
            "WARNING": "⚠️",
            "CRITICAL": "🔴",
        }
        return symbols.get(status, "❓")

    def _format_duration(self, td: timedelta) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = int(td.total_seconds())

        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")

        return " ".join(parts)

    def render_once(self):
        """Render dashboard once (no loop)."""
        self._render()


# Standalone dashboard runner
def main():
    """Run live dashboard with demo data."""
    import asyncio
    from pathlib import Path

    # Import production system
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from demos.demo_alignment_integration import AlignedHoloLoomSystem
        print("✅ Production system imported")
    except ImportError:
        print("❌ Could not import production system")
        print("   Run from repository root: python HoloLoom/alignment/live_monitor.py")
        return

    # Create monitor
    from hololoom.alignment.monitoring import AlignmentMonitor

    monitor = AlignmentMonitor(
        persist_path=Path("./monitor_metrics.json")
    )

    # Create dashboard
    dashboard = LiveDashboard(monitor)

    print("="*80)
    print("ALIGNMENT FRAMEWORK LIVE MONITORING DEMO")
    print("="*80)
    print()
    print("This demo will:")
    print("  1. Create a production alignment system")
    print("  2. Process example queries")
    print("  3. Display live metrics in real-time")
    print()
    print("Press Ctrl+C to stop monitoring")
    print()
    input("Press Enter to start...")

    # Create production system

    print("\n🔧 Initializing alignment framework...")

    system = AlignedHoloLoomSystem(log_dir=Path("./demo_monitor_logs"))

    # Start dashboard
    dashboard.start()

    # Process queries with monitoring
    async def process_with_monitoring():
        queries = [
            "What is Thompson Sampling?",
            "Explain reinforcement learning",
            "How does PPO work?",
            "What is RLHF?",
            "Describe multi-armed bandits",
            "What are the benefits of curiosity-driven learning?",
            "Explain actor-critic methods",
            "What is policy gradient?",
        ] * 10  # 80 queries total

        for i, query_text in enumerate(queries):
            # Track full pipeline
            with monitor.track("pipeline"):
                # Track individual components
                with monitor.track("guardrails"):
                    from hololoom.alignment.safety_guardrails import ActionCategory, ActionRequest
                    request = ActionRequest(action_id=query_text, category=ActionCategory.QUERY)
                    decision = system.guardrails.evaluate(request, text_input=query_text)

                with monitor.track("detector"):
                    from hololoom.alignment.deception_detection import BehavioralProbe, ProbeType
                    probe = BehavioralProbe(ProbeType.GOAL_ALIGNMENT, "Test", "Expected")
                    system.detector.run_probe(probe, "Response")

                with monitor.track("guard"):
                    from hololoom.alignment.instrumental_convergence import ResourceType
                    system.guard.check_resource_usage(ResourceType.MEMORY, 500.0)

                with monitor.track("audit"):
                    from hololoom.alignment.audit_trail import DecisionType, OutcomeType
                    log = system.audit.log_decision(
                        DecisionType.SAFETY_GATE,
                        OutcomeType.APPROVED,
                        "Test decision",
                        query_text=query_text
                    )

            # Small delay to see live updates
            await asyncio.sleep(0.1)

            # Periodically flush
            if (i + 1) % 10 == 0:
                with monitor.track("audit"):
                    system.audit.persist()

    try:
        asyncio.run(process_with_monitoring())

        # Keep dashboard running
        print("\n✅ All queries processed. Monitoring continues...")
        print("   Press Ctrl+C to stop")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Stopping monitoring...")
        dashboard.stop()

        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(monitor.get_summary())

        # Persist metrics
        monitor.persist_metrics()

        print("\n✅ Monitoring session complete")


if __name__ == "__main__":
    main()
