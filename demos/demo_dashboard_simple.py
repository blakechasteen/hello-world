"""
Simple Live Dashboard Demo

Shows the monitoring dashboard with simulated query processing.

Usage:
    python demos/demo_dashboard_simple.py
"""

import asyncio
import time
import sys
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.alignment.monitoring import AlignmentMonitor
from HoloLoom.alignment.live_monitor import LiveDashboard


async def simulate_queries(monitor: AlignmentMonitor, count: int = 100):
    """Simulate query processing with realistic latencies."""
    print(f"\n🚀 Simulating {count} queries with realistic latencies...")
    print("   Watch the dashboard update in real-time!\n")

    for i in range(count):
        # Simulate guardrails (fast)
        latency = random.gauss(0.04, 0.01)
        monitor.record("guardrails", latency)

        # Simulate detector (fast)
        latency = random.gauss(0.035, 0.01)
        monitor.record("detector", latency)

        # Simulate guard (very fast)
        latency = random.gauss(0.001, 0.0005)
        monitor.record("guard", latency)

        # Simulate audit (occasional spikes)
        if random.random() < 0.05:
            latency = random.gauss(5.0, 2.0)
        else:
            latency = random.gauss(0.03, 0.01)
        monitor.record("audit", latency)

        # Calculate pipeline total
        pipeline_latency = random.gauss(0.2, 0.05)
        monitor.record("pipeline", pipeline_latency)

        await asyncio.sleep(0.05)

        if (i + 1) % 10 == 0:
            stats = monitor.get_stats("pipeline")
            print(f"   {i+1}/{count} - P99: {stats['p99']:.3f} ms")


async def main():
    print("="*80)
    print("LIVE MONITORING DASHBOARD DEMO")
    print("="*80)
    print("\nPress Enter to start...")
    input()

    monitor = AlignmentMonitor(
        thresholds={
            "guardrails": 1.0,
            "detector": 2.0,
            "guard": 0.5,
            "audit": 10.0,
            "pipeline": 20.0,
        }
    )

    dashboard = LiveDashboard(monitor, refresh_interval=1.0)
    dashboard.start()
    
    await asyncio.sleep(2)
    await simulate_queries(monitor, count=100)

    print("\n✅ Complete! Press Ctrl+C to stop.\n")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        dashboard.stop()
        print("\n" + monitor.get_summary())


if __name__ == "__main__":
    asyncio.run(main())
