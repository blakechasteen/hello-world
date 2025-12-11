"""
Jenny Demo: M6 - Analytics Collection
=====================================

Demonstrates analytics with:
- Event tracking for renders and compiles
- Performance metrics collection
- Dashboard data generation

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot M6)
"""

import time
from typing import List, Dict, Any

from demos.jenny import register_demo, demo_phase

# Jenny imports
try:
    from HoloLoom.visualization.jenny_analytics import (
        JennyAnalytics,
        RenderEvent,
        CompileEvent,
    )
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False


def create_sample_events() -> List[Dict[str, Any]]:
    """Create sample analytics events for demonstration."""
    return [
        {
            "type": "render",
            "target": "html",
            "panel_count": 3,
            "duration_ms": 45.2,
            "success": True,
        },
        {
            "type": "render",
            "target": "react",
            "panel_count": 3,
            "duration_ms": 52.1,
            "success": True,
        },
        {
            "type": "compile",
            "template": "default_dashboard",
            "duration_ms": 12.5,
            "success": True,
        },
        {
            "type": "render",
            "target": "ar",
            "panel_count": 3,
            "duration_ms": 78.3,
            "success": True,
        },
        {
            "type": "render",
            "target": "html",
            "panel_count": 5,
            "duration_ms": 61.4,
            "success": True,
        },
    ]


@register_demo(
    phase_id="m6",
    name="Analytics Collection",
    order=6,
    requires=["m1"],  # Depends on registry
    tags=["core", "analytics", "metrics"],
    description="Event tracking and performance metrics"
)
@demo_phase("M6: Analytics Collection", emoji="[M6]")
async def demo_m6_analytics() -> dict:
    """
    M6: Analytics Collection - Track events and generate metrics.

    Returns:
        Demo results with analytics info
    """
    if not ANALYTICS_AVAILABLE:
        # Demonstrate with mock analytics
        return await _demo_mock_analytics()

    results = {
        "events_recorded": 0,
        "metrics": {},
    }

    analytics = JennyAnalytics()

    # Overview
    print("  Analytics Collection System:")
    print("  " + "-" * 50)
    print("    JennyAnalytics tracks two event types:")
    print("      - RenderEvent: Panel rendering performance")
    print("      - CompileEvent: Template compilation metrics")
    print()

    # Record some events
    events = create_sample_events()
    for event in events:
        if event["type"] == "render":
            analytics.record_render(
                target=event["target"],
                panel_count=event["panel_count"],
                duration_ms=event["duration_ms"],
                success=event["success"],
            )
        elif event["type"] == "compile":
            analytics.record_compile(
                template=event["template"],
                duration_ms=event["duration_ms"],
                success=event["success"],
            )
    results["events_recorded"] = len(events)

    print(f"  Recorded {len(events)} sample events")
    print()

    # Show metrics
    print("  Performance Metrics:")
    print("  " + "-" * 50)

    render_metrics = analytics.get_render_metrics()
    results["metrics"]["render"] = render_metrics

    print(f"    Total Renders: {render_metrics.get('total', 0)}")
    print(f"    Avg Duration: {render_metrics.get('avg_duration_ms', 0):.1f}ms")
    print(f"    Success Rate: {render_metrics.get('success_rate', 0):.1%}")
    print()

    # Breakdown by target
    print("  By Render Target:")
    print("  " + "-" * 50)
    by_target = render_metrics.get('by_target', {})
    for target, metrics in by_target.items():
        print(f"    {target:10s}: {metrics.get('count', 0)} renders, "
              f"{metrics.get('avg_duration_ms', 0):.1f}ms avg")
    print()

    # Dashboard data
    print("  Dashboard Integration:")
    print("  " + "-" * 50)
    print("    Analytics data feeds into Jenny dashboards:")
    print("      - Real-time render performance graphs")
    print("      - Target comparison charts")
    print("      - Error rate monitoring")
    print("      - Latency percentile tracking (P50/P90/P99)")

    return {
        "status": "success",
        "events_recorded": results["events_recorded"],
        "total_renders": render_metrics.get('total', 0),
        "avg_render_ms": render_metrics.get('avg_duration_ms', 0),
        "success_rate": render_metrics.get('success_rate', 0),
    }


async def _demo_mock_analytics() -> dict:
    """Fallback demo with mock analytics when module not available."""
    print("  Analytics Collection System (Mock):")
    print("  " + "-" * 50)
    print("    JennyAnalytics module not available.")
    print("    Demonstrating with simulated metrics...")
    print()

    # Simulate recording events
    events = create_sample_events()
    print(f"  Simulated {len(events)} events")
    print()

    # Calculate mock metrics
    render_events = [e for e in events if e["type"] == "render"]
    total_renders = len(render_events)
    avg_duration = sum(e["duration_ms"] for e in render_events) / total_renders
    success_rate = sum(1 for e in render_events if e["success"]) / total_renders

    print("  Performance Metrics (Simulated):")
    print("  " + "-" * 50)
    print(f"    Total Renders: {total_renders}")
    print(f"    Avg Duration: {avg_duration:.1f}ms")
    print(f"    Success Rate: {success_rate:.1%}")
    print()

    # Breakdown by target
    print("  By Render Target:")
    print("  " + "-" * 50)
    targets = {}
    for e in render_events:
        target = e["target"]
        if target not in targets:
            targets[target] = {"count": 0, "total_ms": 0}
        targets[target]["count"] += 1
        targets[target]["total_ms"] += e["duration_ms"]

    for target, metrics in targets.items():
        avg = metrics["total_ms"] / metrics["count"]
        print(f"    {target:10s}: {metrics['count']} renders, {avg:.1f}ms avg")
    print()

    # Dashboard info
    print("  Dashboard Integration:")
    print("  " + "-" * 50)
    print("    Full analytics available with HoloLoom.visualization module")

    return {
        "status": "success",
        "events_recorded": len(events),
        "total_renders": total_renders,
        "avg_render_ms": avg_duration,
        "success_rate": success_rate,
        "note": "mock_analytics_used",
    }


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_m6_analytics())
