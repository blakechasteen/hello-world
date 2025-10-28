#!/usr/bin/env python3
"""
Semantic Calculus + Monitoring Dashboard Integration Demo
==========================================================

Demonstrates the semantic calculus MCP server with real-time monitoring.

Usage:
    python demos/semantic_calculus_monitoring_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.semantic_calculus.mcp_server import (
    initialize_semantic_calculus,
    calculus,
    spectrum,
    call_tool as mcp_call_tool
)
from HoloLoom.monitoring.dashboard import MonitoringDashboard, MetricsCollector
import time


async def run_semantic_analysis_with_monitoring():
    """Run semantic calculus analysis and track metrics."""

    print("=" * 70)
    print("SEMANTIC CALCULUS + MONITORING INTEGRATION DEMO")
    print("=" * 70)

    # Initialize semantic calculus
    print("\n[1/3] Initializing semantic calculus...")
    await initialize_semantic_calculus()
    print("    ✓ Semantic calculus initialized")

    # Create monitoring dashboard
    print("\n[2/3] Setting up monitoring dashboard...")
    collector = MetricsCollector()
    dashboard = MonitoringDashboard(collector)
    print("    ✓ Monitoring dashboard ready")

    # Test queries
    print("\n[3/3] Running test queries with monitoring...")
    test_queries = [
        {
            "text": "Machine learning transforms data into actionable insights through automated pattern recognition.",
            "description": "ML definition"
        },
        {
            "text": "Thompson Sampling balances exploration and exploitation using Bayesian inference.",
            "description": "Thompson Sampling"
        },
        {
            "text": "Neural networks learn hierarchical representations from raw data.",
            "description": "Neural networks"
        },
        {
            "text": "I appreciate your thoughtful perspective on this complex issue.",
            "description": "Ethical dialogue"
        }
    ]

    print()
    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}/{len(test_queries)}: {query['description']}")

        # Run analysis and time it
        start = time.time()
        try:
            result = await mcp_call_tool(
                "analyze_semantic_flow",
                {"text": query["text"], "format": "json"}
            )
            latency_ms = (time.time() - start) * 1000
            success = True

            # Record metrics
            collector.record_query(
                pattern="fast",
                latency_ms=latency_ms,
                success=True,
                tool="analyze_semantic_flow",
                backend="semantic_calculus",
                complexity_level="FAST"
            )

            print(f"    ✓ Analyzed in {latency_ms:.1f}ms")

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            collector.record_query(
                pattern="fast",
                latency_ms=latency_ms,
                success=False,
                tool="analyze_semantic_flow",
                backend="semantic_calculus",
                complexity_level="FAST"
            )
            print(f"    ✗ Failed: {e}")

    # Display dashboard
    print("\n" + "=" * 70)
    print("MONITORING DASHBOARD")
    print("=" * 70)
    dashboard.display()

    # Show cache stats
    print("\n" + "=" * 70)
    print("SEMANTIC CALCULUS CACHE STATISTICS")
    print("=" * 70)
    if hasattr(calculus, 'get_cache_stats'):
        cache_stats = calculus.get_cache_stats()
        print(f"  Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
        print(f"  Cache hits: {cache_stats['hits']}")
        print(f"  Cache misses: {cache_stats['misses']}")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1%}")

    print("\n" + "=" * 70)
    print("✓ Demo complete! Integration working successfully.")
    print("=" * 70)


async def main():
    """Run the demo."""
    try:
        await run_semantic_analysis_with_monitoring()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
