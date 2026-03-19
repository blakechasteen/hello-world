#!/usr/bin/env python3
"""
Simple Demo: Context Learning Mechanisms

Shows how to use the learning-enabled router with minimal setup.
"""

import sys
from pathlib import Path

# Add repository root to path (so we can run from any directory)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import asyncio

from hololoom.context import create_query_router
from hololoom.infrastructure.mcp import create_mcp_server, generate_session_id
from hololoom.infrastructure.sql import SQLConfig, load_mock_data


async def main():
    print("=" * 80)
    print("Context Department Learning Demo")
    print("=" * 80)
    print()

    # Step 1: Setup (one-time)
    print("[1/4] Setting up backend...")
    sql_config = SQLConfig(sqlite_path="./data/demo_learning.db")
    mcp_server = await create_mcp_server(sql_config)
    await load_mock_data(mcp_server.sql_backend)
    print("     Backend ready!")
    print()

    # Step 2: Create router with learning enabled
    print("[2/4] Creating router with learning enabled...")
    session_id = generate_session_id()
    router = await create_query_router(
        mcp_server,
        session_id,
        enable_learning=True,        # Track decisions
        enable_calibration=True,     # Adjust confidence
        enable_strategy_updates=True # Adapt weights
    )
    print("     Router ready!")
    print()

    # Step 3: Run some queries
    print("[3/4] Processing queries...")
    queries = [
        "What is the Varroa treatment schedule?",
        "How many hives are in the system?",
        "Show me audit logs for hive_042",
        "What are the inspection policies?",
        "List all treatment types"
    ]

    for i, query in enumerate(queries, 1):
        result = await router.route(query)
        print(f"  Query {i}: '{query[:40]}...'")
        print(f"    Pattern: {result.pattern.value}")
        print(f"    Backends: {', '.join(result.backends_used)}")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Rows: {result.row_count}")
        print(f"    Latency: {result.total_latency_ms:.1f}ms")
        print()

    # Step 4: View learning statistics
    print("[4/4] Learning Statistics:")
    print("-" * 80)

    # Overall stats
    overall = router.learning_tracker.get_overall_performance(window=100)
    print("\nOverall Performance:")
    print(f"  Total queries: {router.learning_tracker.total_events}")
    print(f"  Avg confidence: {overall['avg_confidence']:.2f}")
    print(f"  Avg latency: {overall['avg_latency_ms']:.1f}ms")
    print(f"  Success rate: {overall['success_rate']*100:.1f}%")
    print(f"  Cache hit rate: {overall['cache_hit_rate']*100:.1f}%")

    # Per-backend stats
    print("\nPer-Backend Performance:")
    backend_stats = router.learning_tracker.get_backend_comparison(window=100)
    for backend, metrics in backend_stats.items():
        if metrics.count > 0:
            print(f"  {backend.upper()}:")
            print(f"    Queries: {metrics.count}")
            print(f"    Avg confidence: {metrics.avg_confidence:.2f}")
            print(f"    Avg latency: {metrics.avg_latency_ms:.1f}ms")

    # Calibration stats
    if router.calibrator:
        print("\nCalibration:")
        print(f"  Observations: {router.calibrator.observation_count}")
        curve = router.calibrator.get_calibration_curve()
        if curve.calibrated:
            print(f"  ECE: {curve.ece:.3f} (calibrated!)")
        else:
            print(f"  Status: Collecting data... ({curve.sample_size}/{router.calibrator.min_observations})")

    # Strategy updates
    if router.strategy_updater:
        print("\nStrategy Updates:")
        print(f"  Updates applied: {router.strategy_updater.update_count}")
        print(f"  Rollbacks: {router.strategy_updater.rollback_count}")

    print()
    print("=" * 80)
    print("Demo complete! The system learned from every query.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
