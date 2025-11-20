#!/usr/bin/env python3
"""Quick test of SmartWeavingOrchestrator integration."""

import asyncio
import logging
from smart_weaving_orchestrator import create_smart_orchestrator

logging.basicConfig(level=logging.WARNING)  # Less verbose

async def main():
    print("="*80)
    print("QUICK INTEGRATION TEST")
    print("="*80)
    print()

    # Create orchestrator
    print("Initializing SmartWeavingOrchestrator...")
    orchestrator = create_smart_orchestrator(
        pattern="fast",
        math_budget=50,
        math_style="detailed"
    )
    print("OK Initialized\n")

    # Test queries
    test_queries = [
        "Find documents similar to quantum computing",
        "Optimize the retrieval algorithm",
        "Analyze convergence of learning",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] {query}")
        print("-" * 80)

        try:
            spacetime = await orchestrator.weave(query)

            print(f"\nResponse:")
            print(spacetime.response[:300] + "..." if len(spacetime.response) > 300 else spacetime.response)

            print(f"\nMetadata:")
            print(f"  Confidence: {spacetime.confidence:.1%}")
            print(f"  Tool: {spacetime.tool_used}")
            print(f"  Duration: {spacetime.trace.duration_ms:.0f}ms")

            if hasattr(spacetime.trace, 'analytical_metrics') and spacetime.trace.analytical_metrics:
                math_metrics = spacetime.trace.analytical_metrics.get('math_meaning', {})
                if 'operations_executed' in math_metrics:
                    print(f"  Math ops: {math_metrics['operations_executed']}")
                    print(f"  Math cost: {math_metrics['total_cost']}")
                    print(f"  Math confidence: {math_metrics['confidence']:.1%}")

        except Exception as e:
            print(f"X Error: {e}")

        print("\n")

    # Show stats
    print("="*80)
    print("STATISTICS")
    print("="*80)
    stats = orchestrator.get_statistics()
    print(f"Total weavings: {stats['total_weavings']}")

    if 'math_pipeline' in stats:
        mp = stats['math_pipeline']
        print(f"\nMath Pipeline:")
        print(f"  Executions: {mp['total_executions']}")
        print(f"  Total cost: {mp['total_cost']}")
        print(f"  Avg confidence: {mp['avg_confidence']:.1%}")
        print(f"  Operations used: {list(mp['operations_used'].keys())}")

    print("\nOK Integration test complete!")

if __name__ == "__main__":
    asyncio.run(main())
