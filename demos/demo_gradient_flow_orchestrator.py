"""
Gradient Flow Orchestrator Integration Demo

Demonstrates Phase 1 (Gradient Flow) integrated into WeavingOrchestrator
for physics-based tool selection.

Shows:
1. Gradient flow router running alongside neural policy
2. Blending gradient flow + neural predictions (70/30)
3. Tool selection based on cost/quality/latency metrics

Run:
    python demos/demo_gradient_flow_orchestrator.py
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.config import Config
from hololoom.documentation.types import Query, MemoryShard


def create_test_shards() -> list:
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard1",
            text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem",
            episode="test",
            metadata={"type": "definition"}
        ),
        MemoryShard(
            id="shard2",
            text="Gradient flow routes queries downhill through loss landscapes",
            episode="test",
            metadata={"type": "concept"}
        ),
        MemoryShard(
            id="shard3",
            text="Tool selection balances cost, quality, and latency",
            episode="test",
            metadata={"type": "tradeoff"}
        )
    ]


async def demo_gradient_flow_integration():
    """Demo 1: Gradient flow integration with neural policy."""
    print("=" * 60)
    print("Demo 1: Gradient Flow + Neural Policy Integration")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  - Neural Policy: Provides base tool probabilities")
    print("  - Gradient Flow: Routes to optimal tool via loss landscape")
    print("  - Blending: 70% neural + 30% gradient flow")
    print()

    # Create config
    config = Config.fast()
    shards = create_test_shards()

    # Create orchestrator (gradient flow router auto-initialized)
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Process query
        query = Query(text="What is Thompson Sampling?")

        print(f"Query: {query.text}")
        print()

        # Weave (will use gradient flow + neural blending)
        spacetime = await orchestrator.weave(query)

        print()
        print("Results:")
        print(f"  Tool selected: {spacetime.tool_used}")
        print(f"  Confidence: {spacetime.confidence:.2f}")
        print(f"  Duration: {spacetime.trace.duration_ms:.1f}ms")
        print()

        # Show trace details
        print("Trace:")
        print(f"  Motifs: {', '.join(spacetime.trace.motifs_detected)}")
        print(f"  Embedding scales: {spacetime.trace.embedding_scales_used}")
        print(f"  Context shards: {spacetime.trace.context_shards_count}")
        print()

        print("Stage Timings:")
        for stage, duration in spacetime.trace.stage_durations.items():
            bar = "#" * int(duration / 5)
            print(f"  {stage:20s}: [{bar:<40}] {duration:>6.1f}ms")


async def demo_gradient_flow_vs_neural():
    """Demo 2: Compare gradient flow vs pure neural."""
    print()
    print()
    print("=" * 60)
    print("Demo 2: Gradient Flow vs Pure Neural")
    print("=" * 60)
    print()

    config = Config.fast()
    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Test multiple queries
        queries = [
            "Quick calculation: 2 + 2",           # Should prefer calc (low cost, high quality)
            "Search for recent papers",          # Should prefer search
            "Write to my Notion database",       # Should prefer notion_write
            "Explain gradient flow routing"       # Should prefer answer (balanced)
        ]

        for query_text in queries:
            query = Query(text=query_text)
            spacetime = await orchestrator.weave(query)

            print(f"Query: {query_text}")
            print(f"  -> Tool: {spacetime.tool_used}")
            print(f"  -> Confidence: {spacetime.confidence:.2f}")
            print()


async def demo_gradient_flow_adaptation():
    """Demo 3: Gradient flow adapts to tool metrics."""
    print()
    print("=" * 60)
    print("Demo 3: Gradient Flow Adaptation")
    print("=" * 60)
    print()
    print("Gradient flow automatically routes to optimal tools based on:")
    print("  - Cost (30% weight)")
    print("  - Quality (50% weight)")
    print("  - Latency (20% weight)")
    print()

    config = Config.fast()
    shards = create_test_shards()

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Access gradient router
        if orchestrator.gradient_router:
            print("Tool configurations:")
            for name, tool in orchestrator.gradient_router.tools.items():
                print(f"  {name:15s}: cost={tool.cost:.1f}, quality={tool.quality:.1%}, latency={tool.latency}ms")

            print()
            print("Loss function: L = 0.3*cost + 0.5*(1-quality) + 0.2*latency")
            print()

            # Show loss for each tool
            print("Tool losses (lower is better):")
            for name, tool in orchestrator.gradient_router.tools.items():
                metrics = {"cost": tool.cost, "quality": tool.quality, "latency": tool.latency / 1000.0}
                loss = orchestrator.gradient_router.engine.compute_loss(metrics)
                print(f"  {name:15s}: loss={loss:.3f}")

            print()
            print("Gradient flow routes to tool with lowest loss!")
        else:
            print("Gradient router not initialized")


if __name__ == "__main__":
    print()
    print("=== Gradient Flow Orchestrator Integration Demo ===")
    print()
    print("Phase 1 (Gradient Flow) integrated into production orchestrator!")
    print()

    # Run demos
    asyncio.run(demo_gradient_flow_integration())
    asyncio.run(demo_gradient_flow_vs_neural())
    asyncio.run(demo_gradient_flow_adaptation())

    print()
    print("=" * 60)
    print("[DONE] Gradient flow integration complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("  1. Gradient flow routes queries through loss landscapes")
    print("  2. Blends with neural policy (70% neural + 30% gradient)")
    print("  3. Automatic load balancing based on cost/quality/latency")
    print("  4. Zero manual tuning - physics handles routing!")
    print()
    print("Next: Phase 3 (Thermodynamics) for exploration/exploitation!")
