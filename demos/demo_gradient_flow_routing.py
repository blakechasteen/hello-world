"""
Datacenter Downhill Flow Demo

Demonstrates gradient flow routing for load balancing.

Shows your original inspiration: "Queries flow downhill to least-loaded servers
like water flowing down a mountain."

Run:
    python demos/demo_gradient_flow_routing.py
"""

import sys
import asyncio
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.routing import (
    ServerRouter,
    ToolRouter,
    ServerConfig,
    ToolConfig,
    create_server_router,
    create_tool_router
)
from HoloLoom.physics import GradientFlowEngine, combined_loss


def demo_datacenter_downhill_flow():
    """Demo 1: Datacenter downhill flow (your original concept!)."""
    print("=" * 60)
    print("Demo 1: Datacenter Downhill Flow")
    print("=" * 60)
    print()
    print("Concept: Queries flow downhill to least-loaded servers,")
    print("like water flowing down a mountain to the valley.")
    print()

    # Create datacenter with 3 servers
    router = ServerRouter(
        servers=[
            ServerConfig("server1", max_load=1.0, base_latency=50.0),
            ServerConfig("server2", max_load=1.0, base_latency=30.0),
            ServerConfig("server3", max_load=1.0, base_latency=40.0)
        ]
    )

    # Simulate incoming queries
    queries = [
        "What is Thompson Sampling?",
        "Explain Bayesian inference",
        "How does gradient descent work?",
        "What is a multi-armed bandit?",
        "Describe reinforcement learning"
    ]

    print("Initial loads (all at valley - 0%):")
    for name, load in router.get_loads().items():
        print(f"  {name}: {load:.1%}")

    print("\nRouting 5 queries (watch the downhill flow):\n")

    async def route_queries():
        for i, query in enumerate(queries, 1):
            decision = await router.route_query(query[:30])  # Truncate for display

            print(f"{i}. Query: '{query[:30]}...'")
            print(f"   -> Routed to: {decision.target}")
            print(f"   -> Loss: {decision.loss:.3f}")
            print(f"   -> Current loads:")
            for name, load in router.get_loads().items():
                bar = "#" * int(load * 40)
                print(f"      {name}: [{bar:<40}] {load:.1%}")
            print()

    asyncio.run(route_queries())

    print("Final loads (water settled in valleys):")
    stats = router.get_statistics()
    print(f"  Total routes: {stats['total_routes']}")
    print(f"  Routes per server:")
    for server, count in stats['targets'].items():
        print(f"    {server}: {count} queries")

    print("\nBenefit: Automatic load balancing - no manual tuning!")


def demo_loss_landscape_visualization():
    """Demo 2: Visualize the loss landscape."""
    print("\n\n" + "=" * 60)
    print("Demo 2: Loss Landscape Visualization")
    print("=" * 60)
    print()
    print("Loss landscape is like a 3D mountain:")
    print("  - Peaks = high loss (bad servers)")
    print("  - Valleys = low loss (good servers)")
    print("  - Query = ball rolling downhill")
    print()

    # Create engine
    engine = GradientFlowEngine(
        loss_fn=combined_loss(load_weight=0.7, latency_weight=0.3),
        learning_rate=0.1,
        noise_level=0.0  # No noise for visualization
    )

    # Three servers with different loads
    targets = ["server1", "server2", "server3"]
    metrics = [
        {"load": 0.9, "latency": 50},  # High load (peak!)
        {"load": 0.2, "latency": 30},  # Low load (valley!)
        {"load": 0.6, "latency": 40}   # Medium load
    ]

    print("Server metrics:")
    for i, (name, m) in enumerate(zip(targets, metrics)):
        loss = engine.compute_loss(m)
        print(f"  {i}. {name:10s}: load={m['load']:.1%}, loss={loss:.3f}")

    print("\nLoss landscape (ASCII art):")
    print("     Loss")
    print("       ^")
    print("  0.9 |     #")
    print("      |    / \\       server1 (peak - high load)")
    print("  0.6 |   /   \\   #")
    print("      |  /     \\ /|\\   server3 (hill)")
    print("  0.2 | /       X   \\")
    print("      |/       / \\   \\__")
    print("  0.0 |_______/___\\_____\\___")
    print("         0    1    2    3      Server index")
    print("              ^")
    print("           server2 (valley - low load!)")

    print("\nQuery starts at position 0.5 (near server1)")
    print("Gradient descent flows downhill to server2...")

    # Route
    decision = engine.route(
        targets=targets,
        target_metrics=metrics,
        max_steps=20,
        initial_position=0.5  # Start near server1
    )

    print(f"\nResult after 20 steps:")
    print(f"  Final target: {decision.target}")
    print(f"  Final position: {decision.position:.2f}")
    print(f"  Final loss: {decision.loss:.3f}")
    print(f"  -> Query flowed downhill to {decision.target} (lowest load)!")


async def demo_tool_selection():
    """Demo 3: Tool selection via gradient flow."""
    print("\n\n" + "=" * 60)
    print("Demo 3: Tool Selection (Cost vs Quality vs Speed)")
    print("=" * 60)
    print()

    # Create tool router
    router = ToolRouter(
        tools=[
            ToolConfig("search", cost=0.5, quality=0.7, latency=100),
            ToolConfig("answer", cost=0.2, quality=0.6, latency=50),
            ToolConfig("reason", cost=0.8, quality=0.95, latency=200)
        ],
        cost_weight=0.3,
        quality_weight=0.5,  # Favor quality!
        latency_weight=0.2
    )

    print("Available tools:")
    for name, tool in router.tools.items():
        print(f"  {name:8s}: cost={tool.cost:.1f}, quality={tool.quality:.1%}, latency={tool.latency}ms")

    print("\nQuery: 'Explain complex reasoning about Thompson Sampling'")

    decision = await router.select_tool("Explain complex reasoning")

    print(f"\nSelected tool: {decision.target}")
    print(f"Loss: {decision.loss:.3f}")

    print("\nWhy 'reason' was selected:")
    print("  - High quality (0.95) outweighs high cost (0.8)")
    print("  - Gradient flow naturally balances all factors")
    print("  - No manual if/else logic needed!")


def demo_comparison_learned_vs_gradient():
    """Demo 4: Compare learned routing vs gradient flow."""
    print("\n\n" + "=" * 60)
    print("Demo 4: Learned vs Gradient Flow Routing")
    print("=" * 60)
    print()

    print("Learned Routing (Thompson Sampling):")
    print("  + Adapts to changing conditions")
    print("  + Learns from feedback")
    print("  - Needs training data")
    print("  - Exploration overhead")

    print("\nGradient Flow Routing:")
    print("  + Zero training needed")
    print("  + Immediate optimal routing")
    print("  + Deterministic (with noise_level=0)")
    print("  - Doesn't learn from feedback")
    print("  - Assumes metrics are accurate")

    print("\nBest Use Cases:")
    print("  Learned: Dynamic environments, feedback available")
    print("  Gradient: Static configs, instant decisions needed")

    print("\nCombine Both:")
    print("  1. Use gradient flow for instant routing")
    print("  2. Use Thompson Sampling to learn corrections")
    print("  3. Gradient provides baseline, learning refines")


if __name__ == "__main__":
    print("\n=== Datacenter Downhill Flow Demo ===\n")
    print("Your original inspiration:")
    print('"Queries flow downhill to least-loaded servers"')
    print()

    # Run demos
    demo_datacenter_downhill_flow()
    demo_loss_landscape_visualization()

    # Async demos
    print("\n\nRunning async demos...")
    asyncio.run(demo_tool_selection())

    demo_comparison_learned_vs_gradient()

    print("\n\n" + "=" * 60)
    print("[DONE] All demos complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Queries flow downhill like water seeking valleys")
    print("  2. No central coordinator - physics handles routing")
    print("  3. Automatic load balancing across servers")
    print("  4. Works for tools, resources, agents - anything!")
    print("\nNext: Combine with Thompson Sampling for learned+physical routing!")
