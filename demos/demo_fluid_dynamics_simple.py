"""
Simple Fluid Dynamics Demo

Demonstrates adaptive context packing via Navier-Stokes fluid dynamics.

Shows:
    1. Context flowing from high-pressure to low-pressure regions
    2. Automatic sparse region detection
    3. Reverse prompting to fill gaps
    4. Optimal context window packing

Run:
    python demos/demo_fluid_dynamics_simple.py
"""

import sys
import asyncio
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.physics import (
    AdaptivePacker,
    ContextFlowEngine,
    PressureField,
    VelocityField
)


def demo_basic_flow():
    """Demo 1: Basic pressure flow without graph."""
    print("=" * 60)
    print("Demo 1: Basic Pressure Flow")
    print("=" * 60)

    # Create flow engine
    engine = ContextFlowEngine(viscosity=0.01)

    # Build simple graph
    edges = [
        ("ThompsonSampling", "Bayesian"),
        ("ThompsonSampling", "MultiArmedBandit"),
        ("Bayesian", "PriorDistribution"),
        ("Bayesian", "Posterior"),
        ("MultiArmedBandit", "Regret"),
        ("MultiArmedBandit", "Exploration"),
    ]

    for src, tgt in edges:
        engine.add_edge(src, tgt)

    # Inject high-pressure context (user query)
    print("\n1. Injecting context at ThompsonSampling (high pressure)")
    engine.inject_context(
        node="ThompsonSampling",
        importance=0.95,
        tokens=50,
        text="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem."
    )

    print(f"   Initial state: {engine}")
    print(f"   Pressure at ThompsonSampling: {engine.pressure.get_pressure('ThompsonSampling'):.2f}")
    print(f"   Pressure at Bayesian: {engine.pressure.get_pressure('Bayesian'):.2f}")

    # Propagate
    print("\n2. Propagating via Navier-Stokes (10 timesteps)")
    for i in range(10):
        engine.step(dt=0.01)

    print(f"   After propagation: {engine}")
    print(f"   Pressure at ThompsonSampling: {engine.pressure.get_pressure('ThompsonSampling'):.2f}")
    print(f"   Pressure at Bayesian: {engine.pressure.get_pressure('Bayesian'):.2f}")
    print(f"   Pressure at MultiArmedBandit: {engine.pressure.get_pressure('MultiArmedBandit'):.2f}")

    # Detect sparse regions
    print("\n3. Detecting sparse regions")
    sparse = engine.detect_sparse_regions()
    print(f"   Sparse nodes: {sparse}")

    # Extract context
    print("\n4. Extracting packed context (max 500 tokens)")
    packed = engine.extract_context(max_tokens=500)
    print(f"   Packed {len(packed)} nodes:")
    for node, importance, text in packed[:5]:
        print(f"     - {node:20s} (importance={importance:.2f})")


def demo_adaptive_packer():
    """Demo 2: Adaptive packer with reverse prompting."""
    print("\n\n" + "=" * 60)
    print("Demo 2: Adaptive Packer (Synchronous)")
    print("=" * 60)

    # Create packer
    packer = AdaptivePacker(
        max_tokens=8000,
        viscosity=0.01,
        sparse_threshold=0.3
    )

    # Build knowledge graph
    edges = [
        ("ThompsonSampling", "Bayesian"),
        ("ThompsonSampling", "MultiArmedBandit"),
        ("Bayesian", "PriorDistribution"),
        ("Bayesian", "Posterior"),
        ("MultiArmedBandit", "Regret"),
        ("MultiArmedBandit", "Exploration"),
        ("Exploration", "Exploitation"),
        ("Regret", "CumulativeRegret"),
    ]

    for src, tgt in edges:
        packer.add_edge(src, tgt)

    # Inject query
    print("\n1. Injecting query context")
    packer.inject(
        node="ThompsonSampling",
        importance=0.95,
        tokens=100,
        text="What is Thompson Sampling and how does it work?"
    )

    # Pack synchronously
    print("\n2. Packing context (10 iterations)")
    result = packer.pack_sync(max_iterations=10, dt=0.01)

    print(f"\n3. Results:")
    print(f"   Tokens used: {result.tokens_used} / {result.tokens_available}")
    print(f"   Nodes packed: {len(result.nodes)}")
    print(f"   Flow iterations: {len(result.flow_states)}")

    print(f"\n   Top nodes by pressure:")
    for node, importance, text in result.nodes[:8]:
        print(f"     - {node:20s} (pressure={importance:.2f})")

    # Visualize pressure field
    print("\n4. Pressure Field Visualization:")
    print(packer.visualize_pressure_field())

    # Flow summary
    print("\n5. Flow Summary:")
    summary = packer.get_flow_summary()
    for key, value in summary.items():
        print(f"   {key:25s}: {value}")


async def demo_reverse_prompting():
    """Demo 3: Reverse prompting for sparse regions."""
    print("\n\n" + "=" * 60)
    print("Demo 3: Adaptive Packer with Reverse Prompting")
    print("=" * 60)

    # Mock reverse prompt callback
    async def mock_reverse_prompt(prompt: str):
        """Simulate executing a reverse prompt."""
        print(f"   [Reverse Prompt] {prompt}")
        # Return mock result
        return (80, f"Mock answer to: {prompt}")

    # Create packer with callback
    packer = AdaptivePacker(
        max_tokens=8000,
        viscosity=0.01,
        sparse_threshold=0.4,  # Higher threshold = more reverse prompts
        reverse_prompt_callback=mock_reverse_prompt
    )

    # Build graph with intentional sparse regions
    edges = [
        ("ThompsonSampling", "Bayesian"),
        ("ThompsonSampling", "MultiArmedBandit"),
        ("Bayesian", "PriorDistribution"),  # Will be sparse!
        ("MultiArmedBandit", "Regret"),     # Will be sparse!
    ]

    for src, tgt in edges:
        packer.add_edge(src, tgt)

    # Inject query
    print("\n1. Injecting query")
    packer.inject(
        node="ThompsonSampling",
        importance=0.95,
        tokens=100,
        text="Explain Thompson Sampling"
    )

    # Pack with reverse prompting
    print("\n2. Packing with reverse prompting enabled")
    result = await packer.pack(
        max_iterations=5,
        enable_reverse_prompting=True,
        reverse_prompt_budget=0.2  # 20% of tokens for filling gaps
    )

    print(f"\n3. Results:")
    print(f"   Tokens used: {result.tokens_used} / {result.tokens_available}")
    print(f"   Reverse prompts executed: {len(result.reverse_prompts_executed)}")
    print(f"\n   Reverse prompts:")
    for prompt in result.reverse_prompts_executed:
        print(f"     - {prompt}")

    print(f"\n   Final pressure distribution:")
    for node, pressure in sorted(
        result.final_pressure_map.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        print(f"     {node:25s}: {pressure:.2f}")


def demo_comparison():
    """Demo 4: Compare with vs without fluid dynamics."""
    print("\n\n" + "=" * 60)
    print("Demo 4: Comparison - Manual vs Fluid Dynamics")
    print("=" * 60)

    # Manual approach (simple top-k)
    print("\n1. Manual Approach (Top-K by importance)")
    manual_nodes = [
        ("ThompsonSampling", 0.95),
        ("Bayesian", 0.40),  # Underestimated!
        ("MultiArmedBandit", 0.80),
        ("PriorDistribution", 0.20),  # Underestimated!
        ("Regret", 0.70),
    ]

    manual_sorted = sorted(manual_nodes, key=lambda x: x[1], reverse=True)
    print("   Top nodes:")
    for node, imp in manual_sorted[:3]:
        print(f"     - {node:20s} (importance={imp:.2f})")

    print(f"\n   Problem: Bayesian and PriorDistribution ranked low!")
    print(f"   But they're critical for understanding Thompson Sampling!")

    # Fluid dynamics approach
    print("\n2. Fluid Dynamics Approach")
    packer = AdaptivePacker(max_tokens=8000, viscosity=0.01)

    edges = [
        ("ThompsonSampling", "Bayesian"),
        ("ThompsonSampling", "MultiArmedBandit"),
        ("Bayesian", "PriorDistribution"),
    ]

    for src, tgt in edges:
        packer.add_edge(src, tgt)

    packer.inject("ThompsonSampling", 0.95, 100)

    result = packer.pack_sync(max_iterations=10)

    print("   Top nodes (after fluid propagation):")
    for node, pressure, _ in result.nodes[:5]:
        print(f"     - {node:20s} (pressure={pressure:.2f})")

    print(f"\n   Benefit: Pressure naturally flows to connected nodes!")
    print(f"   Bayesian and PriorDistribution boosted automatically!")


if __name__ == "__main__":
    print("\n=== Fluid Dynamics Context Packing Demo ===\n")

    # Run demos
    demo_basic_flow()
    demo_adaptive_packer()
    demo_comparison()

    # Async demo
    print("\n\nRunning async demo...")
    asyncio.run(demo_reverse_prompting())

    print("\n\n" + "=" * 60)
    print("[DONE] All demos complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. High-pressure (important) context flows to low-pressure (sparse) regions")
    print("  2. Navier-Stokes naturally optimizes context packing")
    print("  3. Reverse prompting fills gaps automatically")
    print("  4. No manual tuning needed - physics does the work!")
    print("\nNext: Try with real HoloLoom knowledge graph!")
