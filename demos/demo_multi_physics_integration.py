"""
Multi-Physics Integration Demo

Shows Phase 1 (Gradient Flow) + Phase 2 (Fluid Dynamics) working together!

Architecture:
    1. Gradient Flow allocates token budget across components (cache, graph, embeddings)
    2. Fluid Dynamics packs each component's context optimally

Run:
    python demos/demo_multi_physics_integration.py
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.physics import (
    MultiPhysicsPacker,
    pack_context_multiphysics
)


async def demo_basic_multiphysics():
    """Demo 1: Basic multi-physics packing."""
    print("=" * 60)
    print("Demo 1: Multi-Physics Context Packing")
    print("=" * 60)
    print()
    print("Phase 1 (Gradient Flow): Allocate budget across components")
    print("Phase 2 (Fluid Dynamics): Pack each component optimally")
    print()

    # Create packer
    packer = MultiPhysicsPacker(max_tokens=8000)

    # Define components with importance scores
    components = {
        "cache": {
            "importance": 0.9,  # Very important!
            "initial_nodes": ["ThompsonSampling", "Bayesian"]
        },
        "graph": {
            "importance": 0.7,  # Important
            "initial_nodes": ["MultiArmedBandit", "Regret"]
        },
        "embeddings": {
            "importance": 0.6,  # Less important
            "initial_nodes": ["Embedding384D"]
        }
    }

    print("Components and importance:")
    for name, config in components.items():
        print(f"  {name:12s}: importance={config['importance']:.1f}")

    print("\nRunning multi-physics packing...")

    # Pack!
    result = await packer.pack(components)

    # Print results
    print("\n" + packer.get_summary(result))

    print("\nKey Insight:")
    print("  - Gradient flow allocated MORE tokens to cache (90% importance)")
    print("  - Fluid dynamics packed each component's context optimally")
    print("  - Physics handles both allocation AND packing!")


async def demo_constrained_allocation():
    """Demo 2: Constrained budget allocation."""
    print("\n\n" + "=" * 60)
    print("Demo 2: Constrained Budget Allocation")
    print("=" * 60)
    print()

    packer = MultiPhysicsPacker(max_tokens=10000)

    components = {
        "critical": {"importance": 1.0},
        "normal": {"importance": 0.5},
        "optional": {"importance": 0.2}
    }

    # Add constraints (min, max tokens)
    constraints = {
        "critical": (3000, 6000),  # At least 3000, at most 6000
        "normal": (1000, 3000),
        "optional": (0, 1000)
    }

    print("Constraints:")
    for name, (min_tok, max_tok) in constraints.items():
        print(f"  {name:12s}: {min_tok}-{max_tok} tokens")

    result = await packer.pack(components, constraints=constraints)

    print("\n" + packer.get_summary(result))

    print("\nBenefit:")
    print("  - Constraints ensure critical components get minimum budget")
    print("  - Gradient flow respects limits while optimizing")


async def demo_comparison_manual_vs_physics():
    """Demo 3: Compare manual allocation vs physics."""
    print("\n\n" + "=" * 60)
    print("Demo 3: Manual vs Physics-Based Allocation")
    print("=" * 60)
    print()

    print("Manual Allocation (equal split):")
    manual = {
        "cache": 2666,
        "graph": 2666,
        "embeddings": 2666
    }
    for name, tokens in manual.items():
        print(f"  {name:12s}: {tokens} tokens (33.3%)")

    print("\nProblem: Doesn't account for importance!")
    print("  - Cache is critical (importance=0.9)")
    print("  - Embeddings less so (importance=0.6)")
    print("  - Equal split wastes tokens on low-importance components")

    print("\nPhysics-Based Allocation (gradient flow):")

    packer = MultiPhysicsPacker(max_tokens=8000)

    components = {
        "cache": {"importance": 0.9},
        "graph": {"importance": 0.7},
        "embeddings": {"importance": 0.6}
    }

    result = await packer.pack(components)

    for alloc in result.allocations:
        pct = alloc.tokens / result.total_tokens_available
        print(f"  {alloc.component:12s}: {alloc.tokens} tokens ({pct:.1%})")

    print("\nBenefit:")
    print("  - Automatically allocates more to cache (highest importance)")
    print("  - Natural distribution based on importance scores")
    print("  - No manual if/else logic!")


async def demo_real_world_scenario():
    """Demo 4: Real-world scenario with HoloLoom."""
    print("\n\n" + "=" * 60)
    print("Demo 4: Real-World HoloLoom Scenario")
    print("=" * 60)
    print()

    print("Scenario: Packing context for complex query")
    print("Query: 'Explain Thompson Sampling with Bayesian inference'")
    print()

    # Simulate HoloLoom components
    components = {
        "recent_context": {
            "importance": 0.95,  # Very fresh, very important!
            "initial_nodes": ["ThompsonSampling", "BayesianInference"]
        },
        "knowledge_graph": {
            "importance": 0.8,  # Core knowledge
            "initial_nodes": ["MultiArmedBandit", "PriorDistribution"]
        },
        "related_memories": {
            "importance": 0.6,  # Helpful but not critical
            "initial_nodes": ["Exploration", "Exploitation"]
        },
        "embeddings": {
            "importance": 0.5,  # Background context
            "initial_nodes": ["Embedding384D"]
        }
    }

    packer = MultiPhysicsPacker(max_tokens=12000)

    result = await packer.pack(components)

    print(packer.get_summary(result))

    print("\nPhysics Pipeline:")
    print("  1. Gradient flow: Allocate 12000 tokens across 4 components")
    print("  2. Fluid dynamics: Pack each component's context")
    print("  3. Result: Optimally packed context window!")

    print("\nNo manual tuning - physics handles everything!")


if __name__ == "__main__":
    print("\n=== Multi-Physics Integration Demo ===\n")
    print("Combining Phase 1 (Gradient Flow) + Phase 2 (Fluid Dynamics)")
    print()

    # Run demos
    asyncio.run(demo_basic_multiphysics())
    asyncio.run(demo_constrained_allocation())
    asyncio.run(demo_comparison_manual_vs_physics())
    asyncio.run(demo_real_world_scenario())

    print("\n\n" + "=" * 60)
    print("[DONE] All demos complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Gradient flow allocates budget optimally")
    print("  2. Fluid dynamics packs each component")
    print("  3. Both phases work together seamlessly")
    print("  4. Zero manual tuning needed!")
    print("\nNext: Integrate into WeavingOrchestrator!")
