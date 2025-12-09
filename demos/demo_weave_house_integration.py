"""
Demo: Multi-Loom Architecture / WeaveHouse Integration
=======================================================

This demo showcases Phase 7 of the Multi-Loom Architecture - integrating
WeaveHouse with the main HoloLoom API.

Features demonstrated:
1. Creating WeaveHouse system via Config.multi_perspective()
2. Using WeaveHouse through HoloLoom API
3. Seeing multi-perspective responses from 5 looms
4. Observing consensus and tension handling
5. Background dreaming consolidation

Date: December 2025
"""

import asyncio
from typing import Dict, Any

# ============================================================================
# Setup
# ============================================================================

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")


def print_result(result: Any, indent: int = 0):
    """Pretty-print a result."""
    prefix = "  " * indent
    if hasattr(result, '__dict__'):
        for key, value in result.__dict__.items():
            if not key.startswith('_'):
                print(f"{prefix}{key}: {value}")
    else:
        print(f"{prefix}{result}")


# ============================================================================
# Demo 1: Basic Multi-Perspective Query
# ============================================================================

async def demo_basic_weave():
    """Demonstrate basic multi-perspective querying."""
    print_section("Demo 1: Basic Multi-Perspective Query")

    from HoloLoom import HoloLoom
    from HoloLoom.config import Config

    # Create HoloLoom with multi-perspective mode
    config = Config.multi_perspective()
    print(f"Config: use_weave_house={config.use_weave_house}")
    print(f"Config: enable_dreaming={config.enable_dreaming}")
    print(f"Config: exploration_depth={config.weave_house_exploration_depth}")

    async with HoloLoom(config=config) as loom:
        print(f"\nHoloLoom initialized:")
        print(f"  Multi-perspective: {loom.is_multi_perspective}")
        print(f"  WeaveHouse: {loom.weave_house is not None}")
        print(f"  DreamOrchestrator: {loom.dream_orchestrator is not None}")

        # Query using multi-perspective weaving
        query = "What is Thompson Sampling?"
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        result = await loom.weave(query)

        print(f"\nResult:")
        print(f"  Response: {result.response[:200]}..." if len(result.response) > 200 else f"  Response: {result.response}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Epistemic Confidence: {result.epistemic_confidence:.2f}")
        print(f"  Perspectives: {len(result.fabrics)}")
        print(f"  Tensions: {len(result.tensions)}")

        # Show individual perspectives
        print(f"\nIndividual Perspectives:")
        for fabric in result.fabrics:
            response_preview = fabric.response[:80] + "..." if len(fabric.response) > 80 else fabric.response
            print(f"  {fabric.perspective}: {response_preview}")
            print(f"    Confidence: {fabric.confidence:.2f}, Epistemic: {fabric.epistemic_confidence:.2f}")

    print("\n[OK] Demo 1 complete")


# ============================================================================
# Demo 2: View All Perspectives
# ============================================================================

async def demo_get_perspectives():
    """Demonstrate getting perspective descriptions."""
    print_section("Demo 2: View All Perspectives")

    from HoloLoom import HoloLoom
    from HoloLoom.config import Config

    async with HoloLoom(config=Config.multi_perspective()) as loom:
        perspectives = await loom.get_perspectives()

        print("Available Loom Perspectives:")
        print("-" * 40)

        for name, description in perspectives.items():
            print(f"  {name}:")
            print(f"    {description}")
            print()

    print("[OK] Demo 2 complete")


# ============================================================================
# Demo 3: Tension Exploration
# ============================================================================

async def demo_tension_exploration():
    """Demonstrate auto-exploration of disagreement zones."""
    print_section("Demo 3: Tension Exploration")

    from HoloLoom import HoloLoom
    from HoloLoom.config import Config

    # Use lower tension threshold to trigger exploration
    config = Config.multi_perspective()
    config.weave_house_tension_threshold = 0.2  # More sensitive
    config.weave_house_exploration_depth = 3    # Deeper exploration

    async with HoloLoom(config=config) as loom:
        # Controversial query that may cause disagreement
        query = "Is it better to explore or exploit in reinforcement learning?"
        print(f"Query: '{query}'")
        print(f"  (Controversial query - may cause loom disagreement)")
        print(f"  Tension threshold: {config.weave_house_tension_threshold}")
        print(f"  Exploration depth: {config.weave_house_exploration_depth}")
        print("-" * 40)

        result = await loom.weave(query)

        print(f"\nResult:")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Tensions detected: {len(result.tensions)}")

        if result.tensions:
            print(f"\nTensions (disagreements between looms):")
            for i, tension in enumerate(result.tensions, 1):
                print(f"  {i}. {tension}")

        if result.exploration_trace:
            print(f"\nExploration trace ({len(result.exploration_trace)} steps):")
            for step in result.exploration_trace[:5]:  # Show first 5 steps
                print(f"  - {step}")

    print("\n[OK] Demo 3 complete")


# ============================================================================
# Demo 4: Background Dreaming
# ============================================================================

async def demo_dreaming():
    """Demonstrate background dream consolidation."""
    print_section("Demo 4: Background Dreaming")

    from HoloLoom import HoloLoom
    from HoloLoom.config import Config

    config = Config.multi_perspective()
    config.dream_consolidation_interval = 5.0  # Short interval for demo (5 seconds)

    async with HoloLoom(config=config) as loom:
        if loom.dream_orchestrator is None:
            print("[SKIP] Dreaming not enabled")
            return

        print(f"DreamOrchestrator active: {loom.dream_orchestrator._running}")

        # Manually trigger a dream cycle
        print("\nTriggering manual dream cycle...")
        stats = await loom.dream_orchestrator.dream_once()

        print(f"\nDream Cycle Stats:")
        print(f"  Cycle number: {stats.cycle_number}")
        print(f"  Total insights collected: {stats.total_insights_collected}")
        print(f"  Math shared: {stats.math_shared}")
        print(f"  Patterns shared: {stats.patterns_shared}")
        print(f"  Corrections shared: {stats.corrections_shared}")

        if hasattr(stats, 'discoveries_shared'):
            print(f"  Discoveries shared: {stats.discoveries_shared}")

        if hasattr(stats, 'duration_ms'):
            print(f"  Duration: {stats.duration_ms:.2f}ms")

    print("\n[OK] Demo 4 complete")


# ============================================================================
# Demo 5: Combined Memory + Multi-Perspective
# ============================================================================

async def demo_combined_api():
    """Demonstrate using both memory API and multi-perspective weaving."""
    print_section("Demo 5: Combined Memory + Multi-Perspective")

    from HoloLoom import HoloLoom
    from HoloLoom.config import Config

    async with HoloLoom(config=Config.multi_perspective()) as loom:
        # First, experience some knowledge
        print("Experiencing knowledge...")
        await loom.experience("Thompson Sampling is a Bayesian approach to exploration")
        await loom.experience("UCB uses upper confidence bounds for exploration")
        await loom.experience("Epsilon-greedy randomly explores with probability epsilon")
        print("  [3 memories stored]")

        # Recall using traditional API
        print("\nRecalling with traditional API...")
        memories = await loom.recall("exploration methods")
        print(f"  Found {len(memories)} related memories")

        # Query using multi-perspective API
        print("\nWeaving with multi-perspective API...")
        result = await loom.weave("Compare exploration strategies in reinforcement learning")

        print(f"\nMulti-perspective result:")
        print(f"  Response: {result.response[:200]}..." if len(result.response) > 200 else f"  Response: {result.response}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Perspectives used: {len(result.fabrics)}")

        # Show that both APIs work together
        print("\nSystem summary:")
        print(loom.summary())

    print("[OK] Demo 5 complete")


# ============================================================================
# Demo 6: Config Alternatives
# ============================================================================

async def demo_config_alternatives():
    """Demonstrate different ways to enable multi-perspective mode."""
    print_section("Demo 6: Config Alternatives")

    from HoloLoom import HoloLoom
    from HoloLoom.config import Config

    # Method 1: Using Config.multi_perspective()
    print("Method 1: Config.multi_perspective()")
    config1 = Config.multi_perspective()
    print(f"  use_weave_house: {config1.use_weave_house}")

    # Method 2: Using use_multi_perspective parameter
    print("\nMethod 2: use_multi_perspective=True")
    loom2 = HoloLoom(use_multi_perspective=True)
    print(f"  is_multi_perspective: {loom2.is_multi_perspective}")

    # Method 3: Setting config.use_weave_house directly
    print("\nMethod 3: config.use_weave_house = True")
    config3 = Config.fused()
    config3.use_weave_house = True
    loom3 = HoloLoom(config=config3)
    print(f"  is_multi_perspective: {loom3.is_multi_perspective}")

    print("\n[OK] Demo 6 complete")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print(" Multi-Loom Architecture / WeaveHouse Integration Demo")
    print(" Phase 7: HoloLoom API Integration")
    print("=" * 60)

    try:
        await demo_basic_weave()
    except Exception as e:
        print(f"[ERROR] Demo 1 failed: {e}")

    try:
        await demo_get_perspectives()
    except Exception as e:
        print(f"[ERROR] Demo 2 failed: {e}")

    try:
        await demo_tension_exploration()
    except Exception as e:
        print(f"[ERROR] Demo 3 failed: {e}")

    try:
        await demo_dreaming()
    except Exception as e:
        print(f"[ERROR] Demo 4 failed: {e}")

    try:
        await demo_combined_api()
    except Exception as e:
        print(f"[ERROR] Demo 5 failed: {e}")

    try:
        await demo_config_alternatives()
    except Exception as e:
        print(f"[ERROR] Demo 6 failed: {e}")

    print("\n" + "=" * 60)
    print(" All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
