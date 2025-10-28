#!/usr/bin/env python3
"""
Pattern Cards Demo
==================

Demonstrates the new Pattern Card system for modular configuration.

Pattern Cards are YAML-based configuration modules that declaratively specify:
- Which mathematical capabilities to enable
- Which memory backends to use
- Which tools are available
- Performance/accuracy tradeoffs

This demo shows:
1. Loading built-in cards (BARE/FAST/FUSED)
2. Card inheritance
3. Runtime overrides
4. Creating custom cards
5. Converting to SemanticCalculusConfig
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.loom.card_loader import PatternCard


def demo_basic_loading():
    """Demo 1: Basic card loading."""
    print("=" * 70)
    print("DEMO 1: Basic Card Loading")
    print("=" * 70)

    print("\nLoading three built-in cards:")
    print("-" * 70)

    for card_name in ["bare", "fast", "fused"]:
        card = PatternCard.load(card_name)

        print(f"\n{card.display_name}")
        print(f"  Description: {card.description}")
        print(f"  Version: {card.version}")

        # Show key config
        sem_calc = card.math_capabilities.semantic_calculus
        if sem_calc.get('enabled'):
            dims = sem_calc.get('config', {}).get('dimensions', 'N/A')
            print(f"  Semantic dimensions: {dims}")
        else:
            print(f"  Semantic calculus: DISABLED")

        scales = card.math_capabilities.spectral_embedding.get('scales', [])
        print(f"  Embedding scales: {scales}")

        print(f"  Memory backend: {card.memory_config.backend}")
        print(f"  Tools: {len(card.tools_config.enabled)} enabled")
        print(f"  Target latency: {card.performance_profile.target_latency_ms}ms")

    print("\n" + "=" * 70)


def demo_inheritance():
    """Demo 2: Card inheritance."""
    print("\nDEMO 2: Card Inheritance")
    print("=" * 70)

    print("\nFUSED card extends FAST card:")
    print("-" * 70)

    fast = PatternCard.load("fast")
    fused = PatternCard.load("fused")

    print(f"\nFAST card config:")
    print(f"  Semantic dimensions: {fast.math_capabilities.semantic_calculus['config']['dimensions']}")
    print(f"  Embedding scales: {fast.math_capabilities.spectral_embedding['scales']}")
    print(f"  Memory backend: {fast.memory_config.backend}")
    print(f"  Tools enabled: {len(fast.tools_config.enabled)}")

    print(f"\nFUSED card config (extends FAST):")
    print(f"  Semantic dimensions: {fused.math_capabilities.semantic_calculus['config']['dimensions']} (OVERRIDE)")
    print(f"  Embedding scales: {fused.math_capabilities.spectral_embedding['scales']} (OVERRIDE)")
    print(f"  Memory backend: {fused.memory_config.backend} (OVERRIDE)")
    print(f"  Tools enabled: {len(fused.tools_config.enabled)} (OVERRIDE)")

    print(f"\nInheritance chain:")
    print(f"  FUSED.extends = '{fused.extends}'")
    print(f"  FUSED inherits from FAST, then overrides specific settings")

    print("\n" + "=" * 70)


def demo_runtime_overrides():
    """Demo 3: Runtime overrides."""
    print("\nDEMO 3: Runtime Overrides")
    print("=" * 70)

    print("\nLoading FAST card with runtime overrides:")
    print("-" * 70)

    # Original fast card
    fast = PatternCard.load("fast")
    print(f"\nOriginal FAST config:")
    print(f"  Semantic dimensions: {fast.math_capabilities.semantic_calculus['config']['dimensions']}")
    print(f"  Cache size: {fast.math_capabilities.semantic_calculus['config']['cache']['size']}")
    print(f"  Target latency: {fast.performance_profile.target_latency_ms}ms")

    # Fast with overrides
    overrides = {
        'math': {
            'semantic_calculus': {
                'config': {
                    'dimensions': 24,  # Override to 24
                    'cache': {
                        'size': 15000  # Larger cache
                    }
                }
            }
        },
        'performance': {
            'target_latency_ms': 150  # Faster target
        }
    }

    fast_custom = PatternCard.load("fast", overrides=overrides)
    print(f"\nFAST with runtime overrides:")
    print(f"  Semantic dimensions: {fast_custom.math_capabilities.semantic_calculus['config']['dimensions']} (OVERRIDDEN)")
    print(f"  Cache size: {fast_custom.math_capabilities.semantic_calculus['config']['cache']['size']} (OVERRIDDEN)")
    print(f"  Target latency: {fast_custom.performance_profile.target_latency_ms}ms (OVERRIDDEN)")

    print("\n" + "=" * 70)


def demo_custom_card():
    """Demo 4: Creating a custom card."""
    print("\nDEMO 4: Creating Custom Cards")
    print("=" * 70)

    print("\nCreating a custom 'research' card by extending FUSED:")
    print("-" * 70)

    # Load fused as base
    research = PatternCard.load("fused")

    # Customize
    research.name = "research"
    research.display_name = "🔬 Deep Research"
    research.description = "Maximum detail for research queries"

    # Increase dimensions even more
    research.math_capabilities.semantic_calculus['config']['dimensions'] = 64
    research.math_capabilities.semantic_calculus['config']['cache']['size'] = 50000

    # Accept slower performance for quality
    research.performance_profile.target_latency_ms = 5000
    research.performance_profile.timeout_ms = 10000

    # Add custom extension
    research.extensions['citation_tracking'] = {
        'enabled': True,
        'format': 'academic'
    }

    print(f"\nCustom RESEARCH card:")
    print(f"  Display name: {research.display_name}")
    print(f"  Semantic dimensions: {research.math_capabilities.semantic_calculus['config']['dimensions']}")
    print(f"  Cache size: {research.math_capabilities.semantic_calculus['config']['cache']['size']}")
    print(f"  Target latency: {research.performance_profile.target_latency_ms}ms (accepts slower)")
    print(f"  Custom extensions: {list(research.extensions.keys())}")

    # Save custom card
    cards_dir = Path(__file__).parent.parent / "HoloLoom" / "cards"
    research.save("research", cards_dir=cards_dir)
    print(f"\n  ✓ Saved to: {cards_dir / 'research.yaml'}")

    print("\n" + "=" * 70)


def demo_semantic_config_conversion():
    """Demo 5: Converting to SemanticCalculusConfig."""
    print("\nDEMO 5: SemanticCalculusConfig Conversion")
    print("=" * 70)

    print("\nConverting cards to SemanticCalculusConfig:")
    print("-" * 70)

    for card_name in ["bare", "fast", "fused"]:
        card = PatternCard.load(card_name)
        sem_config = card.math_capabilities.to_semantic_config()

        print(f"\n{card.display_name} → SemanticCalculusConfig:")

        if sem_config is None:
            print(f"  Semantic calculus: DISABLED")
        else:
            print(f"  Dimensions: {sem_config.dimensions}")
            print(f"  Cache: {sem_config.cache_size} words")
            print(f"  Compute trajectory: {sem_config.compute_trajectory}")
            print(f"  Compute ethics: {sem_config.compute_ethics}")
            print(f"  Framework: {sem_config.ethical_framework}")

    print("\n" + "=" * 70)


def demo_tools_config():
    """Demo 6: Tools configuration."""
    print("\nDEMO 6: Tools Configuration")
    print("=" * 70)

    print("\nComparing tool availability across cards:")
    print("-" * 70)

    cards = {name: PatternCard.load(name) for name in ["bare", "fast", "fused"]}

    # Check specific tools
    test_tools = ["summarize", "analyze", "deep_research", "synthesis"]

    print(f"\n{'Tool':<15} | {'BARE':<6} | {'FAST':<6} | {'FUSED':<6}")
    print("-" * 50)

    for tool in test_tools:
        bare_status = "✓" if cards["bare"].tools_config.is_tool_enabled(tool) else "✗"
        fast_status = "✓" if cards["fast"].tools_config.is_tool_enabled(tool) else "✗"
        fused_status = "✓" if cards["fused"].tools_config.is_tool_enabled(tool) else "✗"

        print(f"{tool:<15} | {bare_status:<6} | {fast_status:<6} | {fused_status:<6}")

    print("\n" + "=" * 70)


def main():
    """Run all demos."""
    print("\n")
    print("=" * 70)
    print("PATTERN CARDS SYSTEM DEMO")
    print("=" * 70)
    print("\nModular configuration for HoloLoom!")
    print()

    demos = [
        demo_basic_loading,
        demo_inheritance,
        demo_runtime_overrides,
        demo_custom_card,
        demo_semantic_config_conversion,
        demo_tools_config,
    ]

    for demo_fn in demos:
        try:
            demo_fn()
        except Exception as e:
            print(f"\nDemo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Pattern Cards provide:")
    print("  ✓ Declarative configuration (YAML)")
    print("  ✓ Composability (inheritance)")
    print("  ✓ Runtime overrides")
    print("  ✓ Custom cards")
    print("  ✓ Version control friendly")
    print("  ✓ Shareable configuration recipes")
    print()
    print("Next steps:")
    print("  1. Create custom cards for your use cases")
    print("  2. Share cards as configuration recipes")
    print("  3. Version control your card configurations")
    print("  4. A/B test different cards")
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
