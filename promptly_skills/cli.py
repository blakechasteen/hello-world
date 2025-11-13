"""
Promptly Strategy CLI - Simple and Elegant

A minimal command-line interface for testing prompting strategies.

Usage:
    python cli.py "explain neural networks"
    python cli.py "explain neural networks" --strategy deep
    python cli.py "solve this problem" --strategy scaffold
    python cli.py "explain transformers" --auto
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.prompting.registry import get_registry
from HoloLoom.prompting.auto_detect import AutoDetector
from HoloLoom.prompting.strategy import StrategyContext


def print_banner():
    """Print elegant banner."""
    print("\n" + "="*60)
    print("  Promptly Strategy Framework")
    print("  Simple. Elegant. Composable. Learning.")
    print("="*60 + "\n")


def print_strategy_list(registry):
    """Print available strategies."""
    print("Available Strategies:\n")

    categories = {}
    for name, strategy in registry.strategies.items():
        cat = strategy.category.value
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, strategy.description))

    for category, strategies in sorted(categories.items()):
        print(f"  {category.upper()}:")
        for name, desc in sorted(strategies):
            print(f"    • {name:12} - {desc}")
        print()


def print_result(result, strategy_name: str):
    """Print strategy result elegantly."""
    print("\n" + "-"*60)
    print(f"  Strategy: {strategy_name}")
    print(f"  Confidence: {result.confidence:.2f}")
    if result.estimated_improvement:
        print(f"  Estimated Improvement: +{result.estimated_improvement*100:.0f}%")
    print("-"*60 + "\n")

    print("Enhanced Query:")
    print("-"*60)
    # Truncate if too long
    if len(result.enhanced_query) > 2000:
        print(result.enhanced_query[:2000])
        print(f"\n... (truncated, {len(result.enhanced_query)} total chars)")
    else:
        print(result.enhanced_query)
    print("-"*60 + "\n")


async def run_strategy(query: str, strategy_name: Optional[str] = None, auto: bool = False):
    """Run a strategy on a query."""
    registry = get_registry()

    # Auto-detect if requested
    if auto:
        print("Auto-detecting best strategies...\n")
        detector = AutoDetector(registry=registry)
        suggestions = await detector.detect(StrategyContext(query=query), top_k=3)

        print("Suggestions:")
        for name, confidence in suggestions:
            print(f"  {name:12} - Confidence: {confidence:.2f}")
        print()

        if not suggestions:
            print("No strategies matched. Using default.\n")
            return

        # Use top suggestion
        strategy_name = suggestions[0][0]
        print(f"Using: {strategy_name}\n")

    # Get strategy
    if strategy_name:
        strategy = registry.get(strategy_name)
        if not strategy:
            print(f"❌ Strategy '{strategy_name}' not found.\n")
            print_strategy_list(registry)
            return
    else:
        print("❌ No strategy specified. Use --strategy <name> or --auto\n")
        print_strategy_list(registry)
        return

    # Run strategy
    context = StrategyContext(query=query)
    result = await strategy.enhance(context)

    # Print result
    print_result(result, strategy_name)


def main():
    """Main CLI entry point."""
    print_banner()

    # Parse simple args
    args = sys.argv[1:]

    if not args or '--help' in args or '-h' in args:
        print("Usage:")
        print("  python cli.py <query> [options]\n")
        print("Options:")
        print("  --strategy <name>  Use specific strategy")
        print("  --auto             Auto-detect best strategy")
        print("  --list             List all strategies")
        print("  --help             Show this help\n")
        print("Examples:")
        print("  python cli.py \"explain neural networks\" --strategy deep")
        print("  python cli.py \"solve this problem\" --auto")
        print("  python cli.py --list\n")
        return

    if '--list' in args:
        registry = get_registry()
        print_strategy_list(registry)
        return

    # Extract query and options
    query = None
    strategy_name = None
    auto = False

    i = 0
    while i < len(args):
        if args[i] == '--strategy' and i + 1 < len(args):
            strategy_name = args[i + 1]
            i += 2
        elif args[i] == '--auto':
            auto = True
            i += 1
        elif not args[i].startswith('--'):
            query = args[i]
            i += 1
        else:
            i += 1

    if not query:
        print("❌ No query provided.\n")
        print("Usage: python cli.py <query> [options]\n")
        return

    # Run async
    asyncio.run(run_strategy(query, strategy_name, auto))


if __name__ == '__main__':
    main()
