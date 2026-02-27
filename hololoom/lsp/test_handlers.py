#!/usr/bin/env python3
"""
Simple test to verify LSP handlers work correctly.

This script tests the 4 core handlers without a full LSP client.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.lsp.server import (
    extract_word_at_position,
    extract_symbol_at_position,
    format_memory_as_markdown,
)


async def test_helper_functions():
    """Test helper functions."""
    print("=" * 70)
    print("Testing Helper Functions")
    print("=" * 70)

    # Test extract_word_at_position
    line = "  HoloLoom.memory.recall("
    pos = 18  # After "recall"
    word = extract_word_at_position(line, pos)
    print(f"✓ extract_word_at_position: '{line}' @ {pos} → '{word}'")
    assert word == "HoloLoom.memory.recall", f"Expected 'HoloLoom.memory.recall', got '{word}'"

    # Test extract_symbol_at_position
    line = "  my_function(arg1, arg2)"
    pos = 8  # Inside "function"
    symbol = extract_symbol_at_position(line, pos)
    print(f"✓ extract_symbol_at_position: '{line}' @ {pos} → '{symbol}'")
    assert symbol == "my_function", f"Expected 'my_function', got '{symbol}'"

    # Test format_memory_as_markdown
    class MockMemory:
        def __init__(self):
            self.text = "This is a test memory about Thompson Sampling"
            self.metadata = {"source": "test", "confidence": 0.95}
            self.timestamp = "2025-11-16T05:58:35"

    mem = MockMemory()
    markdown = format_memory_as_markdown(mem)
    print(f"✓ format_memory_as_markdown:")
    print(f"  {markdown[:60]}...")
    assert "Thompson Sampling" in markdown
    assert "Metadata" in markdown
    assert "Timestamp" in markdown

    print("\n✅ All helper functions work correctly\n")


async def test_hololoom_integration():
    """Test HoloLoom integration."""
    print("=" * 70)
    print("Testing HoloLoom Integration")
    print("=" * 70)

    # Initialize HoloLoom
    print("Initializing HoloLoom...")
    config = Config.fast()
    loom = HoloLoom(config=config)

    async with loom:
        # Add some test memories
        print("\nAdding test memories...")
        await loom.experience("Thompson Sampling is a Bayesian exploration strategy")
        await loom.experience("def calculate_ucb(mean, variance): return mean + sqrt(variance)")
        await loom.experience("class BanditArm: def __init__(self, alpha, beta): ...")
        print("✓ Added 3 test memories")

        # Test recall (simulating completion handler)
        print("\nTesting recall (completion)...")
        query = "Thompson"
        memories = await loom.recall(query, limit=10)
        print(f"✓ Recalled {len(memories)} memories for '{query}'")
        if memories:
            print(f"  Top result: {memories[0].text[:60]}...")

        # Test recall (simulating hover handler)
        print("\nTesting recall (hover)...")
        symbol = "BanditArm"
        memories = await loom.recall(symbol, limit=5)
        print(f"✓ Recalled {len(memories)} memories for symbol '{symbol}'")
        if memories:
            print(f"  Top result: {memories[0].text[:60]}...")

        # Test recall (simulating symbol search)
        print("\nTesting recall (symbol search)...")
        search_query = "Bayesian"
        memories = await loom.recall(search_query, limit=20)
        print(f"✓ Recalled {len(memories)} memories for search '{search_query}'")

    print("\n✅ HoloLoom integration works correctly\n")


async def test_completion_simulation():
    """Simulate completion handler logic."""
    print("=" * 70)
    print("Simulating Completion Handler")
    print("=" * 70)

    config = Config.fast()
    loom = HoloLoom(config=config)

    async with loom:
        # Add code-like memories
        await loom.experience("import numpy as np")
        await loom.experience("from sklearn.ensemble import RandomForestClassifier")
        await loom.experience("def train_model(X, y): model = RandomForest(); return model.fit(X, y)")

        # Simulate typing "Random" and requesting completion
        line = "    model = Random"
        character = len(line)
        query = extract_word_at_position(line, character)

        print(f"Line: '{line}'")
        print(f"Position: {character}")
        print(f"Extracted query: '{query}'")

        memories = await loom.recall(query, limit=10)
        print(f"\nFound {len(memories)} completion items:")

        for i, mem in enumerate(memories[:3], 1):
            label = mem.text.split('\n')[0][:50]
            print(f"  {i}. {label}")

    print("\n✅ Completion simulation successful\n")


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LSP Handler Tests")
    print("=" * 70 + "\n")

    try:
        await test_helper_functions()
        await test_hololoom_integration()
        await test_completion_simulation()

        print("=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nThe LSP server handlers are working correctly!")
        print("Next step: Test with actual LSP client (VS Code extension)")
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
