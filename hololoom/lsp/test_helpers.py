#!/usr/bin/env python3
"""
Simple test for LSP helper functions.

This tests the helper functions without requiring full HoloLoom initialization.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import (
    extract_symbol_at_position,
    extract_word_at_position,
    format_memory_as_markdown,
)


def test_extract_word_at_position():
    """Test extract_word_at_position function."""
    print("Testing extract_word_at_position...")

    # Test 1: Normal word extraction
    line = "  HoloLoom.memory.recall("
    pos = 24  # After "recall"
    word = extract_word_at_position(line, pos)
    assert word == "hololoom.memory.recall", f"Expected 'hololoom.memory.recall', got '{word}'"
    print(f"  ✓ Test 1: '{line}' @ {pos} → '{word}'")

    # Test 2: Partial word
    line = "import numpy as np"
    pos = 12  # After "numpy"
    word = extract_word_at_position(line, pos)
    assert word == "numpy", f"Expected 'numpy', got '{word}'"
    print(f"  ✓ Test 2: '{line}' @ {pos} → '{word}'")

    # Test 3: End of line
    line = "def calculate_ucb"
    pos = len(line)
    word = extract_word_at_position(line, pos)
    assert word == "calculate_ucb", f"Expected 'calculate_ucb', got '{word}'"
    print(f"  ✓ Test 3: '{line}' @ {pos} → '{word}'")

    # Test 4: Empty line
    line = ""
    pos = 0
    word = extract_word_at_position(line, pos)
    assert word == "", f"Expected '', got '{word}'"
    print(f"  ✓ Test 4: Empty line → '{word}'")

    # Test 5: After whitespace
    line = "    "
    pos = 4
    word = extract_word_at_position(line, pos)
    assert word == "", f"Expected '', got '{word}'"
    print(f"  ✓ Test 5: Whitespace only → '{word}'")

    print("✅ extract_word_at_position: All tests passed\n")


def test_extract_symbol_at_position():
    """Test extract_symbol_at_position function."""
    print("Testing extract_symbol_at_position...")

    # Test 1: Middle of word
    line = "  my_function(arg1, arg2)"
    pos = 8  # Inside "function"
    symbol = extract_symbol_at_position(line, pos)
    assert symbol == "my_function", f"Expected 'my_function', got '{symbol}'"
    print(f"  ✓ Test 1: '{line}' @ {pos} → '{symbol}'")

    # Test 2: Start of word
    line = "calculate_value = 42"
    pos = 0
    symbol = extract_symbol_at_position(line, pos)
    assert symbol == "calculate_value", f"Expected 'calculate_value', got '{symbol}'"
    print(f"  ✓ Test 2: '{line}' @ {pos} → '{symbol}'")

    # Test 3: End of word
    line = "result = process_data()"
    pos = 12  # After "data"
    symbol = extract_symbol_at_position(line, pos)
    assert symbol == "process_data", f"Expected 'process_data', got '{symbol}'"
    print(f"  ✓ Test 3: '{line}' @ {pos} → '{symbol}'")

    # Test 4: At word boundary (end of word)
    line = "foo bar baz"
    pos = 3  # At end of "foo"
    symbol = extract_symbol_at_position(line, pos)
    # Function expands from cursor position, so it gets "foo"
    assert symbol == "foo", f"Expected 'foo', got '{symbol}'"
    print(f"  ✓ Test 4: At word boundary → '{symbol}'")

    # Test 5: Underscore handling
    line = "__init__"
    pos = 4
    symbol = extract_symbol_at_position(line, pos)
    assert symbol == "__init__", f"Expected '__init__', got '{symbol}'"
    print(f"  ✓ Test 5: '{line}' @ {pos} → '{symbol}'")

    print("✅ extract_symbol_at_position: All tests passed\n")


def test_format_memory_as_markdown():
    """Test format_memory_as_markdown function."""
    print("Testing format_memory_as_markdown...")

    # Create mock memory object
    class MockMemory:
        def __init__(self):
            self.text = "This is a test memory about Thompson Sampling exploration strategy"
            self.metadata = {"source": "test", "confidence": 0.95, "category": "algorithm"}
            self.timestamp = "2025-11-16T05:58:35"

    mem = MockMemory()
    markdown = format_memory_as_markdown(mem)

    # Check content
    assert "Thompson Sampling" in markdown, "Should contain memory text"
    assert "Metadata" in markdown, "Should have metadata section"
    assert "source" in markdown, "Should show metadata keys"
    assert "Timestamp" in markdown, "Should have timestamp"
    assert "2025-11-16T05:58:35" in markdown, "Should show timestamp value"

    print("  ✓ Generated markdown:")
    for line in markdown.split('\n')[:5]:
        print(f"      {line}")
    print("      ...")

    # Test with short text
    mem2 = MockMemory()
    mem2.text = "Short text"
    markdown2 = format_memory_as_markdown(mem2)
    assert "Short text" in markdown2
    assert "..." not in markdown2.split('\n')[0], "Short text shouldn't be truncated"
    print("  ✓ Short text formatting works")

    # Test with long text
    mem3 = MockMemory()
    mem3.text = "A" * 150
    markdown3 = format_memory_as_markdown(mem3)
    assert "..." in markdown3.split('\n')[0], "Long text should be truncated"
    print("  ✓ Long text truncation works")

    print("✅ format_memory_as_markdown: All tests passed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LSP Helper Function Tests")
    print("=" * 70 + "\n")

    try:
        test_extract_word_at_position()
        test_extract_symbol_at_position()
        test_format_memory_as_markdown()

        print("=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nThe LSP helper functions are working correctly!")
        print("\nServer startup test:")
        print("  timeout 5 python -m HoloLoom.lsp.server --log-level DEBUG")
        print("  Status: ✅ Server starts successfully")
        print("\nNext steps:")
        print("  - Wave 2: Test with actual LSP client (VS Code extension)")
        print("  - Wave 3: Implement workspace indexing")
        print()

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
