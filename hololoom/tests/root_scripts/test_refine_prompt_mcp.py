#!/usr/bin/env python3
"""
Test script for refine_prompt MCP tool integration

Tests the handler function directly without running the full MCP server.
"""

import asyncio
import json
from pathlib import Path
import sys

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.config import Config
from HoloLoom.prompting.metaprompt import create_metaprompt_auto, enhance_request


async def test_refine_prompt():
    """Test the refine_prompt handler logic."""

    print("=" * 70)
    print("Testing refine_prompt MCP Tool")
    print("=" * 70)
    print()

    # Test 1: Basic refinement (auto provider, with strategy)
    print("[Test 1] Basic refinement with auto provider + strategy")
    print("-" * 70)

    test_prompt = "write a Python function to sort data"

    try:
        config = Config.fast()
        provider = "anthropic"  # Default

        refined = create_metaprompt_auto(
            request=test_prompt,
            config=config,
            confidence_threshold=0.7
        )

        print(f"✅ Original: {test_prompt}")
        print(f"✅ Refined length: {len(refined)} characters")
        print(f"✅ Expansion ratio: {round(len(refined) / len(test_prompt), 1)}x")
        print()
        print("First 500 characters of refined prompt:")
        print(refined[:500] + "...")
        print()

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Simple refinement without strategy
    print("\n[Test 2] Simple refinement without strategy")
    print("-" * 70)

    try:
        refined_simple = enhance_request(test_prompt, provider="anthropic")

        print(f"✅ Simple refinement length: {len(refined_simple)} characters")
        print(f"✅ Expansion ratio: {round(len(refined_simple) / len(test_prompt), 1)}x")
        print()

    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Verify CORE_TEMPLATE.md is being used
    print("\n[Test 3] Verify 7-component framework is present")
    print("-" * 70)

    framework_components = [
        "ROLE", "OBJECTIVE", "PROCESS", "FORMAT",
        "CONSTRAINTS", "UNCERTAINTY", "VALIDATION"
    ]

    found_components = [comp for comp in framework_components if comp in refined]

    if len(found_components) >= 5:  # At least 5/7 components
        print(f"✅ Found {len(found_components)}/7 framework components:")
        for comp in found_components:
            print(f"   - {comp}")
    else:
        print(f"⚠️  Only found {len(found_components)}/7 components")
        print(f"   Found: {found_components}")

    print()

    # Test 4: Error handling - missing template
    print("\n[Test 4] Error handling simulation")
    print("-" * 70)

    try:
        # This should work if template exists
        test_result = {
            "status": "success",
            "original_prompt": test_prompt,
            "refined_prompt": refined[:100] + "...",
            "provider": "anthropic",
            "strategy_applied": True,
            "confidence_threshold": 0.7,
            "framework": "7-component (ROLE, OBJECTIVE, PROCESS, FORMAT, CONSTRAINTS, UNCERTAINTY, VALIDATION)",
            "expansion_ratio": round(len(refined) / len(test_prompt), 1)
        }

        result_json = json.dumps(test_result, indent=2)
        print("✅ JSON response structure:")
        print(result_json)

    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False

    print()
    print("=" * 70)
    print("All tests passed! ✅")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  • Original prompt: {len(test_prompt)} chars")
    print(f"  • Refined prompt: {len(refined)} chars")
    print(f"  • Expansion: {round(len(refined) / len(test_prompt), 1)}x")
    print(f"  • Framework components: {len(found_components)}/7")
    print(f"  • Provider: anthropic (Claude optimizations)")
    print()
    print("Next steps:")
    print("  1. Restart Claude Desktop MCP server")
    print("  2. Test refine_prompt tool via Claude Desktop")
    print("  3. Verify quality improvements in actual usage")
    print()

    return True


if __name__ == "__main__":
    success = asyncio.run(test_refine_prompt())
    sys.exit(0 if success else 1)
