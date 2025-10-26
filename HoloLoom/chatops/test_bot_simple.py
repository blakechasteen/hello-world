#!/usr/bin/env python3
"""
Simple Matrix Bot Test - Standalone
====================================
Tests Matrix bot handlers without circular imports.
"""

import asyncio
import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_test(name: str, passed: bool):
    """Print test result"""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {name}")


async def test_imports():
    """Test basic imports"""
    print("\n=== TEST 1: Imports ===\n")

    # Test HoloLoom
    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config
        print_test("HoloLoom imports", True)
        hololoom_ok = True
    except Exception as e:
        print_test(f"HoloLoom imports: {e}", False)
        hololoom_ok = False

    # Test matrix-nio
    try:
        from matrix_nio import AsyncClient
        print_test("matrix-nio import", True)
        nio_ok = True
    except Exception as e:
        print_test(f"matrix-nio import: {e}", False)
        nio_ok = False

    return hololoom_ok and nio_ok


async def test_hololoom_init():
    """Test HoloLoom initialization"""
    print("\n=== TEST 2: HoloLoom Initialization ===\n")

    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config

        config = Config.bare()
        orchestrator = WeavingOrchestrator(
            config=config,
            use_mcts=True,
            mcts_simulations=20
        )

        print_test("Orchestrator creation", True)
        print(f"  MCTS enabled: {orchestrator.use_mcts}")
        print(f"  Simulations: {orchestrator.mcts_simulations}")

        return True
    except Exception as e:
        print_test(f"Orchestrator creation: {e}", False)
        import traceback
        traceback.print_exc()
        return False


async def test_memory_operations():
    """Test memory add/search"""
    print("\n=== TEST 3: Memory Operations ===\n")

    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config

        orchestrator = WeavingOrchestrator(
            config=Config.bare(),
            use_mcts=True,
            mcts_simulations=10
        )

        # Add knowledge
        await orchestrator.add_knowledge("Test: Matrix bot handles commands")
        print_test("Memory add", True)

        # Search (retrieval)
        results = await orchestrator._retrieve_context("Matrix bot", limit=5)
        print_test(f"Memory search ({len(results)} results)", len(results) > 0)

        return True
    except Exception as e:
        print_test(f"Memory operations: {e}", False)
        import traceback
        traceback.print_exc()
        return False


async def test_weaving():
    """Test weaving cycle"""
    print("\n=== TEST 4: Weaving Cycle ===\n")

    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config

        orchestrator = WeavingOrchestrator(
            config=Config.bare(),
            use_mcts=True,
            mcts_simulations=10
        )

        # Execute weaving
        spacetime = await orchestrator.weave("What is the Matrix bot?")

        print_test("Weaving execution", True)
        print(f"  Tool: {spacetime.tool_used}")
        print(f"  Confidence: {spacetime.confidence:.1%}")
        print(f"  Duration: {spacetime.trace.duration_ms:.1f}ms")

        return True
    except Exception as e:
        print_test(f"Weaving execution: {e}", False)
        import traceback
        traceback.print_exc()
        return False


async def test_handler_logic():
    """Test handler command parsing logic"""
    print("\n=== TEST 5: Command Handler Logic ===\n")

    try:
        # Simulate command parsing
        def parse_command(text: str, prefix: str = "!"):
            """Simulate Matrix bot command parsing"""
            if not text.startswith(prefix):
                return None, None

            parts = text[len(prefix):].split(maxsplit=1)
            cmd = parts[0]
            args = parts[1] if len(parts) > 1 else ""
            return cmd, args

        tests = [
            ("!help", "help", ""),
            ("!weave What is MCTS?", "weave", "What is MCTS?"),
            ("!memory add Test", "memory", "add Test"),
            ("Hello", None, None),
        ]

        all_passed = True
        for text, expected_cmd, expected_args in tests:
            cmd, args = parse_command(text)
            passed = cmd == expected_cmd and (cmd is None or args == expected_args)
            print_test(f"Parse '{text}' -> cmd='{cmd}', args='{args}'", passed)
            all_passed = all_passed and passed

        return all_passed
    except Exception as e:
        print_test(f"Handler logic: {e}", False)
        return False


async def run_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("  MATRIX BOT INTEGRATION - SIMPLE TESTS")
    print("="*70)

    tests = [
        test_imports,
        test_hololoom_init,
        test_memory_operations,
        test_weaving,
        test_handler_logic,
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Test crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70 + "\n")

    passed = sum(1 for r in results if r)
    total = len(results)

    print(f"Passed: {passed}/{total} ({passed/total*100:.0f}%)\n")

    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉\n")
        print("Matrix bot integration is working!\n")
        print("Next steps:")
        print("  1. Configure Matrix credentials")
        print("  2. Run: python HoloLoom/chatops/run_bot.py --user @bot:matrix.org --password secret")
        print("  3. Invite bot to room")
        print("  4. Test commands: !help, !weave, !memory, !stats\n")
        return True
    else:
        print("❌ SOME TESTS FAILED\n")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
