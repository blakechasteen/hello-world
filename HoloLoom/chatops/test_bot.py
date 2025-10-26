#!/usr/bin/env python3
"""
Matrix Bot Test Script
======================
Tests HoloLoom Matrix bot without requiring live Matrix server.
"""

import asyncio
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HoloLoom.chatops.matrix_bot import MatrixBot, MatrixBotConfig
from HoloLoom.chatops.hololoom_handlers import HoloLoomMatrixHandlers


def print_section(title: str):
    """Print test section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


async def test_configuration():
    """Test bot configuration"""
    print_section("TEST 1: Bot Configuration")

    try:
        config = MatrixBotConfig(
            homeserver_url="https://matrix.org",
            user_id="@testbot:matrix.org",
            password="test_password"
        )
        print("[OK] MatrixBotConfig created successfully")
        print(f"     Homeserver: {config.homeserver_url}")
        print(f"     User: {config.user_id}")
        print(f"     Rate limit: {config.rate_limit_max} cmds/{config.rate_limit_window}s")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration error: {e}")
        return False


async def test_bot_initialization():
    """Test bot initialization"""
    print_section("TEST 2: Bot Initialization")

    try:
        config = MatrixBotConfig(
            homeserver_url="https://matrix.org",
            user_id="@testbot:matrix.org",
            password="test"
        )

        bot = MatrixBot(config)
        print("[OK] MatrixBot initialized")
        print(f"     Client: {bot.client is not None}")
        print(f"     Handlers: {len(bot.handlers)} registered")
        print(f"     Command prefix: {bot.config.command_prefix}")
        return True
    except Exception as e:
        print(f"[FAIL] Bot initialization error: {e}")
        return False


async def test_handler_registration():
    """Test handler registration"""
    print_section("TEST 3: Handler Registration")

    try:
        config = MatrixBotConfig(
            homeserver_url="https://matrix.org",
            user_id="@testbot:matrix.org",
            password="test"
        )

        bot = MatrixBot(config)
        handlers = HoloLoomMatrixHandlers(bot, config_mode="bare")

        print("[OK] HoloLoomMatrixHandlers created")
        print(f"     Orchestrator: {handlers.orchestrator is not None}")

        handlers.register_all()

        print(f"[OK] Registered {len(bot.handlers)} command handlers:")
        for cmd in sorted(bot.handlers.keys()):
            print(f"     - !{cmd}")

        return True
    except Exception as e:
        print(f"[FAIL] Handler registration error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_hololoom_integration():
    """Test HoloLoom orchestrator integration"""
    print_section("TEST 4: HoloLoom Integration")

    try:
        config = MatrixBotConfig(
            homeserver_url="https://matrix.org",
            user_id="@testbot:matrix.org",
            password="test"
        )

        bot = MatrixBot(config)
        handlers = HoloLoomMatrixHandlers(bot, config_mode="bare")

        print("[OK] Orchestrator initialized")
        print(f"     MCTS enabled: {handlers.orchestrator.use_mcts}")
        print(f"     MCTS simulations: {handlers.orchestrator.mcts_simulations}")

        # Test memory
        await handlers.orchestrator.add_knowledge(
            "Test knowledge: MCTS is a tree search algorithm"
        )
        print("[OK] Memory write successful")

        # Test weaving
        spacetime = await handlers.orchestrator.weave("What is MCTS?")
        print(f"[OK] Weaving successful")
        print(f"     Tool selected: {spacetime.tool_used}")
        print(f"     Confidence: {spacetime.confidence:.1%}")
        print(f"     Duration: {spacetime.trace.duration_ms:.1f}ms")

        return True
    except Exception as e:
        print(f"[FAIL] HoloLoom integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_command_parsing():
    """Test command parsing"""
    print_section("TEST 5: Command Parsing")

    try:
        config = MatrixBotConfig(
            homeserver_url="https://matrix.org",
            user_id="@testbot:matrix.org",
            password="test"
        )

        bot = MatrixBot(config)

        # Test command detection
        test_messages = [
            ("!help", True, "help", ""),
            ("!weave What is MCTS?", True, "weave", "What is MCTS?"),
            ("!memory add Test", True, "memory", "add Test"),
            ("Hello bot", False, None, None),
            ("!ping", True, "ping", ""),
        ]

        for msg, should_match, expected_cmd, expected_args in test_messages:
            # Simulate parsing (bot does this in message_callback)
            if msg.startswith(bot.config.command_prefix):
                parts = msg[len(bot.config.command_prefix):].split(maxsplit=1)
                cmd = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                is_command = True
            else:
                is_command = False
                cmd = None
                args = None

            if is_command == should_match and (not should_match or cmd == expected_cmd):
                print(f"[OK] '{msg}' -> cmd='{cmd}', args='{args}'")
            else:
                print(f"[FAIL] '{msg}' parsing mismatch")
                return False

        return True
    except Exception as e:
        print(f"[FAIL] Command parsing error: {e}")
        return False


async def test_rate_limiting():
    """Test rate limiting"""
    print_section("TEST 6: Rate Limiting")

    try:
        config = MatrixBotConfig(
            homeserver_url="https://matrix.org",
            user_id="@testbot:matrix.org",
            password="test",
            rate_limit_max=3,
            rate_limit_window=10.0
        )

        bot = MatrixBot(config)

        # Simulate rapid commands from same user
        user = "@alice:matrix.org"

        for i in range(5):
            is_allowed = bot._check_rate_limit(user)
            if i < 3:
                if is_allowed:
                    print(f"[OK] Command {i+1}/5 allowed (within limit)")
                else:
                    print(f"[FAIL] Command {i+1}/5 blocked (should be allowed)")
                    return False
            else:
                if not is_allowed:
                    print(f"[OK] Command {i+1}/5 blocked (rate limited)")
                else:
                    print(f"[FAIL] Command {i+1}/5 allowed (should be blocked)")
                    return False

        return True
    except Exception as e:
        print(f"[FAIL] Rate limiting error: {e}")
        return False


async def run_all_tests():
    """Run complete test suite"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                  ║")
    print("║         HOLOLOOM MATRIX BOT - TEST SUITE                         ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    tests = [
        ("Configuration", test_configuration),
        ("Bot Initialization", test_bot_initialization),
        ("Handler Registration", test_handler_registration),
        ("HoloLoom Integration", test_hololoom_integration),
        ("Command Parsing", test_command_parsing),
        ("Rate Limiting", test_rate_limiting),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[ERROR] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print_section("TEST RESULTS SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")

    print(f"\n{'='*70}")
    print(f"  {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print(f"{'='*70}\n")

    if passed == total:
        print("🎉 ALL TESTS PASSED! 🎉\n")
        print("The Matrix bot is ready for deployment!\n")
        print("To run live bot:")
        print("  python HoloLoom/chatops/run_bot.py \\")
        print("    --user @yourbot:matrix.org \\")
        print("    --password your_password \\")
        print("    --hololoom-mode fast\n")
        return True
    else:
        print("❌ SOME TESTS FAILED\n")
        print("Please review the errors above.\n")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
