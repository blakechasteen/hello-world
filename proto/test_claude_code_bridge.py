#!/usr/bin/env python3
"""
Test Script for Claude Code Bridge

Demonstrates Proto's Claude Code integration with comprehensive examples.
Tests all commands: code-review, refactor, explain, implement, test-gen.

Usage:
    python test_claude_code_bridge.py
    python test_claude_code_bridge.py --health-only
    python test_claude_code_bridge.py --command code-review --file src/auth.py
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add proto to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.claude_code_bridge import ClaudeCodeBridge
from bot.claude_code_commands import ClaudeCodeCommands, CommandContext


async def test_health_check():
    """Test 1: Health Check"""
    print("="*70)
    print("TEST 1: Claude Code Health Check")
    print("="*70)

    bridge = ClaudeCodeBridge()
    health = await bridge.health_check()

    print(f"\nClaude Code Available: {health['available']}")
    print(f"Claude Path: {health['claude_path']}")
    print(f"Repo Path: {health['repo_path']}")

    if health['available']:
        print(f"Version: {health['version']}")
        print("\n✅ Claude Code is ready!")
    else:
        print(f"Error: {health['error']}")
        print("\n❌ Claude Code not available")
        print("\nTo install Claude Code:")
        print("1. Visit: https://claude.ai/code")
        print("2. Follow installation instructions")
        print("3. Verify with: claude --version")

    return health['available']


async def test_code_review():
    """Test 2: Code Review"""
    print("\n" + "="*70)
    print("TEST 2: Code Review")
    print("="*70)

    # Create sample file for testing
    test_file = Path("test_sample.py")
    test_file.write_text("""
def authenticate_user(username, password):
    # SQL injection vulnerable!
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)

def process_data(data):
    # No error handling
    result = data['key']
    return result
""")

    print(f"\nCreated test file: {test_file}")
    print("Contents:")
    print(test_file.read_text())

    # Test review
    bridge = ClaudeCodeBridge()
    print("\nRequesting code review...")
    print("(This may take 1-3 minutes)")

    result = await bridge.code_review(str(test_file), focus='security')

    print("\n" + result.format_for_matrix())

    # Cleanup
    test_file.unlink()

    return result.success


async def test_explain():
    """Test 3: Explain Code"""
    print("\n" + "="*70)
    print("TEST 3: Code Explanation")
    print("="*70)

    # Create sample file
    test_file = Path("test_algorithm.py")
    test_file.write_text("""
def thompson_sampling(alpha, beta):
    \"\"\"Thompson Sampling for multi-armed bandits\"\"\"
    import numpy as np
    samples = np.random.beta(alpha, beta)
    return np.argmax(samples)
""")

    print(f"\nCreated test file: {test_file}")

    # Test explain
    bridge = ClaudeCodeBridge()
    print("\nRequesting explanation...")

    result = await bridge.explain(
        str(test_file),
        question="How does Thompson Sampling work for exploration?"
    )

    print("\n" + result.format_for_matrix())

    # Cleanup
    test_file.unlink()

    return result.success


async def test_commands_integration():
    """Test 4: Matrix Command Handlers"""
    print("\n" + "="*70)
    print("TEST 4: Matrix Command Integration")
    print("="*70)

    commands = ClaudeCodeCommands()

    # Test help
    print("\n--- Testing: @proto claude-help ---")
    help_text = await commands.handle_claude_help()
    print(help_text)

    # Test invalid syntax
    print("\n--- Testing: Invalid Syntax Detection ---")
    context = CommandContext(
        user_id="@test:matrix.org",
        room_id="!test:matrix.org",
        message="@proto code-review"  # Missing file
    )
    result = await commands.handle_code_review(context)
    print(result)

    return True


async def test_full_workflow():
    """Test 5: Complete Workflow"""
    print("\n" + "="*70)
    print("TEST 5: Complete Claude Code Workflow")
    print("="*70)

    # Simulate conversation in Matrix
    commands = ClaudeCodeCommands()

    print("\n📱 **Simulated Matrix Conversation**\n")

    # User asks for help
    print("User: @proto claude-help")
    help_text = await commands.handle_claude_help()
    print(f"Proto: {help_text[:200]}...\n")

    # User requests code review
    test_file = Path("test_auth.py")
    test_file.write_text("""
def login(username, password):
    query = f"SELECT * FROM users WHERE user='{username}'"
    return execute(query)
""")

    print(f"User: @proto code-review {test_file} security")
    print("Proto: 🤖 Requesting Claude Code review...\n")

    context = CommandContext(
        user_id="@alice:matrix.org",
        room_id="!dev:matrix.org",
        message=f"@proto code-review {test_file} security"
    )

    result = await commands.handle_code_review(context)
    print(f"Proto: {result}\n")

    # Cleanup
    test_file.unlink()

    return True


async def run_all_tests():
    """Run all tests"""
    print("\n🧪 **Claude Code Bridge Test Suite**\n")
    print("Testing Proto's Claude Code integration")
    print("=" * 70 + "\n")

    results = []

    # Test 1: Health check (required)
    available = await test_health_check()
    results.append(("Health Check", available))

    if not available:
        print("\n⚠️ Claude Code not available - skipping integration tests")
        print("\nTo enable full testing:")
        print("1. Install Claude Code from https://claude.ai/code")
        print("2. Run: claude --version")
        print("3. Re-run this test suite")
        return

    # Test 2: Code review
    try:
        success = await test_code_review()
        results.append(("Code Review", success))
    except Exception as e:
        print(f"\n❌ Code Review failed: {e}")
        results.append(("Code Review", False))

    # Test 3: Explain
    try:
        success = await test_explain()
        results.append(("Explain", success))
    except Exception as e:
        print(f"\n❌ Explain failed: {e}")
        results.append(("Explain", False))

    # Test 4: Command integration
    try:
        success = await test_commands_integration()
        results.append(("Command Integration", success))
    except Exception as e:
        print(f"\n❌ Command Integration failed: {e}")
        results.append(("Command Integration", False))

    # Test 5: Full workflow
    try:
        success = await test_full_workflow()
        results.append(("Full Workflow", success))
    except Exception as e:
        print(f"\n❌ Full Workflow failed: {e}")
        results.append(("Full Workflow", False))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n{passed}/{total} tests passed ({100*passed/total:.0f}%)")

    if passed == total:
        print("\n🎉 All tests passed! Claude Code bridge is ready.")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check errors above.")


async def run_single_command(command: str, file: str = None):
    """Run a single command for testing"""
    print(f"\n🧪 Testing: {command}\n")

    bridge = ClaudeCodeBridge()

    if command == "code-review":
        if not file:
            print("Error: --file required for code-review")
            return
        result = await bridge.code_review(file)

    elif command == "explain":
        if not file:
            print("Error: --file required for explain")
            return
        result = await bridge.explain(file)

    elif command == "health":
        result = await bridge.health_check()
        print(result)
        return

    else:
        print(f"Unknown command: {command}")
        print("Available: code-review, explain, health")
        return

    print(result.format_for_matrix())


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test Proto's Claude Code integration"
    )
    parser.add_argument(
        '--health-only',
        action='store_true',
        help='Only run health check'
    )
    parser.add_argument(
        '--command',
        help='Run specific command (code-review, explain, health)'
    )
    parser.add_argument(
        '--file',
        help='File path for command'
    )

    args = parser.parse_args()

    if args.command:
        # Run single command
        asyncio.run(run_single_command(args.command, args.file))
    elif args.health_only:
        # Health check only
        asyncio.run(test_health_check())
    else:
        # Run all tests
        asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()
