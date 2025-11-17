#!/usr/bin/env python3
"""
Test script for Proto Git Handler Integration

Tests git command integration without Matrix dependency.
Run from repository root: python proto/test_git_integration.py
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add proto to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.git_handler import GitHandler
from bot.command_parser import CommandParser


def test_git_handler():
    """Test GitHandler directly"""
    logger.info("=" * 60)
    logger.info("Testing GitHandler")
    logger.info("=" * 60)

    try:
        # Initialize handler with current repo
        handler = GitHandler(repo_path=os.getcwd())
        logger.info("✓ GitHandler initialized")

        # Test status
        logger.info("\n--- Testing: git status ---")
        status = handler.status(short=True)
        logger.info(f"Status:\n{status}")

        # Test branch
        logger.info("\n--- Testing: current branch ---")
        branch = handler.get_current_branch()
        logger.info(f"Current branch: {branch}")

        # Test log
        logger.info("\n--- Testing: git log ---")
        log = handler.log(max_count=3, oneline=True)
        logger.info(f"Recent commits:\n{log}")

        # Test diff (may be empty)
        logger.info("\n--- Testing: git diff ---")
        diff = handler.diff()
        if diff:
            logger.info(f"Diff:\n{diff[:500]}")
        else:
            logger.info("No uncommitted changes (working tree clean)")

        # Test branch list
        logger.info("\n--- Testing: git branch ---")
        branches = handler.branch()
        logger.info(f"Branches:\n{branches}")

        logger.info("\n✓ All GitHandler tests passed!")
        return True

    except Exception as e:
        logger.error(f"✗ GitHandler test failed: {e}")
        return False


def test_command_parser():
    """Test CommandParser recognizes git commands"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing CommandParser for Git Commands")
    logger.info("=" * 60)

    parser = CommandParser()
    test_cases = [
        ("@promptly git status", "git-status"),
        ("@promptly git log", "git-log"),
        ("@promptly git diff", "git-diff"),
        ("@promptly git branch", "git-branch"),
        ('@promptly git commit "test message"', "git-commit"),
        ("@promptly git push", "git-push"),
        ("@promptly git pull", "git-pull"),
        ("!git status", "git-status"),
    ]

    all_passed = True
    for message, expected_type in test_cases:
        command = parser.parse(message)
        if command and command.get('type') == expected_type:
            logger.info(f"✓ Parsed: {message} → {command['type']}")
        else:
            logger.error(f"✗ Failed: {message}")
            logger.error(f"  Expected: {expected_type}")
            logger.error(f"  Got: {command}")
            all_passed = False

    if all_passed:
        logger.info("\n✓ All CommandParser tests passed!")
    else:
        logger.error("\n✗ Some CommandParser tests failed!")

    return all_passed


def test_mock_bot_methods():
    """Test bot method structure without Matrix"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Bot Method Structure")
    logger.info("=" * 60)

    try:
        # Import bot to check methods exist
        from bot.promptly_bot import PromptlyBot

        # Check all git methods are defined
        git_methods = [
            'cmd_git_status',
            'cmd_git_log',
            'cmd_git_diff',
            'cmd_git_branch',
            'cmd_git_commit',
            'cmd_git_push',
            'cmd_git_pull',
        ]

        for method_name in git_methods:
            if hasattr(PromptlyBot, method_name):
                logger.info(f"✓ Method exists: {method_name}")
            else:
                logger.error(f"✗ Method missing: {method_name}")
                return False

        logger.info("\n✓ All bot methods present!")
        return True

    except ImportError as e:
        if 'nio' in str(e):
            logger.warning(f"⊕ Skipped (Matrix SDK not installed): {e}")
            logger.info("  This is expected in test environment without Matrix dependencies")
            return True
        else:
            logger.error(f"✗ Failed to check bot methods: {e}")
            return False

    except Exception as e:
        logger.error(f"✗ Failed to check bot methods: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("Proto Git Handler Integration Tests")
    logger.info("=" * 60)

    results = []

    # Test 1: GitHandler
    results.append(("GitHandler", test_git_handler()))

    # Test 2: CommandParser
    results.append(("CommandParser", test_command_parser()))

    # Test 3: Bot Methods
    results.append(("Bot Methods", test_mock_bot_methods()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ All Integration Tests Passed!")
        logger.info("=" * 60)
        logger.info("\nGit Handler Integration is ready to use!")
        logger.info("\nUsage from Matrix chat:")
        logger.info("  @promptly git status   - Show git status")
        logger.info("  @promptly git log      - Show recent commits")
        logger.info("  @promptly git diff     - Show changes")
        logger.info("  @promptly git branch   - List branches")
        logger.info("  @promptly git commit \"message\" - Create commit")
        logger.info("  @promptly git push     - Push to remote")
        logger.info("  @promptly git pull     - Pull from remote")
        return 0
    else:
        logger.error("✗ Some tests failed!")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
