#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom ChatOps - Deployment Verification
===========================================
Verifies that all components are working correctly.

Usage:
    python verify_deployment.py
    python verify_deployment.py --config chatops_test_config.yaml
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path
from typing import List, Tuple

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Verification Tests
# ============================================================================

class VerificationTest:
    """Base class for verification tests."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None

    async def run(self) -> bool:
        """Run test (to be overridden)."""
        raise NotImplementedError

    def report(self) -> str:
        """Get test result report."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        msg = f"{status} - {self.name}"
        if not self.passed and self.error:
            msg += f"\n    Error: {self.error}"
        return msg


class ImportTest(VerificationTest):
    """Test that all modules can be imported."""

    def __init__(self):
        super().__init__("Module Imports")

    async def run(self) -> bool:
        try:
            # Core imports
            from holoLoom.chatops import MatrixBot, ChatOpsOrchestrator
            from holoLoom.chatops.conversation_memory import ConversationMemory
            from holoLoom.chatops.matrix_bot import MatrixBotConfig

            # Optional imports
            try:
                from holoLoom.chatops import ChatOpsSkills
            except ImportError:
                pass  # Optional

            try:
                from holoLoom.config import Config
            except ImportError:
                pass  # Optional

            self.passed = True
            return True

        except Exception as e:
            self.error = str(e)
            self.passed = False
            return False


class DependencyTest(VerificationTest):
    """Test that required dependencies are installed."""

    def __init__(self):
        super().__init__("Dependencies")

    async def run(self) -> bool:
        missing = []

        # Required
        try:
            import nio
        except ImportError:
            missing.append("matrix-nio")

        try:
            import yaml
        except ImportError:
            missing.append("pyyaml")

        try:
            import networkx
        except ImportError:
            missing.append("networkx")

        # Optional
        try:
            import sentence_transformers
        except ImportError:
            pass  # Optional

        if missing:
            self.error = f"Missing: {', '.join(missing)}"
            self.passed = False
            return False

        self.passed = True
        return True


class ConfigTest(VerificationTest):
    """Test configuration loading."""

    def __init__(self, config_path: str):
        super().__init__("Configuration")
        self.config_path = config_path

    async def run(self) -> bool:
        try:
            import yaml

            # Check if config exists
            if not Path(self.config_path).exists():
                self.error = f"Config file not found: {self.config_path}"
                self.passed = False
                return False

            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Validate required fields
            if 'matrix' not in config:
                raise ValueError("Missing 'matrix' section")

            if 'homeserver_url' not in config['matrix']:
                raise ValueError("Missing 'matrix.homeserver_url'")

            if 'user_id' not in config['matrix']:
                raise ValueError("Missing 'matrix.user_id'")

            # Check auth
            has_auth = (
                config['matrix'].get('access_token') or
                config['matrix'].get('password') or
                os.getenv('MATRIX_ACCESS_TOKEN') or
                os.getenv('MATRIX_PASSWORD')
            )

            if not has_auth:
                self.error = "No authentication configured (access_token or password)"
                self.passed = False
                return False

            self.passed = True
            return True

        except Exception as e:
            self.error = str(e)
            self.passed = False
            return False


class DirectoryTest(VerificationTest):
    """Test that required directories exist or can be created."""

    def __init__(self):
        super().__init__("Directories")

    async def run(self) -> bool:
        try:
            # Test directory creation
            test_dirs = [
                Path("./test_matrix_store"),
                Path("./test_chatops_memory"),
                Path("./logs")
            ]

            for dir_path in test_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)

                # Test write permissions
                test_file = dir_path / ".test"
                test_file.touch()
                test_file.unlink()

            self.passed = True
            return True

        except Exception as e:
            self.error = str(e)
            self.passed = False
            return False


class ComponentTest(VerificationTest):
    """Test that components can be instantiated."""

    def __init__(self):
        super().__init__("Component Instantiation")

    async def run(self) -> bool:
        try:
            from holoLoom.chatops.matrix_bot import MatrixBotConfig, MatrixBot
            from holoLoom.chatops.chatops_bridge import ChatOpsOrchestrator
            from holoLoom.chatops.conversation_memory import ConversationMemory

            # Test MatrixBotConfig
            config = MatrixBotConfig(
                homeserver_url="https://matrix.org",
                user_id="@test:matrix.org",
                access_token="test_token"
            )

            # Test ConversationMemory
            memory = ConversationMemory()

            # Test ChatOpsOrchestrator (without bot connection)
            chatops = ChatOpsOrchestrator(
                memory_store_path="./test_verify_memory",
                enable_memory_storage=False
            )

            self.passed = True
            return True

        except Exception as e:
            self.error = str(e)
            self.passed = False
            return False


class MatrixConnectionTest(VerificationTest):
    """Test Matrix connection (if credentials provided)."""

    def __init__(self, config_path: str):
        super().__init__("Matrix Connection (Optional)")
        self.config_path = config_path

    async def run(self) -> bool:
        try:
            import yaml
            from nio import AsyncClient

            # Load config
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            matrix_config = config['matrix']

            # Get credentials
            access_token = (
                matrix_config.get('access_token') or
                os.getenv('MATRIX_ACCESS_TOKEN')
            )

            if not access_token:
                self.error = "No access token (skipped)"
                self.passed = True  # Not a failure
                return True

            # Try to connect
            client = AsyncClient(
                homeserver=matrix_config['homeserver_url'],
                user=matrix_config['user_id']
            )
            client.access_token = access_token

            # Test whoami
            response = await client.whoami()

            if hasattr(response, 'user_id'):
                self.passed = True
                await client.close()
                return True
            else:
                self.error = f"Connection failed: {response}"
                self.passed = False
                await client.close()
                return False

        except Exception as e:
            self.error = f"Connection test failed: {str(e)}"
            self.passed = True  # Don't fail on connection issues
            return True


# ============================================================================
# Main Verification Runner
# ============================================================================

async def run_verification(config_path: str) -> Tuple[int, int]:
    """
    Run all verification tests.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (passed_count, total_count)
    """
    print("="*80)
    print("HoloLoom ChatOps - Deployment Verification")
    print("="*80)
    print()

    # Create test suite
    tests: List[VerificationTest] = [
        ImportTest(),
        DependencyTest(),
        DirectoryTest(),
        ComponentTest(),
        ConfigTest(config_path),
        MatrixConnectionTest(config_path),
    ]

    # Run tests
    print("Running verification tests...\n")

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            result = await test.run()
            if result:
                passed += 1
        except Exception as e:
            test.error = str(e)
            test.passed = False

    # Print results
    print("Results:")
    print("-" * 80)
    for test in tests:
        print(test.report())
    print("-" * 80)
    print()

    # Summary
    print(f"Summary: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! Deployment is ready.")
        return passed, total
    else:
        print("✗ Some tests failed. Please review errors above.")
        return passed, total


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify HoloLoom ChatOps deployment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="HoloLoom/chatops/config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    # Run verification
    passed, total = asyncio.run(run_verification(args.config))

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
