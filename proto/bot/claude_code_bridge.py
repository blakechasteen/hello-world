#!/usr/bin/env python3
"""
Claude Code CLI Bridge for Proto

Integrates Claude Code (claude.ai/code) via subprocess for:
- Code review with security analysis
- Refactoring suggestions
- Code explanation
- Implementation assistance

Built for Proto - Conversational Intelligence Hub
"""

import subprocess
import asyncio
import logging
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ClaudeCommandType(Enum):
    """Supported Claude Code command types"""
    REVIEW = "review"
    REFACTOR = "refactor"
    EXPLAIN = "explain"
    IMPLEMENT = "implement"
    TEST_GEN = "test-gen"


@dataclass
class ClaudeResponse:
    """Structured response from Claude Code"""
    success: bool
    command: str
    output: str
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            'success': self.success,
            'command': self.command,
            'output': self.output,
            'error': self.error,
            'duration_seconds': self.duration_seconds,
            'timestamp': self.timestamp.isoformat()
        }

    def format_for_matrix(self) -> str:
        """Format response for Matrix chat display"""
        if not self.success:
            return f"❌ **Claude Code Error**\n\n{self.error}\n\n_({self.duration_seconds:.1f}s)_"

        # Add emoji header based on command
        emoji_map = {
            'review': '🔍',
            'refactor': '🔧',
            'explain': '📚',
            'implement': '⚙️',
            'test-gen': '🧪'
        }
        emoji = emoji_map.get(self.command.split()[0], '🤖')

        header = f"{emoji} **Claude Code {self.command.title()}**\n\n"
        footer = f"\n\n_Completed in {self.duration_seconds:.1f}s_"

        return header + self.output + footer


class ClaudeCodeBridge:
    """
    Bridge to Claude Code CLI.

    Provides subprocess-based integration with Claude Code for code analysis,
    refactoring, and generation tasks. Designed for conversational use via
    Matrix chat.

    Usage:
        bridge = ClaudeCodeBridge()

        # Code review
        result = await bridge.code_review('src/auth.py')
        print(result.format_for_matrix())

        # Refactoring
        result = await bridge.refactor('extract_method', 'src/utils.py:42-67')

        # Explanation
        result = await bridge.explain('src/auth.py', 'How does JWT validation work?')
    """

    def __init__(
        self,
        claude_path: str = "claude",
        repo_path: Optional[str] = None,
        default_timeout: int = 300,
        max_output_lines: int = 1000
    ):
        """
        Initialize Claude Code bridge.

        Args:
            claude_path: Path to claude executable (default: 'claude' from PATH)
            repo_path: Repository path for context (default: current directory)
            default_timeout: Default timeout in seconds (default: 5 minutes)
            max_output_lines: Maximum output lines to prevent flooding
        """
        self.claude_path = claude_path
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.default_timeout = default_timeout
        self.max_output_lines = max_output_lines

        # Verify Claude Code is available
        self._check_availability()

    def _check_availability(self) -> bool:
        """
        Check if Claude Code CLI is available.

        Returns:
            True if available, False otherwise
        """
        try:
            result = subprocess.run(
                [self.claude_path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Claude Code CLI found: {result.stdout.strip()}")
                return True
            else:
                logger.warning("Claude Code CLI not responding correctly")
                return False
        except FileNotFoundError:
            logger.warning(f"Claude Code CLI not found at: {self.claude_path}")
            logger.warning("Install from: https://claude.ai/code")
            return False
        except Exception as e:
            logger.error(f"Error checking Claude Code availability: {e}")
            return False

    async def _run_command(
        self,
        args: List[str],
        timeout: Optional[int] = None,
        stdin_input: Optional[str] = None
    ) -> ClaudeResponse:
        """
        Run Claude Code command asynchronously.

        Args:
            args: Command arguments
            timeout: Timeout in seconds (uses default if None)
            stdin_input: Optional stdin input

        Returns:
            ClaudeResponse with results
        """
        timeout = timeout or self.default_timeout
        start_time = datetime.now()

        try:
            # Build full command
            cmd = [self.claude_path] + args
            cmd_str = ' '.join(args)

            logger.info(f"Running Claude Code: {cmd_str}")

            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin_input else None,
                cwd=str(self.repo_path)
            )

            # Run with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(
                        input=stdin_input.encode() if stdin_input else None
                    ),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                duration = (datetime.now() - start_time).total_seconds()
                return ClaudeResponse(
                    success=False,
                    command=cmd_str,
                    output="",
                    error=f"Command timed out after {timeout}s",
                    duration_seconds=duration
                )

            duration = (datetime.now() - start_time).total_seconds()

            # Decode output
            output = stdout.decode('utf-8', errors='replace').strip()
            error = stderr.decode('utf-8', errors='replace').strip()

            # Truncate if too long
            if output.count('\n') > self.max_output_lines:
                lines = output.split('\n')
                output = '\n'.join(lines[:self.max_output_lines])
                output += f"\n\n... (truncated {len(lines) - self.max_output_lines} lines)"

            # Check exit code
            if process.returncode == 0:
                return ClaudeResponse(
                    success=True,
                    command=cmd_str,
                    output=output,
                    duration_seconds=duration
                )
            else:
                # Command failed
                error_msg = error or output or f"Exit code: {process.returncode}"
                return ClaudeResponse(
                    success=False,
                    command=cmd_str,
                    output=output,
                    error=error_msg,
                    duration_seconds=duration
                )

        except FileNotFoundError:
            return ClaudeResponse(
                success=False,
                command=' '.join(args),
                output="",
                error="Claude Code CLI not found. Install from https://claude.ai/code",
                duration_seconds=0
            )
        except Exception as e:
            logger.error(f"Error running Claude Code: {e}")
            return ClaudeResponse(
                success=False,
                command=' '.join(args),
                output="",
                error=f"Internal error: {str(e)}",
                duration_seconds=0
            )

    async def code_review(
        self,
        file_path: str,
        focus: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> ClaudeResponse:
        """
        Request code review from Claude Code.

        Args:
            file_path: Path to file (relative to repo)
            focus: Optional focus area (e.g., 'security', 'performance', 'style')
            timeout: Optional timeout override

        Returns:
            ClaudeResponse with review results

        Example:
            result = await bridge.code_review('src/auth.py', focus='security')
            print(result.format_for_matrix())
        """
        args = ['code-review', file_path]

        if focus:
            args.extend(['--focus', focus])

        return await self._run_command(args, timeout=timeout)

    async def refactor(
        self,
        pattern: str,
        target: str,
        timeout: Optional[int] = None
    ) -> ClaudeResponse:
        """
        Request refactoring from Claude Code.

        Args:
            pattern: Refactoring pattern (e.g., 'extract_method', 'simplify')
            target: Target file or line range (e.g., 'src/utils.py:42-67')
            timeout: Optional timeout override

        Returns:
            ClaudeResponse with refactoring suggestions

        Example:
            result = await bridge.refactor('extract_method', 'src/utils.py:42-67')
        """
        args = ['refactor', '--pattern', pattern, target]

        return await self._run_command(args, timeout=timeout)

    async def explain(
        self,
        file_path: str,
        question: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> ClaudeResponse:
        """
        Ask Claude Code to explain code.

        Args:
            file_path: Path to file
            question: Optional specific question about the code
            timeout: Optional timeout override

        Returns:
            ClaudeResponse with explanation

        Example:
            result = await bridge.explain('src/auth.py', 'How does JWT work?')
        """
        args = ['explain', file_path]

        if question:
            args.extend(['--question', question])

        return await self._run_command(args, timeout=timeout)

    async def implement(
        self,
        description: str,
        file_path: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> ClaudeResponse:
        """
        Ask Claude Code to implement functionality.

        Args:
            description: Description of what to implement
            file_path: Optional target file
            timeout: Optional timeout override

        Returns:
            ClaudeResponse with implementation

        Example:
            result = await bridge.implement(
                'Add JWT authentication middleware',
                file_path='src/auth/middleware.py'
            )
        """
        args = ['implement', description]

        if file_path:
            args.extend(['--file', file_path])

        return await self._run_command(args, timeout=timeout)

    async def generate_tests(
        self,
        file_path: str,
        test_framework: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> ClaudeResponse:
        """
        Generate unit tests for code.

        Args:
            file_path: Path to file to test
            test_framework: Optional framework (e.g., 'pytest', 'unittest')
            timeout: Optional timeout override

        Returns:
            ClaudeResponse with generated tests

        Example:
            result = await bridge.generate_tests('src/auth.py', 'pytest')
        """
        args = ['test-gen', file_path]

        if test_framework:
            args.extend(['--framework', test_framework])

        return await self._run_command(args, timeout=timeout)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Claude Code bridge health.

        Returns:
            Health status dictionary
        """
        # Try running a simple command
        result = await self._run_command(['--version'], timeout=5)

        return {
            'available': result.success,
            'claude_path': self.claude_path,
            'repo_path': str(self.repo_path),
            'version': result.output if result.success else None,
            'error': result.error
        }


# Convenience functions for quick usage

async def quick_review(file_path: str, repo_path: Optional[str] = None) -> str:
    """Quick code review (returns formatted string)"""
    bridge = ClaudeCodeBridge(repo_path=repo_path)
    result = await bridge.code_review(file_path)
    return result.format_for_matrix()


async def quick_explain(
    file_path: str,
    question: Optional[str] = None,
    repo_path: Optional[str] = None
) -> str:
    """Quick code explanation (returns formatted string)"""
    bridge = ClaudeCodeBridge(repo_path=repo_path)
    result = await bridge.explain(file_path, question)
    return result.format_for_matrix()


async def quick_refactor(
    pattern: str,
    target: str,
    repo_path: Optional[str] = None
) -> str:
    """Quick refactoring (returns formatted string)"""
    bridge = ClaudeCodeBridge(repo_path=repo_path)
    result = await bridge.refactor(pattern, target)
    return result.format_for_matrix()


# Example usage
if __name__ == "__main__":
    import sys

    async def main():
        """Test Claude Code bridge"""
        bridge = ClaudeCodeBridge()

        # Health check
        health = await bridge.health_check()
        print(f"Claude Code available: {health['available']}")
        if health['version']:
            print(f"Version: {health['version']}")

        # Example review (if file provided)
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            print(f"\nReviewing: {file_path}")
            result = await bridge.code_review(file_path)
            print(result.format_for_matrix())
        else:
            print("\nUsage: python claude_code_bridge.py <file_to_review>")

    asyncio.run(main())
