#!/usr/bin/env python3
"""
Claude Code Command Handlers for Proto

Matrix chat command handlers that integrate Claude Code CLI bridge.
Provides conversational interface to code review, refactoring, and explanation.

Commands:
    @proto code-review <file> [focus]
    @proto refactor <pattern> <target>
    @proto explain <file> [question]
    @proto implement <description>
    @proto test-gen <file> [framework]

Built for Proto - Conversational Intelligence Hub
"""

import logging
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Import claude_code_bridge
try:
    from .claude_code_bridge import ClaudeCodeBridge, ClaudeResponse
    CLAUDE_BRIDGE_AVAILABLE = True
except ImportError:
    CLAUDE_BRIDGE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("claude_code_bridge not available")


@dataclass
class CommandContext:
    """Context for command execution"""
    user_id: str
    room_id: str
    message: str
    repo_path: Optional[str] = None


class ClaudeCodeCommands:
    """
    Matrix command handlers for Claude Code integration.

    Provides conversational interface to Claude Code CLI:
    - @proto code-review src/auth.py
    - @proto code-review src/auth.py security
    - @proto refactor extract_method src/utils.py:42-67
    - @proto explain src/auth.py How does JWT work?
    - @proto implement Add JWT auth middleware
    - @proto test-gen src/auth.py pytest

    Usage:
        commands = ClaudeCodeCommands(repo_path='/path/to/repo')

        # In Matrix bot message handler:
        if message.startswith('@proto code-review'):
            result = await commands.handle_code_review(context)
            await send_message(room, result)
    """

    def __init__(
        self,
        repo_path: Optional[str] = None,
        claude_path: str = "claude",
        default_timeout: int = 300
    ):
        """
        Initialize Claude Code command handlers.

        Args:
            repo_path: Repository path for file operations
            claude_path: Path to claude executable
            default_timeout: Default timeout for commands
        """
        self.repo_path = repo_path
        self.claude_path = claude_path
        self.default_timeout = default_timeout

        # Initialize bridge
        if CLAUDE_BRIDGE_AVAILABLE:
            self.bridge = ClaudeCodeBridge(
                claude_path=claude_path,
                repo_path=repo_path,
                default_timeout=default_timeout
            )
        else:
            self.bridge = None

    def _check_availability(self) -> Tuple[bool, str]:
        """
        Check if Claude Code bridge is available.

        Returns:
            (available, error_message) tuple
        """
        if not CLAUDE_BRIDGE_AVAILABLE:
            return False, "❌ Claude Code bridge not available. Missing dependencies."

        if not self.bridge:
            return False, "❌ Claude Code bridge not initialized."

        return True, ""

    async def handle_code_review(self, context: CommandContext) -> str:
        """
        Handle @proto code-review command.

        Syntax:
            @proto code-review <file>
            @proto code-review <file> <focus>

        Args:
            context: Command context

        Returns:
            Formatted response for Matrix
        """
        # Check availability
        available, error = self._check_availability()
        if not available:
            return error

        # Parse command
        # Format: @proto code-review src/auth.py [security|performance|style]
        pattern = r'@proto\s+code-review\s+(\S+)(?:\s+(\S+))?'
        match = re.search(pattern, context.message)

        if not match:
            return (
                "❌ **Invalid syntax**\n\n"
                "Usage:\n"
                "• `@proto code-review <file>`\n"
                "• `@proto code-review <file> <focus>`\n\n"
                "Focus options: security, performance, style, readability\n\n"
                "Example:\n"
                "`@proto code-review src/auth.py security`"
            )

        file_path = match.group(1)
        focus = match.group(2)

        # Show progress message
        progress_msg = f"🤖 **Requesting Claude Code review...**\n\nFile: `{file_path}`"
        if focus:
            progress_msg += f"\nFocus: `{focus}`"
        progress_msg += "\n\n_This may take 1-3 minutes..._"

        # Execute review (caller should send progress_msg first)
        result = await self.bridge.code_review(file_path, focus=focus)

        return result.format_for_matrix()

    async def handle_refactor(self, context: CommandContext) -> str:
        """
        Handle @proto refactor command.

        Syntax:
            @proto refactor <pattern> <target>

        Args:
            context: Command context

        Returns:
            Formatted response for Matrix
        """
        available, error = self._check_availability()
        if not available:
            return error

        # Parse command
        # Format: @proto refactor extract_method src/utils.py:42-67
        pattern = r'@proto\s+refactor\s+(\S+)\s+(\S+)'
        match = re.search(pattern, context.message)

        if not match:
            return (
                "❌ **Invalid syntax**\n\n"
                "Usage:\n"
                "`@proto refactor <pattern> <target>`\n\n"
                "Patterns:\n"
                "• extract_method - Extract method from code block\n"
                "• simplify - Simplify complex logic\n"
                "• rename - Suggest better names\n"
                "• optimize - Performance improvements\n\n"
                "Example:\n"
                "`@proto refactor extract_method src/utils.py:42-67`"
            )

        refactor_pattern = match.group(1)
        target = match.group(2)

        # Execute refactor
        result = await self.bridge.refactor(refactor_pattern, target)

        return result.format_for_matrix()

    async def handle_explain(self, context: CommandContext) -> str:
        """
        Handle @proto explain command.

        Syntax:
            @proto explain <file>
            @proto explain <file> <question>

        Args:
            context: Command context

        Returns:
            Formatted response for Matrix
        """
        available, error = self._check_availability()
        if not available:
            return error

        # Parse command
        # Format: @proto explain src/auth.py [optional question...]
        pattern = r'@proto\s+explain\s+(\S+)(?:\s+(.+))?'
        match = re.search(pattern, context.message)

        if not match:
            return (
                "❌ **Invalid syntax**\n\n"
                "Usage:\n"
                "• `@proto explain <file>`\n"
                "• `@proto explain <file> <question>`\n\n"
                "Examples:\n"
                "• `@proto explain src/auth.py`\n"
                "• `@proto explain src/auth.py How does JWT validation work?`"
            )

        file_path = match.group(1)
        question = match.group(2)

        # Execute explain
        result = await self.bridge.explain(file_path, question=question)

        return result.format_for_matrix()

    async def handle_implement(self, context: CommandContext) -> str:
        """
        Handle @proto implement command.

        Syntax:
            @proto implement <description>

        Args:
            context: Command context

        Returns:
            Formatted response for Matrix
        """
        available, error = self._check_availability()
        if not available:
            return error

        # Parse command
        # Format: @proto implement Add JWT authentication middleware
        pattern = r'@proto\s+implement\s+(.+)'
        match = re.search(pattern, context.message)

        if not match:
            return (
                "❌ **Invalid syntax**\n\n"
                "Usage:\n"
                "`@proto implement <description>`\n\n"
                "Examples:\n"
                "• `@proto implement Add JWT authentication middleware`\n"
                "• `@proto implement Create user registration endpoint`\n"
                "• `@proto implement Add rate limiting to API`"
            )

        description = match.group(1)

        # Execute implement
        result = await self.bridge.implement(description)

        return result.format_for_matrix()

    async def handle_test_gen(self, context: CommandContext) -> str:
        """
        Handle @proto test-gen command.

        Syntax:
            @proto test-gen <file>
            @proto test-gen <file> <framework>

        Args:
            context: Command context

        Returns:
            Formatted response for Matrix
        """
        available, error = self._check_availability()
        if not available:
            return error

        # Parse command
        # Format: @proto test-gen src/auth.py [pytest|unittest]
        pattern = r'@proto\s+test-gen\s+(\S+)(?:\s+(\S+))?'
        match = re.search(pattern, context.message)

        if not match:
            return (
                "❌ **Invalid syntax**\n\n"
                "Usage:\n"
                "• `@proto test-gen <file>`\n"
                "• `@proto test-gen <file> <framework>`\n\n"
                "Frameworks: pytest, unittest, jest, mocha\n\n"
                "Example:\n"
                "`@proto test-gen src/auth.py pytest`"
            )

        file_path = match.group(1)
        framework = match.group(2)

        # Execute test generation
        result = await self.bridge.generate_tests(file_path, test_framework=framework)

        return result.format_for_matrix()

    async def handle_claude_help(self) -> str:
        """
        Show Claude Code help message.

        Returns:
            Help text for Matrix
        """
        return """🤖 **Claude Code Integration**

Available commands:

**Code Review**
`@proto code-review <file> [focus]`
• Review code for quality, security, performance
• Focus: security, performance, style, readability

**Refactoring**
`@proto refactor <pattern> <target>`
• extract_method - Extract methods from code blocks
• simplify - Simplify complex logic
• rename - Suggest better variable names
• optimize - Performance improvements

**Explanation**
`@proto explain <file> [question]`
• Explain how code works
• Ask specific questions about implementation

**Implementation**
`@proto implement <description>`
• Generate code from description
• Create new features, endpoints, utilities

**Test Generation**
`@proto test-gen <file> [framework]`
• Generate unit tests for existing code
• Frameworks: pytest, unittest, jest, mocha

**Examples**
```
@proto code-review src/auth.py security
@proto refactor extract_method src/utils.py:42-67
@proto explain src/auth.py How does JWT work?
@proto implement Add rate limiting middleware
@proto test-gen src/auth.py pytest
```

**Note**: Commands may take 1-5 minutes depending on complexity."""

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Claude Code integration health.

        Returns:
            Health status dictionary
        """
        if not self.bridge:
            return {
                'available': False,
                'error': 'Bridge not initialized'
            }

        return await self.bridge.health_check()


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_commands():
        """Test command handlers"""
        commands = ClaudeCodeCommands()

        # Test help
        help_text = await commands.handle_claude_help()
        print(help_text)
        print("\n" + "="*60 + "\n")

        # Test health check
        health = await commands.health_check()
        print(f"Health check: {health}")

    asyncio.run(test_commands())
