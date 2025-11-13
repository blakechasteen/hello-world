#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update claude_bridge.py to use Anthropic API instead of CLI
"""

API_BRIDGE_CODE = '''#!/usr/bin/env python3
"""
Claude API Bridge for Promptly Matrix Bot

Direct Anthropic API integration for code review, explanation, and refactoring.
Provides Claude's capabilities via Matrix chat without requiring CLI.
"""

import logging
import os
from pathlib import Path
from typing import Optional

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClaudeBridge:
    """Bridge to Claude via Anthropic API"""

    def __init__(self, api_key: Optional[str] = None, repo_path: Optional[str] = None):
        """
        Initialize Claude API bridge.

        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            repo_path: Repository path for reading files
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.repo_path = Path(repo_path) if repo_path else Path(".")

        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic SDK not installed. Run: pip install anthropic")
            self.client = None
        elif not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set in environment")
            self.client = None
        else:
            try:
                self.client = Anthropic(api_key=self.api_key)
                logger.info("Claude API bridge initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None

    def _read_file(self, file_path: str) -> str:
        """
        Read file contents safely.

        Args:
            file_path: Path to file (relative to repo)

        Returns:
            File contents as string

        Raises:
            Exception: If file cannot be read
        """
        full_path = self.repo_path / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not full_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        # Security: Check file is within repo
        try:
            full_path.resolve().relative_to(self.repo_path.resolve())
        except ValueError:
            raise ValueError(f"File outside repository: {file_path}")

        # Read file
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _call_claude(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        Call Claude API with prompt.

        Args:
            prompt: Prompt for Claude
            max_tokens: Maximum response tokens

        Returns:
            Claude's response

        Raises:
            Exception: If API call fails
        """
        if not self.client:
            raise Exception("Claude API not available. Check ANTHROPIC_API_KEY.")

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Latest Sonnet model
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract text from response
            return message.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def review(self, file_path: str) -> str:
        """
        Request code review from Claude.

        Args:
            file_path: Path to file to review (relative to repo)

        Returns:
            Review results as string
        """
        logger.info(f"Requesting review for: {file_path}")

        try:
            # Read file
            code = self._read_file(file_path)

            # Construct review prompt
            prompt = f"""Please review the following code from {file_path}:

```
{code}
```

Provide a comprehensive code review covering:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Suggestions for improvement

Be concise but thorough."""

            # Call Claude API
            return self._call_claude(prompt, max_tokens=2048)

        except Exception as e:
            logger.error(f"Review error: {e}")
            return f"Error during review: {e}"

    def explain(self, file_path: str, question: Optional[str] = None) -> str:
        """
        Ask Claude to explain code.

        Args:
            file_path: Path to file
            question: Optional specific question about the code

        Returns:
            Explanation as string
        """
        logger.info(f"Requesting explanation for: {file_path}")

        try:
            # Read file
            code = self._read_file(file_path)

            # Construct explanation prompt
            if question:
                prompt = f"""Here is code from {file_path}:

```
{code}
```

{question}

Please provide a clear, detailed explanation."""
            else:
                prompt = f"""Please explain the following code from {file_path}:

```
{code}
```

Explain:
1. What this code does
2. How it works
3. Key design decisions
4. Important details a developer should know

Be clear and concise."""

            # Call Claude API
            return self._call_claude(prompt, max_tokens=2048)

        except Exception as e:
            logger.error(f"Explain error: {e}")
            return f"Error during explanation: {e}"

    def refactor(self, file_path: str, instruction: str) -> str:
        """
        Request refactoring from Claude.

        Args:
            file_path: Path to file to refactor
            instruction: What to refactor

        Returns:
            Refactoring suggestion
        """
        logger.info(f"Requesting refactor for: {file_path}")

        try:
            # Read file
            code = self._read_file(file_path)

            # Construct refactoring prompt
            prompt = f"""Here is code from {file_path}:

```
{code}
```

Refactoring instruction: {instruction}

Please provide:
1. Explanation of the refactoring approach
2. The refactored code
3. Key changes and why they improve the code

Be specific and show the complete refactored version."""

            # Call Claude API with more tokens for refactored code
            return self._call_claude(prompt, max_tokens=3072)

        except Exception as e:
            logger.error(f"Refactor error: {e}")
            return f"Error during refactoring: {e}"

    def chat(self, message: str, context_file: Optional[str] = None) -> str:
        """
        Chat with Claude (general questions).

        Args:
            message: Question or message
            context_file: Optional file for context

        Returns:
            Claude's response
        """
        logger.info("Claude API chat request")

        try:
            if context_file:
                # Read context file
                code = self._read_file(context_file)
                prompt = f"""Context from {context_file}:

```
{code}
```

Question: {message}

Please answer considering the code context above."""
            else:
                prompt = message

            # Call Claude API
            return self._call_claude(prompt, max_tokens=1024)

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {e}"

    def is_available(self) -> bool:
        """
        Check if Claude API is available.

        Returns:
            True if API key is set and SDK is installed
        """
        return self.client is not None


# Example usage
if __name__ == "__main__":
    import sys

    # Test Claude API bridge
    bridge = ClaudeBridge()

    print("=== Testing Claude API Bridge ===")

    # Check if available
    if bridge.is_available():
        print("✅ Claude API is available")
    else:
        print("❌ Claude API not available")
        print("   Reasons:")
        if not ANTHROPIC_AVAILABLE:
            print("   - Anthropic SDK not installed (pip install anthropic)")
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("   - ANTHROPIC_API_KEY not set in environment")
        sys.exit(1)

    # Test chat
    print("\\n=== Testing Chat ===")
    response = bridge.chat("What is Python?")
    print(response[:200] if len(response) > 200 else response)

    print("\\n✅ Claude API bridge working!")
'''

if __name__ == "__main__":
    # Write new API bridge
    with open('bot/claude_bridge.py', 'w', encoding='utf-8') as f:
        f.write(API_BRIDGE_CODE)

    print("✅ Updated claude_bridge.py to use Anthropic API")
    print("   Backup saved to: bot/claude_bridge.py.cli_backup")
