#!/usr/bin/env python3
"""
Command Parser for Promptly Matrix Bot

Parses natural language commands from Matrix messages.
"""

import re
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CommandParser:
    """Parse natural language commands for Promptly bot"""

    # Command patterns (regex)
    # Support both @promptly/@promptlybot mentions and ! commands
    COMMANDS = {
        'help': r'(?:@promptly(?:bot)?\s+help|!help)',
        'optimize': r'(?:@promptly(?:bot)?\s+optimize|!optimize)',
        'run': r'(?:@promptly(?:bot)?\s+run\s+(\w+)\s+"([^"]+)"|!run\s+(\w+)\s+"([^"]+)")',
        'code-review': r'(?:@promptly(?:bot)?\s+code-review|!code-review)',
        'save': r'(?:@promptly(?:bot)?\s+save\s+(\w+)|!save\s+(\w+))',
        'list': r'(?:@promptly(?:bot)?\s+list|!list)',
        'schema': r'(?:@promptly(?:bot)?\s+schema|!schema)',
        'verify': r'(?:@promptly(?:bot)?\s+verify\s+"([^"]+)"|!verify\s+"([^"]+)")',
        'refine': r'(?:@promptly(?:bot)?\s+refine|!refine)',
        # Git commands
        'git-status': r'(?:@promptly(?:bot)?\s+git\s+status|!git\s+status)',
        'git-log': r'(?:@promptly(?:bot)?\s+git\s+log|!git\s+log)',
        'git-diff': r'(?:@promptly(?:bot)?\s+git\s+diff|!git\s+diff)',
        'git-branch': r'(?:@promptly(?:bot)?\s+git\s+branch|!git\s+branch)',
        'git-commit': r'(?:@promptly(?:bot)?\s+git\s+commit\s+"([^"]+)"|!git\s+commit\s+"([^"]+)")',
        'git-push': r'(?:@promptly(?:bot)?\s+git\s+push|!git\s+push)',
        'git-pull': r'(?:@promptly(?:bot)?\s+git\s+pull|!git\s+pull)',
    }

    def parse(self, message: str) -> Optional[Dict]:
        """
        Parse message into structured command.

        Args:
            message: Raw message text

        Returns:
            Command dict with 'type' and parameters, or None if no match
        """
        message_lower = message.lower()

        # Try each command pattern
        for cmd_type, pattern in self.COMMANDS.items():
            match = re.search(pattern, message_lower, re.IGNORECASE | re.DOTALL)
            if match:
                logger.info(f"Matched command: {cmd_type}")
                return self.extract_command(cmd_type, match.groups(), message)

        # No match
        logger.warning(f"Could not parse command: {message[:100]}")
        return None

    def extract_command(
        self,
        cmd_type: str,
        groups: tuple,
        full_message: str
    ) -> Dict:
        """
        Extract command parameters from regex groups.

        Args:
            cmd_type: Command type
            groups: Regex match groups
            full_message: Full message text

        Returns:
            Command dict
        """
        if cmd_type == 'help':
            return {'type': 'help'}

        elif cmd_type == 'optimize':
            # Extract task and examples from message
            return self.parse_optimize_command(full_message)

        elif cmd_type == 'run':
            return {
                'type': 'run',
                'workflow': groups[0],
                'input': groups[1]
            }

        elif cmd_type == 'code-review':
            # Extract code from message (after command)
            code, language = self.extract_code_block_with_lang(full_message)
            return {
                'type': 'code-review',
                'code': code,
                'language': language
            }

        elif cmd_type == 'save':
            return {
                'type': 'save',
                'name': groups[0]
            }

        elif cmd_type == 'list':
            return {'type': 'list'}

        elif cmd_type == 'schema':
            # Extract schema fields from message
            return self.parse_schema_command(full_message)

        elif cmd_type == 'verify':
            return {
                'type': 'verify',
                'statement': groups[0]
            }

        elif cmd_type == 'refine':
            return self.parse_refine_command(full_message)

        # Git commands
        elif cmd_type == 'git-status':
            return {'type': 'git-status'}

        elif cmd_type == 'git-log':
            return {'type': 'git-log'}

        elif cmd_type == 'git-diff':
            return {'type': 'git-diff'}

        elif cmd_type == 'git-branch':
            return {'type': 'git-branch'}

        elif cmd_type == 'git-commit':
            # First group or second group (depending on @promptly vs !)
            message = groups[0] if groups[0] else groups[1]
            return {
                'type': 'git-commit',
                'message': message
            }

        elif cmd_type == 'git-push':
            return {'type': 'git-push'}

        elif cmd_type == 'git-pull':
            return {'type': 'git-pull'}

        else:
            return {'type': 'unknown'}

    def parse_optimize_command(self, message: str) -> Dict:
        """
        Parse optimize command with task and examples.

        Expected format:
        @promptly optimize
        Task: [description]
        Examples: [JSON array]

        Args:
            message: Full message text

        Returns:
            Command dict with task and examples
        """
        # Extract task
        task_match = re.search(r'Task:\s*(.+?)(?=Examples:|$)', message, re.IGNORECASE | re.DOTALL)
        task = task_match.group(1).strip() if task_match else "Unknown task"

        # Extract examples
        examples_match = re.search(r'Examples:\s*(\[.+\])', message, re.IGNORECASE | re.DOTALL)
        examples = []

        if examples_match:
            try:
                examples_str = examples_match.group(1)
                examples = json.loads(examples_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse examples JSON: {e}")
                examples = []

        return {
            'type': 'optimize',
            'task': task,
            'examples': examples
        }

    def parse_schema_command(self, message: str) -> Dict:
        """
        Parse schema command with field definitions.

        Expected format:
        @promptly schema
        Fields:
        - name: string (required)
        - age: number
        - email: string (email format)

        Args:
            message: Full message text

        Returns:
            Command dict with schema fields
        """
        fields = []

        # Extract fields section
        fields_match = re.search(r'Fields:\s*(.+)', message, re.IGNORECASE | re.DOTALL)
        if fields_match:
            fields_text = fields_match.group(1)

            # Parse each field line
            for line in fields_text.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    # Parse: - name: string (required)
                    field_match = re.match(r'-\s*(\w+):\s*(\w+)\s*(?:\((.+)\))?', line)
                    if field_match:
                        fields.append({
                            'name': field_match.group(1),
                            'type': field_match.group(2),
                            'constraints': field_match.group(3) if field_match.group(3) else None
                        })

        return {
            'type': 'schema',
            'fields': fields
        }

    def parse_refine_command(self, message: str) -> Dict:
        """
        Parse refine command with strategy and parameters.

        Expected format:
        @promptly refine
        Strategy: elegance
        Max iterations: 3

        Args:
            message: Full message text

        Returns:
            Command dict with refine parameters
        """
        # Extract strategy
        strategy_match = re.search(r'Strategy:\s*(\w+)', message, re.IGNORECASE)
        strategy = strategy_match.group(1) if strategy_match else "auto"

        # Extract max iterations
        iterations_match = re.search(r'Max iterations:\s*(\d+)', message, re.IGNORECASE)
        max_iterations = int(iterations_match.group(1)) if iterations_match else 3

        return {
            'type': 'refine',
            'strategy': strategy,
            'max_iterations': max_iterations
        }

    def extract_code_block(self, message: str) -> str:
        """
        Extract code block from message.

        Supports:
        - Markdown code blocks (```code```)
        - Inline code (`code`)
        - Plain text after command

        Args:
            message: Full message text

        Returns:
            Extracted code
        """
        # Try markdown code block first
        code_block_match = re.search(r'```(?:\w+)?\n?(.+?)```', message, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try inline code
        inline_code_match = re.search(r'`(.+?)`', message, re.DOTALL)
        if inline_code_match:
            return inline_code_match.group(1).strip()

        # Fall back to text after "code-review"
        after_command = re.search(r'code-review\s+(.+)', message, re.IGNORECASE | re.DOTALL)
        if after_command:
            return after_command.group(1).strip()

        return ""

    def extract_code_block_with_lang(self, message: str) -> tuple:
        """
        Extract code block and language from message.

        Supports:
        - Markdown code blocks with language (```python code```)
        - Markdown code blocks without language (```code```)
        - Inline code (`code`)

        Args:
            message: Full message text

        Returns:
            Tuple of (code, language) where language is None if not specified
        """
        # Try markdown code block with language
        code_block_match = re.search(r'```(\w+)\n(.+?)```', message, re.DOTALL)
        if code_block_match:
            return code_block_match.group(2).strip(), code_block_match.group(1)

        # Try markdown code block without language
        code_block_match = re.search(r'```\n?(.+?)```', message, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip(), None

        # Try inline code
        inline_code_match = re.search(r'`(.+?)`', message, re.DOTALL)
        if inline_code_match:
            return inline_code_match.group(1).strip(), None

        # Fall back to text after "code-review"
        after_command = re.search(r'code-review\s+(.+)', message, re.IGNORECASE | re.DOTALL)
        if after_command:
            return after_command.group(1).strip(), None

        return "", None


# Unit tests
if __name__ == "__main__":
    parser = CommandParser()

    # Test help
    cmd = parser.parse("@promptly help")
    assert cmd['type'] == 'help'
    print("✅ help command parsed")

    # Test run
    cmd = parser.parse('@promptly run qa_workflow "What is Thompson Sampling?"')
    assert cmd['type'] == 'run'
    assert cmd['workflow'] == 'qa_workflow'
    assert 'Thompson' in cmd['input']
    print("✅ run command parsed")

    # Test optimize
    cmd = parser.parse("""
@promptly optimize
Task: Answer customer questions
Examples: [
  {"input": "How to reset?", "output": "Click..."},
  {"input": "Where is order?", "output": "Check..."}
]
""")
    assert cmd['type'] == 'optimize'
    assert 'customer' in cmd['task'].lower()
    assert len(cmd['examples']) == 2
    print("✅ optimize command parsed")

    # Test save
    cmd = parser.parse("@promptly save my_prompt")
    assert cmd['type'] == 'save'
    assert cmd['name'] == 'my_prompt'
    print("✅ save command parsed")

    # Test list
    cmd = parser.parse("@promptly list")
    assert cmd['type'] == 'list'
    print("✅ list command parsed")

    print("\n✅ All command parser tests passed!")
