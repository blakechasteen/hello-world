"""
Command Grammar Parser for Elle Voice Assistant

Implements structured command parsing for Milestone 2: Command Mode.
Supports shortcuts, navigation, task operations, and command chaining.

Created: November 22, 2025
Part of: Voice UX Milestone 2
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CommandType(Enum):
    """Structured command types"""
    # Navigation
    BACK = "back"
    NEXT = "next"
    HOME = "home"
    UP = "up"

    # Thread operations
    THREAD_SWITCH = "thread_switch"
    THREAD_LIST = "thread_list"
    THREAD_CREATE = "thread_create"
    THREAD_DELETE = "thread_delete"

    # Task operations
    TASK_RUN = "task_run"
    TASK_STOP = "task_stop"
    TASK_PAUSE = "task_pause"
    TASK_RESUME = "task_resume"
    TASK_STATUS = "task_status"

    # Query operations
    ENTITY_LOOKUP = "entity_lookup"
    SEARCH = "search"

    # Fallback
    CONVERSATIONAL = "conversational"  # Falls back to Milestone 1


@dataclass
class StructuredCommand:
    """Parsed structured command"""
    command_type: CommandType
    parameters: dict
    raw_text: str
    confidence: float = 1.0  # Structured commands have high confidence

    def __str__(self):
        return f"<{self.command_type.value}: {self.parameters}>"


class CommandGrammarParser:
    """
    Parses structured voice commands with shortcuts and chaining.

    Features:
    - Thread shortcuts: "t3", "#3", "+ topic", "- 3"
    - Navigation: "back", "next", "home"
    - Task operations: "run task", "stop", "pause"
    - Entity lookup: "@entity", "find entity"
    - Command chaining: "t3; run analyze"
    - Fallback to conversational mode for unmatched
    """

    def __init__(self):
        """Initialize parser with regex patterns"""
        # Compile patterns for performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns"""
        # Navigation patterns
        self.nav_patterns = {
            CommandType.BACK: re.compile(r"^(back|b)$", re.IGNORECASE),
            CommandType.NEXT: re.compile(r"^(next|n)$", re.IGNORECASE),
            CommandType.HOME: re.compile(r"^(home|h)$", re.IGNORECASE),
            CommandType.UP: re.compile(r"^up$", re.IGNORECASE),
        }

        # Thread operation patterns
        self.thread_patterns = {
            # "thread 3", "t3", "#3", "baking" (single word only)
            CommandType.THREAD_SWITCH: [
                re.compile(r"^thread\s+(\d+)$", re.IGNORECASE),
                re.compile(r"^t(\d+)$", re.IGNORECASE),
                re.compile(r"^#(\d+)$"),
                # Fuzzy name match: single word or short phrase (max 3 words)
                re.compile(r"^([a-zA-Z][a-zA-Z0-9_]{0,20})$"),  # Single word: "baking", "biochar"
                re.compile(r"^([a-zA-Z][a-zA-Z0-9_\s]{0,30})$"),  # Short phrase: "baking bread" (max ~3 words)
            ],
            # "threads", "ts", "list"
            CommandType.THREAD_LIST: re.compile(r"^(threads|ts|list)$", re.IGNORECASE),
            # "new topic", "create topic", "+ topic"
            CommandType.THREAD_CREATE: [
                re.compile(r"^new\s+(.+)$", re.IGNORECASE),
                re.compile(r"^create\s+(.+)$", re.IGNORECASE),
                re.compile(r"^\+\s*(.+)$"),
            ],
            # "delete 3", "del 3", "- 3"
            CommandType.THREAD_DELETE: [
                re.compile(r"^delete\s+(\d+)$", re.IGNORECASE),
                re.compile(r"^del\s+(\d+)$", re.IGNORECASE),
                re.compile(r"^-\s*(\d+)$"),
            ],
        }

        # Task operation patterns
        self.task_patterns = {
            # "run task", "exec task", "> task"
            CommandType.TASK_RUN: [
                re.compile(r"^run\s+([a-zA-Z0-9_]+)$", re.IGNORECASE),
                re.compile(r"^exec\s+([a-zA-Z0-9_]+)$", re.IGNORECASE),
                re.compile(r"^>\s*([a-zA-Z0-9_]+)$"),
            ],
            # "stop", "s", "halt"
            CommandType.TASK_STOP: re.compile(r"^(stop|s|halt)$", re.IGNORECASE),
            # "pause", "p"
            CommandType.TASK_PAUSE: re.compile(r"^(pause|p)$", re.IGNORECASE),
            # "resume", "continue", "c"
            CommandType.TASK_RESUME: re.compile(r"^(resume|continue|c)$", re.IGNORECASE),
            # "status", "st", "?"
            CommandType.TASK_STATUS: re.compile(r"^(status|st|\?)$", re.IGNORECASE),
        }

        # Query operation patterns
        self.query_patterns = {
            # "@entity"
            CommandType.ENTITY_LOOKUP: re.compile(r"^@([a-zA-Z0-9_]+)$"),
            # "find entity"
            CommandType.SEARCH: re.compile(r"^find\s+(.+)$", re.IGNORECASE),
        }

    def parse(self, text: str) -> List[StructuredCommand]:
        """
        Parse command text into structured commands.

        Supports command chaining with semicolons:
        - "t3; run analyze" → [THREAD_SWITCH(3), TASK_RUN(analyze)]

        Args:
            text: Raw command text

        Returns:
            List of StructuredCommand objects (multiple if chained)
        """
        text = text.strip()

        if not text:
            return [self._conversational_fallback(text)]

        # Check for command chaining
        if ";" in text:
            return self._parse_chained(text)

        # Try to parse as single structured command
        command = self._parse_single(text)
        return [command]

    def _parse_chained(self, text: str) -> List[StructuredCommand]:
        """Parse chained commands separated by semicolons"""
        parts = [p.strip() for p in text.split(";")]
        commands = []

        for part in parts:
            if part:
                cmd = self._parse_single(part)
                commands.append(cmd)

        return commands if commands else [self._conversational_fallback(text)]

    def _parse_single(self, text: str) -> StructuredCommand:
        """Parse a single command (no chaining)"""
        # Try navigation patterns FIRST (most specific)
        for cmd_type, pattern in self.nav_patterns.items():
            if pattern.match(text):
                return StructuredCommand(
                    command_type=cmd_type,
                    parameters={},
                    raw_text=text
                )

        # Try task patterns BEFORE thread patterns (avoid greedy thread_switch)
        for cmd_type, patterns in self.task_patterns.items():
            # Handle list of patterns
            if isinstance(patterns, list):
                for pattern in patterns:
                    match = pattern.match(text)
                    if match:
                        return self._build_task_command(cmd_type, match, text)
            else:
                # Single pattern
                match = patterns.match(text)
                if match:
                    return self._build_task_command(cmd_type, match, text)

        # Try query patterns (entity lookup, search)
        for cmd_type, pattern in self.query_patterns.items():
            match = pattern.match(text)
            if match:
                return self._build_query_command(cmd_type, match, text)

        # Try thread patterns (after more specific patterns)
        # Order within thread patterns: LIST, DELETE, CREATE, then SWITCH (most general)
        ordered_thread_types = [
            CommandType.THREAD_LIST,
            CommandType.THREAD_DELETE,
            CommandType.THREAD_CREATE,
            CommandType.THREAD_SWITCH,  # Most general - try last
        ]

        for cmd_type in ordered_thread_types:
            if cmd_type not in self.thread_patterns:
                continue

            patterns = self.thread_patterns[cmd_type]

            # Handle list of patterns
            if isinstance(patterns, list):
                for pattern in patterns:
                    match = pattern.match(text)
                    if match:
                        return self._build_thread_command(cmd_type, match, text)
            else:
                # Single pattern
                match = patterns.match(text)
                if match:
                    return self._build_thread_command(cmd_type, match, text)

        # No structured pattern matched → Conversational fallback
        return self._conversational_fallback(text)

    def _build_thread_command(self, cmd_type: CommandType, match, raw_text: str) -> StructuredCommand:
        """Build thread operation command from regex match"""
        params = {}

        if cmd_type == CommandType.THREAD_SWITCH:
            # Could be thread ID (number) or thread name (string)
            value = match.group(1)
            if value.isdigit():
                params["thread_id"] = int(value)
            else:
                params["thread_name"] = value

        elif cmd_type == CommandType.THREAD_CREATE:
            params["topic"] = match.group(1).strip()

        elif cmd_type == CommandType.THREAD_DELETE:
            params["thread_id"] = int(match.group(1))

        return StructuredCommand(
            command_type=cmd_type,
            parameters=params,
            raw_text=raw_text
        )

    def _build_task_command(self, cmd_type: CommandType, match, raw_text: str) -> StructuredCommand:
        """Build task operation command from regex match"""
        params = {}

        if cmd_type == CommandType.TASK_RUN:
            params["task_name"] = match.group(1)

        # STOP, PAUSE, RESUME, STATUS have no parameters

        return StructuredCommand(
            command_type=cmd_type,
            parameters=params,
            raw_text=raw_text
        )

    def _build_query_command(self, cmd_type: CommandType, match, raw_text: str) -> StructuredCommand:
        """Build query command from regex match"""
        params = {}

        if cmd_type == CommandType.ENTITY_LOOKUP:
            params["entity"] = match.group(1)
        elif cmd_type == CommandType.SEARCH:
            params["query"] = match.group(1)

        return StructuredCommand(
            command_type=cmd_type,
            parameters=params,
            raw_text=raw_text
        )

    def _conversational_fallback(self, text: str) -> StructuredCommand:
        """Fallback to conversational mode for unmatched text"""
        return StructuredCommand(
            command_type=CommandType.CONVERSATIONAL,
            parameters={"text": text},
            raw_text=text,
            confidence=0.5  # Lower confidence for fallback
        )


# ============================================================================
# Demo & Testing
# ============================================================================

def demo_command_parser():
    """Demonstrate command parser capabilities"""
    print("=" * 70)
    print("COMMAND GRAMMAR PARSER DEMO")
    print("=" * 70)

    parser = CommandGrammarParser()

    test_commands = [
        # Navigation
        "back",
        "n",
        "home",

        # Thread shortcuts
        "thread 3",
        "t5",
        "#2",
        "baking",

        # Thread operations
        "threads",
        "ts",
        "+ greenhouse",
        "new biochar research",
        "delete 7",
        "- 4",

        # Task operations
        "run analyze_code",
        "> build_project",
        "stop",
        "pause",
        "c",
        "?",

        # Query operations
        "@Thompson",
        "find Matryoshka",

        # Command chaining
        "t3; run analyze",
        "new research; > search_papers; ?",

        # Conversational fallback
        "What's the weather like today?",
        "Tell me about quantum computing",
    ]

    print("\nParsing commands:")
    print("-" * 70)

    for text in test_commands:
        commands = parser.parse(text)

        print(f"\nInput:  '{text}'")

        if len(commands) == 1:
            cmd = commands[0]
            print(f"Parsed: {cmd.command_type.value}")
            if cmd.parameters:
                print(f"Params: {cmd.parameters}")
            if cmd.command_type == CommandType.CONVERSATIONAL:
                print(f"  -> Fallback to conversational mode")
        else:
            # Multiple commands (chained)
            print(f"Parsed: {len(commands)} chained commands")
            for i, cmd in enumerate(commands, 1):
                print(f"  {i}. {cmd.command_type.value} {cmd.parameters}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_command_parser()
