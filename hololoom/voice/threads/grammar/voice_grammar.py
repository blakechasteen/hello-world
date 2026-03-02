"""
Voice Grammar Patterns

Natural language patterns for thread management, mode switching, and
command detection. Supports both casual and structured command syntax.

Pattern Categories:
- Thread operations (create, switch, list, summarize)
- Mode switching (conversational, command, streaming)
- Meta commands (help, status, settings)
- Conversational queries (everything else)

Examples:
    >>> grammar = VoiceGrammar()
    >>> intent = grammar.classify("start a new thread for planning")
    >>> print(intent.command_type)  # CommandType.THREAD_CREATE
    >>> print(intent.params)        # {'topic': 'planning'}
"""

import re
from enum import Enum, auto
from typing import Optional, Dict, List, Pattern
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Types of voice commands"""

    # Thread operations
    THREAD_CREATE = auto()
    THREAD_SWITCH = auto()
    THREAD_LIST = auto()
    THREAD_SUMMARIZE = auto()
    THREAD_BRANCH = auto()  # Milestone 1
    THREAD_MERGE = auto()   # Milestone 1

    # Mode switching
    MODE_SWITCH = auto()

    # Meta commands
    HELP = auto()
    STATUS = auto()

    # Not a command - conversational query
    CONVERSATIONAL = auto()


@dataclass
class CommandIntent:
    """
    Parsed command intent with parameters.

    Attributes:
        command_type: Type of command detected
        params: Extracted parameters (e.g., thread name, topic)
        confidence: Classification confidence (0.0-1.0)
        original_text: Original voice input
    """
    command_type: CommandType
    params: Dict[str, str]
    confidence: float
    original_text: str

    @property
    def is_conversational(self) -> bool:
        """Check if this is a conversational query (not a command)."""
        return self.command_type == CommandType.CONVERSATIONAL


class VoiceGrammar:
    """
    Natural language grammar for voice commands.

    Combines regex patterns with confidence scoring to classify voice input.
    """

    def __init__(self):
        """Initialize grammar patterns."""
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> Dict[CommandType, List[tuple[Pattern, float]]]:
        """
        Build regex patterns for each command type.

        Returns:
            Dict mapping command types to list of (pattern, confidence) tuples
        """
        patterns = {}

        # Thread creation patterns
        patterns[CommandType.THREAD_CREATE] = [
            (re.compile(r'^(start|create|open|begin) (a )?new thread (for|about) (?P<topic>.+)$', re.I), 0.95),
            (re.compile(r'^new thread:?\s*(?P<topic>.+)$', re.I), 0.90),
            (re.compile(r'^let\'?s talk about (?P<topic>.+)$', re.I), 0.85),
            (re.compile(r'^new conversation about (?P<topic>.+)$', re.I), 0.90),
            (re.compile(r'^thread (for|about) (?P<topic>.+)$', re.I), 0.80),
        ]

        # Thread switching patterns
        patterns[CommandType.THREAD_SWITCH] = [
            (re.compile(r'^(go back to|switch to|return to|resume) (the )?(?P<thread_name>.+) (thread|conversation)$', re.I), 0.95),
            (re.compile(r'^switch (to )?(?P<thread_name>.+)$', re.I), 0.85),
            (re.compile(r'^continue (the )?(?P<thread_name>.+)( thread| conversation)?$', re.I), 0.90),
            (re.compile(r'^back to (?P<thread_name>.+)$', re.I), 0.85),
        ]

        # Thread listing patterns
        patterns[CommandType.THREAD_LIST] = [
            (re.compile(r'^(list|show|display)( all| my)?( active)? threads?$', re.I), 0.95),
            (re.compile(r'^what threads (do i have|are active)(\?)?$', re.I), 0.90),
            (re.compile(r'^(show|list) (my )?conversations?$', re.I), 0.85),
            (re.compile(r'^threads?$', re.I), 0.80),
        ]

        # Thread summarization patterns
        patterns[CommandType.THREAD_SUMMARIZE] = [
            (re.compile(r'^summarize (the |this )?(?P<thread_name>.+) (thread|conversation)$', re.I), 0.95),
            (re.compile(r'^(what|tell me what) (have we|did we) (discussed|talked about) (in |about )?(?P<thread_name>.+)(\?)?$', re.I), 0.90),
            (re.compile(r'^summary of (?P<thread_name>.+)$', re.I), 0.90),
            (re.compile(r'^summarize( this)?$', re.I), 0.85),  # Current thread
        ]

        # Thread branching patterns (Milestone 1)
        patterns[CommandType.THREAD_BRANCH] = [
            (re.compile(r'^(fork|branch|split) this into (?P<branch_name>.+)$', re.I), 0.95),
            (re.compile(r'^(this (is|seems) important|i have an idea)[.,;:]? (create|start) (a )?new (thread|branch)( for)?( (?P<branch_name>.+))?$', re.I), 0.85),
            (re.compile(r'^new branch:?\s*(?P<branch_name>.+)$', re.I), 0.90),
        ]

        # Mode switching patterns
        patterns[CommandType.MODE_SWITCH] = [
            (re.compile(r'^(switch to|enter|activate) (?P<mode>conversational|command|streaming) mode$', re.I), 0.95),
            (re.compile(r'^(?P<mode>conversational|command|streaming) mode$', re.I), 0.85),
        ]

        # Help patterns
        patterns[CommandType.HELP] = [
            (re.compile(r'^(help|what can you do|show commands?)(\?)?$', re.I), 0.95),
            (re.compile(r'^(how do i|what\'?s the command (for|to)) (?P<topic>.+)(\?)?$', re.I), 0.90),
        ]

        # Status patterns
        patterns[CommandType.STATUS] = [
            (re.compile(r'^(status|what\'?s my status|where am i)(\?)?$', re.I), 0.95),
            (re.compile(r'^what (mode|thread) am i in(\?)?$', re.I), 0.90),
        ]

        return patterns

    def classify(self, text: str) -> CommandIntent:
        """
        Classify voice input into command intent.

        Args:
            text: Voice input text

        Returns:
            CommandIntent with classification and extracted parameters

        Example:
            >>> grammar = VoiceGrammar()
            >>> intent = grammar.classify("start a new thread for orchard planning")
            >>> print(intent.command_type)  # CommandType.THREAD_CREATE
            >>> print(intent.params)        # {'topic': 'orchard planning'}
        """
        # Clean input
        text_clean = text.strip()

        # Try to match against all patterns
        best_match = None
        best_confidence = 0.0
        best_command_type = CommandType.CONVERSATIONAL

        for command_type, pattern_list in self.patterns.items():
            for pattern, base_confidence in pattern_list:
                match = pattern.match(text_clean)
                if match:
                    # Extract parameters from named groups
                    params = match.groupdict()

                    # Calculate final confidence (pattern confidence)
                    confidence = base_confidence

                    # Keep best match
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_command_type = command_type
                        best_match = params

        # If no pattern matched, it's a conversational query
        if best_match is None:
            return CommandIntent(
                command_type=CommandType.CONVERSATIONAL,
                params={},
                confidence=1.0,
                original_text=text
            )

        # Return classified intent
        intent = CommandIntent(
            command_type=best_command_type,
            params=best_match,
            confidence=best_confidence,
            original_text=text
        )

        logger.debug(f"Classified: '{text}' → {intent.command_type.name} (confidence: {intent.confidence:.2f})")
        return intent

    def get_command_examples(self, command_type: CommandType) -> List[str]:
        """
        Get example phrases for a command type.

        Args:
            command_type: Type of command

        Returns:
            List of example phrases
        """
        examples = {
            CommandType.THREAD_CREATE: [
                "start a new thread for orchard planning",
                "new thread: biochar production",
                "let's talk about composting methods"
            ],
            CommandType.THREAD_SWITCH: [
                "switch to orchard planning thread",
                "go back to biochar production",
                "continue the composting conversation"
            ],
            CommandType.THREAD_LIST: [
                "list my threads",
                "show all active threads",
                "what threads do i have?"
            ],
            CommandType.THREAD_SUMMARIZE: [
                "summarize orchard planning thread",
                "what have we discussed in biochar production?",
                "summary of this conversation"
            ],
            CommandType.THREAD_BRANCH: [
                "fork this into a new idea about pollination",
                "this is important, create a new branch"
            ],
            CommandType.MODE_SWITCH: [
                "switch to command mode",
                "enter streaming mode",
                "conversational mode"
            ],
            CommandType.HELP: [
                "help",
                "what can you do?",
                "how do i create a thread?"
            ],
            CommandType.STATUS: [
                "status",
                "what mode am i in?",
                "where am i?"
            ]
        }
        return examples.get(command_type, [])

    def get_all_examples(self) -> Dict[str, List[str]]:
        """Get all command examples organized by category."""
        return {
            command_type.name: self.get_command_examples(command_type)
            for command_type in CommandType
            if command_type != CommandType.CONVERSATIONAL
        }
