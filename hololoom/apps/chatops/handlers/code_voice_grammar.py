"""
Code Voice Grammar Patterns

Natural language patterns for voice-to-code operations including
code generation, explanation, editing, and review.

Pattern Categories:
- Code generation (write, create, generate functions/classes)
- Code explanation (explain, describe, what does)
- Code editing (refactor, fix, add, modify)
- Code review (review, check, any issues)

Examples:
    >>> grammar = CodeVoiceGrammar()
    >>> intent = grammar.classify("write a function that sorts users by age")
    >>> print(intent.code_intent)  # CodeIntent.GENERATE
    >>> print(intent.params)       # {'request': 'sorts users by age'}

Author: Claude Code
Date: December 2025
"""

import re
from enum import Enum, auto
from typing import Optional, Dict, List, Pattern
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CodeIntent(Enum):
    """Types of code-related voice commands"""

    # Code operations
    GENERATE = auto()    # Write new code
    EXPLAIN = auto()     # Explain existing code
    EDIT = auto()        # Modify/refactor code
    REVIEW = auto()      # Review code for issues

    # Context operations
    CONTEXT_SET = auto()     # Set code context ("use this code")
    CONTEXT_CLEAR = auto()   # Clear context

    # Language specification
    LANGUAGE = auto()    # Specify language ("in Python", "use TypeScript")

    # Not code-related - general query
    QUERY = auto()


@dataclass
class CodeCommandIntent:
    """
    Parsed code command intent with parameters.

    Attributes:
        code_intent: Type of code command detected
        params: Extracted parameters (request, language, etc.)
        confidence: Classification confidence (0.0-1.0)
        original_text: Original voice input
    """
    code_intent: CodeIntent
    params: Dict[str, str]
    confidence: float
    original_text: str

    @property
    def is_code_command(self) -> bool:
        """Check if this is a code-related command."""
        return self.code_intent != CodeIntent.QUERY

    @property
    def request(self) -> Optional[str]:
        """Get the main request text."""
        return self.params.get('request')

    @property
    def language(self) -> Optional[str]:
        """Get specified programming language if any."""
        return self.params.get('language')


class CodeVoiceGrammar:
    """
    Natural language grammar for code-related voice commands.

    Combines regex patterns with confidence scoring to classify voice input
    for code generation, explanation, editing, and review.
    """

    # Common programming languages for detection
    LANGUAGES = [
        'python', 'javascript', 'typescript', 'java', 'go', 'rust',
        'c', 'cpp', 'c\\+\\+', 'csharp', 'c#', 'ruby', 'php',
        'swift', 'kotlin', 'scala', 'haskell', 'elixir', 'clojure'
    ]

    def __init__(self):
        """Initialize grammar patterns."""
        self.patterns = self._build_patterns()
        self.language_pattern = self._build_language_pattern()

    def _build_language_pattern(self) -> Pattern:
        """Build pattern to detect programming language mentions."""
        lang_group = '|'.join(self.LANGUAGES)
        return re.compile(
            rf'\b(in|using|with)\s+(?P<language>{lang_group})\b',
            re.I
        )

    def _build_patterns(self) -> Dict[CodeIntent, List[tuple[Pattern, float]]]:
        """
        Build regex patterns for each code intent type.

        Returns:
            Dict mapping code intents to list of (pattern, confidence) tuples
        """
        patterns = {}

        # =================================================================
        # CODE GENERATION patterns
        # =================================================================
        patterns[CodeIntent.GENERATE] = [
            # Direct generation requests
            (re.compile(r'^(write|create|generate|make|build) (me )?(a |an |the )?(?P<type>function|method|class|script|program|module|component|api|endpoint) (that |to |for |which )?(?P<request>.+)$', re.I), 0.95),

            # "code for/to/that" patterns
            (re.compile(r'^(write |generate |create )?code (for|to|that) (?P<request>.+)$', re.I), 0.90),

            # "give me" / "show me" patterns
            (re.compile(r'^(give me|show me|i need) (a |an |the |some )?(code|function|script|program) (for|to|that) (?P<request>.+)$', re.I), 0.90),

            # Simple imperative patterns
            (re.compile(r'^(implement|code|program) (?P<request>.+)$', re.I), 0.85),

            # "how do I" as generation (can also be explanation)
            (re.compile(r'^how (do i|would i|can i|to) (write|create|implement|code) (?P<request>.+)$', re.I), 0.80),

            # Language-specific patterns
            (re.compile(r'^(?P<request>.+) in (python|javascript|typescript|go|rust|java)$', re.I), 0.75),
        ]

        # =================================================================
        # CODE EXPLANATION patterns
        # =================================================================
        patterns[CodeIntent.EXPLAIN] = [
            # Direct explanation requests
            (re.compile(r'^(explain|describe|walk me through) (this|the|that) (code|function|class|method|script|program)$', re.I), 0.95),

            # "what does" patterns
            (re.compile(r'^what does (this|the|that) (code|function|class|method)( do)?(\?)?$', re.I), 0.95),

            # "how does" patterns
            (re.compile(r'^how does (this|it|the code|that) work(\?)?$', re.I), 0.90),

            # "tell me about" patterns
            (re.compile(r'^(tell me about|what\'?s|what is) (this|the|that) (code|function|class)(\?)?$', re.I), 0.90),

            # Short forms
            (re.compile(r'^explain (this|it)$', re.I), 0.85),
            (re.compile(r'^what (is|does) this(\?)?$', re.I), 0.85),

            # "break down" patterns
            (re.compile(r'^(break down|analyze|parse) (this|the) (code|function|class)$', re.I), 0.90),
        ]

        # =================================================================
        # CODE EDITING patterns
        # =================================================================
        patterns[CodeIntent.EDIT] = [
            # Refactoring patterns
            (re.compile(r'^(refactor|rewrite|restructure|reorganize) (this|the|that) (code|function|class|method)?( to (?P<request>.+))?$', re.I), 0.95),

            # Flexible refactor patterns (voice-friendly)
            (re.compile(r'^(refactor|rewrite) (this|it) to (?P<request>.+)$', re.I), 0.90),

            # Fix patterns
            (re.compile(r'^(fix|correct|repair|debug) (this|the|that)( bug| issue| error| problem)?( in (?P<location>.+))?$', re.I), 0.95),

            # Add patterns
            (re.compile(r'^add (?P<request>error handling|logging|type hints|types|comments|documentation|tests|validation) (to (this|the|that))?$', re.I), 0.95),

            # Modify patterns
            (re.compile(r'^(change|modify|update|alter) (this|the|that) (to )?(?P<request>.+)$', re.I), 0.90),

            # Make it patterns
            (re.compile(r'^make (this|it) (?P<request>async|synchronous|faster|more efficient|simpler|cleaner|more readable)$', re.I), 0.90),

            # Convert patterns
            (re.compile(r'^convert (this|it) to (?P<request>.+)$', re.I), 0.90),

            # Rename patterns
            (re.compile(r'^rename (this|the) (?P<type>function|variable|class|method) (to )?(?P<new_name>\w+)$', re.I), 0.90),

            # Optimization patterns
            (re.compile(r'^(optimize|improve|enhance) (this|the) (code|function|performance)$', re.I), 0.85),
        ]

        # =================================================================
        # CODE REVIEW patterns
        # =================================================================
        patterns[CodeIntent.REVIEW] = [
            # Direct review requests
            (re.compile(r'^(review|check|examine|inspect) (this|the|that|my) (code|function|class|implementation)$', re.I), 0.95),

            # "check for issues" patterns (voice-friendly)
            (re.compile(r'^(check|review) (this|the|my) (code|function|class) (for|about) (issues|problems|bugs|errors)$', re.I), 0.95),

            # "any issues" patterns
            (re.compile(r'^(are there )?any (issues|problems|bugs|errors) (with|in) (this|the|that)( code)?(\?)?$', re.I), 0.95),

            # "is this" patterns
            (re.compile(r'^is (this|the|my) (code|implementation|approach) (correct|good|ok|okay|right)(\?)?$', re.I), 0.90),

            # "what's wrong" patterns
            (re.compile(r'^what\'?s (wrong|the issue|the problem) (with|in) (this|the)(\?)?$', re.I), 0.90),

            # Critique patterns
            (re.compile(r'^(critique|evaluate|assess) (this|my|the) (code|implementation)$', re.I), 0.85),

            # "can you" patterns
            (re.compile(r'^can you (review|check|look at) (this|my) (code|function|implementation)(\?)?$', re.I), 0.85),
        ]

        # =================================================================
        # CONTEXT patterns
        # =================================================================
        patterns[CodeIntent.CONTEXT_SET] = [
            (re.compile(r'^(use|set|consider) (this|the following) (as |for )?(context|code|reference)$', re.I), 0.95),
            (re.compile(r'^(here\'?s|this is) (the|my) (code|context)$', re.I), 0.90),
            (re.compile(r'^work with (this|the following) (code)?$', re.I), 0.85),
        ]

        patterns[CodeIntent.CONTEXT_CLEAR] = [
            (re.compile(r'^(clear|reset|forget) (the )?(context|code)$', re.I), 0.95),
            (re.compile(r'^start (fresh|over|new)$', re.I), 0.85),
        ]

        return patterns

    def classify(self, text: str) -> CodeCommandIntent:
        """
        Classify voice input into code command intent.

        Args:
            text: Voice input text

        Returns:
            CodeCommandIntent with classification and extracted parameters

        Example:
            >>> grammar = CodeVoiceGrammar()
            >>> intent = grammar.classify("write a function that calculates fibonacci")
            >>> print(intent.code_intent)  # CodeIntent.GENERATE
            >>> print(intent.params)       # {'request': 'calculates fibonacci', 'type': 'function'}
        """
        # Clean input
        text_clean = text.strip()

        # Extract language if mentioned
        detected_language = None
        lang_match = self.language_pattern.search(text_clean)
        if lang_match:
            detected_language = lang_match.group('language').lower()
            # Normalize language names
            if detected_language in ('c++', 'cpp'):
                detected_language = 'cpp'
            elif detected_language in ('c#', 'csharp'):
                detected_language = 'csharp'

        # Try to match against all patterns
        best_match = None
        best_confidence = 0.0
        best_intent = CodeIntent.QUERY

        for code_intent, pattern_list in self.patterns.items():
            for pattern, base_confidence in pattern_list:
                match = pattern.match(text_clean)
                if match:
                    # Extract parameters from named groups
                    params = {k: v for k, v in match.groupdict().items() if v}

                    # Calculate final confidence
                    confidence = base_confidence

                    # Keep best match
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = code_intent
                        best_match = params

        # Build final params
        final_params = best_match if best_match else {}
        if detected_language:
            final_params['language'] = detected_language

        # If no pattern matched, it's a general query
        if best_match is None and detected_language is None:
            return CodeCommandIntent(
                code_intent=CodeIntent.QUERY,
                params={'request': text_clean},
                confidence=1.0,
                original_text=text
            )

        # Return classified intent
        intent = CodeCommandIntent(
            code_intent=best_intent,
            params=final_params,
            confidence=best_confidence,
            original_text=text
        )

        logger.debug(f"Code classified: '{text}' → {intent.code_intent.name} (confidence: {intent.confidence:.2f})")
        return intent

    def get_examples(self, code_intent: CodeIntent) -> List[str]:
        """
        Get example phrases for a code intent type.

        Args:
            code_intent: Type of code command

        Returns:
            List of example phrases
        """
        examples = {
            CodeIntent.GENERATE: [
                "write a function that calculates fibonacci",
                "create a class for user authentication",
                "generate code for sorting a list",
                "make a script that downloads images",
                "code for parsing JSON files"
            ],
            CodeIntent.EXPLAIN: [
                "explain this code",
                "what does this function do?",
                "how does this work?",
                "walk me through this class"
            ],
            CodeIntent.EDIT: [
                "add error handling to this",
                "refactor this to use async",
                "fix the bug in this function",
                "add type hints",
                "make this more efficient"
            ],
            CodeIntent.REVIEW: [
                "review this code",
                "any issues with this?",
                "is this implementation correct?",
                "what's wrong with this?"
            ],
            CodeIntent.CONTEXT_SET: [
                "use this as context",
                "here's my code",
                "work with this code"
            ],
            CodeIntent.CONTEXT_CLEAR: [
                "clear context",
                "start fresh",
                "forget the code"
            ]
        }
        return examples.get(code_intent, [])
