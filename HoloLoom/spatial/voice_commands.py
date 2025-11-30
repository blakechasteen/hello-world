"""
HoloLoom Phase 4: Voice Command System
November 2025

Voice interaction for spatial knowledge graph,
supporting natural language commands in XR.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Pattern
import json
import re


class CommandCategory(Enum):
    """Categories of voice commands."""
    NAVIGATION = "navigation"    # Move around, focus
    MANIPULATION = "manipulation"  # Grab, move, scale
    QUERY = "query"              # Search, ask questions
    SYSTEM = "system"            # Settings, mode changes
    SELECTION = "selection"      # Select, deselect
    VIEW = "view"                # Zoom, rotate view
    MEMORY = "memory"            # Create, delete knowledge


class CommandState(Enum):
    """State of command processing."""
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VoiceCommand:
    """A recognized voice command."""
    command_id: str
    text: str                                # Raw transcript
    intent: str                              # Parsed intent
    category: CommandCategory
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    state: CommandState = CommandState.PROCESSING
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "command_id": self.command_id,
            "text": self.text,
            "intent": self.intent,
            "category": self.category.value,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.value,
            "result": str(self.result) if self.result else None,
            "error": self.error
        }


@dataclass
class CommandPattern:
    """Pattern for matching voice commands."""
    pattern: str                             # Regex pattern
    intent: str                              # Intent name
    category: CommandCategory
    handler: Callable[[VoiceCommand], Any]
    description: str = ""
    examples: List[str] = field(default_factory=list)
    param_extractors: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def match(self, text: str) -> Optional[Dict[str, str]]:
        """Match text against pattern, returning extracted parameters."""
        m = self._compiled.match(text.strip())
        if m:
            return m.groupdict()
        return None


class WakeWordDetector:
    """Detects wake words to activate voice commands."""

    DEFAULT_WAKE_WORDS = ["hey hololoom", "ok loom", "loom", "hey mind"]

    def __init__(self, wake_words: List[str] = None):
        self.wake_words = wake_words or self.DEFAULT_WAKE_WORDS
        self.is_active = False
        self.last_activation: Optional[datetime] = None
        self.activation_timeout = 10.0  # Seconds before going back to sleep

    def check(self, text: str) -> bool:
        """Check if text contains a wake word."""
        text_lower = text.lower().strip()

        for wake_word in self.wake_words:
            if text_lower.startswith(wake_word):
                self.is_active = True
                self.last_activation = datetime.now()
                return True

        return False

    def strip_wake_word(self, text: str) -> str:
        """Remove wake word from beginning of text."""
        text_lower = text.lower().strip()

        for wake_word in self.wake_words:
            if text_lower.startswith(wake_word):
                return text[len(wake_word):].strip()

        return text

    def check_timeout(self) -> bool:
        """Check if activation has timed out."""
        if not self.is_active or not self.last_activation:
            return True

        elapsed = (datetime.now() - self.last_activation).total_seconds()
        if elapsed > self.activation_timeout:
            self.is_active = False
            return True

        return False


class VoiceCommandSystem:
    """
    Processes voice commands for XR knowledge interaction.

    Features:
    - Wake word detection
    - Pattern-based command parsing
    - Natural language understanding
    - Command history
    - Customizable command vocabulary
    """

    def __init__(self, wake_words: List[str] = None):
        self.wake_detector = WakeWordDetector(wake_words)
        self.patterns: List[CommandPattern] = []
        self.history: List[VoiceCommand] = []
        self.max_history = 100
        self._command_counter = 0

        # Callbacks
        self.on_command: Optional[Callable[[VoiceCommand], None]] = None
        self.on_wake: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[VoiceCommand, str], None]] = None

        # Register default patterns
        self._register_defaults()

    def _generate_command_id(self) -> str:
        self._command_counter += 1
        return f"cmd_{self._command_counter:06d}"

    # === Pattern Registration ===

    def register_pattern(
        self,
        pattern: str,
        intent: str,
        category: CommandCategory,
        handler: Callable[[VoiceCommand], Any],
        description: str = "",
        examples: List[str] = None,
        param_extractors: Dict[str, str] = None
    ):
        """
        Register a voice command pattern.

        Args:
            pattern: Regex pattern with named groups for parameters
            intent: Intent name (e.g., "navigate_to")
            category: Command category
            handler: Function to execute
            description: Human description
            examples: Example phrases
            param_extractors: Parameter extraction patterns
        """
        cmd_pattern = CommandPattern(
            pattern=pattern,
            intent=intent,
            category=category,
            handler=handler,
            description=description,
            examples=examples or [],
            param_extractors=param_extractors or {}
        )
        self.patterns.append(cmd_pattern)

    def _register_defaults(self):
        """Register default command patterns."""

        # Navigation commands
        self.register_pattern(
            r"(?:go to|navigate to|show me|find)\s+(?P<target>.+)",
            "navigate_to",
            CommandCategory.NAVIGATION,
            self._handle_navigate,
            "Navigate to a concept",
            ["go to Thompson Sampling", "show me neural networks", "find memory"]
        )

        self.register_pattern(
            r"(?:zoom in|closer)",
            "zoom_in",
            CommandCategory.VIEW,
            self._handle_zoom_in,
            "Zoom in on current view"
        )

        self.register_pattern(
            r"(?:zoom out|farther|back)",
            "zoom_out",
            CommandCategory.VIEW,
            self._handle_zoom_out,
            "Zoom out from current view"
        )

        self.register_pattern(
            r"(?:rotate|spin|turn)\s*(?P<direction>left|right)?",
            "rotate",
            CommandCategory.VIEW,
            self._handle_rotate,
            "Rotate the view",
            ["rotate left", "spin right", "turn"]
        )

        # Selection commands
        self.register_pattern(
            r"(?:select|pick|grab)\s+(?P<target>.+)",
            "select",
            CommandCategory.SELECTION,
            self._handle_select,
            "Select a node",
            ["select Thompson Sampling", "grab this one"]
        )

        self.register_pattern(
            r"(?:deselect|drop|release)\s*(?:all)?",
            "deselect",
            CommandCategory.SELECTION,
            self._handle_deselect,
            "Deselect current selection"
        )

        # Query commands
        self.register_pattern(
            r"(?:what is|explain|tell me about|describe)\s+(?P<topic>.+)",
            "query_explain",
            CommandCategory.QUERY,
            self._handle_explain,
            "Ask about a topic",
            ["what is Thompson Sampling", "explain neural networks"]
        )

        self.register_pattern(
            r"(?:how (?:does|do|is)|why)\s+(?P<question>.+)",
            "query_how",
            CommandCategory.QUERY,
            self._handle_question,
            "Ask how/why questions"
        )

        self.register_pattern(
            r"(?:search|look for|find)\s+(?P<query>.+)",
            "search",
            CommandCategory.QUERY,
            self._handle_search,
            "Search the knowledge graph"
        )

        # Manipulation commands
        self.register_pattern(
            r"(?:move|place|put)\s+(?:this|it)?\s*(?:to|at|near)?\s*(?P<location>.+)?",
            "move",
            CommandCategory.MANIPULATION,
            self._handle_move,
            "Move selected object"
        )

        self.register_pattern(
            r"(?:make (?:it )?(?P<size>bigger|smaller)|scale (?P<direction>up|down))",
            "scale",
            CommandCategory.MANIPULATION,
            self._handle_scale,
            "Scale selected object"
        )

        self.register_pattern(
            r"(?:connect|link)\s+(?P<source>.+?)\s+(?:to|with)\s+(?P<target>.+)",
            "connect",
            CommandCategory.MANIPULATION,
            self._handle_connect,
            "Connect two nodes"
        )

        # Memory commands
        self.register_pattern(
            r"(?:create|add|new)\s+(?:a\s+)?(?:node|concept|idea)\s*(?:called|named)?\s*(?P<name>.+)?",
            "create_node",
            CommandCategory.MEMORY,
            self._handle_create,
            "Create a new knowledge node",
            ["create a node called Machine Learning", "add new concept"]
        )

        self.register_pattern(
            r"(?:delete|remove|forget)\s+(?P<target>.+)",
            "delete",
            CommandCategory.MEMORY,
            self._handle_delete,
            "Delete a node"
        )

        self.register_pattern(
            r"(?:remember|save|note)\s+(?:that\s+)?(?P<content>.+)",
            "remember",
            CommandCategory.MEMORY,
            self._handle_remember,
            "Save a new memory",
            ["remember that Thompson Sampling is Bayesian"]
        )

        # System commands
        self.register_pattern(
            r"(?:help|what can (?:I|you) (?:say|do))",
            "help",
            CommandCategory.SYSTEM,
            self._handle_help,
            "Show available commands"
        )

        self.register_pattern(
            r"(?:cancel|stop|never ?mind)",
            "cancel",
            CommandCategory.SYSTEM,
            self._handle_cancel,
            "Cancel current operation"
        )

        self.register_pattern(
            r"(?:undo|go back)",
            "undo",
            CommandCategory.SYSTEM,
            self._handle_undo,
            "Undo last action"
        )

    # === Command Processing ===

    def process_speech(self, text: str) -> Optional[VoiceCommand]:
        """
        Process speech input and execute matching command.

        Args:
            text: Recognized speech text

        Returns:
            Executed command or None if no match
        """
        if not text or not text.strip():
            return None

        # Check for wake word
        if self.wake_detector.check(text):
            if self.on_wake:
                self.on_wake()

            # Strip wake word and process remainder
            text = self.wake_detector.strip_wake_word(text)
            if not text:
                return None

        # Check if active (wake word was recently said)
        elif not self.wake_detector.is_active:
            return None
        elif self.wake_detector.check_timeout():
            return None

        # Reset timeout
        self.wake_detector.last_activation = datetime.now()

        # Try to match patterns
        for pattern in self.patterns:
            params = pattern.match(text)
            if params is not None:
                command = VoiceCommand(
                    command_id=self._generate_command_id(),
                    text=text,
                    intent=pattern.intent,
                    category=pattern.category,
                    parameters=params
                )

                # Execute handler
                try:
                    result = pattern.handler(command)
                    command.result = result
                    command.state = CommandState.EXECUTED
                except Exception as e:
                    command.error = str(e)
                    command.state = CommandState.FAILED
                    if self.on_error:
                        self.on_error(command, str(e))

                # Add to history
                self.history.append(command)
                if len(self.history) > self.max_history:
                    self.history.pop(0)

                # Fire callback
                if self.on_command:
                    self.on_command(command)

                return command

        # No pattern matched - create failed command
        command = VoiceCommand(
            command_id=self._generate_command_id(),
            text=text,
            intent="unknown",
            category=CommandCategory.SYSTEM,
            state=CommandState.FAILED,
            error="No matching command found"
        )

        self.history.append(command)
        return command

    # === Default Handlers ===

    def _handle_navigate(self, cmd: VoiceCommand) -> str:
        target = cmd.parameters.get("target", "")
        return f"Navigating to: {target}"

    def _handle_zoom_in(self, cmd: VoiceCommand) -> str:
        return "Zooming in"

    def _handle_zoom_out(self, cmd: VoiceCommand) -> str:
        return "Zooming out"

    def _handle_rotate(self, cmd: VoiceCommand) -> str:
        direction = cmd.parameters.get("direction", "right")
        return f"Rotating {direction}"

    def _handle_select(self, cmd: VoiceCommand) -> str:
        target = cmd.parameters.get("target", "")
        return f"Selecting: {target}"

    def _handle_deselect(self, cmd: VoiceCommand) -> str:
        return "Deselected"

    def _handle_explain(self, cmd: VoiceCommand) -> str:
        topic = cmd.parameters.get("topic", "")
        return f"Explaining: {topic}"

    def _handle_question(self, cmd: VoiceCommand) -> str:
        question = cmd.parameters.get("question", "")
        return f"Answering: {question}"

    def _handle_search(self, cmd: VoiceCommand) -> str:
        query = cmd.parameters.get("query", "")
        return f"Searching for: {query}"

    def _handle_move(self, cmd: VoiceCommand) -> str:
        location = cmd.parameters.get("location", "here")
        return f"Moving to: {location}"

    def _handle_scale(self, cmd: VoiceCommand) -> str:
        size = cmd.parameters.get("size") or cmd.parameters.get("direction")
        return f"Scaling: {size}"

    def _handle_connect(self, cmd: VoiceCommand) -> str:
        source = cmd.parameters.get("source", "")
        target = cmd.parameters.get("target", "")
        return f"Connecting {source} to {target}"

    def _handle_create(self, cmd: VoiceCommand) -> str:
        name = cmd.parameters.get("name", "New Node")
        return f"Creating node: {name}"

    def _handle_delete(self, cmd: VoiceCommand) -> str:
        target = cmd.parameters.get("target", "")
        return f"Deleting: {target}"

    def _handle_remember(self, cmd: VoiceCommand) -> str:
        content = cmd.parameters.get("content", "")
        return f"Remembering: {content}"

    def _handle_help(self, cmd: VoiceCommand) -> List[Dict]:
        return self.get_available_commands()

    def _handle_cancel(self, cmd: VoiceCommand) -> str:
        return "Cancelled"

    def _handle_undo(self, cmd: VoiceCommand) -> str:
        return "Undoing last action"

    # === Queries ===

    def get_available_commands(self) -> List[Dict]:
        """Get list of available commands."""
        commands = []
        for pattern in self.patterns:
            commands.append({
                "intent": pattern.intent,
                "category": pattern.category.value,
                "description": pattern.description,
                "examples": pattern.examples
            })
        return commands

    def get_command_history(
        self,
        category: Optional[CommandCategory] = None,
        limit: int = 20
    ) -> List[VoiceCommand]:
        """Get recent command history."""
        history = self.history
        if category:
            history = [c for c in history if c.category == category]
        return history[-limit:]

    def get_successful_commands(self, limit: int = 10) -> List[VoiceCommand]:
        """Get recent successful commands."""
        successful = [c for c in self.history if c.state == CommandState.EXECUTED]
        return successful[-limit:]

    def get_failed_commands(self, limit: int = 10) -> List[VoiceCommand]:
        """Get recent failed commands."""
        failed = [c for c in self.history if c.state == CommandState.FAILED]
        return failed[-limit:]

    def export_state(self) -> str:
        """Export current state as JSON."""
        data = {
            "is_active": self.wake_detector.is_active,
            "patterns_count": len(self.patterns),
            "history_count": len(self.history),
            "recent_commands": [c.to_dict() for c in self.history[-10:]]
        }
        return json.dumps(data, indent=2)


# Quick test
if __name__ == "__main__":
    print("Testing Voice Command System...")

    system = VoiceCommandSystem()

    # Custom callback
    def on_cmd(cmd):
        print(f"Command: {cmd.intent} -> {cmd.result}")

    system.on_command = on_cmd

    # Test commands
    test_phrases = [
        "hey hololoom go to Thompson Sampling",
        "what is neural networks",
        "zoom in",
        "select memory",
        "connect Thompson Sampling to Bayesian methods",
        "remember that exploration is important",
        "help"
    ]

    for phrase in test_phrases:
        print(f"\n>>> '{phrase}'")
        cmd = system.process_speech(phrase)
        if cmd:
            print(f"    Intent: {cmd.intent}")
            print(f"    Params: {cmd.parameters}")
            print(f"    State: {cmd.state.value}")
        else:
            print("    (No command recognized)")

    print(f"\n\nAvailable commands: {len(system.patterns)}")
    print(f"History: {len(system.history)} commands")
