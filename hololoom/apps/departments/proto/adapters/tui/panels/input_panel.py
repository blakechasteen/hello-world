"""Proto TUI Input Panel.

Query input with syntax highlighting and command history.

Status: Phase 3 TUI (2025-12-05)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional
from textual.widgets import Input, Static
from textual.containers import Container
from textual.message import Message


@dataclass
class CommandHistory:
    """Manages command history with navigation."""
    commands: List[str] = field(default_factory=list)
    index: int = -1
    max_size: int = 100

    def add(self, command: str) -> None:
        """Add a command to history."""
        if command and command != (self.commands[-1] if self.commands else None):
            self.commands.append(command)
            if len(self.commands) > self.max_size:
                self.commands.pop(0)
        self.index = -1

    def previous(self) -> Optional[str]:
        """Get previous command in history."""
        if not self.commands:
            return None
        if self.index < len(self.commands) - 1:
            self.index += 1
        return self.commands[-(self.index + 1)]

    def next(self) -> Optional[str]:
        """Get next command in history."""
        if not self.commands:
            return None
        if self.index > 0:
            self.index -= 1
            return self.commands[-(self.index + 1)]
        elif self.index == 0:
            self.index = -1
            return ""
        return None


class CommandInput(Input):
    """Input with command history and tab completion."""

    class Submitted(Message):
        """Message sent when command is submitted."""
        def __init__(self, value: str, mode: str):
            super().__init__()
            self.value = value
            self.mode = mode

    def __init__(
        self,
        placeholder: str = "Enter your query...",
        mode: str = "ask"
    ):
        super().__init__(placeholder=placeholder)
        self.history = CommandHistory()
        self.mode = mode
        self._completions: List[str] = []

    def on_key(self, event) -> None:
        """Handle special keys."""
        if event.key == "up":
            prev = self.history.previous()
            if prev is not None:
                self.value = prev
                event.prevent_default()
        elif event.key == "down":
            next_cmd = self.history.next()
            if next_cmd is not None:
                self.value = next_cmd
                event.prevent_default()
        elif event.key == "tab":
            self._handle_completion()
            event.prevent_default()

    def _handle_completion(self) -> None:
        """Handle tab completion."""
        value = self.value.strip()
        if not value:
            return

        # Built-in commands for completion
        commands = [
            "/ask", "/review", "/repl", "/test", "/explain",
            "/git", "/run", "/security", "/help", "/clear", "/quit"
        ]

        if value.startswith("/"):
            # Complete commands
            matches = [c for c in commands if c.startswith(value)]
            if len(matches) == 1:
                self.value = matches[0] + " "
            elif matches:
                # Show completions somehow
                pass

    def submit_value(self) -> None:
        """Submit the current value."""
        value = self.value.strip()
        if value:
            self.history.add(value)
            self.post_message(self.Submitted(value, self.mode))
            self.value = ""


class InputPanel(Container):
    """Complete input panel with prompt and input field."""

    DEFAULT_CSS = """
    InputPanel {
        height: auto;
        width: 100%;
        padding: 0 1;
        background: $surface;
    }

    InputPanel > .prompt {
        width: auto;
        color: $text-muted;
    }

    InputPanel > CommandInput {
        width: 1fr;
    }
    """

    def __init__(self, mode: str = "ask"):
        super().__init__()
        self.mode = mode
        self._input: Optional[CommandInput] = None
        self._prompt: Optional[Static] = None

    def compose(self):
        """Compose the input panel."""
        self._prompt = Static(f"{self.mode} > ", classes="prompt")
        yield self._prompt

        self._input = CommandInput(mode=self.mode)
        yield self._input

    def set_mode(self, mode: str) -> None:
        """Set the input mode."""
        self.mode = mode
        if self._input:
            self._input.mode = mode
        if self._prompt:
            self._prompt.update(f"{mode} > ")

    def focus_input(self) -> None:
        """Focus the input field."""
        if self._input:
            self._input.focus()

    def clear_input(self) -> None:
        """Clear the input field."""
        if self._input:
            self._input.value = ""
