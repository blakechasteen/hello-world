"""Proto TUI Response Panel.

Response display with markdown rendering and message history.

Status: Phase 3 TUI (2025-12-05)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from textual.widgets import Static
from textual.containers import ScrollableContainer


class MessageType(Enum):
    """Types of messages in the response panel."""
    USER = "user"
    PROTO = "proto"
    SYSTEM = "system"
    ERROR = "error"
    DEBUG = "debug"


@dataclass
class Message:
    """A single message in the conversation."""
    content: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def format(self) -> str:
        """Format message for display."""
        time_str = self.timestamp.strftime("%H:%M:%S")

        if self.message_type == MessageType.USER:
            return f"[dim]{time_str}[/] [bold cyan]You:[/] {self.content}"
        elif self.message_type == MessageType.PROTO:
            return f"[dim]{time_str}[/] [bold green]Proto:[/] {self.content}"
        elif self.message_type == MessageType.SYSTEM:
            return f"[dim]{time_str}[/] [bold yellow]System:[/] {self.content}"
        elif self.message_type == MessageType.ERROR:
            return f"[dim]{time_str}[/] [bold red]Error:[/] {self.content}"
        elif self.message_type == MessageType.DEBUG:
            return f"[dim]{time_str}[/] [dim]Debug:[/] {self.content}"
        else:
            return f"[dim]{time_str}[/] {self.content}"


class MessageWidget(Static):
    """Widget for displaying a single message."""

    DEFAULT_CSS = """
    MessageWidget {
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
    }

    MessageWidget.user {
        background: $surface-darken-1;
    }

    MessageWidget.proto {
        background: $surface;
    }

    MessageWidget.error {
        background: $error 10%;
    }
    """

    def __init__(self, message: Message):
        super().__init__(message.format())
        self.message = message
        self.add_class(message.message_type.value)


class ResponsePanel(ScrollableContainer):
    """Scrollable panel displaying conversation history."""

    DEFAULT_CSS = """
    ResponsePanel {
        height: 100%;
        width: 3fr;
        padding: 1;
        background: $surface;
        border: solid $primary;
    }

    ResponsePanel > .welcome {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }

    ResponsePanel > .thinking {
        color: $warning;
        text-style: italic;
    }
    """

    def __init__(self):
        super().__init__()
        self.messages: List[Message] = []
        self._thinking: bool = False
        self._thinking_widget: Optional[Static] = None

    def compose(self):
        """Compose initial content."""
        yield Static(
            "[dim]Welcome to Proto![/]\n\n"
            "Enter your query below to get started.\n"
            "Use [bold]/help[/] for available commands.",
            classes="welcome"
        )

    def add_message(
        self,
        content: str,
        message_type: MessageType = MessageType.PROTO,
        metadata: Optional[dict] = None
    ) -> None:
        """Add a message to the response panel."""
        # Remove welcome message on first real message
        if len(self.messages) == 0:
            welcome = self.query_one(".welcome", Static)
            if welcome:
                welcome.remove()

        # Stop thinking indicator if active
        self.stop_thinking()

        # Create and store message
        message = Message(
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        self.messages.append(message)

        # Add widget
        widget = MessageWidget(message)
        self.mount(widget)

        # Scroll to bottom
        self.scroll_end(animate=False)

    def add_user_message(self, content: str) -> None:
        """Convenience method for adding user messages."""
        self.add_message(content, MessageType.USER)

    def add_proto_message(self, content: str, metadata: Optional[dict] = None) -> None:
        """Convenience method for adding Proto responses."""
        self.add_message(content, MessageType.PROTO, metadata)

    def add_error(self, content: str) -> None:
        """Convenience method for adding error messages."""
        self.add_message(content, MessageType.ERROR)

    def add_system(self, content: str) -> None:
        """Convenience method for adding system messages."""
        self.add_message(content, MessageType.SYSTEM)

    def start_thinking(self, message: str = "Thinking...") -> None:
        """Show thinking indicator."""
        if self._thinking:
            return

        self._thinking = True
        self._thinking_widget = Static(f"[italic]{message}[/]", classes="thinking")
        self.mount(self._thinking_widget)
        self.scroll_end(animate=False)

    def stop_thinking(self) -> None:
        """Remove thinking indicator."""
        if not self._thinking:
            return

        self._thinking = False
        if self._thinking_widget:
            self._thinking_widget.remove()
            self._thinking_widget = None

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()

        # Remove all message widgets
        for widget in self.query(MessageWidget):
            widget.remove()

        # Remove thinking widget if present
        self.stop_thinking()

        # Show welcome again
        welcome = Static(
            "[dim]Conversation cleared.[/]\n\n"
            "Enter a new query to continue.",
            classes="welcome"
        )
        self.mount(welcome)

    def get_history(self, limit: int = 20) -> List[Message]:
        """Get recent message history."""
        return self.messages[-limit:]

    def export_history(self) -> str:
        """Export conversation history as plain text."""
        lines = []
        for msg in self.messages:
            time_str = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"[{time_str}] {msg.message_type.value}: {msg.content}")
        return "\n".join(lines)
