"""Proto TUI Session Management.

Handles state persistence for conversations, context, and settings.

Status: Phase 3 TUI (2025-12-05)
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SessionMessage:
    """Serializable message for session storage."""
    content: str
    message_type: str  # "user", "proto", "system", "error"
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionContext:
    """Context file information for session storage."""
    path: str
    name: str
    language: str
    lines: int
    added_at: str


@dataclass
class SessionData:
    """Complete session state for persistence."""
    session_id: str
    created_at: str
    updated_at: str
    mode: str = "ask"
    messages: list[SessionMessage] = field(default_factory=list)
    context_files: list[SessionContext] = field(default_factory=list)
    command_history: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "mode": self.mode,
            "messages": [asdict(m) for m in self.messages],
            "context_files": [asdict(c) for c in self.context_files],
            "command_history": self.command_history,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            mode=data.get("mode", "ask"),
            messages=[
                SessionMessage(**m) for m in data.get("messages", [])
            ],
            context_files=[
                SessionContext(**c) for c in data.get("context_files", [])
            ],
            command_history=data.get("command_history", []),
            settings=data.get("settings", {}),
        )


class SessionManager:
    """Manages Proto TUI session persistence."""

    DEFAULT_SESSION_DIR = ".proto_sessions"

    def __init__(self, session_dir: str | None = None):
        """Initialize session manager.

        Args:
            session_dir: Directory for session files. Defaults to ~/.proto_sessions
        """
        if session_dir:
            self.session_dir = Path(session_dir)
        else:
            self.session_dir = Path.home() / self.DEFAULT_SESSION_DIR

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: SessionData | None = None

    def create_session(self, session_id: str | None = None) -> SessionData:
        """Create a new session."""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        now = datetime.now().isoformat()
        self.current_session = SessionData(
            session_id=session_id,
            created_at=now,
            updated_at=now,
        )
        return self.current_session

    def save_session(self, session: SessionData | None = None) -> bool:
        """Save session to disk."""
        session = session or self.current_session
        if session is None:
            return False

        session.updated_at = datetime.now().isoformat()

        session_path = self.session_dir / f"{session.session_id}.json"
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save session: {e}")
            return False

    def load_session(self, session_id: str) -> SessionData | None:
        """Load session from disk."""
        session_path = self.session_dir / f"{session_id}.json"
        if not session_path.exists():
            return None

        try:
            with open(session_path, encoding="utf-8") as f:
                data = json.load(f)
            self.current_session = SessionData.from_dict(data)
            return self.current_session
        except Exception as e:
            print(f"Failed to load session: {e}")
            return None

    def get_latest_session(self) -> SessionData | None:
        """Get the most recently modified session."""
        sessions = list(self.session_dir.glob("*.json"))
        if not sessions:
            return None

        latest = max(sessions, key=lambda p: p.stat().st_mtime)
        return self.load_session(latest.stem)

    def list_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent sessions."""
        sessions = list(self.session_dir.glob("*.json"))
        sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        result = []
        for session_path in sessions[:limit]:
            try:
                with open(session_path, encoding="utf-8") as f:
                    data = json.load(f)
                result.append({
                    "session_id": data.get("session_id"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                    "file_count": len(data.get("context_files", [])),
                })
            except Exception:
                continue

        return result

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        session_path = self.session_dir / f"{session_id}.json"
        if not session_path.exists():
            return False

        try:
            session_path.unlink()
            if self.current_session and self.current_session.session_id == session_id:
                self.current_session = None
            return True
        except Exception:
            return False

    def add_message(
        self,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a message to the current session."""
        if self.current_session is None:
            self.create_session()

        message = SessionMessage(
            content=content,
            message_type=message_type,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
        )
        self.current_session.messages.append(message)

    def add_context_file(
        self,
        path: str,
        name: str,
        language: str = "text",
        lines: int = 0
    ) -> None:
        """Add a context file to the current session."""
        if self.current_session is None:
            self.create_session()

        context = SessionContext(
            path=path,
            name=name,
            language=language,
            lines=lines,
            added_at=datetime.now().isoformat(),
        )
        self.current_session.context_files.append(context)

    def remove_context_file(self, path: str) -> bool:
        """Remove a context file from the current session."""
        if self.current_session is None:
            return False

        for i, ctx in enumerate(self.current_session.context_files):
            if ctx.path == path:
                self.current_session.context_files.pop(i)
                return True
        return False

    def add_to_history(self, command: str) -> None:
        """Add a command to history."""
        if self.current_session is None:
            self.create_session()

        # Avoid duplicates at end
        if (self.current_session.command_history and
                self.current_session.command_history[-1] == command):
            return

        self.current_session.command_history.append(command)

        # Keep history bounded
        max_history = 1000
        if len(self.current_session.command_history) > max_history:
            self.current_session.command_history = \
                self.current_session.command_history[-max_history:]

    def set_mode(self, mode: str) -> None:
        """Set the current mode."""
        if self.current_session is None:
            self.create_session()

        self.current_session.mode = mode

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a session setting."""
        if self.current_session is None:
            return default
        return self.current_session.settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set a session setting."""
        if self.current_session is None:
            self.create_session()

        self.current_session.settings[key] = value

    def export_conversation(self, format: str = "text") -> str:
        """Export conversation in specified format."""
        if self.current_session is None:
            return ""

        if format == "markdown":
            return self._export_markdown()
        elif format == "json":
            return json.dumps(self.current_session.to_dict(), indent=2)
        else:
            return self._export_text()

    def _export_text(self) -> str:
        """Export as plain text."""
        lines = [
            f"Proto Session: {self.current_session.session_id}",
            f"Created: {self.current_session.created_at}",
            f"Mode: {self.current_session.mode}",
            "",
            "--- Conversation ---",
            "",
        ]

        for msg in self.current_session.messages:
            lines.append(f"[{msg.timestamp}] {msg.message_type}: {msg.content}")
            lines.append("")

        return "\n".join(lines)

    def _export_markdown(self) -> str:
        """Export as markdown."""
        lines = [
            f"# Proto Session: {self.current_session.session_id}",
            "",
            f"**Created:** {self.current_session.created_at}",
            f"**Mode:** {self.current_session.mode}",
            "",
            "## Conversation",
            "",
        ]

        for msg in self.current_session.messages:
            if msg.message_type == "user":
                lines.append("### You")
            elif msg.message_type == "proto":
                lines.append("### Proto")
            else:
                lines.append(f"### {msg.message_type.title()}")

            lines.append(f"*{msg.timestamp}*")
            lines.append("")
            lines.append(msg.content)
            lines.append("")

        if self.current_session.context_files:
            lines.append("## Context Files")
            lines.append("")
            for ctx in self.current_session.context_files:
                lines.append(f"- `{ctx.name}` ({ctx.language}, {ctx.lines} lines)")
            lines.append("")

        return "\n".join(lines)


# Convenience function
def create_session_manager(session_dir: str | None = None) -> SessionManager:
    """Create a session manager instance."""
    return SessionManager(session_dir)
