"""Proto TUI Context Panel.

Sidebar showing context files, mode, and status.

Status: Phase 3 TUI (2025-12-05)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from textual.widgets import Static
from textual.containers import Container, Vertical


@dataclass
class FileInfo:
    """Information about a context file."""
    path: str
    name: str
    size: int = 0
    lines: int = 0
    language: str = "unknown"
    added_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_path(cls, path: str) -> "FileInfo":
        """Create FileInfo from a file path."""
        p = Path(path)
        name = p.name

        # Detect language from extension
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".sh": "bash",
            ".sql": "sql",
        }
        language = ext_to_lang.get(p.suffix.lower(), "text")

        # Get file stats if file exists
        size = 0
        lines = 0
        if p.exists():
            size = p.stat().st_size
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    lines = sum(1 for _ in f)
            except Exception:
                pass

        return cls(
            path=str(p.absolute()),
            name=name,
            size=size,
            lines=lines,
            language=language
        )


class FileWidget(Static):
    """Widget for displaying a single file in context."""

    DEFAULT_CSS = """
    FileWidget {
        width: 100%;
        padding: 0 1;
        height: auto;
    }

    FileWidget:hover {
        background: $surface-lighten-1;
    }

    FileWidget.active {
        background: $primary 20%;
    }
    """

    def __init__(self, file_info: FileInfo, active: bool = False):
        # Format display
        icon = self._get_icon(file_info.language)
        lines_str = f"{file_info.lines}L" if file_info.lines else ""

        content = f"{icon} [cyan]{file_info.name}[/] [dim]{lines_str}[/]"
        super().__init__(content)

        self.file_info = file_info
        if active:
            self.add_class("active")

    def _get_icon(self, language: str) -> str:
        """Get icon for language."""
        icons = {
            "python": "🐍",
            "javascript": "📜",
            "typescript": "📘",
            "rust": "🦀",
            "go": "🐹",
            "java": "☕",
            "c": "⚙️",
            "cpp": "⚙️",
            "markdown": "📝",
            "json": "📋",
            "yaml": "📋",
            "bash": "💻",
            "sql": "🗃️",
        }
        return icons.get(language, "📄")


class ContextPanel(Container):
    """Sidebar panel showing context files and status."""

    DEFAULT_CSS = """
    ContextPanel {
        height: 100%;
        width: 1fr;
        min-width: 25;
        max-width: 40;
        padding: 1;
        background: $surface-darken-1;
        border: solid $secondary;
    }

    ContextPanel > .section-title {
        text-style: bold;
        margin-bottom: 1;
    }

    ContextPanel > .section {
        margin-bottom: 2;
    }

    ContextPanel > .mode-display {
        padding: 0 1;
    }

    ContextPanel > .shortcut {
        color: $text-muted;
    }

    ContextPanel > .empty-hint {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    """

    def __init__(self, mode: str = "ask"):
        super().__init__()
        self.mode = mode
        self.context_files: List[FileInfo] = []
        self.active_file: Optional[str] = None

    def compose(self):
        """Compose the context panel."""
        # Context files section
        yield Static("📁 Context Files", classes="section-title")
        yield Static("[dim italic]No files loaded[/]", classes="empty-hint", id="files-hint")

        # Mode section
        yield Static("")
        yield Static("⚡ Mode", classes="section-title")
        yield Static(f"  [yellow]{self.mode}[/]", classes="mode-display", id="mode-display")

        # Shortcuts section
        yield Static("")
        yield Static("⌨️ Shortcuts", classes="section-title")
        yield Static("  [dim]Ctrl+O[/] Open file", classes="shortcut")
        yield Static("  [dim]Ctrl+R[/] Review mode", classes="shortcut")
        yield Static("  [dim]Ctrl+T[/] Test mode", classes="shortcut")
        yield Static("  [dim]Ctrl+L[/] Clear", classes="shortcut")
        yield Static("  [dim]Ctrl+Q[/] Quit", classes="shortcut")

    def add_file(self, path: str) -> bool:
        """Add a file to context."""
        # Check if already added
        for f in self.context_files:
            if f.path == path:
                return False

        file_info = FileInfo.from_path(path)
        self.context_files.append(file_info)

        # Remove empty hint if present
        hint = self.query_one("#files-hint", Static)
        if hint:
            hint.display = False

        # Add file widget
        widget = FileWidget(file_info, active=(path == self.active_file))
        self.mount(widget, after=self.query_one(".section-title"))

        return True

    def remove_file(self, path: str) -> bool:
        """Remove a file from context."""
        # Find and remove file info
        for i, f in enumerate(self.context_files):
            if f.path == path:
                self.context_files.pop(i)
                break
        else:
            return False

        # Remove widget
        for widget in self.query(FileWidget):
            if widget.file_info.path == path:
                widget.remove()
                break

        # Show hint if no files left
        if not self.context_files:
            hint = self.query_one("#files-hint", Static)
            if hint:
                hint.display = True

        return True

    def set_active_file(self, path: str) -> None:
        """Set the active file."""
        self.active_file = path

        # Update widget classes
        for widget in self.query(FileWidget):
            if widget.file_info.path == path:
                widget.add_class("active")
            else:
                widget.remove_class("active")

    def set_mode(self, mode: str) -> None:
        """Set the current mode."""
        self.mode = mode

        mode_display = self.query_one("#mode-display", Static)
        if mode_display:
            mode_display.update(f"  [yellow]{mode}[/]")

    def get_context_files(self) -> List[FileInfo]:
        """Get list of context files."""
        return self.context_files.copy()

    def get_context_paths(self) -> List[str]:
        """Get list of context file paths."""
        return [f.path for f in self.context_files]

    def clear_context(self) -> None:
        """Clear all context files."""
        self.context_files.clear()
        self.active_file = None

        # Remove all file widgets
        for widget in self.query(FileWidget):
            widget.remove()

        # Show hint
        hint = self.query_one("#files-hint", Static)
        if hint:
            hint.display = True
