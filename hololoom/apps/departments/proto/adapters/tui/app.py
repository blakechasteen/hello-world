"""Proto TUI Application.

Main Textual app for Proto code agent with semantic wave animations.

Usage:
    python -m HoloLoom.apps.departments.proto.adapters.tui

Status: Phase 3 TUI (2025-12-05)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Footer, Input, Static

from .wave_animation import (
    WaveGenerator,
    WaveTheme,
    create_wave_system,
)

if TYPE_CHECKING:
    from hololoom.apps.departments.proto.engine import ProtoEngine


class WaveHeader(Static):
    """Animated wave header showing incoming/response state."""

    DEFAULT_CSS = """
    WaveHeader {
        height: auto;
        min-height: 2;
        max-height: 8;
        width: 100%;
        background: $surface;
        color: cyan;
    }
    """

    def __init__(self, generator: WaveGenerator, title: str = "P R O T O"):
        self.generator = generator
        self.title = title
        self._timer: Timer | None = None
        # Generate initial content before super().__init__
        initial_rows = self.generator.generate_top_waves(80, self.title)
        initial_color = self.generator.get_top_color()
        initial_content = '\n'.join(initial_rows)
        super().__init__(f"[{initial_color}]{initial_content}[/]")

    def on_mount(self) -> None:
        """Start animation timer on mount."""
        self._timer = self.set_interval(
            1 / self.generator.config.fps,
            self._animate
        )

    def _animate(self) -> None:
        """Advance animation and update display."""
        self.generator.advance_frame()
        self._update_content()

    def _update_content(self) -> None:
        """Update the wave content."""
        width = self.size.width or 80
        rows = self.generator.generate_top_waves(width, self.title)
        color = self.generator.get_top_color()

        content = '\n'.join(rows)
        self.update(f"[{color}]{content}[/]")

        # Adjust height based on wave depth
        self.styles.height = len(rows)


class WaveFooter(Static):
    """Animated wave footer showing outgoing/request state."""

    DEFAULT_CSS = """
    WaveFooter {
        height: auto;
        min-height: 2;
        max-height: 8;
        width: 100%;
        background: $surface;
        color: blue;
    }
    """

    def __init__(self, generator: WaveGenerator):
        self.generator = generator
        self._timer: Timer | None = None
        # Generate initial content before super().__init__
        initial_rows = self.generator.generate_bottom_waves(80)
        initial_color = self.generator.get_bottom_color()
        initial_content = '\n'.join(initial_rows)
        super().__init__(f"[{initial_color}]{initial_content}[/]")

    def on_mount(self) -> None:
        """Start animation timer on mount."""
        self._timer = self.set_interval(
            1 / self.generator.config.fps,
            self._animate
        )

    def _animate(self) -> None:
        """Animation is driven by WaveHeader, just update content."""
        self._update_content()

    def _update_content(self) -> None:
        """Update the wave content."""
        width = self.size.width or 80
        rows = self.generator.generate_bottom_waves(width)
        color = self.generator.get_bottom_color()

        content = '\n'.join(rows)
        self.update(f"[{color}]{content}[/]")

        # Adjust height based on wave depth
        self.styles.height = len(rows)


class ResponsePanel(Static):
    """Panel displaying Proto responses with markdown rendering."""

    DEFAULT_CSS = """
    ResponsePanel {
        height: 100%;
        width: 3fr;
        padding: 1;
        background: $surface;
        border: solid $primary;
        overflow-y: auto;
    }
    """

    def __init__(self):
        super().__init__("[dim]Ready for your query...[/]")
        self.messages: list[tuple[str, str]] = []  # (role, content)

    def _build_content(self) -> str:
        """Build the panel content."""
        if not self.messages:
            return "[dim]Ready for your query...[/]"

        lines = []
        for role, content in self.messages[-20:]:  # Keep last 20 messages
            if role == "user":
                lines.append(f"[bold cyan]User:[/] {content}")
            elif role == "proto":
                lines.append(f"[bold green]Proto:[/] {content}")
            elif role == "error":
                lines.append(f"[bold red]Error:[/] {content}")
            else:
                lines.append(f"[dim]{role}:[/] {content}")
            lines.append("")

        return '\n'.join(lines)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the response panel."""
        self.messages.append((role, content))
        self.update(self._build_content())

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.update(self._build_content())


class ContextPanel(Static):
    """Sidebar showing context files and status."""

    DEFAULT_CSS = """
    ContextPanel {
        height: 100%;
        width: 1fr;
        min-width: 25;
        max-width: 35;
        padding: 1;
        background: $surface-darken-1;
        border: solid $secondary;
    }
    """

    def __init__(self):
        self.context_files: list[str] = []
        self.mode = "ask"
        # Pass initial content to Static
        super().__init__(self._build_content())

    def _build_content(self) -> str:
        """Build the panel content."""
        lines = [
            "[bold]Context Files[/]",
            ""
        ]

        if self.context_files:
            for f in self.context_files[:10]:
                # Show just filename
                name = f.split('/')[-1].split('\\')[-1]
                lines.append(f"  [cyan]{name}[/]")
        else:
            lines.append("  [dim]No files loaded[/]")

        lines.extend([
            "",
            "[bold]Mode[/]",
            f"  [yellow]{self.mode}[/]",
            "",
            "[bold]Shortcuts[/]",
            "  [dim]Ctrl+O[/] Open file",
            "  [dim]Ctrl+R[/] Review mode",
            "  [dim]Ctrl+T[/] Test mode",
            "  [dim]Ctrl+Q[/] Quit",
        ])

        return '\n'.join(lines)

    def add_context_file(self, path: str) -> None:
        """Add a file to context."""
        if path not in self.context_files:
            self.context_files.append(path)
            self.update(self._build_content())

    def remove_context_file(self, path: str) -> None:
        """Remove a file from context."""
        if path in self.context_files:
            self.context_files.remove(path)
            self.update(self._build_content())

    def set_mode(self, mode: str) -> None:
        """Set the current mode (ask, review, repl, etc.)."""
        self.mode = mode
        self.update(self._build_content())


class QueryInput(Input):
    """Query input with history support."""

    DEFAULT_CSS = """
    QueryInput {
        width: 100%;
        margin: 0 1;
    }
    """

    def __init__(self):
        super().__init__(placeholder="Enter your query...")
        self.history: list[str] = []
        self.history_index = -1

    def on_key(self, event) -> None:
        """Handle up/down for history navigation."""
        if event.key == "up" and self.history:
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                self.value = self.history[-(self.history_index + 1)]
        elif event.key == "down" and self.history:
            if self.history_index > 0:
                self.history_index -= 1
                self.value = self.history[-(self.history_index + 1)]
            elif self.history_index == 0:
                self.history_index = -1
                self.value = ""

    def add_to_history(self, query: str) -> None:
        """Add query to history."""
        if query and query != (self.history[-1] if self.history else None):
            self.history.append(query)
        self.history_index = -1


class ProtoTUI(App):
    """Proto Terminal User Interface with semantic wave animations."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-container {
        height: 1fr;
        width: 100%;
    }

    #content-area {
        height: 1fr;
        width: 100%;
    }

    #input-area {
        height: 3;
        width: 100%;
        padding: 0 1;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+o", "open_file", "Open File", show=True),
        Binding("ctrl+r", "review_mode", "Review Mode", show=True),
        Binding("ctrl+t", "test_mode", "Test Mode", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("escape", "cancel", "Cancel"),
    ]

    # Reactive mode
    mode: reactive[str] = reactive("ask")

    def __init__(
        self,
        engine: ProtoEngine | None = None,
        theme: WaveTheme = WaveTheme.OCEAN
    ):
        super().__init__()
        self.engine = engine

        # Create wave system
        self.wave_controller = create_wave_system(theme)
        self.generator = self.wave_controller.generator

        # Panels (created in compose)
        self._response_panel: ResponsePanel | None = None
        self._context_panel: ContextPanel | None = None
        self._query_input: QueryInput | None = None

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        # Top waves
        yield WaveHeader(self.generator, "P R O T O")

        # Main content area
        with Container(id="main-container"):
            with Horizontal(id="content-area"):
                self._response_panel = ResponsePanel()
                yield self._response_panel

                self._context_panel = ContextPanel()
                yield self._context_panel

        # Input area
        with Container(id="input-area"):
            self._query_input = QueryInput()
            yield self._query_input

        # Bottom waves
        yield WaveFooter(self.generator)

        # Footer with keybindings
        yield Footer()

    def on_mount(self) -> None:
        """Focus input on mount."""
        if self._query_input:
            self._query_input.focus()
        self.wave_controller.on_idle()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle query submission."""
        query = event.value.strip()
        if not query:
            return

        # Add to history
        if self._query_input:
            self._query_input.add_to_history(query)
            self._query_input.value = ""

        # Show user message
        if self._response_panel:
            self._response_panel.add_message("user", query)

        # Process query (no await - @work decorator handles background execution)
        self._process_query(query)

    @work
    async def _process_query(self, query: str) -> None:
        """Process query through Proto engine."""
        # Update waves for processing
        self.wave_controller.on_preparing_request()
        await asyncio.sleep(0.1)

        self.wave_controller.on_sending_request()
        await asyncio.sleep(0.1)

        self.wave_controller.on_waiting_response()

        try:
            if self.engine:
                # Real engine processing
                self.wave_controller.on_processing()
                result = await self.engine.process(query, mode=self.mode)

                self.wave_controller.on_receiving_response()
                await asyncio.sleep(0.1)

                if self._response_panel:
                    self._response_panel.add_message("proto", result.response)

                self.wave_controller.on_complete()
            else:
                # Demo mode without engine
                self.wave_controller.on_processing()
                await asyncio.sleep(1.0)  # Simulate processing

                self.wave_controller.on_receiving_response()
                await asyncio.sleep(0.5)

                if self._response_panel:
                    self._response_panel.add_message(
                        "proto",
                        f"[Demo Mode] Received: {query}\n\n"
                        "Proto engine not connected. Run with --engine to enable."
                    )

                self.wave_controller.on_complete()

        except Exception as e:
            self.wave_controller.on_error()
            if self._response_panel:
                self._response_panel.add_message("error", str(e))

        finally:
            # Return to idle after brief delay
            await asyncio.sleep(1.0)
            self.wave_controller.on_idle()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_open_file(self) -> None:
        """Open a file for context."""
        # TODO: Implement file browser
        self.notify("File browser not yet implemented")

    def action_review_mode(self) -> None:
        """Switch to review mode."""
        self.mode = "review"
        if self._context_panel:
            self._context_panel.set_mode("review")
        self.notify("Switched to review mode")

    def action_test_mode(self) -> None:
        """Switch to test mode."""
        self.mode = "test"
        if self._context_panel:
            self._context_panel.set_mode("test")
        self.notify("Switched to test mode")

    def action_clear(self) -> None:
        """Clear the response panel."""
        if self._response_panel:
            self._response_panel.clear()

    def action_cancel(self) -> None:
        """Cancel current operation."""
        self.wave_controller.on_idle()
        if self._query_input:
            self._query_input.value = ""


def main():
    """Run the Proto TUI."""
    import sys

    # Check for engine flag
    engine = None
    if "--engine" in sys.argv:
        try:
            from hololoom.apps.departments.proto.engine import ProtoEngine
            engine = ProtoEngine()
        except ImportError:
            print("Warning: Could not import ProtoEngine, running in demo mode")

    # Select theme
    theme = WaveTheme.OCEAN
    if "--theme" in sys.argv:
        idx = sys.argv.index("--theme")
        if idx + 1 < len(sys.argv):
            theme_name = sys.argv[idx + 1].upper()
            theme = getattr(WaveTheme, theme_name, WaveTheme.OCEAN)

    app = ProtoTUI(engine=engine, theme=theme)
    app.run()


if __name__ == "__main__":
    main()
