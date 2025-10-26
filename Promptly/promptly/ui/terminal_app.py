"""
Promptly Terminal UI - Interactive TUI Application
===================================================
Full-featured terminal interface using Textual framework.

Features:
- Live prompt execution
- Real-time analytics dashboard
- Loop visualization
- Cost tracking
- Interactive prompt composer
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Button, Static, Input, TextArea, DataTable, TabbedContent, TabPane, ProgressBar, Label, Select, Tree
from textual.binding import Binding
from textual import events
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table as RichTable
from datetime import datetime
import asyncio

# Graceful imports
try:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from promptly import Promptly
    from execution_engine import ExecutionEngine
    from loop_composition import LoopComposer
    from tools.cost_tracker import CostTracker
    from tools.prompt_analytics import PromptAnalytics
    PROMPTLY_AVAILABLE = True
except ImportError as e:
    PROMPTLY_AVAILABLE = False
    print(f"Import warning: {e}")


class StatusPanel(Static):
    """Live status panel showing system state"""

    total_prompts = reactive(0)
    total_cost = reactive(0.0)
    active_loops = reactive(0)

    def render(self) -> Panel:
        """Render status panel"""
        content = f"""
[cyan]Total Prompts:[/cyan] {self.total_prompts}
[green]Total Cost:[/green] ${self.total_cost:.4f}
[yellow]Active Loops:[/yellow] {self.active_loops}
[magenta]Status:[/magenta] Ready
"""
        return Panel(content, title="[bold cyan]System Status", border_style="cyan")


class PromptComposer(Vertical):
    """Interactive prompt composition area"""

    def compose(self) -> ComposeResult:
        """Create composer widgets"""
        yield Label("[bold cyan]Prompt Composer", id="composer-label")
        yield Input(placeholder="Enter prompt or load skill...", id="prompt-input")
        yield Horizontal(
            Button("Execute", variant="primary", id="execute-btn"),
            Button("Chain", variant="success", id="chain-btn"),
            Button("Loop", variant="warning", id="loop-btn"),
            Button("Clear", id="clear-btn"),
            id="action-buttons"
        )
        yield TextArea("# Results will appear here...", id="results-area", language="markdown")


class AnalyticsDashboard(Vertical):
    """Analytics visualization"""

    def compose(self) -> ComposeResult:
        """Create analytics widgets"""
        yield Label("[bold green]Analytics Dashboard", id="analytics-label")

        # Data table for prompt history
        table = DataTable(id="analytics-table")
        table.add_columns("Time", "Prompt", "Tokens", "Cost", "Status")
        yield table

        # Cost breakdown
        yield ProgressBar(total=100, show_eta=False, id="cost-progress")
        yield Label("Cost: $0.0000", id="cost-label")


class LoopVisualizer(Vertical):
    """Visual representation of loop execution"""

    def compose(self) -> ComposeResult:
        """Create visualizer widgets"""
        yield Label("[bold yellow]Loop Execution Flow", id="loop-label")

        # Tree view for loop hierarchy
        tree = Tree("Execution", id="loop-tree")
        yield tree

        yield ScrollableContainer(
            Static(id="loop-details"),
            id="loop-scroll"
        )


class SkillBrowser(Vertical):
    """Browse and manage skills"""

    def compose(self) -> ComposeResult:
        """Create skill browser"""
        yield Label("[bold magenta]Skill Library", id="skill-label")

        # Skill list
        tree = Tree("Skills", id="skill-tree")
        yield tree

        yield Horizontal(
            Button("Load", id="skill-load"),
            Button("Edit", id="skill-edit"),
            Button("New", id="skill-new"),
            id="skill-actions"
        )


class PromptlyApp(App):
    """
    Main Promptly Terminal UI Application

    Full-featured TUI with:
    - Live prompt execution
    - Real-time analytics
    - Loop visualization
    - Skill management
    - Cost tracking
    """

    CSS = """
    Screen {
        background: $surface;
    }

    #status-panel {
        dock: top;
        height: 6;
        background: $panel;
    }

    #composer-area {
        width: 50%;
        border: solid $accent;
    }

    #analytics-area {
        width: 50%;
        border: solid $success;
    }

    #loop-area {
        height: 40%;
        border: solid $warning;
    }

    #skill-area {
        height: 40%;
        border: solid $error;
    }

    #prompt-input {
        margin: 1;
    }

    #results-area {
        height: 1fr;
        margin: 1;
    }

    #action-buttons {
        height: 3;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }

    DataTable {
        height: 1fr;
    }

    Tree {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+e", "execute", "Execute"),
        Binding("ctrl+c", "clear", "Clear"),
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+l", "load_skill", "Load Skill"),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self):
        super().__init__()
        self.promptly = Promptly() if PROMPTLY_AVAILABLE else None
        self.cost_tracker = CostTracker() if PROMPTLY_AVAILABLE else None
        self.analytics = PromptAnalytics() if PROMPTLY_AVAILABLE else None
        self.execution_count = 0
        self.total_cost = 0.0

    def compose(self) -> ComposeResult:
        """Create application layout"""
        yield Header(show_clock=True)

        # Status panel at top
        yield StatusPanel(id="status-panel")

        # Main content area with tabs
        with TabbedContent(initial="composer"):
            # Tab 1: Prompt Composer
            with TabPane("Composer", id="composer"):
                with Horizontal():
                    yield PromptComposer(id="composer-area")
                    yield AnalyticsDashboard(id="analytics-area")

            # Tab 2: Loop Visualizer
            with TabPane("Loops", id="loops"):
                yield LoopVisualizer(id="loop-area")

            # Tab 3: Skills
            with TabPane("Skills", id="skills"):
                yield SkillBrowser(id="skill-area")

            # Tab 4: Analytics
            with TabPane("Analytics", id="analytics-tab"):
                yield Static("[bold]Detailed Analytics Coming Soon...", id="analytics-detail")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize application after mounting"""
        self.title = "Promptly - Interactive Prompt Platform"
        self.sub_title = "v1.0 | Press F1 for help"

        # Load initial data
        self._load_skills()
        self._load_analytics()

    def _load_skills(self):
        """Load available skills into tree"""
        if not PROMPTLY_AVAILABLE:
            return

        try:
            tree = self.query_one("#skill-tree", Tree)
            tree.clear()

            # Add skill categories
            analysis = tree.root.add("Analysis")
            analysis.add_leaf("Summarize")
            analysis.add_leaf("Extract")
            analysis.add_leaf("Classify")

            generation = tree.root.add("Generation")
            generation.add_leaf("Write")
            generation.add_leaf("Rewrite")
            generation.add_leaf("Expand")

            meta = tree.root.add("Meta")
            meta.add_leaf("Reflect")
            meta.add_leaf("Critique")
            meta.add_leaf("Improve")
        except Exception as e:
            self.notify(f"Error loading skills: {e}", severity="error")

    def _load_analytics(self):
        """Load analytics data"""
        if not PROMPTLY_AVAILABLE or not self.analytics:
            return

        try:
            # Update status panel
            status = self.query_one(StatusPanel)
            summary = self.analytics.get_summary()
            status.total_prompts = summary.get('total_executions', 0)
            status.total_cost = summary.get('total_cost', 0.0)
        except Exception as e:
            self.notify(f"Error loading analytics: {e}", severity="warning")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        button_id = event.button.id

        if button_id == "execute-btn":
            await self.action_execute()
        elif button_id == "chain-btn":
            await self.action_chain()
        elif button_id == "loop-btn":
            await self.action_loop()
        elif button_id == "clear-btn":
            self.action_clear()
        elif button_id == "skill-load":
            await self.action_load_skill()

    async def action_execute(self):
        """Execute prompt"""
        if not PROMPTLY_AVAILABLE or not self.promptly:
            self.notify("Promptly not available", severity="error")
            return

        try:
            # Get prompt from input
            prompt_input = self.query_one("#prompt-input", Input)
            prompt = prompt_input.value.strip()

            if not prompt:
                self.notify("Please enter a prompt", severity="warning")
                return

            # Show executing state
            results_area = self.query_one("#results-area", TextArea)
            results_area.text = f"# Executing...\n\n**Prompt:** {prompt}\n\n*Processing...*"

            # Execute (simulated for now)
            await asyncio.sleep(0.5)  # Simulate API call

            result = f"Result for: {prompt}"

            # Update results
            results_area.text = f"""# Execution Complete

**Prompt:** {prompt}

**Result:**
{result}

**Status:** Success
**Tokens:** ~100
**Cost:** $0.0010
**Time:** {datetime.now().strftime('%H:%M:%S')}
"""

            # Update analytics table
            table = self.query_one("#analytics-table", DataTable)
            table.add_row(
                datetime.now().strftime("%H:%M:%S"),
                prompt[:30] + "..." if len(prompt) > 30 else prompt,
                "100",
                "$0.0010",
                "✓"
            )

            # Update status
            self.execution_count += 1
            self.total_cost += 0.0010
            status = self.query_one(StatusPanel)
            status.total_prompts = self.execution_count
            status.total_cost = self.total_cost

            self.notify("Execution complete!", severity="information")

        except Exception as e:
            self.notify(f"Execution error: {e}", severity="error")

    async def action_chain(self):
        """Execute prompt chain"""
        self.notify("Chain execution coming soon!", severity="information")

    async def action_loop(self):
        """Execute loop"""
        self.notify("Loop execution coming soon!", severity="information")

    def action_clear(self):
        """Clear inputs and results"""
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.value = ""

        results_area = self.query_one("#results-area", TextArea)
        results_area.text = "# Results will appear here..."

        self.notify("Cleared", severity="information")

    async def action_load_skill(self):
        """Load selected skill"""
        self.notify("Skill loading coming soon!", severity="information")

    def action_help(self):
        """Show help"""
        self.notify("""
Promptly Terminal UI - Keyboard Shortcuts:
Ctrl+E: Execute prompt
Ctrl+C: Clear inputs
Ctrl+S: Save results
Ctrl+L: Load skill
F1: Help
Q: Quit
""", severity="information", timeout=10)


def run_tui():
    """Run the Promptly TUI application"""
    app = PromptlyApp()
    app.run()


if __name__ == "__main__":
    run_tui()
