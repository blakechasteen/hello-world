#!/usr/bin/env python3
"""
Promptly TUI - Terminal User Interface
A rich, interactive terminal UI for managing prompts, branches, and evaluations
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

# Try importing textual (graceful degradation)
try:
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical, VerticalScroll
    from textual.widgets import (
        Header, Footer, Static, Button, Input, Tree, DataTable,
        TabbedContent, TabPane, TextLog, Label, ListView, ListItem,
        Markdown, DirectoryTree
    )
    from textual.binding import Binding
    from textual.screen import Screen
    from textual import on
    from textual.reactive import reactive
    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False

# Import Promptly
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from promptly import Promptly, PromptlyError
except ImportError:
    # Handle case where we're running from within the directory
    from ..promptly import Promptly, PromptlyError


class PromptViewer(Static):
    """Widget to display prompt details"""

    def __init__(self, prompt_data: Optional[Dict] = None):
        super().__init__()
        self.prompt_data = prompt_data

    def on_mount(self):
        """Render when mounted"""
        self.update_content()

    def update_content(self, prompt_data: Optional[Dict] = None):
        """Update the displayed prompt"""
        if prompt_data:
            self.prompt_data = prompt_data

        if not self.prompt_data:
            self.update("No prompt selected")
            return

        content = f"""# {self.prompt_data['name']}

**Branch:** {self.prompt_data['branch']}
**Version:** {self.prompt_data['version']}
**Commit:** {self.prompt_data['commit_hash']}
**Created:** {self.prompt_data['created_at']}

## Content

```
{self.prompt_data['content']}
```
"""

        if self.prompt_data.get('metadata'):
            content += f"\n## Metadata\n\n```json\n{json.dumps(self.prompt_data['metadata'], indent=2)}\n```"

        self.update(Markdown(content))


class BranchViewer(Static):
    """Widget to display branch tree"""

    def __init__(self, promptly: Promptly):
        super().__init__()
        self.promptly = promptly

    def compose(self) -> ComposeResult:
        """Compose the branch tree"""
        tree = Tree("Branches")
        tree.root.expand()

        try:
            with self.promptly._get_db() as db:
                cursor = db.conn.cursor()
                cursor.execute("SELECT name, head_commit, created_at FROM branches ORDER BY created_at")
                branches = cursor.fetchall()

                current_branch = self.promptly._get_current_branch()

                for branch_row in branches:
                    name = branch_row[0]
                    commit = branch_row[1][:8]
                    label = f"{'[*] ' if name == current_branch else ''}{name} ({commit})"
                    node = tree.root.add(label)

                    # Add prompts under each branch
                    prompts = self.promptly.list_prompts(branch=name)
                    for prompt in prompts[:5]:  # Show first 5
                        node.add_leaf(f"{prompt['name']} v{prompt['version']}")

        except Exception as e:
            tree.root.add_leaf(f"Error: {str(e)}")

        yield tree


class PromptListPanel(VerticalScroll):
    """Panel showing list of prompts"""

    def __init__(self, promptly: Promptly, on_select=None):
        super().__init__()
        self.promptly = promptly
        self.on_select_callback = on_select

    def compose(self) -> ComposeResult:
        """Compose the prompt list"""
        try:
            prompts = self.promptly.list_prompts()

            if not prompts:
                yield Static("No prompts found")
                return

            for prompt in prompts:
                btn = Button(
                    f"{prompt['name']} (v{prompt['version']})",
                    id=f"prompt_{prompt['name']}",
                    variant="primary"
                )
                yield btn

        except Exception as e:
            yield Static(f"Error loading prompts: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle prompt selection"""
        if self.on_select_callback and event.button.id:
            prompt_name = event.button.id.replace("prompt_", "")
            self.on_select_callback(prompt_name)


class LogPanel(VerticalScroll):
    """Panel showing commit log"""

    def __init__(self, promptly: Promptly):
        super().__init__()
        self.promptly = promptly

    def compose(self) -> ComposeResult:
        """Compose the log view"""
        try:
            commits = self.promptly.log(limit=20)

            if not commits:
                yield Static("No commits found")
                return

            for commit in commits:
                commit_info = f"""**{commit['commit_hash']}**
Prompt: {commit['name']} (v{commit['version']})
Branch: {commit['branch']}
Date: {commit['created_at']}
---
"""
                yield Markdown(commit_info)

        except Exception as e:
            yield Static(f"Error loading log: {e}")


class EvalPanel(VerticalScroll):
    """Panel for running evaluations"""

    def __init__(self, promptly: Promptly):
        super().__init__()
        self.promptly = promptly

    def compose(self) -> ComposeResult:
        """Compose the evaluation interface"""
        yield Static("# Evaluation Center", classes="header")
        yield Label("Select a prompt to evaluate:")

        try:
            prompts = self.promptly.list_prompts()
            for prompt in prompts:
                yield Button(
                    f"Evaluate: {prompt['name']}",
                    id=f"eval_{prompt['name']}",
                    variant="success"
                )
        except Exception as e:
            yield Static(f"Error: {e}")

        yield Static("\n## Recent Evaluations", classes="header")
        yield self._get_recent_evals()

    def _get_recent_evals(self) -> Static:
        """Get recent evaluation results"""
        try:
            with self.promptly._get_db() as db:
                cursor = db.conn.cursor()
                cursor.execute("""
                    SELECT prompt_name, score, created_at
                    FROM evaluations
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                evals = cursor.fetchall()

                if not evals:
                    return Static("No evaluations yet")

                content = "\n".join([
                    f"- {row[0]}: Score {row[1]:.2f if row[1] else 'N/A'} ({row[2]})"
                    for row in evals
                ])
                return Markdown(content)
        except:
            return Static("Error loading evaluations")


class ChainPanel(VerticalScroll):
    """Panel for managing chains"""

    def __init__(self, promptly: Promptly):
        super().__init__()
        self.promptly = promptly

    def compose(self) -> ComposeResult:
        """Compose the chain interface"""
        yield Static("# Prompt Chains", classes="header")

        try:
            with self.promptly._get_db() as db:
                cursor = db.conn.cursor()
                cursor.execute("SELECT name, steps, description FROM chains")
                chains = cursor.fetchall()

                if not chains:
                    yield Static("No chains found. Create one to get started!")
                    return

                for chain_row in chains:
                    name = chain_row[0]
                    steps = json.loads(chain_row[1])
                    description = chain_row[2] or "No description"

                    chain_info = f"""## {name}

{description}

**Steps:** {' → '.join(steps)}

"""
                    yield Markdown(chain_info)
                    yield Button(f"Run {name}", id=f"run_chain_{name}", variant="primary")
                    yield Static("---")

        except Exception as e:
            yield Static(f"Error loading chains: {e}")


class DiffPanel(VerticalScroll):
    """Panel for comparing prompt versions"""

    def __init__(self, promptly: Promptly):
        super().__init__()
        self.promptly = promptly

    def compose(self) -> ComposeResult:
        """Compose the diff interface"""
        yield Static("# Diff Viewer", classes="header")
        yield Static("Select two versions of a prompt to compare")

        # TODO: Implement version selection and diff display
        yield Static("\n[Feature coming soon: Side-by-side version comparison]")


class PromptlyTUI(App):
    """Main TUI Application"""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
    }

    #left_panel {
        column-span: 1;
        row-span: 2;
        border: solid cyan;
    }

    #main_content {
        column-span: 1;
        row-span: 2;
        border: solid green;
    }

    .header {
        text-style: bold;
        color: cyan;
    }

    Button {
        margin: 1;
    }

    DataTable {
        height: 100%;
    }

    Tree {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("1", "show_prompts", "Prompts"),
        Binding("2", "show_branches", "Branches"),
        Binding("3", "show_log", "Log"),
        Binding("4", "show_eval", "Eval"),
        Binding("5", "show_chains", "Chains"),
        Binding("ctrl+h", "help", "Help"),
    ]

    TITLE = "Promptly TUI"
    SUB_TITLE = "Prompt Management System"

    def __init__(self):
        super().__init__()
        try:
            self.promptly = Promptly()
            self.promptly._check_init()
        except PromptlyError:
            self.promptly = None

    def compose(self) -> ComposeResult:
        """Compose the application layout"""
        yield Header()

        if not self.promptly:
            yield Static("No promptly repository found. Please initialize one first.")
            yield Footer()
            return

        # Main layout with tabs
        with TabbedContent(initial="prompts"):
            with TabPane("Prompts", id="prompts"):
                with Horizontal():
                    with VerticalScroll(id="left_panel"):
                        yield PromptListPanel(self.promptly, on_select=self.on_prompt_selected)
                    with VerticalScroll(id="main_content"):
                        yield PromptViewer(id="prompt_viewer")

            with TabPane("Branches", id="branches"):
                yield BranchViewer(self.promptly)

            with TabPane("Log", id="log"):
                yield LogPanel(self.promptly)

            with TabPane("Eval", id="eval"):
                yield EvalPanel(self.promptly)

            with TabPane("Chains", id="chains"):
                yield ChainPanel(self.promptly)

            with TabPane("Diff", id="diff"):
                yield DiffPanel(self.promptly)

        yield Footer()

    def on_prompt_selected(self, prompt_name: str):
        """Handle prompt selection"""
        try:
            prompt_data = self.promptly.get(prompt_name)
            if prompt_data:
                viewer = self.query_one("#prompt_viewer", PromptViewer)
                viewer.update_content(prompt_data)
        except Exception as e:
            self.notify(f"Error loading prompt: {e}", severity="error")

    def action_refresh(self):
        """Refresh the current view"""
        self.refresh()
        self.notify("Refreshed")

    def action_show_prompts(self):
        """Show prompts tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "prompts"

    def action_show_branches(self):
        """Show branches tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "branches"

    def action_show_log(self):
        """Show log tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "log"

    def action_show_eval(self):
        """Show eval tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "eval"

    def action_show_chains(self):
        """Show chains tab"""
        tabs = self.query_one(TabbedContent)
        tabs.active = "chains"

    def action_help(self):
        """Show help"""
        help_text = """
# Promptly TUI Help

## Keyboard Shortcuts

- **q**: Quit application
- **r**: Refresh current view
- **1**: Show Prompts tab
- **2**: Show Branches tab
- **3**: Show Log tab
- **4**: Show Eval tab
- **5**: Show Chains tab
- **Ctrl+H**: Show this help
- **Tab**: Navigate between widgets

## Navigation

Use arrow keys or Tab to navigate between panels and elements.
Click on prompts in the left panel to view details.

## Tabs

- **Prompts**: Browse and view prompts
- **Branches**: View branch tree and structure
- **Log**: View commit history
- **Eval**: Run evaluations on prompts
- **Chains**: Manage and run prompt chains
- **Diff**: Compare prompt versions

Press any key to close this help.
"""
        self.push_screen(HelpScreen(help_text))


class HelpScreen(Screen):
    """Help screen overlay"""

    def __init__(self, help_text: str):
        super().__init__()
        self.help_text = help_text

    def compose(self) -> ComposeResult:
        with Container():
            yield Markdown(self.help_text)
            yield Button("Close", variant="primary", id="close_help")

    def on_button_pressed(self, event: Button.Pressed):
        """Close help on button press"""
        if event.button.id == "close_help":
            self.app.pop_screen()

    def on_key(self, event):
        """Close help on any key press"""
        self.app.pop_screen()


def main():
    """Entry point for TUI"""
    if not TEXTUAL_AVAILABLE:
        print("Error: textual not available. Install with: pip install textual")
        print("Run: pip install textual")
        return 1

    try:
        app = PromptlyTUI()
        app.run()
    except Exception as e:
        print(f"Error running TUI: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
