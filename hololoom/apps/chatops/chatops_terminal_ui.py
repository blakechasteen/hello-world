#!/usr/bin/env python3
"""
ChatOps Terminal UI - Complete System Visualization

Interactive terminal UI for the complete ChatOps system:
- Self-Improving Bot status
- Team Learning insights
- Workflow Marketplace browser
- Predictive Quality dashboard
- Multi-Agent coordination
- Live chat simulation

Run: python chatops_terminal_ui.py
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Button, Static, Input, TextArea, DataTable, TabbedContent, TabPane, Label, Tree, ProgressBar
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table as RichTable
from datetime import datetime
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from self_improving_bot import SelfImprovingBot
    from team_learning import TeamLearningSystem
    from workflow_marketplace import WorkflowMarketplace
    from predictive_quality import PredictiveQualitySystem
    from multi_agent import MultiAgentSystem, AgentRole
    CHATOPS_AVAILABLE = True
except ImportError as e:
    CHATOPS_AVAILABLE = False
    print(f"ChatOps import error: {e}")


class SystemStatusPanel(Static):
    """Overall system status"""

    status = reactive("Initializing...")
    total_queries = reactive(0)
    avg_quality = reactive(0.0)
    active_experiments = reactive(0)

    def render(self) -> Panel:
        content = f"""
[cyan]Status:[/cyan] {self.status}
[green]Total Queries:[/green] {self.total_queries}
[yellow]Avg Quality:[/yellow] {self.avg_quality:.2f}
[magenta]Active Experiments:[/magenta] {self.active_experiments}
"""
        return Panel(content, title="[bold cyan]ChatOps System", border_style="cyan")


class SelfImprovingTab(Vertical):
    """Self-Improving Bot dashboard"""

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]Self-Improving Bot Dashboard", id="si-label")

        # Stats table
        table = DataTable(id="si-stats-table")
        table.add_columns("Metric", "Value")
        yield table

        # Active experiments
        yield Label("[bold green]Active Experiments", id="exp-label")
        exp_table = DataTable(id="exp-table")
        exp_table.add_columns("Experiment", "Status", "Variant A", "Variant B", "Winner")
        yield exp_table

        # Patterns learned
        yield Label("[bold yellow]Patterns Learned", id="patterns-label")
        patterns_area = TextArea("Loading patterns...", id="patterns-area", read_only=True)
        yield patterns_area


class TeamLearningTab(Vertical):
    """Team Learning insights"""

    def compose(self) -> ComposeResult:
        yield Label("[bold yellow]Team Learning System", id="tl-label")

        # Training examples
        yield Label("High-Quality Training Examples", id="train-label")
        train_table = DataTable(id="train-table")
        train_table.add_columns("ID", "Query Type", "Quality Score", "Usage")
        yield train_table

        # Best practices
        yield Label("[bold green]Best Practices Identified", id="bp-label")
        bp_area = TextArea("Loading best practices...", id="bp-area", read_only=True)
        yield bp_area

        # Expert profiles
        yield Label("[bold blue]Expert Profiles", id="expert-label")
        expert_table = DataTable(id="expert-table")
        expert_table.add_columns("User", "Expertise", "Contributions", "Avg Quality")
        yield expert_table


class WorkflowMarketplaceTab(Vertical):
    """Workflow Marketplace browser"""

    def compose(self) -> ComposeResult:
        yield Label("[bold magenta]Workflow Marketplace", id="wm-label")

        # Search
        yield Horizontal(
            Input(placeholder="Search workflows...", id="wm-search"),
            Button("Search", variant="primary", id="wm-search-btn"),
            id="wm-search-controls"
        )

        # Workflows table
        table = DataTable(id="wm-table")
        table.add_columns("Workflow", "Category", "Rating", "Downloads", "Author")
        yield table

        # Installed workflows
        yield Label("[bold green]Installed Workflows", id="installed-label")
        installed_table = DataTable(id="installed-table")
        installed_table.add_columns("Workflow", "Version", "Installed", "Auto-Update")
        yield installed_table


class PredictiveQualityTab(Vertical):
    """Predictive quality dashboard"""

    def compose(self) -> ComposeResult:
        yield Label("[bold blue]Predictive Quality System", id="pq-label")

        # Prediction accuracy
        yield Label("Prediction Performance", id="perf-label")
        perf_table = DataTable(id="perf-table")
        perf_table.add_columns("Metric", "Value", "Trend")
        yield perf_table

        # Query difficulty distribution
        yield Label("[bold green]Query Difficulty Distribution", id="diff-label")
        diff_area = TextArea("Loading difficulty stats...", id="diff-area", read_only=True)
        yield diff_area

        # Feature weights
        yield Label("[bold yellow]Learned Feature Weights", id="weights-label")
        weights_table = DataTable(id="weights-table")
        weights_table.add_columns("Feature", "Weight", "Impact")
        yield weights_table


class MultiAgentTab(Vertical):
    """Multi-Agent system dashboard"""

    def compose(self) -> ComposeResult:
        yield Label("[bold green]Multi-Agent System", id="ma-label")

        # Agent status
        yield Label("Agent Status", id="agent-status-label")
        agent_table = DataTable(id="agent-table")
        agent_table.add_columns("Agent", "Role", "Status", "Tasks", "Avg Quality")
        yield agent_table

        # Collaboration history
        yield Label("[bold cyan]Recent Collaborations", id="collab-label")
        collab_table = DataTable(id="collab-table")
        collab_table.add_columns("Task", "Agents", "Status", "Quality")
        yield collab_table

        # Routing statistics
        yield Label("[bold yellow]Routing Statistics", id="route-label")
        route_area = TextArea("Loading routing stats...", id="route-area", read_only=True)
        yield route_area


class LiveChatTab(Vertical):
    """Live chat simulation"""

    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]Live ChatOps Simulation", id="chat-label")

        # Chat area
        chat_area = TextArea("# Welcome to ChatOps\n\nType a query below...", id="chat-area", read_only=True)
        yield chat_area

        # Input
        yield Horizontal(
            Input(placeholder="Enter query...", id="chat-input"),
            Button("Send", variant="primary", id="send-btn"),
            Button("Auto-Demo", variant="success", id="demo-btn"),
            id="chat-controls"
        )

        # System response details
        yield Label("[bold green]Response Details", id="details-label")
        details_area = TextArea("Response details will appear here...", id="details-area", read_only=True)
        yield details_area


class ChatOpsTerminalApp(App):
    """Main ChatOps Terminal UI"""

    CSS = """
    Screen {
        background: $surface;
    }

    Label {
        padding: 1;
        background: $boost;
        color: $text;
    }

    Input {
        margin: 1;
    }

    Button {
        margin-right: 1;
    }

    DataTable {
        height: 1fr;
        margin: 1;
    }

    TextArea {
        height: 1fr;
        margin: 1;
    }

    SystemStatusPanel {
        dock: top;
        height: 7;
    }

    #chat-controls, #wm-search-controls {
        height: auto;
        margin: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("ctrl+d", "demo", "Demo"),
    ]

    def __init__(self):
        super().__init__()
        self.system_ready = False

        # Initialize systems (if available)
        if CHATOPS_AVAILABLE:
            self.self_improving = SelfImprovingBot()
            self.team_learning = TeamLearningSystem()
            self.marketplace = WorkflowMarketplace()
            self.predictive = PredictiveQualitySystem()
            self.multi_agent = MultiAgentSystem()
        else:
            self.self_improving = None
            self.team_learning = None
            self.marketplace = None
            self.predictive = None
            self.multi_agent = None

        self.query_count = 0
        self.total_quality = 0.0

    def compose(self) -> ComposeResult:
        yield Header()
        yield SystemStatusPanel()

        with TabbedContent():
            with TabPane("Chat", id="tab-chat"):
                yield LiveChatTab()

            with TabPane("Self-Improving", id="tab-si"):
                yield SelfImprovingTab()

            with TabPane("Team Learning", id="tab-tl"):
                yield TeamLearningTab()

            with TabPane("Marketplace", id="tab-wm"):
                yield WorkflowMarketplaceTab()

            with TabPane("Predictive Quality", id="tab-pq"):
                yield PredictiveQualityTab()

            with TabPane("Multi-Agent", id="tab-ma"):
                yield MultiAgentTab()

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize on startup"""
        status = self.query_one(SystemStatusPanel)

        if CHATOPS_AVAILABLE:
            status.status = "Ready - All Systems Online"
            self.system_ready = True

            # Initialize dashboards
            await self.update_all_dashboards()
        else:
            status.status = "Error - Systems Unavailable"

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        button_id = event.button.id

        if button_id == "send-btn":
            await self.send_chat_message()
        elif button_id == "demo-btn":
            await self.run_auto_demo()
        elif button_id == "wm-search-btn":
            await self.search_workflows()

    async def send_chat_message(self):
        """Process chat query"""
        if not self.system_ready:
            return

        # Get query
        chat_input = self.query_one("#chat-input", Input)
        query = chat_input.value.strip()

        if not query:
            return

        chat_area = self.query_one("#chat-area", TextArea)
        details_area = self.query_one("#details-area", TextArea)

        # Show query
        current = chat_area.text
        chat_area.load_text(f"{current}\n\n**User:** {query}\n\n**Bot:** Processing...")

        try:
            # 1. Predict quality
            prediction = await self.predictive.predict_quality(query)

            # 2. Route to agent
            response = await self.multi_agent.route(query)

            # 3. Learn from outcome
            self.predictive.learn(
                query,
                prediction.predicted_quality,
                response.get("quality_score", 0.0)
            )

            # Update stats
            self.query_count += 1
            self.total_quality += response.get("quality_score", 0.0)

            # Display response
            bot_response = response.get("answer", "No response")
            chat_area.load_text(f"{current}\n\n**User:** {query}\n\n**Bot:** {bot_response[:200]}...")

            # Show details
            details = f"""# Response Details

**Agent:** {response.get('agent_name', 'Unknown')} ({response.get('agent_role', 'unknown')})
**Quality Score:** {response.get('quality_score', 0.0):.2f}

## Prediction
- **Predicted Quality:** {prediction.predicted_quality:.2f}
- **Difficulty:** {prediction.difficulty_score:.2f}
- **Retry Probability:** {prediction.predicted_retry_probability:.1%}

## Configuration Used
{self._format_dict(prediction.recommended_config)}

## Sections
- **TL;DR:** {response.get('tldr', 'N/A')[:100]}...
"""
            details_area.load_text(details)

            # Update status
            await self.update_system_status()

        except Exception as e:
            chat_area.load_text(f"{current}\n\n**User:** {query}\n\n**Bot:** Error: {e}")

        # Clear input
        chat_input.value = ""

    async def run_auto_demo(self):
        """Run automated demo"""
        demo_queries = [
            "The API is returning 500 errors",
            "Review PR #123",
            "Deploy v2.1.0 to production",
            "Who are the experts on Kubernetes?"
        ]

        for query in demo_queries:
            chat_input = self.query_one("#chat-input", Input)
            chat_input.value = query
            await self.send_chat_message()
            await asyncio.sleep(2)

    async def search_workflows(self):
        """Search workflow marketplace"""
        if not self.marketplace:
            return

        search_input = self.query_one("#wm-search", Input)
        query = search_input.value.strip()

        results = self.marketplace.search(query=query if query else None)

        table = self.query_one("#wm-table", DataTable)
        table.clear()

        for workflow in results[:10]:
            table.add_row(
                workflow.metadata.name,
                workflow.metadata.category,
                f"{workflow.metadata.rating:.1f}",
                str(workflow.metadata.downloads),
                workflow.metadata.author
            )

    async def update_all_dashboards(self):
        """Update all dashboard tabs"""
        await self.update_self_improving_dashboard()
        await self.update_team_learning_dashboard()
        await self.update_marketplace_dashboard()
        await self.update_predictive_dashboard()
        await self.update_multi_agent_dashboard()

    async def update_self_improving_dashboard(self):
        """Update self-improving bot dashboard"""
        if not self.self_improving:
            return

        # Stats
        stats = self.self_improving.get_improvement_stats()
        table = self.query_one("#si-stats-table", DataTable)
        table.clear()

        for key, value in stats.items():
            if not isinstance(value, dict):
                table.add_row(key, str(value))

        # Experiments
        exp_table = self.query_one("#exp-table", DataTable)
        exp_table.clear()

        for exp_id, exp in list(self.self_improving.experiments.items())[:5]:
            exp_table.add_row(
                exp.name,
                exp.status,
                "Control",
                "Variant",
                exp.winner or "TBD"
            )

    async def update_team_learning_dashboard(self):
        """Update team learning dashboard"""
        if not self.team_learning:
            return

        # Training examples
        examples = self.team_learning.get_training_examples(limit=10)
        table = self.query_one("#train-table", DataTable)
        table.clear()

        for ex in examples:
            table.add_row(
                ex.example_id[:15],
                ex.metadata.get("query_type", "general"),
                f"{ex.quality_score:.2f}",
                str(ex.usage_count)
            )

        # Best practices
        bp_area = self.query_one("#bp-area", TextArea)
        bp_text = "# Best Practices\n\n"
        for bp_id, bp in list(self.team_learning.best_practices.items())[:5]:
            bp_text += f"## {bp.title}\n{bp.description}\n\n"
        bp_area.load_text(bp_text)

    async def update_marketplace_dashboard(self):
        """Update marketplace dashboard"""
        if not self.marketplace:
            return

        # Workflows
        workflows = self.marketplace.search()
        table = self.query_one("#wm-table", DataTable)
        table.clear()

        for wf in workflows[:10]:
            table.add_row(
                wf.metadata.name,
                wf.metadata.category,
                f"{wf.metadata.rating:.1f}",
                str(wf.metadata.downloads),
                wf.metadata.author
            )

        # Installed
        installed = self.marketplace.list_installed()
        inst_table = self.query_one("#installed-table", DataTable)
        inst_table.clear()

        for inst in installed[:5]:
            inst_table.add_row(
                inst.workflow_id,
                inst.version,
                inst.installed_at.strftime("%Y-%m-%d"),
                "Yes" if inst.auto_update else "No"
            )

    async def update_predictive_dashboard(self):
        """Update predictive quality dashboard"""
        if not self.predictive:
            return

        # Performance stats
        stats = self.predictive.get_statistics()
        table = self.query_one("#perf-table", DataTable)
        table.clear()

        for key, value in stats.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.3f}", "↑")
            elif isinstance(value, int):
                table.add_row(key, str(value), "→")

        # Feature weights
        weights_table = self.query_one("#weights-table", DataTable)
        weights_table.clear()

        for feature, weight in self.predictive.feature_weights.items():
            impact = "High" if weight > 0.15 else "Medium" if weight > 0.08 else "Low"
            weights_table.add_row(feature, f"{weight:.3f}", impact)

    async def update_multi_agent_dashboard(self):
        """Update multi-agent dashboard"""
        if not self.multi_agent:
            return

        # Agent status
        agents = self.multi_agent.list_agents()
        table = self.query_one("#agent-table", DataTable)
        table.clear()

        for agent in agents:
            table.add_row(
                agent["name"],
                agent["role"],
                "Available" if agent["available"] else "Busy",
                str(agent["active_tasks"]),
                f"{agent['avg_quality_score']:.2f}"
            )

    async def update_system_status(self):
        """Update system status panel"""
        status = self.query_one(SystemStatusPanel)
        status.total_queries = self.query_count
        status.avg_quality = self.total_quality / max(self.query_count, 1)

        if self.self_improving:
            stats = self.self_improving.get_improvement_stats()
            status.active_experiments = stats.get("active_experiments", 0)

    def _format_dict(self, d: dict) -> str:
        """Format dict as markdown"""
        return "\n".join(f"- **{k}:** {v}" for k, v in d.items())

    async def action_refresh(self):
        """Refresh all dashboards"""
        await self.update_all_dashboards()

    async def action_demo(self):
        """Run demo"""
        await self.run_auto_demo()


def main():
    """Run the terminal UI"""
    app = ChatOpsTerminalApp()
    app.run()


if __name__ == "__main__":
    main()
