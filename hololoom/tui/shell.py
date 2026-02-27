"""
HoloLoom TUI Interactive Shell
November 2025

Full-featured terminal interface for HoloLoom.
Integrates pipeline, graph, and metrics displays.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import sys
import time

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.spinner import Spinner
    from rich.status import Status
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from hololoom.tui.pipeline_display import PipelineDisplay, CompactPipelineDisplay, StageStatus
from hololoom.tui.graph_display import GraphDisplay, SourceAttributionDisplay
from hololoom.tui.metrics_panel import MetricsPanel


@dataclass
class QueryResult:
    """Result of a query."""
    query: str
    response: str
    confidence: float
    latency_ms: float
    sources: List[Dict]
    tool_used: str
    cached: bool = False


class HoloLoomTUI:
    """
    Interactive terminal interface for HoloLoom.

    Provides:
    - Live pipeline visualization
    - Knowledge graph exploration
    - Metrics dashboard
    - Query history
    - Help system
    """

    BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║   ██╗  ██╗ ██████╗ ██╗      ██████╗ ██╗      ██████╗  ██████╗ ███╗   ███╗  ║
║   ██║  ██║██╔═══██╗██║     ██╔═══██╗██║     ██╔═══██╗██╔═══██╗████╗ ████║  ║
║   ███████║██║   ██║██║     ██║   ██║██║     ██║   ██║██║   ██║██╔████╔██║  ║
║   ██╔══██║██║   ██║██║     ██║   ██║██║     ██║   ██║██║   ██║██║╚██╔╝██║  ║
║   ██║  ██║╚██████╔╝███████╗╚██████╔╝███████╗╚██████╔╝╚██████╔╝██║ ╚═╝ ██║  ║
║   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝  ║
╠═══════════════════════════════════════════════════════════════╣
║                   Terminal User Interface                      ║
║         Type 'help' for commands, 'quit' to exit              ║
╚═══════════════════════════════════════════════════════════════╝
"""

    COMMANDS = {
        "help": "Show this help message",
        "quit": "Exit the shell",
        "clear": "Clear the screen",
        "status": "Show system status",
        "metrics": "Show metrics dashboard",
        "graph": "Show knowledge graph",
        "history": "Show query history",
        "sources": "Show last query sources",
        "pipeline": "Run pipeline demo",
        "config": "Show configuration",
        "user": "User management (Phase 3)",
        "share": "Share knowledge (Phase 3)",
    }

    def __init__(self, orchestrator=None, console: Optional['Console'] = None):
        """
        Initialize TUI shell.

        Args:
            orchestrator: HoloLoom orchestrator (optional, for real queries)
            console: Rich Console instance
        """
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required. Install with: pip install rich")

        self.console = console or Console()
        self.orchestrator = orchestrator
        self.running = False

        # Display components
        self.pipeline = PipelineDisplay(self.console)
        self.compact_pipeline = CompactPipelineDisplay(self.console)
        self.graph = GraphDisplay(self.console)
        self.sources_display = SourceAttributionDisplay(self.console)
        self.metrics = MetricsPanel(self.console)

        # State
        self.history: List[QueryResult] = []
        self.last_sources: List[Dict] = []
        self.current_user: Optional[str] = None  # Phase 3

        # Simulate some initial metrics
        self._init_demo_state()

    def _init_demo_state(self):
        """Initialize demo state."""
        # Sample metrics
        self.metrics.set_metric("activation", 0.85)
        self.metrics.set_metric("coherence", 0.72)
        self.metrics.set_metric("active_nodes", 24)
        self.metrics.set_metric("latency", 95)
        self.metrics.set_metric("cache_hit_rate", 0.78)
        self.metrics.set_metric("thompson_answer_alpha", 15.0)
        self.metrics.set_metric("thompson_answer_beta", 5.0)

        # Sample graph
        self.graph.add_node("hololoom", "hololoom", activation=0.9)
        self.graph.add_node("thompson", "Thompson Sampling", activation=0.8)
        self.graph.add_node("memory", "Memory System", activation=0.7)
        self.graph.add_node("weaving", "Weaving Cycle", activation=0.6)
        self.graph.add_edge("hololoom", "thompson", "USES")
        self.graph.add_edge("hololoom", "memory", "USES")
        self.graph.add_edge("hololoom", "weaving", "USES")
        self.graph.add_edge("weaving", "memory", "LEADS_TO")

    def print_banner(self):
        """Print welcome banner."""
        self.console.print(self.BANNER, style="bold blue")

    def print_help(self):
        """Print help message."""
        table = Table(title="Available Commands", border_style="blue")
        table.add_column("Command", style="cyan bold")
        table.add_column("Description")

        for cmd, desc in self.COMMANDS.items():
            table.add_row(cmd, desc)

        # Add query help
        table.add_row("[query]", "Any other input is treated as a query", style="dim")

        self.console.print(table)

    def print_status(self):
        """Print system status."""
        status = Table.grid(padding=1)
        status.add_column(style="dim")
        status.add_column(style="bold")

        # Connection status
        orch_status = "Connected" if self.orchestrator else "Demo Mode"
        orch_style = "green" if self.orchestrator else "yellow"
        status.add_row("Orchestrator:", f"[{orch_style}]{orch_status}[/{orch_style}]")

        # User (Phase 3)
        user = self.current_user or "Anonymous"
        status.add_row("Current User:", user)

        # Stats
        status.add_row("Queries:", str(len(self.history)))
        status.add_row("Graph Nodes:", str(len(self.graph.nodes)))

        self.console.print(Panel(status, title="System Status", border_style="green"))

    async def process_query(self, query: str) -> QueryResult:
        """
        Process a query through HoloLoom.

        Uses orchestrator if available, otherwise simulates.
        """
        start_time = time.time()

        # Show pipeline
        self.pipeline.start(query)

        with Live(self.pipeline.render_panel(), console=self.console, refresh_per_second=10) as live:
            if self.orchestrator:
                # Real query
                try:
                    from hololoom.protocols.types import Query
                    spacetime = await self.orchestrator.weave(Query(text=query))

                    # Update pipeline stages from trace
                    for stage_name in ["parse", "retrieve", "decide", "generate"]:
                        self.pipeline.update_stage(stage_name, StageStatus.COMPLETE)
                        live.update(self.pipeline.render_panel())

                    self.pipeline.complete()
                    live.update(self.pipeline.render_panel())

                    # Build result
                    result = QueryResult(
                        query=query,
                        response=str(spacetime.response),
                        confidence=spacetime.confidence,
                        latency_ms=(time.time() - start_time) * 1000,
                        sources=spacetime.metadata.get('sources', []),
                        tool_used=spacetime.metadata.get('tool_used', 'answer'),
                        cached=spacetime.metadata.get('cache_hit', False)
                    )

                    # Update metrics
                    self.metrics.update_from_spacetime(spacetime)

                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    raise
            else:
                # Simulated query
                stages = [
                    ("parse", 0.02, "factual|conceptual"),
                    ("retrieve", 0.05, "Found 8 matches"),
                    ("decide", 0.03, "Tool: answer (0.89)"),
                    ("generate", 0.08, "Streaming...")
                ]

                for stage_name, delay, detail in stages:
                    self.pipeline.update_stage(stage_name, StageStatus.RUNNING, detail)
                    live.update(self.pipeline.render_panel())
                    await asyncio.sleep(delay)
                    self.pipeline.update_stage(stage_name, StageStatus.COMPLETE, detail)
                    live.update(self.pipeline.render_panel())

                self.pipeline.complete()
                live.update(self.pipeline.render_panel())

                # Generate simulated response
                response = self._generate_demo_response(query)
                sources = self._generate_demo_sources(query)

                result = QueryResult(
                    query=query,
                    response=response,
                    confidence=0.87,
                    latency_ms=(time.time() - start_time) * 1000,
                    sources=sources,
                    tool_used="answer",
                    cached=False
                )

                # Update metrics
                self.metrics.set_metric("latency", result.latency_ms)
                self.metrics.set_metric("activation", result.confidence)

        # Store result
        self.history.append(result)
        self.last_sources = result.sources

        return result

    def _generate_demo_response(self, query: str) -> str:
        """Generate demo response."""
        query_lower = query.lower()

        if "thompson" in query_lower:
            return ("Thompson Sampling is a Bayesian approach to the exploration-exploitation "
                    "tradeoff. It maintains a probability distribution over possible action values "
                    "and samples from this distribution to select actions. This naturally balances "
                    "exploring uncertain options while exploiting known good choices.")
        elif "hololoom" in query_lower:
            return ("HoloLoom is a neural decision-making system that combines multi-scale "
                    "embeddings, knowledge graph memory, and Thompson Sampling exploration. "
                    "It uses a 'weaving' metaphor where independent modules are coordinated "
                    "by an orchestrator to produce responses.")
        elif "memory" in query_lower:
            return ("HoloLoom's memory system integrates 11 specialized subsystems including "
                    "vector memory, knowledge graphs, awareness tracking, and physics-based "
                    "spring dynamics. It supports hybrid retrieval combining BM25 and semantic "
                    "similarity search.")
        else:
            return f"Based on my knowledge, here's what I know about '{query}'..."

    def _generate_demo_sources(self, query: str) -> List[Dict]:
        """Generate demo sources."""
        return [
            {"id": "mem_001", "content": f"Primary source for {query[:20]}...", "relevance": 0.92},
            {"id": "mem_002", "content": "Related concept and context", "relevance": 0.78},
            {"id": "mem_003", "content": "Background information", "relevance": 0.65},
        ]

    def print_response(self, result: QueryResult):
        """Print query response."""
        # Response panel
        response_text = Text()
        response_text.append(result.response)

        self.console.print(Panel(
            response_text,
            title="[bold green]Response[/bold green]",
            subtitle=f"Confidence: {result.confidence:.0%} | Latency: {result.latency_ms:.0f}ms | Tool: {result.tool_used}",
            border_style="green"
        ))

    def print_sources(self, sources: List[Dict] = None):
        """Print source attribution."""
        sources = sources or self.last_sources
        if not sources:
            self.console.print("[dim]No sources available[/dim]")
            return
        self.sources_display.print_sources(sources)

    def print_history(self):
        """Print query history."""
        if not self.history:
            self.console.print("[dim]No query history[/dim]")
            return

        table = Table(title="Query History")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Query", max_width=40)
        table.add_column("Conf", justify="right")
        table.add_column("Lat", justify="right")
        table.add_column("Tool")

        for i, result in enumerate(self.history[-10:], 1):
            table.add_row(
                str(i),
                result.query[:40] + "..." if len(result.query) > 40 else result.query,
                f"{result.confidence:.0%}",
                f"{result.latency_ms:.0f}ms",
                result.tool_used
            )

        self.console.print(table)

    def print_config(self):
        """Print configuration."""
        config = Table(title="Configuration")
        config.add_column("Setting", style="cyan")
        config.add_column("Value")

        config.add_row("Mode", "Demo" if not self.orchestrator else "Production")
        config.add_row("User", self.current_user or "Anonymous")
        config.add_row("Graph Nodes", str(len(self.graph.nodes)))
        config.add_row("History Size", str(len(self.history)))

        self.console.print(config)

    # === Phase 3: Multi-User Features ===

    def cmd_user(self, args: str):
        """User management command."""
        parts = args.split()
        if not parts:
            # Show current user
            self.console.print(f"Current user: [bold]{self.current_user or 'Anonymous'}[/bold]")
            return

        subcmd = parts[0]
        if subcmd == "login":
            if len(parts) < 2:
                self.console.print("[red]Usage: user login <username>[/red]")
                return
            self.current_user = parts[1]
            self.console.print(f"[green]✓ Logged in as {self.current_user}[/green]")

        elif subcmd == "logout":
            if self.current_user:
                self.console.print(f"[yellow]Logged out from {self.current_user}[/yellow]")
                self.current_user = None
            else:
                self.console.print("[dim]Not logged in[/dim]")

        elif subcmd == "list":
            # Demo user list
            users = ["alice", "bob", "charlie", self.current_user]
            users = [u for u in users if u]
            self.console.print(f"Known users: {', '.join(users)}")

        else:
            self.console.print(f"[red]Unknown subcommand: {subcmd}[/red]")

    def cmd_share(self, args: str):
        """Share knowledge command (Phase 3)."""
        if not self.current_user:
            self.console.print("[yellow]⚠ Login required to share. Use: user login <name>[/yellow]")
            return

        parts = args.split()
        if not parts:
            self.console.print("[dim]Usage: share <node_id> [user|team|all][/dim]")
            return

        node_id = parts[0]
        scope = parts[1] if len(parts) > 1 else "team"

        # Check if node exists
        if node_id not in self.graph.nodes:
            self.console.print(f"[red]Node '{node_id}' not found[/red]")
            return

        # Simulate sharing
        self.console.print(f"[green]✓ Shared '{node_id}' with {scope}[/green]")
        self.console.print(f"  Attribution: {self.current_user}")
        self.console.print(f"  Visibility: {scope}")

    async def handle_command(self, line: str) -> bool:
        """
        Handle a command line.

        Returns False to exit, True to continue.
        """
        line = line.strip()

        if not line:
            return True

        # Parse command
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Built-in commands
        if cmd in ("quit", "exit", "q"):
            return False

        elif cmd == "help":
            self.print_help()

        elif cmd == "clear":
            self.console.clear()

        elif cmd == "status":
            self.print_status()

        elif cmd == "metrics":
            self.metrics.print_dashboard()

        elif cmd == "graph":
            mode = args if args in ("tree", "table", "ascii", "edges") else "tree"
            self.graph.print(mode=mode)

        elif cmd == "history":
            self.print_history()

        elif cmd == "sources":
            self.print_sources()

        elif cmd == "pipeline":
            self.pipeline.run_simulation()

        elif cmd == "config":
            self.print_config()

        elif cmd == "user":
            self.cmd_user(args)

        elif cmd == "share":
            self.cmd_share(args)

        else:
            # Treat as query
            try:
                result = await self.process_query(line)
                self.print_response(result)
            except Exception as e:
                self.console.print(f"[red]Error processing query: {e}[/red]")

        return True

    async def run(self):
        """Run the interactive shell."""
        self.running = True
        self.print_banner()

        # Status line
        self.console.print("[dim]Type 'help' for commands, 'quit' to exit[/dim]\n")

        while self.running:
            try:
                # Show prompt with compact status
                self.console.print()
                self.metrics.print_compact()

                # Get input
                line = Prompt.ask("[bold blue]hololoom[/bold blue]")

                # Handle command
                continue_running = await self.handle_command(line)
                if not continue_running:
                    self.running = False

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'quit' to exit[/yellow]")

            except EOFError:
                self.running = False

        self.console.print("\n[dim]Goodbye![/dim]")

    def run_sync(self):
        """Run synchronously (convenience method)."""
        asyncio.run(self.run())


class SimpleTUI:
    """
    Simplified TUI for basic terminal output.

    Works without Rich library (fallback mode).
    """

    def __init__(self):
        self.history: List[str] = []

    def print_banner(self):
        """Print ASCII banner."""
        print("""
╔═══════════════════════════════════════════════════════════╗
║                    H O L O L O O M                         ║
║                  Terminal Interface                        ║
╚═══════════════════════════════════════════════════════════╝
        """)

    def print_pipeline(self, stages: List[str]):
        """Print simple pipeline."""
        print("\nPipeline:")
        for i, stage in enumerate(stages):
            status = "●" if i < 2 else "○"
            print(f"  {status} {stage}")

    def run(self):
        """Run simple interactive loop."""
        self.print_banner()
        print("Type 'quit' to exit\n")

        while True:
            try:
                line = input("hololoom> ")
                if line.lower() in ("quit", "exit"):
                    break

                self.history.append(line)
                print(f"\n[Processing: {line}]\n")

            except (KeyboardInterrupt, EOFError):
                break

        print("\nGoodbye!")


# Entry point
def main():
    """Main entry point."""
    if RICH_AVAILABLE:
        tui = HoloLoomTUI()
        tui.run_sync()
    else:
        print("Rich library not found, using simple TUI")
        tui = SimpleTUI()
        tui.run()


if __name__ == "__main__":
    main()
