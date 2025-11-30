"""
HoloLoom TUI Pipeline Display
November 2025

Live pipeline visualization in terminal using Rich library.
Shows Parse → Retrieve → Decide → Generate stages with timing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import time

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class StageStatus(Enum):
    """Pipeline stage status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    BOTTLENECK = "bottleneck"
    ERROR = "error"


@dataclass
class PipelineStage:
    """Single pipeline stage."""
    name: str
    display_name: str
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    detail: str = ""

    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    @property
    def is_bottleneck(self) -> bool:
        """Check if stage is a bottleneck (>50ms)."""
        dur = self.duration_ms
        return dur is not None and dur > 50


@dataclass
class PipelineState:
    """Complete pipeline state."""
    stages: List[PipelineStage] = field(default_factory=list)
    query: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def total_duration_ms(self) -> Optional[float]:
        """Get total pipeline duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


class PipelineDisplay:
    """
    Terminal pipeline visualization.

    Shows live progress through Parse → Retrieve → Decide → Generate stages.
    """

    # Stage configuration
    DEFAULT_STAGES = [
        ("parse", "Parse"),
        ("retrieve", "Retrieve"),
        ("decide", "Decide"),
        ("generate", "Generate")
    ]

    # Status symbols
    STATUS_SYMBOLS = {
        StageStatus.PENDING: "○",
        StageStatus.RUNNING: "◐",
        StageStatus.COMPLETE: "●",
        StageStatus.BOTTLENECK: "⚠",
        StageStatus.ERROR: "✗"
    }

    # Status colors
    STATUS_COLORS = {
        StageStatus.PENDING: "dim",
        StageStatus.RUNNING: "yellow",
        StageStatus.COMPLETE: "green",
        StageStatus.BOTTLENECK: "red",
        StageStatus.ERROR: "red bold"
    }

    def __init__(self, console: Optional['Console'] = None):
        """Initialize pipeline display."""
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required. Install with: pip install rich")

        self.console = console or Console()
        self.state = PipelineState()
        self._live = None
        self._init_stages()

    def _init_stages(self):
        """Initialize pipeline stages."""
        self.state.stages = [
            PipelineStage(name=name, display_name=display)
            for name, display in self.DEFAULT_STAGES
        ]

    def reset(self):
        """Reset pipeline state."""
        self._init_stages()
        self.state.query = ""
        self.state.start_time = None
        self.state.end_time = None

    def start(self, query: str = ""):
        """Start pipeline execution."""
        self.reset()
        self.state.query = query
        self.state.start_time = time.time()

    def update_stage(self, stage_name: str, status: StageStatus, detail: str = ""):
        """Update a stage's status."""
        for stage in self.state.stages:
            if stage.name == stage_name:
                # Handle timing
                if status == StageStatus.RUNNING and stage.status == StageStatus.PENDING:
                    stage.start_time = time.time()
                elif status in (StageStatus.COMPLETE, StageStatus.BOTTLENECK, StageStatus.ERROR):
                    stage.end_time = time.time()
                    # Check for bottleneck
                    if stage.is_bottleneck and status == StageStatus.COMPLETE:
                        status = StageStatus.BOTTLENECK

                stage.status = status
                stage.detail = detail
                break

    def complete(self):
        """Mark pipeline as complete."""
        self.state.end_time = time.time()

    def render_stage_line(self, stage: PipelineStage) -> Text:
        """Render a single stage line."""
        text = Text()

        # Symbol
        symbol = self.STATUS_SYMBOLS[stage.status]
        color = self.STATUS_COLORS[stage.status]
        text.append(f"  {symbol} ", style=color)

        # Stage name (fixed width)
        name_style = "bold" if stage.status == StageStatus.RUNNING else ""
        text.append(f"{stage.display_name:<10}", style=name_style)

        # Progress bar
        if stage.status == StageStatus.PENDING:
            text.append(" [          ] ", style="dim")
        elif stage.status == StageStatus.RUNNING:
            # Animated progress
            elapsed = (time.time() - stage.start_time) * 10 if stage.start_time else 0
            filled = int(elapsed) % 10
            bar = "█" * filled + "░" * (10 - filled)
            text.append(f" [{bar}] ", style="yellow")
        else:
            text.append(" [██████████] ", style=color)

        # Timing
        if stage.duration_ms is not None:
            dur_str = f"{stage.duration_ms:.0f}ms"
            if stage.is_bottleneck:
                text.append(dur_str, style="red bold")
            else:
                text.append(dur_str, style="green")
        else:
            text.append("--", style="dim")

        # Detail
        if stage.detail:
            text.append(f"  {stage.detail}", style="dim italic")

        return text

    def render_ascii_pipeline(self) -> str:
        """Render pipeline as ASCII art (for non-Rich environments)."""
        lines = []
        lines.append("┌─────────────────────────────────────────────┐")
        lines.append("│          HoloLoom Pipeline Status           │")
        lines.append("├─────────────────────────────────────────────┤")

        for stage in self.state.stages:
            symbol = self.STATUS_SYMBOLS.get(stage.status, "?")
            dur = f"{stage.duration_ms:.0f}ms" if stage.duration_ms else "--"

            if stage.status == StageStatus.RUNNING:
                bar = "[▓▓▓▓░░░░░░]"
            elif stage.status in (StageStatus.COMPLETE, StageStatus.BOTTLENECK):
                bar = "[██████████]"
            else:
                bar = "[          ]"

            line = f"│ {symbol} {stage.display_name:<10} {bar} {dur:>6} │"
            lines.append(line)

        lines.append("├─────────────────────────────────────────────┤")

        # Total timing
        if self.state.total_duration_ms:
            total = f"Total: {self.state.total_duration_ms:.0f}ms"
        else:
            total = "Total: --"
        lines.append(f"│ {total:^43} │")
        lines.append("└─────────────────────────────────────────────┘")

        return "\n".join(lines)

    def render_panel(self) -> Panel:
        """Render pipeline as Rich Panel."""
        # Build content
        content = Text()

        # Query if present
        if self.state.query:
            content.append("Query: ", style="bold")
            query_display = self.state.query[:50] + "..." if len(self.state.query) > 50 else self.state.query
            content.append(f"{query_display}\n\n", style="italic")

        # Stages
        for stage in self.state.stages:
            content.append(self.render_stage_line(stage))
            content.append("\n")

        # Total timing
        content.append("\n")
        if self.state.total_duration_ms:
            content.append(f"Total: {self.state.total_duration_ms:.0f}ms", style="bold green")
        elif self.state.start_time:
            elapsed = (time.time() - self.state.start_time) * 1000
            content.append(f"Elapsed: {elapsed:.0f}ms", style="yellow")

        # Check for bottlenecks
        bottlenecks = [s for s in self.state.stages if s.status == StageStatus.BOTTLENECK]
        if bottlenecks:
            content.append("\n\n")
            content.append("⚠ Bottleneck detected: ", style="red bold")
            content.append(", ".join(s.display_name for s in bottlenecks), style="red")

        return Panel(
            content,
            title="[bold blue]Pipeline Status[/bold blue]",
            border_style="blue"
        )

    def print_pipeline(self):
        """Print current pipeline state to console."""
        self.console.print(self.render_panel())

    def run_simulation(self, query: str = "What is Thompson Sampling?"):
        """Run a simulated pipeline for demonstration."""
        import asyncio

        async def simulate():
            self.start(query)

            with Live(self.render_panel(), console=self.console, refresh_per_second=10) as live:
                # Parse stage
                self.update_stage("parse", StageStatus.RUNNING, "Analyzing query type")
                live.update(self.render_panel())
                await asyncio.sleep(0.02)
                self.update_stage("parse", StageStatus.COMPLETE, "factual|technical")
                live.update(self.render_panel())

                # Retrieve stage
                self.update_stage("retrieve", StageStatus.RUNNING, "Searching 12 shards")
                live.update(self.render_panel())
                await asyncio.sleep(0.06)
                self.update_stage("retrieve", StageStatus.COMPLETE, "Found 8 matches")
                live.update(self.render_panel())

                # Decide stage
                self.update_stage("decide", StageStatus.RUNNING, "Thompson draw...")
                live.update(self.render_panel())
                await asyncio.sleep(0.03)
                self.update_stage("decide", StageStatus.COMPLETE, "Tool: answer (0.89)")
                live.update(self.render_panel())

                # Generate stage
                self.update_stage("generate", StageStatus.RUNNING, "Streaming response")
                live.update(self.render_panel())
                await asyncio.sleep(0.1)
                self.update_stage("generate", StageStatus.COMPLETE, "142 tokens")

                self.complete()
                live.update(self.render_panel())

        asyncio.run(simulate())


class CompactPipelineDisplay:
    """
    Compact single-line pipeline display.

    Shows: ○ Parse → ◐ Retrieve → ○ Decide → ○ Generate
    """

    STAGES = ["Parse", "Retrieve", "Decide", "Generate"]

    def __init__(self, console: Optional['Console'] = None):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required")
        self.console = console or Console()
        self.current_stage = -1
        self.completed_stages: List[bool] = [False] * len(self.STAGES)

    def reset(self):
        """Reset display."""
        self.current_stage = -1
        self.completed_stages = [False] * len(self.STAGES)

    def advance(self):
        """Advance to next stage."""
        if self.current_stage >= 0:
            self.completed_stages[self.current_stage] = True
        self.current_stage += 1

    def render(self) -> Text:
        """Render compact pipeline."""
        text = Text()

        for i, stage in enumerate(self.STAGES):
            if self.completed_stages[i]:
                text.append("●", style="green")
            elif i == self.current_stage:
                text.append("◐", style="yellow bold")
            else:
                text.append("○", style="dim")

            text.append(f" {stage}", style="bold" if i == self.current_stage else "")

            if i < len(self.STAGES) - 1:
                text.append(" → ", style="dim")

        return text

    def print(self):
        """Print compact pipeline."""
        self.console.print(self.render())


# Quick test
if __name__ == "__main__":
    print("Testing Pipeline Display...")

    display = PipelineDisplay()
    display.run_simulation()

    print("\n\nCompact display test:")
    compact = CompactPipelineDisplay()
    for _ in range(5):
        compact.print()
        compact.advance()
        import time
        time.sleep(0.3)
