"""
HoloLoom TUI Metrics Panel
November 2025

Terminal dashboard showing system metrics, awareness, and performance.
Tufte-style data visualization in text format.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.text import Text
    from rich.columns import Columns
    from rich.progress import Progress, BarColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class MetricValue:
    """Single metric with history."""
    name: str
    current: float
    unit: str = ""
    history: List[float] = None
    min_val: float = 0.0
    max_val: float = 1.0

    def __post_init__(self):
        if self.history is None:
            self.history = []

    @property
    def trend(self) -> str:
        """Get trend indicator."""
        if len(self.history) < 2:
            return "─"
        if self.history[-1] > self.history[-2]:
            return "↑"
        elif self.history[-1] < self.history[-2]:
            return "↓"
        return "─"

    @property
    def sparkline(self) -> str:
        """Create ASCII sparkline from history."""
        if not self.history or len(self.history) < 2:
            return "▁▁▁▁▁"

        recent = self.history[-10:]  # Last 10 values
        if max(recent) == min(recent):
            return "▄" * len(recent)

        chars = "▁▂▃▄▅▆▇█"
        normalized = [
            (v - min(recent)) / (max(recent) - min(recent))
            for v in recent
        ]
        return "".join(chars[min(7, int(v * 7))] for v in normalized)


class MetricsPanel:
    """
    Terminal metrics dashboard.

    Shows:
    - Awareness metrics (activation, coherence)
    - Performance metrics (latency, throughput)
    - Cache metrics (hit rate, size)
    - Learning metrics (Thompson priors)
    """

    def __init__(self, console: Optional['Console'] = None):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required")

        self.console = console or Console()
        self.metrics: Dict[str, MetricValue] = {}
        self.start_time = time.time()

    def set_metric(self, name: str, value: float, unit: str = "",
                   min_val: float = 0.0, max_val: float = 1.0):
        """Set or update a metric."""
        if name in self.metrics:
            self.metrics[name].current = value
            self.metrics[name].history.append(value)
            # Keep last 100 values
            if len(self.metrics[name].history) > 100:
                self.metrics[name].history = self.metrics[name].history[-100:]
        else:
            self.metrics[name] = MetricValue(
                name=name,
                current=value,
                unit=unit,
                history=[value],
                min_val=min_val,
                max_val=max_val
            )

    def _format_value(self, metric: MetricValue) -> Text:
        """Format metric value with styling."""
        text = Text()

        # Value with unit
        if metric.max_val == 1.0 and metric.min_val == 0.0:
            # Percentage-like
            pct = metric.current * 100
            if pct >= 80:
                style = "green bold"
            elif pct >= 50:
                style = "yellow"
            else:
                style = "red"
            text.append(f"{pct:.0f}%", style=style)
        else:
            text.append(f"{metric.current:.2f}{metric.unit}", style="bold")

        # Trend indicator
        trend_color = "green" if metric.trend == "↑" else "red" if metric.trend == "↓" else "dim"
        text.append(f" {metric.trend}", style=trend_color)

        return text

    def _progress_bar(self, value: float, width: int = 15,
                      low_color: str = "red", mid_color: str = "yellow",
                      high_color: str = "green") -> str:
        """Create ASCII progress bar."""
        filled = int(value * width)
        if value >= 0.7:
            color = high_color
        elif value >= 0.4:
            color = mid_color
        else:
            color = low_color

        bar = "█" * filled + "░" * (width - filled)
        return f"[{color}]{bar}[/{color}]"

    def render_awareness_panel(self) -> Panel:
        """Render awareness metrics panel."""
        content = Text()

        # Activation
        activation = self.metrics.get("activation", MetricValue("activation", 0.0))
        content.append("Activation  ", style="dim")
        content.append(self._progress_bar(activation.current))
        content.append(f" {activation.current:.0%}\n")

        # Coherence
        coherence = self.metrics.get("coherence", MetricValue("coherence", 0.0))
        content.append("Coherence   ", style="dim")
        content.append(self._progress_bar(coherence.current))
        content.append(f" {coherence.current:.0%}\n")

        # Active nodes
        active_nodes = self.metrics.get("active_nodes", MetricValue("active_nodes", 0, max_val=100))
        content.append("Active      ", style="dim")
        content.append(f"[bold cyan]{int(active_nodes.current)}[/bold cyan] nodes\n")

        # Sparklines
        content.append("\n")
        content.append("Trends: ", style="dim")
        content.append(f"[cyan]{activation.sparkline}[/cyan] activation  ")
        content.append(f"[green]{coherence.sparkline}[/green] coherence")

        return Panel(content, title="[bold blue]Awareness[/bold blue]", border_style="blue")

    def render_performance_panel(self) -> Panel:
        """Render performance metrics panel."""
        content = Text()

        # Latency
        latency = self.metrics.get("latency", MetricValue("latency", 0, "ms", max_val=500))
        lat_style = "green" if latency.current < 100 else "yellow" if latency.current < 200 else "red"
        content.append("Latency     ", style="dim")
        content.append(f"[{lat_style}]{latency.current:.0f}ms[/{lat_style}]")
        content.append(f" {latency.trend}\n", style="dim")

        # Throughput
        throughput = self.metrics.get("throughput", MetricValue("throughput", 0, "/s", max_val=100))
        content.append("Throughput  ", style="dim")
        content.append(f"[bold]{throughput.current:.1f}[/bold] q/s\n")

        # Tokens
        tokens = self.metrics.get("tokens", MetricValue("tokens", 0, max_val=1000))
        content.append("Tokens      ", style="dim")
        content.append(f"[bold]{int(tokens.current)}[/bold] last response\n")

        # Uptime
        uptime = time.time() - self.start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        content.append("\n")
        content.append(f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}", style="dim")

        return Panel(content, title="[bold green]Performance[/bold green]", border_style="green")

    def render_cache_panel(self) -> Panel:
        """Render cache metrics panel."""
        content = Text()

        # Hit rate
        hit_rate = self.metrics.get("cache_hit_rate", MetricValue("cache_hit_rate", 0.0))
        content.append("Hit Rate    ", style="dim")
        content.append(self._progress_bar(hit_rate.current))
        content.append(f" {hit_rate.current:.0%}\n")

        # Cache size
        cache_size = self.metrics.get("cache_size", MetricValue("cache_size", 0, max_val=10000))
        content.append("Size        ", style="dim")
        content.append(f"[bold]{int(cache_size.current):,}[/bold] entries\n")

        # Memory
        memory_mb = self.metrics.get("memory_mb", MetricValue("memory_mb", 0, "MB", max_val=1000))
        content.append("Memory      ", style="dim")
        content.append(f"[bold]{memory_mb.current:.0f}[/bold] MB\n")

        # Speedup
        if hit_rate.current > 0:
            # Assume 100x speedup for cache hits
            speedup = hit_rate.current * 100
            content.append("\n")
            content.append(f"Est. Speedup: ", style="dim")
            content.append(f"[bold green]{speedup:.0f}x[/bold green] avg")

        return Panel(content, title="[bold yellow]Cache[/bold yellow]", border_style="yellow")

    def render_thompson_panel(self) -> Panel:
        """Render Thompson Sampling metrics panel."""
        content = Text()

        # Get tool priors
        tools = ["answer", "research", "clarify", "execute"]
        for tool in tools:
            alpha = self.metrics.get(f"thompson_{tool}_alpha", MetricValue(f"thompson_{tool}_alpha", 1.0, max_val=100))
            beta = self.metrics.get(f"thompson_{tool}_beta", MetricValue(f"thompson_{tool}_beta", 1.0, max_val=100))

            # Expected reward
            expected = alpha.current / (alpha.current + beta.current)
            bar = self._progress_bar(expected, width=10)

            content.append(f"{tool:<10}", style="bold")
            content.append(bar)
            content.append(f" α:{alpha.current:.0f} β:{beta.current:.0f}\n")

        # Exploration rate
        exploration = self.metrics.get("exploration_rate", MetricValue("exploration_rate", 0.1))
        content.append("\n")
        content.append("Exploration: ", style="dim")
        content.append(f"[bold]{exploration.current:.0%}[/bold]")

        return Panel(content, title="[bold magenta]Thompson Sampling[/bold magenta]", border_style="magenta")

    def render_dashboard(self) -> Layout:
        """Render full dashboard layout."""
        layout = Layout()

        layout.split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(self.render_awareness_panel()),
            Layout(self.render_cache_panel())
        )

        layout["right"].split_column(
            Layout(self.render_performance_panel()),
            Layout(self.render_thompson_panel())
        )

        return layout

    def render_compact(self) -> Text:
        """Render single-line compact metrics."""
        text = Text()

        # Key metrics
        activation = self.metrics.get("activation", MetricValue("activation", 0.0))
        latency = self.metrics.get("latency", MetricValue("latency", 0.0))
        cache_hit = self.metrics.get("cache_hit_rate", MetricValue("cache_hit_rate", 0.0))

        text.append("Act:", style="dim")
        text.append(f"{activation.current:.0%} ", style="cyan")
        text.append("Lat:", style="dim")
        text.append(f"{latency.current:.0f}ms ", style="yellow")
        text.append("Cache:", style="dim")
        text.append(f"{cache_hit.current:.0%}", style="green")

        return text

    def print_dashboard(self):
        """Print full dashboard."""
        self.console.print(self.render_dashboard())

    def print_compact(self):
        """Print compact metrics line."""
        self.console.print(self.render_compact())

    def update_from_spacetime(self, spacetime: Any):
        """Update metrics from HoloLoom Spacetime result."""
        if hasattr(spacetime, 'confidence'):
            self.set_metric("activation", spacetime.confidence)

        if hasattr(spacetime, 'metadata'):
            meta = spacetime.metadata
            if 'awareness' in meta:
                awareness = meta['awareness']
                self.set_metric("activation", awareness.get('activation_level', 0.0))
                self.set_metric("coherence", awareness.get('coherence', 0.0))
                self.set_metric("active_nodes", awareness.get('active_nodes', 0))

            if 'latency_ms' in meta:
                self.set_metric("latency", meta['latency_ms'])

            if 'cache_hit' in meta:
                self.set_metric("cache_hit_rate", 1.0 if meta['cache_hit'] else 0.0)


class QueryStatsDisplay:
    """Display query-level statistics."""

    def __init__(self, console: Optional['Console'] = None):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required")
        self.console = console or Console()
        self.query_count = 0
        self.total_latency = 0.0
        self.cache_hits = 0

    def record_query(self, latency_ms: float, cache_hit: bool = False):
        """Record a query."""
        self.query_count += 1
        self.total_latency += latency_ms
        if cache_hit:
            self.cache_hits += 1

    def render(self) -> Panel:
        """Render query stats."""
        content = Text()

        content.append("Queries: ", style="dim")
        content.append(f"[bold]{self.query_count}[/bold]\n")

        avg_latency = self.total_latency / max(1, self.query_count)
        content.append("Avg Latency: ", style="dim")
        content.append(f"[bold]{avg_latency:.0f}ms[/bold]\n")

        hit_rate = self.cache_hits / max(1, self.query_count)
        content.append("Cache Hit Rate: ", style="dim")
        content.append(f"[bold]{hit_rate:.0%}[/bold]")

        return Panel(content, title="Query Stats", border_style="cyan")


# Quick test
if __name__ == "__main__":
    print("Testing Metrics Panel...")

    panel = MetricsPanel()

    # Set some sample metrics
    panel.set_metric("activation", 0.85)
    panel.set_metric("coherence", 0.72)
    panel.set_metric("active_nodes", 15, max_val=100)
    panel.set_metric("latency", 95, "ms", max_val=500)
    panel.set_metric("throughput", 12.5, "/s", max_val=100)
    panel.set_metric("tokens", 142, max_val=1000)
    panel.set_metric("cache_hit_rate", 0.78)
    panel.set_metric("cache_size", 5432, max_val=10000)
    panel.set_metric("memory_mb", 256, "MB", max_val=1000)
    panel.set_metric("thompson_answer_alpha", 15.0)
    panel.set_metric("thompson_answer_beta", 5.0)
    panel.set_metric("thompson_research_alpha", 8.0)
    panel.set_metric("thompson_research_beta", 7.0)
    panel.set_metric("thompson_clarify_alpha", 3.0)
    panel.set_metric("thompson_clarify_beta", 4.0)
    panel.set_metric("thompson_execute_alpha", 2.0)
    panel.set_metric("thompson_execute_beta", 8.0)
    panel.set_metric("exploration_rate", 0.15)

    # Simulate some history
    import random
    for _ in range(20):
        panel.set_metric("activation", 0.7 + random.random() * 0.25)
        panel.set_metric("coherence", 0.6 + random.random() * 0.3)

    print("\n=== Full Dashboard ===")
    panel.print_dashboard()

    print("\n=== Compact View ===")
    panel.print_compact()
