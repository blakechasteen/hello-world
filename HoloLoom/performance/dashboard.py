#!/usr/bin/env python3
"""
Performance Dashboard - Real-time monitoring terminal UI
========================================================
Live terminal dashboard for monitoring HoloLoom performance metrics.

Usage:
    python -m HoloLoom.performance.dashboard

Features:
- Real-time latency charts
- System resource monitoring
- Cache hit rates
- Query throughput
- Component-level breakdowns
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Warning: rich not installed - dashboard UI disabled")
    print("Install with: pip install rich")

from HoloLoom.performance.metrics import get_global_metrics
from HoloLoom.performance.profiler import get_global_registry


class PerformanceDashboard:
    """
    Live terminal dashboard for HoloLoom performance monitoring.

    Displays:
    - Query latency (current, p50, p95, p99)
    - System resources (CPU, memory)
    - Cache hit rates
    - Query throughput (QPS)
    - Component breakdowns
    """

    def __init__(self):
        self.console = Console() if HAS_RICH else None
        self.metrics = get_global_metrics()
        self.registry = get_global_registry()
        self.running = False

    def create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(name="latency"),
            Layout(name="throughput")
        )

        layout["right"].split_column(
            Layout(name="system"),
            Layout(name="cache")
        )

        return layout

    def generate_header(self) -> Panel:
        """Generate header panel."""
        if not HAS_RICH:
            return "HoloLoom Performance Dashboard"

        header = Table.grid(expand=True)
        header.add_column(justify="left")
        header.add_column(justify="center")
        header.add_column(justify="right")

        header.add_row(
            "[bold blue]HoloLoom Performance Dashboard[/]",
            "[yellow]Monitoring System Health[/]",
            f"[green]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]"
        )

        return Panel(header, style="white on blue")

    def generate_latency_panel(self) -> Panel:
        """Generate latency metrics panel."""
        if not HAS_RICH:
            return "Latency Metrics"

        stats = self.metrics.get_latency_stats("query_processing")

        table = Table(box=box.SIMPLE, show_header=False, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        if stats:
            table.add_row("Count", str(stats["count"]))
            table.add_row("Mean", f"{stats['mean']:.2f} ms")
            table.add_row("Median (P50)", f"{stats['median']:.2f} ms")
            table.add_row("P95", f"{stats['p95']:.2f} ms")
            table.add_row("P99", f"{stats['p99']:.2f} ms")
            table.add_row("Min", f"{stats['min']:.2f} ms")
            table.add_row("Max", f"{stats['max']:.2f} ms")
        else:
            table.add_row("No data", "N/A")

        return Panel(table, title="[bold]Query Latency[/]", border_style="cyan")

    def generate_throughput_panel(self) -> Panel:
        """Generate throughput metrics panel."""
        if not HAS_RICH:
            return "Throughput Metrics"

        qps = self.metrics.get_throughput("queries", window_seconds=60)
        total = self.metrics.get_counter("total_queries")

        table = Table(box=box.SIMPLE, show_header=False, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Queries/Second", f"{qps:.2f}")
        table.add_row("Total Queries", str(total))
        table.add_row("Window", "60 seconds")

        return Panel(table, title="[bold]Throughput[/]", border_style="green")

    def generate_system_panel(self) -> Panel:
        """Generate system metrics panel."""
        if not HAS_RICH:
            return "System Metrics"

        system = self.metrics.get_system_metrics()

        table = Table(box=box.SIMPLE, show_header=False, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        cpu = system.get("cpu_percent", 0.0)
        memory_mb = system.get("memory_mb", 0.0)
        memory_pct = system.get("memory_percent", 0.0)

        # Color code based on thresholds
        cpu_color = "green" if cpu < 50 else "yellow" if cpu < 80 else "red"
        mem_color = "green" if memory_pct < 50 else "yellow" if memory_pct < 80 else "red"

        table.add_row("CPU Usage", f"[{cpu_color}]{cpu:.1f}%[/]")
        table.add_row("Memory (MB)", f"[{mem_color}]{memory_mb:.1f}[/]")
        table.add_row("Memory (%)", f"[{mem_color}]{memory_pct:.1f}%[/]")

        if "error" in system:
            table.add_row("Status", f"[red]{system['error']}[/]")

        return Panel(table, title="[bold]System Resources[/]", border_style="yellow")

    def generate_cache_panel(self) -> Panel:
        """Generate cache metrics panel."""
        if not HAS_RICH:
            return "Cache Metrics"

        # TODO: Hook up to actual cache stats when available
        table = Table(box=box.SIMPLE, show_header=False, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Hit Rate", "N/A")
        table.add_row("Size", "N/A")
        table.add_row("Evictions", "N/A")

        return Panel(table, title="[bold]Cache Stats[/]", border_style="magenta")

    def generate_footer(self) -> Panel:
        """Generate footer panel."""
        if not HAS_RICH:
            return "Press Ctrl+C to exit"

        footer = Table.grid(expand=True)
        footer.add_column(justify="left")
        footer.add_column(justify="center")
        footer.add_column(justify="right")

        footer.add_row(
            "[dim]Refresh: 1s[/]",
            "[bold]Press Ctrl+C to exit[/]",
            "[dim]HoloLoom v1.0[/]"
        )

        return Panel(footer, style="white on black")

    def generate_dashboard(self) -> Layout:
        """Generate complete dashboard layout."""
        layout = self.create_layout()

        layout["header"].update(self.generate_header())
        layout["latency"].update(self.generate_latency_panel())
        layout["throughput"].update(self.generate_throughput_panel())
        layout["system"].update(self.generate_system_panel())
        layout["cache"].update(self.generate_cache_panel())
        layout["footer"].update(self.generate_footer())

        return layout

    async def run(self, refresh_rate: float = 1.0):
        """
        Run the live dashboard.

        Args:
            refresh_rate: Refresh interval in seconds
        """
        if not HAS_RICH:
            print("Rich library not available - cannot display dashboard")
            print("Install with: pip install rich")
            return

        self.running = True

        with Live(self.generate_dashboard(), refresh_per_second=4, screen=True) as live:
            try:
                while self.running:
                    await asyncio.sleep(refresh_rate)
                    live.update(self.generate_dashboard())
            except KeyboardInterrupt:
                self.running = False

    def stop(self):
        """Stop the dashboard."""
        self.running = False


# CLI entry point
async def main():
    """Main entry point for dashboard."""
    dashboard = PerformanceDashboard()

    print("\n🎛️  Starting HoloLoom Performance Dashboard...")
    print("📊 Monitoring system metrics...")
    print()

    try:
        await dashboard.run(refresh_rate=1.0)
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped")


if __name__ == "__main__":
    asyncio.run(main())
