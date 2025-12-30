"""
HoloLoom Monitoring Dashboard
=============================

Real-time metrics visualization using the rich library.

Features:
- Query count and success rate tracking
- Pattern distribution visualization
- Average latency per pattern (BARE/FAST/FUSED)
- Memory backend hit rates
- Tool usage statistics
- Live dashboard updates

Usage:
    from HoloLoom.monitoring import MonitoringDashboard, MetricsCollector
    
    collector = MetricsCollector()
    dashboard = MonitoringDashboard(collector)
    
    # Track queries
    collector.record_query(pattern="fast", latency_ms=150, success=True)
    
    # Display dashboard
    dashboard.display()
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime
import threading

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich library not available. Install with: pip install rich")


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    pattern: str
    latency_ms: float
    success: bool
    backend: Optional[str] = None
    tool: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    complexity_level: Optional[str] = None
    memory_hits: int = 0


class MetricsCollector:
    """
    Collects and aggregates system metrics.
    
    Thread-safe for concurrent query recording.
    """
    
    def __init__(self):
        self.queries: List[QueryMetrics] = []
        self.pattern_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'successes': 0,
            'failures': 0,
            'total_latency': 0.0,
            'min_latency': float('inf'),
            'max_latency': 0.0,
        })
        self.backend_hits: Counter = Counter()
        self.tool_usage: Counter = Counter()
        self.complexity_distribution: Counter = Counter()
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_query(
        self,
        pattern: str,
        latency_ms: float,
        success: bool,
        backend: Optional[str] = None,
        tool: Optional[str] = None,
        complexity_level: Optional[str] = None,
        memory_hits: int = 0
    ):
        """Record metrics for a query."""
        with self.lock:
            metrics = QueryMetrics(
                pattern=pattern,
                latency_ms=latency_ms,
                success=success,
                backend=backend,
                tool=tool,
                complexity_level=complexity_level,
                memory_hits=memory_hits
            )
            self.queries.append(metrics)
            
            # Update aggregates
            stats = self.pattern_stats[pattern]
            stats['count'] += 1
            if success:
                stats['successes'] += 1
            else:
                stats['failures'] += 1
            stats['total_latency'] += latency_ms
            stats['min_latency'] = min(stats['min_latency'], latency_ms)
            stats['max_latency'] = max(stats['max_latency'], latency_ms)
            
            if backend:
                self.backend_hits[backend] += 1
            if tool:
                self.tool_usage[tool] += 1
            if complexity_level:
                self.complexity_distribution[complexity_level] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        with self.lock:
            total_queries = len(self.queries)
            successful_queries = sum(1 for q in self.queries if q.success)
            
            return {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'failed_queries': total_queries - successful_queries,
                'success_rate': successful_queries / total_queries if total_queries > 0 else 0.0,
                'uptime_seconds': time.time() - self.start_time,
                'avg_latency_ms': sum(q.latency_ms for q in self.queries) / total_queries if total_queries > 0 else 0.0,
            }
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.queries.clear()
            self.pattern_stats.clear()
            self.backend_hits.clear()
            self.tool_usage.clear()
            self.complexity_distribution.clear()
            self.start_time = time.time()


class MonitoringDashboard:
    """
    Real-time monitoring dashboard using rich library.
    
    Displays:
    - Overall statistics (queries, success rate, uptime)
    - Pattern distribution and latencies
    - Backend hit rates
    - Tool usage
    - Complexity level distribution
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.console = Console() if RICH_AVAILABLE else None
    
    def _create_summary_panel(self) -> Panel:
        """Create summary statistics panel."""
        summary = self.collector.get_summary()
        
        uptime_mins = summary['uptime_seconds'] / 60
        success_rate = summary['success_rate'] * 100
        
        text = Text()
        text.append(f"Total Queries: ", style="bold cyan")
        text.append(f"{summary['total_queries']}\n")
        text.append(f"Success Rate: ", style="bold green")
        text.append(f"{success_rate:.1f}% ")
        text.append(f"({summary['successful_queries']}/{summary['total_queries']})\n")
        text.append(f"Avg Latency: ", style="bold yellow")
        text.append(f"{summary['avg_latency_ms']:.1f}ms\n")
        text.append(f"Uptime: ", style="bold magenta")
        text.append(f"{uptime_mins:.1f} minutes")
        
        return Panel(text, title="[bold]System Overview[/bold]", border_style="cyan")
    
    def _create_pattern_table(self) -> Table:
        """Create pattern distribution table."""
        table = Table(title="Pattern Distribution", box=box.ROUNDED)
        
        table.add_column("Pattern", style="cyan", no_wrap=True)
        table.add_column("Count", style="magenta", justify="right")
        table.add_column("Success", style="green", justify="right")
        table.add_column("Avg Latency", style="yellow", justify="right")
        table.add_column("Min/Max", style="blue", justify="right")
        
        for pattern, stats in sorted(self.collector.pattern_stats.items()):
            count = stats['count']
            success_rate = (stats['successes'] / count * 100) if count > 0 else 0
            avg_latency = (stats['total_latency'] / count) if count > 0 else 0
            
            table.add_row(
                pattern,
                str(count),
                f"{success_rate:.0f}%",
                f"{avg_latency:.1f}ms",
                f"{stats['min_latency']:.0f}/{stats['max_latency']:.0f}ms"
            )
        
        return table
    
    def _create_backend_table(self) -> Table:
        """Create backend hit rate table."""
        table = Table(title="Backend Hit Rates", box=box.ROUNDED)
        
        table.add_column("Backend", style="cyan")
        table.add_column("Hits", style="magenta", justify="right")
        table.add_column("Percentage", style="green", justify="right")
        
        total_hits = sum(self.collector.backend_hits.values())
        
        for backend, hits in self.collector.backend_hits.most_common():
            percentage = (hits / total_hits * 100) if total_hits > 0 else 0
            table.add_row(backend, str(hits), f"{percentage:.1f}%")
        
        return table
    
    def _create_tool_table(self) -> Table:
        """Create tool usage table."""
        table = Table(title="Tool Usage Statistics", box=box.ROUNDED)
        
        table.add_column("Tool", style="cyan")
        table.add_column("Uses", style="magenta", justify="right")
        table.add_column("Percentage", style="green", justify="right")
        
        total_uses = sum(self.collector.tool_usage.values())
        
        for tool, uses in self.collector.tool_usage.most_common():
            percentage = (uses / total_uses * 100) if total_uses > 0 else 0
            table.add_row(tool, str(uses), f"{percentage:.1f}%")
        
        return table
    
    def _create_complexity_table(self) -> Table:
        """Create complexity distribution table."""
        table = Table(title="Complexity Levels", box=box.ROUNDED)
        
        table.add_column("Level", style="cyan")
        table.add_column("Count", style="magenta", justify="right")
        table.add_column("Percentage", style="green", justify="right")
        
        total = sum(self.collector.complexity_distribution.values())
        
        # Sort by complexity order
        order = ['LITE', 'FAST', 'FULL', 'RESEARCH']
        for level in order:
            if level in self.collector.complexity_distribution:
                count = self.collector.complexity_distribution[level]
                percentage = (count / total * 100) if total > 0 else 0
                table.add_row(level, str(count), f"{percentage:.1f}%")
        
        return table
    
    def display(self):
        """Display the dashboard once."""
        if not RICH_AVAILABLE:
            print("Dashboard requires rich library. Install with: pip install rich")
            # Fallback to text output
            summary = self.collector.get_summary()
            print(f"\n=== HoloLoom Monitoring ===")
            print(f"Total Queries: {summary['total_queries']}")
            print(f"Success Rate: {summary['success_rate']*100:.1f}%")
            print(f"Avg Latency: {summary['avg_latency_ms']:.1f}ms")
            print(f"Uptime: {summary['uptime_seconds']/60:.1f} minutes")
            return
        
        self.console.clear()
        self.console.print(Panel.fit("[bold cyan]HoloLoom Monitoring Dashboard[/bold cyan]"))
        self.console.print()
        
        # Summary
        self.console.print(self._create_summary_panel())
        self.console.print()
        
        # Pattern distribution
        if self.collector.pattern_stats:
            self.console.print(self._create_pattern_table())
            self.console.print()
        
        # Backend hits
        if self.collector.backend_hits:
            self.console.print(self._create_backend_table())
            self.console.print()
        
        # Tool usage
        if self.collector.tool_usage:
            self.console.print(self._create_tool_table())
            self.console.print()
        
        # Complexity distribution
        if self.collector.complexity_distribution:
            self.console.print(self._create_complexity_table())
    
    def display_live(self, refresh_rate: float = 1.0):
        """
        Display live-updating dashboard.
        
        Args:
            refresh_rate: Update interval in seconds
        """
        if not RICH_AVAILABLE:
            print("Live dashboard requires rich library. Install with: pip install rich")
            return
        
        def generate_layout():
            """Generate dashboard layout."""
            layout = Layout()
            layout.split_column(
                Layout(self._create_summary_panel(), name="summary", size=6),
                Layout(name="tables")
            )
            
            tables = []
            if self.collector.pattern_stats:
                tables.append(self._create_pattern_table())
            if self.collector.backend_hits:
                tables.append(self._create_backend_table())
            if self.collector.tool_usage:
                tables.append(self._create_tool_table())
            if self.collector.complexity_distribution:
                tables.append(self._create_complexity_table())
            
            # Create combined view of tables
            if tables:
                combined = "\n\n".join(str(t) for t in tables)
                layout["tables"].update(Panel(combined, border_style="cyan"))
            
            return layout
        
        with Live(generate_layout(), refresh_per_second=1/refresh_rate, console=self.console) as live:
            try:
                while True:
                    time.sleep(refresh_rate)
                    live.update(generate_layout())
            except KeyboardInterrupt:
                pass


# Singleton instance for easy access
_global_collector: Optional[MetricsCollector] = None


def get_global_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_global_collector():
    """Reset global collector metrics."""
    global _global_collector
    if _global_collector is not None:
        _global_collector.reset()
