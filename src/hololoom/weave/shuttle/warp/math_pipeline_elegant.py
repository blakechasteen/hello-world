#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elegant Math Pipeline - Fluent API with Beautiful Visualizations
=================================================================
A sexy, elegant interface for the Smart Math Pipeline with:
- Fluent API (method chaining)
- Beautiful terminal UI (colors, animations, sparklines)
- Interactive HTML dashboards
- Real-time RL learning visualization
- Performance optimizations (async, caching)

Philosophy: "Beauty is a feature, not a luxury."

Author: HoloLoom Team
Date: 2025-10-29
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

# Rich terminal UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.syntax import Syntax
    from rich.tree import Tree
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None

# Base math pipeline
from HoloLoom.warp.math_pipeline_integration import (
    MathPipelineIntegration,
    MathPipelineResult
)

logger = logging.getLogger(__name__)


# ============================================================================
# Beautiful Terminal UI
# ============================================================================

class BeautifulMathUI:
    """
    Beautiful terminal UI for math pipeline with Rich library.

    Features:
    - Colored output (intents, operations, insights)
    - Live progress bars
    - Sparklines for trends
    - Tables for statistics
    - Panels for results
    """

    def __init__(self):
        """Initialize beautiful UI."""
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None

        # Color scheme
        self.colors = {
            "similarity": "cyan",
            "optimization": "green",
            "analysis": "yellow",
            "verification": "blue",
            "transformation": "magenta",
            "decision": "red",
            "generation": "white",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "info": "blue"
        }

    def print_header(self, title: str, subtitle: str = ""):
        """Print beautiful header."""
        if not self.console:
            print(f"\n{'='*80}\n{title}\n{subtitle}\n{'='*80}\n")
            return

        panel = Panel(
            f"[bold white]{title}[/]\n[dim]{subtitle}[/]",
            box=box.DOUBLE,
            style="bold blue",
            expand=False
        )
        self.console.print(panel)

    def print_query(self, query: str, intent: str):
        """Print query with detected intent."""
        if not self.console:
            print(f"Query: {query}\nIntent: {intent}")
            return

        color = self.colors.get(intent.lower(), "white")
        self.console.print(f"\n[bold white]Query:[/] {query}")
        self.console.print(f"[bold {color}]Intent:[/] {intent.upper()}")

    def print_operations(self, operations: List[str], cost: int, budget: int):
        """Print selected operations with cost."""
        if not self.console:
            print(f"Operations: {', '.join(operations)}")
            print(f"Cost: {cost}/{budget}")
            return

        # Create operation tree
        tree = Tree("🔧 [bold]Operations Selected[/]")
        for i, op in enumerate(operations, 1):
            tree.add(f"[cyan]{i}.[/] {op}")

        self.console.print(tree)

        # Cost bar
        percentage = (cost / budget) * 100 if budget > 0 else 0
        bar_length = 40
        filled = int(bar_length * cost / budget) if budget > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        color = "green" if percentage < 70 else "yellow" if percentage < 90 else "red"
        self.console.print(f"\n[{color}]Cost:[/] {bar} {cost}/{budget} ({percentage:.0f}%)")

    def print_result(self, result: MathPipelineResult):
        """Print beautiful result panel."""
        if not self.console:
            print(f"\nSummary: {result.summary}")
            print(f"Confidence: {result.confidence:.0%}")
            print(f"Time: {result.execution_time_ms:.1f}ms")
            return

        # Build result content
        content = f"[bold white]{result.summary}[/]\n\n"

        # Insights
        if result.insights:
            content += "[bold yellow]✨ Insights:[/]\n"
            for insight in result.insights:
                content += f"  • {insight}\n"
            content += "\n"

        # Metrics
        confidence_color = "green" if result.confidence > 0.8 else "yellow" if result.confidence > 0.5 else "red"
        time_color = "green" if result.execution_time_ms < 10 else "yellow" if result.execution_time_ms < 50 else "red"

        content += f"[bold {confidence_color}]Confidence:[/] {result.confidence:.0%}  "
        content += f"[bold {time_color}]Time:[/] {result.execution_time_ms:.1f}ms  "
        content += f"[bold cyan]Cost:[/] {result.total_cost}"

        # Create panel
        panel = Panel(
            content,
            title="📊 Analysis Result",
            border_style="green",
            box=box.ROUNDED
        )
        self.console.print(panel)

    def print_statistics_table(self, stats: Dict[str, Any]):
        """Print statistics as beautiful table."""
        if not self.console:
            print("\nStatistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return

        table = Table(
            title="📈 Math Pipeline Statistics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("Metric", style="white")
        table.add_column("Value", style="green", justify="right")

        # Add key metrics
        if "total_analyses" in stats:
            table.add_row("Total Analyses", str(stats["total_analyses"]))
        if "avg_operations_per_analysis" in stats:
            table.add_row("Avg Operations", f"{stats['avg_operations_per_analysis']:.1f}")
        if "avg_confidence" in stats:
            table.add_row("Avg Confidence", f"{stats['avg_confidence']:.0%}")
        if "avg_execution_time_ms" in stats:
            table.add_row("Avg Time", f"{stats['avg_execution_time_ms']:.1f}ms")

        self.console.print(table)

    def print_rl_leaderboard(self, leaderboard: List[Dict]):
        """Print RL operation leaderboard."""
        if not self.console or not leaderboard:
            return

        table = Table(
            title="🏆 RL Operation Leaderboard",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold yellow"
        )

        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Operation", style="white")
        table.add_column("Intent", style="magenta")
        table.add_column("Success Rate", style="green", justify="right")
        table.add_column("Count", style="blue", justify="right")

        for i, op in enumerate(leaderboard[:10], 1):
            total = op["successes"] + op["failures"]
            rate = op["successes"] / total if total > 0 else 0

            # Medal for top 3
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."

            table.add_row(
                medal,
                op["operation_name"],
                op["intent"],
                f"{rate:.0%}",
                str(total)
            )

        self.console.print(table)

    def print_sparkline(self, values: List[float], label: str = "Trend"):
        """Print ASCII sparkline."""
        if not values:
            return

        # Normalize to 0-8 range
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1

        # Sparkline characters
        chars = "▁▂▃▄▅▆▇█"

        normalized = [(v - min_val) / range_val for v in values]
        sparkline = "".join(chars[int(n * 7)] for n in normalized)

        if self.console:
            self.console.print(f"[cyan]{label}:[/] {sparkline} [dim]({min_val:.2f} - {max_val:.2f})[/]")
        else:
            print(f"{label}: {sparkline} ({min_val:.2f} - {max_val:.2f})")


# ============================================================================
# Fluent API
# ============================================================================

class ElegantMathPipeline:
    """
    Fluent, elegant API for math pipeline.

    Features:
    - Method chaining
    - Builder pattern
    - Beautiful terminal output
    - Async support
    - Caching

    Example:
        result = await (ElegantMathPipeline()
            .with_budget(50)
            .enable_rl()
            .enable_composition()
            .beautiful_output()
            .analyze("Find similar documents")
        )
    """

    def __init__(self):
        """Initialize elegant pipeline."""
        # Configuration
        self._budget = 50
        self._enable_expensive = False
        self._enable_rl = False
        self._enable_composition = False
        self._enable_testing = False
        self._output_style = "detailed"
        self._use_contextual = False
        self._beautiful = False

        # State
        self._integration: Optional[MathPipelineIntegration] = None
        self._ui: Optional[BeautifulMathUI] = None
        self._cache: Dict[str, MathPipelineResult] = {}
        self._history: List[MathPipelineResult] = []

        logger.info("ElegantMathPipeline initialized")

    # Builder methods (fluent API)

    def with_budget(self, budget: int) -> 'ElegantMathPipeline':
        """Set computational budget."""
        self._budget = budget
        return self

    def enable_expensive_ops(self) -> 'ElegantMathPipeline':
        """Enable expensive operations (Ricci flow, etc.)."""
        self._enable_expensive = True
        return self

    def enable_rl(self) -> 'ElegantMathPipeline':
        """Enable RL learning (Thompson Sampling)."""
        self._enable_rl = True
        return self

    def enable_composition(self) -> 'ElegantMathPipeline':
        """Enable operation composition."""
        self._enable_composition = True
        return self

    def enable_testing(self) -> 'ElegantMathPipeline':
        """Enable rigorous property-based testing."""
        self._enable_testing = True
        return self

    def with_output_style(self, style: str) -> 'ElegantMathPipeline':
        """Set output style (concise/detailed/technical)."""
        self._output_style = style
        return self

    def use_contextual_bandit(self) -> 'ElegantMathPipeline':
        """Use 470-dim contextual bandit (advanced)."""
        self._use_contextual = True
        return self

    def beautiful_output(self) -> 'ElegantMathPipeline':
        """Enable beautiful terminal UI."""
        self._beautiful = True
        if HAS_RICH:
            self._ui = BeautifulMathUI()
        return self

    def lite(self) -> 'ElegantMathPipeline':
        """Configure for LITE mode (budget: 10)."""
        return self.with_budget(10).with_output_style("concise")

    def fast(self) -> 'ElegantMathPipeline':
        """Configure for FAST mode (budget: 50, RL enabled)."""
        return self.with_budget(50).enable_rl()

    def full(self) -> 'ElegantMathPipeline':
        """Configure for FULL mode (budget: 100, all features)."""
        return (self
            .with_budget(100)
            .enable_rl()
            .enable_composition()
            .enable_testing()
        )

    def research(self) -> 'ElegantMathPipeline':
        """Configure for RESEARCH mode (unlimited budget, all features)."""
        return (self
            .with_budget(999)
            .enable_rl()
            .enable_composition()
            .enable_testing()
            .enable_expensive_ops()
            .use_contextual_bandit()
            .with_output_style("technical")
        )

    # Execution methods

    def _ensure_integration(self):
        """Ensure integration is initialized."""
        if self._integration is None:
            self._integration = MathPipelineIntegration(
                enabled=True,
                budget=self._budget,
                enable_expensive=self._enable_expensive,
                enable_rl=self._enable_rl,
                enable_composition=self._enable_composition,
                enable_testing=self._enable_testing,
                output_style=self._output_style,
                use_contextual_bandit=self._use_contextual
            )

    async def analyze(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        context: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Optional[MathPipelineResult]:
        """
        Analyze query (async, with caching).

        Args:
            query: Query string
            query_embedding: Optional query embedding
            context: Optional context
            use_cache: Use cached result if available

        Returns:
            MathPipelineResult or None
        """
        # Check cache
        cache_key = query
        if use_cache and cache_key in self._cache:
            logger.info(f"Cache hit for: {query[:50]}...")
            return self._cache[cache_key]

        # Ensure integration
        self._ensure_integration()

        # Beautiful output
        if self._beautiful and self._ui:
            self._ui.print_header(
                "🔮 Math Pipeline Analysis",
                f"Budget: {self._budget} | Mode: {'RL' if self._enable_rl else 'Basic'}"
            )

        # Generate mock embedding if needed
        if query_embedding is None:
            query_embedding = np.random.randn(384)

        # Analyze (run in executor to not block)
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self._integration.analyze,
            query,
            query_embedding,
            context
        )

        if result:
            # Beautiful output
            if self._beautiful and self._ui:
                self._ui.print_query(query, result.metadata.get("intent", "unknown"))
                self._ui.print_operations(
                    result.operations_used,
                    result.total_cost,
                    self._budget
                )
                self._ui.print_result(result)

            # Cache result
            self._cache[cache_key] = result

            # Add to history
            self._history.append(result)

        return result

    def analyze_sync(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        context: Optional[Dict] = None
    ) -> Optional[MathPipelineResult]:
        """Synchronous version of analyze."""
        return asyncio.run(self.analyze(query, query_embedding, context))

    async def analyze_batch(
        self,
        queries: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[Optional[MathPipelineResult]]:
        """
        Analyze multiple queries (async, parallel).

        Args:
            queries: List of query strings
            use_cache: Use cached results
            show_progress: Show progress bar

        Returns:
            List of results
        """
        if show_progress and self._beautiful and self._ui:
            self._ui.console.print(f"\n[bold cyan]Analyzing {len(queries)} queries...[/]")

        # Analyze in parallel
        tasks = [
            self.analyze(query, use_cache=use_cache)
            for query in queries
        ]

        results = await asyncio.gather(*tasks)

        return results

    # Statistics & visualization

    def statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        self._ensure_integration()
        return self._integration.get_statistics()

    def show_statistics(self):
        """Show beautiful statistics."""
        stats = self.statistics()

        if self._beautiful and self._ui:
            self._ui.print_statistics_table(stats)

            # Show RL leaderboard if available
            if "rl_stats" in stats and "rl_learning" in stats["rl_stats"]:
                leaderboard = stats["rl_stats"]["rl_learning"].get("leaderboard", [])
                if leaderboard:
                    self._ui.print_rl_leaderboard(leaderboard)
        else:
            print("\nStatistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    def show_trends(self):
        """Show execution time trends."""
        if not self._history:
            print("No history yet")
            return

        times = [r.execution_time_ms for r in self._history]
        confidences = [r.confidence for r in self._history]
        costs = [r.total_cost for r in self._history]

        if self._beautiful and self._ui:
            self._ui.print_sparkline(times, "Execution Time (ms)")
            self._ui.print_sparkline(confidences, "Confidence")
            self._ui.print_sparkline(costs, "Cost")
        else:
            print(f"Execution times: {times}")
            print(f"Confidences: {confidences}")
            print(f"Costs: {costs}")

    def clear_cache(self) -> 'ElegantMathPipeline':
        """Clear result cache."""
        self._cache.clear()
        logger.info("Cache cleared")
        return self

    def save_state(self) -> 'ElegantMathPipeline':
        """Save RL state."""
        if self._integration:
            self._integration.save_state()
        return self

    # Context manager support

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Save RL state on exit
        self.save_state()

    def __repr__(self) -> str:
        """String representation."""
        mode = "RESEARCH" if self._budget > 500 else "FULL" if self._budget > 80 else "FAST" if self._budget > 30 else "LITE"
        features = []
        if self._enable_rl:
            features.append("RL")
        if self._enable_composition:
            features.append("Composition")
        if self._enable_testing:
            features.append("Testing")
        if self._beautiful:
            features.append("Beautiful UI")

        return f"ElegantMathPipeline({mode}, {', '.join(features)})"


# ============================================================================
# Convenience Functions
# ============================================================================

async def analyze(
    query: str,
    mode: str = "fast",
    beautiful: bool = True,
    **kwargs
) -> Optional[MathPipelineResult]:
    """
    One-liner analysis function.

    Args:
        query: Query string
        mode: "lite", "fast", "full", or "research"
        beautiful: Enable beautiful output
        **kwargs: Additional context

    Returns:
        MathPipelineResult

    Example:
        result = await analyze("Find similar documents", mode="fast", beautiful=True)
    """
    pipeline = ElegantMathPipeline()

    # Configure mode
    if mode == "lite":
        pipeline = pipeline.lite()
    elif mode == "fast":
        pipeline = pipeline.fast()
    elif mode == "full":
        pipeline = pipeline.full()
    elif mode == "research":
        pipeline = pipeline.research()

    # Beautiful output
    if beautiful:
        pipeline = pipeline.beautiful_output()

    return await pipeline.analyze(query, context=kwargs)


def analyze_sync(query: str, mode: str = "fast", **kwargs) -> Optional[MathPipelineResult]:
    """Synchronous one-liner analysis."""
    return asyncio.run(analyze(query, mode=mode, **kwargs))


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ELEGANT MATH PIPELINE - Fluent API Demo")
    print("="*80)
    print()

    async def demo():
        """Run elegant demo."""

        # Example 1: One-liner
        print("\n" + "="*80)
        print("EXAMPLE 1: One-Liner Analysis")
        print("="*80)

        result = await analyze(
            "Find documents similar to quantum computing",
            mode="fast",
            beautiful=True,
            has_embeddings=True
        )

        # Example 2: Fluent API
        print("\n" + "="*80)
        print("EXAMPLE 2: Fluent API (Method Chaining)")
        print("="*80)

        async with (ElegantMathPipeline()
            .fast()
            .beautiful_output()
        ) as pipeline:

            # Single query
            result = await pipeline.analyze(
                "Optimize the retrieval algorithm",
                context={"requires_optimization": True}
            )

            # Batch queries
            queries = [
                "Find similar items",
                "Analyze the structure",
                "Verify correctness"
            ]

            results = await pipeline.analyze_batch(queries, show_progress=True)

            # Show statistics
            print("\n")
            pipeline.show_statistics()
            pipeline.show_trends()

        # Example 3: Different modes
        print("\n" + "="*80)
        print("EXAMPLE 3: Mode Comparison")
        print("="*80)

        query = "Find the shortest path in the graph"

        for mode in ["lite", "fast", "full"]:
            print(f"\n--- {mode.upper()} Mode ---")
            result = await analyze(query, mode=mode, beautiful=False)
            if result:
                print(f"Operations: {len(result.operations_used)}")
                print(f"Cost: {result.total_cost}")
                print(f"Time: {result.execution_time_ms:.1f}ms")

    # Run demo
    asyncio.run(demo())

    print("\n" + "="*80)
    print("Elegant Math Pipeline - Beauty is a feature! ✨")
    print("="*80)