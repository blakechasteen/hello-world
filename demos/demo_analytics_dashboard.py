#!/usr/bin/env python3
"""
Recursive Reasoning Analytics Demo
===================================

Demonstrates analytics tracking and insights for recursive reasoning strategies.

Shows:
1. Automatic tracking of all executions
2. Strategy performance comparison
3. Quality improvement trends
4. Cost analysis
5. AI-powered recommendations

Usage:
    PYTHONPATH=. python demos/demo_analytics_dashboard.py
"""

import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm
from rich.progress import track

from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.weaving_orchestrator_recursive import RecursiveWeavingOrchestrator
from HoloLoom.protocols.recursive_reasoning import ReasoningStrategy


console = Console()


def create_test_shards():
    """Create sample memory shards"""
    return [
        MemoryShard(
            content="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem",
            entities=["Thompson Sampling", "Bayesian"],
            motifs=["exploration", "exploitation"]
        ),
        MemoryShard(
            content="Recursive loops enable iterative refinement of AI responses",
            entities=["recursive loops", "refinement"],
            motifs=["iteration", "improvement"]
        )
    ]


async def generate_test_data():
    """Generate test data by running multiple queries"""
    console.print(Panel.fit(
        "[bold cyan]Generating Test Data[/bold cyan]\n\n"
        "Running 20 queries across different strategies to populate analytics...",
        border_style="cyan"
    ))

    config = Config.fast()
    shards = create_test_shards()

    orchestrator = RecursiveWeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_recursive=True,
        enable_analytics=True  # Enable analytics tracking
    )

    # Test queries
    queries = [
        "What is Thompson Sampling?",
        "Explain recursive loops",
        "How do neural networks learn?",
        "What is reinforcement learning?",
        "Explain Bayesian inference",
        "What is meta-learning?",
        "How does backpropagation work?",
        "What is attention mechanism?",
        "Explain transformer architecture",
        "What is consciousness?",  # HOFSTADTER candidate
        "Why does gradient descent work?",  # DECOMPOSE candidate
        "Review this code: def foo(): pass",  # CRITIQUE candidate
        "What are the best approaches to learn ML?",  # EXPLORE candidate
        "Improve this explanation: ML is cool",  # REFINE candidate
        "Verify: The earth is round",  # VERIFY candidate
    ]

    strategies = [
        ReasoningStrategy.REFINE,
        ReasoningStrategy.CRITIQUE,
        ReasoningStrategy.DECOMPOSE,
        ReasoningStrategy.EXPLORE,
        ReasoningStrategy.HOFSTADTER,
        ReasoningStrategy.ADAPTIVE,  # Let system choose
    ]

    for i, query_text in enumerate(track(queries, description="Running queries...")):
        query = Query(text=query_text)

        # Alternate between explicit strategies and adaptive
        if i % 3 == 0:
            # Use explicit strategy
            strategy = strategies[i % len(strategies)]
            await orchestrator.weave_with_strategy(
                query,
                strategy=strategy,
                max_iterations=2  # Reduced for demo speed
            )
        else:
            # Use adaptive (auto-select)
            await orchestrator.weave(query)

    console.print("[green]✓ Generated 20 test executions[/green]\n")
    return orchestrator


async def demo_1_summary_statistics(orchestrator):
    """Demo 1: Summary statistics"""
    console.print(Panel.fit(
        "[bold green]Demo 1: Summary Statistics[/bold green]\n\n"
        "Overall performance metrics across all strategies",
        border_style="green"
    ))

    summary = orchestrator.get_analytics_summary()

    console.print(f"\n[bold]Overall Statistics:[/bold]")
    console.print(f"  Total Queries: {summary['total_queries']}")
    console.print(f"  Avg Quality Gain: {summary['avg_quality_gain']:.1%}")
    console.print(f"  Avg Iterations: {summary['avg_iterations']:.1f}")
    console.print(f"  Avg Duration: {summary['avg_duration_ms']:.0f}ms")
    console.print(f"  Total Cost: ${summary['total_cost']:.2f}")
    console.print(f"  Total Tokens: {summary['total_tokens']:,}")


async def demo_2_strategy_comparison(orchestrator):
    """Demo 2: Strategy comparison"""
    console.print(Panel.fit(
        "[bold magenta]Demo 2: Strategy Comparison[/bold magenta]\n\n"
        "Performance breakdown by reasoning strategy",
        border_style="magenta"
    ))

    summary = orchestrator.get_analytics_summary()
    strategies = summary.get("strategies", {})

    if not strategies:
        console.print("[yellow]No strategy data available yet[/yellow]")
        return

    # Create comparison table
    table = Table(title="Strategy Performance Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Avg Iterations", justify="right")
    table.add_column("Quality Gain", justify="right")
    table.add_column("Duration (ms)", justify="right")
    table.add_column("Success Rate", justify="right")

    for strategy_name, metrics in sorted(strategies.items()):
        table.add_row(
            strategy_name,
            str(metrics["count"]),
            f"{metrics['avg_iterations']:.1f}",
            f"{metrics['avg_quality_gain']:.1%}",
            f"{metrics['avg_duration_ms']:.0f}",
            f"{metrics['success_rate']:.0f}%"
        )

    console.print(table)

    # Find best performer
    best_quality = max(strategies.items(), key=lambda x: x[1]['avg_quality_gain'])
    console.print(f"\n[bold green]Best Quality:[/bold green] {best_quality[0]} "
                  f"(+{best_quality[1]['avg_quality_gain']:.1%})")

    fastest = min(strategies.items(), key=lambda x: x[1]['avg_duration_ms'])
    console.print(f"[bold blue]Fastest:[/bold blue] {fastest[0]} "
                  f"({fastest[1]['avg_duration_ms']:.0f}ms)")


async def demo_3_quality_trends(orchestrator):
    """Demo 3: Quality trends over time"""
    console.print(Panel.fit(
        "[bold yellow]Demo 3: Quality Trends[/bold yellow]\n\n"
        "Quality improvements over the last 7 days",
        border_style="yellow"
    ))

    if not orchestrator.analytics:
        console.print("[red]Analytics not enabled[/red]")
        return

    # Get trends for each strategy
    strategies = [
        ReasoningStrategy.REFINE,
        ReasoningStrategy.CRITIQUE,
        ReasoningStrategy.DECOMPOSE,
    ]

    for strategy in strategies:
        trends = orchestrator.analytics.get_quality_trends(strategy=strategy, days=7)

        if not trends:
            continue

        console.print(f"\n[bold]{strategy.value.upper()}:[/bold]")
        for trend in trends:
            bar = "█" * int(trend['avg_quality_gain'] * 50)
            console.print(
                f"  {trend['date']}: {bar} "
                f"{trend['avg_quality_gain']:.1%} ({trend['count']} queries)"
            )


async def demo_4_recommendations(orchestrator):
    """Demo 4: AI-powered recommendations"""
    console.print(Panel.fit(
        "[bold red]Demo 4: AI Recommendations[/bold red]\n\n"
        "Actionable insights based on analytics",
        border_style="red"
    ))

    summary = orchestrator.get_analytics_summary()
    recommendations = summary.get("recommendations", [])

    if not recommendations:
        console.print("[yellow]Not enough data for recommendations yet[/yellow]")
        return

    console.print("[bold]Recommendations:[/bold]\n")
    for i, rec in enumerate(recommendations, 1):
        if "⚠️" in rec:
            console.print(f"{i}. [yellow]{rec}[/yellow]")
        else:
            console.print(f"{i}. {rec}")


async def demo_5_export_data(orchestrator):
    """Demo 5: Export analytics data"""
    console.print(Panel.fit(
        "[bold blue]Demo 5: Export Analytics[/bold blue]\n\n"
        "Export data for external analysis",
        border_style="blue"
    ))

    output_path = "demos/output/analytics_export.csv"

    try:
        orchestrator.export_analytics(output_path)
        console.print(f"[green]✓ Analytics exported to {output_path}[/green]")

        # Show sample
        with open(output_path, 'r') as f:
            lines = f.readlines()[:6]  # Header + 5 rows

        console.print("\n[bold]Sample data:[/bold]")
        for line in lines:
            console.print(f"  {line.strip()}")

    except Exception as e:
        console.print(f"[red]Error exporting: {e}[/red]")


async def main():
    """Run all demos"""
    console.print(Panel.fit(
        "[bold white on blue] Recursive Reasoning Analytics Dashboard [/bold white on blue]\n\n"
        "[italic]Track, analyze, and optimize recursive reasoning performance[/italic]",
        border_style="blue"
    ))

    # Generate test data
    orchestrator = await generate_test_data()

    demos = [
        ("Summary Statistics", demo_1_summary_statistics),
        ("Strategy Comparison", demo_2_strategy_comparison),
        ("Quality Trends", demo_3_quality_trends),
        ("AI Recommendations", demo_4_recommendations),
        ("Export Data", demo_5_export_data),
    ]

    for i, (name, demo_fn) in enumerate(demos, 1):
        console.print(f"\n\n{'='*70}")
        console.print(f"[bold]Demo {i}/{len(demos)}: {name}[/bold]")
        console.print(f"{'='*70}\n")

        try:
            await demo_fn(orchestrator)
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
            import traceback
            traceback.print_exc()

        if i < len(demos):
            Confirm.ask("\n[bold]Continue to next demo?[/bold]", default=True)

    console.print("\n\n" + "="*70)
    console.print("[bold green]✅ Analytics dashboard complete![/bold green]")
    console.print("="*70)

    # Final summary
    console.print("\n[bold]Key Insights:[/bold]")
    summary = orchestrator.get_analytics_summary()
    console.print(f"  • Processed {summary['total_queries']} queries")
    console.print(f"  • Average quality improvement: {summary['avg_quality_gain']:.1%}")
    console.print(f"  • Total cost: ${summary['total_cost']:.2f}")

    if summary.get('strategies'):
        best = max(summary['strategies'].items(), key=lambda x: x[1]['avg_quality_gain'])
        console.print(f"  • Best strategy: {best[0]} (+{best[1]['avg_quality_gain']:.1%})")


if __name__ == "__main__":
    asyncio.run(main())
