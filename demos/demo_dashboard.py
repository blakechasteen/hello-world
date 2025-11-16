#!/usr/bin/env python3
"""
Real-Time Dashboard Demo
========================

Demonstrates HoloLoom Promptly's real-time dashboard with live updates.

This demo:
1. Starts the dashboard server
2. Generates simulated executions
3. Shows live updates in the browser
4. Displays analytics, skills, and recent activity

Usage:
    PYTHONPATH=. python demos/demo_dashboard.py

Then open: http://localhost:8000

Created: 2025-11-16
Phase 5: Real-time Dashboard
"""

import asyncio
import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

from HoloLoom.config import Config
from HoloLoom.documentation.types import Query
from HoloLoom.weaving_orchestrator_recursive import RecursiveWeavingOrchestrator
from HoloLoom.protocols.recursive_reasoning import ReasoningStrategy
from HoloLoom.agentic.skill_agents import execute_skill

console = Console()


async def generate_sample_data():
    """Generate sample executions for dashboard visualization."""
    console.print(Panel.fit(
        "[bold cyan]Generating Sample Data[/bold cyan]\\n\\n"
        "Creating sample executions to populate the dashboard...",
        border_style="cyan"
    ))

    config = Config.fast()

    # Sample queries with different strategies
    queries_and_strategies = [
        ("What is Thompson Sampling?", ReasoningStrategy.REFINE),
        ("Explain the exploration-exploitation tradeoff", ReasoningStrategy.DECOMPOSE),
        ("How does PPO differ from DDPG?", ReasoningStrategy.CRITIQUE),
        ("What are the key features of HoloLoom?", ReasoningStrategy.EXPLORE),
        ("Is recursive reasoning always beneficial?", ReasoningStrategy.VERIFY),
        ("How should I architect a scalable API?", ReasoningStrategy.HOFSTADTER),
    ]

    # Sample code for skills
    sample_code = """
def calculate_discount(price, discount_percent):
    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Invalid discount")
    return price * (1 - discount_percent / 100)
"""

    # Execute recursive weaving queries with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(
            "[cyan]Executing sample queries[/cyan]",
            total=len(queries_and_strategies)
        )

        for i, (query_text, strategy) in enumerate(queries_and_strategies, 1):
            progress.update(
                task,
                description=f"[cyan]Query {i}/{len(queries_and_strategies)}: {query_text[:35]}...[/cyan]"
            )

            async with RecursiveWeavingOrchestrator(
                cfg=config,
                shards=[],
                enable_recursive=True,
                enable_analytics=True,
                default_strategy=strategy,
                max_iterations=3,
                quality_threshold=0.85
            ) as orch:
                query = Query(text=query_text)
                await orch.weave_with_strategy(
                    query=query,
                    strategy=strategy,
                    max_iterations=3
                )

            progress.advance(task)
            # Small delay between executions
            await asyncio.sleep(0.5)

    console.print("[green]✓ All queries executed![/green]\\n")

    # Execute skills with progress bar
    skills_to_test = [
        ("code-reviewer", {"code": sample_code, "language": "python"}),
        ("test-generator", {"code": sample_code, "language": "python"}),
        ("refactoring-expert", {"code": sample_code, "language": "python"}),
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(
            "[cyan]Executing sample skills[/cyan]",
            total=len(skills_to_test)
        )

        for i, (skill_name, params) in enumerate(skills_to_test, 1):
            progress.update(
                task,
                description=f"[cyan]Skill {i}/{len(skills_to_test)}: {skill_name}[/cyan]"
            )

            await execute_skill(
                skill_name=skill_name,
                parameters=params,
                config=config,
                enable_analytics=True
            )

            progress.advance(task)
            await asyncio.sleep(0.5)

    console.print("[green]✓ All skills executed![/green]\\n")
    console.print(f"[green]✅ Sample data generated![/green]")
    console.print(f"[dim]Total executions: {len(queries_and_strategies) + len(skills_to_test)}[/dim]")


async def main():
    """Main demo function."""
    console.print(Panel.fit(
        "[bold white on blue] HoloLoom Promptly Real-Time Dashboard Demo [/bold white on blue]\\n\\n"
        "[italic]Live visualization of memory, reasoning, and analytics[/italic]",
        border_style="blue"
    ))

    console.print("\\n[bold]Setup Instructions:[/bold]")
    console.print("1. This demo will generate sample data")
    console.print("2. Start the dashboard server in another terminal:")
    console.print("   [cyan]cd /home/user/hello-world && PYTHONPATH=. uvicorn HoloLoom.dashboard_server:app --reload --port 8000[/cyan]")
    console.print("3. Open http://localhost:8000 in your browser")
    console.print("4. Watch the dashboard update in real-time!\\n")

    if not Confirm.ask("[bold]Generate sample data?[/bold]", default=True):
        console.print("[yellow]Skipping data generation. Dashboard will start empty.[/yellow]")
        return

    # Generate sample data
    await generate_sample_data()

    console.print("\\n[bold]Dashboard Features:[/bold]")
    console.print("  • [cyan]Analytics Summary[/cyan] - Total queries, quality gains, costs")
    console.print("  • [green]Top Strategies[/green] - Most used reasoning strategies")
    console.print("  • [yellow]Available Skills[/yellow] - 13 professional skills")
    console.print("  • [magenta]Recent Executions[/magenta] - Live feed of activity")
    console.print("  • [blue]WebSocket Updates[/blue] - Real-time every 5 seconds")

    console.print("\\n[bold]REST API Endpoints:[/bold]")
    console.print("  GET /api/analytics/summary - Analytics summary")
    console.print("  GET /api/analytics/trends?days=7 - Quality trends")
    console.print("  GET /api/analytics/strategy/{name} - Strategy metrics")
    console.print("  GET /api/analytics/recommendations - AI recommendations")
    console.print("  GET /api/skills - All available skills")
    console.print("  GET /api/skills/{name} - Skill details")
    console.print("  GET /api/executions/recent?limit=20 - Recent executions")

    console.print("\\n[bold]WebSocket Endpoint:[/bold]")
    console.print("  WS /ws - Real-time updates")
    console.print("    • Connects on page load")
    console.print("    • Sends initial data")
    console.print("    • Broadcasts updates every 5s")
    console.print("    • Auto-reconnects on disconnect")

    console.print("\\n" + "="*70)
    console.print("[green]✅ Demo complete! Check the dashboard in your browser.[/green]")
    console.print("="*70)

    console.print("\\n[bold]Next Steps:[/bold]")
    console.print("1. Run more queries to see live updates")
    console.print("2. Try different strategies (REFINE, CRITIQUE, DECOMPOSE, etc.)")
    console.print("3. Execute skills (code-reviewer, bug-detective, etc.)")
    console.print("4. Watch analytics update in real-time!")


if __name__ == "__main__":
    asyncio.run(main())
